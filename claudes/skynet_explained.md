# SkyNet FPGA 実装 解説ドキュメント

対象読者: HLS/C++ の一般知識はあるが、SkyNet リポジトリ固有のコードパターンは初見のハードウェアエンジニア。
目的: AMD/Xilinx HLS で量子化 AutoEncoder アクセラレータを手書きする際の参考として、SkyNet の設計判断とコード上の実装パターンを、実ソースの該当行を引用しながら整理する。

参照ソース (すべて `/mnt/d/dv/hls-study/SkyNet/` 以下、行番号は今回読んだ時点のもの):
- `FPGA/HLS/net_hls.h`, `net_hls.cc`, `conv1x1.cc`, `dwconv3x3.cc`
- `FPGA/HLS/reorder_weight.cc`, `golden_c.cc`, `output_verify.cc`, `tb.cc`, `script.tcl`
- `FPGA/RTL/script.tcl`
- `FPGA/Deploy/SkyNet.py`
- `GPU/models.py`（PyTorch リファレンスモデル）
- `README.md`
- 論文3本（MLSys'20 / ICML'19 workshop / DAC'19、arXiv 1909.09709 / 1905.08369 / 1904.04421）


---

## 1. Overview（概要と、本リポジトリを参考にする理由）

SkyNet は AutoEncoder ではない。**DAC-SDC (56th DAC System Design Contest)** という低消費電力 UAV 向け物体検出コンテストのために設計された、軽量な物体検出・追跡用 CNN である（`README.md:5`）。GPU（Jetson TX2）と FPGA（Ultra96）の両部門で優勝しており、このリポジトリの `FPGA/` 以下が Vivado HLS による手書き C++ 実装である。

ユーザーが設計しようとしている「量子化 AutoEncoder アクセラレータ」とネットワークの用途（検出 vs 再構成）は異なるが、参考にすべきは **ネットワークの意味的な役割ではなく、以下の "転用可能な" ハードウェア設計パターン**である。

- 固定小数点（`ap_fixed`）による量子化 CNN を、フレームワークからの自動変換ではなく **完全手書きの HLS C++** で実装している。
- 全レイヤーが同一種類の演算（1x1 conv, 3x3 depthwise conv）に還元されており、**単一の PE（Processing Element）ハードウェアをタイムシェアして全レイヤーに使い回す**（resource time-division）という、リソース制約の強いエッジ FPGA（Ultra96, xczu3eg）で汎用的に有効な設計手法を採用している。
- 重み・アクティベーションともに **オンチップに全部乗せず、DDR 経由で都度ストリーミングする** という現実的なメモリ階層設計をしている（BRAM 0.95MB 程度しかない小型 FPGA 向け）。
- Skip/bypass 接続を、DDR 上のオフセット管理と on-the-fly な space-to-depth 変換で処理しており、AutoEncoder のようなスキップ接続を持つネットワークにもそのまま応用できる。

以上の理由から、「物体検出ネットワークだが、ハードウェア実装のパターン集として読む」という位置づけで本ドキュメントを作成する。


---

## 2. Network architecture（ネットワーク構成）

PyTorch 版リファレンス実装 `GPU/models.py:34-86` の `SkyNet` クラスと、HLS 側 `FPGA/HLS/net_hls.cc` のステージコメント（`DW1_CONV...`〜`PW7_CONV...`）を突き合わせると、以下の構成になっている。

| ステージ | 種類 | 入力ch | 出力ch | 空間解像度 (1枚あたり) | 備考 |
|---|---|---|---|---|---|
| dw1 | depthwise 3x3 + pointwise 1x1 (`conv_dw`) | 3 | 48 | 160x320 | `models.py:60` |
| — | MaxPool2d(k=2,s=2) | 48 | 48 | 80x160 | `models.py:61` |
| dw2 | conv_dw | 48 | 96 | 80x160 | `models.py:62` |
| — | MaxPool2d(k=2,s=2) | 96 | 96 | 40x80 | `models.py:63` |
| dw3 | conv_dw | 96 | 192 | 40x80 | `models.py:64`。ここが bypass の分岐元 |
| — | MaxPool2d(k=2,s=2) | 192 | 192 | 20x40 | `models.py:67`（model_p2 側のみ） |
| dw4 | conv_dw | 192 | 384 | 20x40 | `models.py:68` |
| dw5 | conv_dw | 384 | 512 | 20x40 | `models.py:69` |
| reorg | space-to-depth stride2 (`ReorgLayer`) | 192→768 | — | 40x80→20x40 | `models.py:12-31`, dw3 出力を bypass |
| concat | channel concat | 768+512=1280 | — | 20x40 | `models.py:84` |
| dw6 | conv_dw | 1280 | 96 | 20x40 | `models.py:72` |
| pw7 | pointwise 1x1 (bias無し, BN無し) | 96 | 10 | 20x40 | `models.py:73` = 検出ヘッド |

`conv_dw` ブロックは `DW-Conv3x3(groups=inp) → BN → ReLU6 → PW-Conv1x1 → BN → ReLU6` という2層セット（`models.py:49-58`）で、論文でいう "Bundle"（後述 §10）そのものである。10 出力チャンネルは 2 anchor × (x,y,w,h,conf) = 10 の YOLO 風検出ヘッド。

HLS 側 `net_hls.cc` はこの構成を `DW1`〜`DW6`, `PW7` という7段のコードブロックとして直接踏襲しており、コメントに `/// DW1_CONV_3x3 + DW1_CONV_1x1 + POOL` (`net_hls.cc:814`) のように明記されている。ヘッダ `net_hls.h:73-87` の `SkyNet()` 関数シグネチャがトップレベル IP のインタフェースである。

バッチサイズ4枚を **2x2 のタイル状に1枚の大きい画像へステッチしてから同一ハードウェアパスに1回通す**（README記載の "batch size 4"）。これは §8 で詳述する。


---

## 3. Quantization scheme（量子化方式）

型定義は `net_hls.h:45-67` にまとまっている（`CSIM_DEBUG` 未定義時、つまり実機合成相当のパス）。

```cpp
typedef ap_fixed<9,  3, AP_RND, AP_SAT> FIX_FM;      // feature map (activation)
typedef ap_fixed<12, 4, AP_RND, AP_SAT> FIX_FM_acc;  // 1x1 conv の積和アキュムレータ
typedef ap_fixed<11, 4, AP_RND, AP_SAT> FIX_WT;      // 重み・バイアス
```

つまり **量子化はゼロ点付き affine 量子化（zero-point + scale）ではなく、シンプルな固定小数点表現（uniform symmetric fixed-point, `ap_fixed`）**である。`ap_fixed<W,I,AP_RND,AP_SAT>` は Wビット中 I ビットが整数部（符号含む）、残り W-I ビットが小数部で、丸めモード `AP_RND`（四捨五入相当）・飽和モード `AP_SAT`（オーバーフロー時に最大/最小値へクリップ）が全ての演算・代入に自動適用される。

- FIX_FM: 9bit, 整数部3bit・小数部6bit → 表現範囲 約[-4, 4)、LSB=1/64
- FIX_WT: 11bit, 整数部4bit・小数部7bit → 表現範囲 約[-8, 8)、LSB=1/128
- FIX_FM_acc: 12bit, 整数部4bit・小数部8bit → 表現範囲 約[-8, 8)、LSB=1/256（アキュムレータなので小数部を1bit広げて丸め誤差を抑えている）

これは MLSys'20 論文（arXiv:1909.09709）の Table 7 で報告されている量子化探索のうち "scheme 1: 9-bit feature map / 11-bit weight（IoU 0.727、フル精度比 -1.4%）" にちょうど一致する（フル精度 float32 baseline は IoU 0.741）。他の候補（8bit FM や 10bit weight）は精度劣化が大きいため採用されなかった、と論文に明記されている。**このビット幅は学習時に量子化を意識してハイパーパラメータ探索された値であり、HLS 側では単に `ap_fixed<9,3>` 等の型として固定されているだけ**（実行時に可変ではない）。

**scale factor はどこにあるか**: per-channel/per-tensor の明示的な scale 変数は存在しない。すべて `ap_fixed` の整数部・小数部ビット数（＝暗黙の2進小数点位置）が「scale」の役割を果たしている。オフライン変換（`reorder_weight.cc`）は float の重みを `(FIX_WT)dw1_conv_3x3_weight[c][m][n]` のような **単純な C++ 型キャスト**で量子化しているだけで（例: `reorder_weight.cc:176`）、per-layer にスケールを掛けたり zero-point を引いたりする処理は一切ない。つまり「量子化」は学習側（GPU/PyTorch）で fake-quantization を意識して事前に済ませてあり、FPGA側は素直に丸めてビット幅を落とすだけ、という設計である（学習コード自体はこのリポジトリの `FPGA/HLS/` には含まれておらず、`weights_floating.bin` という学習済みfloat重み一式が既に BN 折り込み済みの conv 重み・バイアスとして与えられる前提になっている。BN パラメータそのものは `tb.cc:load_weights()` にも `reorder_weight.cc` にも現れないため、**BatchNorm は学習後にオフラインで畳み込みの重み・バイアスへ fuse 済み**と考えられる。論文側にも "layer fusion to merge both parameters from Conv and its successive BN offline" と明記されている）。

画素値の正規化（`(pixel/255.0 - 0.5) / 0.25`）も同様に、実行時の割り算を避けるため 256 エントリの LUT `img_norm_ch[256]`（`net_hls.cc:491-508`）として事前計算されたテーブル引きに置き換えられている。これも一種の「オフライン量子化テーブル化」の実例。


---

## 4. Requantization handling（層間のリスケール処理）

典型的な affine 量子化アクセラレータのような「積和後に scale 定数を乗算してビットシフトし、次層の量子化スケールへ合わせる」という明示的な requant 乗算は **存在しない**。代わりに `ap_fixed` 型どうしの代入・キャストが自動的に２進小数点を合わせ、丸め(AP_RND)・飽和(AP_SAT)を行う、という C++ の言語機能そのものが requantize の役割を担っている。具体的なタイミングは以下の通り。

1. **depthwise 3x3 conv 内部（`dwconv3x3.cc:34-46`）**: `top[co][h][w] += (weights[co][i][j] * bottom[co][h+i-1][w+j-1]);` において `top` は `FIX_FM`（9,3）型の配列そのもの。3x3=9回のタップ加算のたびに、暗黙の乗算結果が **その都度 FIX_FM の精度に丸め・飽和されてから加算される**。つまり depthwise conv は「タップごとに9回 requantize している」ことになる。
2. **pointwise 1x1 conv 内部（`conv1x1.cc:11-142`）**: 16入力チャンネル分の積和木（`compute_engine_16`, `conv1x1.cc:11-71`）は中間結果を `FIX_32_10`（32bit, 整数部10bit）という **広いビット幅の一時変数**に保持し、加算木の最後にだけ `FIX_FM_acc`（12,4）へ加算する（`conv1x1.cc:118`）。つまり pointwise conv は depthwise conv と異なり、**16タップ分の積和はフル精度に近い形で計算してから1回だけ丸める**という非対称な精度ポリシーになっている。これは実装上の細かい違いだが、丸め誤差の蓄積のされ方が層タイプによって異なる点として明記しておく。
3. **活性化関数と同時に requantize**: `relu_single()`（`net_hls.cc:273-279`, `dwconv3x3.cc:12-18` に同一実装が重複定義）が ReLU6（`d>6→6`, `d<0→0`, それ以外はそのまま）を実装しており、`FIX_FM_acc` → `FIX_FM` への明示キャスト `(FIX_FM)src[c][h][w]` と同時に呼ばれる（例: `relu_copy_buf_to_DDR_acc`, `net_hls.cc:309-329`）。ここが「アキュムレータ精度 → 次層入力精度」への正式な requantize ポイントで、ReLU6 のクリップと固定小数点の丸め・飽和が1つの代入文で同時に行われる。
4. **例外（生ビットコピー）**: `local_buf_copy()`（`net_hls.cc:760-771`）は `dest[c][h][w].range(FM_RG, 0) = src[c][h][w].range(FM_RG, 0);` という **`.range()` によるビット単位の直接コピー**を行っており、`FIX_FM_acc`→`FIX_FM` の代入でありながら ap_fixed の自動リスケールを経由しない。`FIX_FM` は小数部6bit、`FIX_FM_acc` は小数部8bitなので、本来は2進小数点がずれているはずであり、このコピーが数値的に何を意図しているかはソースの静的読解だけでは断定できなかった（DW6 の reorg 一時バッファとして使われている特殊なケースであり、`load_and_reorg_part` 内でも同様に `.range(WT_RG, 0)`（11bit幅）を使った生ビットコピーで `FIX_FM_acc` 型の配列に書き込んでいる箇所がある: `net_hls.cc:720-738`）。**再現実装する場合はここだけ C-simulation で数値を実際にトレースして意図を確認することを推奨する**。

requantize のタイミングは「層の境界ごと」（1x1 conv や 3x3 conv の出力を書き戻す瞬間）であり、タイルごとの中間結果には適用されない。スケール値は実行時パラメータではなく、**すべて `ap_fixed` の型定義（コンパイル時定数）に静的に埋め込まれている**。


---

## 5. Resource time-division architecture（リソースの時分割共有）

SkyNet の HLS 実装の最大の特徴は、**全レイヤー共通で使う PE を1つだけ合成し、制御ループでレイヤーごとに違うタイル数・チャンネル数だけ回す**という設計である。これは `net_hls.cc:805-808` の以下のプラグマで明示的に強制されている。

```cpp
#pragma HLS ALLOCATION instances=CONV_1x1               limit=1 function
#pragma HLS ALLOCATION instances=DW_CONV_3x3             limit=1 function
#pragma HLS ALLOCATION instances=Relu_Max_Pooling        limit=1 function
#pragma HLS ALLOCATION instances=load_image_chunk_norm   limit=1 function
```

`ALLOCATION ... limit=1 function` は「この関数の呼び出しがソースコード中に何十回あっても、合成後のハードウェアインスタンスは1つだけに制限し、呼び出しごとに時分割で共有する」という Vivado HLS 固有の指示である。実際 `SkyNet()` 本体（`net_hls.cc:773-1223`）では `CONV_1x1(...)` が dw1〜pw7 まで **合計十数回、チャンネル数もタイルサイズも変えながら**呼び出されている（例: `net_hls.cc:848`, `890`, `893`, `946`, `1013`, `1074`, `1183`, `1210` など）。これらはすべて同一の物理 PE 配列を再利用する。

PE 内部の並列度は以下の通り（`conv1x1.cc:91-142`）。

- `top`, `bottom`, `weights` はすべて `#pragma HLS array_partition ... dim=1 complete`（`conv1x1.cc:96-99`）でチャンネル方向に完全分割 → 32チャンネル分のレジスタ/BRAMバンクに同時アクセス可能。
- 出力チャンネル `coo` ループは `#pragma HLS unroll`（`conv1x1.cc:116-117`）で32並列展開 → **32個の PE を空間展開**。
- 各 PE は16入力チャンネル分の積和木 `compute_engine_16`（`conv1x1.cc:11-71`、`log2(16)=4` 段のバイナリ加算木）をハードウェア化しており、入力チャンネルは `ci` ループを16刻みで2回まわして32チャンネル分をカバーする（`conv1x1.cc:106`）。
- ループ本体は `#pragma HLS pipeline II=2`（`conv1x1.cc:115`）でパイプライン化。

つまり **32 (出力ch並列) × 16 (積和木) = 512 MAC が空間展開され、これが全レイヤー・全タイルに対して時分割で再利用される固定サイズの PE アレイ**という構造。depthwise 側も同様に `DW_CONV_3x3`（`dwconv3x3.cc:21-59`）で `co` を32並列展開（`dwconv3x3.cc:39-40`）し、`#pragma HLS pipeline`（`dwconv3x3.cc:38`）でタップごとにパイプライン化している。チャンネル数がハードウェアの32レーンに満たない層（例: dw1 の入力3ch）はゼロ埋めして32の倍数に揃えている（`reorder_weight.cc:174-184` 参照）。

`net_hls.cc` 本体側では、レイヤーごとに `CI_N`（入力32ch単位数）・`CO_N`（出力32ch単位数）という変数を都度更新し（例 `net_hls.cc:821-822`, `871-872`, `922-923`, `975`, `1001`, `1038`, `1062`, `1100`, `1146`, `1170-1171`, `1203-1204`）、その値に応じて `for(int co = 0; co < CO_N; co++)` のようなループ回数を変えることで、**同じ CONV_1x1/DW_CONV_3x3 呼び出しをレイヤーごとに異なる反復回数だけ実行する**（=同じハードウェアを異なる「実行時間」だけ使う時分割制御）。これは「レイヤーごとに専用パイプラインを並べる」設計（dataflow 型の per-layer replication）とは対照的な、**面積優先・スループット非優先のシーケンシャル制御アーキテクチャ**である。`#pragma HLS DATAFLOW` は使われておらず、`SkyNet()` トップ関数はレイヤーを順番に呼ぶ単純な逐次コードである。


---

## 6. Skip connection handling（バイパス接続の扱い）

PyTorch 側 (`models.py:12-31, 80-86`) では dw3 の出力（192ch, 40x80）を `ReorgLayer(stride=2)` で **space-to-depth**（H,Wを1/2にしてチャンネルを4倍にする pixel-unshuffle 的操作）し、192→768chに変換した上で、dw5 の出力（512ch, 20x40）と channel 方向に concat して 1280ch にしている。

HLS 側ではこの処理を **DDR 上のオフセット管理＋オンザフライなビット再配置**として実装している。

1. dw3 の 1x1 conv 出力（192ch）は `relu_copy_buf_to_DDR_acc(DDR_buf_burst, 100 + co + (col*2+row)*CO_N, FM_buf_acc, 0, 0)`（`net_hls.cc:954`）で、**タイルごとに `DDR_buf` の 100番地以降**（コメントに明記: `net_hls.cc:1098` "Output of DW3_CONV_1x1_OUT are stored in DDR_buf[100] to DDR_buf[123]"）へいったん退避される。dw3 は 2x2 のタイル分割で処理されるため、4タイル分（各6チャンネルブロック）が 100, 106, 112, 118 の4つの開始オフセットに配置される。
2. dw5 の出力は同様に `DDR_buf[42]`〜`[57]`（`net_hls.cc:1030-1031` コメント）に格納される。
3. dw6 に入る直前、`load_and_reorg()`（`net_hls.cc:746-757`）/`load_and_reorg_part()`（`net_hls.cc:626-742`）が、DDR 上の4つの異なるオフセット (`100+c/4`, `+6`, `+12`, `+18`、すなわち2x2タイルの4象限) から512bitワードを読み出し、**4x4画素ブロック単位でチャンネル方向へ再配置**することで space-to-depth 変換をハードウェア的に実現している（`net_hls.cc:638-742` の `DATA[0..15]` への読み出しと `buf_out_1/2/3/4` への書き戻しがそれ）。この関数は `#pragma HLS pipeline` すら付いておらず、完全にビット演算とインデックス計算だけのソフトウェア的な reorg である。
4. 再配置後、`local_buf_copy()`（`net_hls.cc:760-771`、§4で触れた生ビットコピー）で一時的に `FM_buf_acc` を汎用バッファとして再利用しつつ、`DW_CONV_3x3` を4回（`net_hls.cc:1101-1141` の `for(int c=0; c<CI_N; c+=4)` ループ内で c+0〜c+3 の4回）呼び出して 768ch 分の depthwise conv を処理し、`DDR_buf[58]`〜`[81]` へ書き戻す。
5. dw5 側（512ch）は別ループ（`net_hls.cc:1146-1163`）でそのまま `DW_CONV_3x3` を通し `DDR_buf[82]`〜`[97]` へ。
6. 最終的に `DDR_buf[58]`〜`[97]`（=768+512=1280ch 分、`net_hls.cc:1166` コメント "input in DDR_buf[58] - DDR_buf[97]"）を dw6 の 1x1 conv (`CI_N=1280/32=40`) が読み込み、concat 済みとして扱う。

つまり **on-chip の ping-pong BRAM でバイパスを持ち回すのではなく、DDR 上の固定アドレスオフセットに各ブランチの出力を書き出しておき、後続レイヤーがそのオフセットから読み直す**という、DDR をワーキングメモリ（scratchpad）として使うスタイル。オンチップに保持されるのは 1 タイル分（32ch×44×84）の `FM_buf1〜4`, `FM_buf_acc` のみ（`net_hls.cc:12-16`）で、これらは再利用のたびに上書きされる汎用ワークバッファである。

**"mysterious dw3" のチャンネル並べ替え**: `reorder_weight.cc:54-56, 122-168` に "reordered weights for the mysterious dw3(192->768)" というコメントがあり、PyTorch の `ReorgLayer` が生成するチャンネル順序（`view/transpose` の結果、サブ位置ごとに192chブロックが4つ連続に並ぶ: `[0..191]=位置(0,0)`, `[192..383]=位置(0,1)`, ...）を、HLS 側が要求する「32chレーンごとに4つの位置が交互に並ぶ」順序（ch, ch+192, ch+192*2, ch+192*3 が32個おきに並ぶ）へ、dw6 の重み・バイアス自体を**オフラインで並べ替えている**（`reorder_weight.cc:123-168`）。これは「フレームワークのテンソルレイアウトとハードウェアのアクセスパターンが food合わない場合、活性化ではなく重み側を並べ替えて帳尻を合わせる」という汎用的なテクニックの実例。


---

## 7. On-chip vs DDR（オンチップ / オフチップの切り分け）

トップ関数 `SkyNet()` の m_axi インタフェース宣言（`net_hls.cc:789-802`）は次の通り。

```cpp
#pragma HLS INTERFACE m_axi depth=3*162*322     port=image_in_raw_pad          offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=306*32*32     port=conv_weight_1x1_all      offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=24*32*3*3     port=conv_weight_3x3_all      offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=63*32         port=bias_all                 offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=106272        port=DDR_dw1_pool_out_PL_burst offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=41328         port=DDR_dw2_pool_out_PL_burst offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=473088        port=DDR_buf_burst            offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=5             port=predict_boxes            offset=slave bundle=OUTPUT
#pragma HLS INTERFACE m_axi depth=5             port=constant                 offset=slave bundle=OUTPUT
#pragma HLS INTERFACE m_axi depth=10*32*22*42   port=debug                    offset=slave bundle=OUTPUT
#pragma HLS INTERFACE s_axilite register        port=return
```

**重みは全くオンチップに常駐しない。** `conv_weight_1x1_all`, `conv_weight_3x3_all`, `bias_all` はすべて m_axi 経由（DDR 上）で、レイヤーの処理途中で `load_weight_1x1_from_axi()`（`net_hls.cc:355-367`）や `load_weight_3x3_from_axi()`（`net_hls.cc:371-385`）によって **その都度 32ch 分だけ読み出され**、小さな `weight_buf_1x1[4][32][32]` / `weight_buf_3x3[4][32][3][3]`（`net_hls.cc:19-20`）という一時バッファに乗る。重み全体のサイズは `conv_weight_1x1_all` だけで `depth=306*32*32` 語（512bit幅）= 約 627KB 相当であり、Ultra96 の BRAM 総量（論文記載: 約0.95MB）に対して他のバッファと合わせると到底収まらないため、**DDR 常駐が必須**という設計判断（論文にも "Network parameters can not be accommodated by the FPGA on-chip memory ... we have to store them in the external memory (DRAM)" と明記）。

**中間特徴マップ（アクティベーション）も、タイルより大きい範囲では DDR に置かれる。** `DDR_dw1_pool_out_PL_burst`（dw1後のmaxpool出力, 64/32×82×2×162×2 個の512bitワード）、`DDR_dw2_pool_out_PL_burst`（dw2後）、`DDR_buf_burst`（dw3以降の全中間結果、128×44×84語）がすべて m_axi 経由。§6で見た通り、bypass 用の一時保存もこの `DDR_buf` を使う。

**オンチップに乗るのは「1タイル分」のワーキングバッファだけ**: `FM_buf1[32][44][84]`, `FM_buf2`, `FM_buf3`, `FM_buf4`（`FIX_FM`, 9bit）、`FM_buf_acc`（`FIX_FM_acc`, 12bit）（`net_hls.cc:12-16`）。これらは 32×44×84×9bit ≈ 133KB（acc版は177KB）程度で、BRAM に収まるサイズに抑えられている。加えて `weight_buf_1x1[4][32][32]`, `weight_buf_3x3[4][32][3][3]`, `bias_buf[4][32]`（`net_hls.cc:19-21`）というごく小さな重みの一時コピーもオンチップ。

RTL 側 (`FPGA/RTL/script.tcl:20-21`) では、この2つの m_axi バンドル（`bundle=INPUT`, `bundle=OUTPUT`）がそれぞれ Zynq UltraScale+ の `S_AXI_HP0_FPD`, `S_AXI_HP1_FPD`（高性能 AXI ポート、`m_axi_INPUT_r`, `m_axi_OUTPUT_r`）に接続されている。制御は `s_axilite`（`net_hls.cc:802`）経由の AXI-Lite で、ホスト（ARM PS 側の Python コード）がスタートビットを書き込み完了をポーリングする（`SkyNet.py:204-207`）。

まとめると: **オンチップ = 1タイル分のFM+重みの往復バッファのみ。重み全体・レイヤー間中間結果・バイパス退避先はすべてDDR。** これは「重みをオンチップに全部乗せる」設計ではなく「小さい PE アレイ＋大きい外部メモリ帯域」で面積を切り詰める設計である。


---

## 8. Streaming I/O（入出力のストリーミング）

- **入力画像**: バッチ4枚を、ホスト側 Python (`SkyNet.py:76-106` の `stitch()`) があらかじめ 2x2 グリッド（644x324、各画像を320x160にリサイズして1px paddingを挟んで貼り付け）に **ステッチしてから1枚の画像として DDR に転送**する。HLS側は `image_in_raw_pad[3*162*2*322*2]` という1枚の巨大画像として受け取る（`net_hls.h:73`, `net_hls.cc:773`）。つまりバッチ処理はハードウェア的な並列実行ではなく、**空間的なタイル化（1枚の大きな疑似画像として同一パイプラインに通す）**で実現している。
- ホスト⇔FPGA 間は AXI-Lite の制御レジスタ（`SkyNet.write(0x00, 1)` で開始、`SkyNet.read(0x00)` でビジーフラグをポーリング, `SkyNet.py:204-207`）でハンドシェイクするポーリング方式。AXI-Stream は使われていない（全データ転送は m_axi バーストのメモリマップドアクセス）。
- **画像の正規化**: `load_image_chunk_norm()`（`net_hls.cc:512-544`）が、m_axi 経由で読んだ生の `uint8` 画素を §3 で述べた256エントリLUT `img_norm_ch` でオンザフライに正規化しつつ、44x84 のタイルバッファへ詰め直す。ここで **画像を44x84（=42x82の有効領域＋3x3畳み込み用の1px halo）のタイル単位でDDRからストリーム読み出し**しており、dw1 の処理では `row`(8回)×`col`(8回) の二重ループでタイルを順に処理する（`net_hls.cc:831-853`）。
- **ダブルバッファリング**: dw1 の3x3conv入力について `FM_buf1`/`FM_buf3` を交互に使うピンポン方式が明示的にコード化されている。`if(col%2==0){ DW_CONV_3x3(FM_buf1,...); load_image_chunk_norm(FM_buf3, ...) } else { DW_CONV_3x3(FM_buf3,...); load_image_chunk_norm(FM_buf1,...) }`（`net_hls.cc:837-844`）。次のタイルの DMA ロードと今のタイルの演算を重ねる、典型的な ping-pong バッファリング。dw4/dw5 でも同様のパターンが `FM_buf1`/`FM_buf3` の交互ロードで繰り返される（`net_hls.cc:985-992`, `1045-1052`）。
- **出力（検出結果）**: FPGA 側は `predict_boxes[4][5]`（4画像分のバウンディングボックス生の値）と `constant[4][3]`（どちらのanchorが選ばれたか等のインデックス情報）のみを m_axi 経由で書き出す（`net_hls.cc:1217`, `compute_bounding_box()` 本体は `net_hls.cc:25-270`）。**sigmoid・exp によるボックスデコードは FPGA 上では行わず、ホスト側（`SkyNet.py:109-150` の `compute_bounding_box()`、あるいはテストベンチ `tb.cc:420-456`）が float 演算で行う**。これは、シグモイド／指数関数という非線形演算をわざわざ固定小数点でハードウェア化するコストを避け、後処理をPS(ARM)側に逃がす設計判断である。
- バッチ4枚分の検出結果は、44x84 のタイルバッファを4象限（`compute_bounding_box()` 内の m,n ループ範囲 `1..20 vs 23..42` × `1..40 vs 43..82`, `net_hls.cc:37-38, 98-99, 158-159, 219-220`）に分割してそれぞれ argmax することで抽出しており、これは §2 のステッチ入力と対応している。


---

## 9. Weight/bias/scale storage layout（重み・バイアスの格納レイアウト）

`reorder_weight.cc`（667行）はオフラインの重み前処理ツールで、以下を行う。

1. **チャンネル数のゼロパディング**: ハードウェアが32ch単位で動くため、3ch(dw1入力)や48ch(dw1/dw2出力)のような32の倍数でないチャンネル数を、32の倍数（32, 64, ...）へゼロ埋めする（例: `reorder_weight.cc:172-204`、3→32ch, 48→64ch）。
2. **float→固定小数点キャスト**: `(FIX_WT)dw1_conv_3x3_weight[c][m][n]` のような単純キャストで `ap_fixed<11,4>` へ変換（`reorder_weight.cc:176` 等）。
3. **dw6 の "mysterious" reorder**（§6参照）: `dw6_conv_3x3_weight_reo`, `dw6_conv_1x1_weight_reo` へ、PyTorch の reorg 層が生成するチャンネル順とハードウェアが要求する32chレーン順を合わせるための並べ替え（`reorder_weight.cc:123-168`）。
4. **1x1 conv 重みの32x32ブロック化**: 各レイヤーの `[CO][CI]` 重み行列を 32x32 のタイルに分割し、`fix_conv_weight_1x1_all[index_1x1][32][32]` という**全レイヤー共通の平坦配列**へ `CO`, `CI` ブロックの外側ループでインデックスを振りながら詰め込む（`reorder_weight.cc:352-566`）。3x3 conv 重みとバイアスも同様に `fix_conv_weight_3x3_all[index_3x3][32][3][3]`, `fix_bias_all[index_bias][32]` へ詰め込まれる。
5. **512bitワードへのパッキング**: 32ch分の `FIX_WT`（11bit幅、`WT_RG=10` で `.range(10,0)` として抽出）を、512bit の `uint512` 1ワードに16bitレーン×32個で敷き詰める（`reorder_weight.cc:623-664`）。これが m_axi バス幅（512bit）に一致させるための詰め方であり、CONV_1x1/DW_CONV_3x3 が一度に32チャンネル分の重みをバースト転送できる理由になっている。
6. **単一バイナリへの書き出し**: 最終的に `weights_fixed.bin` というファイルへ、**`conv_weight_1x1_all` 全体 → `conv_weight_3x3_all` 全体 → `bias_all` 全体**の順でフラットに書き出す（`reorder_weight.cc:625-664`）。ホスト側 `SkyNet.py:42-49` はこのファイル（README では `SkyNet.bin` にリネーム）を読み込み、`conv_weight_1x1_all.size`, `conv_weight_3x3_all.size`, `bias_all.size` という **静的に決まったオフセット境界**で3つの配列に分割して DDR へコピーする。つまりオフセット管理はコード上のサイズ定数の一致だけで担保されており、ファイル内に自己記述的なヘッダやスケール値は含まれない。

要するに、単一 blob の内部レイアウトは:

```
[ conv_weight_1x1_all: index_1x1 個 × 32 × uint512 ]
[ conv_weight_3x3_all: index_3x3 個 × 3 × 3 × uint512 ]
[ bias_all           : index_bias 個 × uint512 ]
```

という3セクション構成で、量子化スケールに相当する情報は一切含まれていない（§3 の通りスケールはコンパイル時の型定義に埋め込み済みのため）。


---

## 10. Papers summary（3本の論文の設計思想の要約）

3本の論文（MLSys'20 = arXiv:1909.09709, ICML'19 workshop Best Poster = arXiv:1905.08369, DAC'19 = arXiv:1904.04421）は同じ研究グループによる一連の研究で、**「ハードウェアを後付けで考えるのではなく、ネットワーク設計の最初期からハードウェア制約を織り込む」ボトムアップ設計手法**を共通の骨格としている。DAC'19 と ICML'19 workshop 論文がこの co-design 手法論そのもの（Auto-DNN / Auto-HLS、bottom-up + top-down のバイディレクショナル探索）を提案し、MLSys'20 論文（SkyNet 本体）がその手法を実際に DAC-SDC 向けに適用した結果を報告する、という relationship になっている。

中心的な概念は **"Bundle"**: 3x3 depthwise conv + 1x1 pointwise conv + BatchNorm + ReLU6 という「ハードウェア効率の良い層の組」を最小構成単位として定義し（DAC'19論文: "a set of sequential DNN layers as a basic DNN building block", 1bundleにつき計算IPは最大2つまでに制限）、まず各種 Bundle 候補をターゲットデバイス上で実装・計測してレイテンシ／リソース使用量のモデルを作り（Auto-DNN のステージ1）、そのうえで Bundle を積み木のように積んだネットワーク全体を、20エポックの簡易学習で精度を概算しながら **group-based particle swarm optimization (PSO)**（あるいは stochastic coordinate descent、論文によって表現が異なる）で探索する。適合度関数は `Fit = Accuracy + α × (Estimated_HW_metric - Target)`（α<0）という、精度とハードウェア指標（レイテンシ等）を単一のスカラーに合成した形になっており、ハードウェア制約を満たさない候補を早期に切り捨てる。

SkyNet 固有の設計判断（MLSys'20論文）では、DAC-SDC のデータセット統計（"91%の物体が画像全体の9%未満のサイズ、31%は1%未満"）から **小さい物体の検出精度を上げるには高解像度・低レベルの特徴を検出ヘッドまで残す必要がある**と分析し、dw3 出力（bypass、192→768ch）を dw5 出力（512ch）と concat する構成（Model C、パラメータ数1.82MB）を、bypassなし（Model A, 1.27MB）・単純bypass（Model B, 1.57MB）と比較した上で最終採用している。ReLU6 の採用理由も精度目的だけでなく、"出力レンジが [0,6] に制限されるため元のReLU（[0,∞)）よりも少ないビット数で表現できる" という **量子化を見据えたハードウェア都合の活性化関数選択**として説明されている。

量子化に関しては、MLSys'20論文 Table 7 に FM/weight のビット幅を総当たりで振った実験結果があり（float32 baseline IoU 0.741 → 9bit FM/11bit weight で IoU 0.727 (-1.4%) → 8bit FM/10bit weight で IoU 0.680 (-8.2%)）、**精度劣化を許容範囲(-1.4%程度)に抑えつつビット幅を削れる組み合わせを実験的に選定**しており、この 9bit FM / 11bit weight がそのまま本リポジトリの `FIX_FM`/`FIX_WT` の型定義に反映されている。GPU 側 (TX2) は cuDNN の効率が良いため量子化せず float32 のまま推論している、という対比も論文に明記されている。


---

## 11. Key takeaways（手書きHLS量子化オンチップ設計への転用ポイント）

- **型定義に量子化を埋め込む**: 実行時 scale レジスタではなく `ap_fixed<W,I,AP_RND,AP_SAT>` の型そのものを「量子化フォーマット」として設計し、学習側で事前にそのビット幅を意識した精度検証（本件では table形式の総当たり実験）を行ってから HLS の型として固定する。requantize は明示コードを書かず、型変換（代入・キャスト）に任せる。
- **アキュムレータは入力より広いビット幅にし、丸めは"層の出口"（活性化関数と同時）でだけ行う**。ただし積和木の深さによって「途中で毎回丸める（depthwise 3x3, `dwconv3x3.cc:41`）」か「最後にまとめて丸める（pointwise 1x1, `conv1x1.cc:118`）」かで誤差特性が変わることに留意する。どちらの方式を採るかは意図的に選び、`.range()` によるビット直接コピー（`local_buf_copy`のような）はドキュメント化しないと後で意味不明になる。
- **`#pragma HLS ALLOCATION instances=<func> limit=1 function` による PE の強制共有**は、レイヤー形状が均質（同じ conv 種別の繰り返し）なネットワークで面積を切り詰める強力な手法。1つの PE 関数をチャンネル数・タイル数だけ違うループ回数で何度も呼び出す設計にしておけば、合成時に自動で1インスタンスへ縮退できる。
- **PE の並列度はチャンネル方向の `array_partition complete` + `unroll`、内部の縮約は加算木**という組み合わせ（`conv1x1.cc`）は、量子化CNN・AutoEncoderのconv/transposed-conv両方に転用しやすい定型パターン。
- **重み・大きな中間バッファはDDR常駐、オンチップは1タイル分の作業バッファのみ**というメモリ階層は、BRAM/URAMが限られる小型SoC FPGAで現実的な落とし所。m_axiのバス幅（512bit）に重みパッキングを合わせておくと、DMAバーストが効率化する。
- **バッチ処理を空間タイルとして扱う**（4枚を2x2ステッチして1枚として流す）ことで、制御ロジックを増やさずにスループットを稼ぐ手法は、AutoEncoderでも複数サンプルをまとめて1枚のfeature mapとして扱えるなら流用できる。
- **skip/bypass 接続はオンチップの持ち回しではなくDDR上の固定オフセットで管理**し、必要なら中間層の重み側をオフラインで並べ替えて、活性化側のレイアウト変換コストをゼロにする（`reorder_weight.cc` の "mysterious dw3" reorder）。AutoEncoderのdecoder側でencoder特徴を使う場合も同様の技が使える。
- **非線形な後処理（sigmoid/exp相当）はPL側で固定小数点実装せず、PS/ホスト側にfloatで逃がす**という判断は、AutoEncoderの損失計算や活性化の一部が特殊関数を要する場合に、リソース対効果を考えるうえで参考になる。
- **正規化・活性化テーブルは事前計算LUT化**（`img_norm_ch[256]`）し、実行時の割り算・浮動小数点演算を排除する。

---

## 未確認・不明点（推測で埋めなかった箇所）

- `local_buf_copy()`（`net_hls.cc:760-771`）および `load_and_reorg_part()` 内の `buf_out_4` への `.range(WT_RG,0)` 書き込み（`net_hls.cc:720-738`）が、`FIX_FM_acc`（小数部8bit）と `FIX_WT`/`FIX_FM` 由来のビットパターン（小数部6bitまたは異なるスケール）の間でビット単位コピーを行っている点について、数値的な正しさ（2進小数点が実際に一致しているのか、意図的な近似なのか）はソースの静的読解だけでは確認できなかった。
- BatchNorm パラメータの fuse（畳み込み重み・バイアスへの折り込み）を行うスクリプト自体は本リポジトリの `FPGA/` 以下には含まれておらず、`weights_floating.bin` が生成される過程（`GPU/` 側のどのスクリプトが対応するか）は未特定。
- 実機の DSP/BRAM/LUT 利用率レポート（`.rpt` 等）はリポジトリ内に存在せず、本ドキュメントの資源使用率の数字はすべて論文（MLSys'20 Table等、PYNQ-Z1向けのDAC'19実験値含む）からの引用であり、Ultra96向けの実合成レポートそのものではない。
