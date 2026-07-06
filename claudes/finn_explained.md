# FINN / finn-hlslib 解説ドキュメント

対象読者: AMD/Xilinx Vitis HLS / Vivado HLS で、量子化 AutoEncoder 用の専用アクセラレータをフルスクラッチの C++ で書こうとしているハードウェアエンジニア。`claudes/skynet_explained.md`（SkyNet）・`claudes/tvm-vta_explained.md`（TVM-VTA）・`claudes/design_philosophy.md` を読了済みという前提で書く。

目的: 先の2文書（SkyNet, VTA）はいずれも「オンチップのみ・DDR不使用」というユーザーの最重要条件を満たしていなかった（`design_philosophy.md` §0, §1）。FINN は AMD/Xilinx 自身が、量子化ニューラルネットワークを**全て on-chip（BRAM/URAM/LUTRAM）に載せ、層間を on-chip の AXI-Stream FIFO で結ぶストリーミング・データフロー・アクセラレータ**として設計するために作ったフレームワークであり、この欠落を埋める参照モデルとして本ドキュメントを作成する。

参照ソース（すべて `/mnt/d/dv/hls-study/finn-hlslib/` 以下、行番号は今回読んだ時点のもの）:
- `bnn-library.h`, `README.md`
- `mvau.hpp`, `mac.hpp`, `convlayer.h`, `vvau.hpp`
- `slidingwindow.h`, `streamtools.h`, `eltwise.hpp`
- `activations.hpp`, `weights.hpp`, `interpret.hpp`
- `pool.hpp`, `normalize.hpp`（低優先度、概要のみ）
- `tb/conv_top.cpp`, `tb/channelwise_op_top.cpp`, `tb/eltwise_top.cpp`, `tb/dup_stream_top.cpp`, `tb/data/config.h`
- 論文3本: FINN (arXiv:1612.07119, FPGA'17), FINN-R (arXiv:1809.04570), Benchmarking QNN on FPGAs with FINN (arXiv:2102.01341)
- CICADA/AXOL1TL 関連: arXiv:2411.19506, arXiv:2108.03986（hls4ml、finn-hlslibとは別ツール）

---

## 1. Overview — FINN と finn-hlslib の関係、何が「手書き」で何が「自動生成」か

**finn-hlslib（本リポジトリ）は、量子化ニューラルネットワークの各種レイヤーをテンプレート化した、Xilinx社員（Giulio Gambardella, Thomas B. Preusser, Christoph Doehring ほか）が手で書いた Vitis HLS C++ テンプレートライブラリである**（`README.md:3`: "This repo contains the Vitis HLS C++ library for the hardware acceleration of Quantized Neural Networks (QNN) using FINN"）。中身は `mvau.hpp`, `convlayer.h`, `vvau.hpp`, `slidingwindow.h`, `streamtools.h`, `activations.hpp` などの `.hpp`/`.h` ファイル群で、いずれも C++ テンプレート関数・クラスとして実装されている（`bnn-library.h:52-60` が主要ヘッダを束ねるアグリゲータ）。**これは正真正銘の「手書き HLS C++」であり、ユーザーが求める "hand-written HLS" 要件を満たす一次資料として扱える。**

一方、"FINN" という名前が指すもう一段上の層——ONNX/QONNX 形式のネットワーク記述を受け取り、各レイヤーに対して上記テンプレートをどのパラメータ（SIMD, PE, ビット幅など）でインスタンス化するかを決定し、レイヤー同士を1つのトップ関数として結線する **Python コンパイラ（`Xilinx/finn` リポジトリ、本環境には未クローン）**は、finn-hlslib とは別物である。学習側のフロントエンド（Brevitas: PyTorch向け量子化認識学習ライブラリ、あるいは QONNX 形式一般）が出力した量子化済みネットワークを、FINNコンパイラが「どのテンプレート関数をどの折り畳み度・ビット幅で呼ぶか」という形にコード生成し、最終的に本リポジトリの `.hpp` を `#include` した1つの合成可能な HLS プロジェクトを吐き出す、という関係になっている。

**ユーザーへの含意**: 「手書き HLS」という要件に照らすと、**finn-hlslib のテンプレート関数そのもの（MVAU, VVAU, sliding window, thresholding など）はそのまま参考にしてよい/流用してよい一次資料**である。一方で「どのレイヤーにどの SIMD/PE を割り当てるか」「レイヤー同士をどう連結してトップ関数にするか」という**ネットワーク全体の配線・パラメータ決定はFINPythonコンパイラの仕事であり、今回クローンされていないため本ドキュメントの対象外**。ユーザーがAutoEncoderを完全手書きするなら、このテンプレート層をお手本にしつつ、レイヤー連結とパラメータ選定は自分で（コンパイラなしで）行うことになる——これはむしろSkyNet的な「手でループ回数・チャンネル数を書き下す」スタイルに近い。

---

## 2. 完全オンチップのストリーミング・データフロー・アーキテクチャ

finn-hlslib の各レイヤー関数は例外なく `hls::stream<ap_uint<W>>` 型の引数を入出力に取る（例: `mvau.hpp:93-94` の `Matrix_Vector_Activate_Batch(hls::stream<TI> &in, hls::stream<TO> &out, ...)`、`convlayer.h:107-108` の `ConvLayer_Batch(hls::stream<ap_uint<InStreamW>> &in, hls::stream<ap_uint<OutStreamW>> &out, ...)`）。これらは HLS 内部の FIFO（`hls::stream`）でしかなく、m_axi のような DRAM 経由のメモリマップドインタフェースは一切登場しない。

実際にトップ関数を合成する例が `tb/conv_top.cpp:55-58` にある。

```cpp
void Testbench_conv(stream<ap_uint<IFM_Channels1*INPUT_PRECISION> > & in,
                     stream<ap_uint<OFM_Channels1*ACTIVATION_PRECISION> > & out,
                     unsigned int numReps){
#pragma HLS DATAFLOW
    ConvLayer_Batch<...>(in, out, PARAM::weights, PassThroughActivation<ap_uint<16>>(), numReps, ap_resource_dsp());
}
```

`#pragma HLS DATAFLOW`（`tb/conv_top.cpp:56`）を1つのトップ関数にかけ、その中で複数のレイヤー関数（`ConvLayer_Batch`, `Vector_Vector_Activate_Batch`, `Thresholding_Batch`, `StreamingEltwise` など）を `hls::stream` でパイプライン状に連結すれば、Vitis HLS が自動的に各レイヤーを独立したハードウェアブロックとして並行実行するデータフロー領域を生成する。`convlayer.h` の `ConvLayer_Batch` 自体も内部で `StreamingDataWidthConverter_Batch → ConvolutionInputGenerator → Matrix_Vector_Activate_Batch → StreamingDataWidthConverter_Batch` という4段をストリームで繋ぎ（`convlayer.h:117-127`）、`#pragma HLS INLINE`（`convlayer.h:113`）で親のDATAFLOW領域へ展開される、という構造になっている。**DDR/m_axiが一切現れない**点は、`tb/`以下の全テストベンチ（`conv_top.cpp`, `channelwise_op_top.cpp`, `eltwise_top.cpp`, `dup_stream_top.cpp` 等）に共通しており、finn-hlslibのテンプレート関数の入出力は例外なく `hls::stream` である。

**重みの格納場所**: `weights.hpp` の `BinaryWeights`（`weights.hpp:66-98`）・`FixedPointWeights`（`weights.hpp:110-148`）はいずれも `ap_uint<SIMD*WT::width> m_weights[PE][TILES]`（`weights.hpp:69`, `113`）という**静的なC++配列**であり、コンストラクタで初期値を与えれば合成時にROM/BRAM/LUTRAMとして実体化される（`ap_resource_bram`等の選択はスライディングウィンドウのバッファについては明示されているが——後述§7——weights.hpp自体には`BIND_STORAGE`のようなメモリ資源選択プラグマは付いていない。実際にBRAM/LUTRAM/レジスタのどれになるかはHLSツールの推定に委ねられているか、FINNコンパイラが生成するトップレベルラッパー側でプラグマを追加している可能性が高いが、finn-hlslib単体のソースからは確認できなかった)。いずれにせよ**重み配列は DRAM 上のポインタではなく、関数にコンパイル時定数として渡される静的配列**であり、これがそのままオンチップメモリとして合成される。

なお、**finn-hlslib には「重みをストリームから読む」バリアントも存在する**: `Matrix_Vector_Activate_Stream_Batch`（`mvau.hpp:210-308`）、`Vector_Vector_Activate_Stream_Batch`（`vvau.hpp:189-276`）、`Thresholding_Stream_Batch`（`activations.hpp:336-385`）がそれで、`vvau.hpp:166` のコメントに "The weights are supplied from a stream input to **facilitate memory-compute decoupling**" と明記されている。これは重み全体をオンチップの静的配列として持てない大規模モデル向けの逃げ道であり、採用すればストリーム経由でDRAM等外部から重みを供給することも可能になる。**ユーザーがオンチップオンリーを厳守するなら、この streaming-weight バリアントではなく `BinaryWeights`/`FixedPointWeights` の静的配列バリアントを使うべき**、という選択がここで明示的に必要になる。

**SkyNet・VTAとの対比**: `design_philosophy.md` §1 の表が示す通り、SkyNetは重み全体・層間中間特徴マップの双方をDDR常駐にし（`skynet_explained.md` §7）、VTAは命令列・uop列・重み・活性化・バイアス・出力の全てをDRAM経由でロード/ストアする（`tvm-vta_explained.md` §5）。FINNはこれらと対照的に、**静的配列＋`hls::stream`＋`#pragma HLS DATAFLOW`という3点セットだけで、重み・層間活性化の双方を一度もDRAMに触れさせずに1つのビットストリーム内で完結させる**ことを標準の設計としている。これが本ドキュメントをSkyNet/VTAに追加する理由そのものである。

---

## 3. リソース時分割（"Folding"）— mvau.hpp の SIMD/PE 構造

FINNのリソース時分割の中核は `mvau.hpp:93-180` の `Matrix_Vector_Activate_Batch` にある。テンプレート引数 `MatrixW`（重み行列の幅=入力次元）, `MatrixH`（高さ=出力次元）, `SIMD`（1サイクルで並列処理する入力チャンネル数）, `PE`（1サイクルで並列処理する出力チャンネル数）から、以下の2つの「折り畳み数（fold）」が導出される（`mvau.hpp:100-106`）。

```cpp
unsigned const  NF = MatrixH / PE;   // 出力側の折り畳み数 (neuron fold)
unsigned const  SF = MatrixW / SIMD; // 入力側の折り畳み数 (synapse fold)
...
unsigned const TOTAL_FOLD = NF * SF;
```

これはまさにSkyNetの `CI_N`/`CO_N`（`net_hls.cc`）やVTAの命令駆動ループに相当する「同じ物理演算器を何回再利用するか」を表すカウンタである。メインループ (`mvau.hpp:123-179`) は `reps * TOTAL_FOLD` 回、`#pragma HLS pipeline style=flp II=1`（`mvau.hpp:124`）でII=1にパイプライン化されて回り、1回のイテレーションで:

1. 新しい入力ベクトルを読むか(`nf==0`)、前に読んだ`inputBuf[sf]`を再利用するか(`mvau.hpp:126-135`)を切り替える——同じ入力ベクトルをPE方向の異なる出力行に対して`NF`回使い回すため。
2. `weights.weights(tile)`（`weights.hpp`の`TileIndex`経由）で現在のタイルの重みを取得し、`pe`ループを`#pragma HLS UNROLL`（`mvau.hpp:150-151`）で`PE`本並列展開し、各PEが`mac<SIMD>(...)`（`mac.hpp:163-172`）を呼ぶ（`mvau.hpp:155`）。`mac<SIMD>`自体も内部で`i`を`0..SIMD-1`まで`#pragma HLS unroll`（`mac.hpp:167-168`）で並列展開し、`SIMD`個の積和を1段の加算に融合する。
3. `sf`が`SF`に達したら出力を1回`out.write()`し、`nf`を進めて次の出力行グループへ（`mvau.hpp:159-178`）。

つまり **1サイクルあたり `SIMD × PE` 個のMAC演算がハードウェア的に空間展開され、`MatrixW × MatrixH`（=行列全体のMAC数）をこの`SIMD×PE`ブロックで割った回数（`TOTAL_FOLD`）だけ、同一のハードウェアが時分割で再利用される**。これはSkyNetの「32(出力ch並列)×16(積和木)=512 MACを全レイヤーで時分割共有」（`skynet_explained.md` §5）と設計思想としては同型だが、FINNの場合は**レイヤーごとに個別の物理インスタンスを`ConvLayer_Batch`/`Vector_Vector_Activate_Batch`として合成し（`#pragma HLS DATAFLOW`で層ごとに空間的に並べる）、各レイヤー内部だけで`SIMD`/`PE`折り畳みを行う**という点がSkyNetの「全レイヤー共通の`ALLOCATION limit=1`関数」と異なる。FINNの折り畳みは「1つのレイヤーの中で、その層の行列サイズを`SIMD×PE`のタイルへ分解して時分割する」粒度であり、「複数レイヤーで同じ物理関数インスタンスを共有する」というSkyNet流のレイヤー横断的な`ALLOCATION`共有はfinn-hlslib自体には見当たらない（これは`design_philosophy.md`が示唆する折衷案——演算パターンごとに専用PE関数を用意し`limit=1`で縮退——とは別のレイヤー、すなわち「1つのレイヤー内部の時分割」の話である点に注意)。

`vvau.hpp`（depthwise、§8参照）も全く同じ`NF`/`SF`/`TOTAL_FOLD`の構造を持つが`SIMD=1`固定という制約がある（`vvau.hpp:92`の`static_assert(SIMD == 1, "SIMD parallelism not yet supported.")`）。

MMV（Multiple output pixel per cycle, 複数出力画素の並列計算）という3つ目の折り畳み軸もあり、`convlayer.h:259-309`の`ConvLayer_Batch_MMV`がこれを使う。SIMD/PEがチャンネル方向の並列度なのに対し、MMVは空間方向（複数の出力画素を同時に計算する）の並列度であり、`mvau.hpp`の`Matrix_Vector_Activate_Batch`テンプレート引数`MMV`（`mvau.hpp:89`）がこれに対応する。

---

## 4. 量子化スキーム — 任意ビット幅の `ap_int`/`ap_uint` とテンプレートによる型注入

FINNはもともとBNN(Binarized Neural Network)向けに始まったプロジェクトで、その名残が`interpret.hpp`に色濃く残っている。`Binary`クラス(`interpret.hpp:76-93`)は`ap_uint<1>`を`{-1,+1}`として解釈し直す薄いラッパー、`XnorMul`クラス(`interpret.hpp:58-69`)は2値同士の乗算をXNOR相当の分岐(`m_val==b?1:0`)に置き換える——これは掛け算器を使わずに済ませる、2値ネットワーク特有の最適化である。

しかし**finn-hlslibはビット幅を`ap_int<N>`/`ap_uint<N>`のテンプレートパラメータとして扱っており、Nは1(binary)に限定されない**。実際のテストベンチ設定`tb/data/config.h:11-13`では

```cpp
#define INPUT_PRECISION 8
#define ACTIVATION_PRECISION 16
```

という8bit入力・16bit活性化の非バイナリ構成が使われており、`tb/conv_top.cpp:57`は`Slice<ap_uint<INPUT_PRECISION>>`, `Slice<ap_uint<ACTIVATION_PRECISION>>`という`TSrcI`/`TDstI`型注入によって、同じ`ConvLayer_Batch`テンプレートを任意ビット幅で使い回している。この`TSrcI`/`TDstI`/`TWeightI`という3つのテンプレートパラメータ(`mvau.hpp:90-91`, `convlayer.h:100-102`)が、入力・出力・重みそれぞれの「ストリーム上のビットパターンをどう解釈するか」を注入するインタフェースになっており(`Identity`=素通し、`Slice<T>`=T型として切り出し、`Binary`/`XnorMul`=2値解釈)、**ビット幅そのものはユーザーが与える型`T`のテンプレート引数として自由に選べる**設計である。したがって、ユーザーが必要とする「8bit, 16bit, あるいはそれ以上の量子化ビット幅」は、FINNの2値・3値ヘリテージに関わらず`ap_int<N>`/`ap_uint<N>`を素直に選べば良い——これはFINN自身が実証済みの汎化である(§9のFINN-R論文が明示的に扱うテーマでもある)。

アキュムレータ型は`mvau.hpp:113`の`decltype(activation.init(0,0)) accu[MMV][PE]`のように、活性化クラス(`TA`)の`init()`の戻り値型から推論される——つまりアキュムレータの型(ビット幅)は活性化クラス側(`activations.hpp`)が決め、`mac<SIMD>`(`mac.hpp:163-172`)がその型`T`で加算する。桁あふれを防ぐための「入力より広いアキュムレータ」という設計判断自体はSkyNet/VTAと同じ発想だが、FINNではこれを活性化クラスのテンプレートパラメータとして明示的に選択する形になっている。

---

## 5. Requantization — Thresholding/MultiThreshold: 比較としての再量子化

これがFINN最大の特徴的技法であり、`activations.hpp`に実装がある。

### 仕組み

`ThresholdsActivation`(`activations.hpp:200-222`)は次のような構造を持つ。

```cpp
template<unsigned NF, unsigned PE, unsigned NumTH,
     typename TA, typename TR, int ActVal = 0, typename Compare = comp::less<TA, TA>>
class ThresholdsActivation {
public:
  TA m_thresholds[PE][NF][NumTH];
  ...
  TR activate(unsigned const nf, unsigned const pe, TA const &accu) const {
    TR result = ActVal;
    for (unsigned int i = 0; i < NumTH; i++) {
#pragma HLS unroll
      result += Compare()(m_thresholds[pe][nf][i], accu);
    }
    return result;
  }
};
```
(`activations.hpp:203-221`、`Compare`のデフォルトは`comp::less<TA,TA>`, `activations.hpp:65-66, 90-95`で`a<b`を`ap_uint<1>`として返す)

つまり、あるチャンネル(`pe`,`nf`)に対して`NumTH`個の閾値`m_thresholds[pe][nf][0..NumTH-1]`を昇順に並べておき、積和結果`accu`がそのうち何個の閾値を上回ったかを**単純に数え上げる**(`result += (threshold[i] < accu)`を`NumTH`回)。これは「乗算+シフト+クリップ」による典型的なrequantizeとは全く異なるアプローチで、**閾値比較の総和という組合せ論理(コンパレータの木)だけで、任意の単調非減少な量子化関数(=アキュムレータの値域を離散レベルへ写像する関数)を表現する**。`NumTH`個の閾値でちょうど`NumTH+1`段階の出力レベルを区別できるので、出力が`TR`型でN段階(`ActVal`からの相対値)を表現したいなら`NumTH = N-1`個の閾値を用意すればよい。

このメカニズムを層として使うのが`Thresholding_Batch`(`activations.hpp:278-312`)で、`ConvLayer_Batch`のようなMVAU/VVAUの後段に単独のストリーミング層として繋ぐ(§2のDATAFLOW領域内の1ステージになる)。閾値を静的配列でなくストリームから供給する`Thresholding_Stream_Batch`(`activations.hpp:336-385`)もある——これも§2で見た「メモリ・演算のデカップリング」のバリエーション。

### なぜ乗算器が要らないのか

累積和`accu`とスケール・バイアス・非線形活性化(ReLUなど)を経て次段の量子化レベルへ変換する処理は、通常「シフト(スケーリング)→バイアス加算→非線形→クリップ→丸め」という複数ステップの**算術演算**を要する(VTAの`SHR→MAX/MIN→ビット切り出し`, `tvm-vta_explained.md`§4)。FINNの閾値方式は、この一連の変換全体を**学習後にオフラインで計算した`NumTH`個の閾値の集合**に**あらかじめ畳み込んでおき**、ハードウェア側は単純な大小比較の総和だけで済ませる。閾値はBatchNormのスケール・バイアス・非線形活性化・requantizeスケールをすべて合成した結果の「アキュムレータ値域上のどこで出力レベルが1段上がるか」という境界点そのものであり、乗算器はおろか加算器すら不要(比較器と1ビットカウンタの木だけ)——これが"multiplier-free requantization"の核心である。

### コスト: 閾値テーブルサイズはビット幅に対してどう伸びるか(実際に確認した内容)

`ThresholdsActivation::m_thresholds`は`TA m_thresholds[PE][NF][NumTH]`(`activations.hpp:204`)という**PE×NF×NumTH個のオンチップ・レジスタ/BRAM**であり、`activate()`内の`NumTH`回のループは`#pragma HLS unroll`(`activations.hpp:217`)で完全並列展開される——つまり**`NumTH`個の比較器が空間的に並び、その出力を`NumTH`入力の加算木に接続する**回路になる。

出力が`W`ビットで`2^W`段階の量子化レベルを区別する必要がある場合、これらの段階すべてを区別するには理論上`NumTH = 2^W - 1`個の閾値が必要になる(2値=1閾値, 3値=2閾値, 2bit=3閾値, ...という等差の性質から導かれる、コード自体から読み取れる事実)。**これは`W`に対して指数的に増加する**——8bit出力なら255個の閾値・255個の比較器・255入力の加算木が`PE×NF`個ぶん必要になり、16bit出力なら65535個というオーダーになって明らかに非現実的である。

FINNが元々ターゲットにしてきた2値・3値・せいぜい4〜8bit程度の量子化(§9のFINN-R論文が扱う範囲、`W^x A^y`表記でxもyも概ね1〜8)では、この閾値数(最大でも255程度)は現実的なLUTコストに収まる。しかし**ユーザーが「より高ビット幅の量子化」(例えば12bitや16bitの中間表現)を検討しているなら、閾値比較方式をそのビット幅の出力全体に単純適用するのは非現実的**、というのが実際にコードを読んで確認できる誠実な結論である。実務的な回避策としては、(a) 閾値方式は活性化関数出力(通常はReLU等を経て比較的低ビットに落ちる部分)にだけ使い、それ以外の層間受け渡しは`ap_fixed`的な単純ビットシフト+飽和で済ませる、(b) 閾値の段数を出力ビット幅全体でなく「意味のある折れ点の数」(区分線形近似の折れ点数)に絞る、といった設計判断が必要になる。**これはFINNのソース自体は答えを持っておらず、ユーザーが自分のビット幅要件に応じて判断すべき論点**として明記しておく。

### SkyNet/VTAとの対比

SkyNetは`ap_fixed`の暗黙キャストによる自動丸め・飽和(`skynet_explained.md`§3-4)、VTAは`SHR`(右シフト)+`MIN`/`MAX`(クリップ)という明示的な整数演算列(`tvm-vta_explained.md`§4)でrequantizeする。FINNの閾値方式は、この両者のどちらとも異なる**「requantizeを算術ではなく比較として表現する」第三の設計**であり、特に低〜中ビット幅の量子化においては乗算器・シフタを一切使わずにBatchNorm+非線形活性化+requantizeを1段の比較器ロジックへ統合できる点が最大の利点である。

---

## 6. Skip connection — `DuplicateStreams` と `AddStreams`

`streamtools.h`に両方の実装がある。

### 分岐: `DuplicateStreams`

```cpp
template<unsigned int DataWidth, unsigned int NumTotal>
void DuplicateStreams(hls::stream<ap_uint<DataWidth> > & in, hls::stream<ap_uint<DataWidth> > & out1,
        hls::stream<ap_uint<DataWidth> > & out2) {
    for (unsigned int i = 0; i < NumTotal; i++) {
#pragma HLS pipeline style=flp II=1
        ap_uint<DataWidth> e = in.read();
        out1.write(e);
        out2.write(e);
    }
}
```
(`streamtools.h:584-597`)

ドキュメンテーションコメントに **"Used to generate the inputs to the bypass and convolutional branches in Resnet-50"**(`streamtools.h:574`)と明記されている通り、ResNet系のskip connection分岐点そのものを想定した関数である。`DuplicateStreams_Batch`(`streamtools.h:613-621`)は複数画像分をラップするだけの薄いバッチ版。

### 合流: `AddStreams` / `AddStreamsLayer_Batch`

```cpp
template <unsigned int NumChannels, typename In1_t, typename In2_t, typename Out_t,
          unsigned int NumTotal, int offset = 0>
void AddStreams(hls::stream<ap_uint<NumChannels * In1_t::width>> &in1,
                hls::stream<ap_uint<NumChannels * In2_t::width>> &in2,
                hls::stream<ap_uint<NumChannels * Out_t::width>> &out) {
  for (unsigned int i = 0; i < NumTotal; i++) {
#pragma HLS pipeline style=flp II=1
    ...
    for (unsigned int j = 0; j < NumChannels; j++) {
#pragma HLS UNROLL
      In1_t op1 = e1(...); In2_t op2 = e2(...);
      Out_t sum = op1 + op2 + offset;
      ...
    }
    out.write(e);
  }
}
```
(`streamtools.h:639-662`、コメント: **"Used to merge the outputs of the bypass and convolutional branches in Resnet-50"**, `streamtools.h:699`)

`In1_t`/`In2_t`/`Out_t`をそれぞれ独立のテンプレート型引数にしているため、**両ブランチのビット幅・符号が異なっていても(例えば片方が8bit、もう片方が16bitでも)自然に加算し、`Out_t`という別の(通常はより広い)型へrequantizeせずに素通しできる**——ここでも「型をテンプレートで注入する」というFINN共通のイディオムが効いている。`offset`パラメータ(デフォルト0)はバイアス値を同時に加算できる汎用性を持たせたもの。

`AddStreamsLayer_Batch`(`streamtools.h:714-732`)はさらに、チャンネル方向の並列度`PECount`が入力ストリームの並列度`NumChannels`と異なる場合に、`StreamingDataWidthConverter_Batch`で両ストリームをいったん`PECount`幅へ折り畳んでから`AddStreams_Batch`を呼び、出力を再び`NumChannels`幅へ戻す、という「Add演算のPE折り畳み」を実現している。

### レイテンシ不一致(バッファリング)についての誠実な報告

ユーザーの懸念通り、skip接続は「メインパス(畳み込み等を経由する側)」と「バイパスパス(素通りする側)」でレイテンシが大きく異なりうる。**しかし`streamtools.h`の`DuplicateStreams`/`AddStreams`自体のコードには、これを吸収するための明示的なFIFO深さ指定(`#pragma HLS STREAM depth=...`)は一切見当たらなかった**——両関数とも単純な`hls::stream`変数の読み書きのみで、深さはデフォルト値(Vitis HLSでは通常小さい固定値)に委ねられている。深さ指定の実例としては`normalize.hpp:105`の`#pragma HLS stream variable=buffer depth=FM_SIZE`(1フレーム分の特徴マップ全体を一時保持するバッファ、`max_norm`関数内で最大値スキャンのために使われる、§6のskip接続とは無関係の用途)があるのみで、skip connection用のバイパスFIFOの深さをどう決めるかについての規約はfinn-hlslib自体には見当たらなかった。

**結論として、finn-hlslibのテンプレート関数レベルでは skip 接続のレイテンシバランシング(バイパス側FIFOを十分な深さに取ってストールを防ぐ)は解決されておらず、これは(a) FINN Pythonコンパイラがトップレベルの`hls::stream`インスタンスに対して`#pragma HLS STREAM depth=`を生成時に付与しているか(本リポジトリの範囲外なので確認不能)、(b) ユーザー自身がConv側のパイプライン深さを見積もってバイパスFIFOの深さを手動で決める必要がある、のどちらかである。** これは「未確認・不明点」として後述にも明記する。

---

## 7. Sliding window / line buffer — `slidingwindow.h` によるDRAム無しの畳み込み

`ConvolutionInputGenerator`(`slidingwindow.h:159-*`)がim2col相当の変換をストリーム上で行う、DRAM無し畳み込みの心臓部である。

まずバッファの実装先資源を選択できる4つの`memory_resource`オーバーロードがある(`slidingwindow.h:72-138`):

```cpp
template <typename T> void memory_resource(T inputBuf, ap_resource_dflt const&){
#pragma HLS BIND_STORAGE variable=inputBuf type=RAM_2P
}
template <typename T> void memory_resource(T inputBuf, ap_resource_bram const&){
#pragma HLS BIND_STORAGE variable=inputBuf type=RAM_S2P impl=BRAM
}
template <typename T> void memory_resource(T inputBuf, ap_resource_uram const&){
#pragma HLS BIND_STORAGE variable=inputBuf type=RAM_S2P impl=URAM
}
template <typename T> void memory_resource(T inputBuf, ap_resource_lutram const&){
#pragma HLS BIND_STORAGE variable=inputBuf type=RAM_S2P impl=LUTRAM
}
```
(`slidingwindow.h:73-137`)

呼び出し側が`ap_resource_bram()`/`ap_resource_uram()`/`ap_resource_lutram()`/`ap_resource_dflt()`のいずれかを渡すことで、ラインバッファをBRAM・URAM・LUTRAMのどれに固定するかを明示的に選べる——ターゲットデバイスのオンチップメモリ種別(Versal等のURAM搭載デバイスか、小型ZynqのBRAM/LUTRAMのみか)に応じてユーザーが選択する設計である。

本体`ConvolutionInputGenerator`(`slidingwindow.h:159-260`)は次のロジックでラインバッファ(リングバッファ)を構成する。

```cpp
const unsigned int number_blocks = ConvKernelDim/Stride + 1;
ap_uint<SIMD*Input_precision> inputBuf[number_blocks][Stride * IFMDim * multiplying_factor];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
memory_resource(inputBuf, r);
```
(`slidingwindow.h:175-178`)

つまり、**畳み込みカーネルの高さ`ConvKernelDim`をストライド`Stride`で割った数+1個ぶんの「ラインブロック」を保持するリングバッファ**を持ち(`ARRAY_PARTITION dim=1 complete`でブロックごとに独立したメモリポートを持たせる)、新しい1行を読み込みながら、既に溜まっている過去`ConvKernelDim`行ぶんから畳み込みウィンドウを切り出して出力する、という**古典的なライン・バッファ方式**である。メインループ(`slidingwindow.h:190-260`)は「初期ブロック充填フェーズ(`inp < IFMDim*ConvKernelDim*multiplying_factor`)」と「定常フェーズ(1サイクルで新しい入力画素を1つ読み込みつつ、過去データからウィンドウを1つ出力する)」の2フェーズに分かれており、入力ストリームからの読み出しと出力ウィンドウの生成が同一ループ内で並行に進む(`slidingwindow.h:239-256`の`if`ブロックが「書き込み」、`slidingwindow.h:209-238`が「読み出し(ウィンドウ生成)」)。

このバッファのサイズは`number_blocks × Stride × IFMDim × multiplying_factor`(`multiplying_factor = IFMChannels/SIMD`)であり、**特徴マップ全体ではなく「カーネル高さ分+α」の行数だけをオンチップに保持する**のが要点——これはSkyNetの「44×84タイル」のようなタイル単位のワーキングバッファ(`skynet_explained.md`§7)と同じ発想だが、FINNは**タイル境界という概念自体を持たず、入力画像全体を1本のストリームとして最初から最後まで流し切る**(SkyNetのようにDDR上のタイルを都度ロードし直す必要がない)。これによりconvolutionは完全にDRAM無しで、入力ストリーム→ラインバッファ→ウィンドウ出力→MVAUという一直線のデータフローとして完結する。

---

## 8. Depthwise conv (VVAU) — `vvau.hpp`

`Vector_Vector_Activate_Batch`(`vvau.hpp:80-157`)はMVAUと同じ`NF`/`SF`/`TOTAL_FOLD`の折り畳み構造を持つが、決定的な違いは**`SIMD`が常に1に固定される**点である。

```cpp
static_assert(SIMD == 1, "SIMD parallelism not yet supported.");
unsigned const  NF = Channels / PE;
unsigned const  SF = Kernel_2;   // Kernel*Kernel、SIMDが常に1なのでカーネル画素数と一致
```
(`vvau.hpp:92, 96, 101`)

そして計算本体では、MVAUの`mac<SIMD>`(複数入力チャンネルにわたる積和)の代わりに、1チャンネル分のスカラー乗算だけが行われる。

```cpp
auto const &w = weights.weights(tile);
for(unsigned pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
  auto const wgt = TWeightI()(w[pe]);
  for (unsigned mmv = 0; mmv < MMV; mmv++){
    auto const act = TSrcI()(inElem, mmv);
    accu[mmv][pe] += mul(wgt[0], act(pe,mmv), r);   // wgt[0]: SIMD=1なので単一要素
  }
}
```
(`vvau.hpp:126-134`)

つまりMVAUが「入力チャンネル軸にわたる内積(通常のconv/linearの意味論)」を計算するのに対し、VVAUは**チャンネルをまたいだ積和を一切行わず、各PE(=各出力チャンネル)が自分自身に対応する1本の重みベクトル(カーネル内の`Kernel_2`要素)とだけ内積を取る**——これがdepthwise separable convの数学的定義(チャンネルごとに独立したカーネル)そのものであり、MVAUとVVAUの違いはまさに「チャンネル方向の畳み込み(reduction)があるかないか」に一致する。SkyNetの`conv1x1`(pointwise, MVAU相当)と`dwconv3x3`(depthwise, VVAU相当)という2種類のPEを使い分ける設計(`skynet_explained.md`§5)と、構造的に対応関係がある。

---

## 9. 論文が語る設計思想

### FINN (arXiv:1612.07119, FPGA'17)

原論文の核心は、**「1個の汎用エンジンを命令で使い回す」のではなく、「各レイヤーに専用のハードウェアブロックを割り当て、そのブロックの並列度(=折り畳み度)をユーザーが要求するスループットに合わせて調整する」というヘテロジニアスなストリーミング・アーキテクチャ**を提案した点にある。各レイヤーのハードウェアは重み・活性化をオンチップメモリに統合して持ち、外部メモリアクセスを最小化する。2値化ネットワークにおいては、浮動小数点の乗算をXNORとpopcountという極めて軽量な演算に置き換えられる、という2値化特有の利点も強調されている。ZC706ボード上で消費電力25W未満という条件下、MNISTで1230万分類/秒・レイテンシ0.31µs(精度95.8%)、CIFAR-10/SVHNで21906分類/秒・レイテンシ283µs(精度80.1%/94.9%)という数字を報告しており、論文は「この種のベンチマークにおいて当時報告された中で最速の分類率」だと主張している。

### FINN-R (arXiv:1809.04570)

FINN-RはFINNを2値専用から**任意/混合精度(`W^x A^y`表記、x=重みビット数, y=活性化ビット数)へ一般化**したフレームワークである。Section 3.2で導入される折り畳みパラメータは3つ: `P`(PE数)・`Q`(SIMD数)・`M`(MMV数、同時に処理する出力画素数)で、これらは「レイヤーの性能を広い範囲でスケールさせるが、同時にハードウェアコストへも直接影響する」("allow to scale the performance of a layer implementation in a wide range but also affect the hardware costs directly")と説明されている。Section 4.4のバランシングアルゴリズム(Algorithm 1)は「最も律速しているレイヤーの折り畳みパラメータを反復的に緩めていく("systematically widens the most pressing bottlenecks")」という貪欲法で、デバイスの資源予算を使い切るまでレイヤーごとの並列度を自動調整する——これはfinn-hlslib自体には存在しない、**FINN Pythonコンパイラ側が担う自動チューニング機能**である(§1参照、ユーザーが完全手書きするなら、この自動調整は自分の判断で代替する必要がある)。Section 4.2の"Streamlining"パスは、BatchNormの吸収を2値ネットワークに限定せず一般化したもので、「量子化層の直前にあるスケーリング層を1つの線形変換へ畳み込み("collapsing scaling layers in front of a quantization layer into a single linear transform")、それを多段閾値(multi-level threshold)へ吸収する("absorbs floating-point scaling factors into multi-level thresholds")」——これがまさに§5で見た`ThresholdsActivation`が受け取る閾値の生成元である。AWS-F1で50TOp/s、組み込みデバイスで5TOp/sという、当時「前例のない実測スループット("new unprecedented measured throughput")」を報告している。

### Benchmarking QNN on FPGAs with FINN (arXiv:2102.01341)

この論文はFINNとBrevitas(量子化認識学習)を組み合わせ、2〜8bitの重み・活性化精度と複数の並列化設定でQNNをベンチマークしたもの。**PE/SIMDの折り畳み設定だけがスループット変動の原因であり、重み・活性化のビット幅自体はスループットに大きな影響を与えない**(ビット幅はオンチップメモリ消費量には影響するがデータパスの並列度там自体は変えない、という趣旨)ことを実験的に示している。また、RadioMLのTransformerのような大きなモデルでは、保持すべき重みの絶対量が大きいため、BRAM/URAMの消費が支配的な制約になる、という指摘もある。**ただし本ドキュメントの執筆時点で、この論文はPDFのフルテキスト抽出ができず(利用環境にPDFレンダリング用のpoppler-utils等が入っておらず、WebFetch経由でも本文の構造化テキストが得られなかった)、上記はWeb検索結果の要約からの引用であり、具体的な章番号・表番号・BRAM/URAMの絶対量などの一次確認はできていない**。この点は末尾の「未確認・不明点」に明記する。

**3本を通じた設計思想の対比(SkyNet/VTAとの違い)**: SkyNetは「1つの固定ネットワークのために、全レイヤー共通のPEをコンパイル時に1個へ縮退させ、DDRを容量の逃げ道にする」設計(`design_philosophy.md`§2)、VTAは「1つの汎用GEMM/ALUコアを命令ストリームで時分割し、DRAMを事実上のワーキングメモリにする」設計(同)。対してFINNは、**「レイヤーごとに専用のハードウェアブロックを空間的に並べ(SkyNetの全レイヤー単一PEとは逆)、各ブロック内部だけをSIMD/PEで折り畳み、全体を1つのDATAFLOW領域としてオンチップのストリームで結線する」**という第三の道を取る。VTAのような命令セット・コンパイラは持たず(FINNのPythonコンパイラは折り畳みパラメータを選ぶだけで、実行時の命令発行は行わない——ハードウェアは合成後は完全に固定パイプラインとして動く)、SkyNetのような「DDRを大容量スクラッチパッドとして許容する」判断も取らない、という点が本ドキュメントを追加する最大の理由になっている。

---

## 10. CICADA/AXOL1TL — hls4mlベースのオンチップ量子化AutoEncoder実運用例(参考)

CERN/CMS実験のLevel-1トリガーで実際に稼働している異常検知システム、**CICADA**(畳み込みオートエンコーダ、18×14の2チャンネルカロリメータ画像を入力とする)と**AXOL1TL**(変分オートエンコーダ、潜在ベクトルは`N(μ=0,σ=1)`の8次元)は、いずれも**QKeras による量子化認識学習(bit-exactな学習)を行い、hls4mlでHLSコードへ変換し、Xilinx Virtex-7 FPGA上で完全オンチップ推論として展開されている**(arXiv:2411.19506)。関連する先行研究(arXiv:2108.03986, "Autoencoders on FPGAs for real-time, unsupervised new physics detection at 40 MHz at the Large Hadron Collider")では、同種のオートエンコーダをXilinx Virtex UltraScale+ VU9P上に実装し、**推論レイテンシ約80ns、VU9Pのロジックリソース使用率3%未満**という具体的な数字が報告されている。CMSのLevel-1トリガーはイベントあたり40MHz・数µsのレイテンシ制約下で動作するため、これらのオートエンコーダは他の判定ロジックと同居しながら、**推論の全経路がFPGA上で完結し、DDRとの往復を挟まない**ことが必須要件になっている。

**重要な注記**: これらはfinn-hlslibではなく**hls4ml**という別のコンパイラ/自動生成ツール(Kerasモデルを層ごとのHLSコードへ自動変換する、FastMLコミュニティのツール)を使っており、本ドキュメントの主題である「finn-hlslibの手書きHLSテンプレート」とは直接の実装上の関係はない。ここで触れる意義は、**「量子化オートエンコーダを完全オンチップでFPGAに実装する」というユーザーの設計目標そのものが、CERNの実運用システムで既に実証済みの、仮説ではなく実在するパターンである**ことを示す傍証としてのみである。

---

## 11. ユーザーの手書きオンチップHLS AutoEncoderへの示唆

### そのまま転用できるパターン

- **閾値ベースのrequantization(`ThresholdsActivation`/`Thresholding_Batch`, `activations.hpp:200-312`)**: BatchNorm・非線形活性化・requantizeスケールを学習後にオフラインで1本の閾値配列へ畳み込み、実行時は比較器の総和だけで済ませる。乗算器・シフタが不要になる利点は、特に中間層(ReLU等でクリップされる層)のrequantizeにそのまま使える。ただし§5で述べた通り、閾値数は出力ビット幅に対して`2^W - 1`のオーダーで増えるため、**高ビット幅の出力にそのまま適用するのは避け、低〜中ビット幅の中間活性化に限定して使う**のが妥当。
- **`DuplicateStreams`/`AddStreams`/`AddStreamsLayer_Batch`によるskip接続(`streamtools.h:584-732`)**: 分岐は`in.read()`して2つの出力ストリームへ`write()`するだけ、合流はチャンネルごとの単純加算。型をテンプレート引数として独立に注入できるため、両ブランチのビット幅が異なっていても素直に扱える。
- **`ConvolutionInputGenerator`のラインバッファ(`slidingwindow.h:159-260`)**: 特徴マップ全体ではなく「カーネル高さ+α」の行数だけをリングバッファとして保持し、入力ストリームを流しながら畳み込みウィンドウを生成する。BRAM/URAM/LUTRAMのどれに実装するかを`memory_resource`(`slidingwindow.h:72-138`)経由で明示的に選べる点も含め、DRAM無しconvのお手本になる。
- **`SIMD`/`PE`折り畳み(`mvau.hpp:100-179`)による「1レイヤー内部の」リソース時分割**: `NF = MatrixH/PE`, `SF = MatrixW/SIMD`という2重の折り畳みと、それを1つの結合ループへ平坦化して`II=1`パイプライン化する書き方(`mvau.hpp:122-124`)は、ユーザーがConv/Linear用のPE関数を書く際にそのまま使える定型パターン。
- **MVAU(通常conv/linear)とVVAU(depthwise)の使い分け(`mvau.hpp` vs `vvau.hpp`)**: チャンネル方向のreductionがあるかないかで関数を分ける、という構造化はAutoEncoderのEncoder/Decoderが通常convとdepthwise-separable convの両方を持つ場合にそのまま対応づけられる。
- **`hls::stream`+`#pragma HLS DATAFLOW`によるレイヤー間連結(`tb/conv_top.cpp:56`ほか全テストベンチ)**: SkyNetの「DDR経由の疎結合」・VTAの「4モジュール+FIFO手動配線」のいずれとも違う、**単一トップ関数+DATAFLOW領域+hls::stream**という最も素直な構成がFINNの標準スタイルであり、`design_philosophy.md`§5・§8がすでに推奨していた方向性と一致する。

### FINNの2値・3値ヘリテージに由来する、そのままでは使えない/要注意な部分

- **閾値テーブルのサイズは出力ビット幅に対して指数的に増える(`2^W-1`個)**——これは実際にコード(`activations.hpp:203-221`)を読んで確認した事実であり、FINNが元々2値・3値・低ビット量子化を主戦場にしてきたことの直接の帰結。ユーザーが高ビット幅の中間表現(例えば12〜16bit)を使うなら、閾値方式をそのビット幅全体に適用するのは非現実的で、閾値方式を使う範囲(低ビットの活性化出力)と、単純なシフト+飽和(SkyNet/VTA流)で済ませる範囲を層ごとに切り分ける必要がある。
- **`Binary`/`XnorMul`(`interpret.hpp:58-93`)は2値専用の最適化であり、ユーザーの量子化ビット幅がそれより大きいなら使わない**——ただし他の型注入インタフェース(`Identity`, `Slice<T>`)は任意ビット幅に対応しているため、これらだけを使えば2値ヘリテージの影響は実質的に避けられる。
- **`Matrix_Vector_Activate_Stream_Batch`/`Vector_Vector_Activate_Stream_Batch`/`Thresholding_Stream_Batch`という「重み・閾値をストリームから読む」バリアント(`mvau.hpp:210-308`, `vvau.hpp:189-276`, `activations.hpp:336-385`)は、重みが完全にオンチップに収まらない場合の逃げ道であり、オンチップオンリーを厳守するユーザーは使うべきではない**——`BinaryWeights`/`FixedPointWeights`の静的配列バリアント(`weights.hpp:66-148`)を使うこと。
- **FINN Pythonコンパイラが担う自動折り畳みパラメータ探索(FINN-R Section 4.4のバランシングアルゴリズム)は、finn-hlslib自体には存在しない、コンパイラ側の機能である**。ユーザーが完全手書きでSIMD/PE/MMVを選ぶ場合、この自動チューニングに相当する判断(どのレイヤーが律速するか、資源予算内でどう折り畳み度を配分するか)を自分で行う必要がある——SkyNetが実験的に(論文中のTable探索で)ビット幅・構成を決め打ちしたのと同じ立場になる。
- **finn-hlslibのレイヤー横断的なリソース共有(SkyNetの`ALLOCATION limit=1`のような、複数レイヤーが同一物理関数インスタンスを共有する仕組み)は標準では見当たらない**——FINNの折り畳みは「1レイヤー内部」の時分割であり、「Convレイヤーが10個あるので同じPE関数を10回呼ぶ」というSkyNet的な発想をFINNのテンプレートの上に重ねて実現するのはユーザー自身の設計判断になる。
- **skip接続のレイテンシバランシング(バイパスFIFOの深さ)は、finn-hlslibのテンプレートコード自体には規定がない(§6)**——DATAFLOW領域のデフォルトFIFO深さで足りない場合、`#pragma HLS STREAM depth=`を自分で見積もって付与する必要がある。

---

## 未確認・不明点(推測で埋めなかった箇所)

- `weights.hpp`の`BinaryWeights`/`FixedPointWeights`の`m_weights`配列自体には、`slidingwindow.h`の`memory_resource`のような明示的な`BIND_STORAGE`/メモリ資源選択プラグマが付いていない。実機合成でBRAM/LUTRAM/レジスタのどれになるかはHLSツールの推定に委ねられているか、FINN Pythonコンパイラが生成するトップレベルラッパー側で別途プラグマを付与している可能性があるが、finn-hlslib単体のソースからは確認できなかった。
- Skip connection用のバイパスFIFOの深さ(レイテンシバランシング)について、finn-hlslibの`DuplicateStreams`/`AddStreams`自体には明示的な`#pragma HLS STREAM depth=`指定がなく、これがFINN Pythonコンパイラ側で自動決定されるのか、通常はデフォルト深さのままで実用上問題にならない設計になっているのか、あるいはユーザーが手動で調整すべき項目として位置づけられているのか、本リポジトリの範囲内では確認できなかった。
- arXiv:2102.01341("Benchmarking Quantized Neural Networks on FPGAs with FINN")は、今回の実行環境にPDFレンダリング用のpoppler-utils等がインストールされておらず(ネットワーク経由のパッケージインストール権限もなし)、WebFetch経由でも本文の構造化テキストの抽出に失敗した。そのため同論文についての記述(§9)はWeb検索結果の要約に基づくものであり、具体的な章番号・表番号・BRAM/URAM消費量の絶対値などの一次確認はできていない。
- 閾値数`NumTH`が実際のFINN運用でどの程度の値(何bit相当)まで使われているかについて、finn-hlslibのテストベンチ(`tb/`)内に具体的なビット幅と閾値数を結びつけた実例は見つけられなかった(`tb/channelwise_op_top.cpp`は`Thresholding_Batch`のテンプレート機構を閾値比較ではなく`ChannelWiseOperation`による単純な乗算・加算に転用した例であり、`ThresholdsActivation`そのものを使うテストベンチのファイル名を`tb/`以下で特定できなかった)。`2^W-1`という閾値数の見積もりは`ThresholdsActivation::activate()`のロジック(`activations.hpp:213-221`)から直接導いた理論値であり、実際のFINN運用実績と突き合わせたものではない。
- FINN Python コンパイラ(`Xilinx/finn`リポジトリ)は本環境にクローンされておらず、「どのレイヤーにどのSIMD/PE/ビット幅を割り当てるか」「トップ関数の配線をどう自動生成するか」の具体的な実装は未調査。本ドキュメントの記述はfinn-hlslib(テンプレート層)とFINN-R論文(コンパイラの設計思想の記述)から再構成したものであり、実装コードそのものは確認していない。
