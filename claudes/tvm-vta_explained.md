# TVM-VTA (Versatile Tensor Accelerator) 解説ドキュメント

対象読者: AMD/Xilinx Vitis HLS / Vivado HLS で、量子化 AutoEncoder 用の専用アクセラレータをフルスクラッチの C++ で書こうとしているハードウェアエンジニア。FPGA/HLS 量子化DNN設計の背景知識はほぼ前提にしない。対象は「オンチップのみ・DDR無し・特定ネットワーク専用のハンドライトHLS」。

このドキュメントは `/mnt/d/dv/hls-study/tvm-vta` にクローン済みのリポジトリを実際に読んで書いている。ファイルパス・行番号は原則 `file:line` の形式で明記する。コードの識別子・pragma名・論文引用は原文のまま記載する。

---

## 1. Overview — VTA とは何か

VTA (Versatile Tensor Accelerator) は Apache TVM プロジェクトのサブプロジェクトで、**汎用的・パラメトリックに設定可能な量子化 CNN 推論アクセラレータのテンプレート**であり、TVM コンパイラスタックと co-design されたハードウェア/ソフトウェア一式である。リポジトリの `README.md:22` に明記されている通り:

> "VTA (versatile tensor accelerator) is an open-source deep learning accelerator complemented with an end-to-end TVM-based compiler stack."

**重要: VTA は特定のオートエンコーダ用アクセラレータではない。** 畳み込み(GEMMとして実装)・活性化・プーリング・正規化などを行う汎用の行列積コアであり、どのネットワーク層に対しても TVM コンパイラが命令列(インストラクション+マイクロオペレーション)を生成してこのハードウェアに流し込む、という使い方をする。AutoEncoder 特有のハードウェアは存在しない。

それでも参考にする価値がある理由:
- ISA(命令セット)、4段パイプライン、依存関係キュー、GEMMコア、ALUコアによるrequantizationなど、**HLSで書かれた量子化DNNアクセラレータの成熟した設計パターン集**として非常に具体的で読みやすい。
- Vivado HLS (`hardware/xilinx/src/vta.cc`) が751行というコンパクトな量で、fetch/load/compute/store の全体が把握できる。
- 論文 (arXiv:1807.04188) に明確な設計思想があり、なぜそう設計したかの理由付けが学べる。

**ユーザーのターゲットとの重要な相違点(先に明示しておく)**: 実際にコードを読んで検証した結果、VTA は下記の点でユーザーの要件(オンチップのみ・手書きHLS・単一ネットワーク固定)から大きく外れている。

1. **VTA は DRAM 前提の Load/Store DMA アーキテクチャである。オンチップのみではない。** `hardware/xilinx/src/vta.h:118-211` の `fetch`/`load`/`compute`/`store` の関数シグネチャは全て `volatile bus_T *inputs` 等、DRAM上のデータへの AXI マスターポート引数を取る。オンチップ SRAM (`inp_mem`, `wgt_mem`, `out_mem`, `uop_mem`, `acc_mem`) は「DRAMから読み出したタイルを一時的に保持するワーキングセット」であり、恒久的な重み格納場所ではない。実機の Vivado ブロックデザイン (`hardware/xilinx/scripts/vivado.tcl:289-324`) では Zynq PS (Processing System) の `M_AXI_GP0`/`S_AXI_ACP` 等を介して DDR に接続しており、命令列そのものも DRAM 上に置かれ `fetch` が読み出す設計になっている。
2. **VTA は「手書きでネットワーク固有のHLSを書く」のではなく、コンパイラがJIT生成した命令ストリームをinterpretする汎用CPUに近い。** ハードウェア自体(`vta.cc`)は 1 回だけ HLS で合成し、ネットワークが変わっても再合成しない。ネットワークごとの違いは全て「命令列(LOAD/STORE/GEMM/ALU/FINISHとそのマイクロオペコード)」という形でランタイムに供給される。これはユーザーが目指す「1つの固定ネットワーク用にC++を手で書き込む」という設計方針とは正反対の思想である。
3. 命令自体、及びマイクロオペコード(uop)は `uops` ポインタ経由でDRAMから読み込まれる (`hardware/xilinx/src/vta.cc:483-486`)。つまり「計算そのもの」だけでなく「何を計算するかの記述(命令)」もオフチップに置かれる。

これらの相違点は 5, 9, 10 節でさらに具体的に掘り下げる。

---

## 2. パイプラインアーキテクチャ — FETCH → LOAD → COMPUTE → STORE

VTA のハードウェアコアは `hardware/xilinx/src/vta.cc` の4つの独立した HLS 関数として実装されている:

- `fetch()` : `hardware/xilinx/src/vta.cc:132-167`
- `load()` : `hardware/xilinx/src/vta.cc:169-234`
- `compute()` (内部で `gemm()`, `alu()` を呼ぶ) : `hardware/xilinx/src/vta.cc:415-515`
- `store()` : `hardware/xilinx/src/vta.cc:517-563`

### 命令の流れ

`fetch()` は DRAM 上の命令列 (`insns`) を1つずつ読み、opcode を見て3種類の `hls::stream<insn_T>` ハードウェアFIFOのいずれかへ書き込む(`hardware/xilinx/src/vta.cc:145-166`):

```cpp
if (opcode == VTA_OPCODE_STORE) {
  store_queue.write(raw_insn);
} else if (opcode == VTA_OPCODE_LOAD) {
  if (memory_type == VTA_MEM_ID_INP || memory_type == VTA_MEM_ID_WGT) {
    load_queue.write(raw_insn);
  } else {
    gemm_queue.write(raw_insn);   // UOP/ACC(bias) の LOAD は compute 側が処理
  }
} else {
  gemm_queue.write(raw_insn);     // GEMM/ALU/FINISH は全部 gemm_queue
}
```

つまり論理的には3ステージ(load / compute(=gemm) / store)への振り分けだが、TVM のドキュメントでも「3-stage load-compute-store task pipeline」と表現されている(fetchは命令の振り分け役)。

### 実機での構成: 4つの独立したHLS IP

重要なのは、これら4関数は**1つの巨大なステートマシンとして合成されるのではなく、`hardware/xilinx/scripts/hls.tcl` で個別に `set_top` して4本の別々のIPコアとしてHLS合成される**という点である(`hardware/xilinx/scripts/hls.tcl:100-138`: `vta_fetch`, `vta_load`, `vta_compute`, `vta_store` という4つのプロジェクトがそれぞれ `csynth_design` + `export_design -format ip_catalog` される)。

その4つのIPは `hardware/xilinx/scripts/vivado.tcl` でVivado IP Integrator(ブロックデザイン)上で AXI-Stream FIFO IP (`fifo_generator`)を介して物理的に接続される。例えば:

```tcl
connect_bd_intf_net -intf_net fetch_0_load_queue_V_V \
  [get_bd_intf_pins fetch_0/load_queue_V_V] [get_bd_intf_pins load_queue/S_AXIS]
connect_bd_intf_net -intf_net load_queue_M_AXIS \
  [get_bd_intf_pins load_0/load_queue_V_V] [get_bd_intf_pins load_queue/M_AXIS]
```
(`hardware/xilinx/scripts/vivado.tcl:332,338`)

つまり `fetch`→`load`→`compute`→`store` の命令キューは HLS の `#pragma HLS INTERFACE axis port = ...` (`hardware/xilinx/src/vta.cc:140-142` 等)によって各IPの外部AXI-Streamポートとして現れ、Vivado側で明示的にFIFO IPを挟んで結線される。これは「1つのトップ関数の中で dataflow pragma を使う」設計ではなく、**別々に合成した独立モジュールをRTLレベルで手動配線する**という、Vitis HLS の `dataflow` region よりも粗粒度の構成である点に注意(ユーザーが素直に1トップ関数+`#pragma HLS DATAFLOW`で書く場合とは前提が異なる)。

なお `vta()` 関数 (`hardware/xilinx/src/vta.cc:565-751`) は上記の4モジュールを1つの関数内で呼び出して依存関係を解決しながらシミュレーションするための**シミュレーション専用のラッパー**であり、ヘッダのコメントにも明記されている:

> "VTA wrapper for simulation purpose only. Orchestrates dataflow execution of the fetch, load, GEMM and store stages." (`hardware/xilinx/src/vta.h:214-215`)

実機合成では使われない。

### 依存関係キュー(dependency FIFO)

4ステージ間のデータ依存(例:computeがloadした値を使う前にloadが完了している必要がある、storeがcomputeの結果を書き出す前にcomputeが完了している必要がある、など)は、命令ストリームとは別の **bool型の小さなハードウェアFIFO** で明示的にハンドシェイクされる。これは VTA の設計上の特徴で、命令自体に「前段が終わるまで待て/終わったら通知しろ」というフラグを持たせている。

`hardware/xilinx/src/vta.cc:169-234` の `load()` を見ると、4本の依存キューのうち2本 (`g2l_dep_queue`, `l2g_dep_queue`) を扱う:

```cpp
// Pop dependence token if instructed
if (insn.pop_next_dep) {
  g2l_dep_queue.read();
}
...
// Push dependence token if instructed
if (insn.push_next_dep) {
  l2g_dep_queue.write(1);
}
```

これは命令フォーマット (`include/vta/hw_spec_const.h` が定義する `VTAMemInsn` 構造体、`include/vta/hw_spec.h:84-115`) に埋め込まれた4つの1bitフラグ `pop_prev_dep` / `pop_next_dep` / `push_prev_dep` / `push_next_dep` (`include/vta/hw_spec.h:88-94`) によってコンパイラ側が明示的にスケジューリングする。VTA全体では以下の4本の依存キューがある(`hardware/xilinx/src/vta.h` の引数、`hardware/xilinx/src/vta.cc:598-606` のインスタンス化箇所を参照):

- `l2g_dep_queue` : load → gemm(compute)
- `g2l_dep_queue` : gemm(compute) → load
- `s2g_dep_queue` : store → gemm(compute)
- `g2s_dep_queue` : gemm(compute) → store

各FIFOの深さは `STREAM_IN_DEPTH = 8` (`hardware/xilinx/src/vta.h:39`, シミュレーション用)。実機のVivadoブロックデザインでは命令キューは16byte幅・深さ512、依存キューは1byte幅・深さ1024の `fifo_generator` IP として生成される(`hardware/xilinx/scripts/vivado.tcl:203-218`のコメント)。

つまり VTA のパイプライン間通信は「命令ストリーム(データ本体)」+「トークンベースの依存関係FIFO(同期用)」という2レイヤー構造になっている。これは、あるステージが他ステージの完了を待たずに独立して進める場合(依存フラグが立っていない命令)は自由に並列実行させ、依存がある場合だけブロックする、という**ソフトウェアパイプライニングをコンパイラが命令に埋め込み、ハードウェアはただトークンを見てストールするだけ**、という設計思想を反映している。

---

## 3. リソースの時分割・使い回し — GEMMコア/ALUコアはネットワーク全体で共有される

VTA の compute モジュール (`hardware/xilinx/src/vta.cc:415-515`) は、**1つの物理的なGEMMコアと1つの物理的なALUコアだけを持ち**、ネットワークの全レイヤーがこれを命令ストリームによって時分割で使い回す。

```cpp
} else if (insn.generic.opcode == VTA_OPCODE_GEMM) {
    gemm(raw_copy, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem);
} else if (insn.generic.opcode == VTA_OPCODE_ALU) {
    alu(raw_copy, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem);
}
```
(`hardware/xilinx/src/vta.cc:502-505`)

`gemm()` (`hardware/xilinx/src/vta.cc:236-325`) と `alu()` (`hardware/xilinx/src/vta.cc:327-413`) はどちらも同じ `compute()` 関数から呼ばれる普通のC++関数で(`#pragma HLS INLINE` によりcomputeにインライン展開される。`hardware/xilinx/src/vta.cc:243,334`)、レイヤーごとに別のハードウェアインスタンスがあるわけではない。1回のGEMM命令・ALU命令の中でも、`iter_out`/`iter_in`という2重ループ+uop列でさらに時分割実行する(`hardware/xilinx/src/vta.cc:253-324` の `EXE_OUT_LOOP`/`EXE_IN_LOOP`/`READ_GEMM_UOP`)。つまり:

- **層内の空間展開の粒度は固定**: 1サイクルで計算できるのは `VTA_BATCH × VTA_BLOCK_IN × VTA_BLOCK_OUT` の行列積(既定コンフィグでは 1×16×16 = 256 MAC、後述)だけ。
- **それより大きい層(畳み込みのico/oco/カーネル位置など)は、全て同じ物理GEMMコアに対して命令(GEMM命令のiter_out/iter_in/uopループ)を繰り返し発行することで実現する。**
- ネットワークが変わっても、あるいは層のサイズが変わっても、ハードウェア(GEMMコア/ALUコア)は一切変更されず、TVMコンパイラが生成する命令列(=どのuopを何回ループさせるか)だけが変わる。

これは、たとえば SkyNet のように**層ごとに空間的に専用のパイプラインステージ/演算器を敷き詰めるハンドライトなHLSデータフロー設計**(層Aの畳み込み専用ロジック、層Bの畳み込み専用ロジック…と物理的に並べて全体をストリーミングで流す方式)とは対照的である。VTA は「小さな汎用行列積エンジンを1個作り、命令で使い回す」= **時間方向の多重化(temporal multiplexing)** に振り切っている。これはCPU的・von Neumann的な設計であり、SkyNetのような **空間的な多重化(spatial pipelining)** とは思想が逆である。ユーザーが単一の固定ネットワーク用にHLSを手書きするなら、SkyNet的な空間展開(層ごとに演算器を用意しネットワーク全体をストリーム的にパイプライン化する)の方が本来のゴールに近く、VTAの「1つのGEMMコアを命令で使い回す」設計は必ずしも直接のお手本にはならない(9節・10節で詳述)。

---

## 4. 量子化スキーム — 低ビット幅GEMM + 広いアキュムレータ + ALUによるrequantization

### ビット幅パラメータ

ビット幅は `config/vta_config.json` で `LOG_*` 値として設定し、`include/vta/hw_spec_const.h` でシフトして実際のビット幅マクロに展開される。既定の `config/vta_config.json:4-12`(シミュレーション/pynqデフォルト共通)の値と、`include/vta/hw_spec_const.h:24-48` の展開式から計算すると:

| パラメータ | JSON設定値 (LOG_*) | 実際の値 |
|---|---|---|
| `VTA_INP_WIDTH` (入力/活性化) | `LOG_INP_WIDTH=3` | 8 bit (`ap_int<8>`) |
| `VTA_WGT_WIDTH` (重み) | `LOG_WGT_WIDTH=3` | 8 bit |
| `VTA_ACC_WIDTH` (アキュムレータ) | `LOG_ACC_WIDTH=5` | 32 bit |
| `VTA_OUT_WIDTH` (出力/次段への活性化) | `LOG_OUT_WIDTH=LOG_INP_WIDTH` (`config/pkg_config.py:67`) | 8 bit |
| `VTA_BATCH` | `LOG_BATCH=0` | 1 |
| `VTA_BLOCK_IN`=`VTA_BLOCK_OUT` | `LOG_BLOCK=4` (`LOG_BLOCK_IN`/`LOG_BLOCK_OUT` は共に `LOG_BLOCK` から複製, `config/pkg_config.py:65-66`) | 16 |

これらから `hardware/xilinx/src/vta.h:41-63` で以下の型が作られる:

```cpp
typedef ap_uint<VTA_BUS_WIDTH> bus_T;
typedef ap_int<VTA_INP_WIDTH> inp_T;         // int8
typedef ap_int<VTA_WGT_WIDTH> wgt_T;         // int8
typedef ap_int<VTA_OUT_WIDTH> out_T;         // int8
typedef ap_int<VTA_ACC_WIDTH> acc_T;         // int32
typedef ap_int<VTA_WGT_WIDTH+VTA_INP_WIDTH+1> mul_T;                    // 乗算結果, int17
typedef ap_int<VTA_WGT_WIDTH+VTA_INP_WIDTH+VTA_LOG_BLOCK_IN+1> sum_T;    // 内積の部分和, int21
```

つまり **int8 × int8 の乗算を `mul_T`(int17)で受け、`VTA_BLOCK_IN`(=16)要素分の内積を `sum_T`(int21、`LOG_BLOCK_IN=4`だけビットを足した余裕を持つ)で累積し、最終的に `acc_T`(int32)のアキュムレータへ加算する** という、低精度入力・広ビット幅アキュムレータの典型的な量子化GEMM構成になっている(`hardware/xilinx/src/vta.cc:293-306`)。

### requantization(次層への橋渡し)の実装場所

GEMMの出力はまず32bitアキュムレータ `acc_mem` に足しこまれる (`hardware/xilinx/src/vta.cc:302-304` の `accum += (acc_T) tmp;`)。これを次の層の入力として使える8bitに戻す(requantize/rescale)処理は、**GEMMコアの中では行われず、別のALU命令(`VTA_OPCODE_ALU`)として発行される**。ALUコア (`alu()`, `hardware/xilinx/src/vta.cc:327-413`) がサポートする演算は `include/vta/hw_spec_const.h:126-133` で定義されている4種(HLS実装が対応するのはこの4つのみ。後述):

```c
#define VTA_ALU_OPCODE_MIN 0
#define VTA_ALU_OPCODE_MAX 1
#define VTA_ALU_OPCODE_ADD 2
#define VTA_ALU_OPCODE_SHR 3   // Shift right
#define VTA_ALU_OPCODE_MUL 4   // ISA定数としては定義されているが…
```

実際のHLS実装 (`hardware/xilinx/src/vta.cc:378-396`) が処理しているのは `MIN`/`MAX`/`ADD`/`SHR` の4つだけで、`VTA_ALU_OPCODE_MUL` はISA定数として定義されているのみで、Xilinx HLSの `alu()` 関数内には対応する分岐が存在しない(chisel実装や将来拡張用の予約と考えられる。**これは実際にファイルを読んで確認した事実であり、断定できる**)。

Requantizationの典型パターンは、この `SHR`(算術右シフトによるスケーリング)と `MAX`/`MIN`(クリップ)を**別々のALU命令として連続発行する**ことで実現される:

```cpp
} else if (insn.alu_opcode == VTA_ALU_OPCODE_SHR) {
  // Compute Shift Right
  acc_T shr_val = src_0 >> shft_by;
  dst_tensor[i][b] = shr_val;
  o_tensor[i][b] = (out_T) shr_val.range(VTA_OUT_WIDTH - 1, 0);
}
```
(`hardware/xilinx/src/vta.cc:391-396`)

シフト量 `shft_by` は即値(`use_imm`)またはもう一方のテンソルから取得され、`aluop_shr_arg_T`(`VTA_SHR_ARG_BIT_WIDTH = VTA_LOG_ACC_WIDTH` bit, `include/vta/hw_spec_const.h:156`)という型で表現される。クリップ(飽和)は `MIN`/`MAX` を使って `min(max(x, lo), hi)` のように2命令に分けて表現する(コード上に `CLIP` という単一命令は存在しない。ReLU相当は `MAX` に0を即値として与えれば良く、対称なクリップ(例えばrelu6的な上限クリップ)は `MAX`→`MIN` の2段で実現できる、という設計)。

最終的に `o_tensor` (`out_T`=int8) へ書き戻す際は `acc_T.range(VTA_OUT_WIDTH-1, 0)` で下位ビットを単純に切り出しており(`hardware/xilinx/src/vta.cc:384,390,395`)、これはSHR(スケーリング)の後に幅を落とす操作であって、丸め処理(round-to-nearest等)は明示的には行われていない。つまり **VTAのrequantizationは「右シフト(切り捨て)+ min/maxクリップ+下位ビット切り出し」というシンプルな整数演算の組み合わせであり、per-channel scaleの掛け算(乗算によるリスケール)は `MUL` オペコードがISA上定義されていてもXilinx HLS実装には存在しない**、という点は量子化設計上重要な発見である。

---

## 5. オン/オフチップメモリ — SRAMバッファのサイズと、DRAMとの往復

### オンチップSRAMバッファ

`compute()` (`hardware/xilinx/src/vta.cc:415-515`) 内で以下のオンチップメモリが `static` 配列として宣言され、HLSの `#pragma HLS INTERFACE bram port = ...` (`hardware/xilinx/src/vta.cc:435-437` 等)でBRAMインターフェースとして合成される:

- `uop_mem` : `uop_T uop_mem[VTA_UOP_BUFF_DEPTH]` (マイクロオペレーション列, `hardware/xilinx/src/vta.cc:444`)
- `acc_mem` : `bus_T acc_mem[VTA_ACC_BUFF_DEPTH][ACC_MAT_AXI_RATIO]`(アキュムレータ, `hardware/xilinx/src/vta.cc:447`, `#pragma HLS ARRAY_RESHAPE ... complete dim=2` によりII=1を達成)
- `inp_mem`, `wgt_mem`, `out_mem` は `load()`/`compute()`/`store()` に引数として渡され、実体は `vta()` シミュレーションラッパー内、あるいは実機ではVivadoブロックデザイン上の外部BRAM IP (`blk_mem_gen`, `hardware/xilinx/scripts/vivado.tcl:123-141` の `init_bram_property`)として存在する。

サイズは `config/vta_config.json` (既定 = `pynq_sample.json` と同一設定)の `LOG_*_BUFF_SIZE` から `include/vta/hw_spec_const.h:50-57` を通じて決まる。既定設定でのバイトサイズを実際に計算すると:

| バッファ | 設定 (`LOG_*_BUFF_SIZE`) | サイズ |
|---|---|---|
| `VTA_UOP_BUFF_SIZE` | 15 | 32 KB |
| `VTA_INP_BUFF_SIZE` | 15 | 32 KB |
| `VTA_WGT_BUFF_SIZE` | 18 | 256 KB |
| `VTA_ACC_BUFF_SIZE` | 17 | 128 KB |

合計で約 448 KB がオンチップに常駐する(出力バッファ `out_mem` はACCバッファから派生したサイズ `LOG_OUT_BUFF_SIZE = LOG_ACC_BUFF_SIZE + LOG_OUT_WIDTH - LOG_ACC_WIDTH` で計算され、`config/pkg_config.py:68-71` 参照。ビット幅がACCの1/4なのでOUTバッファはACCバッファよりも小さくなる)。

このサイズはPYNQ-Z1(Zynq-7000, xc7z020)のオンチップBRAM容量に収まるよう調整された値であり、**「1つのレイヤーの入力タイル・重みタイル・その出力タイル」を保持するワーキングセットのサイズであって、ネットワーク全体の重みやアクティベーションを格納するものではない**。重み256KBというサイズだけでも、AutoEncoderのような多層モデルの全重みを保持するには通常全く足りない。

### DRAMとの往復(いつ・何が移動するか)

`load()` (`hardware/xilinx/src/vta.cc:169-234`) は `VTA_OPCODE_LOAD` 命令に従い、`inputs`/`weights` という2本の AXIマスターポート(`#pragma HLS INTERFACE m_axi port = inputs ... bundle = data_port`, `hardware/xilinx/src/vta.cc:177-178`)経由でDRAMから2Dストライドアクセス(パディング付き)でタイルを読み出し、`inp_mem`/`wgt_mem` に書き込む。`compute()` も別途 `biases`(バイアス/アキュムレータの初期値)を `VTA_MEM_ID_ACC` としてDRAMからロードし(`hardware/xilinx/src/vta.cc:487-500`)、`uops`(マイクロオペレーション列そのもの)も `VTA_MEM_ID_UOP` としてDRAMから `memcpy` でロードする(`hardware/xilinx/src/vta.cc:482-486`)。`store()` (`hardware/xilinx/src/vta.cc:517-563`) は `out_mem` の内容を `outputs` ポインタ経由でDRAMへ書き戻す。

つまり **命令列自体(`fetch`が読む`insns`)、マイクロオペレーション列(`uops`)、重み(`weights`)、入力活性化(`inputs`)、バイアス/中間アキュムレータ(`biases`)、出力(`outputs`)の6種類全てがDRAM上に存在し、オンチップSRAMは「現在処理中のタイルの一時キャッシュ」に過ぎない**。これはVivadoブロックデザイン (`hardware/xilinx/scripts/vivado.tcl:289-324`) でZynqの `processing_system7`/`zynq_ultra_ps_e` IPの `M_AXI_GP0`(命令/レジスタ制御用)、`S_AXI_ACP`/`S_AXI_HPC0`(データ用、キャッシュコヒーレントポート)に接続されていることからも明らかで、ARM Cortex-A9/A53上で動くホストソフトウェア(ドライバ、`include/vta/driver.h`)が `VTAMemAlloc`/`VTAMemGetPhyAddr`/`VTAFlushCache` (`include/vta/driver.h:65-142`)といった、物理アドレスとキャッシュ管理を扱うDRAMベースのAPIを提供している。**ユーザーが目指す「モデル全体がオンチップに収まるので外部メモリ往復が不要」という設計とは根本的に異なる。**

---

## 6. Skip connection(残差接続)のネイティブサポートについて

`hardware/xilinx/src/vta.cc`、`include/vta/hw_spec.h`、`include/vta/hw_spec_const.h` を読んだ限り、**skip connection / residual connection 専用のハードウェア機構は存在しない**。

根拠:
- ISAのオペコードは `VTA_OPCODE_LOAD`/`STORE`/`GEMM`/`FINISH`/`ALU` の5種類のみ (`include/vta/hw_spec_const.h:115-124`)。
- ALUオペコードも `MIN`/`MAX`/`ADD`/`SHR`(実装上)/`MUL`(定義のみ)の5種類 (`include/vta/hw_spec_const.h:126-135`) であり、「2つの異なる層の出力を足し合わせる」ための専用命令はない。
- ただし `VTA_ALU_OPCODE_ADD` (`hardware/xilinx/src/vta.cc:385-390`) は `acc_mem` 上の任意の2つのインデックス(`dst_idx`, `src_idx`)にある値を足す汎用の加算命令であり、**もし residual の分岐先の値を `acc_mem`(アキュムレータSRAM)上に「スクラッチとして」残しておければ、後続の層の出力とこの `ADD` 命令で汎用的に加算することは可能**である。つまりハードウェア的な専用回路はないが、`acc_mem` を一種のスクラッチパッドとして使い、命令スケジューリング(TVMコンパイラのメモリ確保・命令生成)側でresidual加算をエミュレートする、という設計になっている。

これは「VTAは特定の演算だけでなく任意の量子化されたテンソル演算グラフを表現できる汎用アクセラレータであり、skip connectionのような特定トポロジ用のハードウェアは持たず、全て命令列とメモリスケジューリング(コンパイラの仕事)に委ねている」という設計思想と整合する。ハンドライトHLSで固定のAutoEncoder(仮にskip connectionを持つ構成だとしても)を書く場合は、このパターン(汎用ALU ADD命令+コンパイラによるバッファ管理)をそのまま持ち込む必要はなく、単に「skip元のテンソルを保持するBRAMを確保し、加算するステージを明示的にパイプラインへ挿入する」という直接的な設計で十分である。

---

## 7. ホストとのインターフェース(ストリーミングI/O) — レジスタ制御 + AXI

トップレベルの各モジュールのポートpragmaを見ると(`hardware/xilinx/src/vta.cc:138-143` の `fetch`、`177-184` の `load`、`427-438` の `compute`、`523-528` の `store`):

- **制御レジスタ**: `#pragma HLS INTERFACE s_axilite port = ... bundle = CONTROL_BUS` で、各モジュールごとに独立した AXI-Lite スレーブインターフェース(`CONTROL_BUS`という名前のbundle)を持つ。例えば `fetch` は `insn_count`(命令数)を、`compute` は `done`(完了フラグ)を、AXI-Liteのメモリマップドレジスタとして公開する(`hardware/xilinx/src/vta.h:120-121,163-164` のコメントにも "AXI-lite memory mapped register" と明記)。ホスト(Zynq PS上のARMコア)はこれらのレジスタをポーリング/書き込みして起動・完了確認を行う。
- **命令/データDMA**: `#pragma HLS INTERFACE m_axi port = insns offset = slave bundle = ins_port` (`fetch`, `hardware/xilinx/src/vta.cc:139`)のように、各モジュールが個別の名前付きAXIマスターバンドル(`ins_port`, `data_port`, `uop_port`)を持ち、DRAMへの読み書きを行う。
- **モジュール間通信**: `#pragma HLS INTERFACE axis port = load_queue` 等 (`hardware/xilinx/src/vta.cc:140-142`)で、命令キュー・依存キューがAXI-Streamインターフェースとして各モジュールの外部ポートに現れる。

実機では、4つのIPの `s_axi_CONTROL_BUS` は共通の `axi_interconnect` (`axi_xbar`, `hardware/xilinx/scripts/vivado.tcl:164-170,327-330`) 経由でZynq PSの `M_AXI_GP0` に接続され、各モジュールごとに固有のベースアドレス(`FETCH_BASE_ADDR`, `LOAD_BASE_ADDR`, `COMPUTE_BASE_ADDR`, `STORE_BASE_ADDR`, 例えばpynqでは `0x43C00000`〜`0x43C03000`, `config/pkg_config.py:187-192`)にレジスタがマップされる。データ用AXIマスターは `smartconnect` (`axi_smc0`) を介してPSの `S_AXI_ACP`(キャッシュコヒーレントアクセスポート)に集約される (`hardware/xilinx/scripts/vivado.tcl:157-162,345-350`)。

要するにホストとのインターフェースは典型的な **Zynq PS+PL構成**: 制御はAXI-Liteレジスタのポーリング、データ移動はAXI(フルマスター)経由のDDR DMA、モジュール間はAXI-Stream FIFO、という3種のAXIプロトコルを使い分けている。

---

## 8. 重み・バイアス・スケールの取り扱い

- **重み**: `VTA_OPCODE_LOAD` 命令(`memory_type == VTA_MEM_ID_WGT`)によって `load()` モジュールがDRAMの `weights` ポインタから `wgt_mem` へロードする(`hardware/xilinx/src/vta.cc:219-227`, `load_2d` テンプレート関数使用、パディング無しの単純ストライドコピー)。
- **バイアス/アキュムレータ初期値**: `VTA_OPCODE_LOAD` 命令(`memory_type == VTA_MEM_ID_ACC`)によって `compute()` モジュールが `biases` ポインタから `acc_mem` へロードする(`hardware/xilinx/src/vta.cc:487-500`, `load_pad_2d` テンプレート使用、パディング対応)。
- **マイクロオペレーション(uop)**: `VTA_OPCODE_LOAD` 命令(`memory_type == VTA_MEM_ID_UOP`)によって `uops` ポインタから `uop_mem` へ `memcpy` される(`hardware/xilinx/src/vta.cc:482-486`)。uopは「どのSRAMインデックス同士を演算するか」というアドレスパターンのみを持つ軽量な符号(`VTAUop` 構造体, `include/vta/hw_spec.h:262-269`: `dst_idx`/`src_idx`/`wgt_idx` の3フィールドのみ)であり、演算内容(足す/掛ける等)はGEMM/ALU命令側の `alu_opcode` 等に載る。
- **スケール(量子化パラメータ)の伝達**: ハードウェアには「スケールレジスタ」のような専用の入出力は存在しない。4節で見た通り、層間のリスケールは `VTA_ALU_OPCODE_SHR` の即値(`insn.imm`, `VTAAluInsn.imm` フィールド, `include/vta/hw_spec.h:242`)、または `use_imm=false` の場合は `acc_mem`/他バッファに事前に置かれたテンソルとして渡される。つまり**per-layerのスケール/シフト量は、ハードウェアレジスタではなく「ALU命令の即値フィールド」としてコンパイラが命令列に埋め込む**、という設計である。これは量子化パラメータの受け渡しを完全にソフトウェア(命令生成)側の責務にしている点で、多くの「レジスタでスケール係数を渡す」ハンドライトHLS設計とは異なるアプローチである。

---

## 9. 論文が語る設計思想(arXiv:1807.04188, "A Hardware–Software Blueprint for Flexible Deep Learning Specialization")

論文の核心は「ハードウェアテンプレートとコンパイラスタックの co-design」である。VTAは特定のフレームワーク・モデル・演算子・データ型にロックインされることを避けるために設計されており、論文は "an end-to-end approach requires integration between frameworks, systems, compilers, and architecture" と述べ、ハードウェア単体でなくスタック全体で最適化することの重要性を強調している。これを実現する具体的な仕組みが、VTAの **2段階ISA (two-level ISA)** である。1つは「DMAロードや深層学習演算子のような可変レイテンシの操作を記述する高レベルのCISC的なタスクISA」(本ドキュメントの `VTA_OPCODE_LOAD/STORE/GEMM/ALU/FINISH`)、もう1つは「行列-行列演算を記述する低レベルの固定レイテンシRISC的なマイクロコードISA」(本ドキュメントの uop、`VTA_UOP_GEM_*`/`VTA_UOP_ALU_*` フィールド)である。この2層構造により、ハードウェアは小さく単純な「行列積+ALUの実行エンジン」のままにしておき、多様な演算子(通常の畳み込み・グループ畳み込み・転置畳み込みなど)への対応は全てソフトウェア側(TVMコンパイラが生成する命令・uop列)が担う。

このアプローチはランタイムのJITコンパイラと一体で機能する。TVMランタイムはマイクロカーネル(uop列)を実行時に動的に生成・管理することで、全レイヤー分のマイクロコードが物理的なuopバッファ(既定で32KB)に同時に収まらない大規模モデルにも対応できる。つまりVTAは「ハードウェアを固定してソフトウェアが後から適応する」のではなく、「アルゴリズム・モデル・演算子・数値形式が今後どう進化しても、ハードウェアを変えることなくコンパイラ側の対応だけで追従できる」ことを狙った設計である。量子化についても、ビット幅(8bit入力/重み、32bit以下のアキュムレータなど)を柔軟なパラメータとして持ちつつ、実際のリスケール処理(shift/clip)はTVMのRelay量子化パスがALU命令列として生成するという役割分担になっている。

この思想は、SkyNetのような**特定のネットワーク1つのために全レイヤーを空間的に専用パイプラインとして敷き詰める「固定データフロー」設計**とは哲学的に対極にある。論文自身も導入部で「固定モデル向けアクセラレータは特定ワークロードに対して魅力的な性能を出せる一方、静的な性質のためハードウェア資源の使い回しができず、より大きい・新しいモデルへの対応が制限される」という趣旨の批判を、汎用性を優先する設計選択の動機として述べている。VTAは「性能の一部を犠牲にしてでも、1つのハードウェア(1回だけ合成したビットストリーム)を多様なモデル・多様な世代のネットワークで使い回せること」を最優先した設計であり、SkyNet的な「1つのネットワークに対して性能を振り切った固定回路を毎回合成する」設計とは、そもそも解こうとしている問題が異なる。

(注記: 上記の論文内容はWebFetchによるPDF/要約の抽出結果に基づく。直接ページを目視して一字一句を検証したわけではないため、細部の引用文言については多少の言い回しの誤差があり得る点に留意されたい。ただし co-design・2段ISA・JITマイクロカーネル生成・固定データフローとの対比、という論旨の骨格は、実際のコード構造(2階層の命令フォーマット、`uop_mem`をDRAMから都度ロードする設計、ALU命令によるsw定義のrequantization)とも整合しており、信頼性は高いと判断している。)

---

## 10. 手書きHLS・AMD/Xilinx・オンチップ量子化設計への示唆

### 転用できるパターン

- **低精度GEMM+広ビット幅アキュムレータの型設計**: int8入力・int8重み → 乗算はint17 (`mul_T`) → 内積の部分和はブロックサイズ分のビット余裕を足したint21 (`sum_T`) → 最終アキュムレータはint32 (`acc_T`)、という「桁あふれしない型を明示的に用意する」パターンはそのまま使える (`hardware/xilinx/src/vta.cc:60,63` の型定義、`hardware/xilinx/src/vta.cc:293-306` の実装)。
- **ALUベースのrequantizationパターン(shift + clip)**: 32bitアキュムレータ → 算術右シフトでスケーリング → min/maxでクリップ → 下位ビット切り出しで8bit化、という3段の操作列 (`hardware/xilinx/src/vta.cc:378-396`) は、量子化されたAutoEncoderの各層出力を次層入力へ渡す際にそのまま応用できる汎用的な考え方。「1つの融合命令で全部やる」のではなく「シンプルな演算(SHR/MAX/MIN)の組み合わせで表現する」という発想は、HLSでの検証のしやすさの観点からも参考になる。
- **明示的な依存関係トークンによるステージ間同期**: パイプラインの各ステージが「前段の完了を待つべきか」を明示的なbool FIFO (`hls::stream<bool>`) でハンドシェイクするパターン (`hardware/xilinx/src/vta.cc:169-234` の `g2l_dep_queue`/`l2g_dep_queue` の読み書き) は、AXI-Streamの `empty()`/`read()` に頼るだけでなく、明示的なREADY/DONEトークンで依存関係を管理したい場合の実装例として有用。
- **ping-pong的なタイル処理のテンプレート関数化**: `load_pad_2d`/`load_2d`/`read_tensor`/`write_tensor` (`hardware/xilinx/src/vta.cc:31-130`) のように、SRAM上のタイル読み書きをテンプレート関数として汎用化し、幅の異なるバス(`bus_T`)から狭い型(`inp_T`/`wgt_T`/`acc_T`)へのパッキング/アンパッキングをbit-sliceで行うパターンは、AXIバス幅とデータ型幅が異なる場合の定石として参考になる。
- **`#pragma HLS ARRAY_RESHAPE`と`#pragma HLS DEPENDENCE ... inter false`の併用でII=1を達成する手法**: `acc_mem` に対する `hardware/xilinx/src/vta.cc:448,450` の指定は、アキュムレータSRAMへの読み書きをII=1でパイプライン化するための実践的なテクニックとして転用できる。
- **タイルの2Dストライド+パディングDMAパターン**: `load_pad_2d` (`hardware/xilinx/src/vta.cc:46-73`) が実装する「y方向にストライド送りしつつ、x方向の前後にゼロパディングを挿入する」2次元ロードは、畳み込みのpadding処理をDMA側で吸収したい場合の実装として参考になる(ただしユーザーの設計はDRAM無しなので、これはオンチップSRAM間、あるいは層間バッファ間のコピーロジックとして読み替える必要がある)。

### 転用すべきでない・そのまま真似ると過剰設計になる部分

- **汎用ISA/JITコンパイラは不要**: VTAの本質的な複雑さ(2段命令セット・uopループ・命令依存フラグ・fetchによる命令ディスパッチ)は「1つのハードウェアで無数のネットワーク・無数の層形状に対応する」ためのものである。単一の固定AutoEncoderしか動かさないなら、層構成・ループ境界・チャネル数は合成時に全て定数として埋め込めるため、命令デコーダ(`fetch`)や汎用ループカウンタ付きuop実行系を作る必要はない。各層はループをアンロール/決め打ちしたRTLとして直接書けばよい。
- **DRAMラウンドトリップの再現は不要**: VTAが4節・5節で見たように「命令・uop・重み・活性化・バイアス・出力の全てをDRAM経由でストリーミングする」設計を持つのは、オンチップSRAMにモデル全体が収まらない前提があるため。ユーザーの設計はモデル全体をオンチップに置く前提なので、`load()`/`store()`のようなDRAM DMAモジュールや、AXIマスターポート、`VTAMemAlloc`/物理アドレス管理のようなホストドライバ層は不要で、単純にBRAM/URAM上のバッファ間で直接データを受け渡すデータフロー設計にできる。
- **命令ストリームによる時分割GEMM/ALU共有は必須ではない**: 3節で見たように、VTAは「1個のGEMMコアを命令で使い回す」ことで面積を節約しているが、これは多様な層形状に対応するためのトレードオフである。単一ネットワーク専用なら、SkyNetのように層ごとに専用の演算ユニット・パイプラインステージを空間的に並べて全体を`#pragma HLS DATAFLOW`でストリーミングさせる設計の方が、レイテンシ・スループットの両面で有利になりやすく、こちらが素直な選択肢になる。
- **依存関係トークンによる粗粒度な4モジュール分割は過剰**: VTAが4つの独立したHLS IPに分割し、Vivado上でAXI-Stream FIFOを手動配線しているのは、fetch/load/compute/storeをそれぞれ独立にクロック・独立に最適化・独立に交換可能にするための設計(かつ元々TVMランタイムがレジスタ経由で個別に起動する前提)。オンチップのみの単一トップHLS関数であれば、`#pragma HLS DATAFLOW`とストリーム変数だけで同等のパイプライン並列性を、より少ない記述量で得られる。
- **skip connection用の特別な仕組みは存在しないので探さなくてよい**: 6節で確認した通りVTAには専用ハードウェアはなく、汎用ALU ADD命令+コンパイラによるバッファ管理でエミュレートしているだけである。AutoEncoder設計で残差/skip接続が必要なら、単純に「retainしておくBRAMバッファ+加算ステージ」を素直に設計すればよく、VTAの命令ベースのスクラッチパッド運用を模倣する必要はない。
- **AXI-Lite経由のレジスタ制御・ホストポーリングモデルはそのまま持ち込む必要はない**: VTAはZynqのPS(ARMコア)がホストとして命令列を都度用意し、各IPのAXI-Liteレジスタを介して起動・完了確認するモデルを取っている(7節)。オンチップのみでホストが存在しない、あるいは単純なfree-runningのアクセラレータであれば、この複雑な制御バス構成(`CONTROL_BUS`,`ins_port`,`data_port`,`uop_port`という複数のAXIバンドル分割)は不要で、単純なstart/done信号やごく小さなAXI-Liteの設定レジスタ数個で十分なことが多い。

---

## 付録: 主要ファイルの対応表

| 内容 | ファイル |
|---|---|
| 4段パイプライン本体(fetch/load/compute/store, GEMMコア, ALUコア) | `hardware/xilinx/src/vta.cc` |
| 上記の関数プロトタイプ宣言・型定義 | `hardware/xilinx/src/vta.h` |
| HLS合成スクリプト(4つのIPとして個別合成) | `hardware/xilinx/scripts/hls.tcl` |
| Vivado IP Integratorでの結線(FIFO, BRAM, Zynq PS接続) | `hardware/xilinx/scripts/vivado.tcl` |
| ISA構造体定義(VTAGenericInsn/VTAMemInsn/VTAGemInsn/VTAAluInsn/VTAUop) | `include/vta/hw_spec.h` |
| ビット幅・バッファサイズ・opcode定数のマクロ展開 | `include/vta/hw_spec_const.h` |
| ホストドライバAPI(DRAM確保・物理アドレス・キャッシュ管理) | `include/vta/driver.h` |
| コンフィグ(ビット幅・バッファサイズ・ターゲットボード) | `config/vta_config.json`, `config/pynq_sample.json`, `config/pkg_config.py` |
| ソフトウェアシミュレーションテスト(ALU/GEMMの単体テスト) | `hardware/xilinx/sim/vta_test.cc`, `tests/hardware/common/` |
| Chisel(Scala/RTL手書き)代替実装 — 対照参考 | `hardware/chisel/src/main/scala/core/*.scala` (Compute.scala, TensorGemm.scala, TensorAlu.scala, Fetch.scala, Load.scala, Store.scala 等) |
| Intel/Cyclone V (DE10-Nano)向けラッパー — 対照参考 | `hardware/intel/`, `hardware/intelfocl/src/vta.cl`(OpenCL) |

Chisel版 (`hardware/chisel/src/main/scala/core/`) はVerilog/Scalaで書かれた同一ISAの別実装で、Xilinx HLS版と同じ命令セット・パイプライン構造を踏襲しているが、より詳細なタイミング制御・イベントカウンタ (`EventCounters.scala`) を持つ。Intel/Cyclone V向け (`hardware/intel/`) はDE10-Nano用のRTL統合スクリプト、`hardware/intelfocl/` はOpenCL(`vta.cl`)によるIntel FPGA SDK for OpenCL向け実装であり、いずれも今回のXilinx HLSとの直接比較のためだけに存在を確認したのみで、詳細読解は行っていない。
