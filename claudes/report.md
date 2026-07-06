# 作業ログ（`claudes/instruction.md` 対応）

## 実施内容の概要

`claudes/instruction.md` の指示に基づき、以下を実施した。

1. リポジトリ直下に2つの参照リポジトリをclone
   - `SkyNet/` ← https://github.com/TomG008/SkyNet
   - `tvm-vta/` ← https://github.com/apache/tvm-vta
2. 2本のサブエージェント（general-purpose）を並列起動し、それぞれのHLSソースコード全文＋関連論文を精読させ、モデルごとの解説資料を作成
   - `claudes/skynet_explained.md`（約39KB）
   - `claudes/tvm-vta_explained.md`（約43KB）
3. 上記2文書の内容を突き合わせ、統合的な設計思想書を作成
   - `claudes/design_philosophy.md`（約34KB）
4. 本作業ログを作成
   - `claudes/report.md`（本ファイル）

## 参照した一次資料

- SkyNet: `FPGA/HLS/` 配下のHLS C++ソース全ファイル（`net_hls.cc/.h`, `conv1x1.cc`, `dwconv3x3.cc`, `reorder_weight.cc`, `golden_c.cc`, `output_verify.cc`, `tb.cc`）、`FPGA/Deploy/SkyNet.py`（ホストコード）、`GPU/models.py`（PyTorchリファレンス実装）、`README.md`
- SkyNet関連論文（ar5ivのHTML render経由で全文取得）:
  - MLSys'20: "SkyNet: a hardware-efficient method for object detection and tracking on embedded systems"（arXiv:1909.09709）
  - ICML'19 workshop (Best Poster): "A Bi-Directional Co-Design Approach to Enable Deep Learning on IoT Devices"（arXiv:1905.08369）
  - DAC'19: "FPGA/DNN Co-Design: An Efficient Design Methodology for IoT Intelligence on the Edge"（arXiv:1904.04421）
- tvm-vta: `hardware/xilinx/src/vta.cc`（751行、fetch/load/compute/store全体）、`hardware/xilinx/src/vta.h`、`hardware/xilinx/scripts/hls.tcl`・`vivado.tcl`、`include/vta/hw_spec.h`・`hw_spec_const.h`・`driver.h`、`config/vta_config.json`・`pkg_config.py`
- VTA論文: "A Hardware–Software Blueprint for Flexible Deep Learning Specialization"（arXiv:1807.04188）

## 主な発見・判断ポイント

- **SkyNetは物体検出・追跡ネットワークであり、AutoEncoderではない。** ただしハードウェア設計パターン（量子化型設計、手書きHLS、PE時分割、skip接続の扱い）は、ネットワークの用途とは独立に転用可能なため、「パターン集」として読む方針をそのまま2文書・設計思想書に明記した。
- **VTAは特定ネットワーク専用アクセラレータではなく、TVMコンパイラと co-design された汎用量子化CNNアクセラレータテンプレート。** ハードウェアは1回だけ合成し、ネットワークごとの違いは実行時に供給される命令列（ISA + マイクロオペレーション）で表現する、という設計思想であり、ユーザーが目指す「単一ネットワーク向け手書きHLS」とは方法論として対極にある。
- **最重要の発見**: SkyNet・VTAのいずれも、ユーザーが希望する「オンチップ設計（DDR不使用）」を満たしていない。両者ともBRAM/URAM容量に対してモデルサイズが大きすぎるため、重み・中間活性化の多くをDDR/DRAM経由でやり取りする設計になっている。この点は設計思想書の冒頭（§0, §1）で明示し、「DDR上のバッファ配置」をそのまま真似るのではなく「オンチップBRAMのバッファ配置」に読み替える、という翻訳の必要性を明記した。
- 量子化・requantizationの実装方式は両者で対照的:
  - SkyNet: `ap_fixed`型の丸め・飽和という言語機能に依存した**暗黙的・コンパイル時固定**なrequantize。
  - VTA: `SHR`→`MIN`/`MAX`→ビット切り詰め、という**明示的なALU命令列**によるrequantize（per-channelの乗算リスケールはISA定義のみでHLS実装は存在しないことをコード読解で確認）。
  - 設計思想書では、全体のアーキテクチャ方針はSkyNet寄り（単一ネットワーク固定・手書き）、requantizeの「操作の書き方」はVTA寄り（明示的な段階分け）を推奨する折衷案を提示した。
- Skip接続は両者とも専用ハードウェアを持たず、SkyNetは「DDR上の固定オフセット＋オフラインの重み並べ替え」、VTAは「汎用ALU ADD命令＋コンパイラによるスクラッチ管理」という、いずれもソフトウェア/オフライン側に処理を寄せる設計だった。

## 未解決・要フォロー事項

- SkyNetの `local_buf_copy()` 等における`.range()`ビット直接コピーの数値的妥当性は、ソース読解のみでは確認できず未解決（`skynet_explained.md` 末尾に明記）。
- VTA論文の引用はWebFetch経由の抽出に基づき、一字一句の検証はしていない（論旨の骨格はコード構造と整合しているため信頼度は高いと判断、`tvm-vta_explained.md` §9注記）。
- AutoEncoder特有の転置畳み込み/アップサンプリング層について、SkyNet・VTAどちらにも直接の実装例がないため、設計思想書の一般原則（型階層・明示的requantize・PE時分割）を適用する形での詳細設計は別途必要（`design_philosophy.md` §9）。
- ターゲットFPGAデバイスとBRAM/URAM容量が未確定のため、「モデル全体が本当にオンチップに収まるか」の見積りは着手前に必須の検算として設計思想書に明記した。

## 成果物一覧

| ファイル | 内容 |
|---|---|
| `claudes/skynet_explained.md` | SkyNetのアーキテクチャ・HLS実装・論文解説 |
| `claudes/tvm-vta_explained.md` | VTAのアーキテクチャ・HLS実装・論文解説 |
| `claudes/design_philosophy.md` | 2モデル統合の設計思想書（本題） |
| `claudes/report.md` | 本作業ログ |
| `SkyNet/`（リポジトリ直下） | clone済み参照リポジトリ |
| `tvm-vta/`（リポジトリ直下） | clone済み参照リポジトリ |
