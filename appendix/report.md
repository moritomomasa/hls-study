# 作業ログ（`appendix/instruction.md` 対応）

## 実施内容の概要

`appendix/instruction.md` の指示に基づき、「AMDのHLSを扱いAIモデルのアクセラレータを作るコツ」を、AIモデルアーキテクチャ観点とFPGAハードウェアアーキテクチャ観点の両方から、抽象原則と具体的な実装技法の両軸で網羅的にまとめた。

先行して存在した `claudes/*.md`（SkyNet / TVM-VTA / FINN の3モデルの詳細比較調査、`claudes/instruction.md` 対応）は「特定3モデルの実装詳細を精読して比較する」調査だったのに対し、本タスクは**その知見を土台にしつつ、モデル形状に依存しない一般原則へ抽象化する**という、一段上のレイヤーの整理を目的とした。将来SkyNet/VTA/FINN以外の形（AutoEncoder、将来的にはTransformer系等）のAIモデルをアクセラレータ化する際の最初の参照資料として使えることを狙っている。

## 実施手順

1. `appendix/instruction.md` を読み、要求（AIモデル観点×FPGA観点の網羅的整理、抽象と具体の両軸、キーワード: hardware-friendly, Co-design, dataflow, pipeline, fifo, pipo, unroll、成果物は`appendix/*`）を把握。
2. リポジトリ構成を確認し、先行調査資料 `claudes/design_philosophy.md`・`claudes/skynet_explained.md`・`claudes/finn_explained.md`・`claudes/tvm-vta_explained.md`（前回セッションの成果物）が既に存在し、本タスクの土台として使えることを確認した。
3. 2本のサブエージェント（general-purpose）を並列起動し、それぞれ独立した観点から精読・執筆させた。
   - **エージェント1（AIモデル観点）**: 先行調査資料4本＋`SkyNet/GPU/models.py`を精読し、AMD公式ドキュメント・量子化ホワイトペーパー（arXiv:1806.08342）・hls4ml関連論文・Hardware-Aware NAS関連論文をWebSearchで補強しながら `appendix/ai_model_hw_friendly.md` を執筆（約30,600字）。
   - **エージェント2（FPGAハードウェア観点）**: 先行調査資料4本に加え、`finn-hlslib/`・`tvm-vta/hardware/xilinx/src/vta.cc`・`SkyNet/FPGA/HLS/`の実ソースを直接読み込み、`appendix/fpga_hls_architecture.md` を執筆（約38,300字）。ユーザー必須キーワード（dataflow, pipeline, fifo, pipo, unroll）はすべて独立節として扱わせた。
4. 両文書を通読し、内容の整合性（相互参照、キーワードの扱い、チェックリストの網羅性）を確認。
5. 両文書を橋渡しする索引ドキュメント `appendix/README.md` を作成。以下を含む。
   - 両文書の役割分担の説明
   - Step 0〜6の設計ワークフロー（両文書のチェックリストを実務の時系列で統合したもの、フィードバックループの明示）
   - キーワード索引表（7キーワードそれぞれについて両文書内の該当節への直接リンク）
   - 両文書に共通して現れる未解決の横断的論点のまとめ（バイパスFIFO深さ決定、Transformer系演算子、Transposed Convolution実装、SIMD/PE自動バランシング）
6. 本作業ログを作成。

## 成果物一覧

| ファイル | 内容 | 分量 |
|---|---|---|
| `appendix/ai_model_hw_friendly.md` | AIモデルアーキテクチャ観点でのhardware-friendly設計ガイド（量子化・演算子選択・次元設計・演算子融合・スキップ接続・AutoEncoder特有考慮・PE共有・Co-design・チェックリスト） | 約30,600字 |
| `appendix/fpga_hls_architecture.md` | FPGAハードウェアアーキテクチャ観点でのHLS実装技法リファレンス（並列性の分類・メモリ階層・dataflow/pipeline/fifo/pipo/unroll/array partitioning/folding/インターフェース設計/量子化型・Co-design実践・チェックリスト） | 約38,300字 |
| `appendix/README.md` | 上記2文書の索引・設計ワークフロー・キーワード索引・横断的な未解決論点まとめ | — |
| `appendix/report.md` | 本作業ログ | — |

## 主な設計判断・工夫点

- **前回調査（claudes/）との役割分担を明確化した**: `claudes/*.md`は「3つの具体的モデルの実装詳細調査」、`appendix/*.md`は「そこから抽象化した、モデル形状に依存しない一般原則集」という位置づけの違いを、両ドキュメントの冒頭に明記させた。これにより重複執筆を避けつつ、`appendix/*.md`の実例引用がすべて`claudes/*.md`・一次ソースの行番号レベルで裏付けを持つ構成にできた。
- **2エージェント並列実行時の作業分担境界を明確化した**: 「モデル側の設計判断」と「HLS実装技法」という観点で明確に線引きし、Co-designという両者にまたがるテーマについては、モデル側エージェントには「モデル探索・パラメータ決定プロセスとしてのCo-design」（§9）を、ハードウェア側エージェントには「HW制約からモデルパラメータを逆算する実務手順」（§5）を担当させることで、内容の重複を避けつつ相互補完的な記述にした。
- **キーワード（hardware-friendly, Co-design, dataflow, pipeline, fifo, pipo, unroll）はすべて独立節として明示的に立てさせた**: 特にFPGA観点の文書では、各キーワードごとに「なぜ必要か（抽象原則）→ Vitis HLSでの具体的書き方（pragma・コード例）→ 先行調査資料中の実例」という統一フォーマットを徹底させ、抽象と具体の両軸という要求を各キーワード単位でも満たす構成にした。
- **未解決課題を隠さず横断的にまとめた**: 前回調査で発見された「スキップ接続のバイパスFIFO深さ決定が3モデルいずれにも解決策がない」という課題を、今回は両文書（モデル側の緩和策／ハードウェア側のデッドロック回避策）の両面から扱った上で、`README.md`側で「両文書に共通する未解決論点」として改めて集約し、読者が「これは自動的には解けない」と正しく認識できるようにした。

## 未解決・要フォロー事項

- 両文書とも、Transformer的な演算子（Softmax/LayerNorm/Attention）・Transposed Convolutionの具体的HLS実装パターンについては、本リポジトリの先行調査（SkyNet/VTA/FINN）に直接の実装例がなく、一般的な設計知見からの記述にとどまっている（`ai_model_hw_friendly.md` §11.2に明記）。将来これらを本格的に扱う場合、hls4ml等の別プロジェクトの追加調査が必要。
- FPGA観点の文書（`fpga_hls_architecture.md`）で引用した数値（BRAM/URAM容量、DSPスライス幅等）はXilinx UltraScale+世代を念頭に置いた目安であり、実際のターゲットデバイスのデータシートでの確認が別途必要（文書内で明記済み）。
- Hardware-Aware NAS関連の学術文献（arXiv:2007.09087, arXiv:2103.10584, arXiv:2005.02563等）はWebSearchの要約に基づく言及であり、一次資料（PDF全文）までは確認していない。
- 両エージェントの分量目安（15,000〜25,000字、20,000〜30,000字）に対し、実際の成果物はやや大きめ（約30,600字・約38,300字）になった。網羅性を優先した結果であり、内容の重複や冗長性は確認上は見られないが、今後の実プロジェクトで参照する中で分割・要約が必要になる可能性はある。
