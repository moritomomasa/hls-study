# AMD HLS AIアクセラレータ設計ガイド — 索引

このディレクトリは、「AMD(Xilinx) Vitis HLSを使い、様々な形のAIモデルをFPGAアクセラレータ化する」ための、**モデル形状に依存しない汎用的な設計知識**を体系化したものである（`instruction.md`の指示に基づく）。

本ディレクトリは2本の主文書と、それらを橋渡しする本索引で構成される。

## 1. 主文書

| ファイル | 観点 | 扱う問い |
|---|---|---|
| [`ai_model_hw_friendly.md`](./ai_model_hw_friendly.md) | **AIモデルアーキテクチャ観点** | 「どういうモデル構造（層構成・チャネル数・演算子選択・量子化方式）を選べば、後段のHLS実装が楽になるか」 |
| [`fpga_hls_architecture.md`](./fpga_hls_architecture.md) | **FPGAハードウェアアーキテクチャ観点** | 「そのモデルを、Vitis HLSでどう書けば（どのpragma・どの構造で）意図通りの面積・スループットになるか」 |

両文書は独立に読めるが、実際の設計作業では**§2「両文書を使った設計ワークフロー」**の順で往復させることを想定している。

いずれも「抽象原則 → 具体的な書き方・実例」の順で統一されており、実例はすべて本リポジトリ内の先行調査（`claudes/*.md`）および一次ソース（`SkyNet/`, `finn-hlslib/`, `tvm-vta/`）の該当行を引用している。断定できない事項はその都度明記してある。

### 前提資料（本ディレクトリの土台）

2文書は、以下の先行調査（`claudes/instruction.md`対応、SkyNet・TVM-VTA・FINNという3つの具体的モデルの詳細実装調査）を土台に、その知見を**特定3モデルの比較から、モデル形状に依存しない一般原則へ抽象化**したものである。実装の一次ソース行番号や、より詳細な単一モデルの解説が必要な場合はこちらを参照する。

- `claudes/design_philosophy.md` — SkyNet/VTA/FINN 3モデル比較設計思想書
- `claudes/skynet_explained.md` / `claudes/finn_explained.md` / `claudes/tvm-vta_explained.md` — 各モデルの解説

---

## 2. 両文書を使った設計ワークフロー

新規にAIアクセラレータを設計する際、以下の順序で両文書のチェックリストを往復させることを推奨する。

```
[Step 0] ターゲットデバイス・制約の確定
   └─ BRAM/URAM/DSP総量、目標クロック、DDR使用可否
      → fpga_hls_architecture.md §2.3（メモリ階層）, §5.1 ステップ1

[Step 1] モデル側の初期設計（hardware-friendly化）
   └─ 演算子選択、量子化方式、次元設計、スキップ接続の配置
      → ai_model_hw_friendly.md §1〜§8 の各節、§10 チェックリスト

[Step 2] オンチップ収容可否・演算強度の見積り（Co-designの最初の往復）
   └─ モデルサイズ vs BRAM予算、layerごとのcompute/memory-bound判定
      → ai_model_hw_friendly.md §1.6 ←→ fpga_hls_architecture.md §2.3〜2.4, §5.1 ステップ2
   └─ 収まらない/律速する場合は Step 1 に戻ってモデル側パラメータを見直す

[Step 3] ハードウェア側の並列度設計
   └─ dataflow構造（DAG化）、レイヤー内unroll/SIMD-PE、レイヤー横断folding
      → fpga_hls_architecture.md §4.1, §4.5, §4.7, §5.1 ステップ3

[Step 4] HLS実装・pragma付与
   → fpga_hls_architecture.md §4（全節）, §6 実装前チェックリスト

[Step 5] 合成・シミュレーションでの検証、ボトルネック解消
   └─ II未達・FIFOデッドロック・リソース超過への対処
      → fpga_hls_architecture.md §5.2, §6 合成後チェックリスト

[Step 6] 実測値のモデル設計へのフィードバック
   └─ 見積りと実測の乖離をコストモデルへ反映し、Step 1〜2 へ戻る
      → ai_model_hw_friendly.md §9.2 手順4 ←→ fpga_hls_architecture.md §5.2 末尾
```

このStep 0〜6は一直線の工程ではなく、**Step 1〜2、Step 2〜3、Step 5〜6の3箇所に明示的なフィードバックループを持つ反復プロセス**である。これがCo-design（両文書共通のキーワード）の実務的な姿であり、「モデルを先に固定してからハードウェアに押し込む」やり方との違いはここにある。

---

## 3. キーワード索引

ユーザー指定の必須キーワードについて、各文書での定義・詳細節への直接リンク。

| キーワード | `ai_model_hw_friendly.md`（モデル観点） | `fpga_hls_architecture.md`（ハードウェア観点） |
|---|---|---|
| **hardware-friendly** | §1「なぜモデル側の設計がHLS実装効率を左右するのか」で定義、全編を貫く軸 | §3「Hardware-Friendly設計の原則」で独立節として定義、Co-designとの違いを明記 |
| **Co-design** | §9「Hardware-Aware Co-design」（SkyNet Bi-Directional Co-Designの一般化） | §5「Co-design実践」（HW制約からモデルパラメータを逆算する具体手順） |
| **dataflow** | §1.3「データフローの規則性」（モデル側がストリーム処理と相性が良い構造か） | §2.2「データフローモデル」（原理）＋ §4.1「dataflow」（pragma・デッドロック回避） |
| **pipeline** | （直接は扱わない。§1.4静的形状がpipeline成立の前提として関連） | §2.1（時間的並列性）＋ §4.2「pipeline」（II、pragma、典型的障害） |
| **fifo** | §6「スキップ接続・分岐構造」（バイパスFIFO問題へのモデル側の緩和策） | §4.3「fifo」（`hls::stream`との関係、深さ決定、デッドロック） |
| **pipo（ping-pong buffer）** | （直接は扱わない） | §4.4「pipo（ping-pong buffer）」（原理、SkyNet/VTA/FINNの実例比較） |
| **unroll** | §4.1「PE数/SIMD数で割り切れるチャネル数設計」（unrollを見越した次元設計） | §4.5「unroll」（完全/部分展開、array partitionとの相互依存） |

---

## 4. 未解決・要フォローの横断的な論点

両文書に共通して現れる、**現時点で自動的には解決しない設計課題**を以下にまとめる。新規プロジェクトを始める際は、これらを「最初から解けるもの」と期待せず、実装者自身が見積もり・検証する前提で計画を立てること。

1. **スキップ接続のバイパスFIFO深さ決定**（`ai_model_hw_friendly.md` §6.4、`fpga_hls_architecture.md` §4.1・§4.3）
   SkyNet・VTA・FINNのいずれにも明確な自動解決策がない。モデル側の緩和策（分岐点を減らす、バイパス経路を短くする）は問題を軽くするが、最終的にはメインパスのレイテンシ見積り→FIFO深さの明示指定→co-simulationでの実測検証、という手作業が必要。
2. **Transformer的な演算子（Softmax/LayerNorm/Attention）のHLS実装パターン**（`ai_model_hw_friendly.md` §3.2・§11.2）
   本リポジトリの先行調査（SkyNet/VTA/FINN）に実装例がなく、一般的な設計知見からの推測にとどまる。将来この種のモデルを扱う場合は、hls4ml等の別プロジェクトの実装例を追加調査する必要がある。
3. **Transposed Convolution / Upsamplingの具体的HLS実装**（`ai_model_hw_friendly.md` §7.1）
   FINNの`upsample.hpp`の存在は確認済みだが内部実装は未読解。AutoEncoderのDecoder側を本格的に設計する際は別途詳細調査が要る。
4. **SIMD/PE折り畳みパラメータの自動バランシング**（`fpga_hls_architecture.md` §5.1）
   FINN PythonコンパイラのSection 4.4相当のアルゴリズムはfinn-hlslib自体には存在せず、完全手書きHLSで行う場合は設計者が手動でレイヤーごとの並列度を調整する必要がある。

---

## 5. 作業ログ

本ディレクトリ作成時の作業内容は [`report.md`](./report.md) を参照。
