# Claude Code Context - Streamlit Data Analytics Platform

このドキュメントは、AIアシスタント（Claude）およびこのプロジェクトに関わる開発者向けのコンテキスト情報です。

## プロジェクト概要

StreamlitベースのWebアプリケーションで、包括的なデータ分析機能を提供します。アンケート分析、購買ログ分析など、ビジネスデータ分析に特化した機能を持ちます。

## 技術スタック

- **フレームワーク**: Streamlit 1.31+
- **言語**: Python 3.11+
- **データ処理**: pandas, numpy
- **統計分析**: scipy, statsmodels, scikit-learn
- **可視化**: plotly, seaborn, matplotlib
- **テキスト分析**: nltk, gensim, janome（日本語対応）
- **多変量解析**: factor-analyzer, prince
- **開発環境**: Docker, Docker Compose
- **テスト**: pytest, pytest-cov
- **コード品質**: black, ruff, mypy

## アーキテクチャ

### モジュール構成

```
app/
├── analysis/           # 分析ロジック（ビジネスロジック層）
│   ├── base_analyzer.py
│   ├── descriptive/    # 記述統計
│   ├── correlation/    # 相関分析
│   ├── regression/     # 回帰分析
│   ├── hypothesis_testing/  # 仮説検定
│   ├── dimensionality/ # 多変量解析
│   ├── clustering/     # クラスタリング
│   ├── text_analysis/  # テキスト分析
│   ├── conjoint/       # コンジョイント分析
│   └── specialized/    # 専門分析
├── auth/               # 認証（セキュリティ層）
├── components/         # UIコンポーネント（プレゼンテーション層）
├── data/               # データ処理（データアクセス層）
├── pages/              # ページ定義（プレゼンテーション層）
└── utils/              # ユーティリティ
```

### 設計パターン

- **MVCパターン**: pages (View), analysis (Model), SessionManager (Controller)
- **シングルトンパターン**: config, SessionManager
- **ファクトリーパターン**: DataLoader（ファイル形式に応じた読み込み）
- **ストラテジーパターン**: DataPreprocessor（異なる前処理手法）

## 重要な設計判断

### 1. セッション状態管理

`st.session_state`を使用して、以下のデータを保持：
- 認証状態
- アップロードされたデータ（rawとprocessed）
- 前処理履歴
- 分析結果

**理由**: Streamlitの特性上、ページ遷移や再実行で状態が失われるため、session_stateで永続化。

### 2. データの不変性

- `raw_data`: 常に元のデータを保持（不変）
- `processed_data`: 前処理後のデータ（可変）

**理由**: ユーザーが前処理を元に戻せるようにするため。

### 3. 分析モジュールの独立性

各分析機能は独立したモジュールとして実装。

**理由**:
- メンテナンス性向上
- テストの容易性
- 機能の追加・削除が簡単

### 4. キャッシング戦略

`@st.cache_data`デコレータを使用：
- データ読み込み
- 重い計算処理

**理由**: パフォーマンス向上とユーザー体験の改善。

## 開発ガイドライン

### 新しい分析機能の追加

1. `app/analysis/[category]/new_analysis.py`を作成
2. 分析関数を実装（`show_[analysis_name]`）
3. `app/pages/analysis.py`に統合
4. テストを`tests/test_analysis/`に追加

### コーディング規約

- **命名規則**:
  - 関数: snake_case
  - クラス: PascalCase
  - 定数: UPPER_SNAKE_CASE
- **docstring**: すべてのpublic関数に必須
- **型ヒント**: 可能な限り使用
- **最大行長**: 100文字

### テスト

- **カバレッジ目標**:
  - 分析モジュール: 90%+
  - データ処理: 85%+
  - その他: 70%+

### Git workflow

1. feature/[feature-name] ブランチで開発
2. 実装とテストを追加
3. black, ruff, mypyでチェック
4. PRを作成してレビュー

## よくある問題と解決方法

### 問題1: セッション状態が消える

**原因**: Streamlitの再実行
**解決**: `SessionManager.init_session_state()`を必ず`main()`の最初で呼ぶ

### 問題2: ファイルアップロードが遅い

**原因**: 大きなファイル
**解決**:
- `MAX_UPLOAD_SIZE_MB`を調整
- chunked uploadの検討

### 問題3: 日本語テキスト分析がエラー

**原因**: Janomeのインストール不足
**解決**: `pip install janome`

### 問題4: 相関行列が表示されない

**原因**: 数値型列が不足
**解決**: データ型変換または列選択の見直し

## パフォーマンス最適化

### 実施済み

1. **キャッシング**: データ読み込みと分析結果
2. **遅延読み込み**: 分析モジュールの動的インポート
3. **ページネーション**: 大きなDataFrameの表示

### 今後の改善案

1. **非同期処理**: 重い分析をバックグラウンドで実行
2. **データベース統合**: 分析履歴の永続化
3. **インクリメンタル処理**: 大規模データの段階的処理

## セキュリティ考慮事項

### 実施済み

1. 環境変数での認証情報管理
2. ファイルサイズ制限
3. ファイルタイプバリデーション
4. セッションタイムアウト

### 今後の強化

1. HTTPS対応（リバースプロキシ）
2. CSRFトークン
3. レート制限
4. 監査ログ

## デプロイメント

### Docker環境（推奨）

```bash
docker-compose up -d
```

### ローカル環境

```bash
streamlit run app/main.py
```

### プロダクション環境

- Nginxをリバースプロキシとして使用
- HTTPS必須
- 環境変数はDocker Secretsで管理
- ログ収集（ELKスタックなど）

## 拡張性

### プラグインアーキテクチャへの移行

将来的には、分析モジュールをプラグインとして動的にロード可能にする。

```python
# 例
plugins = discover_plugins("app/analysis/plugins/")
for plugin in plugins:
    register_analysis(plugin)
```

### マルチユーザー対応

- データベースで user_id ごとにデータ管理
- Redis でセッション管理
- 権限管理（admin, analyst, viewer）

## トラブルシューティングチェックリスト

- [ ] `.env`ファイルは存在するか
- [ ] Dockerデーモンは起動しているか
- [ ] ポート8501は空いているか
- [ ] requirements.txtのすべてのパッケージがインストールされているか
- [ ] サンプルデータファイルは`data/`ディレクトリにあるか

## リソース

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)

## コントリビューション

新機能や改善のアイデアがあれば、issueまたはPRで提案してください。

## 変更履歴

### v1.0.0 (2024-xx-xx)
- 初回リリース
- 全分析機能の実装完了
- Docker環境のセットアップ完了
- サンプルデータ追加
