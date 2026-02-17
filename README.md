# Streamlit Data Analytics Platform

包括的なデータ分析機能を提供するStreamlitアプリケーション

## 概要

このアプリケーションは、ビジネスデータ分析に必要な様々な統計手法と機械学習アルゴリズムを提供する、Webベースのデータ分析プラットフォームです。アンケート分析や購買ログ分析など、実務で頻繁に必要とされる分析機能を簡単に利用できます。

## 主な機能

### 🔐 認証システム
- 環境変数ベースのログイン認証
- セッション管理

### 📁 データ管理
- ファイルアップロード（CSV, Excel, JSON対応）
- サンプルデータの提供
- データプレビューと基本統計表示

### 🔧 データ前処理
- 欠損値処理
- 重複削除
- 外れ値処理
- カテゴリカル変数のエンコーディング
- 特徴量スケーリング

### 📊 記述統計・集計
- クロス集計 / ピボットテーブル
- グループ化集計
- 基本統計量の計算

### 📈 相関分析
- 相関行列とヒートマップ
- ペアプロット
- 散布図

### 📉 回帰分析
- 単回帰分析
- 重回帰分析
- モデル診断（R², 残差プロット、QQプロット）

### 🔬 仮説検定
- t検定（一標本、二標本、対応）
- カイ二乗検定
- ANOVA（分散分析）
- サンプルサイズ計算

### 🎯 多変量解析
- 主成分分析（PCA）
- 因子分析
- コレスポンデンス分析
- コンジョイント分析

### 🔵 クラスタリング
- K-Means（エルボー法サポート）
- 階層的クラスタリング（デンドログラム）
- DBSCAN

### 📝 テキスト分析
- 単語頻度分析
- ワードクラウド
- トピックモデリング（LDA）
- 感情分析
- 日本語対応（Janome使用）

### 🎓 専門分析
- アンケート分析（リッカート尺度、NPS）
- 購買ログ分析（RFM分析、コホート分析）

### 💾 エクスポート
- CSV / Excelでの結果エクスポート

## 必要要件

- Docker & Docker Compose
- または Python 3.11+

## インストールと起動

### Dockerを使用する場合（推奨）

1. リポジトリをクローン
```bash
git clone <repository-url>
cd 07_Streamlit-DataAnalytics-App
```

2. 環境変数の設定
```bash
cp .env.example .env
# .envファイルを編集して認証情報を設定
```

3. Dockerコンテナを起動
```bash
docker-compose up
```

4. ブラウザでアクセス
```
http://localhost:8501
```

### ローカル環境で実行する場合

1. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

2. 環境変数の設定
```bash
cp .env.example .env
# .envファイルを編集
```

3. アプリケーションの起動
```bash
streamlit run app/main.py
```

## 使い方

1. **ログイン**: .envで設定したユーザー名とパスワードでログイン
2. **データアップロード**: CSVまたはExcelファイルをアップロード、またはサンプルデータを選択
3. **データ概要**: アップロードしたデータの基本情報と統計量を確認
4. **前処理**: 必要に応じてデータのクリーニングと変換を実行
5. **分析**: 目的に応じた分析手法を選択して実行
6. **エクスポート**: 分析結果をCSVまたはExcelでダウンロード

## プロジェクト構造

```
07_Streamlit-DataAnalytics-App/
├── app/                    # アプリケーションコード
│   ├── analysis/           # 分析モジュール
│   ├── auth/               # 認証
│   ├── components/         # UIコンポーネント
│   ├── data/               # データ処理
│   ├── pages/              # ページ
│   └── utils/              # ユーティリティ
├── data/                   # サンプルデータ
├── tests/                  # テストコード
├── docs/                   # ドキュメント
├── Dockerfile              # Dockerイメージ定義
├── docker-compose.yml      # Docker Compose設定
├── requirements.txt        # Python依存関係
└── README.md               # このファイル
```

## 環境変数

`.env`ファイルで以下の設定が可能です：

```env
# 認証
APP_USERNAME=admin
APP_PASSWORD=your_password

# アプリケーション設定
APP_NAME=Data Analytics Platform
MAX_UPLOAD_SIZE_MB=200
ALLOWED_FILE_TYPES=csv,xlsx,xls,json

# 分析設定
DEFAULT_CONFIDENCE_LEVEL=0.95
DEFAULT_SIGNIFICANCE_LEVEL=0.05

# ログ設定
LOG_LEVEL=INFO
```

## 開発

### テストの実行
```bash
pytest tests/ -v
```

### コードフォーマット
```bash
black app/
ruff app/
```

### 型チェック
```bash
mypy app/
```

## トラブルシューティング

### Dockerコンテナが起動しない
- ポート8501が他のプロセスで使用されていないか確認
- `docker-compose down`で既存のコンテナを停止してから再起動

### ファイルアップロードができない
- ファイルサイズが設定値（デフォルト200MB）を超えていないか確認
- サポートされているファイル形式（CSV, Excel, JSON）か確認

### 日本語が表示されない（ワードクラウド）
- 日本語フォントがシステムにインストールされているか確認

## ライセンス

MIT License

## 作成者

Data Analytics Team

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## サポート

問題が発生した場合は、GitHubのissueで報告してください。
