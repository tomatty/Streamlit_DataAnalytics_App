# デプロイメントガイド

このドキュメントは、Streamlit Data Analytics Platformを各種環境にデプロイする方法を説明します。

## 目次

1. [ローカル環境での実行](#ローカル環境での実行)
2. [Docker環境での実行](#docker環境での実行)
3. [プロダクション環境へのデプロイ](#プロダクション環境へのデプロイ)
4. [トラブルシューティング](#トラブルシューティング)

---

## ローカル環境での実行

### 前提条件

- Python 3.11以上
- pip

### 手順

1. **リポジトリのクローン**

```bash
git clone <repository-url>
cd 07_Streamlit-DataAnalytics-App
```

2. **仮想環境の作成（推奨）**

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

3. **依存パッケージのインストール**

```bash
pip install -r requirements.txt
```

4. **環境変数の設定**

```bash
cp .env.example .env
```

`.env`ファイルを編集：

```env
APP_USERNAME=admin
APP_PASSWORD=your_secure_password
```

5. **アプリケーションの起動**

```bash
streamlit run app/main.py
```

6. **ブラウザでアクセス**

```
http://localhost:8501
```

---

## Docker環境での実行

### 前提条件

- Docker
- Docker Compose

### 手順

1. **リポジトリのクローン**

```bash
git clone <repository-url>
cd 07_Streamlit-DataAnalytics-App
```

2. **環境変数の設定**

```bash
cp .env.example .env
```

`.env`ファイルを編集して認証情報を設定。

3. **Dockerイメージのビルドと起動**

```bash
docker-compose up -d
```

初回は時間がかかります（依存パッケージのインストール）。

4. **ログの確認**

```bash
docker-compose logs -f
```

5. **ブラウザでアクセス**

```
http://localhost:8501
```

6. **停止**

```bash
docker-compose down
```

### Dockerコマンドリファレンス

```bash
# コンテナの状態確認
docker-compose ps

# ログの表示
docker-compose logs -f streamlit-app

# コンテナに入る
docker-compose exec streamlit-app bash

# 再ビルド
docker-compose build --no-cache

# ボリュームも含めて削除
docker-compose down -v
```

---

## プロダクション環境へのデプロイ

### オプション1: Dockerを使用したデプロイ（推奨）

#### 1. サーバーの準備

```bash
# Ubuntu/Debianの場合
sudo apt update
sudo apt install docker.io docker-compose git
sudo systemctl start docker
sudo systemctl enable docker
```

#### 2. リポジトリのクローン

```bash
cd /opt
sudo git clone <repository-url>
cd 07_Streamlit-DataAnalytics-App
```

#### 3. 環境変数の設定

```bash
sudo cp .env.example .env
sudo nano .env
```

**重要な設定**:
- `APP_PASSWORD`: 強力なパスワードに変更
- `SESSION_TIMEOUT_MINUTES`: 必要に応じて調整

#### 4. Docker Composeの本番用設定

`docker-compose.prod.yml`を作成：

```yaml
version: '3.8'

services:
  streamlit-app:
    build: .
    container_name: data-analytics-app
    restart: always
    ports:
      - "127.0.0.1:8501:8501"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    env_file:
      - .env
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
      - STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

#### 5. Nginx リバースプロキシの設定

```bash
sudo apt install nginx
```

`/etc/nginx/sites-available/analytics`を作成：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_read_timeout 86400;
    }

    # ファイルサイズ制限
    client_max_body_size 200M;
}
```

有効化：

```bash
sudo ln -s /etc/nginx/sites-available/analytics /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 6. SSL/TLS (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

#### 7. アプリケーションの起動

```bash
sudo docker-compose -f docker-compose.prod.yml up -d
```

#### 8. 自動起動の設定

Systemdサービスを作成（`/etc/systemd/system/analytics.service`）：

```ini
[Unit]
Description=Streamlit Analytics App
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/07_Streamlit-DataAnalytics-App
ExecStart=/usr/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/bin/docker-compose -f docker-compose.prod.yml down

[Install]
WantedBy=multi-user.target
```

有効化：

```bash
sudo systemctl enable analytics
sudo systemctl start analytics
```

---

### オプション2: クラウドプラットフォーム

#### Streamlit Community Cloud

1. GitHubリポジトリにプッシュ
2. [Streamlit Community Cloud](https://streamlit.io/cloud)にログイン
3. 「New app」をクリック
4. リポジトリを選択
5. Main file path: `app/main.py`
6. Secretsで環境変数を設定
7. Deploy

**注意**: 無料プランには制限があります。

#### AWS (EC2)

1. EC2インスタンスを作成（Ubuntu推奨）
2. セキュリティグループで80, 443, 8501ポートを開放
3. 上記「Dockerを使用したデプロイ」の手順に従う
4. Elastic IPを割り当て

#### Google Cloud Platform (Cloud Run)

`Dockerfile`を本番用に最適化：

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . .

# ポート設定
ENV PORT 8080
EXPOSE 8080

# Streamlit起動
CMD streamlit run app/main.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true
```

デプロイ：

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/analytics
gcloud run deploy analytics \
    --image gcr.io/PROJECT_ID/analytics \
    --platform managed \
    --region asia-northeast1 \
    --allow-unauthenticated
```

---

## 監視とメンテナンス

### ログ監視

```bash
# Dockerログ
docker-compose logs -f

# Nginxログ
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### バックアップ

```bash
# データディレクトリのバックアップ
sudo tar -czf backup-$(date +%Y%m%d).tar.gz data/

# 定期バックアップ（cron）
0 2 * * * cd /opt/07_Streamlit-DataAnalytics-App && tar -czf /backup/data-$(date +\%Y\%m\%d).tar.gz data/
```

### アップデート

```bash
cd /opt/07_Streamlit-DataAnalytics-App
sudo git pull
sudo docker-compose -f docker-compose.prod.yml down
sudo docker-compose -f docker-compose.prod.yml build
sudo docker-compose -f docker-compose.prod.yml up -d
```

### セキュリティ

1. **ファイアウォール設定**

```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

2. **定期アップデート**

```bash
sudo apt update && sudo apt upgrade -y
```

3. **パスワードポリシー**
   - 強力なパスワードを使用
   - 定期的に変更
   - 2要素認証の検討

---

## トラブルシューティング

### 問題: アプリケーションにアクセスできない

**確認項目**:
1. Dockerコンテナが起動しているか: `docker-compose ps`
2. ポートが開いているか: `netstat -tuln | grep 8501`
3. ファイアウォール設定: `sudo ufw status`
4. Nginxが起動しているか: `sudo systemctl status nginx`

**解決方法**:
```bash
# コンテナ再起動
docker-compose restart

# Nginx再起動
sudo systemctl restart nginx
```

### 問題: メモリ不足

**確認**:
```bash
docker stats
free -h
```

**解決**:
- スワップ領域の追加
- インスタンスサイズの増加
- メモリ使用量の多い分析をバッチ処理に

### 問題: ファイルアップロードが失敗

**確認**:
- Nginxの`client_max_body_size`設定
- `.env`の`MAX_UPLOAD_SIZE_MB`設定

**解決**:
```nginx
# /etc/nginx/sites-available/analytics
client_max_body_size 500M;
```

### 問題: セッションが頻繁に切れる

**解決**:
```env
# .env
SESSION_TIMEOUT_MINUTES=120
```

---

## パフォーマンスチューニング

### Streamlit設定

`.streamlit/config.toml`を作成：

```toml
[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
base = "light"
```

### Docker最適化

```yaml
# docker-compose.prod.yml
services:
  streamlit-app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          memory: 2G
```

---

## まとめ

- **開発**: ローカル環境またはDocker
- **ステージング**: Docker + Nginx
- **本番**: Docker + Nginx + SSL + 監視

問題が発生した場合は、ログを確認し、issueで報告してください。
