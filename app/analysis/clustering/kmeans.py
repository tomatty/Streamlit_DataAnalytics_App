"""
K-Means Clustering module.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def show_kmeans_clustering(df: pd.DataFrame):
    """Display K-Means clustering interface."""
    st.subheader("📊 K-Meansクラスタリング")

    with st.expander("📖 一般的な分析手順", expanded=False):
        st.markdown(
            """
### K-Meansクラスタリングの基本的な流れ

**1. 目的の明確化**
- 顧客セグメンテーション: 購買行動や属性で顧客をグループ化
- 商品分類: 類似商品のグループ化
- 異常検出: 正常パターンから外れたデータの発見
- データ圧縮: 大量のデータポイントを代表点で要約

**2. データの準備**
- **データ形式**:
  - 行：サンプル/観測（例: 顧客、商品、時点）
  - 列：数値型変数（例: 購買額、年齢、頻度）
  - 最低2列以上の数値型変数が必要
- **データ例**:
  ```
  | 顧客ID | 購買額 | 購買回数 | 平均単価 | 経過日数 |
  |--------|-------|---------|---------|---------|
  | 1      | 50000 | 12      | 4167    | 5       |
  | 2      | 8000  | 2       | 4000    | 120     |
  | 3      | 150000| 45      | 3333    | 3       |
  ```
- **標準化が重要**: 単位の異なる変数を同じスケールに揃える
- 欠損値の処理が必要
- 外れ値の影響を受けやすいため事前確認
- **カテゴリー変数の扱い**:
  - K-Meansはユークリッド距離を使うため、数値データが必須
  - **ワンホットエンコーディング（推奨）**: カテゴリーを0/1のバイナリ変数に変換
    - 例: 地域（東京/大阪/名古屋）→ 地域_東京（0/1）、地域_大阪（0/1）、地域_名古屋（0/1）
  - **順序エンコーディング**: 順序のあるカテゴリー（低/中/高 → 1/2/3）
  - カテゴリーが多い場合は次元が増えすぎるため注意
  - 標準化を適用してスケールを揃える

**3. クラスタ数の決定**
- **エルボー法**: イナーシャ（クラスタ内平方和）の減少が緩やかになる点
- **シルエット係数**: 0.5以上が良好、0.7以上が優秀
- **ビジネス観点**: 実務的に意味のあるセグメント数
- 一般的には2-10個程度が解釈しやすい

**4. 結果の解釈**
- **クラスタ中心**: 各クラスタの代表的な特徴
- **クラスタサイズ**: 各クラスタに属するサンプル数
- **クラスタプロファイル**: 各変数の平均値や分布
- ビジネス的な命名: 「優良顧客」「休眠顧客」など

**5. 活用方法**
- マーケティング施策の最適化
- ターゲティング広告
- 在庫管理の効率化
- 推薦システムの構築

**6. 注意点**
- 初期値により結果が変わる（random_state固定で再現性確保）
- 球形のクラスタを前提（複雑な形状には不向き）
- 外れ値に敏感
- クラスタ数を事前に指定する必要がある
            """
        )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("クラスタリングには少なくとも2つの数値型列が必要です。")
        return

    selected_cols = st.multiselect(
        "分析対象列を選択",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )

    if len(selected_cols) < 2:
        st.info("少なくとも2つの列を選択してください。")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        n_clusters = st.slider("クラスタ数", min_value=2, max_value=10, value=3)
    with col2:
        max_iter = st.slider("最大イテレーション数", min_value=100, max_value=1000, value=300, step=50)
    with col3:
        standardize = st.checkbox("データを標準化", value=True)

    # Elbow method
    if st.checkbox("エルボー法でクラスタ数を決定"):
        show_elbow_method(df, selected_cols, standardize)

    if st.button("K-Meansを実行", type="primary"):
        try:
            data_subset = df[selected_cols].dropna()

            if len(data_subset) < n_clusters:
                st.error("データ数がクラスタ数より少ないです。")
                return

            # Standardize if requested
            if standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(data_subset)
            else:
                X = data_subset.values

            # Perform K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=max_iter)
            clusters = kmeans.fit_predict(X)

            # Silhouette score
            silhouette_avg = silhouette_score(X, clusters)

            st.success("K-Meansクラスタリングが完了しました！")

            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.metric("クラスタ数", n_clusters)
            with col2:
                with st.container(border=True):
                    st.metric("シルエットスコア", f"{silhouette_avg:.3f}")

            with st.expander("📖 シルエットスコアの解釈"):
                st.markdown(
                    """
シルエットスコアは、各データ点が自クラスタに対してどれだけ適切に分類されているかを示す指標です（範囲: −1〜+1）。

| スコア | 評価 |
|--------|------|
| 0.71 〜 1.00 | 非常に良いクラスタ構造 |
| 0.51 〜 0.70 | 妥当なクラスタ構造 |
| 0.26 〜 0.50 | 弱いクラスタ構造（重複の可能性あり） |
| −1.00 〜 0.25 | クラスタ構造が不明瞭 |

**計算式:** $s(i) = \\dfrac{b(i) - a(i)}{\\max(a(i),\\ b(i))}$

- $a(i)$: 同一クラスタ内の他の点との平均距離（凝集度）
- $b(i)$: 最も近い他クラスタの点との平均距離（分離度）
                    """
                )

            # Add cluster labels to dataframe
            result_df = data_subset.copy()
            result_df["クラスタ"] = clusters

            # Cluster statistics
            st.markdown("### クラスタ別統計")
            cluster_stats = result_df.groupby("クラスタ")[selected_cols].agg(["mean", "count"])
            cluster_stats.columns = [f"{col}_{agg}" for col, agg in cluster_stats.columns]
            st.dataframe(cluster_stats.reset_index(), use_container_width=True)

            # Visualization (2D)
            if len(selected_cols) >= 2:
                st.markdown("### クラスタの可視化")
                fig = px.scatter(
                    result_df,
                    x=selected_cols[0],
                    y=selected_cols[1],
                    color=result_df["クラスタ"].astype(str),
                    title=f"{selected_cols[0]} vs {selected_cols[1]}",
                    labels={"color": "クラスタ"}
                )

                # Add cluster centers
                if standardize:
                    centers = scaler.inverse_transform(kmeans.cluster_centers_)
                else:
                    centers = kmeans.cluster_centers_

                fig.add_trace(go.Scatter(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    mode="markers",
                    marker=dict(size=15, color="red", symbol="x", line=dict(width=2)),
                    name="クラスタ中心"
                ))

                st.plotly_chart(fig, use_container_width=True)

            # Download results
            csv = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="クラスタリング結果をダウンロード",
                data=csv,
                file_name="kmeans_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")


def show_elbow_method(df: pd.DataFrame, selected_cols: list, standardize: bool):
    """Show elbow method for determining optimal number of clusters."""
    st.markdown("### エルボー法")

    data_subset = df[selected_cols].dropna()

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(data_subset)
    else:
        X = data_subset.values

    # Calculate inertia for different K values
    K_range = range(2, min(11, len(data_subset)))
    inertias = []
    silhouette_scores = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode="lines+markers", name="慣性"))
    fig.update_layout(
        title="エルボー法: クラスタ数 vs 慣性",
        xaxis_title="クラスタ数",
        yaxis_title="慣性"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Silhouette scores
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(x=list(K_range), y=silhouette_scores, mode="lines+markers"))
    fig_sil.update_layout(
        title="シルエットスコア",
        xaxis_title="クラスタ数",
        yaxis_title="シルエットスコア"
    )
    st.plotly_chart(fig_sil, use_container_width=True)
