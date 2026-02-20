"""
Hierarchical Clustering module.
"""
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def show_hierarchical_clustering(df: pd.DataFrame):
    """Display hierarchical clustering interface."""
    st.subheader("📊 階層的クラスタリング")

    with st.expander("📖 一般的な分析手順", expanded=False):
        st.markdown(
            """
### 階層的クラスタリングの基本的な流れ

**1. 目的の明確化**
- 階層構造の可視化: データの包含関係を理解
- クラスタ数の探索: デンドログラムから最適なクラスタ数を決定
- 類似度に基づく分類: 段階的な統合プロセスを観察
- 小規模データの詳細分析: K-Meansより解釈しやすい

**2. データの準備**
- **データ形式**:
  - 行：サンプル/観測（例: 顧客、商品、地域）
  - 列：数値型変数（例: 特徴量、指標）
  - K-Meansと同じ形式
- **サンプル数の制約**:
  - 計算量がO(n²)なので、数千件程度が限界
  - 大規模データにはK-Meansが適している
- **標準化が重要**: 変数のスケールを揃える
- 欠損値の処理が必要
- **カテゴリー変数の扱い**:
  - 距離ベースの手法のため、数値データが必須
  - **ワンホットエンコーディング（推奨）**: カテゴリーを0/1変数に変換
  - **順序エンコーディング**: 順序のあるカテゴリーに使用
  - 標準化を適用してスケールを揃える

**3. 連結法の選択**
- **ワード法（Ward）**: クラスタ内分散を最小化（最も一般的）
  - バランスの取れたクラスタを生成
  - 外れ値に敏感
- **完全連結（Complete）**: クラスタ間の最大距離
  - コンパクトなクラスタを生成
- **平均連結（Average）**: クラスタ間の平均距離
  - 中間的な性質
- **単連結（Single）**: クラスタ間の最小距離
  - チェーン状のクラスタになりやすい

**4. デンドログラムの読み方**
- **縦軸**: クラスタ間の距離（高さ）
- **横軸**: サンプル
- **分岐点の高さ**: 統合時の類似度
  - 高い位置で統合→異質なグループ
  - 低い位置で統合→類似したグループ
- **カット位置**: 水平線を引く高さでクラスタ数が決まる

**5. クラスタ数の決定**
- デンドログラムの大きな段差を探す
- 段差が大きい箇所でカット
- ビジネス的に解釈可能な数

**6. 結果の解釈**
- 各クラスタの特徴を分析
- 階層構造からサブグループを理解
- クラスタ間の関係性を把握

**7. 注意点**
- 計算コストが高い（大規模データには不向き）
- 一度統合したクラスタは分割できない
- 連結法により結果が大きく変わる
- 外れ値の影響を受けやすい
            """
        )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("クラスタリングには少なくとも2つの数値型列が必要です。")
        return

    selected_cols = st.multiselect("分析対象列を選択", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])

    if len(selected_cols) < 2:
        st.info("少なくとも2つの列を選択してください。")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        n_clusters = st.slider("クラスタ数", min_value=2, max_value=10, value=3)
    with col2:
        linkage_method = st.selectbox("連結法", ["ward", "complete", "average", "single"],
                                     format_func=lambda x: {"ward": "ワード法", "complete": "完全連結", "average": "平均連結", "single": "単連結"}[x])
    with col3:
        standardize = st.checkbox("データを標準化", value=True)

    if st.button("階層的クラスタリングを実行", type="primary"):
        try:
            data_subset = df[selected_cols].dropna()

            if len(data_subset) > 100:
                st.warning("データ数が多いため、最初の100サンプルのみを使用します。")
                data_subset = data_subset.head(100)

            if standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(data_subset)
            else:
                X = data_subset.values

            # Dendrogram
            st.markdown("### デンドログラム")
            Z = linkage(X, method=linkage_method)

            fig = ff.create_dendrogram(
                X,
                linkagefun=lambda x: linkage(x, method=linkage_method),
                labels=data_subset.index.tolist()
            )
            fig.update_layout(title="階層的クラスタリング デンドログラム", xaxis_title="サンプル", yaxis_title="距離")
            st.plotly_chart(fig, use_container_width=True)

            # Perform clustering
            hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            clusters = hc.fit_predict(X)

            result_df = data_subset.copy()
            result_df["クラスタ"] = clusters

            st.success("階層的クラスタリングが完了しました！")

            # Cluster statistics
            st.markdown("### クラスタ別統計")
            cluster_stats = result_df.groupby("クラスタ")[selected_cols].agg(["mean", "count"])
            cluster_stats.columns = [f"{col}_{agg}" for col, agg in cluster_stats.columns]
            st.dataframe(cluster_stats.reset_index(), use_container_width=True)

            # Download
            csv = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="クラスタリング結果をダウンロード",
                data=csv,
                file_name="hierarchical_clustering_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
