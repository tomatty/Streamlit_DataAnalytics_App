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
