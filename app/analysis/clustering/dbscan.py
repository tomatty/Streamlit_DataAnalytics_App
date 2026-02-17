"""
DBSCAN Clustering module.
"""
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def show_dbscan_clustering(df: pd.DataFrame):
    """Display DBSCAN clustering interface."""
    st.subheader("📊 DBSCANクラスタリング")

    st.info("DBSCANは密度ベースのクラスタリング手法で、外れ値を検出できます。")

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
        eps = st.number_input("ε (イプシロン)", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    with col2:
        min_samples = st.number_input("最小サンプル数", min_value=2, max_value=20, value=5, step=1)
    with col3:
        standardize = st.checkbox("データを標準化", value=True)

    if st.button("DBSCANを実行", type="primary"):
        try:
            data_subset = df[selected_cols].dropna()

            if standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(data_subset)
            else:
                X = data_subset.values

            # Perform DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X)

            # Add cluster labels
            result_df = data_subset.copy()
            result_df["クラスタ"] = clusters

            # Count clusters and noise points
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)

            st.success("DBSCANクラスタリングが完了しました！")

            col1, col2 = st.columns(2)
            col1.metric("クラスタ数", n_clusters)
            col2.metric("ノイズ点数", n_noise)

            # Cluster statistics
            st.markdown("### クラスタ別統計")
            cluster_stats = result_df.groupby("クラスタ")[selected_cols].agg(["mean", "count"])
            st.dataframe(cluster_stats, use_container_width=True)

            # Visualization
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
                st.plotly_chart(fig, use_container_width=True)

            # Download
            csv = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="クラスタリング結果をダウンロード",
                data=csv,
                file_name="dbscan_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
