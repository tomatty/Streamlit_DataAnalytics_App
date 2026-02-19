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

    with st.expander("📖 一般的な分析手順", expanded=False):
        st.markdown(
            """
### DBSCANクラスタリングの基本的な流れ

**1. 目的の明確化**
- 異常検知: 外れ値・ノイズポイントの発見
- 任意形状のクラスタ発見: 球形でないクラスタも検出可能
- クラスタ数が不明な場合の分類
- 空間データの分析: 地理情報、センサーデータなど

**2. データの準備**
- **データ形式**: K-Meansと同じ（数値型変数）
- **特徴**:
  - クラスタ数を事前に指定する必要がない
  - 密度の低い領域をノイズとして扱える
- **標準化が必須**: 距離ベースのアルゴリズムのため
- サンプル数が少なすぎると全てノイズになる可能性

**3. パラメータの設定**
- **ε（イプシロン）**: 近傍の半径
  - データの密度に応じて調整
  - 小さすぎる→多数のノイズ、大きすぎる→全て1つのクラスタ
  - k-距離グラフで最適値を探索
- **min_samples**: コアポイントの最小近傍点数
  - 一般的には次元数+1以上
  - 大きいほどノイズ判定が厳しくなる

**4. 結果の解釈**
- **コアポイント**: 近傍にmin_samples以上の点がある
- **境界ポイント**: コアポイントの近傍にあるが、自身はコアでない
- **ノイズポイント**: どのクラスタにも属さない外れ値
- **クラスタ数**: 自動的に決定される（ラベル-1がノイズ）

**5. 活用方法**
- 異常取引の検出
- 地理的なホットスポット分析
- センサーデータの異常検知
- 不規則な形状のクラスタ発見

**6. 注意点**
- パラメータ調整が難しい（試行錯誤が必要）
- 密度が大きく異なるクラスタには不向き
- 高次元データではうまく機能しないことがある
- 計算量がO(n log n)〜O(n²)
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
            with col1:
                with st.container(border=True):
                    st.metric("クラスタ数", n_clusters)
            with col2:
                with st.container(border=True):
                    st.metric("ノイズ点数", n_noise)

            # Cluster statistics
            st.markdown("### クラスタ別統計")
            cluster_stats = result_df.groupby("クラスタ")[selected_cols].agg(["mean", "count"])
            cluster_stats.columns = [f"{col}_{agg}" for col, agg in cluster_stats.columns]
            st.dataframe(cluster_stats.reset_index(), use_container_width=True)

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
