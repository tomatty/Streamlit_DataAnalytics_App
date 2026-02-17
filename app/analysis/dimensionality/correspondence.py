"""
Correspondence Analysis module.
"""
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
try:
    import prince
except ImportError:
    prince = None


def show_correspondence_analysis(df: pd.DataFrame):
    """Display Correspondence Analysis interface."""
    st.subheader("📊 コレスポンデンス分析")

    if prince is None:
        st.error("princeライブラリがインストールされていません。`pip install prince`を実行してください。")
        return

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(categorical_cols) < 2:
        st.warning("コレスポンデンス分析には少なくとも2つのカテゴリカル列が必要です。")
        return

    col1, col2 = st.columns(2)
    with col1:
        row_var = st.selectbox("行変数", categorical_cols)
    with col2:
        col_var = st.selectbox("列変数", [c for c in categorical_cols if c != row_var])

    if st.button("コレスポンデンス分析を実行", type="primary"):
        try:
            # Create contingency table
            contingency_table = pd.crosstab(df[row_var], df[col_var])

            # Perform CA
            ca = prince.CA(n_components=2)
            ca = ca.fit(contingency_table)

            st.success("コレスポンデンス分析が完了しました！")

            # Explained inertia
            st.markdown("### 説明された慣性")
            inertia_df = pd.DataFrame({
                "次元": [f"次元{i+1}" for i in range(2)],
                "固有値": ca.eigenvalues_[:2],
                "寄与率(%)": ca.explained_inertia_[:2] * 100,
                "累積寄与率(%)": ca.explained_inertia_[:2].cumsum() * 100
            })
            st.dataframe(inertia_df, use_container_width=True)

            # Plot
            st.markdown("### 対応分析マップ")
            row_coords = ca.row_coordinates(contingency_table)
            col_coords = ca.column_coordinates(contingency_table)

            fig = go.Figure()

            # Row points
            fig.add_trace(go.Scatter(
                x=row_coords[0],
                y=row_coords[1],
                mode="markers+text",
                name=row_var,
                text=row_coords.index,
                textposition="top center",
                marker=dict(size=10, color="blue")
            ))

            # Column points
            fig.add_trace(go.Scatter(
                x=col_coords[0],
                y=col_coords[1],
                mode="markers+text",
                name=col_var,
                text=col_coords.index,
                textposition="bottom center",
                marker=dict(size=10, color="red", symbol="square")
            ))

            fig.update_layout(
                title="コレスポンデンス分析マップ",
                xaxis_title=f"次元1 ({ca.explained_inertia_[0]*100:.1f}%)",
                yaxis_title=f"次元2 ({ca.explained_inertia_[1]*100:.1f}%)",
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
