"""
Crosstab analysis module.
"""
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def show_crosstab_analysis(df: pd.DataFrame):
    """
    Display crosstab analysis interface.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("📊 クロス集計 / ピボットテーブル")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    if not categorical_cols:
        st.warning("カテゴリカル型の列が見つかりません。")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        row_var = st.selectbox("行変数", all_cols, key="crosstab_row")

    with col2:
        col_var = st.selectbox(
            "列変数",
            [c for c in all_cols if c != row_var],
            key="crosstab_col",
        )

    with col3:
        value_var = st.selectbox(
            "値（数値型の場合）",
            ["度数"] + numeric_cols,
            key="crosstab_val",
        )

    aggfunc = None
    if value_var != "度数":
        aggfunc = st.selectbox(
            "集計関数",
            ["mean", "sum", "median", "count", "min", "max", "std"],
            format_func=lambda x: {
                "mean": "平均",
                "sum": "合計",
                "median": "中央値",
                "count": "カウント",
                "min": "最小値",
                "max": "最大値",
                "std": "標準偏差",
            }[x],
        )

    if st.button("クロス集計を実行", type="primary"):
        try:
            if value_var == "度数":
                # Simple frequency crosstab
                crosstab = pd.crosstab(df[row_var], df[col_var], margins=True)
            else:
                # Pivot table with aggregation
                crosstab = pd.pivot_table(
                    df,
                    values=value_var,
                    index=row_var,
                    columns=col_var,
                    aggfunc=aggfunc,
                    margins=True,
                )

            st.success("クロス集計が完了しました！")
            st.markdown("### 結果")
            st.dataframe(crosstab, use_container_width=True)

            # Heatmap visualization
            st.markdown("### ヒートマップ")
            fig = px.imshow(
                crosstab.iloc[:-1, :-1],  # Exclude margins
                labels=dict(x=col_var, y=row_var, color="値"),
                aspect="auto",
                color_continuous_scale="RdYlBu_r",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Download option
            csv = crosstab.to_csv(index=True).encode("utf-8-sig")
            st.download_button(
                label="CSVダウンロード",
                data=csv,
                file_name="crosstab.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
