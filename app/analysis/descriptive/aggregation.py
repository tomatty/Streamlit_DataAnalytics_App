"""
Aggregation analysis module.
"""
import pandas as pd
import streamlit as st
import plotly.express as px


def show_aggregation_analysis(df: pd.DataFrame):
    """
    Display aggregation analysis interface.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("📊 グループ集計")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    if not numeric_cols:
        st.warning("数値型の列が見つかりません。")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        group_by_cols = st.multiselect("グループ化する列", all_cols)

    with col2:
        value_cols = st.multiselect("集計対象列", numeric_cols)

    with col3:
        agg_functions = st.multiselect(
            "集計関数",
            ["sum", "mean", "median", "count", "min", "max", "std", "var"],
            default=["mean"],
            format_func=lambda x: {
                "sum": "合計",
                "mean": "平均",
                "median": "中央値",
                "count": "カウント",
                "min": "最小値",
                "max": "最大値",
                "std": "標準偏差",
                "var": "分散",
            }[x],
        )

    if group_by_cols and value_cols and st.button("集計を実行", type="primary"):
        try:
            # Perform aggregation
            if len(agg_functions) == 1:
                result = df.groupby(group_by_cols)[value_cols].agg(agg_functions[0])
            else:
                result = df.groupby(group_by_cols)[value_cols].agg(agg_functions)

            result = result.reset_index()

            st.success("集計が完了しました！")
            st.markdown("### 集計結果")
            st.dataframe(result, use_container_width=True)

            # Visualization
            if len(group_by_cols) == 1 and len(value_cols) == 1:
                st.markdown("### 可視化")

                chart_type = st.radio(
                    "グラフタイプ",
                    ["bar", "line", "scatter"],
                    format_func=lambda x: {
                        "bar": "棒グラフ",
                        "line": "折れ線グラフ",
                        "scatter": "散布図",
                    }[x],
                    horizontal=True,
                )

                if len(agg_functions) == 1:
                    y_col = value_cols[0]
                else:
                    y_col = st.selectbox("Y軸の列を選択", result.columns[len(group_by_cols):].tolist())

                if chart_type == "bar":
                    fig = px.bar(result, x=group_by_cols[0], y=y_col)
                elif chart_type == "line":
                    fig = px.line(result, x=group_by_cols[0], y=y_col, markers=True)
                else:
                    fig = px.scatter(result, x=group_by_cols[0], y=y_col)

                st.plotly_chart(fig, use_container_width=True)

            # Download option
            csv = result.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="CSVダウンロード",
                data=csv,
                file_name="aggregation.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
