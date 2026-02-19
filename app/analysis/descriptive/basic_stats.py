"""
Basic statistics analysis module.
"""
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def show_basic_statistics(df: pd.DataFrame):
    """
    Display basic statistics for numeric columns.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("📊 数量データ")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        st.warning("数値型の列が見つかりません。")
        return

    # Display descriptive statistics
    st.markdown("### 記述統計")
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    # Distribution plots
    st.markdown("### 分布")

    selected_cols = st.multiselect(
        "列を選択（複数選択可）",
        numeric_cols,
        default=numeric_cols[:1] if numeric_cols else [],
        key="num_dist_select",
    )

    if not selected_cols:
        st.info("列を選択してください。")
        return

    for col in selected_cols:
        st.markdown(f"#### {col}")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Histogram
            fig_hist = px.histogram(
                df,
                x=col,
                title=f"{col} のヒストグラム",
                labels={col: col},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with chart_col2:
            # Box plot
            fig_box = px.box(
                df,
                y=col,
                title=f"{col} の箱ひげ図",
                labels={col: col},
            )
            st.plotly_chart(fig_box, use_container_width=True)


def show_categorical_statistics(df: pd.DataFrame):
    """
    Display statistics for categorical columns.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("📋 カテゴリーデータ")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        st.warning("カテゴリカル型の列が見つかりません。")
        return

    selected_cols = st.multiselect(
        "列を選択（複数選択可）",
        categorical_cols,
        default=categorical_cols[:1] if categorical_cols else [],
        key="cat_dist_select",
    )

    if not selected_cols:
        st.info("列を選択してください。")
        return

    for col in selected_cols:
        st.markdown(f"#### {col}")
        value_counts = df[col].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**度数分布表**")
            freq_df = pd.DataFrame({
                "カテゴリー": value_counts.index,
                "度数": value_counts.values,
                "割合(%)": (value_counts / len(df) * 100).round(1).values,
            })
            st.dataframe(freq_df, use_container_width=True)

        with col2:
            fig_pie = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"{col} の分布",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Bar chart
        fig_bar = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            labels={"x": col, "y": "度数"},
            title=f"{col} の度数分布",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
