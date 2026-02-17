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
    st.subheader("📊 基本統計量")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        st.warning("数値型の列が見つかりません。")
        return

    # Display descriptive statistics
    st.markdown("### 記述統計")
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    # Distribution plots
    st.markdown("### 分布")

    selected_col = st.selectbox("列を選択", numeric_cols)

    if selected_col:
        col1, col2 = st.columns(2)

        with col1:
            # Histogram
            fig_hist = px.histogram(
                df,
                x=selected_col,
                title=f"{selected_col} のヒストグラム",
                labels={selected_col: selected_col},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Box plot
            fig_box = px.box(
                df,
                y=selected_col,
                title=f"{selected_col} の箱ひげ図",
                labels={selected_col: selected_col},
            )
            st.plotly_chart(fig_box, use_container_width=True)


def show_categorical_statistics(df: pd.DataFrame):
    """
    Display statistics for categorical columns.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("📋 カテゴリカル変数の統計")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        st.warning("カテゴリカル型の列が見つかりません。")
        return

    selected_col = st.selectbox("列を選択", categorical_cols, key="cat_select")

    if selected_col:
        # Value counts
        value_counts = df[selected_col].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 度数分布表")
            freq_df = pd.DataFrame({
                "カテゴリー": value_counts.index,
                "度数": value_counts.values,
                "割合(%)": (value_counts / len(df) * 100).values,
            })
            st.dataframe(freq_df, use_container_width=True)

        with col2:
            st.markdown("### 円グラフ")
            fig_pie = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"{selected_col} の分布",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Bar chart
        st.markdown("### 棒グラフ")
        fig_bar = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            labels={"x": selected_col, "y": "度数"},
            title=f"{selected_col} の度数分布",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
