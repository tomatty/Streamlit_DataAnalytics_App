"""
T-test analysis module.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
from app.auth.session_manager import SessionManager


def show_t_test(df: pd.DataFrame):
    """Display t-test analysis interface."""
    st.subheader("📊 t検定")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    test_type = st.radio(
        "検定の種類",
        ["one_sample", "two_sample_independent", "two_sample_paired"],
        format_func=lambda x: {
            "one_sample": "一標本t検定",
            "two_sample_independent": "対応のない二標本t検定",
            "two_sample_paired": "対応のある二標本t検定",
        }[x],
        horizontal=True,
    )

    if test_type == "one_sample":
        show_one_sample_t_test(df, numeric_cols)
    elif test_type == "two_sample_independent":
        show_two_sample_t_test(df, numeric_cols)
    else:
        show_paired_t_test(df, numeric_cols)


def show_one_sample_t_test(df: pd.DataFrame, numeric_cols: list):
    """One-sample t-test."""
    # Get default significance level from settings
    default_alpha = SessionManager.get_setting("significance_level", 0.05)

    col1, col2, col3 = st.columns(3)

    with col1:
        test_col = st.selectbox("検定対象列", numeric_cols)
    with col2:
        mu0 = st.number_input("母平均（μ0）", value=0.0)
    with col3:
        alpha = st.number_input("有意水準（α）", value=float(default_alpha), min_value=0.01, max_value=0.20, step=0.01)

    if st.button("t検定を実行", type="primary"):
        data = df[test_col].dropna()
        t_stat, p_value = stats.ttest_1samp(data, mu0)

        st.success("t検定が完了しました！")
        col1, col2, col3 = st.columns(3)
        col1.metric("t統計量", f"{t_stat:.4f}")
        col2.metric("p値", f"{p_value:.4f}")
        col3.metric("結果", "有意" if p_value < alpha else "有意でない")

        st.info(f"帰無仮説: 母平均 = {mu0}")
        if p_value < alpha:
            st.success(f"p値 < {alpha} のため、帰無仮説を棄却します。")
        else:
            st.warning(f"p値 >= {alpha} のため、帰無仮説を棄却できません。")


def show_two_sample_t_test(df: pd.DataFrame, numeric_cols: list):
    """Two-sample independent t-test."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    col1, col2, col3 = st.columns(3)
    with col1:
        test_col = st.selectbox("検定対象列", numeric_cols)
    with col2:
        group_col = st.selectbox("グループ列", categorical_cols)
    with col3:
        alpha = st.number_input("有意水準（α）", value=0.05, min_value=0.01, max_value=0.10, step=0.01)

    if st.button("t検定を実行", type="primary"):
        groups = df[group_col].unique()
        if len(groups) != 2:
            st.error("グループ列は2つのカテゴリーを含む必要があります。")
            return

        group1 = df[df[group_col] == groups[0]][test_col].dropna()
        group2 = df[df[group_col] == groups[1]][test_col].dropna()

        t_stat, p_value = stats.ttest_ind(group1, group2)

        st.success("t検定が完了しました！")
        col1, col2, col3 = st.columns(3)
        col1.metric("t統計量", f"{t_stat:.4f}")
        col2.metric("p値", f"{p_value:.4f}")
        col3.metric("結果", "有意" if p_value < alpha else "有意でない")

        st.markdown(f"**{groups[0]}の平均:** {group1.mean():.4f}")
        st.markdown(f"**{groups[1]}の平均:** {group2.mean():.4f}")


def show_paired_t_test(df: pd.DataFrame, numeric_cols: list):
    """Paired t-test."""
    col1, col2, col3 = st.columns(3)
    with col1:
        col1_name = st.selectbox("列1", numeric_cols, key="paired_col1")
    with col2:
        col2_name = st.selectbox("列2", [c for c in numeric_cols if c != col1_name], key="paired_col2")
    with col3:
        alpha = st.number_input("有意水準（α）", value=0.05, min_value=0.01, max_value=0.10, step=0.01)

    if st.button("t検定を実行", type="primary"):
        data_subset = df[[col1_name, col2_name]].dropna()
        t_stat, p_value = stats.ttest_rel(data_subset[col1_name], data_subset[col2_name])

        st.success("対応のあるt検定が完了しました！")
        col1, col2, col3 = st.columns(3)
        col1.metric("t統計量", f"{t_stat:.4f}")
        col2.metric("p値", f"{p_value:.4f}")
        col3.metric("結果", "有意" if p_value < alpha else "有意でない")
