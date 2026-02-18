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
        with col1:
            with st.container(border=True):
                st.metric("t統計量", f"{t_stat:.4f}")
        with col2:
            with st.container(border=True):
                st.metric("p値", f"{p_value:.4f}")
        with col3:
            with st.container(border=True):
                st.metric("結果", "有意" if p_value < alpha else "有意でない")

        st.info(f"帰無仮説: 母平均 = {mu0}")
        if p_value < alpha:
            st.success(f"p値 < {alpha} のため、帰無仮説を棄却します。")
        else:
            st.warning(f"p値 >= {alpha} のため、帰無仮説を棄却できません。")

        with st.expander("📖 t検定指標の解釈"):
            st.markdown(
                f"""
**t統計量**: 標本平均と帰無仮説の母平均の差を標準誤差で割った値です。絶対値が大きいほど差が統計的に有意であることを示します。

$$t = \\frac{{\\bar{{x}} - \\mu_0}}{{s / \\sqrt{{n}}}}$$

（$\\bar{{x}}$: 標本平均, $\\mu_0$: 帰無仮説の母平均, $s$: 標本標準偏差, $n$: サンプル数）

**p値（有意確率）**: 帰無仮説が真であると仮定したときに、観察された差以上に極端な結果が得られる確率です。

| p値 | 解釈 |
|-----|------|
| p < 0.001 | 非常に強い証拠で帰無仮説を棄却 |
| p < 0.01 | 強い証拠で棄却 |
| p < 0.05 | 棄却（一般的な有意水準） |
| p ≥ 0.05 | 帰無仮説を棄却できない |

現在の値: t={t_stat:.4f}, p={p_value:.4f}（有意水準: {alpha}）
                """
            )


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
        with col1:
            with st.container(border=True):
                st.metric("t統計量", f"{t_stat:.4f}")
        with col2:
            with st.container(border=True):
                st.metric("p値", f"{p_value:.4f}")
        with col3:
            with st.container(border=True):
                st.metric("結果", "有意" if p_value < alpha else "有意でない")

        st.markdown(f"**{groups[0]}の平均:** {group1.mean():.4f}")
        st.markdown(f"**{groups[1]}の平均:** {group2.mean():.4f}")

        with st.expander("📖 t検定指標の解釈"):
            st.markdown(
                f"""
**t統計量**: 2グループの平均差を標準誤差で割った値。絶対値が大きいほど差が有意です。

$$t = \\frac{{\\bar{{x}}_1 - \\bar{{x}}_2}}{{\\sqrt{{\\frac{{s_1^2}}{{n_1}} + \\frac{{s_2^2}}{{n_2}}}}}}$$

**p値**: 帰無仮説（2グループの母平均が等しい）のもとで現在の差以上が観察される確率。

| p値 | 解釈 |
|-----|------|
| p < 0.05 | 2グループ間に統計的に有意な差あり |
| p ≥ 0.05 | 有意な差を確認できない |

現在の値: t={t_stat:.4f}, p={p_value:.4f}（有意水準: {alpha}）
                """
            )


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
        with col1:
            with st.container(border=True):
                st.metric("t統計量", f"{t_stat:.4f}")
        with col2:
            with st.container(border=True):
                st.metric("p値", f"{p_value:.4f}")
        with col3:
            with st.container(border=True):
                st.metric("結果", "有意" if p_value < alpha else "有意でない")

        with st.expander("📖 対応のあるt検定指標の解釈"):
            st.markdown(
                f"""
**対応のあるt検定**: 同一対象の2時点や2条件を比較します。各ペアの差 $d_i = x_{{1i}} - x_{{2i}}$ を用います。

$$t = \\frac{{\\bar{{d}}}}{{s_d / \\sqrt{{n}}}}$$

（$\\bar{{d}}$: 差の平均, $s_d$: 差の標準偏差, $n$: ペア数）

**p値の解釈**: p < {alpha} で「2条件間に統計的に有意な差がある」と結論付けられます。

現在の値: t={t_stat:.4f}, p={p_value:.4f}（有意水準: {alpha}）
                """
            )
