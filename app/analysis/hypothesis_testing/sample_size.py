"""
Sample size calculation module.
"""
import streamlit as st
import numpy as np
from scipy import stats


def show_sample_size_calculation():
    """Display sample size calculation interface."""
    st.subheader("📊 サンプルサイズ計算")

    calc_type = st.radio(
        "計算の種類",
        ["mean_comparison", "proportion_comparison"],
        format_func=lambda x: {
            "mean_comparison": "平均の比較",
            "proportion_comparison": "比率の比較",
        }[x],
        horizontal=True,
    )

    if calc_type == "mean_comparison":
        show_mean_comparison_sample_size()
    else:
        show_proportion_comparison_sample_size()


def show_mean_comparison_sample_size():
    """Sample size for mean comparison."""
    st.markdown("### 平均値の比較に必要なサンプルサイズ")

    col1, col2 = st.columns(2)

    with col1:
        alpha = st.number_input("有意水準（α）", value=0.05, min_value=0.01, max_value=0.10, step=0.01)
        power = st.number_input("検出力（1-β）", value=0.80, min_value=0.70, max_value=0.99, step=0.01)

    with col2:
        effect_size = st.number_input("効果量（Cohen's d）", value=0.5, min_value=0.1, max_value=2.0, step=0.1)
        test_type = st.selectbox("検定の種類", ["two_tailed", "one_tailed"],
                                format_func=lambda x: {"two_tailed": "両側検定", "one_tailed": "片側検定"}[x])

    if st.button("サンプルサイズを計算", type="primary"):
        # Approximate sample size calculation using formula
        z_alpha = stats.norm.ppf(1 - alpha / (2 if test_type == "two_tailed" else 1))
        z_beta = stats.norm.ppf(power)

        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        n = int(np.ceil(n))

        st.success(f"必要なサンプルサイズ: **各グループ {n} 名** （合計 {n*2} 名）")

        st.info(f"""
        **パラメータ:**
        - 有意水準（α）: {alpha}
        - 検出力（1-β）: {power}
        - 効果量（Cohen's d）: {effect_size}
        - 検定の種類: {"両側検定" if test_type == "two_tailed" else "片側検定"}
        """)

        # Effect size interpretation
        if effect_size < 0.2:
            effect_interpretation = "非常に小さい"
        elif effect_size < 0.5:
            effect_interpretation = "小さい"
        elif effect_size < 0.8:
            effect_interpretation = "中程度"
        else:
            effect_interpretation = "大きい"

        st.markdown(f"**効果量の解釈:** {effect_interpretation}")


def show_proportion_comparison_sample_size():
    """Sample size for proportion comparison."""
    st.markdown("### 比率の比較に必要なサンプルサイズ")

    col1, col2 = st.columns(2)

    with col1:
        alpha = st.number_input("有意水準（α）", value=0.05, min_value=0.01, max_value=0.10, step=0.01)
        power = st.number_input("検出力（1-β）", value=0.80, min_value=0.70, max_value=0.99, step=0.01)

    with col2:
        p1 = st.number_input("グループ1の比率（p1）", value=0.50, min_value=0.01, max_value=0.99, step=0.01)
        p2 = st.number_input("グループ2の比率（p2）", value=0.60, min_value=0.01, max_value=0.99, step=0.01)

    if st.button("サンプルサイズを計算", type="primary"):
        # Sample size calculation for proportions
        p_avg = (p1 + p2) / 2
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n = 2 * p_avg * (1 - p_avg) * ((z_alpha + z_beta) / (p1 - p2)) ** 2
        n = int(np.ceil(n))

        st.success(f"必要なサンプルサイズ: **各グループ {n} 名** （合計 {n*2} 名）")

        st.info(f"""
        **パラメータ:**
        - 有意水準（α）: {alpha}
        - 検出力（1-β）: {power}
        - グループ1の比率（p1）: {p1}
        - グループ2の比率（p2）: {p2}
        - 効果量（p1 - p2）: {abs(p1 - p2):.3f}
        """)
