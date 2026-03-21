"""
Sample size calculator module.
Provides sample size calculations for statistical estimation and testing.
"""
import streamlit as st
import numpy as np
from scipy import stats


def show_sample_size_calculator():
    """Display sample size calculator interface."""
    st.header("🔢 サンプルサイズ計算")

    st.markdown("""
    統計的推定や検定を行う際に必要なサンプルサイズを計算します。
    データのアップロードは不要です。
    """)

    st.markdown("---")

    calc_type = st.radio(
        "計算の種類を選択してください",
        ["mean_estimation", "proportion_estimation", "independence_test"],
        format_func=lambda x: {
            "mean_estimation": "母平均の推定",
            "proportion_estimation": "母比率の推定",
            "independence_test": "独立性の検定",
        }[x],
        horizontal=True,
    )

    st.markdown("---")

    if calc_type == "mean_estimation":
        show_mean_estimation()
    elif calc_type == "proportion_estimation":
        show_proportion_estimation()
    else:
        show_independence_test()


def show_mean_estimation():
    """Sample size calculation for population mean estimation."""
    st.subheader("📊 母平均の推定に必要なサンプルサイズ")

    st.markdown("""
    母集団の平均値を一定の精度で推定するために必要なサンプルサイズを計算します。
    """)

    col1, col2 = st.columns(2)

    with col1:
        confidence_level = st.slider(
            "信頼水準（%）",
            min_value=90,
            max_value=99,
            value=95,
            step=1,
            help="推定の信頼性を表します。一般的には95%が使用されます。"
        )

        margin_of_error = st.number_input(
            "許容誤差（マージン）",
            min_value=0.01,
            max_value=10.0,
            value=0.5,
            step=0.1,
            help="推定値の許容される誤差の範囲です。"
        )

    with col2:
        std_dev = st.number_input(
            "母標準偏差（推定値）",
            min_value=0.1,
            max_value=100.0,
            value=1.0,
            step=0.1,
            help="母集団の標準偏差の推定値です。事前調査やパイロット調査から得られます。"
        )

        finite_population = st.checkbox(
            "有限母集団補正を適用",
            value=False,
            help="母集団サイズが有限で既知の場合にチェックしてください。"
        )

        if finite_population:
            population_size = st.number_input(
                "母集団サイズ（N）",
                min_value=10,
                value=10000,
                step=100,
                help="母集団全体のサイズです。"
            )

    if st.button("サンプルサイズを計算", type="primary", key="calc_mean"):
        # Convert confidence level to alpha
        alpha = 1 - (confidence_level / 100)
        z_score = stats.norm.ppf(1 - alpha / 2)

        # Basic sample size calculation
        n = ((z_score * std_dev) / margin_of_error) ** 2
        n = int(np.ceil(n))

        # Apply finite population correction if needed
        if finite_population:
            n_corrected = n / (1 + ((n - 1) / population_size))
            n_corrected = int(np.ceil(n_corrected))

            st.success(f"必要なサンプルサイズ: **{n_corrected} 件**")
            st.info(f"有限母集団補正前: {n} 件")
        else:
            st.success(f"必要なサンプルサイズ: **{n} 件**")

        # Display parameters
        st.markdown("### パラメータ")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("信頼水準", f"{confidence_level}%")
            st.metric("許容誤差", f"{margin_of_error}")

        with col2:
            st.metric("母標準偏差", f"{std_dev}")
            st.metric("Z値", f"{z_score:.3f}")

        # Interpretation
        st.markdown("### 解釈")
        st.markdown(f"""
        - **{confidence_level}%の信頼水準**で、母平均の推定値が真の値から**±{margin_of_error}**以内に収まるためには、
          最低**{n_corrected if finite_population else n}件**のサンプルが必要です。
        - 許容誤差を小さくするほど、より多くのサンプルが必要になります。
        - 標準偏差が大きいほど、より多くのサンプルが必要になります。
        """)


def show_proportion_estimation():
    """Sample size calculation for population proportion estimation."""
    st.subheader("📊 母比率の推定に必要なサンプルサイズ")

    st.markdown("""
    母集団の比率を一定の精度で推定するために必要なサンプルサイズを計算します。
    """)

    col1, col2 = st.columns(2)

    with col1:
        confidence_level = st.slider(
            "信頼水準（%）",
            min_value=90,
            max_value=99,
            value=95,
            step=1,
            help="推定の信頼性を表します。一般的には95%が使用されます。"
        )

        margin_of_error = st.number_input(
            "許容誤差（マージン）",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="推定値の許容される誤差の範囲です。（例: 0.05 = 5%）"
        )

    with col2:
        expected_proportion = st.number_input(
            "期待される母比率（p）",
            min_value=0.01,
            max_value=0.99,
            value=0.5,
            step=0.01,
            format="%.2f",
            help="母比率の事前推定値です。不明な場合は0.5（最も保守的）を使用します。"
        )

        finite_population = st.checkbox(
            "有限母集団補正を適用",
            value=False,
            help="母集団サイズが有限で既知の場合にチェックしてください。",
            key="finite_pop_proportion"
        )

        if finite_population:
            population_size = st.number_input(
                "母集団サイズ（N）",
                min_value=10,
                value=10000,
                step=100,
                help="母集団全体のサイズです。",
                key="pop_size_proportion"
            )

    if st.button("サンプルサイズを計算", type="primary", key="calc_proportion"):
        # Convert confidence level to alpha
        alpha = 1 - (confidence_level / 100)
        z_score = stats.norm.ppf(1 - alpha / 2)

        # Basic sample size calculation
        p = expected_proportion
        n = (z_score ** 2 * p * (1 - p)) / (margin_of_error ** 2)
        n = int(np.ceil(n))

        # Apply finite population correction if needed
        if finite_population:
            n_corrected = n / (1 + ((n - 1) / population_size))
            n_corrected = int(np.ceil(n_corrected))

            st.success(f"必要なサンプルサイズ: **{n_corrected} 件**")
            st.info(f"有限母集団補正前: {n} 件")
        else:
            st.success(f"必要なサンプルサイズ: **{n} 件**")

        # Display parameters
        st.markdown("### パラメータ")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("信頼水準", f"{confidence_level}%")
            st.metric("許容誤差", f"{margin_of_error:.2%}")

        with col2:
            st.metric("期待される母比率", f"{expected_proportion:.2%}")
            st.metric("Z値", f"{z_score:.3f}")

        # Interpretation
        st.markdown("### 解釈")
        st.markdown(f"""
        - **{confidence_level}%の信頼水準**で、母比率の推定値が真の値から**±{margin_of_error:.2%}**以内に収まるためには、
          最低**{n_corrected if finite_population else n}件**のサンプルが必要です。
        - 期待される母比率が0.5に近いほど、より多くのサンプルが必要になります（最も保守的）。
        - 許容誤差を小さくするほど、より多くのサンプルが必要になります。
        """)

        # Additional note
        if expected_proportion == 0.5:
            st.info("💡 期待される母比率を0.5に設定すると、最も保守的（安全側）な推定になります。")


def show_independence_test():
    """Sample size calculation for chi-square test of independence."""
    st.subheader("📊 独立性の検定に必要なサンプルサイズ")

    st.markdown("""
    2つのカテゴリ変数間の独立性を検定するために必要なサンプルサイズを計算します。
    （カイ二乗検定）
    """)

    col1, col2 = st.columns(2)

    with col1:
        alpha = st.number_input(
            "有意水準（α）",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="第一種の過誤の確率です。一般的には0.05が使用されます。"
        )

        power = st.number_input(
            "検出力（1-β）",
            min_value=0.70,
            max_value=0.99,
            value=0.80,
            step=0.01,
            format="%.2f",
            help="効果を正しく検出できる確率です。一般的には0.80が使用されます。"
        )

    with col2:
        effect_size = st.number_input(
            "効果量（w）",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.05,
            format="%.2f",
            help="期待される効果の大きさです。Cohen's w: 小=0.1, 中=0.3, 大=0.5"
        )

        df = st.number_input(
            "自由度（df）",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help="(行数-1) × (列数-1) で計算されます。2×2分割表の場合は1です。"
        )

    # Effect size interpretation
    st.markdown("### 効果量の目安（Cohen's w）")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**小**: 0.1")
    with col2:
        st.info("**中**: 0.3")
    with col3:
        st.info("**大**: 0.5")

    if st.button("サンプルサイズを計算", type="primary", key="calc_independence"):
        # Calculate critical chi-square value
        critical_chi2 = stats.chi2.ppf(1 - alpha, df)

        # Calculate non-centrality parameter
        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)

        # Approximate sample size using effect size
        # n = (z_alpha + z_beta)^2 / w^2 + df + 1
        n = ((z_alpha + z_beta) ** 2) / (effect_size ** 2) + df + 1
        n = int(np.ceil(n))

        st.success(f"必要なサンプルサイズ: **{n} 件**")

        # Display parameters
        st.markdown("### パラメータ")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("有意水準（α）", f"{alpha:.2f}")
            st.metric("検出力（1-β）", f"{power:.2f}")

        with col2:
            st.metric("効果量（w）", f"{effect_size:.2f}")
            st.metric("自由度（df）", f"{df}")

        # Effect size interpretation
        if effect_size < 0.2:
            effect_interpretation = "小さい（微小な関連性）"
        elif effect_size < 0.4:
            effect_interpretation = "中程度（実務的に意味のある関連性）"
        else:
            effect_interpretation = "大きい（強い関連性）"

        st.markdown("### 解釈")
        st.markdown(f"""
        - **効果量の解釈**: {effect_interpretation}
        - **{alpha:.0%}の有意水準**と**{power:.0%}の検出力**で、効果量{effect_size}の関連性を検出するためには、
          最低**{n}件**のサンプルが必要です。
        - 効果量が小さいほど、より多くのサンプルが必要になります。
        - 検出力を高くするほど、より多くのサンプルが必要になります。
        """)

        # Additional information
        st.markdown("### 自由度の計算方法")
        st.markdown("""
        クロス集計表の自由度は以下のように計算されます：
        - **自由度 = (行数 - 1) × (列数 - 1)**

        例：
        - 2×2分割表: (2-1) × (2-1) = 1
        - 2×3分割表: (2-1) × (3-1) = 2
        - 3×3分割表: (3-1) × (3-1) = 4
        """)
