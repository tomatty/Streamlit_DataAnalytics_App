"""
Chi-square test module.
"""
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.express as px


def show_chi_square_test(df: pd.DataFrame):
    """Display chi-square test interface."""
    st.subheader("📊 カイ二乗検定")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(categorical_cols) < 2:
        st.warning("カイ二乗検定には少なくとも2つのカテゴリカル列が必要です。")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        var1 = st.selectbox("変数1", categorical_cols)
    with col2:
        var2 = st.selectbox("変数2", [c for c in categorical_cols if c != var1])
    with col3:
        alpha = st.number_input("有意水準（α）", value=0.05, min_value=0.01, max_value=0.10, step=0.01)

    if st.button("カイ二乗検定を実行", type="primary"):
        # Create contingency table
        contingency_table = pd.crosstab(df[var1], df[var2])

        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        st.success("カイ二乗検定が完了しました！")

        # Display results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            with st.container(border=True):
                st.metric("χ² 統計量", f"{chi2:.4f}")
        with col2:
            with st.container(border=True):
                st.metric("p値", f"{p_value:.4f}")
        with col3:
            with st.container(border=True):
                st.metric("自由度", f"{dof}")
        with col4:
            with st.container(border=True):
                st.metric("結果", "有意" if p_value < alpha else "有意でない")

        st.markdown("### 分割表（観測度数）")
        st.dataframe(contingency_table, use_container_width=True)

        st.markdown("### 期待度数")
        expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
        st.dataframe(expected_df, use_container_width=True)

        # Heatmap
        fig = px.imshow(contingency_table, labels=dict(color="度数"), text_auto=True)
        fig.update_layout(title="分割表のヒートマップ")
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"帰無仮説: {var1} と {var2} は独立である")
        if p_value < alpha:
            st.success(f"p値 < {alpha} のため、帰無仮説を棄却します。2変数には関連があります。")
        else:
            st.warning(f"p値 >= {alpha} のため、帰無仮説を棄却できません。")

        with st.expander("📖 カイ二乗検定指標の解釈"):
            st.markdown(
                f"""
**χ²統計量（カイ二乗統計量）**: 観測度数と期待度数の差の大きさを表す指標。値が大きいほど2変数間の独立性が低い（関連がある）ことを示します。

$$\\chi^2 = \\sum_{{i,j}} \\frac{{(O_{{ij}} - E_{{ij}})^2}}{{E_{{ij}}}}$$

（$O_{{ij}}$: 観測度数, $E_{{ij}}$: 期待度数）

**自由度（df）**: $(行数-1) \\times (列数-1)$ で計算されます。χ²統計量の有意性はこの自由度によって異なります。

現在の自由度: {dof}（= {int(dof**0.5 + 1)} 行 × {int(dof//int(dof**0.5 + 1) + 1)} 列の分割表の場合の一例）

**p値**: 帰無仮説（2変数が独立）のもとで現在のχ²値以上が観察される確率。p < {alpha} で「2変数間に統計的に有意な関連がある」と結論付けられます。

⚠️ **注意**: χ²検定はセルの期待度数が5以上であることが前提です。期待度数が小さい場合はフィッシャーの正確検定の使用を検討してください。

現在の値: χ²={chi2:.4f}, 自由度={dof}, p={p_value:.4f}（有意水準: {alpha}）
                """
            )
