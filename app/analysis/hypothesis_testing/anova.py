"""
ANOVA (Analysis of Variance) module.
"""
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.express as px


def show_anova(df: pd.DataFrame):
    """Display ANOVA analysis interface."""
    st.subheader("📊 分散分析（ANOVA）")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not numeric_cols or not categorical_cols:
        st.warning("ANOVAには数値型列とカテゴリカル型列が必要です。")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        value_col = st.selectbox("数値変数", numeric_cols)
    with col2:
        group_col = st.selectbox("グループ変数", categorical_cols)
    with col3:
        alpha = st.number_input("有意水準（α）", value=0.05, min_value=0.01, max_value=0.10, step=0.01)

    if st.button("ANOVAを実行", type="primary"):
        # Prepare groups
        groups = df.groupby(group_col)[value_col].apply(lambda x: x.dropna().tolist())

        if len(groups) < 2:
            st.error("少なくとも2つのグループが必要です。")
            return

        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        st.success("ANOVAが完了しました！")

        col1, col2, col3 = st.columns(3)
        with col1:
            with st.container(border=True):
                st.metric("F統計量", f"{f_stat:.4f}")
        with col2:
            with st.container(border=True):
                st.metric("p値", f"{p_value:.4f}")
        with col3:
            with st.container(border=True):
                st.metric("結果", "有意" if p_value < alpha else "有意でない")

        # Group statistics
        st.markdown("### グループ別統計量")
        group_stats = df.groupby(group_col)[value_col].agg(["count", "mean", "std"])
        st.dataframe(group_stats, use_container_width=True)

        # Box plot
        st.markdown("### 箱ひげ図")
        fig = px.box(df, x=group_col, y=value_col, title=f"{value_col} by {group_col}")
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"帰無仮説: すべてのグループの平均は等しい")
        if p_value < alpha:
            st.success(f"p値 < {alpha} のため、帰無仮説を棄却します。グループ間に差があります。")
        else:
            st.warning(f"p値 >= {alpha} のため、帰無仮説を棄却できません。")

        with st.expander("📖 ANOVA指標の解釈"):
            st.markdown(
                f"""
**F統計量**: グループ間の分散（処理効果）をグループ内の分散（誤差）で割った値。値が大きいほどグループ間の差が大きいことを意味します。

$$F = \\frac{{\\text{{グループ間分散（MS}}_{{\\text{{between}}}}\\text{{）}}}}{{\\text{{グループ内分散（MS}}_{{\\text{{within}}}}\\text{{）}}}}$$

| F統計量 | 目安 |
|---------|------|
| F ≈ 1 | グループ間に差なし（帰無仮説に近い） |
| F が大きい | グループ間に差あり（有意性はp値で判断） |

**p値**: 帰無仮説（すべてのグループの母平均が等しい）のもとで現在のF値以上が観察される確率。

⚠️ **注意**: ANOVAはいずれかのグループに差があることを示しますが、どのグループ間に差があるかは **多重比較検定**（Tukey法など）で追加確認が必要です。

現在の値: F={f_stat:.4f}, p={p_value:.4f}（有意水準: {alpha}）
                """
            )
