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

    with st.expander("📖 一般的な分析手順", expanded=False):
        st.markdown(
            """
### 分散分析（ANOVA）の基本的な流れ

**1. 目的の明確化**
- **3つ以上のグループの平均値比較**
  - 例: A、B、C 3つの広告の効果比較
  - 例: 地域別（東日本・西日本・北海道・九州）の売上差
- t検定の拡張版（複数グループを一度に検定）
- グループ間の差が有意かを判定

**2. データの準備**
- **データ形式**:
  ```
  | サンプルID | グループ | 測定値 |
  |-----------|---------|-------|
  | 1         | A       | 85    |
  | 2         | A       | 90    |
  | 3         | B       | 75    |
  | 4         | B       | 80    |
  | 5         | C       | 95    |
  | 6         | C       | 100   |
  ```
- グループ変数: カテゴリカル型（3つ以上の水準）
- 測定値: 数値型
- 各グループに最低3サンプル以上が望ましい
- **カテゴリー変数の扱い**:
  - **変換不要**: グループ変数（カテゴリー変数）と測定値（数値変数）を使用
  - グループ変数は3つ以上のカテゴリー（2つの場合はt検定を使用）
  - 例: 地域（東日本/西日本/北海道/九州）、広告タイプ（A/B/C/D）

**3. 前提条件の確認**
- **正規性**: 各グループのデータが正規分布に従う
- **等分散性**: 全グループの分散が等しい（Levene検定で確認）
- **独立性**: 観測値が独立している
- サンプルサイズのバランスが取れている方が望ましい

**4. 仮説の設定**
- **帰無仮説（H₀）**: すべてのグループの平均が等しい
- **対立仮説（H₁）**: 少なくとも1つのグループの平均が異なる
- 有意水準α（通常0.05）を設定

**5. F検定の実施**
- F統計量 = グループ間分散 / グループ内分散
- p値を計算

**6. 結果の解釈**
- **p値 < 0.05**: 帰無仮説を棄却
  - 少なくとも1つのグループが他と異なる
  - **多重比較が必要**: どのグループ間に差があるか特定
- **p値 ≥ 0.05**: 帰無仮説を棄却できない
  - グループ間に有意差なし

**7. 事後検定（多重比較）**
- ANOVA で有意な場合のみ実施
- **Tukey's HSD**: 全ペア比較（保守的）
- **Bonferroni**: 最も保守的（多重性の調整）
- **Dunnett**: 対照群との比較
- どのグループ間に差があるか特定

**8. 注意点**
- ANOVAは「どこかに差がある」としか言えない（事後検定が必要）
- 多重比較で第一種の誤りが増大
- 外れ値や等分散性の仮定違反に敏感
- 順序や大小関係は考慮しない
            """
        )

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
