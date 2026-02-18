"""
Independence test module.
Implements chi-square test of independence, Fisher's exact test,
and effect size measures (Cramér's V, Phi coefficient).
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


def show_independence_test(df: pd.DataFrame):
    """Display independence test interface."""
    st.subheader("📊 独立性の検定")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # 数値型でも一意値が少ない場合はカテゴリとして扱える
    for col in df.select_dtypes(include=["number"]).columns:
        if df[col].nunique() <= 10:
            categorical_cols.append(col)

    if len(categorical_cols) < 2:
        st.warning("独立性の検定には少なくとも2つのカテゴリカル列（または一意値が10以下の数値列）が必要です。")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        var1 = st.selectbox("変数1（行）", categorical_cols, key="ind_var1")
    with col2:
        var2 = st.selectbox("変数2（列）", [c for c in categorical_cols if c != var1], key="ind_var2")
    with col3:
        alpha = st.number_input(
            "有意水準（α）", value=0.05, min_value=0.01, max_value=0.10, step=0.01, key="ind_alpha"
        )

    if st.button("独立性の検定を実行", type="primary"):
        try:
            ct = pd.crosstab(df[var1], df[var2])
            n = ct.values.sum()
            r, c = ct.shape

            # --- Chi-square test ---
            chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)

            # --- Fisher's exact test (2×2 only) ---
            fisher_available = (r == 2 and c == 2)
            if fisher_available:
                oddsratio, p_fisher = stats.fisher_exact(ct)
            else:
                oddsratio, p_fisher = None, None

            # --- Effect sizes ---
            # Cramér's V
            cramers_v = np.sqrt(chi2 / (n * (min(r, c) - 1))) if min(r, c) > 1 else np.nan
            # Phi coefficient (2×2 only)
            phi = np.sqrt(chi2 / n) if (r == 2 and c == 2) else None

            # --- Minimum expected count check ---
            min_expected = expected.min()
            small_expected_cells = (expected < 5).sum()
            total_cells = expected.size

            st.success("独立性の検定が完了しました！")

            # ---- Results summary ----
            st.markdown("### 検定結果")

            # Chi-square results
            st.markdown("#### カイ二乗検定")
            cols = st.columns(4)
            with cols[0]:
                with st.container(border=True):
                    st.metric("χ² 統計量", f"{chi2:.4f}")
            with cols[1]:
                with st.container(border=True):
                    st.metric("p値", f"{p_chi2:.4f}")
            with cols[2]:
                with st.container(border=True):
                    st.metric("自由度", f"{dof}")
            with cols[3]:
                with st.container(border=True):
                    st.metric("結果", "有意" if p_chi2 < alpha else "有意でない")

            st.info(f"帰無仮説: **{var1}** と **{var2}** は独立である")
            if p_chi2 < alpha:
                st.success(f"p値 ({p_chi2:.4f}) < {alpha} ：帰無仮説を棄却します。2変数間に統計的に有意な関連があります。")
            else:
                st.warning(f"p値 ({p_chi2:.4f}) ≥ {alpha} ：帰無仮説を棄却できません。")

            if small_expected_cells > 0:
                st.warning(
                    f"⚠️ 期待度数が5未満のセルが {small_expected_cells}/{total_cells} 個あります"
                    f"（最小期待度数: {min_expected:.2f}）。"
                    "フィッシャーの正確検定の使用を検討してください。"
                )

            # Fisher's exact test (2×2)
            if fisher_available:
                st.markdown("#### フィッシャーの正確検定（2×2分割表）")
                cols2 = st.columns(3)
                with cols2[0]:
                    with st.container(border=True):
                        st.metric("オッズ比", f"{oddsratio:.4f}")
                with cols2[1]:
                    with st.container(border=True):
                        st.metric("p値（両側）", f"{p_fisher:.4f}")
                with cols2[2]:
                    with st.container(border=True):
                        st.metric("結果", "有意" if p_fisher < alpha else "有意でない")
            else:
                st.info("フィッシャーの正確検定は 2×2 分割表のみ対応しています（現在: " + f"{r}×{c}）。")

            # Effect sizes
            st.markdown("#### 効果量")
            effect_cols = st.columns(2 if phi is not None else 1)
            with effect_cols[0]:
                with st.container(border=True):
                    st.metric("Cramér's V", f"{cramers_v:.4f}")
            if phi is not None:
                with effect_cols[1]:
                    with st.container(border=True):
                        st.metric("Phi係数（φ）", f"{phi:.4f}")

            with st.expander("📖 検定指標と効果量の解釈"):
                st.markdown(
                    f"""
**カイ二乗検定**は、2つのカテゴリカル変数の間に統計的な関連があるかどうかを検定します。

$$\\chi^2 = \\sum_{{i,j}} \\frac{{(O_{{ij}} - E_{{ij}})^2}}{{E_{{ij}}}}, \\quad
E_{{ij}} = \\frac{{\\text{{行合計}}_i \\times \\text{{列合計}}_j}}{{n}}$$

**フィッシャーの正確検定**は、期待度数が小さい（5未満）セルを含む 2×2 分割表に適しています。

**効果量（関連の強さ）:**

| Cramér's V | 効果量の目安（最小次元数に依存） |
|------------|-------------------------------|
| 0.10 未満 | 無視できる関連 |
| 0.10 〜 0.30 | 小さい効果量 |
| 0.30 〜 0.50 | 中程度の効果量 |
| 0.50 以上 | 大きい効果量 |

$$V = \\sqrt{{\\frac{{\\chi^2}}{{n \\cdot (\\min(r,c)-1)}}}}$$

**Phi係数（φ）**: 2×2 分割表専用の効果量。−1〜+1 の範囲をとり、相関係数と同様に解釈できます。

$$\\phi = \\sqrt{{\\frac{{\\chi^2}}{{n}}}}$$

現在の値: χ²={chi2:.4f}, Cramér's V={cramers_v:.4f}{f", φ={phi:.4f}" if phi is not None else ""}
                    """
                )

            st.markdown("---")

            # ---- Contingency table ----
            st.markdown("### 分割表")
            tab_obs, tab_exp, tab_res = st.tabs(["観測度数", "期待度数", "調整済み残差"])

            with tab_obs:
                ct_with_total = ct.copy()
                ct_with_total["合計"] = ct_with_total.sum(axis=1)
                ct_with_total.loc["合計"] = ct_with_total.sum()
                st.dataframe(ct_with_total, use_container_width=True)

            with tab_exp:
                expected_df = pd.DataFrame(
                    expected.round(2),
                    index=ct.index,
                    columns=ct.columns,
                )
                st.dataframe(
                    expected_df.style.background_gradient(cmap="YlOrRd"),
                    use_container_width=True,
                )
                st.caption("セルの値が 5 未満のものに注意してください。")

            with tab_res:
                # Adjusted standardized residuals
                row_totals = ct.sum(axis=1).values
                col_totals = ct.sum(axis=0).values
                adj_residuals = np.zeros_like(ct.values, dtype=float)
                for i in range(r):
                    for j in range(c):
                        denom = np.sqrt(
                            expected[i, j]
                            * (1 - row_totals[i] / n)
                            * (1 - col_totals[j] / n)
                        )
                        adj_residuals[i, j] = (ct.values[i, j] - expected[i, j]) / denom if denom > 0 else 0

                adj_res_df = pd.DataFrame(
                    adj_residuals.round(3),
                    index=ct.index,
                    columns=ct.columns,
                )
                st.dataframe(
                    adj_res_df.style.background_gradient(cmap="RdBu_r", vmin=-3, vmax=3),
                    use_container_width=True,
                )
                st.caption("|調整済み残差| > 1.96 のセルは有意水準5%で統計的に有意な偏りがあります。")

            # ---- Visualizations ----
            st.markdown("### 可視化")
            tab_heat, tab_bar, tab_mosaic = st.tabs(["ヒートマップ", "積み上げ棒グラフ", "バブルチャート"])

            with tab_heat:
                fig_heat = px.imshow(
                    ct,
                    labels=dict(color="観測度数"),
                    text_auto=True,
                    color_continuous_scale="Blues",
                    aspect="auto",
                )
                fig_heat.update_layout(title=f"{var1} × {var2} 分割表ヒートマップ")
                st.plotly_chart(fig_heat, use_container_width=True)

            with tab_bar:
                ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
                ct_long = ct_pct.reset_index().melt(id_vars=var1, var_name=var2, value_name="割合(%)")
                fig_bar = px.bar(
                    ct_long,
                    x=var1,
                    y="割合(%)",
                    color=str(var2),
                    barmode="stack",
                    title=f"{var1}別 {var2} の構成割合",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with tab_mosaic:
                # Bubble chart as mosaic-like visualization
                bubble_data = []
                for i, row_label in enumerate(ct.index):
                    for j, col_label in enumerate(ct.columns):
                        obs = ct.loc[row_label, col_label]
                        exp = expected[i, j]
                        bubble_data.append({
                            var1: str(row_label),
                            var2: str(col_label),
                            "観測度数": obs,
                            "残差": obs - exp,
                        })
                bubble_df = pd.DataFrame(bubble_data)
                fig_bubble = px.scatter(
                    bubble_df,
                    x=var2,
                    y=var1,
                    size="観測度数",
                    color="残差",
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                    title="バブルチャート（サイズ=観測度数、色=残差）",
                    size_max=60,
                )
                st.plotly_chart(fig_bubble, use_container_width=True)

            # Download
            csv = ct.to_csv(index=True).encode("utf-8-sig")
            st.download_button(
                label="分割表をCSVダウンロード",
                data=csv,
                file_name="independence_test_contingency.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
