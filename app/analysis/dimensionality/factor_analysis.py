"""
Factor Analysis module.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo


def show_factor_analysis(df: pd.DataFrame):
    """Display Factor Analysis interface."""
    st.subheader("📊 因子分析")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) < 3:
        st.warning("因子分析には少なくとも3つの数値型列が必要です。")
        return

    selected_cols = st.multiselect("分析対象列を選択", numeric_cols, default=numeric_cols[:min(10, len(numeric_cols))])

    if len(selected_cols) < 3:
        st.info("少なくとも3つの列を選択してください。")
        return

    col1, col2 = st.columns(2)
    with col1:
        n_factors = st.slider("因子数", min_value=1, max_value=min(len(selected_cols)-1, 8), value=min(2, len(selected_cols)-1))
    with col2:
        rotation = st.selectbox("回転法", ["varimax", "promax", "quartimax"],
                               format_func=lambda x: {"varimax": "バリマックス", "promax": "プロマックス", "quartimax": "クォーティマックス"}[x])

    if st.button("因子分析を実行", type="primary"):
        try:
            data_subset = df[selected_cols].dropna()

            if len(data_subset) < 10:
                st.error("有効なデータが不足しています（最低10サンプル必要）。")
                return

            # Bartlett's test
            chi_square_value, p_value = calculate_bartlett_sphericity(data_subset)
            st.markdown("### Bartlett球面性検定")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.metric("χ² 統計量", f"{chi_square_value:.2f}")
            with col2:
                with st.container(border=True):
                    st.metric("p値", f"{p_value:.4f}")
            if p_value < 0.05:
                st.success("p < 0.05: データは因子分析に適しています。")
            else:
                st.warning("p >= 0.05: データは因子分析に適していない可能性があります。")

            # KMO test
            kmo_all, kmo_model = calculate_kmo(data_subset)
            st.markdown("### KMO標本妥当性の測度")
            with st.container(border=True):
                st.metric("KMO", f"{kmo_model:.3f}")
            if kmo_model >= 0.8:
                st.success("KMO >= 0.8: 非常に良い")
            elif kmo_model >= 0.7:
                st.info("KMO >= 0.7: 良い")
            elif kmo_model >= 0.6:
                st.warning("KMO >= 0.6: 普通")
            else:
                st.error("KMO < 0.6: 因子分析に適していない")

            # Perform factor analysis
            fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
            fa.fit(data_subset)

            # Factor loadings
            st.markdown("### 因子負荷量")
            loadings = pd.DataFrame(
                fa.loadings_,
                index=selected_cols,
                columns=[f"因子{i+1}" for i in range(n_factors)]
            )
            st.dataframe(loadings.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1), use_container_width=True)

            # Communalities
            st.markdown("### 共通性")
            communalities = pd.DataFrame({
                "変数": selected_cols,
                "共通性": fa.get_communalities()
            })
            st.dataframe(communalities, use_container_width=True)

            # Variance explained
            variance = fa.get_factor_variance()
            variance_df = pd.DataFrame({
                "因子": [f"因子{i+1}" for i in range(n_factors)],
                "固有値": variance[0],
                "寄与率(%)": variance[1] * 100,
                "累積寄与率(%)": np.cumsum(variance[1]) * 100
            })
            st.markdown("### 分散説明率")
            st.dataframe(variance_df, use_container_width=True)

            st.success("因子分析が完了しました！")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
