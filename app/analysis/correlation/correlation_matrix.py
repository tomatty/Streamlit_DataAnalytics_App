"""
Correlation matrix analysis module.
"""
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np


def show_correlation_analysis(df: pd.DataFrame):
    """
    Display correlation matrix analysis.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("📈 相関行列分析")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("相関分析には少なくとも2つの数値型列が必要です。")
        return

    selected_cols = st.multiselect(
        "分析対象列を選択",
        numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))],
    )

    method = st.selectbox(
        "相関係数の種類",
        ["pearson", "spearman", "kendall"],
        format_func=lambda x: {
            "pearson": "ピアソンの積率相関係数",
            "spearman": "スピアマンの順位相関係数",
            "kendall": "ケンドールの順位相関係数",
        }[x],
    )

    if len(selected_cols) < 2:
        st.info("少なくとも2つの列を選択してください。")
        return

    if st.button("相関分析を実行", type="primary"):
        try:
            # Calculate correlation matrix
            corr_matrix = df[selected_cols].corr(method=method)

            st.success("相関分析が完了しました！")

            # Display correlation matrix
            st.markdown("### 相関行列")
            st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1), use_container_width=True)

            # Heatmap
            st.markdown("### ヒートマップ")
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="相関係数"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                zmin=-1,
                zmax=1,
            )
            fig.update_layout(title="相関係数ヒートマップ")
            st.plotly_chart(fig, use_container_width=True)

            # Find strong correlations
            st.markdown("### 強い相関関係")
            threshold = st.slider("相関係数の閾値", 0.0, 1.0, 0.7, 0.05)

            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        strong_corr.append({
                            "変数1": corr_matrix.columns[i],
                            "変数2": corr_matrix.columns[j],
                            "相関係数": corr_value,
                        })

            if strong_corr:
                strong_corr_df = pd.DataFrame(strong_corr)
                strong_corr_df = strong_corr_df.sort_values("相関係数", key=abs, ascending=False)
                st.dataframe(strong_corr_df, use_container_width=True)
            else:
                st.info(f"閾値 {threshold} 以上の相関関係は見つかりませんでした。")

            # Download option
            csv = corr_matrix.to_csv(index=True).encode("utf-8-sig")
            st.download_button(
                label="CSVダウンロード",
                data=csv,
                file_name="correlation_matrix.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
