"""
Pair plot analysis module.
"""
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


def show_pairplot_analysis(df: pd.DataFrame):
    """
    Display pair plot analysis using seaborn pairplot.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("📊 ペアプロット")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("ペアプロットには少なくとも2つの数値型列が必要です。")
        return

    col1, col2 = st.columns(2)

    with col1:
        selected_cols = st.multiselect(
            "分析対象列を選択（2-5列推奨）",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))],
        )

    with col2:
        hue_col = st.selectbox(
            "色分け列（オプション）",
            ["なし"] + categorical_cols,
        )

    if len(selected_cols) < 2:
        st.info("少なくとも2つの列を選択してください。")
        return

    if len(selected_cols) > 6:
        st.warning("列数が多すぎると表示に時間がかかります。6列以下を推奨します。")

    if st.button("ペアプロットを生成", type="primary"):
        try:
            with st.spinner("ペアプロットを生成中..."):
                plot_df = df[selected_cols + ([hue_col] if hue_col != "なし" else [])].dropna()

                fig, ax = plt.subplots()
                plt.close(fig)

                if hue_col != "なし":
                    pair_grid = sns.pairplot(
                        plot_df,
                        vars=selected_cols,
                        hue=hue_col,
                        diag_kind="kde",
                        plot_kws={"alpha": 0.6},
                    )
                else:
                    pair_grid = sns.pairplot(
                        plot_df,
                        vars=selected_cols,
                        diag_kind="kde",
                        plot_kws={"alpha": 0.6},
                    )

                pair_grid.figure.suptitle("ペアプロット", y=1.02)
                st.pyplot(pair_grid.figure)
                plt.close(pair_grid.figure)

                st.success("ペアプロットを生成しました！")

                # Show correlation for selected pairs
                st.markdown("### 選択列間の相関係数")
                corr_subset = df[selected_cols].corr()
                st.dataframe(
                    corr_subset.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1),
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")


def show_scatter_plot(df: pd.DataFrame):
    """
    Display customizable scatter plot.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("🔵 散布図")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("散布図には少なくとも2つの数値型列が必要です。")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        x_col = st.selectbox("X軸", numeric_cols, key="scatter_x")

    with col2:
        y_col = st.selectbox("Y軸", [c for c in numeric_cols if c != x_col], key="scatter_y")

    with col3:
        color_col = st.selectbox("色分け（オプション）", ["なし"] + all_cols, key="scatter_color")

    size_col = st.selectbox("サイズ（オプション）", ["なし"] + numeric_cols, key="scatter_size")

    if st.button("散布図を生成", type="primary", key="scatter_gen"):
        try:
            kwargs = {"x": x_col, "y": y_col}

            if color_col != "なし":
                kwargs["color"] = color_col

            if size_col != "なし":
                kwargs["size"] = size_col

            kwargs["trendline"] = st.checkbox("トレンドラインを表示", value=False)
            if kwargs["trendline"]:
                kwargs["trendline"] = "ols"
            else:
                del kwargs["trendline"]

            fig = px.scatter(df, **kwargs, title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)

            # Show correlation
            corr = df[[x_col, y_col]].corr().iloc[0, 1]
            with st.container(border=True):
                st.metric("相関係数", f"{corr:.3f}")

            st.success("散布図を生成しました！")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
