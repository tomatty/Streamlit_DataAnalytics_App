"""
Simple linear regression analysis module.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats as stats


def show_simple_regression(df: pd.DataFrame):
    """
    Display simple linear regression analysis.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("📈 単回帰分析")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("回帰分析には少なくとも2つの数値型列が必要です。")
        return

    col1, col2 = st.columns(2)

    with col1:
        x_col = st.selectbox("説明変数（X）", numeric_cols, key="simple_reg_x")

    with col2:
        y_col = st.selectbox(
            "目的変数（Y）",
            [c for c in numeric_cols if c != x_col],
            key="simple_reg_y",
        )

    if st.button("回帰分析を実行", type="primary"):
        try:
            # Prepare data
            X = df[[x_col]].dropna()
            y = df.loc[X.index, y_col]

            # Remove any remaining NaN
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]

            if len(X) < 3:
                st.error("有効なデータが不足しています。")
                return

            # Fit model
            model = LinearRegression()
            model.fit(X, y)

            # Predictions
            y_pred = model.predict(X)

            # Calculate metrics
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)

            # Display results
            st.success("回帰分析が完了しました！")

            # Regression equation
            st.markdown("### 回帰式")
            st.latex(
                f"\\hat{{y}} = {model.intercept_:.4f} + {model.coef_[0]:.4f} \\cdot x"
            )

            # Model performance metrics
            st.markdown("### モデル評価指標")
            col1, col2, col3 = st.columns(3)
            col1.metric("決定係数 (R²)", f"{r2:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            col3.metric("MAE", f"{mae:.4f}")

            # Scatter plot with regression line
            st.markdown("### 回帰直線")
            fig = go.Figure()

            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=X[x_col],
                    y=y,
                    mode="markers",
                    name="データ",
                    marker=dict(color="blue", opacity=0.6),
                )
            )

            # Regression line
            fig.add_trace(
                go.Scatter(
                    x=X[x_col],
                    y=y_pred,
                    mode="lines",
                    name="回帰直線",
                    line=dict(color="red", width=2),
                )
            )

            fig.update_layout(
                title=f"{y_col} vs {x_col}",
                xaxis_title=x_col,
                yaxis_title=y_col,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Residual plot
            st.markdown("### 残差プロット")
            residuals = y - y_pred

            fig_res = go.Figure()
            fig_res.add_trace(
                go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode="markers",
                    marker=dict(color="blue", opacity=0.6),
                )
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(
                title="残差プロット",
                xaxis_title="予測値",
                yaxis_title="残差",
            )
            st.plotly_chart(fig_res, use_container_width=True)

            # Q-Q plot for residuals
            st.markdown("### Q-Qプロット（正規性の確認）")
            fig_qq = go.Figure()

            (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
            fig_qq.add_trace(
                go.Scatter(x=osm, y=osr, mode="markers", name="残差")
            )
            fig_qq.add_trace(
                go.Scatter(
                    x=osm,
                    y=slope * osm + intercept,
                    mode="lines",
                    name="理論分布",
                    line=dict(color="red"),
                )
            )
            fig_qq.update_layout(
                title="Q-Qプロット",
                xaxis_title="理論分位数",
                yaxis_title="サンプル分位数",
            )
            st.plotly_chart(fig_qq, use_container_width=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
