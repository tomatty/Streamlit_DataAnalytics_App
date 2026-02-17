"""
Multiple linear regression analysis module.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import scipy.stats as stats


def show_multiple_regression(df: pd.DataFrame):
    """
    Display multiple linear regression analysis.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("📈 重回帰分析")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) < 3:
        st.warning("重回帰分析には少なくとも3つの数値型列が必要です。")
        return

    y_col = st.selectbox("目的変数（Y）", numeric_cols, key="multi_reg_y")

    x_cols = st.multiselect(
        "説明変数（X）",
        [c for c in numeric_cols if c != y_col],
        key="multi_reg_x",
    )

    if len(x_cols) < 1:
        st.info("少なくとも1つの説明変数を選択してください。")
        return

    if st.button("重回帰分析を実行", type="primary"):
        try:
            # Prepare data
            data_subset = df[[y_col] + x_cols].dropna()

            if len(data_subset) < len(x_cols) + 2:
                st.error("有効なデータが不足しています。")
                return

            X = data_subset[x_cols]
            y = data_subset[y_col]

            # Fit model
            model = LinearRegression()
            model.fit(X, y)

            # Predictions
            y_pred = model.predict(X)

            # Calculate metrics
            r2 = r2_score(y, y_pred)
            adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(x_cols) - 1)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=min(5, len(y) // 2), scoring="r2")

            # Display results
            st.success("重回帰分析が完了しました！")

            # Regression equation
            st.markdown("### 回帰式")
            equation = f"\\hat{{y}} = {model.intercept_:.4f}"
            for i, col in enumerate(x_cols):
                equation += f" + {model.coef_[i]:.4f} \\cdot x_{{{i+1}}}"
            st.latex(equation)

            # Variable mapping
            st.markdown("**変数の対応:**")
            for i, col in enumerate(x_cols):
                st.text(f"  x_{i+1} = {col}")

            # Model performance metrics
            st.markdown("### モデル評価指標")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{r2:.4f}")
            col2.metric("調整済みR²", f"{adj_r2:.4f}")
            col3.metric("RMSE", f"{rmse:.4f}")
            col4.metric("MAE", f"{mae:.4f}")

            st.markdown(f"**交差検証R² (CV=5):** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # Coefficients table
            st.markdown("### 回帰係数")
            coef_df = pd.DataFrame({
                "変数": ["切片"] + x_cols,
                "係数": [model.intercept_] + list(model.coef_),
            })
            st.dataframe(coef_df, use_container_width=True)

            # Feature importance (absolute coefficients)
            st.markdown("### 特徴量の重要度（標準化係数の絶対値）")
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model_scaled = LinearRegression()
            model_scaled.fit(X_scaled, y)

            importance_df = pd.DataFrame({
                "変数": x_cols,
                "標準化係数": model_scaled.coef_,
                "絶対値": np.abs(model_scaled.coef_),
            }).sort_values("絶対値", ascending=False)

            fig_importance = go.Figure()
            fig_importance.add_trace(
                go.Bar(
                    x=importance_df["絶対値"],
                    y=importance_df["変数"],
                    orientation="h",
                    marker_color="lightblue",
                )
            )
            fig_importance.update_layout(
                title="特徴量の重要度",
                xaxis_title="標準化係数の絶対値",
                yaxis_title="変数",
            )
            st.plotly_chart(fig_importance, use_container_width=True)

            # Predicted vs Actual
            st.markdown("### 予測値 vs 実測値")
            fig_pred = go.Figure()
            fig_pred.add_trace(
                go.Scatter(
                    x=y,
                    y=y_pred,
                    mode="markers",
                    name="データ",
                    marker=dict(color="blue", opacity=0.6),
                )
            )
            # Perfect prediction line
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            fig_pred.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="完全予測",
                    line=dict(color="red", dash="dash"),
                )
            )
            fig_pred.update_layout(
                title="予測値 vs 実測値",
                xaxis_title="実測値",
                yaxis_title="予測値",
            )
            st.plotly_chart(fig_pred, use_container_width=True)

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

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
