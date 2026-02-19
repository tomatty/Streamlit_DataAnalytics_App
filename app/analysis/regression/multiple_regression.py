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
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def show_multiple_regression(df: pd.DataFrame):
    """
    Display multiple linear regression analysis.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("📈 重回帰分析")

    with st.expander("📖 一般的な分析手順", expanded=False):
        st.markdown(
            """
### 重回帰分析の基本的な流れ

**1. 目的の明確化**
- 複数要因の同時分析: 各要因の純粋な影響を分離
- 予測精度の向上: 複数の情報を活用
- 重要度の比較: どの変数が最も影響するか
- 交絡の調整: 他の要因をコントロール

**2. データの準備**
- **データ形式**:
  - 行：サンプル/観測
  - 列：説明変数X（複数）、目的変数Y（1つ）
- **データ例**:
  ```
  | 広告費 | 気温 | 曜日ダミー | 売上(Y) |
  |-------|-----|----------|--------|
  | 100   | 25  | 1        | 500    |
  | 150   | 28  | 0        | 750    |
  ```
- **サンプル数**: 説明変数の10倍以上が望ましい
- 欠損値の処理が必要

**3. 変数選択**
- **多重共線性のチェック**: VIF < 10 が望ましい
  - VIF（分散拡大要因）が高い変数は除外検討
  - 相関係数が0.8以上の変数は要注意
- **変数減少法**:
  - ステップワイズ法: 自動的に変数を追加・削除
  - p値やAICを基準に選択

**4. モデルの推定**
- モデル: Y = a + b₁X₁ + b₂X₂ + ... + bₙXₙ + ε
- 各係数は「他の変数を固定したときの影響」を表す

**5. モデルの評価**
- **決定係数（R²）**: 当てはまりの良さ
- **調整済みR²**: 変数の数を考慮した指標
  - 変数追加で必ずしも改善しない
- **F検定**: モデル全体の有意性
- **各係数のp値**: 個別の変数の有意性
- **残差分析**: 前提条件の確認

**6. 結果の解釈**
- 標準化係数（β）: 変数間の重要度比較
- 偏回帰係数: 実務的な影響の大きさ
- 予測精度の評価（RMSE, MAE）

**7. 注意点**
- 多重共線性: 説明変数間の強い相関
- 過学習: 変数が多すぎると訓練データにのみ適合
- 交互作用: 変数間の相乗効果は基本モデルで捉えられない
- 外れ値の影響が大きい
            """
        )

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

            # Statsmodels OLS for detailed statistical inference
            X_sm = sm.add_constant(X)
            ols_model = sm.OLS(y, X_sm).fit()

            st.markdown("### statsmodels 詳細結果")
            coef_table = pd.DataFrame({
                "変数": ["切片"] + x_cols,
                "係数": ols_model.params.values,
                "標準誤差": ols_model.bse.values,
                "t値": ols_model.tvalues.values,
                "p値": ols_model.pvalues.values,
                "95%CI下限": ols_model.conf_int()[0].values,
                "95%CI上限": ols_model.conf_int()[1].values,
            })
            coef_table["有意"] = coef_table["p値"].apply(
                lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            )
            st.dataframe(coef_table.set_index("変数").style.format({
                "係数": "{:.4f}", "標準誤差": "{:.4f}", "t値": "{:.4f}",
                "p値": "{:.4f}", "95%CI下限": "{:.4f}", "95%CI上限": "{:.4f}",
            }), use_container_width=True)

            sm_cols = st.columns(4)
            with sm_cols[0]:
                with st.container(border=True):
                    st.metric("F統計量", f"{ols_model.fvalue:.4f}")
            with sm_cols[1]:
                with st.container(border=True):
                    st.metric("F検定 p値", f"{ols_model.f_pvalue:.4f}")
            with sm_cols[2]:
                with st.container(border=True):
                    st.metric("AIC", f"{ols_model.aic:.2f}")
            with sm_cols[3]:
                with st.container(border=True):
                    st.metric("BIC", f"{ols_model.bic:.2f}")

            with st.expander("📖 statsmodels結果の解釈"):
                st.markdown(
                    """
**標準誤差（SE）**: 係数推定値のばらつき。小さいほど推定が安定している。

**t値**: 係数がゼロと有意に異なるかを検定する統計量。$t = \\hat{\\beta} / SE(\\hat{\\beta})$

**p値の有意水準記号**:
| 記号 | 意味 |
|------|------|
| `***` | p < 0.001（非常に強い証拠） |
| `**` | p < 0.01（強い証拠） |
| `*` | p < 0.05（有意） |
| （なし）| p ≥ 0.05（有意でない） |

**95%信頼区間（CI）**: 係数の真の値が95%の確率で含まれる範囲。区間が0を含む場合は有意でない。

**F統計量**: モデル全体の有意性を検定（すべての係数が同時にゼロかどうか）。

**AIC / BIC**: モデル選択基準。値が小さいほど良いモデル（変数選択の比較に使用）。
                    """
                )

            # Model performance metrics
            st.markdown("### モデル評価指標")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                with st.container(border=True):
                    st.metric("R²", f"{r2:.4f}")
            with col2:
                with st.container(border=True):
                    st.metric("調整済みR²", f"{adj_r2:.4f}")
            with col3:
                with st.container(border=True):
                    st.metric("RMSE", f"{rmse:.4f}")
            with col4:
                with st.container(border=True):
                    st.metric("MAE", f"{mae:.4f}")

            st.markdown(f"**交差検証R² (CV=5):** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            with st.expander("📖 モデル評価指標の解釈"):
                st.markdown(
                    f"""
**R²（決定係数）** と **調整済みR²**: モデルの説明力を示します（範囲: 0〜1）。説明変数が増えるほどR²は上昇しますが、調整済みR²は変数追加による上昇にペナルティをかけるため、変数選択の判断に適しています。

| R² | 評価 |
|----|------|
| 0.9 以上 | 非常に良い当てはまり |
| 0.7 〜 0.9 | 良い当てはまり |
| 0.5 〜 0.7 | 中程度の当てはまり |
| 0.5 未満 | 当てはまりが弱い |

$$R^2_{{adj}} = 1 - (1 - R^2) \\cdot \\frac{{n-1}}{{n-k-1}}$$

（$n$: サンプル数、$k$: 説明変数数）

**RMSE** と **MAE**: いずれも目的変数と同じ単位で解釈できます。RMSE は大きな誤差を重視し、MAE は外れ値の影響を受けにくいです。

現在の値: R²={r2:.4f}, 調整済みR²={adj_r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}
                    """
                )

            # Coefficients table
            st.markdown("### 回帰係数")
            coef_df = pd.DataFrame({
                "変数": ["切片"] + x_cols,
                "係数": [model.intercept_] + list(model.coef_),
            })
            st.dataframe(coef_df, use_container_width=True)

            # VIF (Variance Inflation Factor) for multicollinearity check
            st.markdown("### 多重共線性の確認（VIF）")
            vif_data = pd.DataFrame({
                "変数": x_cols,
                "VIF": [
                    variance_inflation_factor(X.values, i)
                    for i in range(X.shape[1])
                ],
            })
            vif_data["判定"] = vif_data["VIF"].apply(
                lambda v: "✅ 問題なし" if v < 5 else ("⚠️ 注意" if v < 10 else "❌ 多重共線性あり")
            )
            st.dataframe(vif_data, use_container_width=True)
            with st.expander("VIFの解釈"):
                st.markdown(
                    """
| VIF値 | 判定 | 説明 |
|-------|------|------|
| 1〜5未満 | ✅ 問題なし | 多重共線性の影響は軽微 |
| 5〜10未満 | ⚠️ 注意 | 多重共線性の可能性あり、検討が必要 |
| 10以上 | ❌ 多重共線性あり | 深刻な多重共線性、変数の削除や変換を検討 |

**VIF（分散膨張因子）の計算式:**

$$VIF_j = \\frac{1}{1 - R_j^2}$$

- $R_j^2$: 説明変数 $j$ を他の説明変数で回帰したときの決定係数
                    """
                )

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

            # Q-Q plot and histogram for residuals
            st.markdown("### 残差の正規性確認")
            col_qq, col_hist = st.columns(2)

            with col_qq:
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

            with col_hist:
                fig_hist = go.Figure()
                fig_hist.add_trace(
                    go.Histogram(
                        x=residuals,
                        nbinsx=30,
                        name="残差",
                        marker_color="steelblue",
                        opacity=0.7,
                    )
                )
                fig_hist.update_layout(
                    title="残差のヒストグラム",
                    xaxis_title="残差",
                    yaxis_title="頻度",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
