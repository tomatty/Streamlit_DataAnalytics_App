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
import statsmodels.api as sm


def show_simple_regression(df: pd.DataFrame):
    """
    Display simple linear regression analysis.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("📈 単回帰分析")

    with st.expander("📖 一般的な分析手順", expanded=False):
        st.markdown(
            """
### 単回帰分析の基本的な流れ

**1. 目的の明確化**
- 因果関係の推定: XがYに与える影響を定量化
- 予測: Xの値からYの値を推定
- トレンド分析: 時系列データの傾向把握
- 効果測定: 施策の効果を数値化

**2. データの準備**
- **データ形式**:
  - 行：サンプル/観測（例: 時点、顧客、店舗）
  - 列：説明変数X（1つ）、目的変数Y（1つ）
- **データ例**:
  ```
  | 広告費(X) | 売上(Y) |
  |----------|--------|
  | 100      | 500    |
  | 150      | 750    |
  | 200      | 900    |
  ```
- 欠損値の処理が必要
- 外れ値の影響を受けやすいため事前確認
- **カテゴリー変数の扱い**:
  - 線形回帰ではカテゴリー変数を数値化する必要がある
  - **ワンホットエンコーディング（ダミー変数化）**: 順序のないカテゴリー変数に使用
    - 例: 地域（東京/大阪/名古屋）→ 地域_大阪（0/1）、地域_名古屋（0/1）
    - 多重共線性を避けるため、n-1個のダミー変数を作成
  - **順序エンコーディング**: 順序のあるカテゴリー変数に使用
    - 例: 満足度（低/中/高）→ 1/2/3

**3. 前提条件の確認**
- **線形性**: XとYが直線的な関係にある
- **等分散性**: 残差の分散が一定
- **正規性**: 残差が正規分布に従う
- **独立性**: 観測値が互いに独立
- 散布図で線形関係を目視確認

**4. 回帰式の推定**
- モデル: Y = a + bX + ε
  - a: 切片（Xが0のときのY）
  - b: 傾き（Xが1増えたときのYの変化量）
  - ε: 誤差項
- 最小二乗法で係数を推定

**5. モデルの評価**
- **決定係数（R²）**: 0-1の値、1に近いほど当てはまりが良い
  - 0.7以上: 強い関係
  - 0.4-0.7: 中程度の関係
  - 0.4未満: 弱い関係
- **p値**: 係数の有意性検定（p < 0.05で有意）
- **残差プロット**: 等分散性の確認

**6. 結果の解釈**
- 回帰係数の符号と大きさ
- 説明変数の単位あたりの影響
- 予測精度の評価

**7. 注意点**
- 相関関係であり因果関係ではない
- 外挿（データ範囲外の予測）は危険
- 外れ値に敏感
- 非線形関係には不適
            """
        )

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

            # Statsmodels OLS for detailed statistical inference
            X_sm = sm.add_constant(X[x_col].values)
            ols_model = sm.OLS(y.values, X_sm).fit()

            # Regression equation
            st.markdown("### 回帰式")
            st.latex(
                f"\\hat{{y}} = {model.intercept_:.4f} + {model.coef_[0]:.4f} \\cdot x"
            )

            # Statsmodels detailed results
            st.markdown("### statsmodels 詳細結果")
            coef_table = pd.DataFrame({
                "変数": ["切片", x_col],
                "係数": ols_model.params,
                "標準誤差": ols_model.bse,
                "t値": ols_model.tvalues,
                "p値": ols_model.pvalues,
                "95%CI下限": ols_model.conf_int()[0],
                "95%CI上限": ols_model.conf_int()[1],
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
**標準誤差（SE）**: 係数推定値のばらつきの大きさ。小さいほど推定が安定している。

**t値**: 係数がゼロと有意に異なるかを検定する統計量。$t = \\hat{\\beta} / SE(\\hat{\\beta})$

**p値の有意水準記号**:
| 記号 | 意味 |
|------|------|
| `***` | p < 0.001（非常に強い証拠） |
| `**` | p < 0.01（強い証拠） |
| `*` | p < 0.05（有意） |
| （なし）| p ≥ 0.05（有意でない） |

**95%信頼区間（CI）**: 係数の真の値が95%の確率で含まれる範囲。区間が0を含む場合は有意でない。

**F統計量**: モデル全体の有意性を検定。p値が小さいほど回帰モデルが有意。

**AIC / BIC**: モデルの情報量基準。値が小さいほど良いモデル（複数モデルの比較に使用）。
                    """
                )

            # Model performance metrics
            st.markdown("### モデル評価指標")
            col1, col2, col3 = st.columns(3)
            with col1:
                with st.container(border=True):
                    st.metric("決定係数 (R²)", f"{r2:.4f}")
            with col2:
                with st.container(border=True):
                    st.metric("RMSE", f"{rmse:.4f}")
            with col3:
                with st.container(border=True):
                    st.metric("MAE", f"{mae:.4f}")

            with st.expander("📖 モデル評価指標の解釈"):
                st.markdown(
                    f"""
**決定係数（R²）**: モデルが目的変数の変動をどの程度説明できるかを示します（範囲: 0〜1）。

| R² | 評価 |
|----|------|
| 0.9 以上 | 非常に良い当てはまり |
| 0.7 〜 0.9 | 良い当てはまり |
| 0.5 〜 0.7 | 中程度の当てはまり |
| 0.5 未満 | 当てはまりが弱い |

$$R^2 = 1 - \\frac{{\\sum(y_i - \\hat{{y}}_i)^2}}{{\\sum(y_i - \\bar{{y}})^2}}$$

**RMSE（二乗平均平方根誤差）**: 予測誤差の標準偏差。目的変数と同じ単位で解釈でき、外れ値の影響を受けやすい。

$$RMSE = \\sqrt{{\\frac{{1}}{{n}}\\sum_{{i=1}}^{{n}}(y_i - \\hat{{y}}_i)^2}}$$

**MAE（平均絶対誤差）**: 予測誤差の絶対値の平均。外れ値の影響を受けにくく、直感的に解釈しやすい。

$$MAE = \\frac{{1}}{{n}}\\sum_{{i=1}}^{{n}}|y_i - \\hat{{y}}_i|$$

現在の値: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}
                    """
                )

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
