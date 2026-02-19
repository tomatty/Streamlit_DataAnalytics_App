"""
Conjoint Analysis module (simplified implementation).
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def show_conjoint_analysis(df: pd.DataFrame):
    """Display Conjoint Analysis interface."""
    st.subheader("📊 コンジョイント分析")

    with st.expander("📖 一般的な分析手順", expanded=False):
        st.markdown(
            """
### コンジョイント分析の基本的な流れ

**1. 目的の明確化**
- 製品・サービスの最適な組み合わせの発見
- 各属性（機能、価格、デザインなど）の重要度測定
- 顧客の選好構造の理解
- 新製品開発における意思決定支援
- 価格設定戦略の策定

**2. 調査設計（データ収集前）**
- **属性の選定**: 製品を特徴づける要素（3-6個が適切）
  - 例: スマートフォン → ブランド、価格、画面サイズ、バッテリー容量
- **水準の設定**: 各属性の選択肢（2-4水準が適切）
  - 例: 価格 → 3万円、5万円、7万円、10万円
- **プロファイル作成**: 属性の組み合わせパターン
  - 完全要因計画（全組み合わせ）または直交計画（一部抽出）
- **評価方法の決定**:
  - 順位法: プロファイルを順位付け
  - 評定法: 各プロファイルを点数評価（1-10点など）
  - 選択型: 複数プロファイルから1つ選択

**3. データの準備**
- **データ形式**:
  - 行：プロファイル（製品の組み合わせパターン）
  - 列：総合評価 + 各属性
  - **プロファイル形式**が基本（各行が1つの製品案の評価）
- **データ例（スマートフォン評価）**:
  ```
  | プロファイル | 総合評価 | ブランド | 価格   | 画面サイズ | バッテリー |
  |------------|---------|---------|-------|-----------|----------|
  | 1          | 7       | A社     | 5万円  | 6インチ    | 4000mAh  |
  | 2          | 5       | B社     | 3万円  | 5インチ    | 3000mAh  |
  | 3          | 8       | A社     | 7万円  | 6.5インチ  | 5000mAh  |
  | 4          | 4       | C社     | 3万円  | 5インチ    | 3000mAh  |
  ```
- **列の種類**:
  - 総合評価列：数値型（1-10点、1-7点など）
  - 属性列（カテゴリカル）：ブランド、色、サイズなど
  - 属性列（数値型）：価格、容量、重さなど
- 欠損値の処理
- **サンプル数の目安**: プロファイル数 ≥ (属性数 × 平均水準数) × 3

**4. 分析の実行**
- 線形回帰モデルで部分効用値を推定
- 属性のダミー変数化（カテゴリカル変数の場合）
- 各属性水準の効用値を算出

**5. 結果の解釈**
- **部分効用値（Part-worth utilities）**:
  - 各属性水準が総合評価に与える影響度
  - 正の値: 評価を上げる、負の値: 評価を下げる
  - 絶対値が大きいほど影響が大きい
- **相対的重要度（Relative importance）**:
  - 各属性が意思決定に占める重要性の割合（%）
  - 合計100%になる
  - 最も重要な属性を特定

**6. 活用方法**
- **最適製品の設計**: 効用値が最大になる組み合わせを選択
- **市場シミュレーション**: 競合製品との比較
- **セグメンテーション**: 顧客層ごとの選好の違いを分析
- **価格戦略**: 価格弾力性の測定
- **What-if分析**: 属性変更の影響予測

**7. 注意点**
- サンプル数は属性×水準数の3倍以上が望ましい
- 属性が多すぎると回答者の負担が大きい（疲労効果）
- 非現実的な組み合わせは除外する
- 交互作用効果は考慮されない（基本モデルの場合）
            """
        )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("コンジョイント分析には数値型の評価列と属性列が必要です。")
        return

    preference_col = st.selectbox("総合評価（目的変数）", numeric_cols)
    attribute_cols = st.multiselect(
        "属性（説明変数）",
        [c for c in all_cols if c != preference_col],
        help="製品の属性を選択してください"
    )

    if len(attribute_cols) < 1:
        st.info("少なくとも1つの属性を選択してください。")
        return

    if st.button("コンジョイント分析を実行", type="primary"):
        try:
            # Prepare data
            data_subset = df[[preference_col] + attribute_cols].dropna()

            # Handle categorical variables with one-hot encoding
            X = pd.get_dummies(data_subset[attribute_cols], drop_first=True)
            y = data_subset[preference_col]

            # Fit linear regression model (sklearn)
            model = LinearRegression()
            model.fit(X, y)

            # Fit OLS model (statsmodels) for detailed statistics
            X_with_const = sm.add_constant(X)
            ols_model = sm.OLS(y, X_with_const)
            ols_results = ols_model.fit()

            st.success("コンジョイント分析が完了しました！")

            # Regression analysis results
            st.markdown("### 回帰分析の詳細結果")
            st.caption("各係数の統計的有意性を確認できます。p値が0.05未満の場合、その属性は統計的に有意です。")

            # Create summary dataframe
            summary_df = pd.DataFrame({
                "変数": X_with_const.columns,
                "係数": ols_results.params,
                "標準誤差": ols_results.bse,
                "t値": ols_results.tvalues,
                "p値": ols_results.pvalues,
                "95%CI下限": ols_results.conf_int()[0],
                "95%CI上限": ols_results.conf_int()[1],
            })

            # Add significance stars
            def add_significance(p):
                if p < 0.001:
                    return "***"
                elif p < 0.01:
                    return "**"
                elif p < 0.05:
                    return "*"
                else:
                    return ""

            summary_df["有意"] = summary_df["p値"].apply(add_significance)

            # Display the table
            st.dataframe(
                summary_df.style.format({
                    "係数": "{:.4f}",
                    "標準誤差": "{:.4f}",
                    "t値": "{:.4f}",
                    "p値": "{:.4f}",
                    "95%CI下限": "{:.4f}",
                    "95%CI上限": "{:.4f}",
                }).background_gradient(subset=["p値"], cmap="RdYlGn_r", vmin=0, vmax=0.1),
                width="stretch"
            )

            # Model fit statistics
            st.markdown("#### モデル適合度")
            fit_cols = st.columns(4)
            with fit_cols[0]:
                with st.container(border=True):
                    st.metric("F統計量", f"{ols_results.fvalue:.2f}")
            with fit_cols[1]:
                with st.container(border=True):
                    st.metric("F検定p値", f"{ols_results.f_pvalue:.4f}")
            with fit_cols[2]:
                with st.container(border=True):
                    st.metric("AIC", f"{ols_results.aic:.2f}")
            with fit_cols[3]:
                with st.container(border=True):
                    st.metric("BIC", f"{ols_results.bic:.2f}")

            with st.expander("📖 統計指標の解釈"):
                st.markdown(
                    """
**回帰係数（係数）**: 各属性が総合評価に与える影響の大きさ
- 正の値: その属性は評価を上げる
- 負の値: その属性は評価を下げる

**p値**: 係数が統計的に有意かどうかを示す指標
- p < 0.05: 統計的に有意（その属性は評価に影響している）
- p ≥ 0.05: 統計的に有意でない（偶然の可能性）

**有意水準の目安**:
- ***: p < 0.001（非常に強い有意性）
- **: p < 0.01（強い有意性）
- *: p < 0.05（有意）
- （なし）: p ≥ 0.05（有意でない）

**95%信頼区間（CI）**: 係数の真の値が存在する範囲（95%の確率）
- 区間が0を含まない場合、その係数は有意

**F統計量・F検定p値**: モデル全体の有意性
- F検定p値 < 0.05 なら、モデル全体が有意

**AIC・BIC**: モデルの良さを示す指標（小さいほど良い）
- モデル選択時に使用
                    """
                )

            # Part-worth utilities
            st.markdown("### 部分効用値（Part-worth utilities）")

            # Parse attribute names and levels from dummy variable names
            utility_list = []
            for col, coef in zip(X.columns, model.coef_):
                # Try to split by underscore to separate attribute and level
                parts = col.split("_", 1)
                if len(parts) == 2:
                    attr_name, level = parts
                else:
                    # If no underscore, treat the whole column as both attribute and level
                    attr_name = col
                    level = col
                utility_list.append({"属性": attr_name, "水準": level, "効用値": coef})

            # Add reference levels (utility = 0) for each attribute
            # These are the levels that were dropped by drop_first=True
            original_data = data_subset[attribute_cols]
            for attr_col in attribute_cols:
                # Check if this attribute was one-hot encoded (categorical)
                if original_data[attr_col].dtype == 'object' or original_data[attr_col].dtype.name == 'category':
                    # Get the first level (reference level)
                    first_level = sorted(original_data[attr_col].unique())[0]
                    # Check if this reference level is not already in the list
                    attr_name = attr_col
                    if not any(u["属性"] == attr_name and u["水準"] == first_level for u in utility_list):
                        utility_list.append({"属性": attr_name, "水準": str(first_level), "効用値": 0.0})

            utilities_extended = pd.DataFrame(utility_list)

            # Display table with attribute and level information
            utilities_table = utilities_extended[["属性", "水準", "効用値"]].sort_values("効用値", ascending=False).reset_index(drop=True)
            st.dataframe(utilities_table, width="stretch")

            # Visualize part-worth utilities as line chart
            st.markdown("### 効用値グラフ")
            st.caption("各属性の水準ごとの効用値を折れ線グラフで表示します。効用値が高いほど、その水準が総合評価に正の影響を与えます。")

            if len(utilities_extended) > 0:
                # Separate categorical and numeric attributes
                categorical_attrs = []
                numeric_attrs = []

                for attr_name in utilities_extended["属性"].unique():
                    attr_data = utilities_extended[utilities_extended["属性"] == attr_name]
                    # If more than one level, it's categorical
                    if len(attr_data) > 1:
                        categorical_attrs.append(attr_name)
                    else:
                        numeric_attrs.append(attr_name)

                # Create subplots for categorical attributes
                if categorical_attrs:
                    n_attrs = len(categorical_attrs)
                    fig = make_subplots(
                        rows=1, cols=n_attrs,
                        subplot_titles=categorical_attrs,
                        horizontal_spacing=0.08
                    )

                    for i, attr_name in enumerate(categorical_attrs, start=1):
                        attr_data = utilities_extended[utilities_extended["属性"] == attr_name].sort_values("水準")

                        fig.add_trace(
                            go.Scatter(
                                x=attr_data["水準"],
                                y=attr_data["効用値"],
                                mode="lines+markers+text",
                                text=[f"{val:.2f}" for val in attr_data["効用値"]],
                                textposition="top center",
                                line=dict(width=2, color='steelblue'),
                                marker=dict(size=10, color='steelblue'),
                                showlegend=False,
                                name=attr_name
                            ),
                            row=1, col=i
                        )

                        # Add zero line
                        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=1, col=i)

                        # Update axes for this subplot
                        fig.update_xaxes(title_text="水準", tickangle=-45, row=1, col=i)
                        if i == 1:
                            fig.update_yaxes(title_text="効用値", row=1, col=i)

                    fig.update_layout(
                        title_text="部分効用値グラフ（カテゴリカル属性）",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, width="stretch")

                # Display numeric attributes as bar chart
                if numeric_attrs:
                    st.markdown("#### 数値型属性の係数")
                    numeric_data = utilities_extended[utilities_extended["属性"].isin(numeric_attrs)]

                    fig_numeric = go.Figure(go.Bar(
                        x=numeric_data["属性"],
                        y=numeric_data["効用値"],
                        text=[f"{val:.3f}" for val in numeric_data["効用値"]],
                        textposition='outside',
                        marker=dict(color=numeric_data["効用値"], colorscale='RdBu', showscale=False)
                    ))
                    fig_numeric.update_layout(
                        title="数値型属性の効用係数",
                        xaxis_title="属性",
                        yaxis_title="係数（1単位あたりの効用変化）",
                        height=400
                    )
                    fig_numeric.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_numeric, width="stretch")

                with st.expander("📖 効用値グラフの読み方"):
                    st.markdown(
                        """
**カテゴリカル属性のグラフ：**
- 各サブプロットが1つの属性を表します
- X軸：その属性の水準（例: ブランドA, B, C）
- Y軸：部分効用値
- 折れ線：水準間の効用値の変化
- 赤い破線：効用値0のライン（これより上は正の影響、下は負の影響）

**数値型属性のグラフ：**
- 各バーが1つの属性を表します
- Y軸：効用係数（その属性が1単位増えたときの効用変化）
- 例：価格の係数が-0.01なら、価格が100円上がると効用が1減少

**解釈のポイント：**
- 効用値が高い水準ほど、顧客の評価を高める
- 同じ属性内で効用値の差が大きいほど、その属性の選択が重要
- 数値型属性は線形関係を仮定（1単位増えるごとに一定の効用変化）
                        """
                    )

            # Relative importance
            st.markdown("### 相対的重要度")
            # Calculate the range (max - min) of utility values for each attribute
            importance_list = []
            for attr_name in utilities_extended["属性"].unique():
                attr_utilities = utilities_extended[utilities_extended["属性"] == attr_name]["効用値"]
                utility_range = attr_utilities.max() - attr_utilities.min()
                importance_list.append({"属性": attr_name, "範囲": utility_range})

            importance_df = pd.DataFrame(importance_list)
            total_range = importance_df["範囲"].sum()
            importance_df["重要度(%)"] = (importance_df["範囲"] / total_range * 100).round(2)
            importance_df = importance_df[["属性", "重要度(%)"]].sort_values("重要度(%)", ascending=False)

            col_table, col_chart = st.columns([1, 1])
            with col_table:
                st.dataframe(importance_df, width="stretch")
            with col_chart:
                fig_importance = go.Figure(go.Bar(
                    x=importance_df["重要度(%)"],
                    y=importance_df["属性"],
                    orientation='h',
                    marker=dict(color=importance_df["重要度(%)"], colorscale='Blues', showscale=False),
                    text=importance_df["重要度(%)"].apply(lambda x: f"{x:.1f}%"),
                    textposition='outside'
                ))
                fig_importance.update_layout(
                    title="相対的重要度",
                    xaxis_title="重要度(%)",
                    yaxis_title="",
                    height=max(300, len(importance_df) * 50),
                    yaxis=dict(categoryorder='total ascending')
                )
                st.plotly_chart(fig_importance, width="stretch")

            # Model fit
            r2 = model.score(X, y)
            with st.container(border=True):
                st.metric("決定係数 (R²)", f"{r2:.4f}")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
