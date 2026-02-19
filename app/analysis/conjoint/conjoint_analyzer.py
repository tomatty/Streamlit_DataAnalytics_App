"""
Conjoint Analysis module (simplified implementation).
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression


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
- **総合評価列**: 回答者の評価スコア（数値型）
- **属性列**: 各プロファイルの属性値
  - 数値型: 価格、サイズなど
  - カテゴリカル型: ブランド、色など（ダミー変数化される）
- 欠損値の処理
- 回答者ごとのデータ構造確認

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

            # Fit linear regression model
            model = LinearRegression()
            model.fit(X, y)

            st.success("コンジョイント分析が完了しました！")

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
                fig = go.Figure()

                # Create x-axis labels: "Attribute: Level"
                utilities_extended = utilities_extended.copy()
                utilities_extended["x_label"] = utilities_extended["属性"] + ": " + utilities_extended["水準"]

                # Sort by attribute and level for consistent ordering
                utilities_extended = utilities_extended.sort_values(["属性", "水準"])

                # Get all unique x labels
                all_x_labels = utilities_extended["x_label"].tolist()

                # For each attribute, create a line that connects only its levels
                for attr_name in utilities_extended["属性"].unique():
                    attr_data = utilities_extended[utilities_extended["属性"] == attr_name]

                    # Create arrays with None for positions that don't belong to this attribute
                    x_vals = []
                    y_vals = []
                    text_vals = []

                    for x_label in all_x_labels:
                        if x_label in attr_data["x_label"].values:
                            row = attr_data[attr_data["x_label"] == x_label].iloc[0]
                            x_vals.append(x_label)
                            y_vals.append(row["効用値"])
                            text_vals.append(f"{row['効用値']:.2f}")
                        else:
                            x_vals.append(x_label)
                            y_vals.append(None)
                            text_vals.append("")

                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines+markers+text",
                        name=attr_name,
                        text=text_vals,
                        textposition="top center",
                        line=dict(width=2),
                        marker=dict(size=8),
                        connectgaps=False  # Don't connect across None values
                    ))

                fig.update_layout(
                    title="部分効用値グラフ",
                    xaxis_title="属性と水準",
                    yaxis_title="効用値",
                    hovermode="closest",
                    showlegend=True,
                    height=500,
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black', zerolinewidth=1)
                )
                fig.update_xaxis(tickangle=-45)
                st.plotly_chart(fig, width="stretch")

                with st.expander("📖 効用値グラフの読み方"):
                    st.markdown(
                        """
**グラフの見方：**
- **X軸**: 各属性とその水準（例: CPU: Celeron, HDD容量: 5GB）
- **Y軸**: 部分効用値（正の値は評価を上げる、負の値は評価を下げる）
- **折れ線**: 各属性内での水準間の効用値の変化
- **ゼロ線**: 効用値0のライン（これより上は正の影響、下は負の影響）

**解釈のポイント：**
- 効用値が高い水準ほど、顧客の評価を高める
- 同じ属性内で効用値の差が大きいほど、その属性の選択が重要
- 数値型属性（価格など）は係数として表示され、1単位あたりの効用変化を示す
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
