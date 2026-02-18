"""
Conjoint Analysis module (simplified implementation).
"""
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


def show_conjoint_analysis(df: pd.DataFrame):
    """Display Conjoint Analysis interface."""
    st.subheader("📊 コンジョイント分析")

    st.info("コンジョイント分析は、製品やサービスの属性が全体的な評価にどの程度寄与しているかを分析します。")

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
            utilities = pd.DataFrame({
                "属性": X.columns,
                "効用値": model.coef_
            }).sort_values("効用値", ascending=False)
            st.dataframe(utilities, use_container_width=True)

            # Relative importance
            st.markdown("### 相対的重要度")
            importance = (utilities["効用値"].abs() / utilities["効用値"].abs().sum() * 100).values
            importance_df = pd.DataFrame({
                "属性": utilities["属性"],
                "重要度(%)": importance
            })
            st.dataframe(importance_df, use_container_width=True)

            # Model fit
            r2 = model.score(X, y)
            with st.container(border=True):
                st.metric("決定係数 (R²)", f"{r2:.4f}")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
