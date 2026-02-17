"""
Sentiment analysis module (basic implementation).
"""
import pandas as pd
import streamlit as st
import plotly.express as px


def show_sentiment_analysis(df: pd.DataFrame):
    """Display sentiment analysis interface."""
    st.subheader("😊 感情分析")

    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if not text_cols:
        st.warning("テキスト列が見つかりません。")
        return

    st.info("感情分析には事前にラベル付けされた感情列が必要です。")

    col1, col2 = st.columns(2)
    with col1:
        text_col = st.selectbox("テキスト列を選択", text_cols)
    with col2:
        sentiment_col = st.selectbox("感情ラベル列を選択", text_cols)

    if st.button("感情分析を実行", type="primary"):
        try:
            data_subset = df[[text_col, sentiment_col]].dropna()

            # Sentiment distribution
            sentiment_counts = data_subset[sentiment_col].value_counts()

            st.success("感情分析が完了しました！")

            # Display distribution
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 感情分布")
                st.dataframe(sentiment_counts, use_container_width=True)

            with col2:
                st.markdown("### 感情比率")
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="感情の分布"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Bar chart
            fig_bar = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                labels={"x": "感情", "y": "件数"},
                title="感情別件数"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Sample texts for each sentiment
            st.markdown("### 感情別サンプルテキスト")
            for sentiment in sentiment_counts.index:
                with st.expander(f"{sentiment} のサンプル"):
                    samples = data_subset[data_subset[sentiment_col] == sentiment][text_col].head(5)
                    for i, text in enumerate(samples, 1):
                        st.text(f"{i}. {text}")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
