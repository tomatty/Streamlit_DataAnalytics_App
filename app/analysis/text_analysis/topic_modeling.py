"""
Topic modeling module using LDA.
"""
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
from app.auth.session_manager import SessionManager


def show_topic_modeling(df: pd.DataFrame):
    """Display topic modeling interface."""
    st.subheader("📚 トピックモデリング（LDA）")

    # Get default values from settings
    default_n_topics = SessionManager.get_setting("n_topics", 5)
    default_max_features = SessionManager.get_setting("max_features", 100)

    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if not text_cols:
        st.warning("テキスト列が見つかりません。")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        text_col = st.selectbox("テキスト列を選択", text_cols)
    with col2:
        n_topics = st.slider("トピック数", min_value=2, max_value=20, value=int(default_n_topics))
    with col3:
        max_features = st.number_input("最大特徴量数", min_value=10, max_value=1000, value=int(default_max_features), step=10)

    if st.button("トピックモデリングを実行", type="primary"):
        try:
            texts = df[text_col].dropna().astype(str)

            if len(texts) < n_topics:
                st.error(f"データ数がトピック数より少ないです。少なくとも{n_topics}個のテキストが必要です。")
                return

            # Vectorize
            vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
            doc_term_matrix = vectorizer.fit_transform(texts)

            # LDA
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(doc_term_matrix)

            st.success("トピックモデリングが完了しました！")

            # Display topics
            st.markdown("### トピック別のキーワード")
            feature_names = vectorizer.get_feature_names_out()

            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                st.markdown(f"**トピック {topic_idx + 1}:** {', '.join(top_words)}")

            # Document-topic distribution
            doc_topic_dist = lda.transform(doc_term_matrix)

            st.markdown("### 文書のトピック分布")
            result_df = pd.DataFrame(doc_topic_dist, columns=[f"トピック{i+1}" for i in range(n_topics)])
            result_df["主トピック"] = result_df.idxmax(axis=1)
            st.dataframe(result_df.head(20), use_container_width=True)

            # Topic distribution
            topic_counts = result_df["主トピック"].value_counts()
            fig = px.pie(values=topic_counts.values, names=topic_counts.index, title="トピック分布")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
