"""
Word frequency analysis module with Japanese support.
"""
import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
try:
    from janome.tokenizer import Tokenizer
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False


def show_word_frequency_analysis(df: pd.DataFrame):
    """Display word frequency analysis interface."""
    st.subheader("📝 単語頻度分析")

    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if not text_cols:
        st.warning("テキスト列が見つかりません。")
        return

    col1, col2 = st.columns(2)
    with col1:
        text_col = st.selectbox("テキスト列を選択", text_cols)
    with col2:
        language = st.selectbox("言語", ["日本語", "英語"])

    if language == "日本語" and not JANOME_AVAILABLE:
        st.error("Janomeがインストールされていません。`pip install janome`を実行してください。")
        return

    max_words = st.slider("表示する単語数", min_value=10, max_value=100, value=30)

    if st.button("単語頻度分析を実行", type="primary"):
        try:
            # Get text data
            texts = df[text_col].dropna()

            if len(texts) == 0:
                st.error("有効なテキストデータがありません。")
                return

            # Tokenize
            if language == "日本語":
                tokenizer = Tokenizer()
                words = []
                for text in texts:
                    tokens = tokenizer.tokenize(str(text))
                    words.extend([token.base_form for token in tokens if token.part_of_speech.split(",")[0] in ["名詞", "動詞", "形容詞"]])
            else:
                words = []
                for text in texts:
                    words.extend(str(text).lower().split())

            if len(words) == 0:
                st.error("有効な単語が抽出できませんでした。")
                return

            # Count frequencies
            word_counts = Counter(words)
            top_words = word_counts.most_common(max_words)

            st.success(f"単語頻度分析が完了しました！（総単語数: {len(words)}）")

            # Display frequency table
            st.markdown("### 頻出単語ランキング")
            freq_df = pd.DataFrame(top_words, columns=["単語", "出現回数"])
            freq_df["順位"] = range(1, len(freq_df) + 1)
            freq_df = freq_df[["順位", "単語", "出現回数"]]
            st.dataframe(freq_df, use_container_width=True)

            # Bar chart
            st.markdown("### 単語頻度グラフ")
            fig = px.bar(
                freq_df.head(20),
                x="単語",
                y="出現回数",
                title=f"Top 20 単語"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Word cloud
            st.markdown("### ワードクラウド")
            try:
                if language == "日本語":
                    # Use Japanese font
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color="white",
                        font_path="/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc" if plt.get_backend() == "MacOSX" else None
                    ).generate_from_frequencies(dict(word_counts))
                else:
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color="white"
                    ).generate_from_frequencies(dict(word_counts))

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"ワードクラウドの生成に失敗しました: {str(e)}")

            # Download
            csv = freq_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="単語頻度データをダウンロード",
                data=csv,
                file_name="word_frequency.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
