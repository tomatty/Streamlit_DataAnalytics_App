"""
Word frequency analysis module with Japanese support.
"""
import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    from janome.tokenizer import Tokenizer
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False

_JP_FONT_CANDIDATES = [
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "IPAGothic",
    "TakaoGothic",
    "VL Gothic",
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Yu Gothic",
]


def _find_jp_font() -> str | None:
    """
    Find an available Japanese font on the system.

    Returns the path to the font file if found, otherwise None.
    Works across macOS, Linux (including Docker), and Windows.
    """
    try:
        fm._load_fontmanager(try_read_cache=False)
    except Exception:
        pass

    available_fonts = {f.name: f.fname for f in fm.fontManager.ttflist}

    for candidate in _JP_FONT_CANDIDATES:
        if candidate in available_fonts:
            return available_fonts[candidate]

    return None


def _parse_rgb_color(color: str) -> tuple[float, float, float]:
    """
    Convert WordCloud's 'rgb(r, g, b)' string to matplotlib-compatible tuple.

    Args:
        color: Color string in format 'rgb(r, g, b)' where r, g, b are 0-255

    Returns:
        Tuple of (r, g, b) normalized to 0-1 range
    """
    import re
    match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
    if match:
        r, g, b = map(int, match.groups())
        return (r / 255.0, g / 255.0, b / 255.0)
    # Fallback to black if parsing fails
    return (0.0, 0.0, 0.0)


def _render_wordcloud_matplotlib(wc: WordCloud, font_path: str | None) -> plt.Figure:
    """
    Re-render WordCloud layout using matplotlib's FreeType renderer.

    WordCloud uses PIL internally, which cannot always access Japanese glyphs
    from macOS .ttc files. This function takes the already-computed layout
    (word positions, sizes, orientations) and draws each word with matplotlib,
    which handles Japanese fonts via FreeType correctly.
    """
    WC_WIDTH = wc.width
    WC_HEIGHT = wc.height

    font_props = fm.FontProperties(fname=font_path) if font_path else None
    if font_path:
        fm.fontManager.addfont(font_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(wc.background_color)
    ax.set_facecolor(wc.background_color)
    # Match PIL coordinate system: origin top-left, y increases downward
    ax.set_xlim(0, WC_WIDTH)
    ax.set_ylim(WC_HEIGHT, 0)
    ax.axis("off")
    plt.tight_layout(pad=0)

    for (word, _), font_size, (x, y), orientation, color in wc.layout_:
        rot = 90 if orientation else 0
        # Convert PIL/WordCloud color format to matplotlib format
        mpl_color = _parse_rgb_color(color) if isinstance(color, str) and color.startswith('rgb(') else color
        ax.text(
            x, y, word,
            fontsize=font_size * 0.75,
            fontproperties=font_props,
            color=mpl_color,
            rotation=rot,
            ha="left",
            va="top",
        )

    return fig


def show_word_frequency_analysis(df: pd.DataFrame):
    """Display word frequency analysis interface."""
    st.subheader("📝 単語頻度分析")

    with st.expander("📖 一般的な分析手順", expanded=False):
        st.markdown(
            """
### 単語頻度分析の基本的な流れ

**1. 目的の明確化**
- 頻出単語の把握: どの単語が多く使われているか
- トレンドの発見: 話題のキーワードを特定
- 顧客の声の分析: レビュー・アンケートの自由記述から傾向を把握
- コンテンツの特徴抽出: 文書の特徴的な単語を発見

**2. データの準備**
- **データ形式**:
  ```
  | ID | テキスト列                          |
  |----|-----------------------------------|
  | 1  | この商品は使いやすく満足しています   |
  | 2  | サポートが親切で助かりました         |
  | 3  | 価格が高いですが品質は良いです       |
  ```
- テキスト列（object型）が必要
- 1行に1つのテキスト（レビュー、コメント、回答など）
- 最低でも10件以上のテキストが望ましい

**3. 言語の選択**
- **日本語**: 形態素解析（Janome）で単語分割
  - 「使いやすい」→「使い」「やすい」に分割
  - 品詞フィルタ（名詞・動詞・形容詞など）
- **英語**: 空白で単語分割
  - ストップワード除去（the, is, a など）

**4. 前処理**
- **ノイズ除去**: URL、記号、数字など
- **ストップワード除去**: 意味の薄い単語（「これ」「その」など）
- **正規化**: 大文字小文字の統一
- **語幹抽出**: 活用形の統一（英語の場合）

**5. 頻度の集計**
- 単語ごとの出現回数をカウント
- 上位N件を抽出（通常20-50件）
- 出現文書数も考慮（TF-IDFを使うとより高度）

**6. 可視化**
- **棒グラフ**: 上位単語の頻度を比較
- **ワードクラウド**: 直感的に頻出単語を表示
  - 大きい文字ほど頻度が高い
  - 視覚的なインパクトが強い

**7. 結果の解釈**
- 頻出単語から主要なテーマを把握
- 意外な単語の発見（新しいニーズ、問題点）
- ポジティブ/ネガティブな単語の分布
- 時系列での変化（トレンド分析）

**8. 注意点**
- 頻度だけでは文脈がわからない
- ストップワード設定で結果が変わる
- 同義語が別単語として扱われる
- 短いテキストでは意味のある分析が難しい
            """
        )

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

    max_words = st.slider("分析する単語数（ワードクラウド上限）", min_value=10, max_value=100, value=50)

    if st.button("単語頻度分析を実行", type="primary"):
        try:
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
                    words.extend([
                        token.base_form for token in tokens
                        if token.part_of_speech.split(",")[0] in ["名詞", "動詞", "形容詞"]
                    ])
            else:
                words = []
                for text in texts:
                    words.extend(str(text).lower().split())

            if len(words) == 0:
                st.error("有効な単語が抽出できませんでした。")
                return

            word_counts = Counter(words)
            top_words = word_counts.most_common(max_words)
            freq_df = pd.DataFrame(top_words, columns=["単語", "出現回数"])
            freq_df["順位"] = range(1, len(freq_df) + 1)
            freq_df = freq_df[["順位", "単語", "出現回数"]]

            # Persist results so slider reruns don't lose them
            st.session_state["wf_freq_df"] = freq_df
            st.session_state["wf_word_counts"] = dict(word_counts)
            st.session_state["wf_total_words"] = len(words)
            st.session_state["wf_max_words"] = max_words
            st.session_state["wf_language"] = language

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")

    # Render results (survives slider reruns via session_state)
    if "wf_freq_df" not in st.session_state:
        return

    freq_df = st.session_state["wf_freq_df"]
    word_counts = st.session_state["wf_word_counts"]
    total_words = st.session_state["wf_total_words"]
    saved_max_words = st.session_state["wf_max_words"]
    saved_language = st.session_state["wf_language"]

    st.success(f"単語頻度分析が完了しました！（総単語数: {total_words}）")

    # Frequency table
    st.markdown("### 頻出単語ランキング")
    st.dataframe(freq_df, use_container_width=True)

    # Horizontal bar chart with display-count control
    st.markdown("### 単語頻度グラフ")
    chart_n = st.slider(
        "グラフ表示数",
        min_value=5,
        max_value=min(saved_max_words, len(freq_df)),
        value=min(20, len(freq_df)),
        key="chart_n_slider",
    )
    chart_df = freq_df.head(chart_n).sort_values("出現回数", ascending=True)
    fig = px.bar(
        chart_df,
        x="出現回数",
        y="単語",
        orientation="h",
        title=f"Top {chart_n} 単語",
    )
    fig.update_layout(yaxis={"tickfont": {"size": 11}})
    st.plotly_chart(fig, use_container_width=True)

    # Word cloud
    st.markdown("### ワードクラウド")
    try:
        font_path = _find_jp_font() if saved_language == "日本語" else None
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            font_path=font_path,
            max_words=saved_max_words,
        ).generate_from_frequencies(word_counts)

        if saved_language == "日本語" and font_path:
            fig_wc = _render_wordcloud_matplotlib(wc, font_path)
        else:
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")

        st.pyplot(fig_wc)
        plt.close(fig_wc)
    except Exception as e:
        st.warning(f"ワードクラウドの生成に失敗しました: {str(e)}")

    # Download
    csv = freq_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="単語頻度データをダウンロード",
        data=csv,
        file_name="word_frequency.csv",
        mime="text/csv",
    )
