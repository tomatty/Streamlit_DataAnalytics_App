"""
Topic modeling module using LDA.
"""
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px


def show_topic_modeling(df: pd.DataFrame):
    """Display topic modeling interface."""
    st.subheader("📚 トピックモデリング（LDA）")

    with st.expander("📖 一般的な分析手順", expanded=False):
        st.markdown(
            """
### トピックモデリング（LDA）の基本的な流れ

**1. 目的の明確化**
- 大量文書の主要テーマ発見: 数百〜数千件の文書から話題を抽出
- 顧客の関心事の分類: レビューやアンケートの自由記述をトピック分け
- コンテンツの自動タグ付け: 記事やドキュメントを自動分類
- トレンドの把握: 時期ごとの話題の変化を追跡

**2. データの準備**
- **データ形式**:
  ```
  | ID | 文書テキスト                                    |
  |----|-----------------------------------------------|
  | 1  | 新製品のデザインが優れており使いやすい...       |
  | 2  | カスタマーサポートの対応が迅速で丁寧だった...   |
  | 3  | 価格設定が妥当で品質も高く満足している...       |
  ```
- **必要なデータ量**: 最低50件、できれば100件以上の文書
- 各文書がある程度の長さがあることが望ましい（短すぎると意味がない）

**3. LDA（Latent Dirichlet Allocation）とは**
- **潜在トピック**: 文書の背後にある隠れたテーマを自動発見
- **確率モデル**: 各文書は複数のトピックの混合で構成される
  - 例: 文書1 = トピックA 70% + トピックB 30%
- **トピックの定義**: 単語の分布で表現される
  - 例: トピックA = 「品質」30% + 「デザイン」25% + 「使いやすさ」20% + ...

**4. パラメータの設定**
- **トピック数**: 発見したいテーマの数（2-20程度が一般的）
  - 少なすぎる→粗い分類
  - 多すぎる→細かすぎて解釈困難
  - パープレキシティやコヒーレンスで最適値を探索
- **最大特徴量数**: 使用する単語の種類の上限
  - 多いほど詳細だが計算コスト増

**5. 前処理**
- ストップワード除去（「これ」「その」など）
- 低頻度語の除去（2回以下の単語など）
- 高頻度語の除去（全文書に出現する単語）
- 数字・記号の除去

**6. モデルの学習**
- 文書-単語行列の作成
- LDAアルゴリズムで潜在トピックを推定
- 各トピックの代表単語を抽出

**7. 結果の解釈**
- **各トピックの特徴**:
  - 上位単語から意味を読み取る
  - トピックに名前を付ける（「品質関連」「サポート関連」など）
- **文書の所属トピック**:
  - 各文書がどのトピックに強く関連するか
  - トピックの分布を確認

**8. 活用方法**
- 文書の自動分類
- トピック別の時系列トレンド分析
- 各トピックに対する感情分析
- 重要トピックの優先順位付け

**9. 注意点**
- トピック数の設定が結果に大きく影響
- ランダム性があり、実行ごとに結果が変わる
- 短い文書では精度が低い
- トピックの解釈は主観的
- 計算時間がかかる（大規模データの場合）
            """
        )

    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if not text_cols:
        st.warning("テキスト列が見つかりません。")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        text_col = st.selectbox("テキスト列を選択", text_cols)
    with col2:
        n_topics = st.slider("トピック数", min_value=2, max_value=20, value=5)
    with col3:
        max_features = st.number_input("最大特徴量数", min_value=10, max_value=1000, value=100, step=10)

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
