"""
Analysis page module - integrates all analysis functionality.
"""
import streamlit as st
from app.auth.session_manager import SessionManager


def show_analysis(analysis_category: str | None = None):
    """Display analysis page with all analysis options."""
    st.header("📈 分析")

    if not SessionManager.has_data():
        st.warning(
            "データがアップロードされていません。ファイルアップロードページからデータをアップロードしてください。"
        )
        return

    data = SessionManager.get_data()

    if analysis_category is None:
        analysis_category = st.selectbox(
            "分析カテゴリー",
            [
                "記述統計・集計",
                "仮説検定",
                "相関分析",
                "回帰分析",
                "決定木分析",
                "多変量解析",
                "クラスタリング",
                "テキスト分析",
                "専門分析",
            ]
        )

    st.markdown("---")

    # Display appropriate analysis based on category
    if analysis_category == "記述統計・集計":
        show_descriptive_analysis(data)
    elif analysis_category == "仮説検定":
        show_hypothesis_testing_page(data)
    elif analysis_category == "相関分析":
        show_correlation_analysis_page(data)
    elif analysis_category == "回帰分析":
        show_regression_analysis_page(data)
    elif analysis_category == "決定木分析":
        show_tree_analysis_page(data)
    elif analysis_category == "多変量解析":
        show_multivariate_analysis_page(data)
    elif analysis_category == "クラスタリング":
        show_clustering_analysis_page(data)
    elif analysis_category == "テキスト分析":
        show_text_analysis_page(data)
    elif analysis_category == "専門分析":
        show_specialized_analysis_page(data)


def show_descriptive_analysis(data):
    """Descriptive statistics and aggregation."""
    from app.analysis.descriptive.crosstab import show_crosstab_analysis
    from app.analysis.descriptive.aggregation import show_aggregation_analysis

    analysis_type = st.radio(
        "分析タイプ",
        ["crosstab", "aggregation"],
        format_func=lambda x: {"crosstab": "クロス集計", "aggregation": "グループ集計"}[x],
        horizontal=True
    )

    if analysis_type == "crosstab":
        show_crosstab_analysis(data)
    else:
        show_aggregation_analysis(data)


def show_correlation_analysis_page(data):
    """Correlation analysis."""
    from app.analysis.correlation.correlation_matrix import show_correlation_analysis
    from app.analysis.correlation.pairplot import show_pairplot_analysis, show_scatter_plot

    analysis_type = st.radio(
        "分析タイプ",
        ["correlation", "pairplot", "scatter"],
        format_func=lambda x: {"correlation": "相関行列", "pairplot": "ペアプロット", "scatter": "散布図"}[x],
        horizontal=True
    )

    if analysis_type == "correlation":
        show_correlation_analysis(data)
    elif analysis_type == "pairplot":
        show_pairplot_analysis(data)
    else:
        show_scatter_plot(data)


def show_regression_analysis_page(data):
    """Regression analysis."""
    from app.analysis.regression.simple_regression import show_simple_regression
    from app.analysis.regression.multiple_regression import show_multiple_regression

    analysis_type = st.radio(
        "分析タイプ",
        ["simple", "multiple"],
        format_func=lambda x: {"simple": "単回帰分析", "multiple": "重回帰分析"}[x],
        horizontal=True
    )

    if analysis_type == "simple":
        show_simple_regression(data)
    else:
        show_multiple_regression(data)


def show_hypothesis_testing_page(data):
    """Hypothesis testing."""
    from app.analysis.hypothesis_testing.t_test import show_t_test
    from app.analysis.hypothesis_testing.chi_square import show_chi_square_test
    from app.analysis.hypothesis_testing.anova import show_anova
    from app.analysis.hypothesis_testing.independence_test import show_independence_test
    from app.analysis.hypothesis_testing.sample_size import show_sample_size_calculation

    analysis_type = st.radio(
        "分析タイプ",
        ["t_test", "chi_square", "independence", "anova", "sample_size"],
        format_func=lambda x: {
            "t_test": "t検定",
            "chi_square": "カイ二乗検定",
            "independence": "独立性の検定",
            "anova": "ANOVA",
            "sample_size": "サンプルサイズ計算"
        }[x],
        horizontal=True
    )

    if analysis_type == "t_test":
        show_t_test(data)
    elif analysis_type == "chi_square":
        show_chi_square_test(data)
    elif analysis_type == "independence":
        show_independence_test(data)
    elif analysis_type == "anova":
        show_anova(data)
    else:
        show_sample_size_calculation()


def show_tree_analysis_page(data):
    """Decision tree analysis."""
    from app.analysis.tree.decision_tree import show_decision_tree
    from app.analysis.tree.lightgbm_tree import show_lightgbm_tree

    analysis_type = st.radio(
        "分析タイプ",
        ["decision_tree", "lightgbm"],
        format_func=lambda x: {
            "decision_tree": "決定木（sklearn）",
            "lightgbm": "LightGBM",
        }[x],
        horizontal=True,
    )

    if analysis_type == "decision_tree":
        show_decision_tree(data)
    else:
        show_lightgbm_tree(data)


def show_multivariate_analysis_page(data):
    """Multivariate analysis."""
    from app.analysis.dimensionality.pca import show_pca_analysis
    from app.analysis.dimensionality.factor_analysis import show_factor_analysis
    from app.analysis.dimensionality.correspondence import show_correspondence_analysis
    from app.analysis.conjoint.conjoint_analyzer import show_conjoint_analysis

    analysis_type = st.radio(
        "分析タイプ",
        ["pca", "factor", "correspondence", "conjoint"],
        format_func=lambda x: {
            "pca": "主成分分析（PCA）",
            "factor": "因子分析",
            "correspondence": "コレスポンデンス分析",
            "conjoint": "コンジョイント分析"
        }[x],
        horizontal=True
    )

    if analysis_type == "pca":
        show_pca_analysis(data)
    elif analysis_type == "factor":
        show_factor_analysis(data)
    elif analysis_type == "correspondence":
        show_correspondence_analysis(data)
    else:
        show_conjoint_analysis(data)


def show_clustering_analysis_page(data):
    """Clustering analysis."""
    from app.analysis.clustering.kmeans import show_kmeans_clustering
    from app.analysis.clustering.hierarchical import show_hierarchical_clustering
    from app.analysis.clustering.dbscan import show_dbscan_clustering

    analysis_type = st.radio(
        "分析タイプ",
        ["kmeans", "hierarchical", "dbscan"],
        format_func=lambda x: {"kmeans": "K-Means", "hierarchical": "階層的", "dbscan": "DBSCAN"}[x],
        horizontal=True
    )

    if analysis_type == "kmeans":
        show_kmeans_clustering(data)
    elif analysis_type == "hierarchical":
        show_hierarchical_clustering(data)
    else:
        show_dbscan_clustering(data)


def show_text_analysis_page(data):
    """Text analysis."""
    from app.analysis.text_analysis.word_frequency import show_word_frequency_analysis
    from app.analysis.text_analysis.topic_modeling import show_topic_modeling
    from app.analysis.text_analysis.sentiment import show_sentiment_analysis

    analysis_type = st.radio(
        "分析タイプ",
        ["word_freq", "topic", "sentiment"],
        format_func=lambda x: {"word_freq": "単語頻度", "topic": "トピックモデリング", "sentiment": "感情分析"}[x],
        horizontal=True
    )

    if analysis_type == "word_freq":
        show_word_frequency_analysis(data)
    elif analysis_type == "topic":
        show_topic_modeling(data)
    else:
        show_sentiment_analysis(data)


def show_specialized_analysis_page(data):
    """Specialized analysis."""
    from app.analysis.specialized.survey_analysis import show_survey_analysis
    from app.analysis.specialized.purchase_log import show_purchase_log_analysis

    analysis_type = st.radio(
        "分析タイプ",
        ["survey", "purchase"],
        format_func=lambda x: {"survey": "アンケート分析", "purchase": "購買ログ分析"}[x],
        horizontal=True
    )

    if analysis_type == "survey":
        show_survey_analysis(data)
    else:
        show_purchase_log_analysis(data)
