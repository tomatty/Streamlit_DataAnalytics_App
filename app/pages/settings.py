"""
Settings page module.
Allows users to configure analysis settings.
"""
import streamlit as st
from app.auth.session_manager import SessionManager


def show_settings():
    """Display settings page with configuration options."""
    st.header("⚙️ 分析設定")
    st.markdown("分析に使用するデフォルト設定を変更できます。")

    # Get current settings
    current_settings = SessionManager.get_all_settings()

    st.markdown("---")

    # Hypothesis Testing Settings
    st.subheader("📊 仮説検定設定")

    col1, col2 = st.columns(2)

    with col1:
        confidence_level = st.slider(
            "信頼度 (Confidence Level)",
            min_value=0.80,
            max_value=0.99,
            value=float(current_settings.get("confidence_level", 0.95)),
            step=0.01,
            help="仮説検定で使用する信頼度を設定します",
        )

    with col2:
        significance_level = st.slider(
            "有意水準 (Significance Level)",
            min_value=0.01,
            max_value=0.20,
            value=float(current_settings.get("significance_level", 0.05)),
            step=0.01,
            help="仮説検定で使用する有意水準（α）を設定します",
        )

    st.markdown("---")

    # Clustering Settings
    st.subheader("🔵 クラスタリング設定")

    max_clustering_iterations = st.number_input(
        "最大イテレーション数",
        min_value=100,
        max_value=1000,
        value=int(current_settings.get("max_clustering_iterations", 300)),
        step=50,
        help="K-Meansクラスタリングの最大イテレーション数を設定します",
    )

    st.markdown("---")

    # Text Analysis Settings
    st.subheader("📝 テキスト分析設定")

    col3, col4 = st.columns(2)

    with col3:
        max_features = st.number_input(
            "最大特徴数 (Max Features)",
            min_value=10,
            max_value=500,
            value=int(current_settings.get("max_features", 100)),
            step=10,
            help="テキスト分析で抽出する最大単語数を設定します",
        )

    with col4:
        n_topics = st.number_input(
            "トピック数 (Number of Topics)",
            min_value=2,
            max_value=20,
            value=int(current_settings.get("n_topics", 5)),
            step=1,
            help="トピックモデリングで抽出するトピック数を設定します",
        )

    st.markdown("---")

    # Save button
    col_save, col_reset = st.columns([1, 1])

    with col_save:
        if st.button("💾 設定を保存", use_container_width=True, type="primary"):
            # Update settings
            new_settings = {
                "confidence_level": confidence_level,
                "significance_level": significance_level,
                "max_clustering_iterations": max_clustering_iterations,
                "max_features": max_features,
                "n_topics": n_topics,
            }
            SessionManager.update_all_settings(new_settings)
            st.success("✅ 設定を保存しました")
            st.rerun()

    with col_reset:
        if st.button("🔄 デフォルトに戻す", use_container_width=True):
            # Reset to default values from config
            from app.config import config

            default_settings = {
                "confidence_level": config.analysis.default_confidence_level,
                "significance_level": config.analysis.default_significance_level,
                "max_clustering_iterations": config.analysis.max_clustering_iterations,
                "max_features": config.analysis.default_max_features,
                "n_topics": config.analysis.default_n_topics,
            }
            SessionManager.update_all_settings(default_settings)
            st.success("✅ デフォルト設定に戻しました")
            st.rerun()

    # Display current settings summary
    st.markdown("---")
    st.subheader("📋 現在の設定")

    settings_summary = f"""
    **仮説検定:**
    - 信頼度: {confidence_level:.2%}
    - 有意水準: {significance_level:.2%}

    **クラスタリング:**
    - 最大イテレーション数: {max_clustering_iterations}

    **テキスト分析:**
    - 最大特徴数: {max_features}
    - トピック数: {n_topics}
    """

    st.info(settings_summary)
