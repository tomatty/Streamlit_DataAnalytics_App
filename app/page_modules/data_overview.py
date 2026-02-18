"""
Data overview page module.
Displays dataset summary, basic statistics, and data preview.
"""
import streamlit as st
from app.auth.session_manager import SessionManager
from app.components.data_preview import (
    show_basic_info,
    show_column_info,
    show_missing_values,
    show_data_preview,
)
from app.analysis.descriptive.basic_stats import (
    show_basic_statistics,
    show_categorical_statistics,
)


def show_data_overview():
    """Display data overview page."""
    st.header("📊 データ概要")

    if not SessionManager.has_data():
        st.warning(
            "データがアップロードされていません。ファイルアップロードページからデータをアップロードしてください。"
        )
        return

    data = SessionManager.get_data()

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["データプレビュー", "基本情報", "統計量", "欠損値分析"]
    )

    with tab1:
        show_data_preview(data)

    with tab2:
        show_basic_info(data)
        st.markdown("---")
        show_column_info(data)

    with tab3:
        show_basic_statistics(data)
        st.markdown("---")
        show_categorical_statistics(data)

    with tab4:
        show_missing_values(data)
