"""
Main application entry point.
Handles authentication and routing.
"""
import os
import sys

# Ensure the project root is on sys.path so that `app.*` imports work when
# Streamlit Cloud executes this file directly (it adds app/ to sys.path, not
# the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.auth.authenticator import Authenticator
from app.auth.session_manager import SessionManager
from app.config import config
from app.constants import MSG_LOGIN_SUCCESS, MSG_LOGIN_FAILED


def show_login_page():
    """Display the login page."""
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        st.title("🔐 ログイン")
        st.markdown("---")

        with st.form("login_form"):
            username = st.text_input("ユーザー名", placeholder="ユーザー名を入力")
            password = st.text_input(
                "パスワード", type="password", placeholder="パスワードを入力"
            )
            submit = st.form_submit_button("ログイン", use_container_width=True)

            if submit:
                if Authenticator.login(username, password):
                    st.success(MSG_LOGIN_SUCCESS)
                    st.rerun()
                else:
                    st.error(MSG_LOGIN_FAILED)
        st.write("ユーザー名: freeuser")
        st.write("パスワード: freeuser01")
        st.markdown("---")
        st.caption(f"📊 {config.app.name}")


def show_main_app():
    """Display the main application after authentication."""
    # Check for session timeout
    if Authenticator.check_session_timeout():
        st.warning("セッションがタイムアウトしました。再度ログインしてください。")
        st.rerun()

    # Sidebar
    with st.sidebar:
        st.title(f"📊 {config.app.name}")
        st.markdown(f"**ユーザー:** {Authenticator.get_current_user()}")
        st.markdown("---")

        menu_option = st.radio(
            "メニュー",
            [
                "🔢 サンプルサイズ計算",
                "📁 ファイルアップロード",
                "📊 データ概要",
                "🔧 データ前処理",
                "📈 分析",
                "💾 エクスポート",
            ],
            index=0,
        )

        analysis_category = None
        if menu_option == "📈 分析":
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
                ],
                key="sidebar_analysis_category",
            )

        st.markdown("---")

        if st.button("🚪 ログアウト", use_container_width=True):
            Authenticator.logout()
            st.rerun()

    if menu_option == "🔢 サンプルサイズ計算":
        show_sample_size_calculator_page()
    elif menu_option == "📁 ファイルアップロード":
        show_file_upload_page()
    elif menu_option == "📊 データ概要":
        show_data_overview_page()
    elif menu_option == "🔧 データ前処理":
        show_preprocessing_page()
    elif menu_option == "📈 分析":
        show_analysis_page(analysis_category)
    elif menu_option == "💾 エクスポート":
        show_export_page()


def show_sample_size_calculator_page():
    """Display sample size calculator page."""
    from app.page_modules.sample_size_calculator import show_sample_size_calculator

    show_sample_size_calculator()


def show_file_upload_page():
    """Display file upload page."""
    from app.components.file_upload import show_file_upload, show_data_clear_option

    show_file_upload()
    st.markdown("---")
    show_data_clear_option()


def show_data_overview_page():
    """Display data overview page."""
    from app.page_modules.data_overview import show_data_overview

    show_data_overview()


def show_preprocessing_page():
    """Display preprocessing page."""
    from app.page_modules.preprocessing import show_preprocessing

    show_preprocessing()


def show_analysis_page(analysis_category: str | None = None):
    """Display analysis page."""
    from app.page_modules.analysis import show_analysis

    show_analysis(analysis_category)


def show_export_page():
    """Display export page."""
    from app.utils.export import show_export_page as show_export

    data = SessionManager.get_data()
    show_export(data)



def main():
    """Main application function."""
    st.set_page_config(
        page_title=config.app.name,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # Restore session from URL query parameter (?s=TOKEN).
    # st.query_params is populated from the HTTP request URL on every page load,
    # so this works reliably across reloads without any JavaScript timing concerns.
    Authenticator.restore_from_url()

    # Initialize remaining session state defaults
    SessionManager.init_session_state()

    if not Authenticator.is_authenticated():
        show_login_page()
    else:
        show_main_app()


if __name__ == "__main__":
    main()
