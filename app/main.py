"""
Main application entry point.
Handles authentication and routing.
"""
import streamlit as st
from app.auth.authenticator import Authenticator
from app.auth.session_manager import SessionManager
from app.config import config
from app.constants import MSG_LOGIN_SUCCESS, MSG_LOGIN_FAILED


def show_login_page():
    """Display the login page."""
    st.set_page_config(
        page_title=f"{config.app.name} - Login",
        page_icon="🔐",
        layout="centered",
    )

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.title("🔐 ログイン")
        st.markdown("---")

        # Login form
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

        st.markdown("---")
        st.caption(f"📊 {config.app.name}")


def show_main_app():
    """Display the main application after authentication."""
    # Check for session timeout
    if Authenticator.check_session_timeout():
        st.warning("セッションがタイムアウトしました。再度ログインしてください。")
        st.rerun()

    st.set_page_config(
        page_title=config.app.name,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar
    with st.sidebar:
        st.title(f"📊 {config.app.name}")
        st.markdown(f"**ユーザー:** {Authenticator.get_current_user()}")
        st.markdown("---")

        # Main navigation menu
        menu_option = st.radio(
            "メニュー",
            [
                "📁 ファイルアップロード",
                "📊 データ概要",
                "🔧 データ前処理",
                "📈 分析",
                "💾 エクスポート",
                "⚙️ 分析設定",
            ],
            index=0,
        )

        st.markdown("---")

        # Logout button
        if st.button("🚪 ログアウト", use_container_width=True):
            Authenticator.logout()
            st.rerun()

    # Main content area
    if menu_option == "📁 ファイルアップロード":
        show_file_upload_page()
    elif menu_option == "📊 データ概要":
        show_data_overview_page()
    elif menu_option == "🔧 データ前処理":
        show_preprocessing_page()
    elif menu_option == "📈 分析":
        show_analysis_page()
    elif menu_option == "💾 エクスポート":
        show_export_page()
    elif menu_option == "⚙️ 分析設定":
        show_settings_page()


def show_file_upload_page():
    """Display file upload page."""
    from app.components.file_upload import show_file_upload, show_data_clear_option

    show_file_upload()
    st.markdown("---")
    show_data_clear_option()


def show_data_overview_page():
    """Display data overview page."""
    from app.pages.data_overview import show_data_overview

    show_data_overview()


def show_preprocessing_page():
    """Display preprocessing page."""
    from app.pages.preprocessing import show_preprocessing

    show_preprocessing()


def show_analysis_page():
    """Display analysis page."""
    from app.pages.analysis import show_analysis

    show_analysis()


def show_export_page():
    """Display export page."""
    from app.utils.export import show_export_page as show_export

    data = SessionManager.get_data()
    show_export(data)


def show_settings_page():
    """Display settings page."""
    from app.pages.settings import show_settings

    show_settings()


def main():
    """Main application function."""
    # Initialize session state
    SessionManager.init_session_state()

    # Check authentication
    if not Authenticator.is_authenticated():
        show_login_page()
    else:
        show_main_app()


if __name__ == "__main__":
    main()
