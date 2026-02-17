"""
File upload component.
Handles file uploads and sample data selection.
"""
import streamlit as st
from app.data.loader import DataLoader
from app.data.validator import DataValidator
from app.auth.session_manager import SessionManager
from app.config import config
from app.constants import (
    SESSION_UPLOADED_FILE_NAME,
    SESSION_DATA_METADATA,
    MSG_FILE_UPLOAD_SUCCESS,
    MSG_FILE_UPLOAD_ERROR,
    SAMPLE_SURVEY,
    SAMPLE_PURCHASE_LOG,
    SAMPLE_TEXT_DATA,
)


def show_file_upload():
    """Display file upload interface with sample data option."""
    st.header("📁 ファイルアップロード")

    # Upload method selection
    upload_method = st.radio(
        "データソースを選択",
        ["ファイルをアップロード", "サンプルデータを使用"],
        horizontal=True,
    )

    if upload_method == "ファイルをアップロード":
        show_file_upload_interface()
    else:
        show_sample_data_interface()

    # Display current data info if available
    if SessionManager.has_data():
        st.markdown("---")
        st.success("✅ データが読み込まれています")
        data = SessionManager.get_data()
        file_name = st.session_state.get(SESSION_UPLOADED_FILE_NAME, "Unknown")
        st.info(f"📄 **ファイル名:** {file_name}")
        st.info(f"📊 **サイズ:** {data.shape[0]} 行 × {data.shape[1]} 列")


def show_file_upload_interface():
    """Display file upload interface."""
    st.subheader("ファイルをアップロード")

    uploaded_file = st.file_uploader(
        "CSVまたはExcelファイルを選択してください",
        type=["csv", "xlsx", "xls", "json"],
        help="対応形式: CSV, Excel (xlsx/xls), JSON",
    )

    if uploaded_file is not None:
        # Validate file type
        if not DataValidator.validate_file_type(uploaded_file.name):
            st.error(f"{MSG_FILE_UPLOAD_ERROR}: サポートされていないファイル形式です")
            return

        # Validate file size
        if not DataValidator.validate_file_size(uploaded_file.size):
            st.error(
                f"{MSG_FILE_UPLOAD_ERROR}: ファイルサイズが大きすぎます "
                f"(最大: {config.app.max_upload_size_mb}MB)"
            )
            return

        try:
            # Load data
            with st.spinner("データを読み込んでいます..."):
                df = DataLoader.load_file(uploaded_file, uploaded_file.name)

                # Validate DataFrame
                is_valid, error_msg = DataValidator.validate_dataframe(df)
                if not is_valid:
                    st.error(f"{MSG_FILE_UPLOAD_ERROR}: {error_msg}")
                    return

                # Store data in session
                SessionManager.set_data(df, is_raw=True)
                st.session_state[SESSION_UPLOADED_FILE_NAME] = uploaded_file.name

                # Get and store metadata
                metadata = DataLoader.get_data_metadata(df)
                st.session_state[SESSION_DATA_METADATA] = metadata

                st.success(MSG_FILE_UPLOAD_SUCCESS)
                st.rerun()

        except Exception as e:
            st.error(f"{MSG_FILE_UPLOAD_ERROR}: {str(e)}")


def show_sample_data_interface():
    """Display sample data selection interface."""
    st.subheader("サンプルデータを使用")

    sample_datasets = {
        "アンケートデータ": SAMPLE_SURVEY,
        "購買ログデータ": SAMPLE_PURCHASE_LOG,
        "テキストデータ": SAMPLE_TEXT_DATA,
    }

    selected_sample = st.selectbox(
        "サンプルデータセットを選択",
        list(sample_datasets.keys()),
        help="分析機能を試すためのサンプルデータセットです",
    )

    if st.button("サンプルデータを読み込む", type="primary"):
        try:
            sample_file = sample_datasets[selected_sample]

            with st.spinner(f"{selected_sample}を読み込んでいます..."):
                df = DataLoader.load_sample_data(sample_file)

                # Validate DataFrame
                is_valid, error_msg = DataValidator.validate_dataframe(df)
                if not is_valid:
                    st.error(f"{MSG_FILE_UPLOAD_ERROR}: {error_msg}")
                    return

                # Store data in session
                SessionManager.set_data(df, is_raw=True)
                st.session_state[SESSION_UPLOADED_FILE_NAME] = f"{selected_sample} (サンプル)"

                # Get and store metadata
                metadata = DataLoader.get_data_metadata(df)
                st.session_state[SESSION_DATA_METADATA] = metadata

                st.success(f"✅ {selected_sample}を読み込みました")
                st.rerun()

        except Exception as e:
            st.error(f"{MSG_FILE_UPLOAD_ERROR}: {str(e)}")


def show_data_clear_option():
    """Display option to clear loaded data."""
    if SessionManager.has_data():
        if st.button("🗑️ データをクリア", help="読み込んだデータをクリアします"):
            SessionManager.clear_data()
            st.success("データをクリアしました")
            st.rerun()
