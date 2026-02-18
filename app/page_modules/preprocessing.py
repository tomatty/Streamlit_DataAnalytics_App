"""
Data preprocessing page module.
Provides UI for data cleaning and transformation.
"""
import streamlit as st
from app.auth.session_manager import SessionManager
from app.data.preprocessor import DataPreprocessor
from app.components.data_preview import show_data_preview


def show_preprocessing():
    """Display data preprocessing page."""
    st.header("🔧 データ前処理")

    if not SessionManager.has_data():
        st.warning(
            "データがアップロードされていません。ファイルアップロードページからデータをアップロードしてください。"
        )
        return

    data = SessionManager.get_data()

    # Tabs for different preprocessing operations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["欠損値処理", "重複削除", "外れ値処理", "エンコーディング", "スケーリング"]
    )

    with tab1:
        show_missing_value_handling(data)

    with tab2:
        show_duplicate_removal(data)

    with tab3:
        show_outlier_handling(data)

    with tab4:
        show_encoding(data)

    with tab5:
        show_scaling(data)

    # Show preprocessing history
    st.markdown("---")
    show_preprocessing_history()

    # Show data preview after preprocessing history
    st.markdown("---")
    show_data_preview(data, title="現在のデータプレビュー", page_size=20)


def show_missing_value_handling(data):
    """Display missing value handling interface."""
    st.subheader("欠損値処理")

    missing_cols = data.columns[data.isnull().any()].tolist()

    if not missing_cols:
        st.success("欠損値はありません！")
        return

    col1, col2 = st.columns(2)

    with col1:
        selected_cols = st.multiselect("対象列を選択", missing_cols, default=missing_cols)

    with col2:
        method = st.selectbox(
            "処理方法",
            ["drop", "fill_mean", "fill_median", "fill_mode"],
            format_func=lambda x: {
                "drop": "欠損値を含む行を削除",
                "fill_mean": "平均値で補完",
                "fill_median": "中央値で補完",
                "fill_mode": "最頻値で補完",
            }[x],
        )

    if st.button("欠損値処理を実行", type="primary"):
        try:
            processed_data = DataPreprocessor.handle_missing_values(
                data, method, selected_cols
            )
            SessionManager.set_data(processed_data, is_raw=False)
            SessionManager.add_preprocessing_step({
                "operation": "handle_missing_values",
                "method": method,
                "columns": selected_cols,
            })
            st.success("欠損値処理が完了しました！")
            st.rerun()
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")


def show_duplicate_removal(data):
    """Display duplicate removal interface."""
    st.subheader("重複削除")

    n_duplicates = data.duplicated().sum()
    st.info(f"重複行数: {n_duplicates}")

    if n_duplicates == 0:
        st.success("重複行はありません！")
        return

    subset_cols = st.multiselect(
        "重複判定に使用する列（空の場合は全列）",
        data.columns.tolist(),
    )

    if st.button("重複行を削除", type="primary"):
        try:
            processed_data = DataPreprocessor.remove_duplicates(
                data, subset=subset_cols if subset_cols else None
            )
            SessionManager.set_data(processed_data, is_raw=False)
            SessionManager.add_preprocessing_step({
                "operation": "remove_duplicates",
                "subset": subset_cols if subset_cols else "all",
            })
            st.success(f"{n_duplicates}行の重複を削除しました！")
            st.rerun()
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")


def show_outlier_handling(data):
    """Display outlier handling interface."""
    st.subheader("外れ値処理")

    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        st.warning("数値型の列が見つかりません。")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_col = st.selectbox("対象列を選択", numeric_cols)

    with col2:
        method = st.selectbox(
            "検出方法",
            ["iqr", "zscore"],
            format_func=lambda x: {
                "iqr": "IQR法",
                "zscore": "Zスコア法",
            }[x],
        )

    with col3:
        threshold = st.number_input(
            "閾値",
            min_value=0.1,
            max_value=10.0,
            value=1.5 if method == "iqr" else 3.0,
            step=0.1,
        )

    if st.button("外れ値を削除", type="primary"):
        try:
            processed_data = DataPreprocessor.handle_outliers(
                data, selected_col, method, threshold
            )
            removed_rows = len(data) - len(processed_data)
            SessionManager.set_data(processed_data, is_raw=False)
            SessionManager.add_preprocessing_step({
                "operation": "handle_outliers",
                "column": selected_col,
                "method": method,
                "threshold": threshold,
            })
            st.success(f"{removed_rows}行の外れ値を削除しました！")
            st.rerun()
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")


def show_encoding(data):
    """Display categorical encoding interface."""
    st.subheader("カテゴリカル変数のエンコーディング")

    categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        st.warning("カテゴリカル型の列が見つかりません。")
        return

    col1, col2 = st.columns(2)

    with col1:
        selected_cols = st.multiselect(
            "対象列を選択（複数選択可）",
            categorical_cols,
            default=categorical_cols[:1],
        )

    with col2:
        method = st.selectbox(
            "エンコーディング方法",
            ["label", "onehot"],
            format_func=lambda x: {
                "label": "ラベルエンコーディング",
                "onehot": "ワンホットエンコーディング",
            }[x],
        )

    if not selected_cols:
        st.info("対象列を選択してください。")
        return

    if st.button("エンコーディングを実行", type="primary"):
        try:
            processed_data = DataPreprocessor.encode_categorical(
                data, selected_cols, method
            )
            SessionManager.set_data(processed_data, is_raw=False)
            SessionManager.add_preprocessing_step({
                "operation": "encode_categorical",
                "columns": selected_cols,
                "method": method,
            })
            st.success(f"エンコーディングが完了しました！（対象列: {', '.join(selected_cols)}）")
            st.rerun()
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")


def show_scaling(data):
    """Display feature scaling interface."""
    st.subheader("特徴量のスケーリング")

    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        st.warning("数値型の列が見つかりません。")
        return

    col1, col2 = st.columns(2)

    with col1:
        selected_cols = st.multiselect("対象列を選択", numeric_cols)

    with col2:
        method = st.selectbox(
            "スケーリング方法",
            ["standard", "minmax"],
            format_func=lambda x: {
                "standard": "標準化（平均0、分散1）",
                "minmax": "正規化（0-1）",
            }[x],
        )

    if selected_cols and st.button("スケーリングを実行", type="primary"):
        try:
            processed_data = DataPreprocessor.scale_features(
                data, selected_cols, method
            )
            SessionManager.set_data(processed_data, is_raw=False)
            SessionManager.add_preprocessing_step({
                "operation": "scale_features",
                "columns": selected_cols,
                "method": method,
            })
            st.success("スケーリングが完了しました！")
            st.rerun()
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")


def show_preprocessing_history():
    """Display preprocessing history."""
    st.subheader("前処理履歴")

    steps = SessionManager.get_preprocessing_steps()

    if not steps:
        st.info("前処理履歴はありません。")
    else:
        for i, step in enumerate(steps, 1):
            st.text(f"{i}. {step}")

        if st.button("元のデータに戻す"):
            raw_data = SessionManager.get_raw_data()
            SessionManager.set_data(raw_data, is_raw=False)
            SessionManager.clear_preprocessing_history()
            st.success("元のデータに戻しました！")
            st.rerun()
