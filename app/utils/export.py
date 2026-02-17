"""
Export utilities module.
"""
import pandas as pd
import streamlit as st
from io import BytesIO


def export_to_csv(df: pd.DataFrame, filename: str = "data.csv") -> bytes:
    """
    Export DataFrame to CSV.

    Args:
        df: DataFrame to export
        filename: Output filename

    Returns:
        bytes: CSV data
    """
    return df.to_csv(index=False).encode("utf-8-sig")


def export_to_excel(df: pd.DataFrame, filename: str = "data.xlsx") -> bytes:
    """
    Export DataFrame to Excel.

    Args:
        df: DataFrame to export
        filename: Output filename

    Returns:
        bytes: Excel data
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return output.getvalue()


def show_export_page(df: pd.DataFrame):
    """Display export interface."""
    st.header("💾 エクスポート")

    if df is None:
        st.warning("エクスポートするデータがありません。")
        return

    st.info(f"データサイズ: {len(df)} 行 × {len(df.columns)} 列")

    # Export options
    export_format = st.radio(
        "エクスポート形式",
        ["csv", "excel"],
        format_func=lambda x: {"csv": "CSV", "excel": "Excel (xlsx)"}[x],
        horizontal=True
    )

    filename = st.text_input(
        "ファイル名",
        value=f"exported_data.{export_format if export_format == 'csv' else 'xlsx'}"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("現在のデータをエクスポート", type="primary"):
            try:
                if export_format == "csv":
                    data = export_to_csv(df, filename)
                    mime_type = "text/csv"
                else:
                    data = export_to_excel(df, filename)
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                st.download_button(
                    label=f"📥 {filename} をダウンロード",
                    data=data,
                    file_name=filename,
                    mime=mime_type
                )
                st.success("エクスポートの準備ができました！")
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")

    with col2:
        # Column selection
        if st.checkbox("列を選択してエクスポート"):
            selected_cols = st.multiselect("エクスポートする列を選択", df.columns.tolist(), default=df.columns.tolist())

            if selected_cols and st.button("選択列をエクスポート"):
                try:
                    df_selected = df[selected_cols]

                    if export_format == "csv":
                        data = export_to_csv(df_selected, filename)
                        mime_type = "text/csv"
                    else:
                        data = export_to_excel(df_selected, filename)
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                    st.download_button(
                        label=f"📥 {filename} をダウンロード（選択列のみ）",
                        data=data,
                        file_name=filename,
                        mime=mime_type,
                        key="export_selected"
                    )
                    st.success("エクスポートの準備ができました！")
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")

    # Data preview
    st.markdown("---")
    st.markdown("### エクスポート対象データのプレビュー")
    st.dataframe(df.head(20), use_container_width=True)
