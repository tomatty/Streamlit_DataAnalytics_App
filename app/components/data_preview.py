"""
Data preview component.
Displays paginated data tables with filtering.
"""
import streamlit as st
import pandas as pd


def show_data_preview(df: pd.DataFrame, title="データプレビュー", page_size=20):
    """
    Display paginated data preview.

    Args:
        df: DataFrame to display
        title: Title for the preview section
        page_size: Number of rows per page
    """
    st.subheader(title)

    # Calculate total pages
    total_rows = len(df)
    total_pages = (total_rows - 1) // page_size + 1

    # Page selection
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        page = st.number_input(
            f"ページ (全{total_pages}ページ)",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
        )

    # Calculate start and end indices
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)

    # Display page info
    st.caption(f"表示: {start_idx + 1} - {end_idx} 行 / 全 {total_rows} 行")

    # Display data
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)


def show_basic_info(df: pd.DataFrame):
    """
    Display basic DataFrame information.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("基本情報")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("行数", f"{len(df):,}")

    with col2:
        st.metric("列数", f"{len(df.columns):,}")

    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("メモリ使用量", f"{memory_mb:.2f} MB")

    with col4:
        missing_total = df.isnull().sum().sum()
        st.metric("欠損値", f"{missing_total:,}")


def show_column_info(df: pd.DataFrame):
    """
    Display column information.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("列情報")

    # Create column info DataFrame
    col_info = pd.DataFrame({
        "列名": df.columns,
        "データ型": df.dtypes.values,
        "欠損値数": df.isnull().sum().values,
        "欠損率(%)": (df.isnull().sum() / len(df) * 100).values,
        "ユニーク数": [df[col].nunique() for col in df.columns],
    })

    st.dataframe(col_info, use_container_width=True)


def show_missing_values(df: pd.DataFrame):
    """
    Display missing values analysis.

    Args:
        df: DataFrame to analyze
    """
    st.subheader("欠損値分析")

    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

    if len(missing_data) == 0:
        st.success("欠損値はありません！")
    else:
        missing_df = pd.DataFrame({
            "列名": missing_data.index,
            "欠損値数": missing_data.values,
            "欠損率(%)": (missing_data / len(df) * 100).values,
        })

        st.dataframe(missing_df, use_container_width=True)

        # Visualization
        import plotly.express as px

        fig = px.bar(
            missing_df,
            x="列名",
            y="欠損率(%)",
            title="列別欠損率",
            labels={"欠損率(%)": "欠損率 (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)
