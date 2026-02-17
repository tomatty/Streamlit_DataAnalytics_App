"""
Purchase log analysis module.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta


def show_purchase_log_analysis(df: pd.DataFrame):
    """Display purchase log analysis interface."""
    st.subheader("🛒 購買ログ分析")

    analysis_type = st.radio(
        "分析タイプ",
        ["rfm", "cohort", "category"],
        format_func=lambda x: {"rfm": "RFM分析", "cohort": "コホート分析", "category": "カテゴリー分析"}[x],
        horizontal=True
    )

    if analysis_type == "rfm":
        show_rfm_analysis(df)
    elif analysis_type == "cohort":
        show_cohort_analysis(df)
    else:
        show_category_analysis(df)


def show_rfm_analysis(df: pd.DataFrame):
    """RFM (Recency, Frequency, Monetary) analysis."""
    st.markdown("### RFM分析")

    st.info("RFM分析には、顧客ID、購入日、金額の列が必要です。")

    col1, col2, col3 = st.columns(3)
    with col1:
        customer_col = st.selectbox("顧客ID列", df.columns.tolist())
    with col2:
        date_col = st.selectbox("購入日列", df.columns.tolist())
    with col3:
        amount_col = st.selectbox("金額列", df.select_dtypes(include=["number"]).columns.tolist())

    if st.button("RFM分析を実行", type="primary"):
        try:
            # Convert date column if needed
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])

            # Calculate RFM metrics
            snapshot_date = df_copy[date_col].max() + timedelta(days=1)

            rfm = df_copy.groupby(customer_col).agg({
                date_col: lambda x: (snapshot_date - x.max()).days,  # Recency
                customer_col: "count",  # Frequency
                amount_col: "sum"  # Monetary
            }).rename(columns={
                date_col: "Recency",
                customer_col: "Frequency",
                amount_col: "Monetary"
            })

            # Add RFM scores
            rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop")
            rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
            rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")

            rfm["RFM_Score"] = (
                rfm["R_Score"].astype(int) * 100 +
                rfm["F_Score"].astype(int) * 10 +
                rfm["M_Score"].astype(int)
            )

            st.success("RFM分析が完了しました！")

            # Display RFM table
            st.markdown("### RFM集計表")
            st.dataframe(rfm.head(20), use_container_width=True)

            # RFM distribution
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_r = px.histogram(rfm, x="Recency", title="Recency分布", nbins=20)
                st.plotly_chart(fig_r, use_container_width=True)
            with col2:
                fig_f = px.histogram(rfm, x="Frequency", title="Frequency分布", nbins=20)
                st.plotly_chart(fig_f, use_container_width=True)
            with col3:
                fig_m = px.histogram(rfm, x="Monetary", title="Monetary分布", nbins=20)
                st.plotly_chart(fig_m, use_container_width=True)

            # Segment customers
            rfm["Segment"] = "その他"
            rfm.loc[(rfm["R_Score"] >= 4) & (rfm["F_Score"] >= 4), "Segment"] = "優良顧客"
            rfm.loc[(rfm["R_Score"] >= 4) & (rfm["F_Score"] <= 2), "Segment"] = "新規顧客"
            rfm.loc[(rfm["R_Score"] <= 2) & (rfm["F_Score"] >= 4), "Segment"] = "離脱危険顧客"
            rfm.loc[(rfm["R_Score"] <= 2) & (rfm["F_Score"] <= 2), "Segment"] = "休眠顧客"

            st.markdown("### 顧客セグメント")
            segment_counts = rfm["Segment"].value_counts()
            fig_seg = px.pie(values=segment_counts.values, names=segment_counts.index, title="顧客セグメント分布")
            st.plotly_chart(fig_seg, use_container_width=True)

            # Download
            csv = rfm.to_csv(index=True).encode("utf-8-sig")
            st.download_button(
                label="RFM分析結果をダウンロード",
                data=csv,
                file_name="rfm_analysis.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")


def show_cohort_analysis(df: pd.DataFrame):
    """Cohort analysis."""
    st.markdown("### コホート分析")
    st.info("コホート分析には、顧客ID、購入日の列が必要です。")

    col1, col2 = st.columns(2)
    with col1:
        customer_col = st.selectbox("顧客ID列", df.columns.tolist(), key="cohort_customer")
    with col2:
        date_col = st.selectbox("購入日列", df.columns.tolist(), key="cohort_date")

    if st.button("コホート分析を実行", type="primary"):
        try:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])

            # Get first purchase date for each customer
            df_copy["CohortMonth"] = df_copy.groupby(customer_col)[date_col].transform("min").dt.to_period("M")
            df_copy["PurchaseMonth"] = df_copy[date_col].dt.to_period("M")

            # Calculate cohort index
            df_copy["CohortIndex"] = (df_copy["PurchaseMonth"] - df_copy["CohortMonth"]).apply(lambda x: x.n)

            # Create cohort table
            cohort_data = df_copy.groupby(["CohortMonth", "CohortIndex"])[customer_col].nunique().reset_index()
            cohort_pivot = cohort_data.pivot(index="CohortMonth", columns="CohortIndex", values=customer_col)

            # Calculate retention rates
            retention = cohort_pivot.div(cohort_pivot.iloc[:, 0], axis=0) * 100

            st.success("コホート分析が完了しました！")

            st.markdown("### コホート別顧客数")
            st.dataframe(cohort_pivot, use_container_width=True)

            st.markdown("### リテンション率（%）")
            st.dataframe(retention.style.background_gradient(cmap="YlGnBu"), use_container_width=True)

            # Heatmap
            fig = px.imshow(
                retention,
                labels=dict(x="月次", y="コホート", color="リテンション率(%)"),
                aspect="auto",
                color_continuous_scale="YlGnBu"
            )
            fig.update_layout(title="コホート別リテンション率")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")


def show_category_analysis(df: pd.DataFrame):
    """Category analysis."""
    st.markdown("### カテゴリー分析")

    category_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not category_cols or not numeric_cols:
        st.warning("カテゴリー列と数値列が必要です。")
        return

    col1, col2 = st.columns(2)
    with col1:
        category_col = st.selectbox("カテゴリー列", category_cols)
    with col2:
        value_col = st.selectbox("集計値列", numeric_cols)

    if st.button("カテゴリー分析を実行", type="primary"):
        try:
            # Aggregate by category
            category_stats = df.groupby(category_col)[value_col].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)

            st.success("カテゴリー分析が完了しました！")

            st.markdown("### カテゴリー別集計")
            st.dataframe(category_stats, use_container_width=True)

            # Pie chart
            fig_pie = px.pie(
                values=category_stats["sum"],
                names=category_stats.index,
                title=f"{category_col}別{value_col}の割合"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Bar chart
            fig_bar = px.bar(
                x=category_stats.index,
                y=category_stats["sum"],
                title=f"{category_col}別{value_col}",
                labels={"x": category_col, "y": value_col}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
