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
        ["rfm", "cohort", "category", "pareto"],
        format_func=lambda x: {
            "rfm": "RFM分析",
            "cohort": "コホート分析",
            "category": "カテゴリー分析",
            "pareto": "パレート分析",
        }[x],
        horizontal=True
    )

    if analysis_type == "rfm":
        show_rfm_analysis(df)
    elif analysis_type == "cohort":
        show_cohort_analysis(df)
    elif analysis_type == "category":
        show_category_analysis(df)
    else:
        show_pareto_analysis(df)


def show_rfm_analysis(df: pd.DataFrame):
    """RFM (Recency, Frequency, Monetary) analysis."""
    st.markdown("### RFM分析")

    st.info("RFM分析には、顧客ID、購入日、金額の列が必要です。")

    with st.expander("📐 RFMスコアの計算式定義"):
        st.markdown(
            """
**RFM分析**は、顧客を3つの指標で評価する手法です。

| 指標 | 定義 | スコア計算 |
|------|------|------------|
| **R (Recency)** | 最終購入日からの経過日数<br>`snapshot_date - 最終購入日` | 経過日数が**少ない**ほど高スコア（5段階: 5=直近、1=古い） |
| **F (Frequency)** | 購入回数<br>`顧客ごとの購入件数合計` | 購入回数が**多い**ほど高スコア（5段階: 1=少ない、5=多い） |
| **M (Monetary)** | 購入金額合計<br>`顧客ごとの金額合計` | 金額が**大きい**ほど高スコア（5段階: 1=低額、5=高額） |

**総合RFMスコアの計算式:**

$$\\text{RFM\\_Score} = R\\_Score \\times 100 + F\\_Score \\times 10 + M\\_Score$$

各スコアは五分位数（quantile）で1〜5に分類されます。
- スコア範囲: 111（最低）〜 555（最高）
            """
        )

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


def show_pareto_analysis(df: pd.DataFrame):
    """Pareto analysis (80/20 rule) for purchase log data."""
    st.markdown("### パレート分析（ABC分析）")

    st.info(
        "パレート分析は「売上の80%は上位20%の商品・顧客が生み出す」という法則に基づき、"
        "重要度でアイテムをA（上位0〜80%）、B（80〜95%）、C（95〜100%）に分類します。"
    )

    category_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not category_cols or not numeric_cols:
        st.warning("パレート分析にはカテゴリー列と数値列が必要です。")
        return

    col1, col2 = st.columns(2)
    with col1:
        item_col = st.selectbox("分析対象列（商品・顧客など）", category_cols, key="pareto_item")
    with col2:
        value_col = st.selectbox("集計値列（売上・件数など）", numeric_cols, key="pareto_value")

    col_a, col_b, _ = st.columns(3)
    with col_a:
        threshold_a = st.number_input("Aランク閾値（累積%）", min_value=50, max_value=90, value=80, step=5)
    with col_b:
        threshold_b = st.number_input("Bランク閾値（累積%）", min_value=threshold_a + 1, max_value=99, value=95, step=5)

    if st.button("パレート分析を実行", type="primary", key="pareto_run"):
        try:
            import plotly.graph_objects as go

            # Aggregate by item
            pareto_df = (
                df.groupby(item_col)[value_col]
                .sum()
                .reset_index()
                .sort_values(value_col, ascending=False)
                .reset_index(drop=True)
            )
            pareto_df.columns = [item_col, "合計値"]

            # Calculate cumulative percentage
            total = pareto_df["合計値"].sum()
            pareto_df["構成比(%)"] = pareto_df["合計値"] / total * 100
            pareto_df["累積構成比(%)"] = pareto_df["構成比(%)"].cumsum()

            # Assign rank
            pareto_df["ランク"] = pareto_df["累積構成比(%)"].apply(
                lambda x: "A" if x <= threshold_a else ("B" if x <= threshold_b else "C")
            )

            st.success("パレート分析が完了しました！")

            # Summary metrics
            rank_summary = pareto_df.groupby("ランク").agg(
                件数=("合計値", "count"),
                合計値=("合計値", "sum"),
                構成比=("構成比(%)", "sum"),
            ).reindex(["A", "B", "C"])
            rank_summary["件数割合(%)"] = rank_summary["件数"] / len(pareto_df) * 100

            col1, col2, col3 = st.columns(3)
            for col_widget, rank, icon in zip(
                [col1, col2, col3], ["A", "B", "C"], ["🟢", "🟡", "🔴"]
            ):
                with col_widget:
                    with st.container(border=True):
                        row = rank_summary.loc[rank]
                        st.metric(
                            f"{icon} ランク{rank}",
                            f"{int(row['件数'])}件 ({row['件数割合(%)']:.1f}%)",
                            f"売上構成比 {row['構成比']:.1f}%",
                        )

            # Pareto chart (bar + cumulative line)
            st.markdown("### パレート図")
            fig = go.Figure()

            # Bar chart for individual values
            fig.add_trace(
                go.Bar(
                    x=pareto_df[item_col],
                    y=pareto_df["合計値"],
                    name="合計値",
                    marker_color=[
                        "#2196F3" if r == "A" else ("#FF9800" if r == "B" else "#F44336")
                        for r in pareto_df["ランク"]
                    ],
                    yaxis="y1",
                )
            )

            # Cumulative line chart
            fig.add_trace(
                go.Scatter(
                    x=pareto_df[item_col],
                    y=pareto_df["累積構成比(%)"],
                    name="累積構成比(%)",
                    mode="lines+markers",
                    line=dict(color="black", width=2),
                    marker=dict(size=4),
                    yaxis="y2",
                )
            )

            # Threshold lines on secondary axis
            for threshold, label, color in [
                (threshold_a, f"A/B境界 {threshold_a}%", "blue"),
                (threshold_b, f"B/C境界 {threshold_b}%", "orange"),
            ]:
                fig.add_hline(
                    y=threshold,
                    line_dash="dot",
                    line_color=color,
                    annotation_text=label,
                    annotation_position="right",
                    yref="y2",
                )

            fig.update_layout(
                title="パレート図",
                xaxis_title=item_col,
                yaxis=dict(title="合計値", side="left"),
                yaxis2=dict(
                    title="累積構成比(%)",
                    side="right",
                    overlaying="y",
                    range=[0, 105],
                    ticksuffix="%",
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                bargap=0.2,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detail table with rank coloring
            st.markdown("### パレート分析詳細")

            def highlight_rank(row):
                color_map = {"A": "background-color: #c8e6c9", "B": "background-color: #fff9c4", "C": "background-color: #ffcdd2"}
                return [color_map.get(row["ランク"], "")] * len(row)

            st.dataframe(
                pareto_df.style.apply(highlight_rank, axis=1),
                use_container_width=True,
            )

            # Download
            csv = pareto_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="パレート分析結果をダウンロード",
                data=csv,
                file_name="pareto_analysis.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
