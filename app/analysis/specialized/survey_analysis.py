"""
Survey analysis module.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def show_survey_analysis(df: pd.DataFrame):
    """Display survey analysis interface."""
    st.subheader("📋 アンケート分析")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    analysis_type = st.radio(
        "分析タイプ",
        ["likert", "nps"],
        format_func=lambda x: {"likert": "リッカート尺度分析", "nps": "NPS分析"}[x],
        horizontal=True
    )

    if analysis_type == "likert":
        show_likert_analysis(df, numeric_cols)
    else:
        show_nps_analysis(df, numeric_cols)


def show_likert_analysis(df: pd.DataFrame, numeric_cols: list):
    """Likert scale analysis."""
    st.markdown("### リッカート尺度分析")

    likert_cols = st.multiselect(
        "リッカート尺度の質問項目を選択",
        numeric_cols,
        help="1-5または1-7のスケールで評価された質問項目"
    )

    if not likert_cols:
        st.info("質問項目を選択してください。")
        return

    if st.button("リッカート尺度分析を実行", type="primary"):
        try:
            # Calculate statistics
            stats_df = df[likert_cols].describe().T
            stats_df["mode"] = df[likert_cols].mode().iloc[0]

            st.markdown("### 質問項目別統計")
            st.dataframe(stats_df, use_container_width=True)

            # Distribution for each item
            st.markdown("### 回答分布")
            for col in likert_cols:
                value_counts = df[col].value_counts().sort_index()

                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"{col} の回答分布",
                    labels={"x": "評価", "y": "回答数"}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Stacked bar chart
            st.markdown("### 質問項目別回答割合")
            proportion_data = []
            for col in likert_cols:
                value_counts = df[col].value_counts(normalize=True).sort_index() * 100
                for value, pct in value_counts.items():
                    proportion_data.append({"質問": col, "評価": value, "割合(%)": pct})

            prop_df = pd.DataFrame(proportion_data)
            fig = px.bar(
                prop_df,
                x="質問",
                y="割合(%)",
                color="評価",
                title="質問項目別回答割合",
                barmode="stack"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.success("リッカート尺度分析が完了しました！")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")


def show_nps_analysis(df: pd.DataFrame, numeric_cols: list):
    """Net Promoter Score (NPS) analysis."""
    st.markdown("### NPS（ネットプロモータースコア）分析")

    nps_col = st.selectbox(
        "NPS質問項目を選択（0-10のスケール）",
        numeric_cols
    )

    if st.button("NPS分析を実行", type="primary"):
        try:
            nps_scores = df[nps_col].dropna()

            # Categorize scores
            promoters = (nps_scores >= 9).sum()
            passives = ((nps_scores >= 7) & (nps_scores <= 8)).sum()
            detractors = (nps_scores <= 6).sum()
            total = len(nps_scores)

            # Calculate NPS
            nps = ((promoters - detractors) / total) * 100

            st.markdown("### NPS計算結果")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                with st.container(border=True):
                    st.metric("NPS", f"{nps:.1f}")
            with col2:
                with st.container(border=True):
                    st.metric("推奨者", f"{promoters} ({promoters/total*100:.1f}%)")
            with col3:
                with st.container(border=True):
                    st.metric("中立者", f"{passives} ({passives/total*100:.1f}%)")
            with col4:
                with st.container(border=True):
                    st.metric("批判者", f"{detractors} ({detractors/total*100:.1f}%)")

            # NPS interpretation
            if nps > 50:
                st.success("NPSが50以上: 優秀")
            elif nps > 0:
                st.info("NPSが0以上: 良好")
            else:
                st.warning("NPSが0未満: 改善が必要")

            # Distribution
            st.markdown("### スコア分布")
            value_counts = nps_scores.value_counts().sort_index()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title="NPSスコア分布",
                labels={"x": "スコア", "y": "回答数"}
            )

            # Add category colors
            colors = []
            for score in value_counts.index:
                if score >= 9:
                    colors.append("green")
                elif score >= 7:
                    colors.append("yellow")
                else:
                    colors.append("red")

            fig.update_traces(marker_color=colors)
            st.plotly_chart(fig, use_container_width=True)

            st.success("NPS分析が完了しました！")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
