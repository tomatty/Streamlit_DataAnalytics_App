"""
Correspondence Analysis module.
"""
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
try:
    import prince
except ImportError:
    prince = None


def show_correspondence_analysis(df: pd.DataFrame):
    """Display Correspondence Analysis interface."""
    st.subheader("📊 コレスポンデンス分析")

    if prince is None:
        st.error("princeライブラリがインストールされていません。`pip install prince`を実行してください。")
        return

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(categorical_cols) < 2:
        st.warning("コレスポンデンス分析には少なくとも2つのカテゴリカル列が必要です。")
        return

    col1, col2 = st.columns(2)
    with col1:
        row_var = st.selectbox("行変数", categorical_cols)
    with col2:
        col_var = st.selectbox("列変数", [c for c in categorical_cols if c != row_var])

    if st.button("コレスポンデンス分析を実行", type="primary"):
        try:
            # Create contingency table
            contingency_table = pd.crosstab(df[row_var], df[col_var])

            # Perform CA
            ca = prince.CA(n_components=2)
            ca = ca.fit(contingency_table)

            st.success("コレスポンデンス分析が完了しました！")

            # Explained inertia
            st.markdown("### 説明された慣性")
            inertia_df = pd.DataFrame({
                "次元": [f"次元{i+1}" for i in range(2)],
                "固有値": ca.eigenvalues_[:2],
                "寄与率(%)": ca.explained_inertia_[:2] * 100,
                "累積寄与率(%)": ca.explained_inertia_[:2].cumsum() * 100
            })
            st.dataframe(inertia_df, use_container_width=True)

            with st.expander("📖 コレスポンデンス分析指標の解釈"):
                total_inertia_pct = ca.explained_inertia_[:2].sum() * 100
                st.markdown(
                    f"""
**慣性（Inertia）**: 分割表内の変数間の対応関係（独立性からの逸脱）の強さを表す指標です。χ²統計量をデータ全体のサンプル数で割った値に相当します。

$$\\text{{全慣性}} = \\frac{{\\chi^2}}{{n}}$$

**固有値（Eigenvalue）**: 各次元が説明する慣性の量。値が大きいほど次元の重要度が高いです。

**寄与率**: 各次元が全慣性のどの割合を説明するかを示します。

| 2次元の累積寄与率 | 判断 |
|-----------------|------|
| 80% 以上 | 2次元マップで十分な情報を表現できている |
| 60〜80% | おおむね有効だが情報損失あり |
| 60% 未満 | 重要な情報が2次元に収まっていない可能性 |

現在の2次元累積寄与率: **{total_inertia_pct:.1f}%**

**対応分析マップの読み方**:
- 同じ象限に近い点は互いに関連が強い
- 原点（0, 0）に近い点は特定の次元との関連が弱い
- 行ポイント（青）と列ポイント（赤）が近い場合、その組み合わせは強く関連している
                    """
                )

            # Plot
            st.markdown("### 対応分析マップ")
            row_coords = ca.row_coordinates(contingency_table)
            col_coords = ca.column_coordinates(contingency_table)

            fig = go.Figure()

            # Row points
            fig.add_trace(go.Scatter(
                x=row_coords[0],
                y=row_coords[1],
                mode="markers+text",
                name=row_var,
                text=row_coords.index,
                textposition="top center",
                marker=dict(size=10, color="blue")
            ))

            # Column points
            fig.add_trace(go.Scatter(
                x=col_coords[0],
                y=col_coords[1],
                mode="markers+text",
                name=col_var,
                text=col_coords.index,
                textposition="bottom center",
                marker=dict(size=10, color="red", symbol="square")
            ))

            fig.update_layout(
                title="コレスポンデンス分析マップ",
                xaxis_title=f"次元1 ({ca.explained_inertia_[0]*100:.1f}%)",
                yaxis_title=f"次元2 ({ca.explained_inertia_[1]*100:.1f}%)",
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
