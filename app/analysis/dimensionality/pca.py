"""
Principal Component Analysis (PCA) module.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def show_pca_analysis(df: pd.DataFrame):
    """Display PCA analysis interface."""
    st.subheader("📊 主成分分析（PCA）")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("PCAには少なくとも2つの数値型列が必要です。")
        return

    selected_cols = st.multiselect("分析対象列を選択", numeric_cols, default=numeric_cols[:min(10, len(numeric_cols))])

    if len(selected_cols) < 2:
        st.info("少なくとも2つの列を選択してください。")
        return

    n_components = st.slider("主成分数", min_value=2, max_value=min(len(selected_cols), 10), value=min(3, len(selected_cols)))

    if st.button("PCAを実行", type="primary"):
        try:
            # Prepare data
            data_subset = df[selected_cols].dropna()

            if len(data_subset) < 3:
                st.error("有効なデータが不足しています。")
                return

            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_subset)

            # Perform PCA
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(data_scaled)

            # Create DataFrame with principal components
            pc_cols = [f"PC{i+1}" for i in range(n_components)]
            pc_df = pd.DataFrame(data=principal_components, columns=pc_cols)

            st.success("PCAが完了しました！")

            # Explained variance
            st.markdown("### 説明された分散")
            explained_var = pd.DataFrame({
                "主成分": pc_cols,
                "寄与率(%)": pca.explained_variance_ratio_ * 100,
                "累積寄与率(%)": np.cumsum(pca.explained_variance_ratio_) * 100,
            })
            st.dataframe(explained_var, use_container_width=True)

            with st.expander("📖 PCA指標の解釈"):
                cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
                st.markdown(
                    f"""
**寄与率（Explained Variance Ratio）**: 各主成分が元データの全変動をどの割合だけ説明するかを示します。

**累積寄与率**: 第1〜第k主成分までの寄与率の合計。主成分数の選択基準として使用します。

| 累積寄与率の目安 | 判断 |
|----------------|------|
| 80% 以上 | 十分な情報を保持 |
| 70〜80% | 概ね許容範囲 |
| 70% 未満 | 情報の損失が大きい可能性 |

現在の累積寄与率: {', '.join([f'PC1〜{i+1}={v:.1f}%' for i, v in enumerate(cumulative)])}

**因子負荷量（Loadings）**: 各主成分と元変数の相関係数。絶対値が大きいほど（目安: 0.4以上）その変数が主成分の解釈に重要です。

**固有値基準（Kaiser基準）**: 固有値 > 1 の主成分のみ採用するという選択法もあります。スクリープロットの「折れ曲がり点（elbow）」も参考にしてください。
                    """
                )

            # Scree plot
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(
                x=pc_cols,
                y=pca.explained_variance_ratio_ * 100,
                name="寄与率"
            ))
            fig_scree.add_trace(go.Scatter(
                x=pc_cols,
                y=np.cumsum(pca.explained_variance_ratio_) * 100,
                mode="lines+markers",
                name="累積寄与率"
            ))
            fig_scree.update_layout(
                title="スクリープロット",
                xaxis_title="主成分",
                yaxis_title="寄与率 (%)"
            )
            st.plotly_chart(fig_scree, use_container_width=True)

            # Loadings
            st.markdown("### 因子負荷量")
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=pc_cols,
                index=selected_cols
            )
            st.dataframe(loadings.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1), use_container_width=True)

            # Biplot
            if n_components >= 2:
                st.markdown("### バイプロット（PC1 vs PC2）")
                fig_biplot = go.Figure()

                # Scatter plot
                fig_biplot.add_trace(go.Scatter(
                    x=pc_df["PC1"],
                    y=pc_df["PC2"],
                    mode="markers",
                    name="データ",
                    marker=dict(size=5, opacity=0.6)
                ))

                # Loading vectors
                for i, var in enumerate(selected_cols):
                    fig_biplot.add_trace(go.Scatter(
                        x=[0, pca.components_[0, i] * 3],
                        y=[0, pca.components_[1, i] * 3],
                        mode="lines+text",
                        name=var,
                        line=dict(color="red"),
                        text=["", var],
                        textposition="top center"
                    ))

                fig_biplot.update_layout(
                    title="バイプロット",
                    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                    showlegend=False
                )
                st.plotly_chart(fig_biplot, use_container_width=True)

            # Download principal components
            csv = pc_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="主成分スコアをダウンロード",
                data=csv,
                file_name="pca_scores.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
