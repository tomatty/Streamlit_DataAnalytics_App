"""
LightGBM decision tree analysis module.
Supports both classification and regression tasks.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error,
)
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def show_lightgbm_tree(df: pd.DataFrame):
    """Display LightGBM analysis interface."""
    st.subheader("⚡ LightGBM 決定木分析")

    if lgb is None:
        st.error("lightgbmがインストールされていません。`pip install lightgbm`を実行してください。")
        return

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(all_cols) < 2:
        st.warning("LightGBM分析には少なくとも2つの列が必要です。")
        return

    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("目的変数（ターゲット）", all_cols, key="lgb_target")
    with col2:
        task_type = st.radio(
            "タスク種別",
            ["分類（Classification）", "回帰（Regression）"],
            key="lgb_task",
            horizontal=True,
        )

    feature_cols = st.multiselect(
        "説明変数（特徴量）",
        [c for c in numeric_cols if c != target_col],
        default=[c for c in numeric_cols if c != target_col][:min(5, len(numeric_cols))],
        key="lgb_features",
    )

    if len(feature_cols) < 1:
        st.info("少なくとも1つの説明変数を選択してください。")
        return

    st.markdown("#### ハイパーパラメータ設定")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        n_estimators = st.slider("木の本数（n_estimators）", 50, 500, 100, step=50, key="lgb_n")
    with p_col2:
        learning_rate = st.select_slider(
            "学習率（learning_rate）",
            options=[0.01, 0.05, 0.1, 0.2, 0.3],
            value=0.1,
            key="lgb_lr",
        )
    with p_col3:
        max_depth = st.slider("最大深さ（max_depth）", -1, 10, 4, key="lgb_depth",
                              help="-1 は無制限")
    with p_col4:
        test_size = st.slider("テストデータ割合", 0.1, 0.4, 0.2, step=0.05, key="lgb_test")

    num_leaves = st.slider("葉の最大数（num_leaves）", 4, 128, 31, key="lgb_leaves")

    if st.button("LightGBM分析を実行", type="primary", key="lgb_run"):
        try:
            data_subset = df[feature_cols + [target_col]].dropna()

            if len(data_subset) < 10:
                st.error("有効なデータが不足しています（最低10行必要）。")
                return

            X = data_subset[feature_cols]
            y_raw = data_subset[target_col]

            is_regression = task_type.startswith("回帰")

            if is_regression:
                y = y_raw.astype(float)
                model = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    num_leaves=num_leaves,
                    random_state=42,
                    verbose=-1,
                )
                scoring = "r2"
            else:
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str))
                n_classes = len(le.classes_)
                objective = "multiclass" if n_classes > 2 else "binary"
                model = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    num_leaves=num_leaves,
                    objective=objective,
                    random_state=42,
                    verbose=-1,
                )
                scoring = "accuracy"

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)

            st.success("LightGBM分析が完了しました！")

            # --- Metrics ---
            st.markdown("### モデル評価指標")

            if is_regression:
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)

                m_cols = st.columns(4)
                with m_cols[0]:
                    with st.container(border=True):
                        st.metric("R²（テスト）", f"{r2:.4f}")
                with m_cols[1]:
                    with st.container(border=True):
                        st.metric("RMSE", f"{rmse:.4f}")
                with m_cols[2]:
                    with st.container(border=True):
                        st.metric("MAE", f"{mae:.4f}")
                with m_cols[3]:
                    with st.container(border=True):
                        st.metric("CV R²（平均）", f"{cv_scores.mean():.4f}")

                with st.expander("📖 回帰指標の解釈"):
                    st.markdown(
                        f"""
**R²**: テストデータに対する決定係数。1に近いほど良い予測。

**RMSE / MAE**: 予測誤差。目的変数と同じ単位で解釈できる。

**CV R²**: 5-fold交差検証の平均R²。過学習の検出に使用。

現在の値: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, CV R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}
                        """
                    )
            else:
                acc = accuracy_score(y_test, y_pred)

                m_cols = st.columns(3)
                with m_cols[0]:
                    with st.container(border=True):
                        st.metric("正解率（テスト）", f"{acc:.4f}")
                with m_cols[1]:
                    with st.container(border=True):
                        st.metric("CV 正解率（平均）", f"{cv_scores.mean():.4f}")
                with m_cols[2]:
                    with st.container(border=True):
                        st.metric("CV 標準偏差", f"{cv_scores.std():.4f}")

                # Classification report
                st.markdown("#### 分類レポート")
                report_dict = classification_report(
                    y_test, y_pred,
                    target_names=le.classes_.astype(str),
                    output_dict=True,
                )
                report_df = pd.DataFrame(report_dict).T
                st.dataframe(
                    report_df.style.format("{:.3f}", subset=["precision", "recall", "f1-score"]),
                    use_container_width=True,
                )

                # Confusion matrix
                st.markdown("#### 混同行列")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="予測クラス", y="実際のクラス", color="件数"),
                    x=le.classes_.astype(str),
                    y=le.classes_.astype(str),
                    text_auto=True,
                    color_continuous_scale="Blues",
                )
                fig_cm.update_layout(title="混同行列")
                st.plotly_chart(fig_cm, use_container_width=True)

                with st.expander("📖 分類指標の解釈"):
                    st.markdown(
                        f"""
**正解率（Accuracy）**: 全予測のうち正しく分類された割合。

**Precision / Recall / F1**: クラスごとの精度・再現率・調和平均。

**CV 正解率**: 5-fold交差検証の平均正解率。過学習の検出に使用。

現在の値: 正解率={acc:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}
                        """
                    )

            # --- Feature Importance ---
            st.markdown("### 特徴量の重要度")
            importance_tab1, importance_tab2 = st.tabs(["gain（情報利得）", "split（分割回数）"])

            imp_gain = model.feature_importances_
            imp_df_gain = pd.DataFrame({
                "特徴量": feature_cols,
                "重要度（gain）": imp_gain,
            }).sort_values("重要度（gain）", ascending=False)

            with importance_tab1:
                fig_gain = px.bar(
                    imp_df_gain,
                    x="重要度（gain）",
                    y="特徴量",
                    orientation="h",
                    title="特徴量の重要度（gain: 情報利得の合計）",
                    color="重要度（gain）",
                    color_continuous_scale="Greens",
                )
                fig_gain.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_gain, use_container_width=True)

            with importance_tab2:
                booster = model.booster_
                imp_split = booster.feature_importance(importance_type="split")
                imp_df_split = pd.DataFrame({
                    "特徴量": feature_cols,
                    "重要度（split）": imp_split,
                }).sort_values("重要度（split）", ascending=False)

                fig_split = px.bar(
                    imp_df_split,
                    x="重要度（split）",
                    y="特徴量",
                    orientation="h",
                    title="特徴量の重要度（split: 使用された分割回数）",
                    color="重要度（split）",
                    color_continuous_scale="Oranges",
                )
                fig_split.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_split, use_container_width=True)

            with st.expander("📖 特徴量重要度の解釈"):
                st.markdown(
                    """
**gain（情報利得）**: その特徴量が使われた分岐での情報利得の合計。予測への実質的な貢献度を表す。

**split（分割回数）**: その特徴量が木全体で分岐に使われた回数。頻繁に使われるほど高い。

一般的に **gain** がモデルへの貢献をより正確に反映します。
                    """
                )

            # --- Learning curves (loss) ---
            st.markdown("### 学習曲線（LightGBM ブースティング損失）")
            model_cv = lgb.LGBMRegressor(**model.get_params()) if is_regression else lgb.LGBMClassifier(**model.get_params())
            eval_metric = "rmse" if is_regression else "binary_logloss" if len(np.unique(y)) == 2 else "multi_logloss"
            callbacks = [lgb.log_evaluation(period=-1), lgb.record_evaluation(evals_result := {})]
            model_cv.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_metric=eval_metric,
                callbacks=callbacks,
            )

            metric_key = list(evals_result.get("training", {}).keys())
            if metric_key:
                mk = metric_key[0]
                train_loss = evals_result["training"][mk]
                valid_loss = evals_result["valid_1"][mk]
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(
                    x=list(range(1, len(train_loss) + 1)), y=train_loss,
                    mode="lines", name="訓練", line=dict(color="blue"),
                ))
                fig_lc.add_trace(go.Scatter(
                    x=list(range(1, len(valid_loss) + 1)), y=valid_loss,
                    mode="lines", name="テスト", line=dict(color="orange"),
                ))
                fig_lc.update_layout(
                    title=f"学習曲線（{mk}）",
                    xaxis_title="ブースティング回数",
                    yaxis_title=mk,
                )
                st.plotly_chart(fig_lc, use_container_width=True)

            # --- Predicted vs Actual (regression) ---
            if is_regression:
                st.markdown("### 予測値 vs 実測値")
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=y_test, y=y_pred, mode="markers",
                    marker=dict(color="green", opacity=0.6), name="データ",
                ))
                min_val = min(float(np.min(y_test)), float(np.min(y_pred)))
                max_val = max(float(np.max(y_test)), float(np.max(y_pred)))
                fig_pred.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode="lines", line=dict(color="red", dash="dash"), name="完全予測",
                ))
                fig_pred.update_layout(title="予測値 vs 実測値", xaxis_title="実測値", yaxis_title="予測値")
                st.plotly_chart(fig_pred, use_container_width=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
