"""
LightGBM decision tree analysis module.
Supports both classification and regression tasks.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import graphviz
    import lightgbm as lgb
except ImportError:
    lgb = None
    graphviz = None

_SS = "lgb_results"  # session_state key


def _get_tree_depth(node: dict) -> int:
    """Recursively compute the depth of a LightGBM tree node."""
    if "leaf_index" in node:
        return 0
    return 1 + max(
        _get_tree_depth(node["left_child"]),
        _get_tree_depth(node["right_child"]),
    )


def _build_depth_limited_digraph(
    tree_dict: dict,
    max_display_depth: int,
    feature_names: list,
    precision: int = 3,
) -> str:
    """Build a graphviz source string from a LightGBM tree dict, limited to max_display_depth."""
    graph = graphviz.Digraph(
        graph_attr={"rankdir": "TB", "fontsize": "12"},
        node_attr={"fontsize": "11"},
        edge_attr={"fontsize": "10"},
    )
    counter = [0]

    def add_node(node: dict, parent_id: str | None, edge_label: str, depth: int) -> None:
        node_id = str(counter[0])
        counter[0] += 1

        is_leaf = "leaf_index" in node

        if is_leaf or depth >= max_display_depth:
            if is_leaf:
                value = node.get("leaf_value", 0)
                count = node.get("leaf_count", 0)
                label = f"leaf: {value:.{precision}f}\ncount: {count}"
            else:
                count = node.get("internal_count", 0)
                label = f"...\ncount: {count}"
            graph.node(node_id, label=label, shape="ellipse", style="filled", fillcolor="#ffffcc")
        else:
            feat = node.get("split_feature", "?")
            if feature_names and isinstance(feat, int) and feat < len(feature_names):
                feat = feature_names[feat]
            threshold = node.get("threshold", "?")
            try:
                threshold_str = f"{float(threshold):.{precision}f}"
            except (ValueError, TypeError):
                threshold_str = str(threshold)
            gain = node.get("split_gain", 0)
            count = node.get("internal_count", 0)
            label = f"{feat} <= {threshold_str}\ngain: {gain:.{precision}f}\ncount: {count}"
            graph.node(node_id, label=label, shape="box", style="filled", fillcolor="#dae8fc")

        if parent_id is not None:
            graph.edge(parent_id, node_id, label=edge_label)

        if not is_leaf and depth < max_display_depth:
            add_node(node["left_child"], node_id, "yes", depth + 1)
            add_node(node["right_child"], node_id, "no", depth + 1)

    add_node(tree_dict.get("tree_structure", {}), None, "", 0)
    return graph.source


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

            le = None
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

            # Fit with eval for learning curves
            evals_result: dict = {}
            eval_metric = "rmse" if is_regression else (
                "binary_logloss" if len(np.unique(y)) == 2 else "multi_logloss"
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_metric=eval_metric,
                callbacks=[lgb.log_evaluation(period=-1), lgb.record_evaluation(evals_result)],
            )
            y_pred = model.predict(X_test)

            # Cross-validation
            if is_regression:
                n_cv = min(5, len(y) // 2)
            else:
                min_class_count = int(np.bincount(y).min())
                n_cv = min(5, min_class_count)
            cv_scores = cross_val_score(model, X, y, cv=n_cv, scoring=scoring) if n_cv >= 2 else None

            # Metrics
            if is_regression:
                metrics = {
                    "r2": float(r2_score(y_test, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "cv_mean": float(cv_scores.mean()) if cv_scores is not None else None,
                    "cv_std": float(cv_scores.std()) if cv_scores is not None else None,
                }
            else:
                metrics = {
                    "acc": float(accuracy_score(y_test, y_pred)),
                    "cv_mean": float(cv_scores.mean()) if cv_scores is not None else None,
                    "cv_std": float(cv_scores.std()) if cv_scores is not None else None,
                }

            st.session_state[_SS] = {
                "model": model,
                "feature_cols": feature_cols,
                "is_regression": is_regression,
                "n_estimators": n_estimators,
                "le": le,
                "y_test": y_test,
                "y_pred": y_pred,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "metrics": metrics,
                "evals_result": evals_result,
                "eval_metric_key": eval_metric,
            }

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            return

    # ---- Render results from session_state ----
    if _SS not in st.session_state:
        return

    res = st.session_state[_SS]
    model = res["model"]
    feature_cols = res["feature_cols"]
    is_regression = res["is_regression"]
    n_estimators = res["n_estimators"]
    le = res["le"]
    y_test = res["y_test"]
    y_pred = res["y_pred"]
    X_train = res["X_train"]
    X_test = res["X_test"]
    y_train = res["y_train"]
    metrics = res["metrics"]
    evals_result = res["evals_result"]

    st.success("LightGBM分析が完了しました！")

    # --- Metrics ---
    st.markdown("### モデル評価指標")

    if is_regression:
        r2, rmse, mae = metrics["r2"], metrics["rmse"], metrics["mae"]
        cv_str = f"{metrics['cv_mean']:.4f}" if metrics["cv_mean"] is not None else "N/A"
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
                st.metric("CV R²（平均）", cv_str)
        if metrics["cv_mean"] is None:
            st.warning("サンプル数が不足しているため交差検証をスキップしました。")
        with st.expander("📖 回帰指標の解釈"):
            cv_detail = f"CV R²={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}" if metrics["cv_mean"] is not None else "CV: スキップ"
            st.markdown(f"""
**R²**: テストデータに対する決定係数。1に近いほど良い予測。

**RMSE / MAE**: 予測誤差。目的変数と同じ単位で解釈できる。

**CV R²**: 交差検証の平均R²。過学習の検出に使用。

現在の値: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, {cv_detail}
            """)
    else:
        acc = metrics["acc"]
        cv_mean_str = f"{metrics['cv_mean']:.4f}" if metrics["cv_mean"] is not None else "N/A"
        cv_std_str = f"{metrics['cv_std']:.4f}" if metrics["cv_std"] is not None else "N/A"
        m_cols = st.columns(3)
        with m_cols[0]:
            with st.container(border=True):
                st.metric("正解率（テスト）", f"{acc:.4f}")
        with m_cols[1]:
            with st.container(border=True):
                st.metric("CV 正解率（平均）", cv_mean_str)
        with m_cols[2]:
            with st.container(border=True):
                st.metric("CV 標準偏差", cv_std_str)
        if metrics["cv_mean"] is None:
            st.warning("一部クラスのサンプル数が不足しているため交差検証をスキップしました。")

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
            cv_detail = f"CV={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}" if metrics["cv_mean"] is not None else "CV: スキップ"
            st.markdown(f"""
**正解率（Accuracy）**: 全予測のうち正しく分類された割合。

**Precision / Recall / F1**: クラスごとの精度・再現率・調和平均。

**CV 正解率**: 交差検証の平均正解率。過学習の検出に使用。

現在の値: 正解率={acc:.4f}, {cv_detail}
            """)

    # --- Feature Importance ---
    st.markdown("### 特徴量の重要度")
    imp_tab1, imp_tab2 = st.tabs(["gain（情報利得）", "split（分割回数）"])
    booster = model.booster_

    with imp_tab1:
        imp_gain = booster.feature_importance(importance_type="gain")
        imp_df_gain = pd.DataFrame({
            "特徴量": feature_cols, "重要度（gain）": imp_gain,
        }).sort_values("重要度（gain）", ascending=False)
        fig_gain = px.bar(
            imp_df_gain, x="重要度（gain）", y="特徴量", orientation="h",
            title="特徴量の重要度（gain: 情報利得の合計）",
            color="重要度（gain）", color_continuous_scale="Greens",
        )
        fig_gain.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_gain, use_container_width=True)

    with imp_tab2:
        imp_split = booster.feature_importance(importance_type="split")
        imp_df_split = pd.DataFrame({
            "特徴量": feature_cols, "重要度（split）": imp_split,
        }).sort_values("重要度（split）", ascending=False)
        fig_split = px.bar(
            imp_df_split, x="重要度（split）", y="特徴量", orientation="h",
            title="特徴量の重要度（split: 使用された分割回数）",
            color="重要度（split）", color_continuous_scale="Oranges",
        )
        fig_split.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_split, use_container_width=True)

    with st.expander("📖 特徴量重要度の解釈"):
        st.markdown("""
**gain（情報利得）**: その特徴量が使われた分岐での情報利得の合計。予測への実質的な貢献度を表す。

**split（分割回数）**: その特徴量が木全体で分岐に使われた回数。

一般的に **gain** がモデルへの貢献をより正確に反映します。
        """)

    # --- Tree visualization ---
    st.markdown("### 木の構造")
    sl_col1, sl_col2 = st.columns(2)
    with sl_col1:
        tree_index = st.slider(
            "表示する木のインデックス（0 = 最初の木）",
            min_value=0, max_value=n_estimators - 1, value=0,
            key="lgb_tree_idx",
        )

    try:
        dump = model.booster_.dump_model()
        trees = dump.get("tree_info", [])
        if tree_index < len(trees):
            actual_depth = _get_tree_depth(trees[tree_index].get("tree_structure", {}))
        else:
            actual_depth = 1
    except Exception as e:
        st.warning(f"木の構造の取得に失敗しました: {e}")
        actual_depth = 1

    with sl_col2:
        if actual_depth > 1:
            display_depth = st.slider(
                "表示する分岐の深さ",
                min_value=1, max_value=actual_depth,
                value=min(3, actual_depth),
                key="lgb_display_depth",
            )
        else:
            display_depth = actual_depth
            st.caption(f"木の深さ: {actual_depth}（スライダー不要）")

    try:
        if graphviz is not None and tree_index < len(trees):
            source = _build_depth_limited_digraph(
                trees[tree_index],
                max_display_depth=display_depth,
                feature_names=feature_cols,
                precision=3,
            )
            st.caption(f"木 #{tree_index}　実際の深さ: {actual_depth}　表示深さ: {display_depth}")
            st.graphviz_chart(source, use_container_width=True)
        else:
            st.warning("graphvizが利用できません。")
    except Exception as e:
        st.warning(f"樹形図の描画に失敗しました: {e}")

    # --- Learning curves ---
    st.markdown("### 学習曲線")
    metric_keys = list(evals_result.get("training", {}).keys())
    if metric_keys:
        mk = metric_keys[0]
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
