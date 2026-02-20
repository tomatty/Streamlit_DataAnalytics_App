"""
Decision Tree analysis module.
Supports both classification and regression tasks.
"""
import matplotlib
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree

# Search for an available Japanese font (installed via packages.txt or japanize-matplotlib)
_JP_FONT_CANDIDATES = [
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "IPAGothic",
    "TakaoGothic",
    "VL Gothic",
    "Hiragino Sans",
    "Yu Gothic",
]
try:
    _fm._load_fontmanager(try_read_cache=False)
except Exception:
    pass
_available = {f.name for f in _fm.fontManager.ttflist}
for _candidate in _JP_FONT_CANDIDATES:
    if _candidate in _available:
        matplotlib.rcParams["font.family"] = _candidate
        break

_SS = "dt_results"  # session_state key


def show_decision_tree(df: pd.DataFrame):
    """Display Decision Tree analysis interface."""
    st.subheader("🌳 決定木分析")

    with st.expander("📖 一般的な分析手順", expanded=False):
        st.markdown(
            """
### 決定木分析の基本的な流れ

**1. 目的の明確化**
- 分類問題: カテゴリーを予測（例: 顧客が購入するか/しないか）
- 回帰問題: 数値を予測（例: 売上金額の予測）
- ルール抽出: if-then形式の判断基準を可視化
- 特徴量重要度の把握: どの変数が予測に重要か

**2. データの準備**
- **データ形式**:
  ```
  | サンプルID | 特徴量1 | 特徴量2 | 特徴量3 | 目的変数 |
  |-----------|--------|--------|--------|---------|
  | 1         | 25     | 50000  | 3      | 購入    |
  | 2         | 35     | 75000  | 5      | 非購入  |
  ```
- 特徴量（説明変数）: 予測に使う変数
- 目的変数: 予測したい変数
- **カテゴリー変数の扱い**:
  - **ラベルエンコーディング（推奨）**: カテゴリーを整数に変換
    - 例: 性別（男/女）→ 0/1、地域（東京/大阪/名古屋）→ 0/1/2
    - 決定木は分岐条件として使うだけなので、順序は関係ない
    - ワンホットエンコーディングは不要（木が深くなり非効率）
  - 目的変数がカテゴリーの場合は自動的にラベルエンコーディング

**3. ハイパーパラメータの設定**
- **max_depth（最大深さ）**: 木の深さの上限（3-10程度）
  - 深いほど複雑なパターンを学習するが、過学習のリスク
- **min_samples_split**: 分割に必要な最小サンプル数（2-20程度）
  - 大きいほどシンプルな木になる

**4. モデルの学習と評価**
- 訓練データとテストデータに分割（通常80:20）
- **分類タスク**: 正解率、混同行列、適合率・再現率
- **回帰タスク**: R²、RMSE、MAE

**5. 結果の解釈**
- 樹形図: どのような判断基準で予測しているか
- 特徴量重要度: どの変数が重要か
- ルール抽出: ビジネスで活用できる判断基準

**6. 注意点**
- 過学習しやすい（深すぎる木は訓練データに過度に適合）
- max_depthやmin_samples_splitで制御
- アンサンブル手法（Random Forest、LightGBM）でさらに精度向上可能
            """
        )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(all_cols) < 2:
        st.warning("決定木分析には少なくとも2つの列が必要です。")
        return

    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("目的変数（ターゲット）", all_cols, key="dt_target")
    with col2:
        task_type = st.radio(
            "タスク種別",
            ["分類（Classification）", "回帰（Regression）"],
            key="dt_task",
            horizontal=True,
        )

    feature_cols = st.multiselect(
        "説明変数（特徴量）",
        [c for c in numeric_cols if c != target_col],
        default=[c for c in numeric_cols if c != target_col][:min(5, len(numeric_cols))],
        key="dt_features",
    )

    if len(feature_cols) < 1:
        st.info("少なくとも1つの説明変数を選択してください。")
        return

    col3, col4, col5 = st.columns(3)
    with col3:
        max_depth = st.slider("最大深さ（max_depth）", min_value=1, max_value=10, value=4, key="dt_depth")
    with col4:
        min_samples_split = st.slider("分割最小サンプル数", min_value=2, max_value=20, value=5, key="dt_mss")
    with col5:
        test_size = st.slider("テストデータ割合", min_value=0.1, max_value=0.4, value=0.2, step=0.05, key="dt_test")

    if st.button("決定木分析を実行", type="primary", key="dt_run"):
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
                model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                )
            else:
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str))
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if is_regression:
                n_cv = min(5, len(y) // 2)
                cv_scores = cross_val_score(model, X, y, cv=n_cv, scoring="r2") if n_cv >= 2 else None
                metrics = {
                    "r2": r2_score(y_test, y_pred),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "cv_mean": float(cv_scores.mean()) if cv_scores is not None else None,
                    "cv_std": float(cv_scores.std()) if cv_scores is not None else None,
                }
            else:
                min_class_count = int(np.bincount(y).min())
                n_cv = min(5, min_class_count)
                cv_scores = cross_val_score(model, X, y, cv=n_cv, scoring="accuracy") if n_cv >= 2 else None
                metrics = {
                    "acc": float(accuracy_score(y_test, y_pred)),
                    "cv_mean": float(cv_scores.mean()) if cv_scores is not None else None,
                    "cv_std": float(cv_scores.std()) if cv_scores is not None else None,
                    "depth": int(model.get_depth()),
                }

            # Save to session_state
            st.session_state[_SS] = {
                "model": model,
                "feature_cols": feature_cols,
                "is_regression": is_regression,
                "max_depth": max_depth,
                "le": le,
                "y_test": y_test,
                "y_pred": y_pred,
                "X": X,
                "metrics": metrics,
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
    max_depth = res["max_depth"]
    le = res["le"]
    y_test = res["y_test"]
    y_pred = res["y_pred"]
    X = res["X"]
    metrics = res["metrics"]

    st.success("決定木分析が完了しました！")

    # --- Metrics ---
    st.markdown("### モデル評価指標")

    if is_regression:
        r2, rmse, mae = metrics["r2"], metrics["rmse"], metrics["mae"]
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
                cv_val = f"{metrics['cv_mean']:.4f}" if metrics["cv_mean"] is not None else "N/A"
                st.metric("CV R²（平均）", cv_val)

        if metrics["cv_mean"] is None:
            st.warning("サンプル数が不足しているため交差検証をスキップしました。")

        with st.expander("📖 回帰指標の解釈"):
            cv_detail = f"CV R²={metrics['cv_mean']:.4f}" if metrics["cv_mean"] is not None else "CV: スキップ"
            st.markdown(
                f"""
**R²**: テストデータに対する決定係数。1に近いほど良い予測。

**RMSE / MAE**: 予測誤差。目的変数と同じ単位で解釈できる。

**CV R²**: 交差検証の平均R²。過学習の検出に使用。

現在の値: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, {cv_detail}
                """
            )
    else:
        acc = metrics["acc"]
        m_cols = st.columns(3)
        with m_cols[0]:
            with st.container(border=True):
                st.metric("正解率（テスト）", f"{acc:.4f}")
        with m_cols[1]:
            with st.container(border=True):
                cv_val = f"{metrics['cv_mean']:.4f}" if metrics["cv_mean"] is not None else "N/A"
                st.metric("CV 正解率（平均）", cv_val)
        with m_cols[2]:
            with st.container(border=True):
                st.metric("木の深さ", f"{metrics['depth']}")

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
            width="stretch",
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
        st.plotly_chart(fig_cm, width="stretch")

        with st.expander("📖 分類指標の解釈"):
            st.markdown(
                f"""
**正解率（Accuracy）**: 全予測のうち正しく分類された割合。

**Precision / Recall / F1**: クラスごとの精度・再現率・調和平均。

**CV 正解率**: 交差検証の平均正解率。過学習の検出に使用。

現在の値: 正解率={acc:.4f}{f", CV={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}" if metrics['cv_mean'] is not None else ", CV: スキップ"}
                """
            )

    # --- Feature Importance ---
    st.markdown("### 特徴量の重要度")
    importance_df = pd.DataFrame({
        "特徴量": feature_cols,
        "重要度": model.feature_importances_,
    }).sort_values("重要度", ascending=False)

    fig_imp = px.bar(
        importance_df,
        x="重要度",
        y="特徴量",
        orientation="h",
        title="特徴量の重要度（Gini不純度ベース）",
        color="重要度",
        color_continuous_scale="Blues",
    )
    fig_imp.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, width="stretch")

    # --- Tree visualization ---
    st.markdown("### 木の構造")
    tab_plot, tab_text = st.tabs(["樹形図", "テキスト表示"])

    with tab_plot:
        col_depth, col_prop = st.columns([2, 1])
        with col_depth:
            display_depth = st.slider(
                "表示する深さ",
                min_value=1,
                max_value=min(max_depth, 6),
                value=min(3, max_depth),
                key="dt_display_depth",
            )
        with col_prop:
            show_proportion = st.checkbox(
                "割合で表示",
                value=False,
                key="dt_show_proportion",
                help="サンプル数の代わりに全体に対する割合（確率）を表示します",
            )

        class_names = le.classes_.astype(str).tolist() if not is_regression else None
        actual_depth = model.get_depth()
        # Cap visible leaves to avoid over-expanding the figure
        visible_leaves = min(2 ** display_depth, model.get_n_leaves())
        fig_width = max(10, min(visible_leaves * 2.5, 48))
        fig_height = max(5, display_depth * 3.2)
        fontsize = max(9, 13 - display_depth)
        fig_tree, ax = plt.subplots(figsize=(fig_width, fig_height))
        plot_tree(
            model,
            max_depth=display_depth,
            feature_names=feature_cols,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=fontsize,
            impurity=True,
            proportion=show_proportion,
            precision=3,
            ax=ax,
        )
        ax.set_title(
            f"決定木（表示深さ: {display_depth} / 実際の深さ: {actual_depth}）",
            fontsize=12,
        )
        st.pyplot(fig_tree, width="stretch")
        plt.close(fig_tree)

        with st.expander("📖 樹形図の見方"):
            if show_proportion:
                st.markdown(
                    """
**各ノードの表示内容：**
- **分割条件**: 特徴量 <= 閾値（内部ノードのみ）
- **gini/entropy**: 不純度（0に近いほど純粋）
- **samples**: 全体に対するサンプルの割合（0.0〜1.0）
- **value**: 各クラスのサンプル割合（分類）または予測値（回帰）
- **class**: 最も多いクラス（分類のみ）

「割合で表示」オン時は、各ノードのサンプル数と各クラスの数が割合（確率）で表示されます。
                    """
                )
            else:
                st.markdown(
                    """
**各ノードの表示内容：**
- **分割条件**: 特徴量 <= 閾値（内部ノードのみ）
- **gini/entropy**: 不純度（0に近いほど純粋）
- **samples**: サンプル数
- **value**: 各クラスのサンプル数（分類）または予測値（回帰）
- **class**: 最も多いクラス（分類のみ）

「割合で表示」をオンにすると、確率表示に切り替わります。
                    """
                )

    with tab_text:
        tree_text = export_text(model, feature_names=feature_cols, max_depth=min(max_depth, 4))
        st.code(tree_text, language="text")

    # --- Predicted vs Actual (regression) ---
    if is_regression:
        st.markdown("### 予測値 vs 実測値")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            marker=dict(color="steelblue", opacity=0.6),
            name="データ",
        ))
        min_val = min(float(np.min(y_test)), float(np.min(y_pred)))
        max_val = max(float(np.max(y_test)), float(np.max(y_pred)))
        fig_pred.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", line=dict(color="red", dash="dash"), name="完全予測",
        ))
        fig_pred.update_layout(title="予測値 vs 実測値", xaxis_title="実測値", yaxis_title="予測値")
        st.plotly_chart(fig_pred, width="stretch")
