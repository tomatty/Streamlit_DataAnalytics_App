"""
Application constants.
"""

# Session state keys
SESSION_AUTHENTICATED = "authenticated"
SESSION_LOGIN_TIME = "login_time"
SESSION_USERNAME = "username"
SESSION_UPLOADED_FILE_NAME = "uploaded_file_name"
SESSION_RAW_DATA = "raw_data"
SESSION_PROCESSED_DATA = "processed_data"
SESSION_DATA_METADATA = "data_metadata"
SESSION_PREPROCESSING_STEPS = "preprocessing_steps"
SESSION_ANALYSIS_HISTORY = "analysis_history"
SESSION_CURRENT_ANALYSIS = "current_analysis"
SESSION_CURRENT_PAGE = "current_page"
SESSION_SETTINGS = "analysis_settings"

# Page names
PAGE_LOGIN = "login"
PAGE_DATA_OVERVIEW = "data_overview"
PAGE_PREPROCESSING = "preprocessing"
PAGE_ANALYSIS = "analysis"

# File types
SUPPORTED_FILE_TYPES = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "json": "application/json",
}

# Analysis types
ANALYSIS_DESCRIPTIVE = "descriptive"
ANALYSIS_CORRELATION = "correlation"
ANALYSIS_REGRESSION = "regression"
ANALYSIS_HYPOTHESIS_TESTING = "hypothesis_testing"
ANALYSIS_DIMENSIONALITY = "dimensionality"
ANALYSIS_CLUSTERING = "clustering"
ANALYSIS_TEXT = "text"
ANALYSIS_SPECIALIZED = "specialized"

# Aggregation functions
AGG_FUNCTIONS = ["sum", "mean", "median", "std", "var", "min", "max", "count"]

# Clustering algorithms
CLUSTERING_KMEANS = "kmeans"
CLUSTERING_HIERARCHICAL = "hierarchical"
CLUSTERING_DBSCAN = "dbscan"

# Sample data paths
SAMPLE_DATA_DIR = "data"
SAMPLE_SURVEY = "sample_survey.csv"
SAMPLE_PURCHASE_LOG = "sample_purchase_log.csv"
SAMPLE_TEXT_DATA = "sample_text_data.csv"

# UI Messages
MSG_LOGIN_SUCCESS = "ログインに成功しました"
MSG_LOGIN_FAILED = "ユーザー名またはパスワードが正しくありません"
MSG_LOGOUT_SUCCESS = "ログアウトしました"
MSG_FILE_UPLOAD_SUCCESS = "ファイルのアップロードに成功しました"
MSG_FILE_UPLOAD_ERROR = "ファイルのアップロードに失敗しました"
MSG_NO_DATA = "データがアップロードされていません"
MSG_ANALYSIS_SUCCESS = "分析が完了しました"
MSG_ANALYSIS_ERROR = "分析中にエラーが発生しました"

# Color palette for visualizations
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ff9800",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40",
}
