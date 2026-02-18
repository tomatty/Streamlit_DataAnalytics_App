"""
Data preprocessing module.
Handles data cleaning and transformation.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class DataPreprocessor:
    """Handles data preprocessing operations."""

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str, columns: list = None) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.

        Args:
            df: DataFrame to process
            method: Method to handle missing values ('drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_value')
            columns: List of columns to process (None = all columns)

        Returns:
            pd.DataFrame: Processed DataFrame
        """
        df_copy = df.copy()

        if columns is None:
            columns = df_copy.columns.tolist()

        if method == "drop":
            df_copy = df_copy.dropna(subset=columns)
        elif method == "fill_mean":
            for col in columns:
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif method == "fill_median":
            for col in columns:
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif method == "fill_mode":
            for col in columns:
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)

        return df_copy

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            df: DataFrame to process
            subset: Columns to consider for duplicate detection

        Returns:
            pd.DataFrame: DataFrame without duplicates
        """
        return df.drop_duplicates(subset=subset)

    @staticmethod
    def handle_outliers(df: pd.DataFrame, column: str, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in specified column.

        Args:
            df: DataFrame to process
            column: Column name
            method: Method to detect outliers ('iqr' or 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        df_copy = df.copy()

        if method == "iqr":
            Q1 = df_copy[column].quantile(0.25)
            Q3 = df_copy[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_copy = df_copy[
                (df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)
            ]
        elif method == "zscore":
            z_scores = np.abs(
                (df_copy[column] - df_copy[column].mean()) / df_copy[column].std()
            )
            df_copy = df_copy[z_scores < threshold]

        return df_copy

    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: list, method: str = "label") -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: DataFrame to process
            columns: List of column names to encode
            method: Encoding method ('label' or 'onehot')

        Returns:
            pd.DataFrame: DataFrame with encoded columns
        """
        df_copy = df.copy()

        if method == "label":
            le = LabelEncoder()
            for column in columns:
                df_copy[column] = le.fit_transform(df_copy[column])
        elif method == "onehot":
            df_copy = pd.get_dummies(df_copy, columns=columns, prefix=columns)

        return df_copy

    @staticmethod
    def scale_features(df: pd.DataFrame, columns: list, method: str = "standard") -> pd.DataFrame:
        """
        Scale numeric features.

        Args:
            df: DataFrame to process
            columns: List of columns to scale
            method: Scaling method ('standard' or 'minmax')

        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        df_copy = df.copy()

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        df_copy[columns] = scaler.fit_transform(df_copy[columns])

        return df_copy

    @staticmethod
    def filter_rows(df: pd.DataFrame, column: str, operator: str, value) -> pd.DataFrame:
        """
        Filter rows based on condition.

        Args:
            df: DataFrame to filter
            column: Column name
            operator: Comparison operator ('==', '!=', '>', '<', '>=', '<=')
            value: Value to compare

        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if operator == "==":
            return df[df[column] == value]
        elif operator == "!=":
            return df[df[column] != value]
        elif operator == ">":
            return df[df[column] > value]
        elif operator == "<":
            return df[df[column] < value]
        elif operator == ">=":
            return df[df[column] >= value]
        elif operator == "<=":
            return df[df[column] <= value]
        else:
            raise ValueError(f"Unknown operator: {operator}")

    @staticmethod
    def select_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Select specific columns.

        Args:
            df: DataFrame
            columns: List of column names to keep

        Returns:
            pd.DataFrame: DataFrame with selected columns
        """
        return df[columns]
