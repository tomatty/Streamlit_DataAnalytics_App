"""
Data validation module.
Validates uploaded data and file formats.
"""
import pandas as pd
from pathlib import Path
from app.config import config


class DataValidator:
    """Validates data and files."""

    @staticmethod
    def validate_file_type(file_name: str) -> bool:
        """
        Validate if file type is allowed.

        Args:
            file_name: Name of the file

        Returns:
            bool: True if file type is allowed, False otherwise
        """
        file_extension = file_name.split(".")[-1].lower()
        return file_extension in config.app.allowed_file_types

    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """
        Validate if file size is within limits.

        Args:
            file_size: Size of file in bytes

        Returns:
            bool: True if file size is acceptable, False otherwise
        """
        max_size_bytes = config.app.max_upload_size_mb * 1024 * 1024
        return file_size <= max_size_bytes

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            tuple: (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "データが空です"

        if len(df.columns) == 0:
            return False, "列が存在しません"

        if len(df) == 0:
            return False, "行が存在しません"

        return True, ""

    @staticmethod
    def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> tuple[bool, str]:
        """
        Validate that required columns exist in DataFrame.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            tuple: (is_valid, error_message)
        """
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            return False, f"必要な列が見つかりません: {', '.join(missing_columns)}"
        return True, ""

    @staticmethod
    def validate_numeric_columns(df: pd.DataFrame, columns: list[str]) -> tuple[bool, str]:
        """
        Validate that specified columns contain numeric data.

        Args:
            df: DataFrame to validate
            columns: List of column names to check

        Returns:
            tuple: (is_valid, error_message)
        """
        for col in columns:
            if col not in df.columns:
                return False, f"列が存在しません: {col}"
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"列が数値型ではありません: {col}"
        return True, ""
