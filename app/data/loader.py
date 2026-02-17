"""
Data loading module.
Handles loading data from various file formats.
"""
import pandas as pd
import json
from pathlib import Path
from typing import Union
import streamlit as st


class DataLoader:
    """Handles loading data from various file formats."""

    @staticmethod
    @st.cache_data
    def load_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file or file-like object
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            raise ValueError(f"CSVファイルの読み込みに失敗しました: {str(e)}")

    @staticmethod
    @st.cache_data
    def load_excel(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from Excel file.

        Args:
            file_path: Path to Excel file or file-like object
            **kwargs: Additional arguments for pd.read_excel

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            return pd.read_excel(file_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Excelファイルの読み込みに失敗しました: {str(e)}")

    @staticmethod
    @st.cache_data
    def load_json(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from JSON file.

        Args:
            file_path: Path to JSON file or file-like object
            **kwargs: Additional arguments for pd.read_json

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            return pd.read_json(file_path, **kwargs)
        except Exception as e:
            raise ValueError(f"JSONファイルの読み込みに失敗しました: {str(e)}")

    @staticmethod
    def load_file(file, file_name: str) -> pd.DataFrame:
        """
        Load data from uploaded file based on file extension.

        Args:
            file: Uploaded file object
            file_name: Name of the file

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            ValueError: If file format is not supported
        """
        file_extension = file_name.split(".")[-1].lower()

        if file_extension == "csv":
            return DataLoader.load_csv(file)
        elif file_extension in ["xlsx", "xls"]:
            return DataLoader.load_excel(file)
        elif file_extension == "json":
            return DataLoader.load_json(file)
        else:
            raise ValueError(
                f"サポートされていないファイル形式です: {file_extension}"
            )

    @staticmethod
    def get_data_metadata(df: pd.DataFrame) -> dict:
        """
        Extract metadata from DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            dict: Metadata including shape, dtypes, missing values, etc.
        """
        metadata = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
            "numeric_columns": list(df.select_dtypes(include=["number"]).columns),
            "categorical_columns": list(
                df.select_dtypes(include=["object", "category"]).columns
            ),
            "datetime_columns": list(df.select_dtypes(include=["datetime"]).columns),
        }
        return metadata

    @staticmethod
    def load_sample_data(sample_name: str) -> pd.DataFrame:
        """
        Load predefined sample data.

        Args:
            sample_name: Name of the sample dataset

        Returns:
            pd.DataFrame: Sample data

        Raises:
            ValueError: If sample dataset not found
        """
        sample_path = Path("data") / sample_name
        if not sample_path.exists():
            raise ValueError(f"サンプルデータが見つかりません: {sample_name}")

        return DataLoader.load_csv(sample_path)
