"""
Base analyzer class.
Provides common functionality for all analysis modules.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd


class BaseAnalyzer(ABC):
    """Base class for all analyzers."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize analyzer.

        Args:
            data: DataFrame to analyze
        """
        self.data = data
        self.results = {}

    @abstractmethod
    def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Perform analysis.

        Args:
            **kwargs: Analysis-specific parameters

        Returns:
            Dict containing analysis results
        """
        pass

    def get_results(self) -> Dict[str, Any]:
        """
        Get analysis results.

        Returns:
            Dict containing analysis results
        """
        return self.results

    def validate_columns(self, columns: list, required_type=None) -> bool:
        """
        Validate that columns exist and optionally check their types.

        Args:
            columns: List of column names
            required_type: Required data type (e.g., 'numeric', 'categorical')

        Returns:
            bool: True if valid, raises ValueError otherwise
        """
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"列が見つかりません: {col}")

            if required_type == "numeric":
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    raise ValueError(f"列が数値型ではありません: {col}")
            elif required_type == "categorical":
                if not pd.api.types.is_object_dtype(self.data[col]) and not pd.api.types.is_categorical_dtype(self.data[col]):
                    raise ValueError(f"列がカテゴリカル型ではありません: {col}")

        return True
