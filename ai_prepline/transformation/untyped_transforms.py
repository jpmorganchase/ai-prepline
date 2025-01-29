"""
A collection of handy transforms on columns of any data type
"""

from typing import Any, List, Optional

import pandas as pd

from ai_prepline.base.transform import BaseTransform

OCCURRENCE_FILTER_MIN_THRESHOLD = 2


class FilterColumnByValues(BaseTransform):
    """
    Filters dataframe by removing rows where column values match the entries
    provided in the exclude list.
    """

    def __init__(self, target_column_to_filter: str, values_to_filter_out: List[Any]):
        """
        Initialize the transformation

        :param target_column_to_filter: column against to match the filter
        :param values_to_filter_out: list of values which should be excluded if matched
            in the `target_column_to_filter`
        """
        self.target_column_to_filter = target_column_to_filter
        self.values_to_filter_out = values_to_filter_out

    def transform(self, X: pd.DataFrame, y: Optional[list] = None) -> pd.DataFrame:
        return X[~X[self.target_column_to_filter].isin(self.values_to_filter_out)]


class FilterByOccurrenceThreshold(BaseTransform):
    """
    Filters dataframe by dropping all rows for which the associated target column
    value is duplicated (occurs) >= the occurrence_threshold.
    """

    def __init__(self, target_column_to_filter: str, occurrence_threshold: int):
        """
        Initialise the transformation

        :param target_column_to_filter: column against to match the filter
        :param occurrence_threshold: >= threshold used to remove all rows where column
            values do not satisfy it
        """
        if occurrence_threshold < OCCURRENCE_FILTER_MIN_THRESHOLD:
            raise ValueError(
                f"Occurrence threshold has to be > {OCCURRENCE_FILTER_MIN_THRESHOLD}"
            )
        self.target_column_to_filter = target_column_to_filter
        self.occurrence_threshold = occurrence_threshold

    def transform(self, X: pd.DataFrame, y: Optional[list] = None) -> pd.DataFrame:
        return X[
            ~X[self.target_column_to_filter]
            .duplicated(keep=False)
            .groupby(X[self.target_column_to_filter])
            .transform("sum")
            .ge(self.occurrence_threshold)
        ]
