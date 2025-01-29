"""
Utility functions for using pandera to conduct data quality checks. 
"""

import datetime
import logging

import pandas as pd
import pandera as pa

from ai_prepline.base.transform import BaseTransform

logger = logging.getLogger(__name__)


def column_contains_datetime(series: pd.Series):
    """
    Check that a pandas.Series contains datetime objects.
    To be used with pandera data checks.

    :param series: pd.Series
    :return : output to pass to a pa.Check(.)
    """
    return series.apply(lambda y: isinstance(y, datetime.datetime))


class CheckAgainstSchema(BaseTransform):
    """
    Transform that checks dataframes against a pre-defined pandera scheme.
    This is useful as a way to fold such checks between steps of an sklearn pipeline.
    """

    def __init__(self, schema: pa.DataFrameSchema, strict: bool = True):
        """
        :param schema: a pandera DataFrameSchema for testing properties of data.
        :param strict: bool, raises a pa.errors.SchemaError on failure.
        """
        self.schema = schema
        self.strict = strict

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Runs the consistency checks defined in self.schema. On success,
        it returns the frame X as is.
        """
        try:
            return self.schema.validate(X)
        except pa.errors.SchemaError as exc:
            logger.error(exc)
            if self.strict:
                raise

            return X
