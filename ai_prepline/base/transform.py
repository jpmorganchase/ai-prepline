import abc
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransform(BaseEstimator, TransformerMixin, abc.ABC):
    """
    Base class for transformers. Override the .transform() method to implement
    the dataset transformation logic
    """

    # pylint: disable=unused-argument
    def fit(self, X: pd.DataFrame, y: Optional[list] = None) -> "BaseTransform":
        """
        Fit dataframe

        :param X: dataframe to fit
        :return: noop - return self
        """
        return self

    # pylint: disable=unused-argument
    @abc.abstractmethod
    def transform(self, X: pd.DataFrame, y: Optional[list] = None) -> pd.DataFrame:
        """
        Override me - Implement the data transformation

        :param X: dataframe to transform
        :param y: labels, noop, defaults to None
        :return: transformed data
        """
