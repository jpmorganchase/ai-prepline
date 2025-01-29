import functools
from typing import Any, List

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ts_data_fixture() -> pd.DataFrame:
    mock_time_series = pd.DataFrame(
        {
            "feature_a": list(range(1, 20)),
            "feature_b": [i * 2 for i in range(1, 20)],
        },
        index=list(range(1, 20)),
    )

    mock_time_series.feature_a.loc[2] = np.nan
    mock_time_series.feature_b.loc[4] = np.nan
    return mock_time_series


@pytest.fixture
def create_df():
    def _create_df(row_values: List[Any], col_name="Col"):
        """
        Creates a dataframe with "Col" column and a list of values for that column
        """
        return pd.DataFrame({col_name: row_values})

    return _create_df


@pytest.fixture(scope="function")
def get_logs(caplog):
    def wrapped(caplog):
        return list(caplog.messages)

    return functools.partial(wrapped, caplog)
