from typing import List

import numpy as np
import pandas as pd
import pytest

from ai_prepline.transformation.time_series_transforms import TimeSeriesTransform

pd.options.mode.chained_assignment = None


def generate_result(
    ts_data_fixture: pd.DataFrame,
    time_series_col: List[str],
    window_size: int,
    time_horizon: int,
    output_dimension: int,
    offset: int,
) -> pd.DataFrame:
    tst = TimeSeriesTransform()

    result = tst.rolling_window_io_generation(
        time_series_df=ts_data_fixture,
        time_series_col=time_series_col,
        window_size=window_size,
        time_horizon=time_horizon,
        output_dimension=output_dimension,
        offset=offset,
    )
    return result


@pytest.mark.parametrize(
    [
        "time_series_col",
        "window_size",
        "time_horizon",
        "output_dimension",
        "offset",
        "expected",
    ],
    [
        ("feature_a", 3, 0, 1, 1, str(np.array([1, np.nan, 3]))),
        (
            ["feature_a", "feature_b"],
            3,
            0,
            1,
            1,
            str(np.array([[1, 2], [np.nan, 4], [3, 6]])),
        ),
    ],
)
def test_param_time_series_col(
    ts_data_fixture,
    time_series_col,
    window_size,
    time_horizon,
    output_dimension,
    offset,
    expected,
):
    result = generate_result(
        ts_data_fixture,
        time_series_col,
        window_size,
        time_horizon,
        output_dimension,
        offset,
    )

    assert str(result.x.iloc[0]) == expected


@pytest.mark.parametrize(
    [
        "time_series_col",
        "window_size",
        "time_horizon",
        "output_dimension",
        "offset",
        "expected",
    ],
    [
        ("feature_a", 3, 0, 1, 1, 3),
    ],
)
def test_param_window_size(
    ts_data_fixture,
    time_series_col,
    window_size,
    time_horizon,
    output_dimension,
    offset,
    expected,
):
    result = generate_result(
        ts_data_fixture,
        time_series_col,
        window_size,
        time_horizon,
        output_dimension,
        offset,
    )

    assert len(result.x.iloc[0]) == expected


@pytest.mark.parametrize(
    [
        "time_series_col",
        "window_size",
        "time_horizon",
        "output_dimension",
        "offset",
        "expected",
    ],
    [
        ("feature_a", 3, 0, 1, 1, str(np.array([4.0]))),
        ("feature_a", 3, 5, 1, 1, str(np.array([9.0]))),
    ],
)
def test_param_time_horizon(
    ts_data_fixture,
    time_series_col,
    window_size,
    time_horizon,
    output_dimension,
    offset,
    expected,
):
    result = generate_result(
        ts_data_fixture,
        time_series_col,
        window_size,
        time_horizon,
        output_dimension,
        offset,
    )

    assert str(result.y.iloc[0]) == expected


@pytest.mark.parametrize(
    [
        "time_series_col",
        "window_size",
        "time_horizon",
        "output_dimension",
        "offset",
        "expected",
    ],
    [
        ("feature_a", 3, 0, 1, 1, 1),
        ("feature_a", 3, 0, 3, 1, 3),
    ],
)
def test_param_output_dimension(
    ts_data_fixture,
    time_series_col,
    window_size,
    time_horizon,
    output_dimension,
    offset,
    expected,
):
    result = generate_result(
        ts_data_fixture,
        time_series_col,
        window_size,
        time_horizon,
        output_dimension,
        offset,
    )

    assert len(result.y.iloc[0]) == expected


@pytest.mark.parametrize(
    [
        "time_series_col",
        "window_size",
        "time_horizon",
        "output_dimension",
        "offset",
        "expected1",
        "expected2",
    ],
    [
        (
            "feature_a",
            3,
            0,
            1,
            1,
            str(np.array([1, np.nan, 3])),
            str(np.array([np.nan, 3.0, 4.0])),
        ),
        (
            "feature_a",
            3,
            0,
            1,
            3,
            str(np.array([1, np.nan, 3])),
            str(np.array([4.0, 5.0, 6.0])),
        ),
    ],
)
def test_param_offset(
    ts_data_fixture,
    time_series_col,
    window_size,
    time_horizon,
    output_dimension,
    offset,
    expected1,
    expected2,
):
    result = generate_result(
        ts_data_fixture,
        time_series_col,
        window_size,
        time_horizon,
        output_dimension,
        offset,
    )

    assert str(result.x.iloc[:2].values[0]) == expected1
    assert str(result.x.iloc[:2].values[1]) == expected2


@pytest.mark.parametrize(
    [
        "time_series_col",
        "window_size",
        "time_horizon",
        "output_dimension",
        "offset",
        "expected",
    ],
    [
        ("feature_a", 3, 0, 1, 1, str(np.array([1, np.nan, 3]))),
    ],
)
def test_transform_capability(
    ts_data_fixture,
    time_series_col,
    window_size,
    time_horizon,
    output_dimension,
    offset,
    expected,
):
    tst = TimeSeriesTransform(
        time_series_col=time_series_col,
        window_size=window_size,
        time_horizon=time_horizon,
        output_dimension=output_dimension,
        offset=offset,
    )

    result = tst.transform(ts_data_fixture)

    assert str(result.x.iloc[0]) == expected
