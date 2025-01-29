from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from ai_prepline.transformation.untyped_transforms import (
    FilterByOccurrenceThreshold,
    FilterColumnByValues,
)


@pytest.mark.parametrize(
    ["sample", "values", "expected"],
    [
        (["A", "B", "C"], ["A", "B"], ["C"]),
        (
            [
                "A",
                "B",
                "C",
            ],
            ["A", "D"],
            ["B", "C"],
        ),
        ([0, 1, 2], [2, 0], [1]),
        ([0, 1, 2], [3], [0, 1, 2]),
    ],
)
def test_filter_column_by_values(sample, values, expected, create_df):
    data = create_df(sample, col_name="test")
    tr = FilterColumnByValues("test", values)
    pipe = Pipeline([("filter", tr)])
    transformed = pipe.fit_transform(data)
    np.testing.assert_equal(transformed["test"].to_list(), expected)


@pytest.mark.parametrize(
    ["sample", "occurrence", "expected", "raises"],
    [
        (["A", "B", "C", "C", "C"], 2, ["A", "B"], does_not_raise()),
        (["A", "B", "B", "C", "C", "C"], 3, ["A", "B", "B"], does_not_raise()),
        (["A", "B", "C"], 2, ["A", "B", "C"], does_not_raise()),
        ([0, 1, 1, 2], 1, [0, 2], pytest.raises(ValueError)),
        ([0, 1, 1, 2], 2, [0, 2], does_not_raise()),
        ([0, 0, 0, 0, 1, 1, 1, 2], 2, [2], does_not_raise()),
    ],
)
def test_filter_column_by_occurrence(sample, occurrence, expected, create_df, raises):
    data = create_df(sample, col_name="test")
    with raises:
        tr = FilterByOccurrenceThreshold("test", occurrence)
        pipe = Pipeline([("filter", tr)])
        transformed = pipe.fit_transform(data)
        np.testing.assert_equal(transformed["test"].to_list(), expected)
