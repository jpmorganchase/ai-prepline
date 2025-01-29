from typing import Any, List

import pandas as pd
import pandera as pa
import pytest

from ai_prepline.data_checks.data_check import (
    CheckAgainstSchema,
    column_contains_datetime,
)


def create_df(row_values: List[Any], col_name="Col"):
    """
    Creates a dataframe with "Col" column and a list of values for that column
    """
    return pd.DataFrame({col_name: row_values})


@pytest.mark.parametrize(
    ["row_values", "datetime_cast", "expected"],
    [
        (
            ["1/1/2002", "1/2/2002"],
            "datetime64[ns]",
            True,
        ),
        (
            ["1/1/2002", "1/2/2002"],
            "datetime64[ns, UTC]",
            True,
        ),
        (
            ["1/1/2002", "1/2/2002"],
            None,
            False,
        ),
    ],
)
def test_column_contains_datetime(
    row_values: List[str], datetime_cast: str, expected: bool
):
    df = create_df(row_values=row_values)
    if datetime_cast:
        df.Col = df.Col.astype(datetime_cast)

    schema = pa.DataFrameSchema(
        {
            "Col": pa.Column(
                None,
                pa.Check(
                    column_contains_datetime,
                ),
            )
        }
    )

    cl = CheckAgainstSchema(schema, strict=True)

    try:
        cl.transform(df)
        assert True is expected
    except pa.errors.SchemaError:
        assert False is expected


@pytest.mark.parametrize(["row_values", "expected"], [([2, 3], True), ([1, 0], False)])
def test_check_against_schema(row_values: List[int], expected: bool):
    df = create_df(row_values=row_values)
    schema = pa.DataFrameSchema(
        {"Col": pa.Column(int, pa.Check.greater_than_or_equal_to(1))}
    )

    cl = CheckAgainstSchema(schema, strict=True)

    try:
        cl.transform(df)
        assert True is expected
    except pa.errors.SchemaError:
        assert False is expected


@pytest.mark.parametrize(["row_values"], [([-2, -3],)])
def test_check_against_schema_exception_catch(row_values: List[int]):
    df = create_df(row_values=row_values)
    schema = pa.DataFrameSchema(
        {"Col": pa.Column(int, pa.Check.greater_than_or_equal_to(1))}
    )

    cl = CheckAgainstSchema(schema, strict=False)

    # If strict == False, the .transform() should catch the exception internally.
    try:
        cl.transform(df)
        assert True
    except pa.errors.SchemaError:
        assert False
