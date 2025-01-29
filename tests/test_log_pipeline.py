from typing import Any, List

import numpy as np
import pandas as pd
import pandera as pa
import pytest
from sklearn.pipeline import Pipeline

from ai_prepline.data_checks.data_check import (
    CheckAgainstSchema,
    column_contains_datetime,
)
from ai_prepline.pipeline_generators.logfile_pipelines import (
    create_input_output_pandera_checks,
    generate_logdata_pipeline,
    prepare_datetime_checks,
    prepare_drain3_miner,
)


def create_df(row_values: List[Any], col_name="Col"):
    """
    Creates a dataframe with "Col" column and a list of values for that column
    """
    return pd.DataFrame({col_name: row_values})


@pytest.mark.parametrize(
    ["row_values", "max_date", "min_date", "datetime_format", "expected"],
    [
        (["1/1/2025"], None, None, "datetime64[ns]", True),
        (["1/1/2022"], "1/1/2022", "1/1/2022", "datetime64[ns]", True),
        (
            ["1/1/2022"],
            "1/1/2022",
            "1/1/2022",
            "datetime64[ns, UTC]",
            True,
        ),
        (
            ["1/1/2022"],
            "1/1/2022",
            "1/1/2022",
            "datetime64[ns, EST]",
            True,
        ),
        (["1/1/2022"], None, "1/1/2022", "datetime64[ns]", True),
        (["1/1/2022"], None, "1/1/2023", "datetime64[ns]", False),
        (["4/1/2021", "1/1/2022"], "1/1/2021", "1/1/2021", "datetime64[ns]", False),
    ],
)
def test_prepare_datetime_checks(
    row_values, min_date, max_date, datetime_format, expected
):
    df = create_df(row_values)
    df["Col"] = df["Col"].astype(datetime_format)

    checks = prepare_datetime_checks(max_date, min_date)
    checks.append(pa.Check(column_contains_datetime))

    schema = pa.DataFrameSchema({"Col": pa.Column(None, checks=checks)})

    try:
        schema.validate(df)
        assert True is expected
    except pa.errors.SchemaError:
        assert False is expected


@pytest.mark.parametrize(
    [
        "df_as_dict",
        "convert_to_datetime",
        "minimum_elements_per_window_check",
        "expected",
    ],
    [
        (
            {
                "text": ["hello"],
                "id": ["1"],
                "time": ["1/1/2001"],
                "start": ["1/1/2001"],
                "end": ["1/1/2001"],
                "_count_elements_per_group": [1],
                "message_template_concatenated": ["test"],
                "setfit_input": [""],
            },
            True,
            1,
            True,
        ),
        (
            {
                "text": ["hello"],
                "id": ["1"],
                "time": ["1/1/2001"],
                "start": ["1/1/2001"],
                "end": ["1/1/2001"],
                "_count_elements_per_group": [1],
                "message_template_concatenated": ["test"],
                "setfit_input": [""],
            },
            False,
            1,
            False,
        ),
        (
            {
                "text": ["hello"],
                "id": ["1"],
                "time": ["1/1/2001"],
                "start": ["1/1/2001"],
                "end": ["1/1/2001"],
                "_count_elements_per_group": [1],
                "message_template_concatenated": ["test"],
                "setfit_input": [""],
            },
            True,
            2,
            False,
        ),
    ],
)
def test_create_input_output_pandera_checks(
    df_as_dict, convert_to_datetime, minimum_elements_per_window_check, expected
):
    df = pd.DataFrame().from_dict(df_as_dict)

    if convert_to_datetime:
        for col in ["time", "start", "end"]:
            df[col] = df[col].astype(np.datetime64)

    checks = prepare_datetime_checks(max_date_check=None, min_date_check=None)
    checks.append(pa.Check(column_contains_datetime))
    input_schema, output_schema = create_input_output_pandera_checks(
        "text",
        "id",
        "time",
        minimum_elements_per_window_check=minimum_elements_per_window_check,
        datetime_checks=checks,
    )

    try:
        input_schema.validate(df)
        output_schema.validate(df)
        assert True is expected
    except pa.errors.SchemaError:
        assert False is expected


def test_prepare_drain3_miner_init():
    miner = prepare_drain3_miner()
    assert miner is not None


def test_generate_logdata_pipeline_init():
    pipeline_list, check_against_output_schema = generate_logdata_pipeline(
        target_text_column="text", device_id_column="id"
    )

    assert (len(pipeline_list) > 0) and isinstance(
        check_against_output_schema, CheckAgainstSchema
    )


@pytest.mark.parametrize(
    [
        "df_as_dict",
        "expected_nrows_shape",
    ],
    [
        (
            {
                "text": ["a"],
                "id": ["1"],
                "time": ["1/1/2001"],
            },
            1,
        ),
        (
            {
                "text": ["a", "b"],
                "id": ["1", "1"],
                "time": ["1/1/2001", "1/1/2001"],
            },
            1,
        ),
        (
            {
                "text": ["a", "b"],
                "id": ["1", "2"],
                "time": ["1/1/2001", "1/1/2001"],
            },
            2,
        ),
        (
            {
                "text": ["a", "b"],
                "id": ["1", "1"],
                "time": ["1/1/2001 1:00:00", "2/1/2001 1:00:00"],
            },
            3,  # window end-points are inclusive
        ),
    ],
)
def test_shapes_logdata_pipeline(df_as_dict, expected_nrows_shape):
    pipeline, _ = generate_logdata_pipeline(
        target_text_column="text",
        device_id_column="id",
        timestamp_column_name="time",
        run_output_check=False,
    )
    pipeline = Pipeline(pipeline)

    df = pd.DataFrame().from_dict(df_as_dict)
    df["time"] = df["time"].astype(np.datetime64)
    pipeline.fit(df)
    new_df = pipeline.transform(df)
    assert new_df.shape[0] == expected_nrows_shape


@pytest.mark.parametrize(
    [
        "df_as_dict",
        "expected_nrows_shape",
    ],
    [
        (
            {
                "text": ["a", "b"],
                "text_2": ["c", "d"],
                "id": ["1", "1"],
                "time": ["1/1/2001 1:00:00", "1/1/2001 1:10:00"],
            },
            1,
        ),
    ],
)
def test_shapes_logdata_pipeline_two_columns(df_as_dict, expected_nrows_shape):
    pipeline, _ = generate_logdata_pipeline(
        target_text_column="text",
        device_id_column="id",
        timestamp_column_name="time",
        other_columns_to_concatenate=["text_2"],
        run_output_check=True,
    )
    pipeline = Pipeline(pipeline)

    df = pd.DataFrame().from_dict(df_as_dict)
    df["time"] = df["time"].astype(np.datetime64)

    pipeline.fit(df)
    new_df = pipeline.transform(df)

    assert new_df.shape[0] == expected_nrows_shape
