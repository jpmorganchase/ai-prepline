# pylint: disable=duplicate-code
from typing import Any, List

import numpy as np
import pandas as pd
import pandera as pa
import pytest
from sklearn.pipeline import Pipeline

from ai_prepline.data_checks.data_check import column_contains_datetime
from ai_prepline.pipeline_generators.f5_logfile_pipelines import (
    create_input_output_pandera_checks,
    generate_f5_logdata_pipeline,
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
        (["1/1/2022"], "1/1/2022", "1/1/2022", "datetime64[ns, UTC]", True),
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
        "expected",
    ],
    [
        (
            {
                "text": ["CROND[27824]: (syscheck) CMD (/usr/bin/system_check -q)"],
                "id": ["1"],
                "time": ["1/1/2001"],
                "level": ["critical"],
                "message_template": [
                    "CROND<::>: (syscheck) \
                                     CMD (/usr/bin/system_check -q)"
                ],
                "message_template_id": [1],
                "message_template_variable": ["[27824]"],
            },
            True,
            True,
        ),
        (
            {
                "text": ["CROND[27824]: (syscheck) CMD (/usr/bin/system_check -q)"],
                "id": ["1"],
                "time": ["1/1/2001"],
                "level": ["critical"],
                "message_template": [
                    "CROND<::>: (syscheck) \
                                     CMD (/usr/bin/system_check -q)"
                ],
                "message_template_id": [1],
                "message_template_variable": ["[27824]"],
            },
            False,
            False,
        ),
        (
            {
                "text": ["CROND[27824]: (syscheck) CMD (/usr/bin/system_check -q)"],
                "id": ["1"],
                "time": ["1/1/2001"],
                "level": ["error"],
                "message_template": [
                    "CROND<::>: (syscheck) \
                                     CMD (/usr/bin/system_check -q)"
                ],
                "message_template_id": [0],
                "message_template_variable": ["[27824]"],
            },
            True,
            False,
        ),
    ],
)
def test_create_input_output_pandera_checks(df_as_dict, convert_to_datetime, expected):
    df = pd.DataFrame().from_dict(df_as_dict)

    if convert_to_datetime:
        df["time"] = df["time"].astype(np.datetime64)

    checks = prepare_datetime_checks(max_date_check=None, min_date_check=None)
    checks.append(pa.Check(column_contains_datetime))
    input_schema, output_schema = create_input_output_pandera_checks(
        "text",
        "id",
        "time",
        "level",
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


def test_generate_f5_logdata_pipeline_init():
    pipeline_list = generate_f5_logdata_pipeline(
        target_text_column="msg",
        device_id_column="device",
    )

    assert len(pipeline_list) > 0


@pytest.mark.parametrize(
    [
        "df_as_dict",
        "expected_ncols_shape",
    ],
    [
        (
            {
                "text": ["a"],
                "id": ["1"],
                "time": ["1/1/2001"],
                "level": ["info"],  # Optional column
            },
            7,
        ),
        (
            {
                "text": ["CROND[27824]: (syscheck) CMD (/usr/bin/system_check -q)"],
                "id": ["1"],
                "time": ["1/1/2022"],
            },
            6,
        ),
    ],
)
def test_shapes_f5_logdata_pipeline(df_as_dict, expected_ncols_shape):
    pipeline = generate_f5_logdata_pipeline(
        target_text_column="text",
        device_id_column="id",
        timestamp_column_name="time",
        run_input_check=False,
        run_output_check=False,
    )
    pipeline = Pipeline(pipeline)

    df = pd.DataFrame().from_dict(df_as_dict)
    df["time"] = df["time"].astype(np.datetime64)
    pipeline.fit(df)
    new_df = pipeline.transform(df)
    assert new_df.shape[1] == expected_ncols_shape
