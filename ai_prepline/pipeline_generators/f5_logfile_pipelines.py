"""
Pipeline generators for working with log data.
"""

# pylint: disable=duplicate-code
from typing import Any, List, Optional, Tuple

import pandera as pa
from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from ai_prepline.data_checks.data_check import (
    CheckAgainstSchema,
    column_contains_datetime,
)
from ai_prepline.transformation import string_processing

DEFAULT_DRAIN3_CONFIG = "f5_drain3.ini"


def generate_f5_logdata_pipeline(
    target_text_column: str,
    device_id_column: str,
    timestamp_column_name: str = "timestamp",
    message_level: str = "level",
    run_input_check: bool = True,
    run_output_check: bool = True,
    strict_datacheck: bool = True,
    min_date_check: Optional[str] = None,
    max_date_check: Optional[str] = None,
    drain3_file_persistence_path: Optional[str] = None,
    drain3_config_path: Optional[str] = None,
) -> List[Tuple[str, Any]]:
    """
    Generate a pre-processing pipeline for use with log data.
    Use the list of transforms with sklearn.pipeline.Pipeline().

    :param target_text_column: column name that contains log_messages to process.
    :param device_id_column: column with the host/asset identifiers,
        e.g., IP or asset name.
    :param timestamp_column_name: name of the column that contains the timestamps,
        defaults to column name "timestamp"
    :param message_level: log level of the message, defaults to column name "level"
    :param run_input_check: whether to run the initial input data check,
        defaults to True
    :param run_output_check: whether to run the final data check on output,
        defaults to True
    :param strict_datacheck: whether to raise an exception on pandera failures,
        defaults to True
    :param min_date_check: str, min date that should appear in data.
        This will raise an assertion if false.
    :param max_date_check: max date that should appear in data.
        This will raise an assertion if false.
    :param drain3_file_persistence_path: str, where drain3 can store masking data
        while it works. Those are only used by drain3.
    :param drain3_config_path: str, path to load the drain3.ini configuration file.
        This file should contain the regex patterns that drain3 will use to mask
        identifiers.

    Returns a List[str, BaseTransform] which can be used with sklearn Pipeline and
        pandera data-schema that can be used to check properties of the output.
    """
    miner = prepare_drain3_miner(drain3_config_path, drain3_file_persistence_path)
    datetime_checks = prepare_datetime_checks(max_date_check, min_date_check)

    # Check against input schema
    input_schema, output_schema = create_input_output_pandera_checks(
        target_text_column,
        device_id_column,
        timestamp_column_name,
        message_level,
        datetime_checks,
    )

    check_input_schema = [
        (
            "check_input_schema",
            CheckAgainstSchema(input_schema, strict=strict_datacheck),
        ),
    ]

    # Mine for message templates
    mine_and_add_drain_data = [
        (
            "miner",
            string_processing.MineForMessageTemplate(
                miner=miner,
                target_text_column=target_text_column,
                remove_drain3_masks=False,
                add_drain_variables=True,
            ),
        ),
    ]

    # Check against output schema
    check_output_schema = [
        (
            "check_output_schema",
            CheckAgainstSchema(output_schema, strict=strict_datacheck),
        ),
    ]

    # NOTE: Order matters
    # Full pipeline
    pipeline_list = []

    if run_input_check:
        pipeline_list += check_input_schema

    pipeline_list += mine_and_add_drain_data

    if run_output_check:
        pipeline_list += check_output_schema

    return pipeline_list


def create_input_output_pandera_checks(
    target_text_column: str,
    device_id_column: str,
    timestamp_column_name: str,
    message_level: str,
    datetime_checks,
) -> Tuple[pa.DataFrameSchema, pa.DataFrameSchema]:
    """
    Pandera-style DataFrameSchemas. Using schema.validate(df) we check the properties
    of dataframes.

    :param target_text_column: column name that contains log_messages to process.
    :param device_id_column: column with the asset identifiers, e.g., IP or asset name.
    :param datetime_checks: List of checks for datetimes.
    :param timestamp_column_name: name of the column that contains the timestamps,
        defaults to column name "timestamp"
    :param message_level: log level of the message, defaults to column name "level"

    Returns a Tuple Pandera Schema objects that can be used to check
        properties of the output.
    """

    datetime_checks.append(pa.Check(column_contains_datetime))

    input_schema = {
        target_text_column: pa.Column(str),
        device_id_column: pa.Column(str),
        timestamp_column_name: pa.Column(None, checks=datetime_checks),
        message_level: pa.Column(str),
    }

    output_schema = {
        target_text_column: pa.Column(str),
        device_id_column: pa.Column(str),
        timestamp_column_name: pa.Column(None, checks=datetime_checks),
        message_level: pa.Column(str),
        "message_template": pa.Column(str),
        "message_template_id": pa.Column(int, pa.Check.in_range(1, 99999)),
        "message_template_variable": pa.Column(str),
    }

    input_schema = pa.DataFrameSchema(input_schema)
    output_schema = pa.DataFrameSchema(output_schema)

    return input_schema, output_schema


def prepare_datetime_checks(
    max_date_check: Optional[str] = None, min_date_check: Optional[str] = None
):
    """
    Prepare a series of checks for the timestamp column we use.
    """
    datetime_checks = []

    if max_date_check:
        datetime_checks.append(pa.Check.less_than_or_equal_to(max_date_check))

    if min_date_check:
        datetime_checks.append(pa.Check.greater_than_or_equal_to(min_date_check))
    return datetime_checks


def prepare_drain3_miner(
    drain3_config_path: Optional[str] = None,
    drain3_file_persistence_path: Optional[str] = None,
) -> TemplateMiner:
    """
    Prepares a drain3 miner.
    """
    drain3_config_path = drain3_config_path or DEFAULT_DRAIN3_CONFIG
    template_miner_config = TemplateMinerConfig()
    template_miner_config.load(drain3_config_path)

    drain3_file_persistence_path = (
        drain3_file_persistence_path or "drain_file_persistence.bin"
    )

    persistence_handler = FilePersistence(drain3_file_persistence_path)

    miner = TemplateMiner(
        persistence_handler=persistence_handler, config=template_miner_config
    )

    return miner
