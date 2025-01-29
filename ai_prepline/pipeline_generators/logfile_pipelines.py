"""
Example pipeline generators for working with log data.
"""

from abc import ABC
from typing import List, Optional, Tuple

import pandera as pa
from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from ai_prepline.data_checks.data_check import (
    CheckAgainstSchema,
    column_contains_datetime,
)
from ai_prepline.transformation import string_processing, time_transforms

DEFAULT_DRAIN3_CONFIG = "drain3.ini"


def generate_logdata_pipeline(
    target_text_column: str,
    device_id_column: str,
    timestamp_column_name: str = "time",
    other_columns_to_concatenate: Optional[List[str]] = None,
    drop_duplicates: bool = True,
    separator: str = ", ",
    stride: str = "1h",
    window_size: str = "1h",
    run_output_check: bool = True,
    strict_datacheck: bool = True,
    minimum_elements_per_window_check: int = 1,
    max_date_check: Optional[str] = None,
    min_date_check: Optional[str] = None,
    drain3_config_path: Optional[str] = None,
    drain3_file_persistence_path: Optional[str] = None,
) -> Tuple[List[Tuple[str, ABC]], ABC]:
    """
    Generate a pre-processing pipeline for use with log data.
    Use the list of transforms with sklearn.pipeline.Pipeline().

    :param target_text_column: column name that contains log_messages to process.
    :param device_id_column: column with the asset identifiers, e.g., IP or asset name.
    :param separator: separator to use when concatenating and deduplicating text.
    :param other_columns_to_concatenate: Optional list of other columns to concatenate
        according to the time windows and device_id.
    :param drop_duplicates: Optional bool, default == True. If false, we don't drop
        records that are duplicate according to "asset_id" and start of time-window.
        In other words, if the original frame had n records, we output n records.

    Date and timestamp-specific parameters::
    :param stride: gap between starting times of consecutive time windows.
    :param window_size: gap between the starting time and ending time of windows.
    :param timestamp_column_name: name of the column that contains the timestamps.
        Should be datetime data. Used by the AddWindowIdentifier step.


    Pandera data-check parameters::
     :param minimum_elements_per_window_check: int, parameter for pandera, raises an
        exception if there are less than this elements per window.
    :param max_date_check: str, max date that should appear in data.
        This will raise an assertion if false.
    :param min_date_check: str, min date that should appear in data.
        This will raise an assertion if false.
    :param run_output_check: bool, whether to run the final data check on output.
    :param strict_datacheck: bool, whether to raise an exception on pandera failures.


    Drain3-specific parameters::
    :param drain3_config_path: str, path to load the drain3.ini configuration file.
        This file should contain the regex patterns that drain3 will use to mask
        identifiers.
    :param drain3_file_persistence_path: str, where drain3 can store masking data
        while it works. Those are only used by drain3.

    Returns a List[str, BaseTransform] which can be used with sklearn Pipeline and
        pandera data-schema that can be used to check properties of the output.

    This pipeline generator also includes data checks by pandera.

    """
    other_columns_to_concatenate = other_columns_to_concatenate or []

    miner = prepare_drain3_miner(drain3_config_path, drain3_file_persistence_path)
    datetime_checks = prepare_datetime_checks(max_date_check, min_date_check)

    input_schema, output_schema = create_input_output_pandera_checks(
        target_text_column,
        device_id_column,
        timestamp_column_name,
        minimum_elements_per_window_check,
        datetime_checks,
        other_columns_to_concatenate,
    )

    check_input_schema = [
        (
            "check_input_schema",
            CheckAgainstSchema(input_schema, strict=strict_datacheck),
        ),
    ]

    mine_keywords_remove_masks = [
        (
            "miner",
            string_processing.MineForMessageTemplate(
                miner=miner,
                target_text_column=target_text_column,
                template_column="message_template",
                remove_drain3_masks=True,
            ),
        ),
        (
            "remove_stopwords",
            string_processing.RemoveStopwords(
                column="message_template",
                stop_words=set(),
                remove_special_symbols=True,
            ),
        ),
    ]

    # the "start" column comes from the time_transforms.AddWindowIdentifier step.
    # It holds the beginning of each time-window.
    group_by_clms = [device_id_column, "start"]

    add_time_window_identifiers = [
        (
            "add_window_identifier",
            time_transforms.AddWindowIdentifier(
                stride=stride,
                window_size=window_size,
                timestamp_column_name=timestamp_column_name,
            ),
        ),
    ]

    clms_to_concatenate = ["message_template"] + other_columns_to_concatenate
    names_for_concatenated_clms = [
        f"{clm_name}_concatenated" for clm_name in clms_to_concatenate
    ]

    concatenate_text_columns = [
        (
            "concatenate_message",
            string_processing.ConcatTextPerWindow(
                group_by_clms=group_by_clms,
                text_column_to_concatenate=clms_to_concatenate,
                concatenated_column_name=names_for_concatenated_clms,
                separator_for_concatenated_text=separator,
                drop_duplicates=drop_duplicates,
            ),
        ),
    ]

    deduplicate_message_template = [
        (
            "deduplicator",
            string_processing.DeduplicateColumnText(
                text_column_to_deduplicate=("message_template" + "_concatenated"),
                deduplicated_column_name="setfit_input",
                separator=separator,
                prepend_statistics=[len],
                filter_by_columns=group_by_clms,
            ),
        )
    ]

    check_against_output_schema = CheckAgainstSchema(
        output_schema, strict=strict_datacheck
    )

    # The full pipeline!
    pipeline_list = (
        check_input_schema
        + mine_keywords_remove_masks
        + add_time_window_identifiers
        + concatenate_text_columns
        + deduplicate_message_template
    )

    # NOTE: This step needs to be last in the pipeline.
    if run_output_check:
        pipeline_list += [
            (
                "output_schema_check",
                check_against_output_schema,
            ),
        ]

    return (
        pipeline_list,
        check_against_output_schema,
    )


def create_input_output_pandera_checks(
    target_text_column: str,
    device_id_column: str,
    timestamp_column_name: str,
    minimum_elements_per_window_check: int,
    datetime_checks,
    other_columns_to_concatenate: Optional[List[str]] = None,
) -> Tuple[pa.DataFrameSchema, pa.DataFrameSchema]:
    """
    Pandera-style DataFrameSchemas. Using schema.validate(df) we check the properties
    of dataframes.

    :param target_text_column: column name that contains log_messages to process.
    :param device_id_column: column with the asset identifiers, e.g., IP or asset name.
    :param separator: separator to use when concatenating and deduplicating text.
    :param minimum_elements_per_window_check: int, parameter for pandera, raises an
        exception if there are less than this elements per window.
    :param datetime_checks: List of checks for datetimes.
    :param other_columns_to_concatenate: Optional[List[str]]

    """

    other_columns_to_concatenate = other_columns_to_concatenate or []
    datetime_checks.append(pa.Check(column_contains_datetime))

    input_schema = {
        target_text_column: pa.Column(str),
        device_id_column: pa.Column(str),
        timestamp_column_name: pa.Column(None, checks=datetime_checks),
    }

    # comparing to 1-1-1900 is an indirect way to check
    # if an entry is a date without putting constraints
    # on whether a timezone exists, etc.
    output_schema = {
        target_text_column: pa.Column(str),
        "start": pa.Column(
            None,
            checks=[
                pa.Check(column_contains_datetime),
            ],
        ),
        "end": pa.Column(
            None,
            checks=[
                pa.Check(column_contains_datetime),
            ],
        ),
        "message_template_concatenated": pa.Column(str),
        "_count_elements_per_group": pa.Column(
            int,
            pa.Check.greater_than_or_equal_to(minimum_elements_per_window_check),
        ),
        "setfit_input": pa.Column(str),
    }

    # make sure any other columns we want to concatenate exist in the input and output

    def other_columns_to_concatenate_schema(postfix_string):
        return {
        (key + postfix_string): pa.Column(str) for key in other_columns_to_concatenate
    }

    input_schema = {**input_schema, **other_columns_to_concatenate_schema("")}
    output_schema = {
        **output_schema,
        **other_columns_to_concatenate_schema("_concatenated"),
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
        drain3_file_persistence_path or "drain_file_persistence.json"
    )

    persistence_handler = FilePersistence(drain3_file_persistence_path)

    miner = TemplateMiner(
        persistence_handler=persistence_handler, config=template_miner_config
    )

    return miner
