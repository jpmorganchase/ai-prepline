import re
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.pipeline import Pipeline

from ai_prepline.transformation.string_processing import (
    ApplyVectorizer,
    ConcatTextPerWindow,
    DeduplicateColumnText,
    ExplodeColumns,
    ExtractTextReGroups,
    FilterColumnByKeyword,
    MineForMessageTemplate,
    RemoveStopwords,
    SplitTexttoLists,
)
from ai_prepline.transformation.time_transforms import AddWindowIdentifier


@pytest.mark.parametrize(
    ["row_values", "expected"],
    [
        (
            [
                "a, b, c, d",
            ],
            ["a", "b", "c", "d"],
        ),
        (["a,"], ["a,"]),
    ],
)
def test_explode_single_column(row_values, expected, create_df):
    frame = create_df(row_values, col_name="text")

    splitter = SplitTexttoLists(
        [
            "text",
        ],
        ", ",
    )
    explode = ExplodeColumns()

    new_frame = explode.transform(splitter.transform(frame))

    assert list(new_frame.text.values) == expected


@pytest.mark.parametrize(
    ["row_values", "row_values2", "expected1", "expected2"],
    [
        (
            [
                "a, b, c",
            ],
            ["d, e, f"],
            ["a", "b", "c"],
            ["d", "e", "f"],
        ),
    ],
)
def test_explode_two_columns(row_values, row_values2, expected1, expected2, create_df):
    frame = create_df(row_values, col_name="text")
    frame["text2"] = row_values2

    splitter = SplitTexttoLists(
        ["text", "text2"],
        ", ",
    )

    explode = ExplodeColumns()
    new_frame = explode.transform(splitter.transform(frame))

    assert list(new_frame.text.values) == expected1
    assert list(new_frame.text2.values) == expected2


@pytest.mark.parametrize(
    ["row_values", "separator", "expected"],
    [
        (["a, a, a, a"], ", ", ["a"]),
        (["a, a, b, a"], ", ", ["a, b"]),
        (["a b a b"], ", ", ["a b a b"]),
        (["a b a b"], " ", ["a b"]),
        (["a b a b "], "a ", ["a b "]),
        (["a b a b"], "a ", ["a b a b"]),
    ],
)
def test_deduplicate_transform(row_values, separator, expected, create_df):
    frame = create_df(row_values=row_values, col_name="text")
    deduplicator = DeduplicateColumnText(
        text_column_to_deduplicate="text",
        separator=separator,
        deduplicated_column_name="text",
    )
    new_frame = deduplicator.transform(frame)

    assert list(new_frame.text.values) == expected


@pytest.mark.parametrize(
    ["row_values", "separator", "expected"],
    [
        (["a, a, a, a"], ", ", ["a"]),
        (["a, a, b, a"], ", ", ["a, b"]),
        (["a b a b"], ", ", ["a b a b"]),
        (["a b a b"], " ", ["a b"]),
        (["a b a b "], "a ", ["a b "]),
        (["a b a b"], "a ", ["a b a b"]),
    ],
)
def test_deduplicate_filter_by_clms_transform(
    row_values, separator, expected, create_df
):
    frame = create_df(row_values=row_values, col_name="text")
    frame["id"] = 1
    frame = pd.concat([frame, frame], axis=0)

    deduplicator = DeduplicateColumnText(
        text_column_to_deduplicate="text",
        separator=separator,
        filter_by_columns=["id"],
        deduplicated_column_name="text_dedup",
    )
    new_frame = deduplicator.transform(frame).drop_duplicates()
    assert list(new_frame.text_dedup.values) == expected


@pytest.mark.parametrize(
    ["row_values", "separator", "statistics", "expected"],
    [
        (["a, a, a, a"], ", ", [len], ["4, a"]),
        (["a, a, a, a"], ",", [len], ["4, a, a"]),
        (["a, a"], "#", [len], ["a, a"]),
        (["a a"], " ", [len], ["2, a"]),
        (["a b"], " ", [len], ["2, a b"]),
        (["a b"], " ", [len, lambda x: 1], ["2, 1, a b"]),
    ],
)
def test_deduplicate_transform_with_len_stat(
    row_values, separator, statistics, expected, create_df
):
    frame = create_df(row_values=row_values, col_name="text")
    deduplicator = DeduplicateColumnText(
        text_column_to_deduplicate="text",
        deduplicated_column_name="text",
        separator=separator,
        prepend_statistics=statistics,
    )
    new_frame = deduplicator.transform(frame)

    assert list(new_frame.text.values) == expected


def test_vectorizer_transform(create_df):
    test_df = create_df(["row1", "row2"], "Col")
    # mocks
    vectorizer = MagicMock()
    vectorizer.transform.return_value = pd.DataFrame(
        {"feature1": ["mockf1"], "feature2": ["mockf2"]}
    )

    vecTransformer = ApplyVectorizer(vectorizer, "Col")
    transformed = vecTransformer.transform(test_df)
    assert_frame_equal(
        transformed, pd.DataFrame({"feature1": ["mockf1"], "feature2": ["mockf2"]})
    )


def test_vectorizer_transform_column_not_exists(create_df):
    with pytest.raises(KeyError):
        test_df = create_df(["row1", "row2"])
        vectorizer = MagicMock()
        vectorizer.transform.return_value = pd.DataFrame(
            {"feature1": ["mockf1"], "feature2": ["mockf2"]}
        )

        vecTransformer = ApplyVectorizer(vectorizer, "A non existent column")
        _ = vecTransformer.transform(test_df)


@pytest.mark.parametrize(
    ["stop_words", "strings", "remove_special_symbols", "expected"],
    [
        (["a"], ["a"], [False], [""]),
        (["a"], ["a b"], False, ["b"]),
        (["a"], ["ab"], False, ["ab"]),
        (
            ["a", "the"],
            ["a test for removing the stopwords"],
            False,
            ["test for removing stopwords"],
        ),
        (
            [""],
            ["Should $ not - include @ special ? symbols"],
            True,
            ["Should not include special symbols"],
        ),
    ],
)
def test_stop_words(stop_words, strings, remove_special_symbols, expected, create_df):
    test_df = create_df(strings)
    transformer = RemoveStopwords(
        column="Col",
        stop_words=set(stop_words),
        remove_special_symbols=remove_special_symbols,
    )
    transformed = transformer.transform(test_df)
    assert transformed.equals(pd.DataFrame({"Col": expected}))


@pytest.mark.parametrize(
    ["stop_words", "row_val", "expected"],
    [([], ["a#!$!"], ["a"]), (["hi"], ["hi @"], [""]), ([], ["hi @"], ["hi "])],
)
def test_remove_special_chars(stop_words, row_val, expected):
    test_df = pd.DataFrame({"text": row_val})
    transformer = RemoveStopwords(
        stop_words=stop_words, column="text", remove_special_symbols=True
    )
    transformed = transformer.transform(test_df)
    assert transformed.equals(pd.DataFrame({"text": expected}))


def test_stop_words_column_not_exists(create_df):
    with pytest.raises(KeyError):
        test_df = create_df(["row1", "row2"])
        transformer = RemoveStopwords(
            stop_words=[], column="A non existent column", remove_special_symbols=True
        )
        _ = transformer.transform(test_df)


@pytest.mark.parametrize(
    ["list_of_strings", "expected"],
    [
        (["<:IP:>"], [""]),
        (["John went to <:*:>"], ["John went to"]),
        (
            ["The connection on port <:PORT:> took <:*:> ms"],
            ["The connection on port took ms"],
        ),
    ],
)
def test_remove_drain3_masks(list_of_strings, expected, create_df):
    test_df = create_df(list_of_strings)
    # creating a test_target column so we have a text column to index
    test_df["test_target"] = [""]
    miner = MagicMock()
    miner.add_log_message.return_value = {"template_mined": list_of_strings[0]}
    transformer = MineForMessageTemplate(
        miner,
        template_column="test_template",
        target_text_column="test_target",
        remove_drain3_masks=True,
    )
    transformer.transform(test_df)
    assert list(test_df["test_template"].values) == expected


@pytest.mark.parametrize(
    ["list_of_strings", "template", "expected"],
    [
        (
            ["CROND[27824]: (syscheck) CMD (/usr/bin/system_check -q)"],
            (1, "CROND<::>: (syscheck) CMD (/usr/bin/system_check -q)"),
            ["27824"],
        ),
        (
            ["CROND: (syscheck) CMD (/usr/bin/system_check -q)"],
            (1, "CROND: (syscheck) CMD (/usr/bin/system_check -q)"),
            [""],
        ),
        (
            [
                'logger[14205]: [ssl_acc] 204.27.33.248 \
                    - - [01/Apr/2023:03:14:54 +0000] "/index.php" 302 199'
            ],
            (
                1,
                "logger<::>: [ssl_acc] <:IP:> - - [<:NUM:>/Apr/<:NUM:>:\
                    <:NUM:>:<:NUM:>:<:NUM:> <:NUM:>] <:*:> <:NUM:> <:*:>",
            ),
            [
                "14205",
                "204.27.33.248",
                "01",
                "2023",
                "03",
                "14",
                "54",
                "+0000",
                """/index.php""",
                "302",
                "199",
            ],
        ),
    ],
)
def test_add_drain_template_variables(list_of_strings, template, expected, create_df):
    test_df = create_df(list_of_strings)
    miner = MagicMock()
    miner.add_log_message.return_value = {
        "template_mined": template[1],
        "cluster_id": template[0],
    }
    miner.extract_parameters.return_value = [(item,) for item in expected]

    transformer = MineForMessageTemplate(
        miner,
        template_column="test_template",
        target_text_column="Col",
        add_drain_variables=True,
    )
    transformer.transform(test_df)

    # Check if varaibles returned are as expected
    assert test_df["message_template_variable"].values[0] == expected

    # Check if funciton calls works with correct arguments
    miner.add_log_message.assert_called_once_with(list_of_strings[0])
    miner.extract_parameters.assert_called_once_with(template[1], list_of_strings[0])


@pytest.mark.parametrize(
    ["clm1", "clm2", "expected"],
    [(["1", "1"], ["a", "b"], ["a, b"])],
)
def test_concat_not_in_place(clm1, clm2, expected):
    frame = pd.DataFrame({"id": clm1, "text": clm2})
    grouper = ConcatTextPerWindow(
        group_by_clms=["id"],
        text_column_to_concatenate="text",
        concatenated_column_name="test",
        drop_duplicates=True,
    )

    result = grouper.transform(frame)
    assert list(result["test"].values) == expected
    assert not result.equals(frame)


@pytest.mark.parametrize(
    ["clm1", "clm2", "expected"],
    [(["1", "1"], ["1/1/2023", "2/1/2024"], ["2023-01-01, 2024-01-02"])],
)
def test_concat_datetime(clm1, clm2, expected):
    frame = pd.DataFrame({"id": clm1, "dates": clm2})
    frame.dates = pd.to_datetime(frame.dates, format="%d/%m/%Y")

    grouper = ConcatTextPerWindow(
        group_by_clms=["id"],
        text_column_to_concatenate="dates",
        concatenated_column_name="date_concat",
        drop_duplicates=True,
    )

    result = grouper.transform(frame)
    assert list(result["date_concat"].values) == expected
    assert not result.equals(frame)


@pytest.mark.parametrize(
    ["clm1", "clm2", "clm3", "expected"],
    [
        (["1", "1"], ["a", "b"], ["c", "d"], (["a, b"], ["c, d"])),
        (["1", "1"], ["a", "b"], [" ", "d"], (["a, b"], [" , d"])),
    ],
)
def test_concat_multicol(clm1, clm2, clm3, expected):
    frame = pd.DataFrame({"id": clm1, "text": clm2, "dummy": clm3})

    grouper = ConcatTextPerWindow(
        group_by_clms=["id"],
        text_column_to_concatenate=["text", "dummy"],
        concatenated_column_name=["test", "dummy_test"],
        drop_duplicates=True,
    )

    result = grouper.transform(frame)
    assert list(result["test"].values) == expected[0]
    assert list(result["dummy_test"].values) == expected[1]
    assert not result.equals(frame)


@pytest.mark.parametrize(
    ["clm1", "clm2", "expected"],
    [(["1", "1"], ["a", "b"], ["a, b"]), (["1", "2"], ["a", "b"], ["a", "b"])],
)
def test_grouping(clm1, clm2, expected):
    frame = pd.DataFrame({"id": clm1, "text": clm2})
    grouper = ConcatTextPerWindow(
        group_by_clms=["id"],
        text_column_to_concatenate="text",
        concatenated_column_name="test",
        drop_duplicates=True,
    )

    result = grouper.transform(frame)
    assert list(result.drop_duplicates(subset=["test"])["test"]) == expected


@pytest.mark.parametrize(
    ["clm1", "clm2", "expected"],
    [
        (["1/1/2001", "2/1/2001"], ["a", "b"], ["a", "b"]),
        ((["1/1/2001", "1/1/2001"], ["a", "b"], ["a, b"])),
        ((["1/1/2001 00:00:00", "1/1/2001 00:00:00"], ["a", "b"], ["a, b"])),
    ],
)
def test_grouping_dated(clm1, clm2, expected):
    frame = pd.DataFrame({"date": clm1, "text": clm2})
    frame.date = pd.to_datetime(frame.date)

    grouper = ConcatTextPerWindow(
        group_by_clms=["date"],
        text_column_to_concatenate="text",
        concatenated_column_name="test",
        drop_duplicates=True,
    )

    result = grouper.transform(frame).drop_duplicates(subset=["test"])
    assert list(result["test"]) == expected


@pytest.mark.parametrize(
    ["dates", "stride", "window_size", "expected_id"],
    [
        (["1/1/2022 1:00", "1/1/2022 2:00"], "1h", "1h", [0, 0, 1]),
        (["1/1/2022 1:00", "1/1/2022 2:00"], "61min", "1h", [0, 0]),
        (["1/1/2022 1:00", "1/1/2022 2:00"], "1h", "59min", [0, 1]),
        (["1/1/2022 1:00", "1/1/2022 2:10"], "55min", "59min", [0, 1]),
    ],
)
def test_add_window_id(dates, stride, window_size, expected_id, create_df):
    test_df = create_df(dates, "time")
    test_df.time = pd.to_datetime(test_df.time)
    window_id = AddWindowIdentifier(
        stride=stride, window_size=window_size, timestamp_column_name="time"
    )
    window_id.fit(test_df)
    test_df = window_id.transform(test_df)
    assert test_df.window.to_list() == expected_id


@pytest.mark.parametrize(
    ["dates", "stride", "window_size"],
    [
        (["1/1/2022 1:00", "1/1/2022 2:00"], "1h", "1h"),
        (["1/1/2022 1:00", "1/1/2022 2:00"], "61min", "1h"),
        (["1/1/2022 1:00", "1/1/2022 2:00"], "1h", "59min"),
        (["1/1/2022 1:00", "1/1/2022 2:10", "1/1/2022 3:10"], "60min", "60min"),
    ],
)
def test_assert_window_contains_x(dates, stride, window_size, create_df):
    test_df = create_df(dates, "time")
    test_df.time = pd.to_datetime(test_df.time)
    window_id = AddWindowIdentifier(
        stride=stride, window_size=window_size, timestamp_column_name="time"
    )
    window_id.fit(test_df)
    test_df = window_id.transform(test_df)

    check_all_windows_are_valid = (test_df.time >= test_df.start) & (
        test_df.time <= test_df.end
    )
    assert check_all_windows_are_valid.all()


def test_remove_drain3_masks_column_not_exists(create_df):
    with pytest.raises(KeyError):
        # Do not add a "test_target" column
        test_df = create_df(["row1", "row2"])
        miner = MagicMock()
        transformer = MineForMessageTemplate(
            miner,
            template_column="test_template",
            target_text_column="A non existent column",
            remove_drain3_masks=True,
        )
        _ = transformer.transform(test_df)


@pytest.mark.parametrize(
    ["stop_words", "expected"],
    [
        (
            ["a", "to", "the"],
            [
                ["Sally went store", "John went store", "Sam went store"],
                ["<:*:> went store", "<:*:> went store", "<:*:> went store"],
            ],
        )
    ],
)
def test_stop_words_drain_pipeline(stop_words, expected, create_df):
    test_df = create_df(
        ["Sally went to the store", "John went to the store", "Sam went to the store"],
        "remove_stopwords_then_template_column",
    )

    transformer_stopWords = RemoveStopwords(
        "remove_stopwords_then_template_column", stop_words
    )

    miner = MagicMock()
    miner.add_log_message.return_value = {"template_mined": "<:*:> went store"}
    transformer_drain3 = MineForMessageTemplate(
        miner, "remove_stopwords_then_template_column"
    )

    pipe = Pipeline(
        [
            ("stopWordRemover", transformer_stopWords),
            ("templateMiner", transformer_drain3),
        ]
    )
    transformed = pipe.fit_transform(test_df)

    assert transformed.equals(
        pd.DataFrame(
            {
                "remove_stopwords_then_template_column": expected[0],
                "message_template": expected[1],
            }
        )
    )


@pytest.mark.parametrize("sample", [("Jane Doe", "Robin Hood", "Arthur")])
@pytest.mark.parametrize(
    ["pattern", "extracted_column_names", "fillna_value", "err", "expected"],
    [
        (
            r"(\w+) ?(\w+)?",
            ["first", "last"],
            np.nan,
            does_not_raise(),
            {"first": ["Jane", "Robin", "Arthur"], "last": ["Doe", "Hood", np.nan]},
        ),
        (
            "(?P<first>\\w+) ?(?P<last>\\w+)?",
            None,
            "Unknown",
            does_not_raise(),
            {
                "first": ["Jane", "Robin", "Arthur"],
                "last": ["Doe", "Hood", "Unknown"],
            },
        ),
        (
            re.compile("(?P<first>\\w+) ?(?P<last>\\w+)?"),
            None,
            "",
            does_not_raise(),
            {
                "first": ["Jane", "Robin", "Arthur"],
                "last": ["Doe", "Hood", ""],
            },
        ),
        (r"(\w+) ?(\w+)?", None, np.nan, pytest.raises(ValueError), {}),
    ],
)
def test_extract_re_group_transform(
    sample, pattern, extracted_column_names, fillna_value, err, expected, create_df
):
    data = create_df(sample, col_name="test")
    expected = {"test": sample, **expected}

    with err:
        extract_re = ExtractTextReGroups(
            "test",
            pattern,
            fillna_value=fillna_value,
            extracted_column_names=extracted_column_names,
        )
        pipe = Pipeline([("extractRe", extract_re)])
        transformed = pipe.fit_transform(data)
        assert_frame_equal(transformed, pd.DataFrame(expected))


@pytest.mark.parametrize(
    ["sample", "keywords", "expected", "err"],
    [
        (["This is a Test", "One more"], ["is"], ["One more"], does_not_raise()),
        (
            ["This is a Test", "One more"],
            [],
            ["This is a Test", "One more"],
            pytest.raises(ValueError),
        ),
        (
            ["This is a Test", "One more"],
            ["not", "in"],
            ["This is a Test", "One more"],
            does_not_raise(),
        ),
        (
            ["This is a test for lowercase", "One more"],
            ["LOWERCASE"],
            ["One more"],
            does_not_raise(),
        ),
        (
            ["Test more we say!", "testing is fun", "One more"],
            ["test"],
            ["One more"],
            does_not_raise(),
        ),
        (
            [r"I contain regex expression \w!", "I do not", r"So do I [a-Z]+"],
            [r"\w", "Z]+"],
            ["I do not"],
            does_not_raise(),
        ),
    ],
)
def test_filter_by_keyword(sample, keywords, expected, err, create_df):
    data = create_df(sample, col_name="test")
    with err:
        transformed = FilterColumnByKeyword("test", keywords).fit_transform(data)
        np.testing.assert_equal(transformed["test"].to_list(), expected)
