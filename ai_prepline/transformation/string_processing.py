"""
A collection of handy transforms for dataframes that contain columns with text data.
These will play well with sklearn.pipeline.Pipeline, and can be run by themselves.
"""

import re
from typing import Any, Callable, Collection, List, Optional, Union

import pandas as pd
from drain3 import TemplateMiner

from ai_prepline.base.transform import BaseTransform

COLUMN_KEYWORD_FILTER_MIN_KW_LENGTH = 1


class DeduplicateColumnText(BaseTransform):
    """
    Deduplicate strings in column of dataframe. This transform respects word order.

    A string like "a, a, a, a" would be mapped to f"{prepend_statistics}, a".
    """

    def __init__(
        self,
        text_column_to_deduplicate: str,
        deduplicated_column_name: str,
        separator: Optional[str] = None,
        prepend_statistics: Optional[List[Callable[[List[str]], int]]] = None,
        filter_by_columns: Optional[List[str]] = None,
    ):
        """
        Initialize a DeduplicateColumnText object.

        :param text_column_to_deduplicate: name of column that we wish to deduplicate.
        :param deduplicated_column_name: name of column in which to store
            deduplicated text.
        :param separator: optional, separator for the different parts of a string.
        :param prepend_statistics: optional, functions to evaluate on each split
            string and prepend to the main string. By default no statistic
            is prepended.
        :param filter_by_columns: optional, list of columns to pick
             before deduplication happens.
            If used, a left join will take place at the end to attach the
                deduplicated column data.
        """

        self.text_column_to_deduplicate = text_column_to_deduplicate

        self.deduplicated_column_name = deduplicated_column_name

        self.separator = separator or ","

        self.prepend_statistics = prepend_statistics or []
        self.filter_by_columns = filter_by_columns or []

    def transform(self, X, y=None):
        clms = [self.text_column_to_deduplicate] + self.filter_by_columns

        Y = (
            X.drop_duplicates(subset=self.filter_by_columns)[clms]
            if len(self.filter_by_columns) > 0
            else X
        )

        Y[self.deduplicated_column_name] = Y[self.text_column_to_deduplicate].apply(
            self._deduplicate
        )

        if len(self.filter_by_columns) > 0:
            clms = self.filter_by_columns + [self.deduplicated_column_name]
            Y = X.merge(
                Y[clms],
                how="left",
                left_on=self.filter_by_columns,
                right_on=self.filter_by_columns,
            )

        return Y

    def _deduplicate(self, msg) -> str:
        if self.separator not in msg:
            return msg

        strings = msg.split(self.separator)
        statistics = [f"{fun(strings)}, " for fun in self.prepend_statistics]

        return "".join(statistics) + self.separator.join(
            sorted(set(strings), key=strings.index)
        )


class SplitTexttoLists(BaseTransform):
    """
    For every element of a str column, splits the text into a list
    of the str elements that are separated by the separator.

    Example: "a, b, c" with separator ", " will become
        ["a", "b", "c"].
    """

    def __init__(self, column_names: List[str], separator=", "):
        self.column_names = column_names
        self.separator = separator

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X.loc[:, self.column_names] = X.loc[:, self.column_names].apply(
            lambda x: x.str.split(self.separator), axis=1
        )
        return X


class ExplodeColumns(BaseTransform):
    """
    Apply .explode() to all columns containing lists.
    """

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.apply(pd.Series.explode)


class ApplyVectorizer(BaseTransform):
    """
    ApplyVectorizer takes a BaseTransformer instance and applies the
    transformation to the column.
    """

    def __init__(self, vectorizer: BaseTransform, column: str):
        """
        Initialize an ApplyVectorizer transform object

        :param vectorizer: vectorizer to apply
        :param column: column on which the vectorizer will be applied
        """
        self.vectorizer = vectorizer
        self.column = column

    def transform(self, X: pd.DataFrame, y: Optional[list] = None) -> pd.DataFrame:
        """
        Returns the text features w.r.t to a specific vectorizer, e.g., CountVectorizer,
        this is useful when we want to add all operations in the same sklearn pipeline.

        :param X: the dataframe to transform
        :param y: default None as this is not a classifier
        :return: The transformed text features
        """
        self.vectorizer.fit(X[self.column])
        text_features = self.vectorizer.transform(X[self.column])
        return text_features


class RemoveStopwords(BaseTransform):
    """
    Removes stop words, drain3 masks, and special symbols
    """

    def __init__(
        self,
        column: str,
        stop_words: Collection,
        remove_special_symbols: bool = False,
    ):
        """
        Initialize a RemoveStopwords transform object

        :param column: text column to remove stop words from
        :param stop_words: set of stop words we will remove from the column
        :param remove_special_symbols: if true remove non-alphanumeric symbols
        """
        self.column = column
        self.stop_words = stop_words
        self.remove_special_symbols = remove_special_symbols

    def transform(self, X: pd.DataFrame, y: Optional[list] = None) -> pd.DataFrame:
        """
        Remove stopwords from a column in place and return the transformed dataframe

        :param X: dataframe to remove stop words from
        :param y: default None as this is not a classifier
        :return: transformed dataframe
        """
        if self.remove_special_symbols:
            X[self.column] = X[self.column].str.replace(
                "[^.,a-zA-Z0-9 ]", "", regex=True
            )

        if len(self.stop_words) > 0:
            X[self.column] = X[self.column].apply(self.__remove_stop_words)

        return X

    def __remove_stop_words(self, s: str) -> str:
        """Removes stop words from a string"""
        sentence_with_words_removed = [
            # the split() method with no arguments splits on any white space
            word
            for word in s.split()
            if word.lower() not in self.stop_words
        ]
        return " ".join(sentence_with_words_removed)


class MineForMessageTemplate(BaseTransform):
    """
    Transformer that uses a drain3 template miner to create templates out of
    raw log messages.

    A template is text where things like IP addresses, paths, device
    names, etc., are masked. drain3 will look for a drain3.ini configuration file with
    options for creation of the template. One such example file exists in config/

    Instances of this class require a TemplateMiner from drain3. A simple way to
    set this up is:

    ```python
    from drain3.file_persistence import FilePersistence
    fp = FilePersistence(DATA_PATH / "drain.json")
    miner = TemplateMiner(persistence_handler=fp)

    MineForMessageTemplate(miner = miner, target_text_column = your_text_column)
    ```

    If you want to load your own drain3 config file, a .ini file,
    then a good way is:

    ```python
    from drain3.file_persistence import FilePersistence
    from drain3.template_miner_config import TemplateMinerConfig

    fp = FilePersistence(DATA_PATH / "drain.json")
    drain3_config = TemplateMinerConfig()
    drain3_config.load('yourfile.ini')
    miner = TemplateMiner(persistence_handler=fp, config=drain3_config)

    MineForMessageTemplate(miner = miner, target_text_column = your_text_column)
    ```

    To add more metadata from drain miner. E.g. Template ID and Variables for each template, use add_drain_variables flag.

    Example:
    >>> MineForMessageTemplate(
        miner = miner, target_text_column = your_text_column,
        target_id_column = your_id_column, target_var_column = your_variable_column,
        add_drain_variables=True)

    This will add two columns to the dataframe, template_id and template_vars.
    Template variables are extracted by generating a regex pattern from the template.
    The regex is then matched with the original message to get the variables.
    Template ID is the ID of the message template retuned by drain which can be used later for modeling purposes.

    NOTE: If no config is provided, default options will be loaded,
    see https://github.com/logpai/Drain3/blob/e0e942886845315ec4eac8b8de68859d9e106908/drain3/template_miner.py#L36 # pylint: disable=line-too-long

    """

    def __init__(
        self,
        miner: TemplateMiner,
        target_text_column: str,
        template_column: str = "message_template",
        template_id_column: str = "message_template_id",
        template_var_column: str = "message_template_variable",
        add_drain_variables: bool = False,
        remove_drain3_masks: bool = False,
    ):
        """
        Initialize a MineForMessageTemplate object

        :param miner: a drain3 template miner object
        :param target_text_column: column that contains messages we want to mine
        :param template_column: name of column to store the templated messages to
        :param template_id_column: name of column to store the id of templated
                messages to
        :param template_var_column: name of column to store the
                varaibles of templated messages to
        :param add_drain_variables: If True, adds template id
                and variables for each message template
        :param remove_drain3_masks: if true, removes the drain3 masks in the template
        """
        self.miner = miner
        self.target_text_column = target_text_column
        self.template_column = template_column
        self.template_id_column = template_id_column
        self.template_var_column = template_var_column
        self.add_drain_variables = add_drain_variables
        self.remove_drain3_masks = remove_drain3_masks

    def transform(self, X: pd.DataFrame, y: Optional[list] = None) -> pd.DataFrame:
        """
        Creates a drain3 template for the each item in a text column and optionally
        we remove drain3 mask depending on the `self.remove_drain3_masks` flag

        :param X: dataframe to transform
        :param y: default None as this is not a classifier
        :return: transformed dataframe
        """

        message_templates = []
        message_ids = []
        message_variables = []

        for message in X[self.target_text_column]:
            res = self.miner.add_log_message(message)
            message_templates.append(res["template_mined"])
            if self.add_drain_variables:
                message_ids.append(res["cluster_id"])
                m_vars = self.miner.extract_parameters(res["template_mined"], message)
                message_variables.append(
                    [re.sub(r"[\[\]]", "", v[0]) for v in m_vars]
                    if m_vars is not None
                    else []
                )

        # TODO: transform() changes the dimension of the input, rethink
        X[self.template_column] = message_templates
        if self.remove_drain3_masks:
            X[self.template_column] = (
                X[self.template_column]
                .str.replace(r"\<:.+?:\>", "", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
        if self.add_drain_variables:
            X[self.template_id_column] = message_ids
            X[self.template_var_column] = message_variables

        return X


class ConcatTextPerWindow(BaseTransform):
    """
    This transform attaches an extra column with a concatenation of text in each group.
    The groups are defined according to the group_by_clms.

    This transformation is useful when we want to aggregate text per time_windows or
    other variables.

    .transform() returns the a new dataframe with the additional column
    for the concatenated text. Operation is not done in-place.
    """

    def __init__(
        self,
        group_by_clms: List[str],
        text_column_to_concatenate: Union[str, List[str]],
        concatenated_column_name: Union[str, List[str]],
        separator_for_concatenated_text: str = ", ",
        drop_duplicates: bool = True,
    ):
        """
        :param group_by_clms: list of dataframe columns that we want to group by
        :param text_column_to_concatenate: column(s) with text to concatenate
        :param name_for_new_column_with_concatenated_text: the name(s) for
            the column(s) in which to store the concatenated text

        :param separator_for_concatenated_text: what character to use to separate the
            different parts of the concatenated text
        :param drop_duplicates: whether to keep only the first example for every
            combination of group_by_clms
        """
        self.group_by_clms = group_by_clms
        self.text_column_to_concatenate = self._listify(text_column_to_concatenate)

        self.concatenated_column_name = self._listify(concatenated_column_name)

        self.separator_for_concatenated_text = separator_for_concatenated_text
        self.drop_duplicates = drop_duplicates

    @staticmethod
    def _listify(element):
        return element if isinstance(element, list) else [element]

    def transform(self, X: pd.DataFrame, y: Optional[list] = None) -> pd.DataFrame:
        for text_col_to_concatenate, concatenated_col_name in zip(
            self.text_column_to_concatenate, self.concatenated_column_name
        ):
            text_concat_per_window = (
                X.astype({text_col_to_concatenate: str})
                .groupby(self.group_by_clms)[text_col_to_concatenate]
                .apply(f"{self.separator_for_concatenated_text}".join)
                .reset_index()
            )

            text_concat_per_window.rename(
                columns={text_col_to_concatenate: concatenated_col_name},
                inplace=True,
            )

            X = X.merge(
                text_concat_per_window,
                left_on=self.group_by_clms,
                right_on=self.group_by_clms,
                how="left",
            )

        X = X.merge(
            X.groupby(self.group_by_clms).size().reset_index(),
            left_on=self.group_by_clms,
            right_on=self.group_by_clms,
            how="left",
        ).rename(columns={0: "_count_elements_per_group"})

        # NOTE: do not attempt to do in-place, causes a bug with .fit()
        return (
            X.drop_duplicates(subset=self.group_by_clms, keep="first")
            if self.drop_duplicates
            else X
        )


class ExtractTextReGroups(BaseTransform):
    # pylint: disable=anomalous-backslash-in-string

    """
    Extracts new text columns from a source column using regex patterns.
    This transformation performs in_place and supports capturing or named Perl (?...)
    pattern extension syntax.

    Example:
        >>> data = pd.DataFrame({'users': ['Jane Doe', 'Robin Hood', 'Arthur]})
        >>> t = ExtractTextReGroups(
            text_column_to_extract_from='users',
            pattern = re.compile('(?P<first>\\w+) ?(?P<last>\\w+)?')
            )
        >>> t.transform(data)
                users  first  last
        0    Jane Doe   Jane   Doe
        1  Robin Hood  Robin  Hood
        2      Arthur  Arthur

    If there is a failed match missing values are imputed with `fillna_value` which is
    set to empty string by default. If you are not using named groups just pass the
    `extracted_column_names` param with the names for the extracted columns.

    Example:
        >>> data = pd.DataFrame({'users': ['Jane Doe', 'Robin Hood', 'Arthur']})
        >>> t = ExtractTextReGroups(
            text_column_to_extract_from='users',
            pattern = '(\w+) ?(\w+)?',
            extracted_column_names=['First name', 'Last name'],
            fillna_value='UNKNOWN'
            )

                users First name Last name
        0    Jane Doe       Jane       Doe
        1  Robin Hood      Robin      Hood
        2      Arthur     Arthur   UNKNOWN
    """

    def __init__(
        self,
        text_column_to_extract_from: str,
        pattern: Union[str, re.Pattern],
        fillna_value: Optional[Any] = "",
        extracted_column_names: Optional[str] = None,
    ):
        """
        Initialise the transformation

        :param text_column_to_extract_from: source column name to extract from
        :param pattern: regex pattern to be used
        :param fillna_value: value used to impute any missing matches, defaults to ""
        :param extracted_column_names: if named groups are not in the pattern this param
            must be provided with the names of the columns to be created from extracted
            matches, defaults to None
        """
        self.text_column_to_extract_from = text_column_to_extract_from
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.fillna_value = fillna_value

        self.extracted_column_names = (
            list(self.pattern.groupindex.keys()) or extracted_column_names
        )

        if not self.extracted_column_names:
            raise ValueError("No target columns to extract!")

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X[self.extracted_column_names] = X[
            self.text_column_to_extract_from
        ].str.extract(self.pattern)
        X[self.extracted_column_names] = X[self.extracted_column_names].fillna(
            self.fillna_value
        )
        return X


class FilterColumnByKeyword(BaseTransform):
    """
    Filter out rows where target column values contain one of the provided keywords
    This transformation is case insensitive
    """

    def __init__(
        self,
        target_column_to_filter: str,
        keywords_to_filter_out: List[str],
    ):
        """
        Initialize the transformation

        :param target_column_to_filter: column against to match the filter
        :param keywords_to_filter_out: list of keyword values which should be excluded
        :param split_re_expression: how to split the column value for keyword check
            if matched in the `target_column_to_filter`
        """

        keywords = [kw.lower().strip() for kw in keywords_to_filter_out if kw]
        if len(keywords) < COLUMN_KEYWORD_FILTER_MIN_KW_LENGTH:
            raise ValueError(
                f"At least {COLUMN_KEYWORD_FILTER_MIN_KW_LENGTH} "
                "valid keyword must be present in the keyword list"
            )

        self.target_column_to_filter = target_column_to_filter
        self.keywords_to_filter_out = "|".join([re.escape(k) for k in keywords])

    def transform(self, X: pd.DataFrame, y: Optional[list] = None) -> pd.DataFrame:
        return X[
            ~X[self.target_column_to_filter].str.contains(
                self.keywords_to_filter_out, regex=True, case=False, na=False
            )
        ]
