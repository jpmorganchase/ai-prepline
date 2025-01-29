import logging
from datetime import datetime
from typing import List, Optional, Union

import numpy
import pandas as pd

from ai_prepline.base.transform import BaseTransform

logger = logging.getLogger(__name__)


class AddWindowIdentifier(BaseTransform):
    """
    Takes a dataframe as input and adds a column with a time-window identifier.
    Requires a column with timestamps.
    """

    def __init__(
        self,
        timestamp_column_name: str,
        stride: str = "2h",
        window_size: str = "4h",
        table_format: str = "long",
    ):
        """
        :param timestamp_column_name: the column of the dataframe that contains timestamps. Should
        be of type datetime.
        :param window_size: time window size e.g., "2h" or "1min" for 2 hour or  1 minute.
        :param stride: distance between start times of windows.
        :param table_format: compact or long. "compact" uses a column of lists to denote which rows
        belong to which windows, for example:

        row | [0,1,2] --> row belongs to the 0th, 1st, and 2nd time windows.

        "long" is the expected format that we will be using with the model
        and other transforms.

        The time window limits are produced by self.__compute_time_windows()

        For more info on how to express time, please see [1].

        [1] https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html#pandas-timedelta # pylint: disable=line-too-long
        """
        self.stride = pd.Timedelta(stride)
        self.window_size = pd.Timedelta(window_size)
        self.timestamp_column_name = timestamp_column_name

        if table_format not in ["compact", "long"]:
            raise NotImplementedError(
                ("Unknown option for table_format Options are `long` an `compact`")
            )
        self.table_format = table_format
        self.min_date = None
        self.timewindows = None

    def __compute_time_windows(
        self,
        timestamps: pd.Series,
    ):
        """
        Computes the time-window ranges given the dataframe X.

        :param X: dataframe with data
        """
        self.min_date = min(timestamps)
        max_date = max(timestamps)

        max_date_float = datetime_to_float(max_date)
        min_date_float = datetime_to_float(self.min_date)
        stride_float = datetime_to_float(self.stride)

        # add 1 to keep make the windows inclusive of the last timestamp
        num_of_windows = (
            int(numpy.ceil((max_date_float - min_date_float) / (stride_float))) + 1
        )

        timewindow_start = [
            self.min_date + k * self.stride for k in range(num_of_windows)
        ]
        timewindow_end = [
            self.min_date + k * self.stride + self.window_size
            for k in range(num_of_windows)
        ]

        return pd.DataFrame({"start": timewindow_start, "end": timewindow_end})

    def return_window_indexes(self, x: pd.Timestamp) -> List[int]:
        """
        Maps a pd.Timestamp to the indexes of the time windows that contain it.

        :param x: the target timestamp.
        """

        x_seconds = datetime_to_float(x)
        window_size_seconds = datetime_to_float(self.window_size)
        stride_seconds = datetime_to_float(self.stride)
        min_date_seconds = datetime_to_float(self.min_date)

        index_min = numpy.ceil(
            (x_seconds - window_size_seconds - min_date_seconds) / (stride_seconds)
        )
        index_max = numpy.floor((x_seconds - min_date_seconds) / stride_seconds)

        index_min, index_max = int(index_min), int(index_max)

        # add +1 to index_max as we want the upper bound to be inclusive
        return list(range(index_min, index_max + 1))

    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[list] = None,  # pylint: disable=unused-argument
    ):
        """
        Find which rows belong to which time-windows and add an identifier to the
        dataset to find them. This is an in-place operation that changes the original X.

        :param X: dataframe with the relevant data
        :param timestamp_column_name: the column that contains the timestamps.
        :param table_format: "long" or "wide". For modelling is the default "long"
            format. It takes some time to fold the data into this format so "wide"
            is provided for table debugging

        NOTE: If the windows are intersecting (even at the endpoints),
        then a row of X could belong to more than a single window. Rows will
        be duplicated when that happens.

        For example, if window_size = 1h and stride = 1h, then the timestamps
        ["1/1/2022 1:00", "1/1/2022 2:00"] would result into the windows 1:00-2:00 and
        2:00-3:00. Thus there are two elements in the 1:00-2:00 window and one row in
        the 2:00-3:00 window
        """

        X = X.dropna(subset=[self.timestamp_column_name])
        X = X.reset_index(drop=True)
        self.timewindows = self.__compute_time_windows(X[self.timestamp_column_name])

        X["window"] = X[self.timestamp_column_name].apply(self.return_window_indexes)

        if self.table_format == "long":
            X = X.explode("window").merge(
                self.timewindows, left_on="window", right_index=True
            )

        return X


def datetime_to_float(x: Union[pd.Timestamp, pd.Timedelta]) -> float:
    """
    This function returns the number of seconds since unix epoch.

    :param x: x needs to be either a timestamp or a timedelta.

    """
    if isinstance(x, pd.Timestamp):
        return datetime.timestamp(x)

    return x.total_seconds()
