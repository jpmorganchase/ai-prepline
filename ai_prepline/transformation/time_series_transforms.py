"""
A collection of tools to transform time series structured data into a tabular
form of input and output pairs, suitable for converting time series forecasting
problems into supervised machine learning tasks.
"""

from typing import Any, List, Optional

import pandas as pd

from ai_prepline.base.transform import BaseTransform


class TimeSeriesTransform(BaseTransform):
    """A class dedicated to convert time series data into a tabular format (input
    and labels) suitable for traditional machine learning supervised tasks.

    This class produces inputs and corresponding labels using a rolling window
    system. Additional transformations to the input window are also available.
    """

    def __init__(
        self,
        time_series_col: Optional[List[str]] = None,
        window_size: int = 10,
        time_horizon: int = 0,
        output_dimension: int = 1,
        offset: int = 1,
    ):
        """
        Initialize a TimeSeriesTransform object, so that the transform method
        can be used.
        """
        self.time_series_col = time_series_col or []
        self.window_size = window_size
        self.time_horizon = time_horizon
        self.output_dimension = output_dimension
        self.offset = offset

    def transform(self, X: pd.DataFrame, y: Optional[List[Any]] = None) -> pd.DataFrame:
        return self.rolling_window_io_generation(
            time_series_df=X,
            time_series_col=self.time_series_col,
            window_size=self.window_size,
            time_horizon=self.time_horizon,
            output_dimension=self.output_dimension,
            offset=self.offset,
        )

    @staticmethod
    def rolling_window_io_generation(
        time_series_df: pd.DataFrame = None,
        time_series_col: Optional[List[str]] = None,
        window_size: int = 10,
        time_horizon: int = 0,
        output_dimension: int = 1,
        offset: int = 1,
    ) -> pd.DataFrame:
        """Generates input/output pairs for supervised learning time series
        forecasting.

        It accomodates uni and multivariate data, and/or single and multistep
        ahead prediction, and/or single and multi dimensional output.

        Example of the rolling window in practice:

             window_size  time_horizon   output_dimension
        ----[-----------]--------------(------------------)--------------> time

              offset -->
                      window_size  time_horizon   output_dimension
        -------------[-----------]--------------(------------------)-----> time


        :param time_series_df: a pandas dataframe where the index indicates
        the timestamp and where each column represents the the values of a
        given metric.
        :param time_series_col: list of the column names to be included in
        the conversion.
        :param window_size: size of the window,
        :param time_horizon: position of the label/start of the label
        relative to the end of the window,
        :param output_dimension: dimension of the label, i.e. nr of points
        in the future to be predicted,
        :param offset: number of units by which the rolling window moves
        over when points are being generated.
        """

        time_series_df = time_series_df.sort_index()

        total_system_span = window_size + time_horizon + output_dimension

        io_pairs = []

        if len(time_series_df) >= total_system_span:
            time_series = time_series_df[time_series_col]

            for position in range(len(time_series)):
                if position % offset == 0:
                    system = time_series[position : position + total_system_span]
                    x = system.values[:window_size]
                    y = system.values[-output_dimension:]
                    start_idx = system.index.values[-output_dimension]

                    if len(system) < total_system_span:
                        break

                    io_pairs.append([start_idx, x, y])

        io_pairs = pd.DataFrame(io_pairs, columns=["timestamp", "x", "y"]).set_index(
            "timestamp"
        )

        return io_pairs
