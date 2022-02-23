import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
import typing as t
from itertools import chain

from src.utils import trading_stats


def _validate(items: t.List[str], mandatory: t.List[str]):
    col_not_found = [col for col in mandatory if col not in items]
    if col_not_found:
        raise AttributeError(f"{col_not_found} expected but not found")


class DfAccessorBase(metaclass=ABCMeta):
    mandatory_cols: t.List[str]

    def __init__(self, df: pd.DataFrame):
        self._validate(df)
        self._obj = df

    @classmethod
    def _validate(cls, df: pd.DataFrame):
        _validate(df.columns.to_list(), cls.mandatory_cols)


class SeriesAccessorBase(metaclass=ABCMeta):
    mandatory_cols: t.List[str]

    def __init__(self, obj: pd.Series):
        self._validate(obj)
        self._obj = obj

    @classmethod
    def _validate(cls, obj: pd.Series):
        _validate(obj.index.to_list(), cls.mandatory_cols)


def starts(df, col, val):
    return df[(df[col].shift(1) == val) & (df[col] != val)]


def starts_na(df, col) -> pd.Series:
    return df[pd.isna(df[col].shift(1)) & pd.notna(df[col])]


def ends_na(df, col) -> pd.Series:
    return df[pd.isna(df[col].shift(-1)) & pd.notna(df[col])]


def regime_slices(df, regime_col, regime_val=None):

    assert regime_val in [1, -1, None]

    starts = starts_na(df, regime_col)
    ends = ends_na(df, regime_col)
    res = []
    for i, end_date in enumerate(ends.index.to_list()):
        data_slice = df.loc[starts.index[i] : end_date]
        if regime_val is not None and data_slice[regime_col].iloc[0] == regime_val:
            res.append(data_slice)

    return res


@pd.api.extensions.register_series_accessor("pivot_row")
class PivotRow(SeriesAccessorBase):
    mandatory_cols = ["start", "end", "rg"]

    def __init__(self, obj: pd.Series):
        super().__init__(obj)

    def slice(self, dates: pd.Series):
        """
        query for dates between the self._obj pivot rows start and end dates
        TODO, option for inclusive/exclusive
        :return:
        """
        return (self._obj.start < dates) & (dates < self._obj.end)


def date_slice(start, end, dates):
    return (start < dates) & (dates < end)


class Table(metaclass=ABCMeta):
    mandatory_cols: t.List[str]

    def __init__(self, data: pd.DataFrame, name: str):
        self._validate(data)
        self._data = data
        self._name = name

    def _validate(self, df: pd.DataFrame):
        _validate(df.columns.to_list(), self.__class__.mandatory_cols)

    @property
    @abstractmethod
    def mandatory_cols(self):
        """subclasses must define a class attribute mandatory_cols"""

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name


class PriceTable(Table):
    mandatory_cols = ["open", "high", "low", "close"]

    def __init__(self, data: pd.DataFrame, name: str, add_simple_returns=False):
        super().__init__(data, name)

    @property
    def open(self):
        return self.data.open

    @property
    def high(self):
        return self.data.high

    @property
    def low(self):
        return self.data.low

    @property
    def close(self):
        return self.data.close

    @property
    def d1_returns(self):
        return trading_stats.simple_log_returns(self.close)


class PivotTable(Table, metaclass=ABCMeta):
    def __init__(self, data, name, start_date_col, end_date_col):
        super().__init__(data, name)
        self._start_col = start_date_col
        self._end_col = end_date_col

    def unpivot(self, freq, valid_dates):
        """unpivot the dataframe, filtered by give valid dates"""
        un_pivoted = unpivot(self.data, self._start_col, self._end_col, freq)
        return un_pivoted.loc[un_pivoted.date.isin(valid_dates)].set_index('date')

    @property
    def count(self) -> int:
        """total row count"""
        return len(self.data)

    @property
    def counts(self):
        """accumulating row count"""
        return range(1, len(self.data) + 1)


class SignalTable(PivotTable):
    # TODO add current weight column
    mandatory_cols = ["entry", "trail_stop", "fixed_stop", "dir", 'exit_signal_date', 'partial_exit_date']

    def __init__(self, data: pd.DataFrame, name='signals'):
        super().__init__(data=data, name=name, start_date_col='entry', end_date_col='exit_signal_date')

    @property
    def entry(self):
        return self.data.entry

    @property
    def trail_stop(self):
        return self.data.trail_stop

    @property
    def fixed_stop(self):
        return self.data.fixed_stop

    @property
    def dir(self):
        return self.data.dir

    @property
    def exit_signal_date(self):
        return self.data.exit_signal_date

    @property
    def partial_exit_date(self):
        return self.data.partial_exit_date


# def date_ranges(start: pd.Series, end: pd.Series, freq: str):
#     """concatenate date ranges for querying"""
#     return chain.from_iterable(
#         pd.date_range(start=start_value, end=end.iloc[idx], freq=freq)
#         for idx, start_value in start.iteritems()
#     )


def unpivot(
    pivot_table: pd.DataFrame,
    start_date_col: str,
    end_date_col: str,
    freq: str,
    new_date_col="date",
):
    """unpivot the given table given start and end dates"""
    unpivot_table = pivot_table.copy()
    unpivot_table[new_date_col] = unpivot_table.apply(
        lambda x: pd.date_range(x[start_date_col], x[end_date_col], freq=freq), axis=1
    )
    unpivot_table = unpivot_table.explode(new_date_col, ignore_index=True)
    # unpivot_table = unpivot_table.drop(columns=[start_date_col, end_date_col])
    return unpivot_table


