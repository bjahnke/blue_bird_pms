import pandas as pd
import numpy as np
from abc import ABCMeta
import typing as t

from pandas import Timedelta

from strategy_utils import Side, SignalStatus
import trade_stats
from datetime import datetime, timedelta
import operator as op


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


@pd.api.extensions.register_dataframe_accessor("price_data")
class PriceDataAccessor(SeriesAccessorBase):
    mandatory_cols = ["open", "high", "low", "close"]

    def __init__(self, obj: pd.Series):
        super().__init__(obj)

    def init(self):
        """init class with variables"""


class PriceData:
    def __init__(self, obj: pd.DataFrame):
        self._obj = obj
        self.open = obj.open
        self.high = obj.high
        self.low = obj.low
        self.close = obj.close
