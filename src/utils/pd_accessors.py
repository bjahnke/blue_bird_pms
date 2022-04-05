from __future__ import annotations
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
import typing as t
from itertools import chain

from src.money_management import eqty_risk_shares
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
    def mandatory_cols(self) -> t.List[str]:
        """subclasses must define a class attribute mandatory_cols"""

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name

    @classmethod
    def init_empty_df(cls, index=None) -> pd.DataFrame:
        df_kwargs = {'columns': cls.mandatory_cols}
        if index is not None:
            df_kwargs['index'] = index
        return pd.DataFrame(**df_kwargs)

    @classmethod
    def concat(cls, *tables: Table) -> pd.DataFrame:
        datas = []
        for table in tables:
            assert table.__class__ == cls
            datas.append(table.data)
        return pd.condat(datas)


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
        return un_pivoted.loc[un_pivoted.date.isin(valid_dates)].set_index("date")

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
    mandatory_cols = [
        "entry",
        "trail_stop",
        "fixed_stop",
        "dir",
        "exit_signal_date",
        "partial_exit_date",
        "fixed_stop_price"
    ]

    def __init__(self, data: pd.DataFrame, name="signals"):
        super().__init__(
            data=data,
            name=name,
            start_date_col="entry",
            end_date_col="exit_signal_date",
        )

    @property
    def entry(self) -> 'pd.Series[pd.Timestamp]':
        return self.data.entry

    @property
    def trail_stop(self) -> 'pd.Series[pd.Timestamp]':
        return self.data.trail_stop

    @property
    def fixed_stop(self) -> 'pd.Series[pd.Timestamp]':
        return self.data.fixed_stop

    @property
    def dir(self) -> 'pd.Series[int]':
        return self.data.dir

    @property
    def exit_signal_date(self) -> 'pd.Series[pd.Timestamp]':
        return self.data.exit_signal_date

    @property
    def partial_exit_date(self) -> 'pd.Series[pd.Timestamp]':
        """Note: can also be None"""
        return self.data.partial_exit_date

    def pyramid_all(self, base_risk) -> pd.Series:
        """get all risk amortized per regime id"""
        risk = []
        for rg_id in self.data.rg_id.unique():
            signals_by_rg = self.data.loc[self.data.rg_id == rg_id]
            if signals_by_rg.empty:
                break
            amortized_risk = SignalTable(signals_by_rg).pyramid(base_risk)
            risk.append(amortized_risk)
        risk = pd.concat(risk).reset_index(drop=True)
        return risk

    def pyramid(self, base_risk) -> pd.Series:
        """Note: should only be used per regime, not on entire signal table"""
        row_count = pd.Series(data=self.counts) - 1
        return trading_stats.pyramid(row_count, root=2) * base_risk

    def absolute_stop(self, benchmark_prices: PriceTable):
        return self.data.fixed_stop_price.values * benchmark_prices.close.loc[self.fixed_stop]

    def entry_prices(self, price_table: PriceTable):
        return price_table.close.loc[self.entry].reset_index(drop=True)

    def exit_prices(self, price_table: PriceTable):
        return price_table.close.loc[self.exit_signal_date].reset_index(drop=True)

    def partial_exit_prices(self, price_table: PriceTable):
        return price_table.close.loc[self.partial_exit_date.dropna()].reset_index(drop=True)

    def static_returns(self, price_table: PriceTable):
        return (self.exit_prices(price_table) - self.entry_prices(price_table)) * self.dir

    def eqty_risk_shares(self, price_table: PriceTable, eqty, risk, lot=None, fx=None):
        return eqty_risk_shares(
            px=self.entry_prices(price_table),
            sl=self.data.fixed_stop_price,
            eqty=eqty,
            risk=risk,
            lot=lot,
            fx=fx
        )

    def get_open_shares(self, price_data):
        """
        TODO
            - add partial_exit_pct to signal df
            - for each row, get current_open_pct:
                - current_open_pct = 0 if exit_date < price_data.iloc[-1].index
                - else: current_open_pct = partial_exit_pct if partial_exit_date != nan
                - else: current_open_pct = 1.0
            - current_shares = current_open_pct * shares
            - return current_shares.cumsum()
        :param price_data:
        :return:
        """


class PositionTable(PivotTable):
    """
    Table defines pivot periods where
    """

    mandatory_cols = ["signal_id" "initial_size", "remaining_fraction", "start", "end"]

    def __init__(self, data: pd.DataFrame, name="signals"):
        super().__init__(
            data=data,
            name=name,
            start_date_col="start",
            end_date_col="end",
        )


class FrenchStop(Table):
    mandatory_cols = ['rg_id', 'stop_price']

    def __init__(self, data: pd.DataFrame, name=''):
        super().__init__(data, name)

    def update(
        self,
        pt: pd.DataFrame,
        st: pd.DataFrame,
        rg_end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        set stop loss for prior active entries to cost of previous entry
        """
        data_copy = self.data.copy()
        if st.empty:
            return data_copy

        # use the last idx up to -2 depending on size of signal table
        signal = st.iloc[-min(len(st), 2)]
        new_french_stop = pt.close.loc[signal.entry]
        data_copy.loc[st.iloc[-1].entry: rg_end, 'stop_price'] = new_french_stop
        data_copy.loc[st.iloc[-1].entry: rg_end, 'rg_id'] = signal.rg_id

        return data_copy

# def date_ranges(start: pd.Series, end: pd.Series, freq: str):
#     """concatenate date ranges for querying"""
#     return chain.from_iterable(
#         pd.date_range(start=start_value, end=end.iloc[idx], freq=freq)
#         for idx, start_value in start.iteritems()
#     )


class PeakTable(PivotTable):
    mandatory_cols = [
        'start',
        'end',
        'type',
        'lvl'
    ]

    def __init__(self, data: pd.DataFrame, name="peaks"):
        super().__init__(
            data=data,
            name=name,
            start_date_col="start",
            end_date_col="end",
        )

    def start_price(self, price_data):
        """get price by start date and type, result can be assigned to series"""
        return self._get_price(price_data, 'start')

    def end_price(self, price_data):
        """get price by end date and type, result can be assigned to series"""
        return self._get_price(price_data, 'end')

    def _get_price(self, price_data, date_col):
        """get price of the given date col based on type (swing hi/swing lo)"""
        _pt = self.data.copy()
        return np.where(
            _pt.type == 1,
            price_data.low.loc[_pt[date_col]],
            price_data.high.loc[_pt[date_col]],
        )


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
        lambda x: pd.RangeIndex(start=x[start_date_col], stop=x[end_date_col], step=1), axis=1
    )
    unpivot_table = unpivot_table.explode(new_date_col, ignore_index=True)
    # unpivot_table = unpivot_table.drop(columns=[start_date_col, end_date_col])
    return unpivot_table
