"""
contains original code by Laurent Bernut relating to
swing and regime definition
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import typing

import typing as t
import src.utils.pd_accessors as pda
from abc import ABC, abstractmethod


class NotEnoughDataError(Exception):
    """unable to collect enough swing data to initialize strategy"""


def average_true_range(
        df: pd.DataFrame,
        window: int,
        _h: str = 'high',
        _l: str = 'low',
        _c: str = 'close',
):
    """
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
    """
    _max = df[_h].combine(df[_c].shift(), max)
    _min = df[_l].combine(df[_c].shift(), min)
    atr = (_max - _min).rolling(window=window).mean()
    # atr = (_max - _min).ewm(span=window, min_periods=window).mean()
    return atr


def relative(
        df: pd.DataFrame,
        bm_df: pd.DataFrame,
        bm_col: str = "close",
        _o: str = "open",
        _h: str = "high",
        _l: str = "low",
        _c: str = "close",
        ccy_df: typing.Optional[pd.DataFrame] = None,
        ccy_col: typing.Optional[str] = None,
        dgt: typing.Optional[int] = None,
        rebase: typing.Optional[bool] = True,
) -> pd.DataFrame:
    """
    df: df
    bm_df, bm_col: df benchmark dataframe & column name
    ccy_df,ccy_col: currency dataframe & column name
    dgt: rounding decimal
    start/end: string or offset
    rebase: boolean rebase to beginning or continuous series
    """

    # BJ: No, input dataframe should already be sliced
    # # Slice df dataframe from start to end period: either offset or datetime
    # df = df[start:end]

    # inner join of benchmark & currency: only common values are preserved
    df = df.join(bm_df[[bm_col]], how="inner")
    adjustment = df[bm_col].copy()
    if ccy_df is not None:
        df = df.join(ccy_df[[ccy_col]], how="inner")
        adjustment = df[bm_col].mul(df["ccy"])
        if dgt is not None:
            adjustment = round(adjustment, dgt)

    # rename benchmark name as bm and currency as ccy
    # df.rename(columns={bm_col: bm_col, ccy_col: "ccy"}, inplace=True)

    # Adjustment factor: calculate the scalar product of benchmark and currency

    df["bmfx"] = adjustment.fillna(method="ffill")

    if rebase is True:
        df["bmfx"] = df["bmfx"].div(df["bmfx"][0])

    # Divide absolute price by fxcy adjustment factor and rebase to first value
    _ro = "r" + str(_o)
    _rh = "r" + str(_h)
    _rl = "r" + str(_l)
    _rc = "r" + str(_c)

    df[_ro] = df[_o].div(df["bmfx"])
    df[_rh] = df[_h].div(df["bmfx"])
    df[_rl] = df[_l].div(df["bmfx"])
    df[_rc] = df[_c].div(df["bmfx"])

    if dgt is not None:
        df["r" + str(_o)] = round(df["r" + str(_o)], dgt)
        df["r" + str(_h)] = round(df["r" + str(_h)], dgt)
        df["r" + str(_l)] = round(df["r" + str(_l)], dgt)
        df["r" + str(_c)] = round(df["r" + str(_c)], dgt)

    # drop after function is called, user decides
    # df = df.drop([bm_col, "ccy", "bmfx"], axis=1)

    return df


def simple_relative(df, bm_close, rebase=True):
    """simplified version of relative calculation"""
    bm = bm_close.ffill()
    if rebase is True:
        bm = bm.div(bm[0])
    return df.div(bm, axis=0)


def relative_all_rebase(df, bm_close, axis):
    return df.div(bm_close * df.iloc[0], axis=axis) * bm_close[0]


class AbcSwingParams:
    def __init__(self, extreme_levels, atr_levels, peaks, sw_type):
        self.extreme_levels = extreme_levels
        self.atr_levels = atr_levels
        self.peaks = peaks
        self.sw_type = sw_type

    @abstractmethod
    def update_params(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_cum_f(sw_type):
        assert sw_type in [-1, 1]
        return 'cummin' if sw_type == 1 else 'cummax'


class BaseSwingParams(AbcSwingParams):
    def __init__(self, atr: pd.Series, price: pd.Series, sw_type: int):
        self._cum_f = self.__class__.get_cum_f(sw_type)
        param_attrs = self.__class__._init_params(atr, price, self._cum_f)
        param_attrs['sw_type'] = sw_type
        super().__init__(**param_attrs)
        self._base_atr = atr
        self._base_price = price

    @staticmethod
    def _init_params(atr: pd.Series, price: pd.Series, cum_f: str,):
        extremes = getattr(price, cum_f)()
        extremes_changed = extremes != extremes.shift(1)
        peaks = price.loc[extremes_changed]
        _atr_at_peaks = atr.copy()
        _atr_at_peaks.loc[~extremes_changed] = np.nan
        _atr_at_peaks = _atr_at_peaks.ffill()
        return {'extreme_levels': extremes, 'atr_levels': _atr_at_peaks, 'peaks': peaks}

    def update_params(self, date_index):
        self._base_price = self._base_price.loc[self._base_price.index > date_index]
        self._base_atr = self._base_atr.loc[self._base_atr.index > date_index]
        res = self.__class__._init_params(self._base_atr, self._base_price, self._cum_f)
        self.extreme_levels = res['extreme_levels']
        self.atr_levels = res['atr_levels']
        self.peaks = res['peaks']


class DerivedSwingParams(AbcSwingParams):
    def __init__(self, index, atr, raw_peaks, sw_type):
        self._cum_f = self.__class__.get_cum_f(sw_type)
        param_attrs = self.__class__._init_params(index, atr, raw_peaks, self._cum_f)
        param_attrs['sw_type'] = sw_type
        super().__init__(**param_attrs)
        self._base_atr = atr
        self._base_index = index

    @staticmethod
    def _init_params(index, atr, raw_peaks, cum_f):
        """
        :param index: index of price series
        :param atr: average true range series
        :param raw_peaks: peaks discovered on the lower level
        :return:
        """
        extreme_levels = pd.DataFrame(index=index, columns=['ext'])
        extreme_levels['ext'] = raw_peaks
        extreme_levels = extreme_levels['ext']
        extreme_levels = getattr(extreme_levels, cum_f)().ffill()
        atr_levels = atr.copy()
        atr_levels.loc[~atr_levels.index.isin(raw_peaks.index)] = np.nan
        atr_levels = atr_levels.ffill()
        return {'extreme_levels': extreme_levels, 'atr_levels': atr_levels, 'peaks': raw_peaks}

    def update_params(self, date_index):
        self._base_index = self._base_index[self._base_index.get_loc(date_index) + 1:]
        self.peaks = self.peaks.loc[self.peaks.index > date_index]
        res = self.__class__._init_params(self._base_index, self._base_atr, self.peaks, self._cum_f)
        self.extreme_levels = res['extreme_levels']
        self.atr_levels = self.atr_levels[self.atr_levels.index > date_index]

    @staticmethod
    def swing_to_raw_peaks(swing_table) -> t.Tuple[pd.Series, pd.Series]:
        raw_peaks = swing_table.copy()
        raw_peaks.index = raw_peaks.start
        sw_hi_peaks = raw_peaks.loc[raw_peaks.type == -1]
        sw_lo_peaks = raw_peaks.loc[raw_peaks.type == 1]
        return sw_hi_peaks.st_px, sw_lo_peaks.st_px


def init_swings(
        df,
        dist_pct,
        retrace_pct,
        n_num,
        lvl=3,
        lvl_limit=3
):
    """
    new init swings function
    :param df:
    :param dist_pct:
    :param retrace_pct:
    :param n_num:
    :param lvl:
    :param lvl_limit:
    :return:
    """
    sw_list = []
    atr = average_true_range(df, n_num)
    _px = df[['close', 'high', 'low']].copy()
    _px['avg_px'] = df[['high', 'low', 'close']].mean(axis=1)

    _atr_valid_dt = atr.first_valid_index()
    _px = _px.loc[_atr_valid_dt:]
    _atr = atr.loc[_atr_valid_dt:]

    # sw_hi_params = BaseSwingParams(_atr, _px.avg_px, -1)
    # sw_lo_params = BaseSwingParams(_atr, _px.avg_px, 1)

    vlty_lookup = {3: 5, 2: 2.5, 1: 1}

    i = 1
    while True:
        vlty_mult = vlty_lookup[i]
        adjusted_atr = _atr * vlty_mult

        sw_hi_params = BaseSwingParams(adjusted_atr, _px.avg_px, -1)
        sw_lo_params = BaseSwingParams(adjusted_atr, _px.avg_px, 1)

        swings = volatility_swings(_px, sw_hi_params, sw_lo_params)
        swings['lvl'] = i
        swings['st_px'] = pda.PeakTable(swings).start_price(df)
        swings['en_px'] = pda.PeakTable(swings).end_price(df)

        sw_hi, sw_lo = DerivedSwingParams.swing_to_raw_peaks(swings)

        df[f'hi{i}'] = sw_hi
        df[f'lo{i}'] = sw_lo

        sw_list.append(swings)
        _prior_swings = swings

        if i == lvl:
            break
        i += 1


    peak_table = pd.concat(sw_list).sort_values(by='start', ascending=True).reset_index(drop=True)
    return df, peak_table


def volatility_swings(px: pd.DataFrame, hi_sw_params: AbcSwingParams, lo_sw_params):
    """
    alternate looking for swing hi/lo starting with whichever swing is found sooner
    :param px:
    :param hi_sw_params:
    :param lo_sw_params:
    :return:
    """
    # TODO need separate param when base vlty is needed for retrace
    # atr = atr * vol_mult
    _px = px.copy()
    # _atr = atr.copy()
    swing_data = []
    latest_swing_data, swing_params = initial_volatility_swing(_px, hi_sw_params, lo_sw_params)
    while None not in latest_swing_data.values():
        latest_swing_data['type'] = swing_params.sw_type
        swing_data.append(latest_swing_data.values())
        swing_params = hi_sw_params if swing_params == lo_sw_params else lo_sw_params
        latest_swing_date = latest_swing_data['peak_date']
        latest_swing_discovery_date = latest_swing_data['peak_discovery_date']
        # try:
        _px = _px.loc[_px.index > latest_swing_date]
        swing_params.update_params(latest_swing_date)
        # except TypeError:
        #     pass
        # swap swing type from previous and search
        latest_swing_data = get_next_peak_data(_px.close, swing_params)
        if None in latest_swing_data.values():
            break
        elif latest_swing_data['peak_discovery_date'] < latest_swing_discovery_date:
            latest_swing_data['peak_discovery_date'] = latest_swing_discovery_date

    return pd.DataFrame(data=swing_data, columns=['start', 'end', 'type'])


def initial_volatility_swing(_px, hi_sw_params, lo_sw_params):
    """get data for the first swing in the series"""
    high_peak_data = get_next_peak_data(_px.close, hi_sw_params)
    low_peak_data = get_next_peak_data(_px.close, lo_sw_params)
    swing_data_selector = {
        high_peak_data['peak_discovery_date']: (high_peak_data, hi_sw_params),
        low_peak_data['peak_discovery_date']: (low_peak_data, lo_sw_params),
    }
    discovery_compare = []
    if None not in high_peak_data.values():
        discovery_compare.append(high_peak_data['peak_discovery_date'])
    if None not in low_peak_data.values():
        discovery_compare.append(low_peak_data['peak_discovery_date'])
    if len(discovery_compare) > 0:
        if len(discovery_compare) > 1:
            latest_swing_discovery_date = np.minimum(*discovery_compare)
        else:
            latest_swing_discovery_date = discovery_compare[0]
        res = swing_data_selector[latest_swing_discovery_date]
    else:
        # if none in both, just return one of None data dicts
        res = swing_data_selector[high_peak_data['peak_discovery_date']]
    return res


def get_next_peak_data(close_price, swing_params: AbcSwingParams) -> t.Dict[str, t.Any]:
    """
    returns the first date where close price crosses distance threshold
    """
    # if lvl_col in px.columns.to_list():
    #     extremes = px[lvl_col]
    # else:

    # if dir_ == -1:
    #     cum_f = 'cummax'
    #     def f(divisor, _):
    #         return abs(retrace / divisor)
    # else:
    #     cum_f = 'cummin'
    #     def f(divisor, round_val):
    #         return round(retrace / divisor, round_val)
    # retrace = 1
    distance_threshold = abs(close_price - swing_params.extreme_levels) - swing_params.atr_levels
    peak_discovery_date = close_price.loc[distance_threshold > 0].first_valid_index()
    if peak_discovery_date is not None:
        date_query = swing_params.peaks.index <= peak_discovery_date
        price_query = swing_params.peaks == swing_params.extreme_levels.loc[peak_discovery_date]
        peak_date = swing_params.peaks.loc[date_query & price_query].index[-1]
        # date_query = px.index <= peak_discovery_date
        # price_query = px.avg_px == extremes.loc[peak_discovery_date]
        # peak_date = px.loc[date_query & price_query].iloc[-1].name
    else:
        return {'peak_date': None, 'peak_discovery_date': None}

    return {'peak_date': peak_date, 'peak_discovery_date': peak_discovery_date}


@dataclass
class RegimeFcLists:
    fc_vals: t.List
    fc_dates: t.List
    rg_ch_dates: t.List
    rg_ch_vals: t.List

    def update(self, data: t.Dict):
        self.fc_vals.append(data['fc_val'])
        self.fc_dates.append(data['fc_date'])
        self.rg_ch_dates.append(data['rg_ch_date'])
        self.rg_ch_vals.append(data['rg_ch_val'])


def plot(_d, _plot_window=0, _use_index=False, axis=None):
    """"""
    cols = [
        'close',
        "hi3",
        "lo3",
        # "clg",
        # "flr",
        # "rg_ch",
        "rg",
        # 'hi2_lag',
        # 'hi3_lag',
        # 'lo2_lag',
        # 'lo3_lag'
    ]
    _axis = (
        _d[cols]
            .iloc[_plot_window:]
            .plot(
                style=["grey", "ro", "go"],  # "kv", "k^", "c:"],
                figsize=(15, 5),
                secondary_y=["rg"],
                # grid=True,
                # title=str.upper(_ticker),
                use_index=_use_index,
                ax=axis
            )
    )
    return _axis


def regime_floor_ceiling(
        df: pd.DataFrame,
        flr,
        clg,
        rg,
        rg_ch,
        stdev,
        threshold,
        peak_table,
        sw_lvl: int = 3,
        _h: str = "high",
        _l: str = "low",
        _c: str = "close",
):

    _peak_table = peak_table.loc[peak_table.lvl == sw_lvl]
    _sw_hi_peak_table = _peak_table.loc[_peak_table.type == -1]
    _sw_lo_peak_table = _peak_table.loc[_peak_table.type == 1]

    fc_find_floor = hof_find_fc(
        df=df,
        price_col=_l,
        extreme_func='min',
        stdev=stdev,
        sw_type=1,
        threshold=threshold
    )
    fc_find_ceiling = hof_find_fc(
        df=df,
        price_col=_h,
        extreme_func='max',
        stdev=stdev,
        sw_type=-1,
        threshold=threshold
    )
    fc_floor_found = hof_fc_found(
        df=df,
        cum_func='cummin',
        fc_type=1
    )
    fc_ceiling_found = hof_fc_found(
        df=df,
        cum_func='cummax',
        fc_type=-1
    )
    calc_breakdown = hof_break_pullback(
        df=df,
        extreme_idx_f='idxmin',
        extreme_val_f='cummax',
    )
    calc_breakout = hof_break_pullback(
        df=df,
        extreme_idx_f='idxmax',
        extreme_val_f='cummin'
    )

    # Range initialisation to 1st swing
    fc_data_cols = ['test', 'fc_val', 'fc_date', 'rg_ch_date', 'rg_ch_val', 'type']
    init_floor_data = {col: (df.index[0] if col == 'fc_date' else 0) for col in fc_data_cols}
    init_ceiling_data = init_floor_data.copy()
    init_floor_data['type'] = 1
    init_ceiling_data['type'] = -1

    fc_data = pd.DataFrame(columns=fc_data_cols)
    fc_data = pd.concat(
        [
            fc_data,
            pd.DataFrame(data=init_floor_data, index=[0]),
            pd.DataFrame(data=init_ceiling_data, index=[1]),
        ]
    )

    # Boolean variables
    ceiling_found = floor_found = breakdown = breakout = False

    _sw_hi_len = len(_sw_hi_peak_table)
    _sw_lo_len = len(_sw_lo_peak_table)
    loop_size = np.maximum(_sw_hi_len, _sw_lo_len)
    _hi_idxs = [i if i < _sw_hi_len else -1 for i in range(loop_size)]
    _lo_idxs = [i if i < _sw_lo_len else -1 for i in range(loop_size)]

    swing_discovery_date = None

    # Loop through swings
    for i in range(loop_size):
        # asymmetric swing list: default to last swing if shorter list
        sw_lo_data = _sw_lo_peak_table.iloc[_lo_idxs[i]]
        sw_hi_data = _sw_hi_peak_table.iloc[_hi_idxs[i]]
        swing_discovery_date = np.maximum(sw_lo_data.end, sw_hi_data.end)  # latest swing index

        # CLASSIC CEILING DISCOVERY
        if ceiling_found is False:
            # Classic ceiling test
            floors = fc_data.loc[fc_data.type == 1]
            res = fc_find_ceiling(fc_ix=floors.fc_date.iloc[-1], latest_swing=sw_hi_data)
            if len(res) > 0:
                # Boolean flags reset
                ceiling_found = True
                floor_found = breakdown = breakout = False
                # Append lists
                fc_data = pd.concat([fc_data, pd.DataFrame(data=res, index=[len(fc_data)])])

                # EXCEPTION HANDLING: price penetrates discovery swing
        # 1. if ceiling found, calculate regime since rg_ch_ix using close.cummin
        elif ceiling_found is True:
            res, df = fc_ceiling_found(
                rg_ch_data=fc_data.iloc[-1],
                latest_hi_lo_sw_discovery=swing_discovery_date
            )
            if res is True:
                breakout = True
                floor_found = ceiling_found = breakdown = False

        # 3. if breakout, test for bearish pullback from highest high since rg_ch_ix
        if breakout is True:
            df = calc_breakout(
                rg_ch_data=fc_data.iloc[-1],
                latest_sw_discovery=swing_discovery_date
            )

        # CLASSIC FLOOR DISCOVERY
        if floor_found is False:
            # Classic floor test
            current_ceiling = fc_data.loc[fc_data.type == -1].iloc[-1]
            res = fc_find_floor(fc_ix=current_ceiling.fc_date, latest_swing=sw_lo_data)
            if len(res) > 0:
                # Boolean flags reset
                floor_found = True
                ceiling_found = breakdown = breakout = False
                fc_data = pd.concat([fc_data, pd.DataFrame(data=res, index=[len(fc_data)])])

        # EXCEPTION HANDLING: price penetrates discovery swing
        # 1. if floor found, calculate regime since rg_ch_ix using close.cummin
        elif floor_found is True:
            res, df = fc_floor_found(
                rg_ch_data=fc_data.iloc[-1],
                latest_hi_lo_sw_discovery=swing_discovery_date
            )
            if res is True:
                breakdown = True
                ceiling_found = floor_found = breakout = False

        # 3. if breakdown,test for bullish rebound from lowest low since rg_ch_ix

        if breakdown is True:
            df = calc_breakdown(
                rg_ch_data=fc_data.iloc[-1],
                latest_sw_discovery=swing_discovery_date
            )
    #             breakdown = False
    #             breakout = True

    # no data excluding the initialized floor/ceiling
    if len(fc_data.iloc[2:]) == 0:
        raise NotEnoughDataError

    # POPULATE FLOOR,CEILING, RG CHANGE COLUMNS

    # remove init floor ceiling rows
    fc_data = fc_data.iloc[2:]
    floors_data = fc_data.loc[fc_data.type == 1]
    ceilings_data = fc_data.loc[fc_data.type == -1]

    df.loc[floors_data.fc_date, flr] = floors_data.fc_val.values
    df.loc[ceilings_data.fc_date, clg] = ceilings_data.fc_val.values
    df.loc[fc_data.rg_ch_date, rg_ch] = fc_data.rg_ch_val.values
    df[rg_ch] = df[rg_ch].fillna(method="ffill")

    # regime from last swing
    if swing_discovery_date is not None:
        df.loc[swing_discovery_date:, rg] = np.where(
            ceiling_found,  # if ceiling found, highest high since rg_ch_ix
            np.sign(df[swing_discovery_date:][_c].cummax() - fc_data.rg_ch_val.iloc[-1]),
            np.where(
                floor_found,  # if floor found, lowest low since rg_ch_ix
                np.sign(df[swing_discovery_date:][_c].cummin() - fc_data.rg_ch_val.iloc[-1]),
                # np.sign(df[swing_discovery_date:][_c].rolling(5).mean() - rg_ch_list[-1]),
                np.nan
            ),
        )
    df[rg] = df[rg].fillna(method="ffill")
    #     #     df[rg+'_no_fill'] = df[rg]
    return df


def find_fc(
        df,
        fc_ix: pd.Timestamp,
        price_col: str,
        extreme_func: str,
        stdev,
        sw_type: int,
        threshold: float,
        latest_swing: pd.Series,
) -> t.Dict:
    """
    tests to find a new fc between the last opposite fc and current swing.
    New fc found if the distance from the most extreme to the latest swing
    meets the minimum threshold

    if finding floor, Get min value between last ceiling and most recent swing low
    :param latest_swing:
    :param fc_ix:
    :param df:
    :param price_col:
    :param extreme_func:
    :param stdev:
    :param sw_type:
    :param threshold:
    :return:
    """
    res = {}

    # try again with next swing if fc > current swing
    if fc_ix >= latest_swing.start:
        return res

    data_range = df.loc[fc_ix: latest_swing.start, price_col]
    extreme_rows = data_range.loc[data_range == getattr(data_range, extreme_func)()]
    fc_val = extreme_rows.iloc[0]
    fc_date = extreme_rows.index[0]
    fc_test = round((latest_swing.st_px - fc_val) / stdev[latest_swing.start], 1)
    fc_test *= sw_type

    if fc_test >= threshold:
        res = {
            'test': fc_test,
            'fc_val': fc_val,
            'fc_date': fc_date,
            'rg_ch_date': latest_swing.end,
            'rg_ch_val': latest_swing.st_px,
            'type': sw_type
        }
    return res


def hof_find_fc(df, price_col, extreme_func, stdev, sw_type, threshold):
    def _fc_found(fc_ix, latest_swing):
        return find_fc(df, fc_ix, price_col, extreme_func, stdev, sw_type, threshold, latest_swing)
    return _fc_found


def fc_found(
        df,
        latest_hi_lo_sw_discovery,
        rg_data: pd.Series,
        cum_func: str,
        fc_type: int,
        close_col='close',
        rg_col='rg',
):
    """
    set regime to where the newest swing was DISCOVERED

    """
    close_data = df.loc[rg_data.rg_ch_date: latest_hi_lo_sw_discovery, close_col]
    close_extremes = getattr(close_data, cum_func)()
    df.loc[rg_data.rg_ch_date: latest_hi_lo_sw_discovery, rg_col] = np.sign(
        close_extremes - rg_data.rg_ch_val
    )

    # 2. if price.cummax/cummin penetrates swing: regime turns bullish/bearish, breakout/breakdown
    test_break = False
    if (df.loc[rg_data.rg_ch_date: latest_hi_lo_sw_discovery, rg_col] * fc_type > 0).any():
        # Boolean flags reset
        test_break = True

    return test_break, df


def hof_fc_found(df, cum_func, fc_type, close_col='close', rg_col='rg'):
    def _fc_found(rg_ch_data, latest_hi_lo_sw_discovery):
        return fc_found(
            df=df,
            latest_hi_lo_sw_discovery=latest_hi_lo_sw_discovery,
            rg_data=rg_ch_data,
            cum_func=cum_func,
            fc_type=fc_type,
            close_col=close_col,
            rg_col=rg_col
        )
    return _fc_found


def hof_break_pullback(df, extreme_idx_f, extreme_val_f):
    def _break_pullback(rg_ch_data, latest_sw_discovery):
        return break_pullback(
            df=df,
            rg_ch_data=rg_ch_data,
            latest_hi_lo_sw_discovery=latest_sw_discovery,
            extreme_idx_func=extreme_idx_f,
            extreme_val_func=extreme_val_f,
            rg_col='rg',
            close_col='close'
        )
    return _break_pullback


def break_pullback(
        df,
        rg_ch_data,
        latest_hi_lo_sw_discovery,
        extreme_idx_func: str,
        extreme_val_func: str,
        rg_col='rg',
        close_col='close'
):
    # brkout_high_ix = df.loc[
    #                  rg_ch_dates[-1]: latest_hi_lo_swing, close_col
    #                  ].idxmax()
    data_range = df.loc[rg_ch_data.rg_ch_date: latest_hi_lo_sw_discovery, close_col]
    break_extreme_date = getattr(data_range, extreme_idx_func)()

    # brkout_low = df[brkout_high_ix: latest_hi_lo_swing][close_col].cummin()
    break_vals = df.loc[break_extreme_date: latest_hi_lo_sw_discovery, close_col]
    break_val = getattr(break_vals, extreme_val_func)()

    df.loc[break_extreme_date: latest_hi_lo_sw_discovery, rg_col] = np.sign(
        break_val - rg_ch_data.rg_ch_val
    )

    return df
