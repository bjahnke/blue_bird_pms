"""
contains original code by Laurent Bernut relating to
swing and regime definition
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import typing
from scipy.signal import find_peaks
import typing as t
import src.utils.pd_accessors as pda


def regime_breakout(df, _h, _l, window):
    hl = np.where(
        df[_h] == df[_h].rolling(window).max(),
        1,
        np.where(df[_l] == df[_l].rolling(window).min(), -1, np.nan),
    )
    roll_hl = pd.Series(index=df.index, data=hl).fillna(method="ffill")
    return roll_hl


def lower_upper_ohlc(df, is_relative=False):
    if is_relative == True:
        rel = "r"
    else:
        rel = ""
    if "Open" in df.columns:
        ohlc = [rel + "Open", rel + "High", rel + "Low", rel + "Close"]
    elif "open" in df.columns:
        ohlc = [rel + "open", rel + "high", rel + "low", rel + "close"]

    try:
        _o, _h, _l, _c = [ohlc[h] for h in range(len(ohlc))]
    except:
        _o = _h = _l = _c = np.nan
    return _o, _h, _l, _c


def regime_args(df, lvl, is_relative=False):
    if ("Low" in df.columns) & (is_relative == False):
        reg_val = [
            "Lo1",
            "Hi1",
            "Lo" + str(lvl),
            "Hi" + str(lvl),
            "rg",
            "clg",
            "flr",
            "rg_ch",
        ]
    elif ("low" in df.columns) & (is_relative == False):
        reg_val = [
            "lo1",
            "hi1",
            "lo" + str(lvl),
            "hi" + str(lvl),
            "rg",
            "clg",
            "flr",
            "rg_ch",
        ]
    elif ("Low" in df.columns) & (is_relative == True):
        reg_val = [
            "rL1",
            "rH1",
            "rL" + str(lvl),
            "rH" + str(lvl),
            "rrg",
            "rclg",
            "rflr",
            "rrg_ch",
        ]
    elif ("low" in df.columns) & (is_relative == True):
        reg_val = [
            "rl1",
            "rh1",
            "rl" + str(lvl),
            "rh" + str(lvl),
            "rrg",
            "rclg",
            "rflr",
            "rrg_ch",
        ]

    try:
        rt_lo, rt_hi, slo, shi, rg, clg, flr, rg_ch = [
            reg_val[s] for s in range(len(reg_val))
        ]
    except:
        rt_lo = rt_hi = slo = shi = rg = clg = flr = rg_ch = np.nan
    return rt_lo, rt_hi, slo, shi, rg, clg, flr, rg_ch


def hilo_alternation(hilo, dist=None, hurdle=None):
    i = 0
    while (
            np.sign(hilo.shift(1)) == np.sign(hilo)
    ).any():  # runs until duplicates are eliminated

        # removes swing lows > swing highs
        hilo.loc[
            (np.sign(hilo.shift(1)) != np.sign(hilo))
            & (hilo.shift(1) < 0)  # hilo alternation test
            & (np.abs(hilo.shift(1)) < np.abs(hilo))  # previous datapoint:  high
            ] = np.nan  # high[-1] < low, eliminate low

        hilo.loc[
            (np.sign(hilo.shift(1)) != np.sign(hilo))
            & (hilo.shift(1) > 0)  # hilo alternation
            & (np.abs(hilo) < hilo.shift(1))  # previous swing: low
            ] = np.nan  # swing high < swing low[-1]

        # alternation test: removes duplicate swings & keep extremes
        hilo.loc[
            (np.sign(hilo.shift(1)) == np.sign(hilo))
            & (hilo.shift(1) < hilo)  # same sign
            ] = np.nan  # keep lower one

        hilo.loc[
            (np.sign(hilo.shift(-1)) == np.sign(hilo))
            & (hilo.shift(-1) < hilo)  # same sign, forward looking
            ] = np.nan  # keep forward one

        # removes noisy swings: distance test
        if pd.notnull(dist):
            hilo.loc[
                (np.sign(hilo.shift(1)) != np.sign(hilo))
                & (np.abs(hilo + hilo.shift(1)).div(dist, fill_value=1) < hurdle)
                ] = np.nan

        # reduce hilo after each pass
        hilo = hilo.dropna().copy()
        i += 1
        if i == 4:  # breaks infinite loop
            break
        return hilo


def historical_swings(df, _o='open', _h='high', _l='low', _c='close', round_place=2, lvl_limit=3):
    reduction = df[[_o, _h, _l, _c]].copy()

    reduction["avg_px"] = round(reduction[[_h, _l, _c]].mean(axis=1), round_place)
    highs = reduction["avg_px"].values
    lows = -reduction["avg_px"].values
    reduction_target = len(reduction) // 100
    #     print(reduction_target )

    n = 0
    while len(reduction) >= reduction_target:
        highs_list = find_peaks(highs, distance=1, width=0)
        lows_list = find_peaks(lows, distance=1, width=0)
        hilo = reduction.iloc[lows_list[0]][_l].sub(
            reduction.iloc[highs_list[0]][_h], fill_value=0
        )

        # Reduction dataframe and alternation loop
        hilo_alternation(hilo, dist=None, hurdle=None)
        reduction["hilo"] = hilo

        # Populate reduction df
        n += 1
        hi_lvl_col = str(_h)[:2] + str(n)
        lo_lvl_col = str(_l)[:2] + str(n)

        reduce_hi = reduction.loc[reduction["hilo"] < 0, _h]
        reduce_lo = reduction.loc[reduction["hilo"] > 0, _l]
        reduction[hi_lvl_col] = reduce_hi
        reduction[lo_lvl_col] = reduce_lo

        # Populate main dataframe
        df[hi_lvl_col] = reduce_hi
        df[lo_lvl_col] = reduce_lo

        # Reduce reduction
        reduction = reduction.dropna(subset=["hilo"])
        reduction.fillna(method="ffill", inplace=True)
        highs = reduction[hi_lvl_col].values
        lows = -reduction[lo_lvl_col].values

        if n >= lvl_limit:
            break

    return df


class NotEnoughDataError(Exception):
    """unable to collect enough swing data to initialize strategy"""


def cleanup_latest_swing(df, shi, slo, rt_hi, rt_lo):
    """
    removes false positives
    """
    # latest swing
    try:
        shi_dt = df.loc[pd.notnull(df[shi]), shi].index[-1]
        s_hi = df.loc[pd.notnull(df[shi]), shi][-1]
        slo_dt = df.loc[pd.notnull(df[slo]), slo].index[-1]
        s_lo = df.loc[pd.notnull(df[slo]), slo][-1]
    except (IndexError, KeyError):
        raise NotEnoughDataError

    len_shi_dt = len(df[:shi_dt])
    len_slo_dt = len(df[:slo_dt])

    # Reset false positives to np.nan
    for _ in range(2):
        if (
            len_shi_dt > len_slo_dt and
            (
                df.loc[shi_dt:, rt_hi].max() > s_hi or s_hi < s_lo
            )
        ):
            df.loc[shi_dt, shi] = np.nan
            len_shi_dt = 0
        elif (
            len_slo_dt > len_shi_dt and
            (
                df.loc[slo_dt:, rt_lo].min() < s_lo or s_hi < s_lo
            )
        ):
            df.loc[slo_dt, slo] = np.nan
            len_slo_dt = 0
        else:
            pass

    return df


def latest_swing_variables(df, shi, slo, rt_hi, rt_lo, _h='high', _l='low', _c='close'):
    """

    :param df:
    :param shi:
    :param slo:
    :param rt_hi:
    :param rt_lo:
    :param _h:
    :param _l:
    :param _c:
    :return:
    """
    try:
        # get that latest swing hi/lo dates
        shi_query = pd.notnull(df[shi])
        slo_query = pd.notnull(df[slo])

        shi_dt = df.loc[shi_query, shi].index[-1]
        slo_dt = df.loc[slo_query, slo].index[-1]

        s_hi = df.loc[shi_query, shi][-1]
        s_lo = df.loc[slo_query, slo][-1]
    except IndexError:
        raise NotEnoughDataError

    if slo_dt > shi_dt:
        # swing low date is more recent
        swg_var = [
            1,
            s_lo,
            slo_dt,
            rt_lo,
            shi,
            df.loc[slo_dt:, _h].max(),
            df.loc[slo_dt:, _h].idxmax(),
            'high'
        ]
    elif shi_dt > slo_dt:
        swg_var = [
            -1,
            s_hi,
            shi_dt,
            rt_hi,
            slo,
            df.loc[shi_dt:, _l].min(),
            df.loc[shi_dt:, _l].idxmin(),
            'low'
        ]
    else:
        swg_var = [0] * 7

    return swg_var


def test_distance(base_sw_val, hh_ll, dist_vol, dist_pct):
    """
    :param ud: direction
    :param base_sw_val: base, swing hi/lo
    :param hh_ll: lowest low or highest high
    :param dist_vol:
    :param dist_pct:
    :return:
    """
    # priority: 1. Vol 2. pct 3. dflt
    if dist_vol > 0:
        distance_test = np.sign(abs(hh_ll - base_sw_val) - dist_vol)
    elif dist_pct > 0:
        distance_test = np.sign(abs(hh_ll / base_sw_val - 1) - dist_pct)
    else:
        distance_test = np.sign(dist_pct)
    return distance_test > 0


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


def retest_swing(
        df,
        ud: int,
        _rt,
        hh_ll_dt: pd.Timestamp,
        hh_ll: float,
        _swg: str,
        idx_extreme_f: str,
        extreme_f: str,
        cum_extreme_f: str,
        _c: str = 'close',
):
    """
    :param ud:
    :param cum_extreme_f:
    :param extreme_f:
    :param idx_extreme_f:
    :param df:
    :param _rt:
    :param hh_ll_dt: date of hh_ll
    :param hh_ll: lowest low or highest high
    :param _c: close col str
    :param _swg: series to assign the value, shi for swing hi, slo for swing lo
    :return:
    """
    rt_sgmt = df.loc[hh_ll_dt:, _rt]
    discovery_lag = None

    if rt_sgmt.count() > 0:  # Retests exist and distance test met
        rt_dt = getattr(rt_sgmt, idx_extreme_f)()
        rt_hurdle = getattr(rt_sgmt, extreme_f)()
        rt_px = getattr(df.loc[rt_dt:, _c], cum_extreme_f)()
        df.loc[rt_dt, "rt"] = rt_hurdle

        breach_query = (np.sign(rt_px - rt_hurdle) == -np.sign(ud))
        discovery_lag = rt_sgmt.loc[breach_query].first_valid_index()

        if discovery_lag is not None:
            df.at[hh_ll_dt, _swg] = hh_ll

    return df, discovery_lag


def retrace_swing(
        df,
        ud,
        _swg,
        hh_ll_dt,
        hh_ll,
        vlty,
        retrace_vol,
        retrace_pct,
        _c='close'
):
    if ud == 1:
        extreme_f = 'min'
        extreme_idx_f = 'idxmin'

        def f(divisor, _):
            return abs(retrace / divisor)
    # else ub assumed to be -1
    else:
        extreme_f = 'max'
        extreme_idx_f = 'idxmax'

        def f(divisor, round_val):
            return round(retrace / divisor, round_val)

    data_range = df.loc[hh_ll_dt:, _c]
    retrace = getattr(data_range, extreme_f)() - hh_ll
    discovery_lag = None

    if (
        vlty > 0 and
        retrace_vol > 0 and
        f(vlty, 1) - retrace_vol > 0 or
        retrace_pct > 0 and
        f(hh_ll, 4) - retrace_pct > 0
    ):
        discovery_lag = getattr(data_range, extreme_idx_f)()
        df.at[hh_ll_dt, _swg] = hh_ll

    # if _sign == 1:  #
    #     retrace = df.loc[hh_ll_dt:, _c].min() - hh_ll
    #     if (
    #             (vlty > 0)
    #             & (retrace_vol > 0)
    #             & ((abs(retrace / vlty) - retrace_vol) > 0)
    #     ):
    #         df.at[hh_ll_dt, _swg] = hh_ll
    #     elif (retrace_pct > 0) & ((abs(retrace / hh_ll) - retrace_pct) > 0):
    #         df.at[hh_ll_dt, _swg] = hh_ll
    #
    # elif _sign == -1:
    #     retrace = df.loc[hh_ll_dt:, _c].max() - hh_ll
    #     if (
    #             (vlty > 0)
    #             & (retrace_vol > 0)
    #             & ((round(retrace / vlty, 1) - retrace_vol) > 0)
    #     ):
    #         df.at[hh_ll_dt, _swg] = hh_ll
    #     elif (retrace_pct > 0) & ((round(retrace / hh_ll, 4) - retrace_pct) > 0):
    #         df.at[hh_ll_dt, _swg] = hh_ll
    # else:
    #     retrace = 0
    return df, discovery_lag


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


class LatestSwingData:
    def __init__(
            self,
            ud,  # direction +1up, -1down
            base_sw,  # base, swing hi/lo
            bs_dt,  # swing date
            _rt,  # series name used to detect swing, rt_lo for swing hi, rt_hi for swing lo
            sw_col,  # series to assign the value, shi for swing hi, slo for swing lo
            hh_ll,  # lowest low or highest high
            hh_ll_dt,  # date of hh_ll
            price_col,
    ):
        self.ud = ud

        if self.ud == 1:
            extreme = 'max'
            cum_extreme = 'min'
        else:
            extreme = 'min'
            cum_extreme = 'max'

        self.idx_extreme_f = f'idx{extreme}'
        self.extreme_f = f'{extreme}'
        self.cum_extreme_f = f'cum{cum_extreme}'

        self.base_sw = base_sw
        self.bs_dt = bs_dt
        self.rt = _rt
        self.sw_col = sw_col
        self.extreme_val = hh_ll
        self.extreme_date = hh_ll_dt
        self.price_col = price_col

    @classmethod
    def init_from_latest_swing(cls, df, shi, slo, rt_hi, rt_lo, _h='high', _l='low', _c='close'):
        shi_query = pd.notnull(df[shi])
        slo_query = pd.notnull(df[slo])
        try:
            # get that latest swing hi/lo dates
            sw_hi_date = df.loc[shi_query, shi].index[-1]
            sw_lo_date = df.loc[slo_query, slo].index[-1]
        except IndexError:
            raise NotEnoughDataError

        if sw_lo_date > sw_hi_date:
            # swing low date is more recent
            s_lo = df.loc[slo_query, slo][-1]
            swg_var = cls(
                ud=1,
                base_sw=s_lo,
                bs_dt=sw_lo_date,
                _rt=rt_lo,
                sw_col=shi,
                hh_ll=df.loc[sw_lo_date:, _h].max(),
                hh_ll_dt=df.loc[sw_lo_date:, _h].idxmax(),
                price_col='high'
            )
        #  (shi_dt > slo_dt) assuming that shi_dt == slo_dt is impossible
        else:
            s_hi = df.loc[shi_query, shi][-1]
            swg_var = cls(
                ud=-1,
                base_sw=s_hi,
                bs_dt=sw_hi_date,
                _rt=rt_hi,
                sw_col=slo,
                hh_ll=df.loc[sw_hi_date:, _l].min(),
                hh_ll_dt=df.loc[sw_hi_date:, _l].idxmin(),
                price_col='low'
            )
        return swg_var

    def test_distance(self, dist_vol, dist_pct):
        return test_distance(
            base_sw_val=self.base_sw,
            hh_ll=self.extreme_val,
            dist_vol=dist_vol,
            dist_pct=dist_pct
        )

    def retest_swing(self, df):
        return retest_swing(
            df=df,
            ud=self.ud,
            _rt=self.rt,
            hh_ll_dt=self.extreme_date,
            hh_ll=self.extreme_val,
            _swg=self.sw_col,
            idx_extreme_f=self.idx_extreme_f,
            extreme_f=self.extreme_f,
            cum_extreme_f=self.cum_extreme_f
        )

    def retrace_swing(self, df, vlty, retrace_vol, retrace_pct):
        return retrace_swing(
            df=df,
            ud=self.ud,
            _swg=self.sw_col,
            hh_ll_dt=self.extreme_date,
            hh_ll=self.extreme_val,
            vlty=vlty,
            retrace_vol=retrace_vol,
            retrace_pct=retrace_pct,
        )

    def volatility_swing(self, df, dist_pct, vlty, retrace_pct, retrace_vol_mult=2.5, dist_vol_mult=5):
        """detect last swing via volatility test"""
        dist_vol = vlty * dist_vol_mult
        res = self.test_distance(dist_vol, dist_pct)
        if res is True:
            retrace_vol = vlty * retrace_vol_mult
            df = self.retest_swing(df)
            df = self.retrace_swing(df, vlty=vlty, retrace_vol=retrace_vol, retrace_pct=retrace_pct)
        return df


def old_init_swings(
        df: pd.DataFrame,
        dist_pct: float,
        retrace_pct: float,
        n_num: int,
        lvl=3,
        lvl_limit=3,
):
    _o, _h, _l, _c = ['open', 'high', 'low', 'close']
    shi = f'hi{lvl}'
    slo = f'lo{lvl}'
    rt_hi = 'hi1'
    rt_lo = 'lo1'

    df = historical_swings(df, lvl_limit=lvl_limit)
    df = cleanup_latest_swing(df, shi=shi, slo=slo, rt_hi=rt_hi, rt_lo=rt_lo)

    latest_sw_vars = LatestSwingData.init_from_latest_swing(df, shi, slo, rt_hi, rt_lo)

    volatility_series = average_true_range(df=df, window=n_num)
    _dist_vol_series = volatility_series * 5
    df['rol_hi'] = df['high'].rolling(n_num).max()
    df['rol_lo'] = df['low'].rolling(n_num).min()

    df['hi_vol'] = (df['rol_hi'] - _dist_vol_series).ffill()
    df['lo_vol'] = (df['rol_lo'] + _dist_vol_series).ffill()
    _retrace_vol_series = volatility_series * 2.5
    vlty = round(volatility_series[latest_sw_vars.extreme_date], 2)

    # px = df.loc[bs_dt: hh_ll_dt, price_col]
    # vol = _dist_vol_series.loc[bs_dt: hh_ll_dt]
    # _df = pd.DataFrame()
    # diff = np.sign(abs(px - base_sw) - vol)
    # _df['diff'] = diff
    # _df.diff = _df.loc[(_df.diff > 0)]
    # diff = np.where(diff > 0, 1, 0) * ud
    # vol_lvl = base_sw - vol
    # _t = pd.DataFrame({
    #     price_col: px,
    #     'vlty_test': diff,
    #     'vol_lvl': vol_lvl
    # })
    # _t['base'] = base_sw
    discovery_lag = None
    dist_vol = vlty * 5
    res = latest_sw_vars.test_distance(dist_vol, dist_pct)
    if res is True:
        retrace_vol = vlty * 2.5
        df, retest_swing_lag = latest_sw_vars.retest_swing(df)
        df, retrace_swing_lag = latest_sw_vars.retrace_swing(
            df, vlty=vlty, retrace_vol=retrace_vol, retrace_pct=retrace_pct
        )
        lag_compare = []
        if retest_swing_lag is not None:
            lag_compare.append(retest_swing_lag)

        if retrace_swing_lag is not None:
            lag_compare.append(retrace_swing_lag)

        if len(lag_compare) > 0:
            discovery_lag = np.maximum(*lag_compare)

    return df, discovery_lag


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
    for i in range(1, lvl+1):
        swings = init_volatility_swings(df, atr, i)
        swings['st_px'] = pda.PeakTable(swings).start_price(df)
        swings['en_px'] = pda.PeakTable(swings).end_price(df)
        df.loc[swings.start.loc[swings.type == -1], f'hi{i}'] = swings.loc[swings.type == -1, 'st_px'].values
        df.loc[swings.start.loc[swings.type == 1], f'lo{i}'] = swings.loc[swings.type == 1, 'st_px'].values
        sw_list.append(swings)
    peak_table = pd.concat(sw_list).sort_values(by='start', ascending=True).reset_index(drop=True)
    return df, peak_table


def init_volatility_swings(px: pd.DataFrame, atr, lvl=3):
    vol_mult = {3: 5, 2: 2, 1: 1}[lvl]
    atr = atr * vol_mult
    _px = px[['close']].copy()
    _px['avg_px'] = px[['high', 'low', 'close']].mean(axis=1)
    _atr = atr.copy()
    swing_data = []
    latest_swing_data = initial_volatility_swing(_px, atr)
    while None not in latest_swing_data:
        swing_data.append(latest_swing_data + (lvl,))
        latest_swing_type = latest_swing_data[-1] * -1
        latest_swing_date = latest_swing_data[0]
        try:
            _px = _px.loc[_px.index > latest_swing_date]
        except TypeError:
            pass
        _atr = _atr.loc[_atr.index > latest_swing_date]

        # swap swing type from previous and search
        latest_swing_data = get_next_peak_data(_px, _atr, latest_swing_type)

    return pd.DataFrame(data=swing_data, columns=['start', 'end', 'type', 'lvl'])


def initial_volatility_swing(px, atr):
    """get data for the first swing in the series"""
    high_peak_date, high_discovery_date, lo_dir = get_next_peak_data(px, atr, -1)
    low_peak_date, low_discovery_date, hi_dir = get_next_peak_data(px, atr, 1)
    swing_data_selector = {
        high_discovery_date: (high_peak_date, high_discovery_date, lo_dir),
        low_discovery_date: (low_peak_date, low_discovery_date, hi_dir)
    }
    discovery_compare = []
    if high_discovery_date is not None:
        discovery_compare.append(high_discovery_date)
    if low_discovery_date is not None:
        discovery_compare.append(low_discovery_date)
    if len(discovery_compare) > 0:
        latest_swing_discovery_date = np.minimum(*discovery_compare)
        res = swing_data_selector[latest_swing_discovery_date]
    else:
        # no swings discovered
        res = None, None, None
    return res


def get_next_peak_data(px, atr, dir_):
    """
    returns the first date where close price crosses distance threshold
    """
    assert dir_ in [1, -1], f'got {dir_}'
    cum_f = 'cummax' if dir_ == -1 else 'cummin'

    # if dir_ == -1:
    #     cum_f = 'cummax'
    #     def f(divisor, _):
    #         return abs(retrace / divisor)
    # else:
    #     cum_f = 'cummin'
    #     def f(divisor, round_val):
    #         return round(retrace / divisor, round_val)

    _atr_valid_dt = atr.first_valid_index()
    _px = px.loc[_atr_valid_dt:]
    _atr = atr.loc[_atr_valid_dt:]

    extremes = getattr(_px.avg_px, cum_f)()
    extremes_changed = extremes != extremes.shift(1)

    # only use volatility at peak
    _atr_at_peaks = pd.Series(index=_atr.index)
    _atr_at_peaks.loc[extremes_changed] = _atr.loc[extremes_changed]
    _atr_at_peaks = _atr_at_peaks.ffill()
    # retrace = 1
    distance_threshold = abs(_px.close - extremes) - _atr_at_peaks
    peak_discovery_date = _px.loc[distance_threshold > 0].first_valid_index()
    if peak_discovery_date is not None:
        date_query = _px.index <= peak_discovery_date
        price_query = _px.avg_px == extremes.loc[peak_discovery_date]
        peak_date = _px.loc[date_query & price_query].iloc[-1].name
    else:
        return None, None, None

    return peak_date, peak_discovery_date, dir_


@dataclass
class RegimeFcLists:
    fc_vals: t.List
    fc_dates: t.List
    rg_ch_dates: t.List
    rg_ch_vals: t.List

    def update(self, data: t.Dict):
        self.fc_vals.append(data['extreme'])
        self.fc_dates.append(data['extreme_date'])
        self.rg_ch_dates.append(data['swing_date'])
        self.rg_ch_vals.append(data['swing_val'])


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
    shi = f"hi{sw_lvl}"
    slo = f"lo{sw_lvl}"

    _peak_table = peak_table.loc[peak_table.lvl == sw_lvl]
    _sw_hi_peak_table = _peak_table.loc[_peak_table.type == -1]
    _sw_lo_peak_table = _peak_table.loc[_peak_table.type == 1]

    # Lists instantiation
    threshold_test, rg_ch_ix_list, rg_ch_list = [], [], []
    floor_ix_list, floor_list, ceiling_ix_list, ceiling_list = [], [], [], []

    floor_lists = RegimeFcLists(
        fc_vals=floor_list,
        fc_dates=floor_ix_list,
        rg_ch_dates=rg_ch_ix_list,
        rg_ch_vals=rg_ch_list
    )
    ceiling_lists = RegimeFcLists(
        fc_vals=ceiling_list,
        fc_dates=ceiling_ix_list,
        rg_ch_dates=rg_ch_ix_list,
        rg_ch_vals=rg_ch_list
    )

    # Range initialisation to 1st swing
    floor_ix_list.append(df.index[0])
    ceiling_ix_list.append(df.index[0])

    # Boolean variables
    ceiling_found = floor_found = breakdown = breakout = False

    # Swing lists
    # _sw_hi_peak_table = pda.PeakTable(_sw_hi_peak_table).start_price(df)

    swing_highs = df.loc[pd.notnull(df[shi]), shi]
    swing_highs_ix = list(swing_highs.index)
    swing_highs = list(swing_highs)

    # _sw_lo_peak_table[] = pda.PeakTable(_sw_lo_peak_table).start_price(df)
    swing_lows = df.loc[pd.notnull(df[slo]), slo]
    swing_lows_ix = list(swing_lows.index)
    swing_lows = list(swing_lows)

    _sw_hi_len = len(_sw_hi_peak_table)
    _sw_lo_len = len(_sw_lo_peak_table)
    loop_size = np.maximum(_sw_hi_len, _sw_lo_len)
    _hi_idxs = [i if i < _sw_hi_len else -1 for i in range(loop_size)]
    _lo_idxs = [i if i < _sw_lo_len else -1 for i in range(loop_size)]

    fc_data = pd.DataFrame()

    swing_discovery_date = None

    # Loop through swings
    for i in range(loop_size):
        # asymmetric swing list: default to last swing if shorter list
        sw_lo_data = _sw_lo_peak_table.iloc[_lo_idxs[i]]
        sw_hi_data = _sw_hi_peak_table.iloc[_hi_idxs[i]]

        # s_lo_ix = swing_lows_ix[_lo_idxs[i]]
        # s_lo = swing_lows[_lo_idxs[i]]
        # s_hi_ix = swing_highs_ix[_hi_idxs[i]]
        # s_hi = swing_highs[_hi_idxs[i]]

        # _s_lo_discovery = _sw_lo_peak_table.end.loc[_sw_lo_peak_table.start == s_lo_ix].iloc[0]
        # _s_hi_discovery = _sw_hi_peak_table.end.loc[_sw_hi_peak_table.start == s_hi_ix].iloc[0]
        swing_discovery_date = np.maximum(sw_lo_data.end, sw_hi_data.end)  # latest swing index

        # swing_discovery_date = peak_table.loc[
        #     (peak_table.lvl == sw_lvl) &
        #     # (peak_table.type == 1) &
        #     (peak_table.start == _swing_max_ix),
        #     'end'
        # ]
        # swing_discovery_date = swing_discovery_date.iloc[0]

        # CLASSIC CEILING DISCOVERY
        if ceiling_found is False:
            # Classic ceiling test
            res = find_fc(
                df,
                fc_ix=floor_ix_list[-1],
                latest_swing_ix=sw_hi_data.end,
                latest_swing_val=sw_hi_data.st_px,
                price_col=_h,
                extreme_func='max',
                stdev=stdev,
                sw_type=-1,
                threshold=threshold
            )
            if len(res) > 0:
                # Boolean flags reset
                ceiling_found = True
                floor_found = breakdown = breakout = False
                # Append lists
                ceiling_lists.update(res)
                fc_data = pd.concat([fc_data, pd.DataFrame(data=res, index=[len(fc_data)])])

                # EXCEPTION HANDLING: price penetrates discovery swing
        # 1. if ceiling found, calculate regime since rg_ch_ix using close.cummin
        elif ceiling_found is True:
            res, df = fc_found(
                df=df,
                latest_rg_ch_dt=rg_ch_ix_list[-1],
                latest_rg_ch_val=rg_ch_list[-1],
                latest_hi_lo_sw_discovery=swing_discovery_date,
                cum_func='cummax',
                fc_type=1,
            )
            if res is True:
                breakout = True
                floor_found = ceiling_found = breakdown = False

        # 3. if breakout, test for bearish pullback from highest high since rg_ch_ix

        # if breakout is True:
        #     df = break_pullback(
        #         df=df,
        #         rg_ch_dates=rg_ch_ix_list,
        #         rg_ch_vals=rg_ch_list,
        #         latest_hi_lo_sw_discovery=swing_discovery_date,
        #         extreme_idx_func='idxmax',
        #         extreme_val_func='cummin'
        #     )

        # CLASSIC FLOOR DISCOVERY
        if floor_found is False:
            # Classic floor test
            res = find_fc(
                df=df,
                fc_ix=ceiling_ix_list[-1],
                latest_swing_ix=sw_lo_data.end,
                latest_swing_val=sw_lo_data.st_px,
                price_col=_l,
                extreme_func='min',
                stdev=stdev,
                sw_type=1,
                threshold=threshold
            )
            if len(res) > 0:
                # Boolean flags reset
                floor_found = True
                ceiling_found = breakdown = breakout = False
                floor_lists.update(res)
                fc_data = pd.concat([fc_data, pd.DataFrame(data=res, index=[len(fc_data)])])

        # EXCEPTION HANDLING: price penetrates discovery swing
        # 1. if floor found, calculate regime since rg_ch_ix using close.cummin
        elif floor_found is True:
            res, df = fc_found(
                df=df,
                latest_rg_ch_dt=rg_ch_ix_list[-1],
                latest_rg_ch_val=rg_ch_list[-1],
                latest_hi_lo_sw_discovery=swing_discovery_date,
                cum_func='cummin',
                fc_type=-1
            )
            if res is True:
                breakdown = True
                ceiling_found = floor_found = breakout = False

        # 3. if breakdown,test for bullish rebound from lowest low since rg_ch_ix

        # if breakdown is True:
        #     df = break_pullback(
        #         df=df,
        #         rg_ch_dates=rg_ch_ix_list,
        #         rg_ch_vals=rg_ch_list,
        #         latest_hi_lo_sw_discovery=swing_discovery_date,
        #         extreme_idx_func='idxmin',
        #         extreme_val_func='cummax'
        #     )
    #             breakdown = False
    #             breakout = True

    if len(rg_ch_list) == 0:
        raise NotEnoughDataError

    # POPULATE FLOOR,CEILING, RG CHANGE COLUMNS

    df.loc[floor_ix_list[1:], flr] = floor_list
    df.loc[ceiling_ix_list[1:], clg] = ceiling_list
    df.loc[rg_ch_ix_list, rg_ch] = rg_ch_list
    df[rg_ch] = df[rg_ch].fillna(method="ffill")

    # regime from last swing
    if swing_discovery_date is not None:
        df.loc[swing_discovery_date:, rg] = np.where(
            ceiling_found,  # if ceiling found, highest high since rg_ch_ix
            np.sign(df[swing_discovery_date:][_c].cummax() - rg_ch_list[-1]),
            np.where(
                floor_found,  # if floor found, lowest low since rg_ch_ix
                np.sign(df[swing_discovery_date:][_c].cummin() - rg_ch_list[-1]),
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
        latest_swing_ix: pd.Timestamp,
        latest_swing_val: float,
        price_col: str,
        extreme_func: str,
        stdev,
        sw_type: int,
        threshold: float,
) -> t.Dict:
    """
    tests to find a new fc between the last opposite fc and current swing.
    New fc found if the distance from the most extreme to the latest swing
    meets the minimum threshold
    :param fc_ix:
    :param df:
    :param fc_ix_list:
    :param latest_swing_ix:
    :param latest_swing_val:
    :param price_col:
    :param extreme_func:
    :param stdev:
    :param sw_type:
    :param threshold:
    :return:
    """
    data_range = df.loc[fc_ix: latest_swing_ix, price_col]
    extreme_rows = data_range.loc[data_range == getattr(data_range, extreme_func)()]
    extreme = extreme_rows.iloc[0]
    extreme_ix = extreme_rows.index[0]
    fc_test = round((latest_swing_val - extreme) / stdev[latest_swing_ix], 1)
    fc_test *= sw_type
    res = {}
    if fc_test >= threshold:
        res = {
            'test': fc_test,
            'extreme': extreme,
            'extreme_date': extreme_ix,
            'swing_date': latest_swing_ix,
            'swing_val': latest_swing_val,
            'type': sw_type
        }
    return res


def fc_found(
        df,
        latest_rg_ch_dt,
        latest_rg_ch_val,
        latest_hi_lo_sw_discovery,
        cum_func: str,
        fc_type: int,
        close_col='close',
        rg_col='rg',
):
    """set regime to where the newest swing was DISCOVERED"""
    close_data = df.loc[latest_rg_ch_dt: latest_hi_lo_sw_discovery, close_col]
    close_extremes = getattr(close_data, cum_func)()
    df.loc[latest_rg_ch_dt: latest_hi_lo_sw_discovery, rg_col] = np.sign(
        close_extremes - latest_rg_ch_val
    )

    # 2. if price.cummax/cummin penetrates swing: regime turns bullish/bearish, breakout/breakdown
    test_break = False
    if (df.loc[latest_rg_ch_dt: latest_hi_lo_sw_discovery, rg_col] * fc_type > 0).any():
        # Boolean flags reset
        test_break = True

    return test_break, df


def break_pullback(
        df,
        rg_ch_dates,
        rg_ch_vals,
        latest_hi_lo_sw_discovery,
        extreme_idx_func: str,
        extreme_val_func: str,
        rg_col='rg',
        close_col='close'
):
    # brkout_high_ix = df.loc[
    #                  rg_ch_dates[-1]: latest_hi_lo_swing, close_col
    #                  ].idxmax()
    data_range = df.loc[rg_ch_dates[-1]: latest_hi_lo_sw_discovery, close_col]
    break_extreme_date = getattr(data_range, extreme_idx_func)()

    # brkout_low = df[brkout_high_ix: latest_hi_lo_swing][close_col].cummin()
    break_vals = df.loc[break_extreme_date: latest_hi_lo_sw_discovery, close_col]
    break_val = getattr(break_vals, extreme_val_func)()

    df.loc[break_extreme_date: latest_hi_lo_sw_discovery, rg_col] = np.sign(
        break_val - rg_ch_vals[-1]
    )

    return df
