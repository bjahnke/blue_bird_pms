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


def historical_swings(df, _o, _h, _l, _c, round_place=2):
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

        if n >= 9:
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


def latest_swing_variables(df, shi, slo, rt_hi, rt_lo, _h, _l, _c):
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
        shi_dt = df.loc[pd.notnull(df[shi]), shi].index[-1]
        slo_dt = df.loc[pd.notnull(df[slo]), slo].index[-1]

        s_hi = df.loc[pd.notnull(df[shi]), shi][-1]
        s_lo = df.loc[pd.notnull(df[slo]), slo][-1]
    except IndexError:
        raise NotEnoughDataError

    if slo_dt > shi_dt:
        swg_var = [
            1,
            s_lo,
            slo_dt,
            rt_lo,
            shi,
            df.loc[slo_dt:, _h].max(),
            df.loc[slo_dt:, _h].idxmax(),
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
        ]
    else:
        swg_var = [0] * 7

    return swg_var


def test_distance(ud, bs, hh_ll, dist_vol, dist_pct):
    """

    :param ud: direction
    :param bs: base, swing hi/lo
    :param hh_ll: lowest low or highest high
    :param dist_vol:
    :param dist_pct:
    :return:
    """
    # priority: 1. Vol 2. pct 3. dflt
    if dist_vol > 0:
        distance_test = np.sign(abs(hh_ll - bs) - dist_vol)
    elif dist_pct > 0:
        distance_test = np.sign(abs(hh_ll / bs - 1) - dist_pct)
    else:
        distance_test = np.sign(dist_pct)

    return int(max(distance_test, 0) * ud)


def average_true_range(
        df: pd.DataFrame,
        _h: str,
        _l: str,
        _c: str,
        window: int
):
    """
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
    """
    _max = df[_h].combine(df[_c].shift(), max)
    _min = df[_l].combine(df[_c].shift(), min)
    atr = (_max - _min).rolling(window=window).mean()
    return atr


def retest_swing(
        df,
        _sign: int,
        _rt,
        hh_ll_dt: pd.Timestamp,
        hh_ll: float,
        _c: str,
        _swg
):
    """
    :param df:
    :param _sign:
    :param _rt:
    :param hh_ll_dt: date of hh_ll
    :param hh_ll: lowest low or highest high
    :param _c: close col str
    :param _swg: series to assign the value, shi for swing hi, slo for swing lo
    :return:
    """
    rt_sgmt = df.loc[hh_ll_dt:, _rt]

    if (rt_sgmt.count() > 0) & (_sign != 0):  # Retests exist and distance test met
        if _sign == 1:  #
            rt_list = [
                rt_sgmt.idxmax(),
                rt_sgmt.max(),
                df.loc[rt_sgmt.idxmax():, _c].cummin(),
            ]

        elif _sign == -1:
            rt_list = [
                rt_sgmt.idxmin(),
                rt_sgmt.min(),
                df.loc[rt_sgmt.idxmin():, _c].cummax(),
            ]
        rt_dt, rt_hurdle, rt_px = [rt_list[h] for h in range(len(rt_list))]

        if str(_c)[0] == "r":
            df.loc[rt_dt, "rrt"] = rt_hurdle
        elif str(_c)[0] != "r":
            df.loc[rt_dt, "rt"] = rt_hurdle

        if (np.sign(rt_px - rt_hurdle) == -np.sign(_sign)).any():
            df.at[hh_ll_dt, _swg] = hh_ll
    return df


def retrace_swing(
        df, _sign, _swg, _c, hh_ll_dt, hh_ll, vlty, retrace_vol, retrace_pct
):
    if _sign == 1:  #
        retrace = df.loc[hh_ll_dt:, _c].min() - hh_ll

        if (
                (vlty > 0)
                & (retrace_vol > 0)
                & ((abs(retrace / vlty) - retrace_vol) > 0)
        ):
            df.at[hh_ll_dt, _swg] = hh_ll
        elif (retrace_pct > 0) & ((abs(retrace / hh_ll) - retrace_pct) > 0):
            df.at[hh_ll_dt, _swg] = hh_ll

    elif _sign == -1:
        retrace = df.loc[hh_ll_dt:, _c].max() - hh_ll
        if (
                (vlty > 0)
                & (retrace_vol > 0)
                & ((round(retrace / vlty, 1) - retrace_vol) > 0)
        ):
            df.at[hh_ll_dt, _swg] = hh_ll
        elif (retrace_pct > 0) & ((round(retrace / hh_ll, 4) - retrace_pct) > 0):
            df.at[hh_ll_dt, _swg] = hh_ll
    else:
        retrace = 0
    return df


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


def init_swings(
        df: pd.DataFrame,
        dist_pct: float,
        retrace_pct: float,
        n_num: int,
        lvl=3,
):
    _o, _h, _l, _c = lower_upper_ohlc(df)
    shi = f'hi{lvl}'
    slo = f'lo{lvl}'
    rt_hi = 'hi1'
    rt_lo = 'lo1'

    df = historical_swings(df, _o=_o, _h=_h, _l=_l, _c=_c)
    # df = cleanup_latest_swing(df, shi=shi, slo=slo, rt_hi=rt_hi, rt_lo=rt_lo)

    (
        ud,  # direction +1up, -1down
        bs,  # base, swing hi/lo
        bs_dt,  # swing date
        _rt,  # series name used to detect swing, rt_lo for swing hi, rt_hi for swing lo
        _swg,  # series to assign the value, shi for swing hi, slo for swing lo
        hh_ll,  # lowest low or highest high
        hh_ll_dt  # date of hh_ll
    ) = latest_swing_variables(
        df, shi, slo, rt_hi, rt_lo, _h, _l, _c
    )
    # volatility_series = average_true_range(df=df, _h=_h, _l=_l, _c=_c, window=n_num)
    # vlty = round(volatility_series[hh_ll_dt], 2)
    # dist_vol = 5 * vlty

    # _sign = test_distance(ud, bs, hh_ll, dist_vol, dist_pct)
    # df = retest_swing(df, _sign, _rt, hh_ll_dt, hh_ll, _c, _swg)
    # retrace_vol = 2.5 * vlty
    # df = retrace_swing(df, _sign, _swg, _c, hh_ll_dt, hh_ll, vlty, retrace_vol, retrace_pct)
    return df


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

    # Lists instantiation
    threshold_test, rg_ch_ix_list, rg_ch_list = [], [], []
    floor_ix_list, floor_list, ceiling_ix_list, ceiling_list = [], [], [], []

    floor_lists = RegimeFcLists(floor_list, floor_ix_list, rg_ch_ix_list, rg_ch_list)
    ceiling_lists = RegimeFcLists(ceiling_list, ceiling_ix_list, rg_ch_ix_list, rg_ch_list)

    # Range initialisation to 1st swing
    floor_ix_list.append(df.index[0])
    ceiling_ix_list.append(df.index[0])

    # Boolean variables
    ceiling_found = floor_found = breakdown = breakout = False

    # Swing lists
    swing_highs = df.loc[pd.notnull(df[shi]), shi]
    swing_highs_ix = list(swing_highs.index)
    swing_highs = list(swing_highs)

    swing_lows = df.loc[pd.notnull(df[slo]), slo]
    swing_lows_ix = list(swing_lows.index)
    swing_lows = list(swing_lows)

    loop_size = np.maximum(len(swing_highs), len(swing_lows))
    fc_data = pd.DataFrame()

    swing_discovery_date = None

    # Loop through swings
    for i in range(loop_size):
        # asymmetric swing list: default to last swing if shorter list
        _lo_i = i if i < len(swing_lows_ix) else -1
        s_lo_ix = swing_lows_ix[_lo_i]
        s_lo = swing_lows[_lo_i]

        _hi_i = i if i < len(swing_highs_ix) else -1
        s_hi_ix = swing_highs_ix[_hi_i]
        s_hi = swing_highs[_hi_i]

        _swing_max_ix = np.maximum(s_lo_ix, s_hi_ix)  # latest swing index

        swing_discovery_date = peak_table.loc[
            (peak_table.lvl == sw_lvl) &
            # (peak_table.type == 1) &
            (peak_table.start == _swing_max_ix),
            'end'
        ]
        swing_discovery_date = swing_discovery_date.iloc[0]

        # CLASSIC CEILING DISCOVERY
        if ceiling_found is False:
            # Classic ceiling test
            res = find_fc(
                df,
                fc_ix_list=floor_ix_list,
                latest_swing_ix=s_hi_ix,
                latest_swing_val=s_hi,
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
                rg_ch_dates=rg_ch_ix_list,
                rg_ch_vals=rg_ch_list,
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
                fc_ix_list=ceiling_ix_list,
                latest_swing_ix=s_lo_ix,
                latest_swing_val=s_lo,
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
                rg_ch_dates=rg_ch_ix_list,
                rg_ch_vals=rg_ch_list,
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
                np.sign(df[swing_discovery_date:][_c].rolling(5).mean() - rg_ch_list[-1]),
            ),
        )
        df[rg] = df[rg].fillna(method="ffill")
    #     #     df[rg+'_no_fill'] = df[rg]
    return df


def find_fc(
        df,
        fc_ix_list: t.List[pd.Timestamp],
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
    data_range = df.loc[fc_ix_list[-1]: latest_swing_ix, price_col]
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
        rg_ch_dates,
        rg_ch_vals,
        latest_hi_lo_sw_discovery,
        cum_func: str,
        fc_type: int,
        close_col='close',
        rg_col='rg',
):
    """set regime to where the newest swing was DISCOVERED"""
    close_data = df.loc[rg_ch_dates[-1]: latest_hi_lo_sw_discovery, close_col]
    close_extremes = getattr(close_data, cum_func)()
    df.loc[rg_ch_dates[-1]: latest_hi_lo_sw_discovery, rg_col] = np.sign(
        close_extremes - rg_ch_vals[-1]
    )

    # 2. if price.cummax/cummin penetrates swing: regime turns bullish/bearish, breakout/breakdown
    test_break = False
    if (df.loc[rg_ch_dates[-1]: latest_hi_lo_sw_discovery, rg_col] * fc_type > 0).any():
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
