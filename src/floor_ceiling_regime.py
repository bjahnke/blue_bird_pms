"""
this module contains code which ties all aspects of strategy together into a functional model
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
import typing as t
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import yfinance as yf
import pd_accessors as pda
from src.utils import trading_stats as ts, regime
import src.utils.regime


def all_retest_swing(df, rt: str, dist_pct, retrace_pct, n_num, is_relative=False):
    """
    for back testing entries from swings
    get all retest values by working backward, storing the current retest value then
    slicing out data from current retest onward
    :return:
    """
    all_retests = pd.Series(data=np.NAN, index=df.index)
    working_df = df.copy()
    retest_count = 0
    ix = 2220
    ax = None
    lvl4 = None
    while True:
        ix += 1
        # working_df = working_df[['open', 'close', 'high', 'low']].copy()

        try:
            working_df = df[["open", "close", "high", "low"]].iloc[:ix].copy()
            working_df = src.utils.regime.init_swings(
                working_df, dist_pct, retrace_pct, n_num, is_relative=is_relative
            )
            retest_val_lookup = ~pd.isna(working_df[rt])
            retest_value_row = working_df[rt].loc[retest_val_lookup]
            retest_value_index = retest_value_row.index[0]
            all_retests.at[retest_value_index] = retest_value_row
        except KeyError:
            # working_df = working_df.iloc[:-1]
            pass

        else:
            if ax is None:
                try:
                    ax = working_df[["close", "hi4", "lo4", "hi2", "lo2"]].plot(
                        style=["grey", "rv", "g^", "r.", "g.", "ko"],
                        figsize=(15, 5),
                        grid=True,
                        ax=ax,
                    )
                    ax = all_retests.plot(style=["k."], ax=ax)
                    plt.ion()
                    plt.show()
                    plt.pause(0.001)
                except:
                    print("lvl4 not in index")
                    pass
        if ax is not None:

            try:
                ax.clear()
                lvl4 = True
                working_df[["close", "hi4", "lo4", "hi2", "lo2"]].plot(
                    style=["grey", "rv", "g^", "r.", "g."],
                    figsize=(15, 5),
                    grid=True,
                    ax=ax,
                )
                all_retests.plot(style=["k."], ax=ax)
                plt.pause(0.001)
            except:
                if lvl4 is True:
                    print("switch")
                    lvl4 = False
                ax.clear()
                working_df[["close", "hi3", "lo3", "hi2", "lo2"]].plot(
                    style=["grey", "rv", "g^", "r.", "g."],
                    figsize=(15, 5),
                    grid=True,
                    ax=ax,
                )
                all_retests.plot(style=["k."], ax=ax)
                plt.pause(0.001)

            # plt.show()
            # plt.clear()
            # working_df = working_df.loc[:retest_value_index]

        # print(len(working_df.index.to_list()))
        count = all_retests.count()
        if count > retest_count:
            retest_count = count
            print(f"retest count: {retest_count}")

    # return all_retests


def relative(
    df, _o, _h, _l, _c, bm_df, bm_col, ccy_df, ccy_col, dgt, start, end, rebase=True
):
    """
    df: df
    bm_df, bm_col: df benchmark dataframe & column name
    ccy_df,ccy_col: currency dataframe & column name
    dgt: rounding decimal
    start/end: string or offset
    rebase: boolean rebase to beginning or continuous series
    """
    # Slice df dataframe from start to end period: either offset or datetime
    df = df[start:end]

    # inner join of benchmark & currency: only common values are preserved
    df = df.join(bm_df[[bm_col]], how="inner")
    df = df.join(ccy_df[[ccy_col]], how="inner")

    # rename benchmark name as bm and currency as ccy
    df.rename(columns={bm_col: "bm", ccy_col: "ccy"}, inplace=True)

    # Adjustment factor: calculate the scalar product of benchmark and currency
    df["bmfx"] = round(df["bm"].mul(df["ccy"]), dgt).fillna(method="ffill")
    if rebase == True:
        df["bmfx"] = df["bmfx"].div(df["bmfx"][0])

    # Divide absolute price by fxcy adjustment factor and rebase to first value
    df["r" + str(_o)] = round(df[_o].div(df["bmfx"]), dgt)
    df["r" + str(_h)] = round(df[_h].div(df["bmfx"]), dgt)
    df["r" + str(_l)] = round(df[_l].div(df["bmfx"]), dgt)
    df["r" + str(_c)] = round(df[_c].div(df["bmfx"]), dgt)
    df = df.drop(["bm", "ccy", "bmfx"], axis=1)

    return df


def update_sw_lag(swing_lags: pd.Series, swings: pd.Series, discovered_sw_dates):
    """
    Note this function updates args passed for swing_lags and discovered_sw_dates
    :param swing_lags:
    :param swings:
    :param discovered_sw_dates:
    :return:
    """
    swing_lags = swing_lags.reindex(swings.index)
    latest_sw = swings.loc[~pd.isna(swings)].iloc[-1:]
    if latest_sw.index not in discovered_sw_dates:
        swing_lags.loc[latest_sw.index[0] :] = latest_sw[0]
        discovered_sw_dates.append(latest_sw.index)

    return swing_lags


def full_peak_lag(df, asc_peaks) -> pd.DataFrame:
    """
    calculates distance from highest level peak to the time it was discovered
    value is not nan if peak, does not matter if swing low or swing high
    :param df:
    :param asc_peaks: peak level columns in ascending order
    :return:
    """
    # desire lag for all peak levels greater than 1,
    # so if [hi1, hi2, hi3] given,
    # group by [[hi1, hi2], [hi1, hi2, hi3]] to get lag for level 2 and level 3
    lag_groups = []
    for end_idx in range(1, len(asc_peaks)):
        lag_group = [asc_peaks[i] for i in range(end_idx + 1)]
        lag_groups.append(lag_group)
    full_pivot_table = pd.DataFrame(columns=["start", "end", "type"])

    for lag_group in lag_groups:
        highest_peak_col = lag_group[-1]
        highest_peak = df[highest_peak_col]
        prior_peaks = df[lag_group[-2]]

        # will hold the lowest level peaks
        follow_peaks, lag_pivot_table = get_follow_peaks(highest_peak, prior_peaks)
        i = len(lag_group) - 3
        while i >= 0:
            lag_pivot_table = lag_pivot_table.drop(columns=[prior_peaks.name])
            prior_peaks_col = lag_group[i]
            prior_peaks = df[prior_peaks_col]
            follow_peaks, short_lag_pivot_table = get_follow_peaks(
                follow_peaks, prior_peaks
            )
            lag_pivot_table[prior_peaks_col] = short_lag_pivot_table[prior_peaks_col]

            i -= 1
        lag_pivot_table = lag_pivot_table.melt(
            id_vars=[prior_peaks.name],
            value_vars=[highest_peak_col],
            var_name="type",
            value_name="start",
        )
        lag_pivot_table = lag_pivot_table.rename(columns={prior_peaks.name: "end"})
        full_pivot_table = pd.concat([full_pivot_table, lag_pivot_table])

    full_pivot_table = full_pivot_table[["start", "end", "type"]].reset_index(drop=True)
    full_pivot_table["lvl"] = pd.to_numeric(full_pivot_table.type.str.slice(start=-1))
    full_pivot_table["type"] = np.where(
        full_pivot_table.type.str.slice(stop=-1) == "hi", -1, 1
    )
    return full_pivot_table


def get_follow_peaks(
    current_peak: pd.Series, prior_peaks: pd.Series
) -> t.Tuple[pd.Series, pd.DataFrame]:
    """
    calculates lage between current peak and next level peak.
    helper function, must be used sequentially from current level down to lvl 1 peak
    to get full lag
    :param current_peak:
    :param prior_peaks:
    :return:
    """
    pivot_table = pd.DataFrame(columns=[current_peak.name, prior_peaks.name])
    follow_peaks = pd.Series(index=current_peak.index, dtype=pd.Float64Dtype())

    for r in current_peak.dropna().iteritems():
        # slice df starting with r swing, then exclude r swing, drop nans, then only keep the first row
        # gets the first swing of the prior level after the current swing of the current level.
        current_peak_date = r[0]
        follow_peak = prior_peaks.loc[current_peak_date:].iloc[1:].dropna().iloc[:1]
        if len(follow_peak) > 0:
            follow_peaks.loc[follow_peak.index[0]] = follow_peak.iloc[0]
            pivot_table = pivot_table.append(
                {
                    current_peak.name: current_peak_date,
                    prior_peaks.name: follow_peak.index[0],
                },
                ignore_index=True,
            )
    return follow_peaks, pivot_table


def regime_ranges(df, rg_col: str):
    start_col = "start"
    end_col = "end"
    loop_params = [(start_col, df[rg_col].shift(1)), (end_col, df[rg_col].shift(-1))]
    boundaries = {}
    for name, shift in loop_params:
        rg_boundary = df[rg_col].loc[
            ((df[rg_col] == -1) & (pd.isna(shift) | (shift != -1)))
            | ((df[rg_col] == 1) & ((pd.isna(shift)) | (shift != 1)))
        ]
        rg_df = pd.DataFrame(data={rg_col: rg_boundary})
        rg_df.index.name = name
        rg_df = rg_df.reset_index()
        boundaries[name] = rg_df

    boundaries[start_col][end_col] = boundaries[end_col][end_col]
    return boundaries[start_col][[start_col, end_col, rg_col]]


class NoEntriesError(Exception):
    """no entries detected"""


def get_all_entry_candidates(
    price: pd.DataFrame,
    regimes: pd.DataFrame,
    peaks: pd.DataFrame,
    entry_lvls: t.List[int],
    highest_peak_lvl: int,
):
    """
    set fixed stop for first signal in each regime to the recent lvl 3 peak
    build raw signal table, contains entry signal date and direction of trade
    regimes: start(date), end(date), rg(date)
    TODO
        - add fixed_stop_date to output
        - add trail_stop_date to output
    peaks: start(date: peak location), end(date: peak discovery), type
    :param price:
    :param highest_peak_lvl:
    :param entry_lvls:
    :param peaks:
    :param regimes:
    :return: raw_signals_df: entry, fixed stop, trail stop, dir
    """

    raw_signals_list = []

    # rename the table prior to collecting entries
    entry_table = peaks.rename(columns={"start": "trail_stop", "end": "entry"})

    for rg_idx, rg_info in regimes.iterrows():
        rg_entries = get_regime_signal_candidates(
            regime=rg_info,
            entry_table=entry_table,
            entry_lvls=entry_lvls,
            highest_peak_lvl=highest_peak_lvl,
        )
        if rg_entries.empty:
            continue

        rg_entries = validate_entries(price, rg_entries, rg_info.rg)
        raw_signals_list.append(rg_entries)
    try:
        signal_candidates = pd.concat(raw_signals_list).reset_index(drop=True)
    except ValueError:
        raise NoEntriesError

    signal_candidates = signal_candidates.drop(columns=["lvl", "type"])
    return signal_candidates


def get_regime_signal_candidates(
    regime: pd.Series, entry_table: pd.DataFrame, entry_lvls, highest_peak_lvl
):
    """get all regime candidates for a single regime"""
    # set 'start'
    rg_entries = entry_table.loc[
        regime.pivot_row.slice(entry_table.entry)
        # & regime.pivot_row.slice(entry_table.trail_stop)
        & (entry_table.type == regime.rg)
        & (entry_table.lvl.isin(entry_lvls))
    ].copy()
    rg_entries["dir"] = regime.rg
    rg_entries["fixed_stop"] = rg_entries.trail_stop
    rg_entries = rg_entries.sort_values(by="trail_stop")
    try:
        first_sig = rg_entries.iloc[0]
    except IndexError:
        return rg_entries
    peaks_since_first_sig = entry_table.loc[
        entry_table.trail_stop < first_sig.trail_stop
    ]
    prior_major_peaks = peaks_since_first_sig.loc[
        (peaks_since_first_sig.lvl == highest_peak_lvl)
        & (peaks_since_first_sig.type == first_sig.type)
    ]
    try:
        rg_entries.fixed_stop.iat[0] = prior_major_peaks.trail_stop.iat[-1]
    except IndexError:
        # skip if no prior level 3 peaks
        pass
    return rg_entries


@dataclass
class TrailStop:
    """
    pos_price_col: price column to base trail stop movement off of
    neg_price_col: price column to check if stop was crossed
    cum_extreme: cummin/cummax, name of function to use to calculate trailing stop direction
    """

    neg_price_col: str
    pos_price_col: str
    cum_extreme: str
    dir: int

    def init_trail_stop(
        self, price: pd.DataFrame, initial_trail_price, entry_date, rg_end_date
    ) -> pd.Series:
        """
        :param rg_end_date:
        :param entry_date:
        :param price:
        :param initial_trail_price:
        :return:
        """
        entry_price = price.close.loc[entry_date]
        trail_pct_from_entry = (entry_price - initial_trail_price) / entry_price
        extremes = price.loc[entry_date:rg_end_date, self.pos_price_col]

        # when short, pct should be negative, pushing modifier above one
        trail_modifier = 1 - trail_pct_from_entry
        # trail stop reaction must be delayed one bar since same bar reaction cannot be determined
        trail_stop: pd.Series = (
            getattr(extremes, self.cum_extreme)() * trail_modifier
        ).shift(1)
        trail_stop.iat[0] = trail_stop.iat[1]

        return trail_stop

    def exit_signal(self, price: pd.DataFrame, trail_stop: pd.Series) -> pd.Series:
        """detect where price has crossed price"""
        return ((trail_stop - price[self.neg_price_col]) * self.dir) >= 0

    def target_exit_signal(self, price: pd.DataFrame, target_price) -> pd.Series:
        """detect where price has crossed price"""
        return ((target_price - price[self.pos_price_col]) * self.dir) <= 0

    def get_stop_price(
        self, price: pd.DataFrame, stop_date, offset_pct: float
    ) -> float:
        """calculate stop price given a date and percent to offset the stop point from the peak"""
        pct_from_peak = 1 - (offset_pct * self.dir)
        return price[self.neg_price_col].at[stop_date] * pct_from_peak

    def cap_trail_stop(self, trail_data: pd.Series, cap_price) -> pd.Series:
        """apply cap to trail stop"""
        trail_data.loc[((trail_data - cap_price) * self.dir) > 0] = cap_price
        return trail_data

    def validate_entries(self, price, entry_candidates):
        """entry price must be within trail stop/fixed stop"""
        return validate_entries(price, entry_candidates, self.dir)


def validate_entries(price, entry_candidates, direction):
    """entry price must be within trail stop/fixed stop"""
    stop_price_col = "high" if direction == -1 else "low"
    entry_prices = price.loc[entry_candidates.entry, "close"]
    trail_prices = price.loc[entry_candidates.trail_stop, stop_price_col]
    valid_entry_query = ((entry_prices.values - trail_prices.values) * direction) > 0
    return entry_candidates.loc[valid_entry_query]


def get_target_price(stop_price, entry_price, r_multiplier):
    """
    get target price derived from distance from entry to stop loss times r
    :param entry_price:
    :param stop_price:
    :param r_multiplier: multiplier to apply to distance from entry and stop loss
    :return:
    """
    return entry_price + ((entry_price - stop_price) * r_multiplier)


def draw_stop_line(
    stop_calc: TrailStop,
    price: pd.DataFrame,
    trail_stop_date,
    fixed_stop_date,
    entry_date,
    offset_pct,
    r_multiplier,
    rg_end_date,
) -> t.Tuple[pd.Series, pd.Timestamp, pd.Timestamp, pd.Series]:
    """
    trail stop to entry price, then reset to fixed stop price after target price is reached
    :param rg_end_date:
    :param stop_calc:
    :param entry_date:
    :param r_multiplier:
    :param price:
    :param trail_stop_date:
    :param fixed_stop_date:
    :param offset_pct:
    :return:
    """
    entry_price = price.close.loc[entry_date]
    trail_price = stop_calc.get_stop_price(price, trail_stop_date, offset_pct)
    stop_line = stop_calc.init_trail_stop(price, trail_price, entry_date, rg_end_date)
    stop_line = stop_calc.cap_trail_stop(stop_line, entry_price)

    fixed_stop_price = stop_calc.get_stop_price(price, fixed_stop_date, offset_pct)

    target_price = get_target_price(fixed_stop_price, entry_price, r_multiplier)

    target_exit_signal = stop_calc.target_exit_signal(price, target_price)
    partial_exit_date = stop_line.loc[target_exit_signal].first_valid_index()

    if partial_exit_date is not None:
        stop_line.loc[partial_exit_date:] = fixed_stop_price

    stop_loss_exit_signal = stop_calc.exit_signal(price, stop_line)
    exit_signal_date = stop_line.loc[stop_loss_exit_signal].first_valid_index()
    if exit_signal_date is None:
        exit_signal_date = price.index[-1]
    stop_line = stop_line.loc[:exit_signal_date]

    return stop_line, exit_signal_date, partial_exit_date, stop_loss_exit_signal


def draw_french_stop(signals):
    pass


def process_signal_data(
    price_data: pd.DataFrame,
    regimes: pd.DataFrame,
    entry_candidates: pd.DataFrame,
    offset_pct=0.01,
    r_multiplier=1.5,
):
    # sourcery skip: merge-duplicate-blocks, remove-redundant-if
    """process signal data"""
    trail_map = {
        1: TrailStop(
            pos_price_col="high", neg_price_col="low", cum_extreme="cummax", dir=1
        ),
        -1: TrailStop(
            pos_price_col="low", neg_price_col="high", cum_extreme="cummin", dir=-1
        ),
    }
    valid_entries = pd.DataFrame()
    stop_lines = []

    for rg_idx, rg_info in regimes.iterrows():
        stop_calc = trail_map[rg_info.rg]
        start = rg_info.start
        end = rg_info.end

        # next candidate must be higher/lower than prev entry price depending on regime
        while True:
            rg_entry_candidates = entry_candidates.loc[
                pda.date_slice(start, end, entry_candidates.entry)
            ]
            # rg_price_data = price_data.loc[start:end]
            try:
                entry_prices = price_data.loc[rg_entry_candidates.entry, "close"]
                entry_query = ((entry_prices.values - entry_price) * rg_info.rg) > 0
                rg_entry_candidates = rg_entry_candidates.loc[entry_query]
            except NameError:
                pass
            if len(rg_entry_candidates) == 0:
                break

            entry_signal = rg_entry_candidates.iloc[0]

            entry_price = price_data.close.loc[entry_signal.entry]
            # TODO store this stuff signal/stop_loss dataframes?
            (
                stop_line,
                exit_signal_date,
                partial_exit_date,
                stop_loss_exit_signal,
            ) = draw_stop_line(
                stop_calc=stop_calc,
                price=price_data,
                trail_stop_date=entry_signal.trail_stop,
                fixed_stop_date=entry_signal.fixed_stop,
                entry_date=entry_signal.entry,
                offset_pct=offset_pct,
                r_multiplier=r_multiplier,
                rg_end_date=end,
            )
            # crop last signal at entry
            if len(stop_lines) > 0:
                try:
                    stop_lines[-1] = stop_lines[-1].iloc[
                        : stop_lines[-1].index.get_loc(entry_signal.entry)
                    ]
                except KeyError:
                    # unless we moved to the next regime
                    pass
            stop_lines.append(stop_line)
            entry_signal_data = entry_signal.copy()
            entry_signal_data["exit_signal_date"] = exit_signal_date
            entry_signal_data["partial_exit_date"] = partial_exit_date
            if valid_entries.empty:
                valid_entries = pd.DataFrame([entry_signal_data])
            else:
                valid_entries = valid_entries.append(
                    entry_signal_data, ignore_index=True
                )

            start = exit_signal_date
            if partial_exit_date is not None:
                if exit_signal_date <= partial_exit_date:
                    # start = rg_price_data.iloc[rg_price_data.index.get_loc(exit_signal_date) + 1].index[0]
                    start = exit_signal_date
                else:
                    # if exit greater than partial exit, then potentially another signal can be added
                    start = partial_exit_date

    stop_prices = pd.concat(stop_lines)
    return valid_entries, stop_prices


def fc_scale_strategy(
    price_data: pd.DataFrame,
    distance_pct=0.05,
    retrace_pct=0.05,
    swing_window=63,
    sw_lvl=3,
    regime_threshold=0.5,
    trail_offset_pct=0.01,
    r_multiplier=1.5,
    entry_lvls: t.List[int] = None,
    highest_peak_lvl: int = 3,
    side_only: t.Optional[int] = None,
):
    if entry_lvls is None:
        entry_lvls = [2]

    peak_table, enhanced_price_data = init_peak_table(
        price_data=price_data,
        distance_pct=distance_pct,
        retrace_pct=retrace_pct,
        swing_window=swing_window,
        sw_lvl=sw_lvl,
    )

    standard_dev = price_data.close.rolling(swing_window).std(ddof=0)

    regime_table, enhanced_price_data = init_regime_table(
        enhanced_price_data=enhanced_price_data,
        sw_lvl=sw_lvl,
        standard_dev=standard_dev,
        regime_threshold=regime_threshold,
        side_only=side_only,
    )

    valid_entries, stop_loss_series = init_signal_stop_loss_tables(
        price_data,
        regime_table,
        peak_table,
        entry_lvls,
        highest_peak_lvl,
        offset_pct=trail_offset_pct,
        r_multiplier=r_multiplier,
    )

    return (
        enhanced_price_data,
        peak_table,
        regime_table,
        valid_entries,
        stop_loss_series,
    )


def init_peak_table(
    price_data: pd.DataFrame, distance_pct, retrace_pct, swing_window, sw_lvl
):
    """initialization of peak table bundled together"""
    swings = src.utils.regime.init_swings(
        df=price_data,
        dist_pct=distance_pct,
        retrace_pct=retrace_pct,
        n_num=swing_window,
        lvl=sw_lvl,
    )

    hi_peak_table = full_peak_lag(swings, ["hi1", "hi2", "hi3"])
    lo_peak_table = full_peak_lag(swings, ["lo1", "lo2", "lo3"])
    return pd.concat([hi_peak_table, lo_peak_table]).reset_index(drop=True), swings


def init_regime_table(
    enhanced_price_data: pd.DataFrame,
    sw_lvl,
    standard_dev,
    regime_threshold,
    side_only=None,
):
    """initialization of regime table bundled together"""
    shi = f"hi{sw_lvl}"
    slo = f"lo{sw_lvl}"

    data_with_regimes = src.utils.regime.regime_floor_ceiling(
        df=enhanced_price_data,
        slo=slo,
        shi=shi,
        flr="flr",
        clg="clg",
        rg="rg",
        rg_ch="rg_ch",
        stdev=standard_dev,
        threshold=regime_threshold,
    )

    if side_only is not None:
        data_with_regimes = data_with_regimes.loc[data_with_regimes.rg == side_only]

    return regime_ranges(data_with_regimes, "rg"), data_with_regimes


def init_signal_stop_loss_tables(
    price_data,
    regime_table,
    peak_table,
    entry_lvls,
    highest_peak_lvl,
    offset_pct,
    r_multiplier,
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    raw_signals = get_all_entry_candidates(
        price_data, regime_table, peak_table, entry_lvls, highest_peak_lvl
    )

    return process_signal_data(
        price_data,
        regime_table,
        raw_signals,
        offset_pct=offset_pct,
        r_multiplier=r_multiplier,
    )


def init_french_stop_table(entry_table: pd.DataFrame, regime_table: pd.DataFrame) -> pd.DataFrame:
    """when new re-entry occurs, prior entries within regime have stop loss set to previous fixed stop loss"""
    for idx, rg_data in regime_table.iterrows():
        
        rg_entries = entry_table.loc[
            pda.date_slice(rg_data.start, rg_data.end, entry_table.entry)
        ]


def calc_stats(
    price_data: pd.DataFrame,
    signals: pd.DataFrame,
    min_periods: int,
    window: int,
    percentile: float,
    limit,
    freq: str,
) -> pd.DataFrame:
    """
    get full stats of strategy, rolling and expanding
    :param freq:
    :param signals:
    :param price_data:
    :param min_periods:
    :param window:
    :param percentile:
    :param limit:
    :return:
    """

    # TODO include regime returns

    signal_table = pda.SignalTable(signals.copy())
    signal_table.data["trade_count"] = signal_table.counts
    signals_un_pivot = signal_table.unpivot(freq=freq, valid_dates=price_data.index)
    signals_un_pivot = signals_un_pivot.loc[
        ~signals_un_pivot.index.duplicated(keep="last")
    ]

    passive_returns_1d = ts.simple_log_returns(price_data.close)
    signals_un_pivot["returns_1d"] = passive_returns_1d
    # don't use entry date to calculate returns
    signals_un_pivot.loc[signal_table.entry, "returns_1d"] = 0
    strategy_returns_1d = signals_un_pivot["returns_1d"] * signals_un_pivot.dir

    # Performance
    cumul_passive = ts.cumulative_returns_pct(passive_returns_1d, min_periods)
    cumul_returns = ts.cumulative_returns_pct(strategy_returns_1d, min_periods)
    cumul_excess = cumul_returns - cumul_passive - 1
    cumul_returns_pct = ts.cumulative_returns_pct(strategy_returns_1d, min_periods)

    # Robustness metrics
    grit_expanding = ts.expanding_grit(cumul_returns)
    grit_roll = ts.rolling_grit(cumul_returns, window)

    tr_expanding = ts.expanding_tail_ratio(cumul_returns, percentile, limit)
    tr_roll = ts.rolling_tail_ratio(cumul_returns, window, percentile, limit)

    profits_expanding = ts.expanding_profits(strategy_returns_1d)
    losses_expanding = ts.expanding_losses(strategy_returns_1d)
    pr_expanding = ts.profit_ratio(profits=profits_expanding, losses=losses_expanding)

    profits_roll = ts.rolling_profits(strategy_returns_1d, window)
    losses_roll = ts.rolling_losses(strategy_returns_1d, window)
    pr_roll = ts.profit_ratio(profits=profits_roll, losses=losses_roll)

    # Cumulative t-stat
    win_count = (
        strategy_returns_1d[strategy_returns_1d > 0]
        .expanding()
        .count()
        .fillna(method="ffill")
    )
    total_count = (
        strategy_returns_1d[strategy_returns_1d != 0]
        .expanding()
        .count()
        .fillna(method="ffill")
    )

    csr_expanding = ts.common_sense_ratio(pr_expanding, tr_expanding)
    csr_roll = ts.common_sense_ratio(pr_roll, tr_roll)

    # Trade Count
    trade_count = signals_un_pivot["trade_count"]
    signal_roll = trade_count.diff(window)

    win_rate = (win_count / total_count).fillna(method="ffill")
    avg_win = profits_expanding / total_count
    avg_loss = losses_expanding / total_count
    edge_expanding = ts.expectancy(win_rate, avg_win, avg_loss).fillna(method="ffill")
    sqn_expanding = ts.t_stat(trade_count, edge_expanding)

    win_roll = strategy_returns_1d.copy()
    win_roll[win_roll < 0] = np.nan
    win_rate_roll = win_roll.rolling(window, min_periods=0).count() / window
    avg_win_roll = profits_roll / window
    avg_loss_roll = losses_roll / window

    edge_roll = ts.expectancy(
        win_rate=win_rate_roll, avg_win=avg_win_roll, avg_loss=avg_loss_roll
    )
    sqn_roll = ts.t_stat_expanding(signal_count=signal_roll, expectancy=edge_roll)

    score_expanding = ts.robustness_score(grit_expanding, csr_expanding, sqn_expanding)
    score_roll = ts.robustness_score(grit_roll, csr_roll, sqn_roll)
    stat_sheet_dict = {
        # Note: commented out items should be included afterwords
        # 'ticker': symbol,
        # 'tstmt': ticker_stmt,
        # 'st': st,
        # 'mt': mt,
        "perf": cumul_returns_pct,
        "excess": cumul_excess,
        # 'score': round(score_expanding[-1], 1),  # TODO remove (risk_adj_returns used for score)
        # 'score_roll': round(score_roll[-1], 1),  # TODO remove (risk_adj_returns used for score)
        "trades": trade_count,
        "win": win_rate,
        "win_roll": win_rate_roll,
        "avg_win": avg_win,
        "avg_win_roll": avg_win_roll,
        "avg_loss": avg_loss,
        "avg_loss_roll": avg_loss_roll,
        # 'geo_GE': round(geo_ge, 4),
        "expectancy": edge_expanding,
        "edge_roll": edge_roll,
        "grit": grit_expanding,
        "grit_roll": grit_roll,
        "csr": csr_expanding,
        "csr_roll": csr_roll,
        "pr": pr_expanding,
        "pr_roll": pr_roll,
        "tail": tr_expanding,
        "tail_roll": tr_roll,
        "sqn": sqn_expanding,
        "sqn_roll": sqn_roll,
        "risk_adjusted_returns": score_expanding,
        "risk_adj_returns_roll": score_roll,
    }
    for key, value in stat_sheet_dict.items():
        pd.DataFrame.from_dict({key: value})

    return pd.DataFrame.from_dict(stat_sheet_dict)


def rolling_plot(
    price_data: pd.DataFrame, ndf, stop_loss_t, ticker,
):
    """
    recalculates the strategy on a rolling window of the given data to visualize how
    the strategy behaves
    """
    _open = "open"
    _high = "high"
    _low = "low"
    _close = "close"
    use_index = False
    initial_size = 600
    plot_window = 250
    axis = None
    index = initial_size
    fp_rg = None
    hi2_lag = None
    lo2_lag = None
    hi2_discovery_dts = []
    lo2_discovery_dts = []
    d = price_data[[_open, _high, _low, _close]].copy().iloc[:index]

    ndf["stop_loss"] = stop_loss_t
    a = ndf[
        [_close, "hi3", "lo3", "clg", "flr", "rg_ch", "hi2", "lo2", "stop_loss"]
    ].plot(
        style=["grey", "ro", "go", "kv", "k^", "c:", "r.", "g."],
        figsize=(15, 5),
        grid=True,
        title=str.upper(ticker),
        use_index=use_index,
    )

    ndf["rg"].plot(
        style=["b-."],
        # figsize=(15, 5),
        # marker='o',
        secondary_y=["rg"],
        ax=a,
        use_index=use_index,
    )
    plt.show()
    # all_retest_swing(data, 'rt', distance_percent, retrace_percent, swing_window)
    # data[['close', 'hi3', 'lo3', 'rt']].plot(
    #     style=['grey', 'rv', 'g^', 'ko'],
    #     figsize=(10, 5), grid=True, title=str.upper(ticker))

    # data[['close', 'hi3', 'lo3']].plot(
    #     style=['grey', 'rv', 'g^'],
    #     figsize=(20, 5), grid=True, title=str.upper(ticker))

    # plt.show()
    """
    ohlc = ['Open','High','Low','Close']
    _o,_h,_l,_c = [ohlc[h] for h in range(len(ohlc))]
    rg_val = ['Hi3','Lo3','flr','clg','rg','rg_ch',1.5]
    slo, shi,flr,clg,rg,rg_ch,threshold = [rg_val[s] for s in range(len(rg_val))]
    stdev = df[_c].rolling(63).std(ddof=0)
    df = regime_floor_ceiling(df,_h,_l,_c,slo, shi,flr,clg,rg,rg_ch,stdev,threshold)

    df[[_c,'Hi3', 'Lo3','clg','flr','rg_ch','rg']].plot(    
    style=['grey', 'ro', 'go', 'kv', 'k^','c:','y-.'],     
    secondary_y= ['rg'],figsize=(20,5),    
    grid=True, 
    title = str.upper(ticker))

    """

    for idx, row in price_data.iterrows():
        if (num := price_data.index.get_loc(idx)) <= index:
            print(f"iter index {num}")
            continue
        d.at[idx] = row
        try:
            res = fc_scale_strategy(d)
            d = res[0]

            if fp_rg is None:
                fp_rg = d.rg.copy()
                fp_rg = fp_rg.fillna(0)
                hi2_lag = d.hi2.copy()
                lo2_lag = d.lo2.copy()
            else:
                fp_rg = fp_rg.reindex(d.rg.index)
                new_val = d.rg.loc[pd.isna(fp_rg)][0]
                fp_rg.loc[idx] = new_val

                hi2_lag = update_sw_lag(hi2_lag, d.hi2, hi2_discovery_dts)
                lo2_lag = update_sw_lag(lo2_lag, d.lo2, lo2_discovery_dts)

        except KeyError:
            pass
        else:
            pass
            # live print procedure
            try:
                data_plot_window = len(d.index) - plot_window
                if axis is None:
                    axis = (
                        d[[_close, "hi3", "lo3", "clg", "flr", "rg_ch", "rg"]]
                        .iloc[index - plot_window :]
                        .plot(
                            style=["grey", "ro", "go", "kv", "k^", "c:", "b-."],
                            figsize=(15, 5),
                            secondary_y=["rg"],
                            grid=True,
                            title=str.upper(ticker),
                            use_index=use_index,
                        )
                    )
                    fp_rg.iloc[data_plot_window:].plot(
                        style="y-.", secondary_y=True, use_index=use_index, ax=axis
                    )
                    hi2_lag.iloc[data_plot_window:].plot(
                        style="r.", use_index=use_index, ax=axis
                    )
                    lo2_lag.iloc[data_plot_window:].plot(
                        style="g.", use_index=use_index, ax=axis
                    )
                    plt.ion()
                    plt.show()
                    plt.pause(0.001)
                else:
                    plt.gca().cla()
                    axis.clear()
                    d[[_close, "hi3", "lo3", "clg", "flr", "rg_ch", "rg"]].iloc[
                        data_plot_window:
                    ].plot(
                        style=["grey", "ro", "go", "kv", "k^", "c:", "b-."],
                        figsize=(15, 5),
                        secondary_y=["rg"],
                        grid=True,
                        title=str.upper(ticker),
                        ax=axis,
                        use_index=use_index,
                    )
                    fp_rg.iloc[data_plot_window:].plot(
                        style="y-.", secondary_y=True, use_index=use_index, ax=axis
                    )
                    hi2_lag.iloc[data_plot_window:].plot(
                        style="r.", use_index=use_index, ax=axis
                    )
                    lo2_lag.iloc[data_plot_window:].plot(
                        style="g.", use_index=use_index, ax=axis
                    )
                    # d.rt.iloc[window:].plot(style='k.', use_index=use_index, ax=axis)
                    plt.pause(0.001)
            except Exception as e:
                print(e)
        print(idx)

    # plt.close()
    a = ndf[[_close, "hi3", "lo3", "clg", "flr", "rg_ch"]].plot(
        style=["grey", "ro", "go", "kv", "k^", "c:"],
        figsize=(15, 5),
        grid=True,
        title=str.upper(ticker),
        use_index=use_index,
    )
    ndf["rg"].plot(
        style=["b-."],
        # figsize=(15, 5),
        # marker='o',
        secondary_y=["rg"],
        ax=a,
        use_index=use_index,
    )
    fp_rg.plot(style="y-.", secondary_y=True, use_index=use_index, ax=a)
    hi2_lag.plot(style="r.", use_index=use_index, ax=axis)
    lo2_lag.plot(style="g.", use_index=use_index, ax=axis)
    plt.show()


def yf_get_stock_data(symbol, days, interval: str) -> pd.DataFrame:
    """get price data from yahoo finance"""
    data = yf.ticker.Ticker(symbol).history(
        start=(datetime.now() - timedelta(days=days)),
        end=datetime.now(),
        interval=interval,
    )

    data = data.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
    )
    return data[["open", "high", "low", "close"]]


def get_cached_data(symbol, days, interval) -> pd.DataFrame:
    """get price data from local storage"""
    file_name = f"{symbol}_{interval}_{days}d.csv"
    data = pd.read_csv(fr"..\strategy_output\price_data\{file_name}")
    data = data.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
    )

    data_cols = data.columns.to_list()

    if 'Date' in data_cols:
        date_col = 'Date'
    elif 'Datetime' in data_cols:
        date_col = 'Datetime'
    else:
        raise

    data[date_col] = pd.to_datetime(data[date_col])
    data = data.set_index(data[date_col])

    return data[["open", "high", "low", "close"]]


def get_wikipedia_stocks(url):
    """get stock data (names, sectors, etc) from wikipedia"""
    wiki_df = pd.read_html(url)[0]
    tickers_list = list(wiki_df["Symbol"])
    return tickers_list[:], wiki_df


def scan_all(
    symbols,
    get_data_method: t.Callable[[str], pd.DataFrame],
    regime_data,
    bench_df: pd.DataFrame = None,
    distance_pct=0.05,
    retrace_pct=0.05,
    swing_window=63,
    sw_lvl=3,
    regime_threshold=0.5,
    entry_lvls=None,
    highest_peak_lvl=3,
    trail_offset_pct=0.01,
    r_multiplier=1.5,
    freq='15T'
):
    if entry_lvls is None:
        entry_lvls = [2]

    """scan all data"""
    for i, symbol in enumerate(symbols):
        try:
            data = get_data_method(symbol)
        except KeyError:
            # if data is messed up, usually cached data
            continue

        if data.empty:
            print(f"{symbol}, no data")
            continue

        if bench_df is not None:
            # TODO rebase only necessary for visual performance comparison?
            working_data = regime.relative(
                df=data, bm_df=bench_df, bm_col="spy_close", rebase=False
            )
            working_data = working_data[["ropen", "rhigh", "rlow", "rclose"]]
            working_data = working_data.rename(
                columns={
                    "ropen": "open",
                    "rhigh": "high",
                    "rlow": "low",
                    "rclose": "close",
                }
            )
        else:
            working_data = data.copy()

        try:
            # ticker_regime_score = regime_data.loc[regime_data['Symbol'] == symbol, 'score'].iloc[0]
            # if ticker_regime_score > 0:
            #     side_only = 1
            # elif ticker_regime_score < 0:
            #     side_only = -1
            # else:
            #     # skip if regime score is 0: trend is sideways?
            #     continue
            side_only = None

            ndf, peak_t, rg_t, valid_t, stop_loss_t = fc_scale_strategy(
                price_data=working_data,
                side_only=side_only,
                distance_pct=distance_pct,
                retrace_pct=retrace_pct,
                swing_window=swing_window,
                sw_lvl=sw_lvl,
                regime_threshold=regime_threshold,
                entry_lvls=entry_lvls,
                highest_peak_lvl=highest_peak_lvl,
                trail_offset_pct=trail_offset_pct,
                r_multiplier=r_multiplier
            )
        except (regime.NotEnoughDataError, NoEntriesError):
            continue

        stat_sheet = calc_stats(
            data,
            valid_t,
            min_periods=50,
            window=200,
            percentile=0.05,
            limit=5,
            freq=freq,
        )
        print(f"({i}/{len(symbols)}) {symbol}")
        yield {
            "symbol": symbol,
            "ndf": ndf,
            "peak_t": peak_t,
            "rg_t": rg_t,
            "valid_t": valid_t,
            "stop_loss_t": stop_loss_t,
            "stat_sheet": stat_sheet,
        }


def cached_main():
    """use to run scanner on locally stored price data"""
    days = 58
    interval = "15m"
    freq = '15T'
    regime_data = pd.read_csv(r"..\strategy_output\scan\stock_info\sp500_regimes.csv")
    bench_df = get_cached_data("SPY", days=days, interval=interval)
    bench_df = bench_df.rename(columns={"close": "spy_close"})
    tickers = regime_data["Symbol"].to_list()

    stat_overview = pd.DataFrame()

    def get_data_method(symb):
        return get_cached_data(symb, days=days, interval=interval)

    print("scanning...")
    try:
        for scan_data in scan_all(
            tickers, get_data_method, regime_data, bench_df=bench_df, trail_offset_pct=0.01, freq=freq
        ):
            stat_sheet = scan_data["stat_sheet"].reset_index().copy()
            stat_sheet_overview = stat_sheet.iloc[-2].copy()
            stat_sheet_overview["symbol"] = scan_data["symbol"]
            stat_overview = stat_overview.append(stat_sheet_overview)
    except:
        # re raise uncaught exceptions here so stat_overview can be observed
        raise

    stat_overview.to_csv(
        fr"..\strategy_output\scan\overviews\sp500_{interval}_{days}.csv"
    )

    return stat_overview


def main():
    """scans on recent data"""
    days = 58
    interval = "15m"
    freq = '15T'
    sp500_wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    tickers, _ = get_wikipedia_stocks(sp500_wiki)

    def get_data_method(symb):
        return yf_get_stock_data(symb, days=days, interval=interval)

    # INPUTS END, Code start

    stat_overview = pd.DataFrame()

    bench_df = get_data_method("SPY")
    bench_df = bench_df.rename(columns={"close": "spy_close"})

    print("scanning...")
    try:
        for scan_data in scan_all(
            tickers, get_data_method, None, bench_df=bench_df, trail_offset_pct=0.01, freq=freq
        ):
            stat_sheet = scan_data["stat_sheet"].reset_index().copy()
            # TODO -2 because yf gives to the minute data despite before bar closes
            stat_sheet_overview = stat_sheet.iloc[-2].copy()
            stat_sheet_overview["symbol"] = scan_data["symbol"]
            stat_overview = stat_overview.append(stat_sheet_overview)
    except:
        # re raise uncaught exceptions here so stat_overview can be observed
        raise

    stat_overview.to_csv(
        fr"..\strategy_output\scan\overviews\sp500_{interval}_{days}.csv"
    )

    return stat_overview


def download_data(tickers, days, interval):
    """download ticker data to this project directory"""
    for i, ticker in enumerate(tickers):
        data = yf_get_stock_data(ticker, days, interval)
        file_name = f"{ticker}_{interval}_{days}d.csv"
        data.to_csv(fr"..\strategy_output\price_data\{file_name}")
        print(f"({i}/{len(tickers)}) {file_name}")


def price_data_to_relative_series(
    symbols: t.List[str], bench_symbol: str, interval: str, days: int, from_cache=False
) -> pd.DataFrame:
    """
    TODO optional fx data
    translate all given tickers to relative data and plot
    """
    file_name_fmt = "{0}_{1}_{2}d.csv".format
    data_path_fmt = r"..\strategy_output\price_data\{0}".format
    r_out_fmt = "r{0}_{1}_{2}d.csv".format

    bench_file_name = file_name_fmt(bench_symbol, interval, days)
    bench_data = pd.read_csv(data_path_fmt(bench_file_name))
    bench_data = bench_data.set_index("Datetime").rename(columns={"close": "spy_close"})
    r_data = pd.DataFrame(index=bench_data.index)

    for i, symbol in enumerate(symbols):
        file_name = file_name_fmt(symbol, interval, days)
        price_data = pd.read_csv(data_path_fmt(file_name))
        price_data = price_data.set_index("Datetime")
        # if symbol == 'FB':
        #     price_data.close.plot()
        #     plt.show()

        # TODO NOTE: try added because original price data was overwritten
        #   remove with price data can be corrected (write files to different folders in the future
        try:
            r_symbol_data = regime.relative(price_data, bench_data, bm_col="spy_close")[
                ["ropen", "rhigh", "rlow", "rclose"]
            ]
        except KeyError:
            continue
        r_symbol_data.to_csv(data_path_fmt(r_out_fmt(symbol, interval, days)))
        r_data[symbol] = r_symbol_data["rclose"]
        print(f"({i}/{len(symbols)}) {symbol}")

    return r_data


if __name__ == "__main__":
    main()
    # res = price_data_to_relative_series(rd.Symbol.to_list(), 'SPY', '15m', 58)
    # latest_data = r_data.iloc[-2]
    # latest_data = latest_data.sort_values()
    # r_data.plot()
    # plt.show()
    # top_performers = latest_data.iloc[-10:]
    # bot_performers = latest_data.iloc[:10]
    # r_data[
    #     top_performers.index.to_list() +
    #     bot_performers.index.to_list()
    # ].plot()
    # plt.show()
    print("d")
