"""
this module contains code which ties all aspects of strategy together into a functional model
"""
from dataclasses import dataclass, field
import typing as t
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import src.utils.pd_accessors as pda
from src.utils import trading_stats as ts, regime
import src.utils.regime
import src.money_management as mm
import old_regime_functions as orf


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
            df = pd.DataFrame({
                    prior_peaks.name: [follow_peak.index[0]],
                    current_peak.name: [current_peak_date],
            })
            pivot_table = pd.concat([pivot_table, df], axis=0, ignore_index=True)
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
            rg_data=rg_info,
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

    signal_candidates = signal_candidates[['entry', 'en_px', 'dir', 'trail_stop', 'fixed_stop']]
    return signal_candidates


def get_regime_signal_candidates(
    rg_data: pd.Series, entry_table: pd.DataFrame, entry_lvls, highest_peak_lvl, entry_limit=None
):
    """get all regime candidates for a single regime"""

    rg_entries = entry_table.loc[
        pda.PivotRow(rg_data).slice(entry_table.entry)
        # & regime.pivot_row.slice(entry_table.trail_stop)
        & (entry_table.type == rg_data.rg)
        & (entry_table.lvl.isin(entry_lvls))
    ].copy()
    rg_entries["dir"] = rg_data.rg
    rg_entries["fixed_stop"] = rg_entries.trail_stop
    rg_entries = rg_entries.sort_values(by="entry")

    try:
        first_sig = rg_entries.iloc[0]
    except IndexError:
        return rg_entries

    prior_major_peaks = entry_table.loc[
        (entry_table.trail_stop < first_sig.trail_stop)
        & (entry_table.lvl == highest_peak_lvl)
        & (entry_table.type == first_sig.type)
    ]
    try:
        rg_entries.fixed_stop.iat[0] = prior_major_peaks.trail_stop.iat[-1]
    except IndexError:
        # skip if no prior level 3 peaks
        pass
    return rg_entries


def retest_swing_candidates(
        rg_data: pd.Series,
        peak_table: pd.DataFrame,
        entry_lvls,
        highest_peak_lvl,
        entry_limit=None,
):
    """
    enter on swing hi discovery if regime is bearish,
    which is more likely to be a better price, theoretically
    """
    rg_peaks = peak_table.loc[peak_table.end < rg_data.end]
    rg_entries = rg_peaks.loc[
        (rg_peaks.end > rg_data.start)
        & (rg_peaks.lvl.isin(entry_lvls))
        & (rg_peaks.type != rg_data.rg)
    ].copy()

    if rg_entries.empty:
        return rg_entries

    rg_entries['dir'] = rg_data.rg
    rg_entries = rg_entries.rename(columns={'end': 'entry'})

    # set stop loss reference date to a prior (opposite) swing
    # that exceeds entry price
    stop_peaks = rg_peaks.loc[
        (rg_peaks.type == rg_data.rg) &
        (rg_peaks.lvl == 2)
    ]

    rg_entries['trail_stop'] = np.nan
    for idx, entry_data in rg_entries.iterrows():
        prev_stop_peaks = stop_peaks.loc[stop_peaks.end <= entry_data.entry]
        i = len(prev_stop_peaks) - 1
        while i >= 0:
            prev_stop = prev_stop_peaks.iloc[i]
            if ((entry_data.en_px - prev_stop.st_px) * rg_data.rg) > 0:
                rg_entries.at[idx, 'trail_stop'] = prev_stop.start
                break
            i -= 1

    # TODO what does it look like when no stop within entry? (fixed stop is nan)
    rg_entries = rg_entries.dropna()
    rg_entries['fixed_stop'] = rg_entries.trail_stop

    if rg_entries.empty:
        return rg_entries

    first_sig = rg_entries.iloc[0]
    prior_major_peaks = rg_peaks.loc[
        (rg_peaks.end < first_sig.trail_stop)
        & (rg_peaks.lvl == highest_peak_lvl)
        & (rg_peaks.type == first_sig.type)
        & (((first_sig.entry - rg_peaks.st_px) * rg_data.rg) > 0)
    ]
    try:
        rg_entries.fixed_stop.iat[0] = prior_major_peaks.start.iat[-1]
    except IndexError:
        # skip if no prior level 3 peaks
        pass
    return rg_entries


def validate_entries(price: pd.DataFrame, entry_candidates: pd.DataFrame, direction: int):
    """entry price must be within trail stop/fixed stop"""
    assert direction in [1, -1]
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
    stop_calc: mm.TrailStop,
    price: pd.DataFrame,
    trail_stop_date,
    fixed_stop_date,
    entry_date,
    offset_pct,
    r_multiplier,
    rg_end_date,
    atr,
) -> t.Tuple[pd.Series, pd.Timestamp, pd.Timestamp, pd.Series, float]:
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
    _stop_modifier = atr * 2


    entry_price = price.close.loc[entry_date]
    # trail_price = stop_calc.get_stop_price(price, trail_stop_date, offset_pct)
    # stop_line = stop_calc.init_trail_stop(price, trail_price, entry_date, rg_end_date)
    stop_line = stop_calc.init_atr_stop(
        price, price.close.loc[trail_stop_date],
        entry_date, rg_end_date, _stop_modifier
    )
    fixed_stop_price = stop_calc.get_stop_price(price, fixed_stop_date, offset_pct)
    stop_line = stop_calc.cap_trail_stop(stop_line, entry_price)
    target_price = get_target_price(fixed_stop_price, entry_price, r_multiplier)
    target_exit_signal = stop_calc.target_exit_signal(price, target_price)
    partial_exit_date = stop_line.loc[target_exit_signal].first_valid_index()

    if partial_exit_date is not None:
        stop_line.loc[partial_exit_date:] = fixed_stop_price
    else:
        partial_exit_date = np.nan

    stop_loss_exit_signal = stop_calc.exit_signal(price, stop_line)
    exit_signal_date = stop_line.loc[stop_loss_exit_signal].first_valid_index()
    # signal is active until signal end date is not the current date
    if exit_signal_date is None:
        exit_signal_date = rg_end_date
    stop_line = stop_line.loc[:exit_signal_date]

    return stop_line, exit_signal_date, partial_exit_date, stop_loss_exit_signal, fixed_stop_price


def draw_fixed_stop(
    stop_calc: mm.TrailStop,
    price: pd.DataFrame,
    trail_stop_date,
    fixed_stop_date,
    entry_date,
    offset_pct,
    r_multiplier,
    rg_end_date,
) -> t.Tuple[pd.Series, pd.Timestamp, pd.Timestamp, pd.Series, float]:
    entry_price = price.close.loc[entry_date]
    trail_price = stop_calc.get_stop_price(price, trail_stop_date, offset_pct)
    # stop_line = stop_calc.init_trail_stop(price, trail_price, entry_date, rg_end_date)
    # stop_line = stop_calc.cap_trail_stop(stop_line, entry_price)
    stop_line = stop_calc.init_stop_loss(price, trail_price, entry_date, rg_end_date)
    fixed_stop_price = stop_calc.get_stop_price(price, fixed_stop_date, offset_pct)
    target_price = get_target_price(fixed_stop_price, entry_price, r_multiplier)
    target_exit_signal = stop_calc.target_exit_signal(price, target_price)
    partial_exit_date = stop_line.loc[target_exit_signal].first_valid_index()

    if partial_exit_date is not None:
        stop_line.loc[partial_exit_date:] = fixed_stop_price
    else:
        partial_exit_date = np.nan

    stop_loss_exit_signal = stop_calc.exit_signal(price, stop_line)
    exit_signal_date = stop_line.loc[stop_loss_exit_signal].first_valid_index()
    # signal is active until signal end date is not the current date
    if exit_signal_date is None:
        exit_signal_date = rg_end_date
    stop_line = stop_line.loc[:exit_signal_date]

    return stop_line, exit_signal_date, partial_exit_date, stop_loss_exit_signal, fixed_stop_price


def process_signal_data(
    r_price_data: pd.DataFrame,
    regimes: pd.DataFrame,
    entry_candidates: pd.DataFrame,
    peak_table: pd.DataFrame,
    offset_pct=0.01,
    r_multiplier=1.5,
) -> t.Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Execute stop loss logic to discover valid entries from the candidate list.
    Valid entry occurs if all are true:
        - entry price does not exceed stop price (fixed/trail??)
        - entry price does not exceed prior entry price
        - risk of prior trade is reduced (partial or full exit or first entry)
    returns table of valid entries and time series containing stop loss values throughout regime
    """
    # sourcery skip: merge-duplicate-blocks, remove-redundant-if

    atr = regime.average_true_range(r_price_data, 14)

    trail_map = {
        1: mm.TrailStop(
            pos_price_col="close", neg_price_col="close", cum_extreme="cummax", dir=1
        ),
        -1: mm.TrailStop(
            pos_price_col="close", neg_price_col="close", cum_extreme="cummin", dir=-1
        ),
    }
    valid_entries = pd.DataFrame(columns=entry_candidates.columns.to_list())
    stop_lines = []
    french_stop = pda.FrenchStop.init_empty_df(index=r_price_data.index)

    for rg_idx, rg_info in regimes.iterrows():
        stop_calc = trail_map[rg_info.rg]
        start = rg_info.start
        end = rg_info.end
        entry_signal = None
        entry_price = None

        # next candidate must be higher/lower than prev entry price depending on regime
        while True:
            rg_price_data = r_price_data.loc[start:end]
            rg_peak_table = peak_table.loc[
                (peak_table.end >= rg_info.start) &
                (peak_table.end < rg_info.end) &
                (peak_table.lvl == 3) &
                (peak_table.type == rg_info.rg)
            ]

            rg_entry_candidates = entry_candidates.loc[
                pda.date_slice(start, end, entry_candidates.entry)
            ]
            if rg_entry_candidates.empty:
                break

            # rg_entry_candidates = reduce_regime_candidates(
            #     rg_entry_candidates,
            #     r_price_data,
            #     entry_price,
            #     entry_signal,
            #     rg_info,
            #     rg_peak_table
            # )

            rg_entry_candidates = reduce_regime_candidates_new_leg(
                rg_entry_candidates,
                r_price_data,
                entry_price,
                entry_signal,
                rg_info,
                rg_peak_table
            )

            if rg_entry_candidates.empty:
                break

            entry_signal = rg_entry_candidates.iloc[0]
            entry_price = r_price_data.close.loc[entry_signal.entry]

            (
                stop_line,
                exit_signal_date,
                partial_exit_date,
                stop_loss_exit_signal,
                fixed_stop_price,
            ) = draw_stop_line(
                stop_calc=stop_calc,
                price=r_price_data,
                trail_stop_date=entry_signal.trail_stop,
                fixed_stop_date=entry_signal.fixed_stop,
                entry_date=entry_signal.entry,
                offset_pct=offset_pct,
                r_multiplier=r_multiplier,
                rg_end_date=end,
                atr=atr
            )
            # (
            #     stop_line,
            #     exit_signal_date,
            #     partial_exit_date,
            #     stop_loss_exit_signal,
            #     fixed_stop_price,
            # ) = draw_fixed_stop(
            #     stop_calc=stop_calc,
            #     price=r_price_data,
            #     trail_stop_date=entry_signal.trail_stop,
            #     fixed_stop_date=entry_signal.fixed_stop,
            #     entry_date=entry_signal.entry,
            #     offset_pct=offset_pct,
            #     r_multiplier=r_multiplier,
            #     rg_end_date=end,
            # )
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
            entry_signal_data["fixed_stop_price"] = fixed_stop_price
            entry_signal_data["rg_id"] = rg_info.name

            valid_entries = pd.concat([valid_entries, entry_signal_data.to_frame().transpose()], ignore_index=True)

            french_stop = pda.FrenchStop(french_stop).update(
                r_price_data,
                valid_entries.loc[
                    (valid_entries.dir == rg_info.rg) &
                    (valid_entries.entry >= rg_info.start)
                ],
                rg_end=end
            )
            french_exit_signal = stop_calc.exit_signal(rg_price_data, french_stop.stop_price)
            french_exit_date = french_stop.loc[french_exit_signal].first_valid_index()

            # set exits for still open reduced-risk positions

            if french_exit_date is not None:
                update_stop_query = (
                    (pd.notna(valid_entries.partial_exit_date)) &
                    (french_exit_date < valid_entries.exit_signal_date)
                )
                update_stop_query.iloc[-1] = False
                # valid_entries.iloc[:-2].loc[update_stop_query, 'exit_signal_date'] = french_exit_date
                # valid_entries.iloc[:-2, valid_entries.columns.get_loc('exit_signal_date')] = french_exit_date
                valid_entries.loc[update_stop_query, 'exit_signal_date'] = french_exit_date

            start = exit_signal_date
            if not pd.isna(partial_exit_date):
                if exit_signal_date <= partial_exit_date:
                    # start = rg_price_data.iloc[rg_price_data.index.get_loc(exit_signal_date) + 1].index[0]
                    start = exit_signal_date
                else:
                    # if exit greater than partial exit, then potentially another signal can be added
                    start = partial_exit_date

    if len(stop_lines) > 0:
        stop_prices = pd.concat(stop_lines)
    else:
        stop_prices = pd.Series()
        assert valid_entries.empty  # TODO not sure if this can ever be false, so notify me in case it ever is
        valid_entries = pda.SignalTable.init_empty_df()

    return valid_entries, stop_prices, french_stop


def reduce_regime_candidates(
        rg_entry_candidates: pd.DataFrame,
        price_data: pd.DataFrame,
        entry_price: t.Union[float, None],
        entry_signal: t.Union[pd.Series, None],
        rg_info: pd.Series,
        rg_peak_table: pd.DataFrame,
):
    """
    logic for reducing regime candidates,
    For bull regimes, new entries must be higher than previous entry, unless first entry (vice versa for bear)
    additionally:
    # if prev entry check yields no new entries and a new leg exists after it
    # get all entries after most recent new leg
    # otherwise, only get oll candidates after new leg if the next entry is
    # after the leg

    """
    entry_prices = price_data.loc[rg_entry_candidates.entry, "close"]
    try:
        # filter for entries that are within the previous entry
        new_rg_entry_candidates = rg_entry_candidates.loc[
            ((entry_prices.values - entry_price) * rg_info.rg) > 0
        ]
        # get new legs
        _sw_after_entry = rg_peak_table.loc[rg_peak_table.end > entry_signal.entry]
    except TypeError:
        # previous entry data is none (it is the first signal) make no new changes
        new_rg_entry_candidates = rg_entry_candidates

    return new_rg_entry_candidates


def reduce_regime_candidates_new_leg(
        rg_entry_candidates: pd.DataFrame,
        price_data: pd.DataFrame,
        entry_price: t.Union[float, None],
        entry_signal: t.Union[pd.Series, None],
        rg_info: pd.Series,
        rg_peak_table: pd.DataFrame,

):
    """
    logic for reducing regime candidates,
    For bull regimes, new entries must be higher than previous entry, unless first entry (vice versa for bear)
    additionally:
    # if prev entry check yields no new entries and a new leg exists after it
    # get all entries after most recent new leg
    # otherwise, only get oll candidates after new leg if the next entry is
    # after the leg

    """
    entry_prices = price_data.loc[rg_entry_candidates.entry, "close"]
    try:
        # filter for entries that are within the previous entry
        new_rg_entry_candidates = rg_entry_candidates.loc[
            ((entry_prices.values - entry_price) * rg_info.rg) > 0
        ]
        # get new legs
        _sw_after_entry = rg_peak_table.loc[rg_peak_table.end > entry_signal.entry]
    except TypeError:
        # previous entry data is none (it is the first signal) make no new changes
        new_rg_entry_candidates = rg_entry_candidates
    else:
        if not _sw_after_entry.empty:
            # if entries still exist, only select legs that are prior to next entry
            # otherwise it is too early to use new leg
            if not new_rg_entry_candidates.empty:
                _sw_after_entry = _sw_after_entry.loc[
                    _sw_after_entry.end <= new_rg_entry_candidates.iloc[0].entry
                ]

            # if not all new legs were filtered out yet, then it is time to use
            # leg to select new entries
            if not _sw_after_entry.empty:
                new_rg_entry_candidates = rg_entry_candidates.loc[
                    rg_entry_candidates.entry >= _sw_after_entry.iloc[0].end
                ]
    return new_rg_entry_candidates


@dataclass
class FcStrategyTables:
    enhanced_price_data: pd.DataFrame
    peak_table: pd.DataFrame
    regime_table: pd.DataFrame
    valid_entries: pd.DataFrame
    stop_loss_series: pd.Series
    french_stop: pd.DataFrame
    stats_history: pd.DataFrame = field(init=False)


def fc_scale_strategy(
    price_data: pd.DataFrame,
    abs_price_data: pd.DataFrame,
    distance_pct=0.05,
    retrace_pct=0.05,
    swing_window=63,
    sw_lvl=3,
    regime_threshold=0.5,
    trail_offset_pct=0.01,
    r_multiplier=1.5,
    entry_lvls: t.List[int] = None,
    highest_peak_lvl: int = 3,
) -> FcStrategyTables:
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
        peak_table=peak_table
    )

    valid_entries, stop_loss_series, french_stop = init_signal_stop_loss_tables(
        price_data,
        regime_table,
        peak_table,
        entry_lvls,
        highest_peak_lvl,
        offset_pct=trail_offset_pct,
        r_multiplier=r_multiplier,
        abs_price_data=abs_price_data,
    )

    return FcStrategyTables(
        enhanced_price_data,
        peak_table,
        regime_table,
        valid_entries,
        stop_loss_series,
        french_stop
    )


def init_peak_table(
    price_data: pd.DataFrame, distance_pct, retrace_pct, swing_window, sw_lvl
):
    """initialization of peak table bundled together"""
    swings, peak_table = src.utils.regime.init_swings(
        df=price_data,
        dist_pct=distance_pct,
        retrace_pct=retrace_pct,
        n_num=swing_window,
        lvl=sw_lvl,
    )

    # hi_peak_table = full_peak_lag(swings, ["hi1", "hi2", "hi3"])
    # lo_peak_table = full_peak_lag(swings, ["lo1", "lo2", "lo3"])
    # peak_table = pd.concat([hi_peak_table, lo_peak_table]).reset_index(drop=True)
    # if final_discovery_lag is not None:
    #     null_query = pd.isnull(peak_table.end)
    #     if null_query.sum() == 1:
    #         peak_table.loc[null_query, 'end'] = final_discovery_lag
    #     else:
    #         raise Exception('Only expect to fill one swing')

    return peak_table, swings


def init_regime_table(
    enhanced_price_data: pd.DataFrame,
    sw_lvl,
    standard_dev,
    regime_threshold,
    peak_table,
):
    """initialization of regime table bundled together"""

    data_with_regimes = src.utils.regime.regime_floor_ceiling(
        df=enhanced_price_data,
        peak_table=peak_table,
        sw_lvl=sw_lvl,
        flr="flr",
        clg="clg",
        rg="rg",
        rg_ch="rg_ch",
        stdev=standard_dev,
        threshold=regime_threshold,
    )

    return regime_ranges(data_with_regimes, "rg"), data_with_regimes


def init_signal_stop_loss_tables(
    price_data,
    regime_table,
    peak_table,
    entry_lvls,
    highest_peak_lvl,
    offset_pct,
    r_multiplier,
    abs_price_data,
) -> t.Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    raw_signals = get_all_entry_candidates(
        price_data, regime_table, peak_table, entry_lvls, highest_peak_lvl
    )
    # over_under_by_price = np.sign(price_data.close - abs_price_data.close)
    # over_under_by_signal = over_under_by_price.loc[raw_signals.entry].reset_index(drop=True)
    # signals_filter = (
    #         (over_under_by_signal == raw_signals.dir) |
    #         (over_under_by_signal == 0)
    # )
    # raw_signals = raw_signals.loc[signals_filter].reset_index(drop=True)
    if raw_signals.empty:
        raise NoEntriesError

    return process_signal_data(
        price_data,
        regime_table,
        raw_signals,
        peak_table,
        offset_pct=offset_pct,
        r_multiplier=r_multiplier,
    )


def win_rate_calc(
    price_data: pd.DataFrame,
    signals: pd.DataFrame,
    min_periods,
    round_to=2,
):
    """"""
    price_data = round(price_data, round_to)
    signal_table = pda.SignalTable(signals.copy())
    signal_table.data["trade_count"] = signal_table.counts
    signals_un_pivot = signal_table.unpivot(valid_dates=price_data.index)
    signals_un_pivot = signals_un_pivot.loc[
        ~signals_un_pivot.index.duplicated(keep="last")
    ]
    signals_un_pivot = signals_un_pivot[['dir', 'trade_count']]
    signals_un_pivot = expand_index(signals_un_pivot, price_data.index)
    signals_un_pivot.dir = signals_un_pivot.dir.fillna(0)

    passive_returns_1d = ts.simple_log_returns(price_data.close)
    signals_un_pivot["strategy_returns_1d"] = passive_returns_1d * signals_un_pivot.dir
    # don't use entry date to calculate returns
    signals_un_pivot.loc[signal_table.entry, "strategy_returns_1d"] = 0
    strategy_returns_1d = signals_un_pivot.strategy_returns_1d.copy()
    cumul_returns = ts.cumulative_returns_pct(strategy_returns_1d, min_periods)
    # Cumulative t-stat
    win_count = (
        strategy_returns_1d.loc[strategy_returns_1d > 0]
        .expanding()
        .count()
        .fillna(method="ffill")
    )

    total_count = (
        strategy_returns_1d.loc[strategy_returns_1d != 0]
        .expanding()
        .count()
        .fillna(method="ffill")
    )
    win_rate = (win_count / total_count).fillna(method="ffill")
    return win_rate


def calc_stats(
    price_data: pd.DataFrame,
    signals: pd.DataFrame,
    min_periods: int,
    window: int,
    percentile: float,
    limit,
    round_to=2,
) -> t.Union[None, pd.DataFrame]:
    """
    get full stats of strategy, rolling and expanding
    :param round_to:
    :param freq:
    :param signals:
    :param price_data:
    :param min_periods:
    :param window:
    :param percentile:
    :param limit:
    :return:
    """
    price_data = round(price_data, round_to)
    # TODO include regime returns
    signal_table = pda.SignalTable(signals.copy())
    signal_table.data["trade_count"] = signal_table.counts
    signals_un_pivot = signal_table.unpivot(valid_dates=price_data.index)
    signals_un_pivot = signals_un_pivot.loc[
        ~signals_un_pivot.index.duplicated(keep="last")
    ]
    signals_un_pivot = signals_un_pivot[['dir', 'trade_count']]
    signals_un_pivot = expand_index(signals_un_pivot, price_data.index)
    signals_un_pivot.dir = signals_un_pivot.dir.fillna(0)

    passive_returns_1d = ts.simple_log_returns(price_data.close)
    signals_un_pivot["strategy_returns_1d"] = passive_returns_1d * signals_un_pivot.dir
    # don't use entry date to calculate returns
    signals_un_pivot.loc[signal_table.entry, "strategy_returns_1d"] = 0
    strategy_returns_1d = signals_un_pivot.strategy_returns_1d.copy()

    # Performance
    cumul_passive = ts.cumulative_returns_pct(passive_returns_1d, min_periods)
    cumul_returns = ts.cumulative_returns_pct(strategy_returns_1d, min_periods)
    cumul_excess = cumul_returns - cumul_passive - 1
    cumul_returns_pct = cumul_returns.copy()

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
        strategy_returns_1d.loc[strategy_returns_1d > 0]
        .expanding()
        .count()
        .fillna(method="ffill")
    )

    total_count = (
        strategy_returns_1d.loc[strategy_returns_1d != 0]
        .expanding()
        .count()
        .fillna(method="ffill")
    )

    csr_expanding = ts.common_sense_ratio(pr_expanding, tr_expanding)
    csr_roll = ts.common_sense_ratio(pr_roll, tr_roll)
    csr_roll = expand_index(csr_roll, price_data.index).ffill()

    # Trade Count
    trade_count = signals_un_pivot["trade_count"]
    trade_count = expand_index(trade_count, price_data.index).ffill().fillna(0)
    signal_roll = trade_count.diff(window)

    win_rate = (win_count / total_count).fillna(method="ffill")
    avg_win = profits_expanding / total_count
    avg_loss = losses_expanding / total_count
    edge_expanding = ts.expectancy(win_rate, avg_win, avg_loss).fillna(method="ffill")
    sqn_expanding = ts.t_stat(trade_count, edge_expanding)

    win_roll = strategy_returns_1d.copy()
    win_roll[win_roll <= 0] = np.nan
    win_rate_roll = win_roll.rolling(window, min_periods=0).count() / window
    avg_win_roll = profits_roll / window
    avg_loss_roll = losses_roll / window

    edge_roll = ts.expectancy(
        win_rate=win_rate_roll, avg_win=avg_win_roll, avg_loss=avg_loss_roll
    )
    sqn_roll = ts.t_stat(signal_count=signal_roll, trading_edge=edge_roll)

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

    historical_stat_sheet = pd.DataFrame.from_dict(stat_sheet_dict)
    # historical_stat_sheet = historical_stat_sheet.ffill()

    return historical_stat_sheet


def expand_index(gap_data, full_index):
    """insert indexes into the given gap data"""
    try:
        expanded_idx = gap_data.__class__(index=full_index, columns=gap_data.columns, dtype='float64')
    except AttributeError:
        expanded_idx = gap_data.__class__(index=full_index, dtype='float64')
    expanded_idx.loc[gap_data.index] = gap_data
    return expanded_idx


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
    # main()
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
