"""
this module contains code which ties all aspects of strategy together into a functional model
"""
from dataclasses import dataclass, field
import typing as t
import numpy as np
import pandas as pd
import pandas_accessors.accessors as pda
import pandas_accessors.utils as ts

import src.regime
import src.regime as regime


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
    stop_loss_offset_pct: float,
    r_multiplier: float
):
    """
    set fixed stop for first signal in each regime to the recent lvl 3 peak
    build raw signal table, contains entry signal date and direction of trade
    regimes: start(date), end(date), rg(date)
    TODO
        - add fixed_stop_date to output
        - add trail_stop_date to output
    peaks: start(date: peak location), end(date: peak discovery), type
    :param stop_loss_offset_pct:
    :param r_multiplier:
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

    signal_candidates = signal_candidates[['entry', 'en_px', 'st_px', 'dir', 'trail_stop', 'fixed_stop']]

    if stop_loss_offset_pct == 0:
        pct_from_peak = 1
    else:
        pct_from_peak = 1 - (stop_loss_offset_pct * signal_candidates.dir)
    signal_candidates['fixed_stop_price'] = price.loc[signal_candidates.fixed_stop.values, 'close'].values * pct_from_peak
    signal_candidates['r_pct'] = (
            (signal_candidates.en_px - signal_candidates.fixed_stop_price) / signal_candidates.en_px
    )
    signal_candidates = signal_candidates.loc[
        (signal_candidates.r_pct != 0.00) &
        (abs(signal_candidates.r_pct) < 0.05)
        # (signal_candidates.vlty_break < 40) &
        # (signal_candidates.pct_break < .035)
    ].reset_index(drop=True)
    signal_candidates['target_price'] = (
            signal_candidates.en_px + (signal_candidates.en_px * signal_candidates.r_pct * r_multiplier)
    )

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
        (entry_table.entry <= first_sig.entry)
        & (entry_table.lvl == highest_peak_lvl)
        & (entry_table.type == first_sig.type)
        & (((first_sig.en_px - entry_table.st_px) * first_sig.type) > 0)
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
    stop_calc: ts.TrailStop,
    price: pd.DataFrame,
    trail_stop_date,
    fixed_stop_price,
    entry_date,
    entry_price,
    target_price,
    rg_end_date,
    atr,
) -> t.Tuple[pd.Series, pd.Timestamp, pd.Timestamp, pd.Series, float]:
    """
    trail stop to entry price, then reset to fixed stop price after target price is reached
    :param target_price:
    :param entry_price:
    :param atr:
    :param fixed_stop_price:
    :param rg_end_date:
    :param stop_calc:
    :param entry_date:
    :param price:
    :param trail_stop_date:
    :return:
    """
    _stop_modifier = atr * 2
    # trail_price = stop_calc.get_stop_price(price, trail_stop_date, offset_pct)
    # stop_line = stop_calc.init_trail_stop(price, trail_price, entry_date, rg_end_date)
    stop_line = stop_calc.init_trail_stop(
        price, price.close.loc[trail_stop_date],
        entry_date, rg_end_date
    )
    stop_line = stop_calc.cap_trail_stop(stop_line, entry_price)
    stop_line = stop_calc.init_atr_stop(stop_line, entry_date, rg_end_date, _stop_modifier)

    stop_loss_exit_signal = stop_calc.exit_signal(price, stop_line)
    exit_signal_date = stop_line.loc[stop_loss_exit_signal].first_valid_index()

    target_exit_signal = stop_calc.target_exit_signal(price, target_price)
    partial_exit_date = stop_line.loc[target_exit_signal].first_valid_index()

    if exit_signal_date is None:
        exit_signal_date = rg_end_date

    if partial_exit_date is not None and partial_exit_date < exit_signal_date:
        stop_line.loc[partial_exit_date:] = fixed_stop_price
    else:
        partial_exit_date = np.nan

    # signal is active until signal end date is not the current date
    stop_line = stop_line.loc[:exit_signal_date]

    return stop_line, exit_signal_date, partial_exit_date, stop_loss_exit_signal, fixed_stop_price


def draw_fixed_stop(
    stop_calc: ts.TrailStop,
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
        1: ts.TrailStop(
            pos_price_col="close", neg_price_col="close", cum_extreme="cummax", dir=1
        ),
        -1: ts.TrailStop(
            pos_price_col="close", neg_price_col="close", cum_extreme="cummin", dir=-1
        ),
    }
    valid_entries = pd.DataFrame(columns=entry_candidates.columns.to_list())
    valid_entries['partial_exit_date'] = np.nan
    valid_entries['rg_id'] = np.nan
    stop_lines = []

    french_stop_table = pd.DataFrame(columns=['start', 'end'])
    french_stop_lines = []

    for rg_idx, rg_info in regimes.iterrows():
        stop_calc = trail_map[rg_info.rg]
        start = rg_info.start
        end = rg_info.end
        entry_signal = None

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

            if entry_signal is not None:
                trail_stop_price = r_price_data.close.at[entry_signal.trail_stop]
                rg_entry_candidates = reduce_regime_candidates(
                    rg_entry_candidates,
                    trail_stop_price,
                    entry_signal,
                    rg_info,
                    rg_peak_table
                )

            if rg_entry_candidates.empty:
                break

            entry_signal = rg_entry_candidates.iloc[0]

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
                fixed_stop_price=entry_signal.fixed_stop_price,
                entry_date=entry_signal.entry,
                entry_price=entry_signal.en_px,
                target_price=entry_signal.target_price,
                rg_end_date=end,
                atr=atr
            )
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

            start = exit_signal_date
            if not pd.isna(partial_exit_date):
                if exit_signal_date <= partial_exit_date:
                    # start = rg_price_data.iloc[rg_price_data.index.get_loc(exit_signal_date) + 1].index[0]
                    start = exit_signal_date
                else:
                    # if exit greater than partial exit, then potentially another signal can be added
                    start = partial_exit_date

        partial_exit_entries = valid_entries.loc[
            valid_entries.partial_exit_date.notna() &
            (valid_entries.rg_id == rg_info.name)
        ]

        french_start_date = None
        fs_id = 0
        french_stop_line = None
        french_exit_date = None
        for i, (idx, current_entry) in enumerate(partial_exit_entries.iterrows()):
            if i == 0:
                continue
            prior_entry_price = partial_exit_entries.iloc[i-1].en_px
            if french_start_date is None:
                french_start_date = current_entry.partial_exit_date
                french_stop_line = pda.FrenchStop.init_empty_df(
                    index=r_price_data.loc[french_start_date:].index
                )

            french_stop_line.loc[
                current_entry.partial_exit_date:, 'stop_price'
            ] = prior_entry_price if prior_entry_price < current_entry.en_px else prior_entry_price
            french_stop_line.loc[
                current_entry.partial_exit_date:, 'rg_id'
            ] = rg_info.name
            french_exit_signal = stop_calc.exit_signal(rg_price_data, french_stop_line.stop_price)
            french_exit_date = french_stop_line.loc[french_exit_signal].first_valid_index()

            if french_exit_date is not None:
                french_stop_line.loc[french_exit_date:] = np.nan
                french_stop_table.at[fs_id] = pd.Series(data={'start': french_start_date, 'end': french_exit_date})
                query_prior_partial_exits = (
                    valid_entries.partial_exit_date.notna() &
                    (valid_entries.rg_id == rg_info.name) &
                    (valid_entries.partial_exit_date < french_exit_date) &
                    (valid_entries.partial_exit_date > french_stop_table.iloc[-1].end)
                )
                valid_entries.loc[query_prior_partial_exits, 'exit_signal_date'] = french_exit_date
                french_stop_lines.append(french_stop_line.loc[french_start_date: french_exit_date])

                french_start_date = None
                fs_id += 1

        if isinstance(french_stop_line, pd.DataFrame) and french_exit_date is None:
            french_stop_lines.append(french_stop_line)

    french_stop_line = pda.FrenchStop.init_empty_df(index=r_price_data.index)
    if len(french_stop_lines) > 0:
        combined_french_line = pd.concat(french_stop_lines)
        french_stop_line.loc[combined_french_line.index] = combined_french_line

    if len(stop_lines) > 0:
        stop_prices = pd.concat(stop_lines)
    else:
        stop_prices = pd.Series()
        assert valid_entries.empty  # TODO not sure if this can ever be false, so notify me in case it ever is
        valid_entries = pda.SignalTable.init_empty_df()

    return valid_entries, stop_prices, french_stop_line


def reduce_regime_candidates(
        rg_entry_candidates: pd.DataFrame,
        entry_limit: t.Union[float, None],
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
    try:
        # filter for entries that are within the previous entry
        new_rg_entry_candidates = rg_entry_candidates.loc[
            ((rg_entry_candidates.st_px.values - entry_limit) * rg_info.rg) > 0
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
        _sw_after_entry = rg_peak_table.loc[
            rg_peak_table.start > entry_signal.entry
        ]
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
                    (rg_entry_candidates.entry >= _sw_after_entry.iloc[0].end) &
                    (((rg_entry_candidates.en_px - _sw_after_entry.iloc[0].st_px) * rg_info.rg) > 0)
                ]
    return new_rg_entry_candidates


@dataclass
class FcStrategyTables:
    enhanced_price_data: pd.DataFrame
    peak_table: pd.DataFrame
    regime_table: pd.DataFrame
    valid_entries: pd.DataFrame = field(default=None)
    stop_loss_series: pd.Series = field(default=None)
    french_stop: pd.DataFrame = field(default=None)
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


def fc_scale_strategy_live(
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
    find_retest_swing: bool = True,
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

    if find_retest_swing:
        retest_swing = src.regime.retest_from_latest_base_swing(
            swings=peak_table,
            price_data=price_data,
            retest_swing_lvl=1,
            base_swing_lvl=3,
        )
        if retest_swing is not None:
            retest_swing_df = pd.DataFrame(retest_swing).transpose()
            peak_table = pd.concat([peak_table, retest_swing_df], ignore_index=True)

            swing_type = 'lo' if retest_swing['type'] == 1 else 'hi'
            swing_col = f'{swing_type}{retest_swing.lvl}'
            enhanced_price_data.at[retest_swing.start, swing_col] = retest_swing.st_px

    standard_dev = price_data.close.rolling(swing_window).std(ddof=0)

    regime_table, enhanced_price_data = init_regime_table(
        enhanced_price_data=enhanced_price_data,
        sw_lvl=sw_lvl,
        standard_dev=standard_dev,
        regime_threshold=regime_threshold,
        peak_table=peak_table
    )

    return FcStrategyTables(
        enhanced_price_data,
        peak_table,
        regime_table
    )




def init_peak_table(
    price_data: pd.DataFrame, distance_pct, retrace_pct, swing_window, sw_lvl
):
    """initialization of peak table bundled together"""
    swings, peak_table = regime.init_swings(
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

    data_with_regimes = regime.regime_floor_ceiling(
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
    regime_table = regime_ranges(data_with_regimes, "rg")
    regime_table['type'] = 'fc'
    return regime_table, data_with_regimes


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
        price_data, regime_table, peak_table, entry_lvls, highest_peak_lvl, offset_pct, r_multiplier
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

