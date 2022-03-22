import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
import src.floor_ceiling_regime as sfcr
from datetime import datetime, timedelta
import typing as t
import src.utils.regime as regime
from src.pd_accessors import PriceTable
import src.pd_accessors as pda


class StockDataGetter:
    def __init__(self, data_getter_method: t.Callable):
        self._data_getter_method = data_getter_method
        self.strategy_exceptions = []
        self.no_data = []

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        data = self._data_getter_method(symbol)
        if data is None:
            return pd.DataFrame()
        # data = data.rename(
        #     columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
        # )
        PriceTable(data, symbol)
        return data[["open", "high", "low", "close"]]

    def yield_strategy_data(
        self,
        bench_symbol,
        symbols: t.List[str],
        strategy: t.Callable[[pd.DataFrame, pd.DataFrame], t.Any],
        expected_exceptions: t.Tuple
    ) -> t.Union[t.Tuple[pd.DataFrame, sfcr.FcStrategyTables], t.Tuple[None, None]]:
        for i, symbol in enumerate(symbols):
            print(f"({i}/{len(symbols)}) {symbol}")  # , end='\r')
            bench_data = self.get_stock_data(bench_symbol)
            symbol_data = self.get_stock_data(symbol)
            strategy_data = None
            if not symbol_data.empty:
                try:
                    strategy_data = strategy(symbol_data, bench_data)
                except expected_exceptions as e:
                    self.strategy_exceptions.append(e)
            else:
                self.no_data.append(symbol)
                symbol_data = None

            yield symbol, symbol_data, bench_data, strategy_data


def enhanced_price_data_plot(data, ax=None):
    _open = "open"
    _high = "high"
    _low = "low"
    _close = "close"
    a = data[
        [_close, "hi3", "lo3", "clg", "flr", "rg_ch", "hi2", "lo2", 'stop_loss', 'french_stop']
    ].plot(
        style=["grey", "ro", "go", "kv", "k^", "c:", "r.", "g."],
        figsize=(15, 5),
        grid=True,
        use_index=False,
        ax=ax
    )

    data["rg"].plot(
        style=["b-."],
        # figsize=(15, 5),
        # marker='o',
        secondary_y=["rg"],
        ax=a,
        use_index=False,
    )


def rolling_plot(
    price_data: pd.DataFrame,
    ndf,
    stop_loss_t,
    ticker,
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
            res = sfcr.fc_scale_strategy(d)
            d = res.enhanced_price_data

            if fp_rg is None:
                fp_rg = d.rg.copy()
                fp_rg = fp_rg.fillna(0)
                hi2_lag = d.hi2.copy()
                lo2_lag = d.lo2.copy()
            else:
                fp_rg = fp_rg.reindex(d.rg.index)
                new_val = d.rg.loc[pd.isna(fp_rg)][0]
                fp_rg.loc[idx] = new_val

                hi2_lag = sfcr.update_sw_lag(hi2_lag, d.hi2, hi2_discovery_dts)
                lo2_lag = sfcr.update_sw_lag(lo2_lag, d.lo2, lo2_discovery_dts)

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
                        .iloc[index - plot_window:]
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
    PriceTable(data, symbol)
    data = data[["open", "high", "low", "close"]]
    return data


def get_cached_data(symbol, days, interval) -> pd.DataFrame:
    """get price data from local storage"""
    file_name = f"{symbol}_{interval}_{days}d.csv"
    data = pd.read_csv(rf"..\strategy_output\price_data\{file_name}")
    data_cols = data.columns.to_list()

    if "Date" in data_cols:
        date_col = "Date"
    elif "Datetime" in data_cols:
        date_col = "Datetime"
    else:
        raise

    data[date_col] = pd.to_datetime(data[date_col])
    data = data.set_index(data[date_col])

    return data


def data_to_relative(data, bench_df):
    """make relative function compatible with my code"""
    bm_close = 'bm_close'
    bench_df = bench_df.rename(columns={'close': bm_close})
    working_data = regime.relative(
        df=data, bm_df=bench_df, bm_col=bm_close, rebase=True
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
    return working_data


def get_wikipedia_stocks(url):
    """get stock data (names, sectors, etc) from wikipedia"""
    wiki_df = pd.read_html(url)[0]
    tickers_list = list(wiki_df["Symbol"])
    return tickers_list[:], wiki_df


def yf_download_data(tickers, days, interval) -> pd.DataFrame:
    """download stock data from yf and concat to big price history file"""
    data = yf.download(
        tickers,
        start=(datetime.now() - timedelta(days=days)),
        end=datetime.now(),
        interval=interval
    )

    data = data.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
    )
    return data[["open", "high", "low", "close"]]


def run_scanner(scanner, stat_calculator, relative_side_only=True):
    stat_overview = pd.DataFrame()
    entry_data = {}
    strategy_data_lookup = {}
    for symbol, symbol_data, bench_data, strategy_data in scanner:
        if symbol_data is None or strategy_data is None:
            continue

        signals = strategy_data.valid_entries.copy()

        # only process long for outperformers, short for underperformers
        if relative_side_only:
            symbol_data['over_under'] = np.where((symbol_data.close-strategy_data.enhanced_price_data.close) > 0, -1, 1)
            signals_filter = symbol_data.over_under.loc[signals.entry].reset_index(drop=True) == signals.dir
            signals = signals.loc[signals_filter].reset_index(drop=True)
            if signals.empty:
                continue

        stat_sheet_historical = stat_calculator(symbol_data, signals)
        if stat_sheet_historical is None:
            continue

        strategy_data.stat_historical = stat_sheet_historical
        strategy_data_lookup[symbol] = strategy_data

        # TODO fixed? TODO -2 because yf gives to the minute data despite before bar closes
        stat_sheet_final_scores = stat_sheet_historical.iloc[-1].copy()
        stat_sheet_final_scores['symbol'] = symbol
        signal_table = pda.SignalTable(signals)
        price_table = PriceTable(symbol_data, '')

        entries = signals
        entries['abs_entry'] = signal_table.entry_prices(price_table)
        entries['abs_exit'] = signal_table.exit_prices(price_table)
        entries['abs_return'] = signal_table.static_returns(price_table)
        entries['partial_exit'] = signal_table.partial_exit_prices(price_table)

        risk = signal_table.pyramid_all(-0.0075)

        # signals['win_roll'] = stat_sheet_historical.win_roll.loc[signals.entry].values
        # signals['avg_win_roll'] = stat_sheet_historical.avg_win_roll.loc[signals.entry].values
        # signals['avg_loss_roll'] = stat_sheet_historical.avg_loss_roll.loc[signals.entry].values
        signals['risk'] = risk

        # win_rate_query = (signals.win_roll > .5)
        # signals.loc[win_rate_query, 'risk'] = smm.other_kelly(
        #     win_rate=signals['win_roll'].loc[win_rate_query],
        #     avg_win=signals['avg_win_roll'].loc[win_rate_query],
        #     avg_loss=signals['avg_win_roll'].loc[win_rate_query],
        # )
        # signals['risk'] = signals['risk'] * .5
        signals['shares'] = signal_table.eqty_risk_shares(strategy_data.enhanced_price_data, 30000, signals['risk'])
        entries['partial_profit'] = (entries.partial_exit - entries.abs_entry) * (entries.shares * (2 / 3))
        entries['rem_profit'] = (entries.abs_exit - entries.abs_entry) * (entries.shares * (1 / 3))
        entries['partial_total'] = entries.partial_profit + entries.rem_profit
        entries['no_partial_total'] = (entries.abs_exit - entries.abs_entry) * entries.shares
        entries['total'] = entries['partial_total']
        entries.loc[pd.isna(entries.total), 'total'] = entries.loc[pd.isna(entries.total), 'no_partial_total']
        entries['total'] = entries.total.cumsum()

        entry_data[symbol] = entries

        # fig, axes = plt.subplots(nrows=3, ncols=1)

        # strategy_data.enhanced_price_data['stop_loss'] = strategy_data.stop_loss_series
        # strategy_data.enhanced_price_data['french_stop'] = strategy_data.french_stop.stop_price
        # enhanced_price_data_plot(strategy_data.enhanced_price_data, ax=axes[0])

        # bench_data["close"] = bench_data["close"].div(bench_data["close"][0])
        # symbol_data['stop_loss'] = strategy_data.enhanced_price_data['stop_loss'] * bench_data.close
        # symbol_data[['close', 'stop_loss']].plot(use_index=False, ax=axes[1])
        # pd.DataFrame({
        #     'abs_rel_delta': abs(symbol_data.close - strategy_data.enhanced_price_data.close),
        #     '': abs(symbol_data.close - symbol_data.stop_loss)
        # }).plot(use_index=False, ax=axes[2])
        # plt.show()
        stat_sheet_final_scores['weight_total'] = entries.total.iloc[-1]
        stat_overview = pd.concat([stat_overview, stat_sheet_final_scores.to_frame().transpose()], ignore_index=True)

    stat_overview = stat_overview.reset_index(drop=True)
    return stat_overview, strategy_data_lookup
