import src.utils.regime as regime
import src.scanner as scanner
import src.floor_ceiling_regime as sfcr
import json
import typing as t
import pandas as pd
from matplotlib import pyplot as plt
import pickle


class DataLoader:
    def __init__(self, data: t.Dict, base_file_path: str):
        self._data = data
        self._base_file_path = base_file_path

    @classmethod
    def init_from_paths(cls, data_path, base_path):
        with open(data_path, 'r') as data_fp:
            data = json.load(data_fp)
        with open(base_path, 'r') as base_fp:
            base = json.load(base_fp)['base']
        return cls(data, base)

    @property
    def files(self):
        return self._data['files']

    @property
    def base(self):
        return self._base_file_path

    def file_path(self, file_name):
        return fr'{self.base}\{file_name}'

    def history_path(self, bench: str, interval: str):
        return self.file_path(self.files['history'][str(interval)][bench])

    def bench_path(self, bench: str, interval: str):
        return self.file_path(self.files['bench'][str(interval)][bench])


def main_re_download_data():
    """re download stock and bench data, write to locations specified in paths.json"""
    sp500_wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ticks, _ = scanner.get_wikipedia_stocks(sp500_wiki)
    data_loader = DataLoader.init_from_paths('other.json', 'base.json')
    bench = 'SPY'
    days = 59
    interval = 15
    interval_str = f'{interval}m'
    history_path = data_loader.history_path(bench=bench, interval=interval_str)
    bench_path = data_loader.bench_path(bench=bench, interval=interval_str)
    downloaded_data = scanner.yf_download_data(ticks, days, interval_str)
    bench_data = scanner.yf_get_stock_data('SPY', days, interval_str)
    relative = regime.simple_relative(downloaded_data, bench_data.close)
    # TODO add relative data to schema
    relative.to_csv(f'{data_loader.base}\\spy_relative_history_{interval_str}.csv')
    downloaded_data.to_csv(history_path)
    bench_data.to_csv(bench_path)


def run_backtest():
    sp500_wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ticks, _ = scanner.get_wikipedia_stocks(sp500_wiki)
    days = 59
    interval = 15
    interval_str = f'{interval}m'
    bench_str = "SPY"

    data_loader = DataLoader.init_from_paths('other.json', 'base.json')
    price_data = pd.read_csv(data_loader.history_path(bench_str, interval_str), index_col=0, header=[0, 1]).iloc[
                 1:].astype('float64')
    price_data.index = pd.to_datetime(price_data.index, utc=True)
    price_glob = PriceGlob(price_data).swap_level()
    bench = pd.read_csv(data_loader.bench_path(bench_str, interval_str), index_col=0).astype('float64')
    bench.index = pd.to_datetime(bench.index, utc=True)
    scan = scanner.StockDataGetter(
        # data_getter_method=lambda s: scanner.yf_get_stock_data(s, days=days, interval=interval_str),
        data_getter_method=lambda s: price_glob.get_prices(s),
    ).yield_strategy_data(
        bench_symbol="SPY",
        # symbols=ticks,
        symbols=['ADBE'],
        strategy=lambda pdf_, _: (
            sfcr.fc_scale_strategy(
                price_data=scanner.data_to_relative(pdf_, bench),
                abs_price_data=pdf_,
                distance_pct=0.05,
                retrace_pct=0.05,
                swing_window=63,
                sw_lvl=3,
                regime_threshold=0.5,
                trail_offset_pct=0.005,
                r_multiplier=1.5,
                entry_lvls=None,
                highest_peak_lvl=3,
            )
        ),
        expected_exceptions=(regime.NotEnoughDataError, sfcr.NoEntriesError)
    )
    stat_overview_, strategy_lookup = scanner.run_scanner(
        scanner=scan,
        stat_calculator=lambda data_, entry_signals_: sfcr.calc_stats(
            data_,
            entry_signals_,
            min_periods=50,
            window=200,
            percentile=0.05,
            limit=5,
            freq=f'{interval}T',
            # freq='1D',
            # freq='5T',
        ),
        relative_side_only=True
    )
    # stat_overview_ = stat_overview_.sort_values('risk_adj_returns_roll', axis=1, ascending=False)
    stat_overview_.to_csv(data_loader.file_path(f'stat_overview_{interval_str}.csv'))
    pkl_fp = data_loader.file_path('strategy_lookup.pkl')
    with open(pkl_fp, 'wb') as f:
        pickle.dump(strategy_lookup, f)
    print('done')


class PriceGlob:
    """table of price data for multiple tickers, columns multi index is ticker and (open, high, low, close)"""

    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

    def swap_level(self):
        """swap the level of columns, allow for searching all price data by ticker"""
        return self.__class__(self._data.swaplevel(-1, -2, axis=1))

    def relative_rebased(self, bench_close):
        """relative close prices, rebased to 1"""
        bench_close_frame = self._data.copy()
        bench_close_frame.close = bench_close
        return self._data.close.div(bench_close_frame.close * self._data.close.iloc[0]) * bench_close_frame.close.iloc[0]

    def relative_rebase_all(self, bench_close):
        """all relative prices, rebased to 1"""
        bench_close_frame = self._data.copy()
        bench_close_frame.open = bench_close
        bench_close_frame.high = bench_close
        bench_close_frame.low = bench_close
        bench_close_frame.close = bench_close
        return self._data.div(bench_close_frame * self._data.iloc[0]) * bench_close_frame.iloc[0]

    def relative_all(self, bench_close, rebase=True):
        """all relative, rebased to stock price"""
        bench_close_frame = self._data.copy()
        bench_close_frame.open = bench_close
        bench_close_frame.high = bench_close
        bench_close_frame.low = bench_close
        bench_close_frame.close = bench_close
        if rebase is True:
            bench_close_frame = bench_close_frame.div(bench_close_frame.iloc[0])

        return self.__class__(self._data.div(bench_close_frame))

    def get_prices(self, column):
        d = None
        try:
            d = self._data[column]
        except KeyError:
            pass
        return d


def with_relative(data, bm_close, rebase=True):
    ndf = data.copy().to_frame()
    ndf['r'] = regime.simple_relative(data, bm_close, rebase)
    return ndf


def quick_plot(idx, price_data, r_close, bench):
    sorted_close = r_close.iloc[-1].sort_values().dropna()
    with_relative(
        price_data.close[
            sorted_close.index[idx]
        ],
        bench.close
    ).plot(figsize=(8, 3), use_index=False)


def main_roll_scan():
    _data_loader = DataLoader.init_from_paths('other.json', 'base.json')
    _strategy_path = _data_loader.file_path('strategy_lookup.pkl')
    with open(_strategy_path, 'rb') as f:
        _strategy_lookup = pickle.load(f)

    _bench_str = 'SPY'
    _interval = '15m'
    _price_data = pd.read_csv(_data_loader.history_path(_bench_str, _interval), index_col=0, header=[0, 1]).iloc[
                  1:].astype('float64')
    _price_data.index = pd.to_datetime(_price_data.index, utc=True)
    _bench = pd.read_csv(_data_loader.bench_path(_bench_str, _interval), index_col=0).astype('float64')
    _bench.index = pd.to_datetime(_bench.index, utc=True)
    _relative_rebased = PriceGlob(_price_data).relative_rebased(_bench.close)
    _strategy_overview = pd.read_csv(_data_loader.file_path('stat_overview_15m.csv'))

    _price_data_by_symbol = PriceGlob(_price_data).swap_level()
    _symbol = 'CVX'
    scanner.rolling_plot(
        _price_data_by_symbol.data[_symbol],
        _strategy_lookup[_symbol].enhanced_price_data[['open', 'high', 'low', 'close']],
        _strategy_lookup[_symbol].stop_loss_series,
        peak_table=_strategy_lookup[_symbol].peak_table,
        ticker=_symbol,
        plot_loop=True,
        plot_rolling_lag=False
    )
    print('d')


if __name__ == '__main__':
    run_backtest()


