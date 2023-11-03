import src.regime.utils as utils
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import pytest
from sqlalchemy import create_engine, text
import typing as t
import numpy as np

symbol_query = 'SELECT * FROM {table} where {table}.symbol = \'{symbol}\''

def get_test_data() -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get test data
    :return:
    """
    return pd.read_pickle('stock_data.pkl'), pd.read_pickle('peak.pkl'), pd.read_pickle('regime.pkl')


def test_add_peak_regime_data():
    """
    Test add_peak_regime_data, that the peak data was expanded properly into the time series
    """
    stock_table, peak_table, regime_table = get_test_data()
    result = utils.add_peak_regime_data(stock_table, regime_table, peak_table)
    for peak in peak_table.itertuples():
        swing_str = 'lo' if peak.type == 1 else 'hi'
        peak_col_name = f'{swing_str}{peak.lvl}'
        discovery_peak_col_name = f'd{swing_str}{peak.lvl}'

        assert result[peak_col_name].iloc[peak.start] == peak.st_px
        assert result[peak_col_name].iloc[peak.start] == result.close.iloc[peak.start]
        assert result[discovery_peak_col_name].iloc[peak.end] == peak.en_px
        assert result[discovery_peak_col_name].iloc[peak.end] == result.close.iloc[peak.end]


def test_retest_from_latest_base_swing():
    """
    Test retest_from_latest_base_swing
    :return:
    """
    stock_table, peak_table, regime_table = get_test_data()
    result = utils.retest_from_latest_base_swing(
        peak_table,
        stock_table,
        regime_table.rg.iloc[-1],
        base_swing_lvl=3
    )
    assert result is not None


def test_find_all_retest_swing():
    stock_table, peak_table, regime_table = get_test_data()
    # base swing is the latest swing
    base_swing_lvl: t.Literal[2, 3] = 3
    retest_swing_lvl: t.Literal[1, 2] = 1

    base_swing = peak_table.loc[peak_table.lvl == base_swing_lvl].iloc[-1]
    latest_swing = utils.retest_from_latest_base_swing(
        peak_table,
        stock_table,
        regime_table.rg.iloc[-1],
        base_swing_lvl=base_swing_lvl
    )

    result = utils.find_all_retest_swing(
        peak_table,
        base_swing,
        stock_table,
        regime_table.rg.iloc[-1],
        retest_swing_lvl=retest_swing_lvl,
    )

    if latest_swing is not None:
        assert not result.empty
        # assert that latest swing is in result
        assert latest_swing.start in result.start.values
        assert latest_swing.end in result.end.values
        assert latest_swing.type in result.type.values
        assert latest_swing.lvl in result.lvl.values
        assert latest_swing.st_px in result.st_px.values
        assert latest_swing.en_px in result.en_px.values


@pytest.fixture
def test_data():
    # Create a DataFrame with some test data
    data = pd.DataFrame({
        'high': [1, 2, 3, 2, 1, 4, 5, 6, 5, 4, 3, 2, 1, 0, 1],
        'low': [1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
        'close': [1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 4]
    })
    return data

def test_historical_swings_values(test_data):
    # Call the function with the test data
    result = utils.historical_swings(test_data)

    # Check the values in the new columns
    assert (result['hi1'] == [1, 2, 3, np.nan, np.nan, 4, 5, 6, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]).all()
    assert (result['lo1'] == [1, np.nan, 1, 2, 3, np.nan, 1, np.nan, 1, 2, 3, 4, 5, 6, 7]).all()
    assert (result['hi2'] == [np.nan, np.nan, np.nan, np.nan, np.nan, 4, 5, 6, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]).all()
    assert (result['lo2'] == [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, np.nan, 1, 2, 3, 4, 5, 6, 7]).all()
    assert (result['hi3'] == [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1]).all()
    assert (result['lo3'] == [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1]).all()


def test_find_swings_with_last_known_peak_data(test_data):
    # Create a DataFrame with the last known peak data
    last_known_peak_data = pd.DataFrame({
        'type': [-1],
        'lvl': [1],
        'start': [2]
    })

    # Call the function with the test data and the last known peak data
    result = utils.find_swings(test_data, 1, -1, last_known_peak_data)

    # Check the values in the result
    assert (result == [np.nan, np.nan, 3, 2, 1]).all()
