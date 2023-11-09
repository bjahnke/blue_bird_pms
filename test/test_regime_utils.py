import src.regime.utils as utils
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import pytest
from sqlalchemy import create_engine, text
import typing as t
import numpy as np
import os 

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

NAN_1 = [np.nan]
NAN_2 = NAN_1 * 3
NAN_3 = NAN_1 * 7

class SwMocks:
    """
    Test data for swing tests
    """
    
    class Price:
        SW_HI_1 = [1,2,1]
        SW_HI_2 = SW_HI_1 + [3] + SW_HI_1
        SW_HI_3 = SW_HI_2 + [4] + SW_HI_2
        SW_LO_1 = [4,3,4]
        SW_LO_2 = SW_LO_1 + [2] + SW_LO_1
        SW_LO_3 = SW_LO_2 + [1] + SW_LO_2

    class Peak:
        SW_HI_1 = NAN_1 + [1] + NAN_1
        SW_HI_2 = NAN_2 + [3] + NAN_2
        SW_HI_3 = NAN_3 + [4] + NAN_3
        SW_LO_1 = NAN_1 + [3] + NAN_1
        SW_LO_2 = NAN_2 + [2] + NAN_2
        SW_LO_3 = NAN_3 + [1] + NAN_3
        


@pytest.fixture
def test_data():
    # Create a DataFrame with some test data
    data = pd.DataFrame({
        'close': SwMocks.Price.SW_HI_3 + SwMocks.Price.SW_LO_3,
    })
    return data


def test_init_compare():
    # Create a DataFrame with some test data
    swings = pd.Series([1, 2, 3, 3, 2, 1, 2, 3, 2, 1])

    # Call the function with the test data
    result = utils.init_compare(swings, True)

    # the duplicate three should be ignored
    # Check the values in the new DataFrame
    assert (result['prior'] == [1, 2, 3, 2, 1, 2, 3]).all()
    assert (result['sw'] == [2, 3, 2, 1, 2, 3, 2]).all()
    assert (result['next'] == [3, 2, 1, 2, 3, 2, 1]).all()


def test_find_swings():
    # Create a DataFrame with some test data
    swings = pd.Series([1, 2, 3, 2, 1, 2, 3, 2, 1])

    # Call the function with the test data
    result = utils.find_swings(swings, 1, -1)

    # Check the values in the new DataFrame
    assert (result == [3, 3]).all()
    assert (result.index == [2, 6]).all()