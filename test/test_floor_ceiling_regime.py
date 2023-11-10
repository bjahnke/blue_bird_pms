import pandas as pd
import pytest
from src.floor_ceiling_regime.floor_ceiling_regime import regime_ranges
import src.regime.utils as utils

class TestRegimeRanges:
    """
    Test the regime_ranges function
    """
    @pytest.fixture
    def data_with_regime(self):
        """
        load pickle data
        """
        return pd.read_pickle('.\\test\\regime.pkl'), pd.read_pickle('.\\test\\stock_data.pkl')
    
    @pytest.fixture
    def data_with_regime(self):
        """
        load pickle data
        """
        stock_data = pd.read_pickle('.\\test\\stock_data.pkl')
        stock_data = stock_data.loc[stock_data.is_relative == False].copy()
        stock_data.index = stock_data.bar_number
        stock_data.index.name = 'index'
        
        stdev = stock_data.close.rolling(63).std(ddof=0)
        px, swings = utils.init_swings(stock_data)
        regime_data = utils.regime_floor_ceiling(
            df=px,
            stdev=stdev,
            threshold=1.5,
            peak_table=swings,
        )
        return regime_data
    
    @pytest.fixture
    def rg_table(self, data_with_regime):
        """
        load pickle data
        """
        return pd.read_pickle('.\\test\\regime.pkl')

    def test_regime_ranges1(self, data_with_regime, rg_table):
        # Test case 1: all positive values
        regime_table = regime_ranges(data_with_regime, 'rg')
        pd.testing.assert_frame_equal(regime_table, rg_table)

    def test_regime_ranges2(self, data_with_regime):
        # Test case 2: all negative values
        df = pd.DataFrame({'rg': [-1, -1, -1, -1]})
        expected_output = pd.DataFrame({'start': [0], 'end': [3], 'rg': [-1]})
        pd.testing.assert_frame_equal(regime_ranges(df, 'rg'), expected_output)
    
    def test_regime_ranges3(self, data_with_regime):
        # Test case 3: alternating positive and negative values
        df = pd.DataFrame({'rg': [1, -1, 1, -1]})
        expected_output = pd.DataFrame({'start': [0, 1, 2, 3], 'end': [1, 2, 3, 4], 'rg': [1, -1, 1, -1]})
        pd.testing.assert_frame_equal(regime_ranges(df, 'rg'), expected_output)
    
    def test_regime_ranges4(self, data_with_regime):
        # Test case 4: single positive value
        df = pd.DataFrame({'rg': [1]})
        expected_output = pd.DataFrame({'start': [0], 'end': [0], 'rg': [1]})
        pd.testing.assert_frame_equal(regime_ranges(df, 'rg'), expected_output)
    
    def test_regime_ranges5(self, data_with_regime):
        # Test case 5: single negative value
        df = pd.DataFrame({'rg': [-1]})
        expected_output = pd.DataFrame({'start': [0], 'end': [0], 'rg': [-1]})
        pd.testing.assert_frame_equal(regime_ranges(df, 'rg'), expected_output)
    
    def test_regime_ranges6(self, data_with_regime):
        # Test case 6: empty dataframe
        df = pd.DataFrame({'rg': []})
        expected_output = pd.DataFrame({'start': [], 'end': [], 'rg': []})
        pd.testing.assert_frame_equal(regime_ranges(df, 'rg'), expected_output)


class TestRegimeFloorCeiling:
    @pytest.fixture
    def data_with_regime(self):
        """
        load pickle data
        """
        stock_data = pd.read_pickle('.\\test\\stock_data.pkl')
        stock_data = stock_data.loc[stock_data.is_relative == False].copy()
        stock_data.index = stock_data.bar_number
        stock_data.index.name = 'index'
        
        stdev = stock_data.close.rolling(63).std(ddof=0)
        px, swings = utils.init_swings(stock_data)
        # regime_data = utils.regime_floor_ceiling(
        #     df=px,
        #     stdev=stdev,
        #     threshold=1.5,
        #     peak_table=swings,
        # )
        return px, swings, stdev
    
    def test_regime_floor_ceiling1(self, data_with_regime):
        # Test case 1: all positive values
        px, swings, stdev = data_with_regime
        regime_data, fc_data = utils.regime_floor_ceiling(
            df=px,
            stdev=stdev,
            threshold=1.5,
            peak_table=swings,
        )
        expected = pd.read_pickle('.\\test\\price_regime.pkl')
        regime = pd.read_pickle('.\\test\\regime.pkl')
        pd.testing.assert_frame_equal(regime_data, expected)