import pandas as pd
import pytest
import src.regime.utils as utils

class TestFindSwings:
    """
    A collection of unit tests for the `find_swings` function in the `utils` module.
    """

    @pytest.fixture
    def swings(self):
        return pd.Series([1, 2, 3, 2, 1, 2, 3, 2, 1])

    def test_find_swings_no_last_known_peak_data(self, swings):
        result = utils.find_swings(swings, 1, -1)
        assert (result == [3, 3]).all()
        assert (result.index == [2, 6]).all()

    def test_find_swings_with_last_known_peak_data(self, swings):
        last_known_peak_data = pd.DataFrame({
            'type': [-1],
            'lvl': [1],
            'start': [2]
        })
        result = utils.find_swings(swings, 1, -1, last_known_peak_data)
        assert (result == [3]).all()
        assert (result.index == [6]).all()

    def test_find_swings_with_no_swings(self):
        swings = pd.Series([1, 1, 1, 1, 1])
        result = utils.find_swings(swings, 1, -1)
        assert result.empty

    def test_find_swings_with_one_swing(self):
        swings = pd.Series([1, 2, 1])
        result = utils.find_swings(swings, 1, -1)
        assert (result == [2]).all()
        assert (result.index == [1]).all()

        swings = pd.Series([2, 1, 2])
        result = utils.find_swings(swings, 1, 1)
        assert (result == [1]).all()
        assert (result.index == [1]).all()

    def test_find_swings_with_multiple_peaks_at_same_level(self):
        swings = pd.Series([1, 2, 3, 2, 1, 2, 3, 2, 1])

        result = utils.find_swings(swings, 1, -1)
        assert (result == [3, 3]).all()
        assert (result.index == [2, 6]).all()


class TestFullPeakLag:
    """
    A collection of unit tests for the `full_peak_lag` function in the `utils` module.
    """

    # add fixture for data with one peak
    @pytest.fixture
    def data_with_one_peak(self):
        return pd.DataFrame({
            'close': [1.0, 2.0, 3.0, 2.0, 1.0],
            'hi1': [None, None, 3.0, None, None],
            'lo1': [None, None, None, None, None],
            'hi2': [None, None, None, None, None],
            'lo2': [None, None, None, None, None],
            'hi3': [None, None, None, None, None],
            'lo3': [None, None, None, None, None],
        })
    
    @pytest.fixture
    def data_with_multiple_peaks(self):
        return pd.DataFrame({
            'close': [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            'hi1': [None, None, 3.0, None, None, None, 3.0, None, None],
            'lo1': [None, None, None, None, None, None, None, None, None],
            'hi2': [None, None, None, None, None, None, None, None, None],
            'lo2': [None, None, None, None, None, None, None, None, None],
            'hi3': [None, None, None, None, None, None, None, None, None],
            'lo3': [None, None, None, None, None, None, None, None, None],
        })
    

    @pytest.fixture
    def data_with_three_level_peaks(self):
        # data has level 1,2,3 peaks
        return pd.DataFrame({
            'close': [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 4.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0],
            'hi1': [None, 2.0, None, 3.0, None, 2.0, None, 4.0, None, 2.0, None, 3.0, None, 2.0, None],
            'lo1': [None, None, 1.0, None, 1.0, None, 1.0, None, 1.0, None, 1.0, None, 1.0, None, 1.0],
            'hi2': [None, None, None, 3.0, None, None, None, 4.0, None, None, None, 3.0, None, None, None],
            'lo2': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            'hi3': [None, None, None, None, None, None, None, 4.0, None, None, None, None, None, None, None],
            'lo3': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        })


    def test_full_peak_lag_with_no_data(self):
        """
        Test the full_peak_lag function when given an empty DataFrame as input.
        """
        data = pd.DataFrame()
        asc_peaks = ['hi1', 'hi2', 'hi3']
        res = utils.full_peak_lag(data, asc_peaks)
        assert res.empty

    def test_full_peak_lag_with_one_peak():
        """
        Test full_peak_lag function with a DataFrame containing one peak.

        The function creates a DataFrame with one peak and calls the full_peak_lag function with the DataFrame and a list of
        ascending peaks. The expected result is a DataFrame with the start and end indices of the peak, the peak type, level,
        start price, and end price. The function uses pd.testing.assert_frame_equal to compare the expected result with the
        actual result.
        """
    def test_full_peak_lag_with_one_peak(self, data_with_one_peak):
        data = data_with_one_peak
        asc_peaks = ['hi1']
        result = utils.full_peak_lag(data, asc_peaks)
        expected = pd.DataFrame({
            'start': [2],
            'end': [3],
            'type': [-1],
            'lvl': [1],
            'st_px': [3.0],
            'en_px': [2.0]
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_full_peak_lag_with_multiple_peaks(self, data_with_multiple_peaks):
        """
        Test full_peak_lag function with multiple peaks in ascending order.
        """
        data = data_with_multiple_peaks
        asc_peaks = ['hi1', 'hi2', 'hi3']
        result = utils.full_peak_lag(data, asc_peaks)
        expected = pd.DataFrame({
            'start': [2, 6],
            'end': [3, 7],
            'type': [-1, -1],
            'lvl': [1, 1],
            'st_px': [3.0, 3.0],
            'en_px': [2.0, 2.0]
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_full_peak_lag_with_no_peaks(self):
        data = pd.DataFrame({
            'close': [1, 2, 3, 2, 1],
            'hi1': [None, None, None, None, None]
        })
        asc_peaks = ['hi1']
        result = utils.full_peak_lag(data, asc_peaks)
        assert result.empty

    def test_full_peak_lag_with_three_level_peaks(self, data_with_three_level_peaks):
        """
        Test full_peak_lag function with multiple peaks in ascending order.
        """
        data = data_with_three_level_peaks
        asc_peaks = ['hi1', 'hi2', 'hi3']
        result = utils.full_peak_lag(data, asc_peaks)
        expected = pd.read_csv('.\\test\\expected_3_level_peaks.csv', index_col=0)
        pd.testing.assert_frame_equal(result, expected)


class TestHistoricalSwings:
    """
    A collection of unit tests for the `historical_swings` function in the `utils` module.
    """

    def test_historical_swings_with_no_data(self):
        data = pd.DataFrame(columns=['high', 'low', 'close'])
        result = utils.historical_swings(data)
        columns = data.columns.tolist() + ['hi1', 'lo1', 'hi2', 'lo2', 'hi3', 'lo3']
        assert (result.columns.tolist() == columns)
        assert result.empty

    def test_historical_swings_with_one_level(self):
        data = pd.DataFrame({
            'high': [1.0, 2.0, 3.0, 2.0, 1.0],
            'low': [0.0, 1.0, 2.0, 1.0, 0.0],
            'close': [1.0, 2.0, 3.0, 2.0, 1.0]
        })
        result = utils.historical_swings(data, lvl_limit=1)
        expected = pd.DataFrame({
            'high': [1.0, 2.0, 3.0, 2.0, 1.0],
            'low': [0.0, 1.0, 2.0, 1.0, 0.0],
            'close': [1.0, 2.0, 3.0, 2.0, 1.0],
            'hi1': [None, None, 3.0, None, None],
            'lo1': [None, None, None, None, None]
        }).astype('float64')
        pd.testing.assert_frame_equal(result, expected)

    def test_historical_swings_with_multiple_levels(self):
        data = pd.DataFrame({
            'high': [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            'low': [0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
            'close': [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
        })
        result = utils.historical_swings(data, lvl_limit=3)
        expected = pd.DataFrame({
            'high': [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            'low': [0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
            'close': [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            'hi1': [None, None, 3.0, None, None, None, 3.0, None, None],
            'lo1': [None, None, None, None, 0.0, None, None, None, None],
            'hi2': [None, None, None, None, None, None, None, None, None],
            'lo2': [None, None, None, None, None, None, None, None, None],
            'hi3': [None, None, None, None, None, None, None, None, None],
            'lo3': [None, None, None, None, None, None, None, None, None]
        }).astype('float64')
        pd.testing.assert_frame_equal(result, expected)

    def test_historical_swings_with_last_known_peak_data(self):
        data = pd.DataFrame({
            'high': [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            'low': [0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
            'close': [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
        })
        last_known_peak_data = pd.DataFrame({
            'type': [-1],
            'lvl': [1],
            'start': [2]
        })
        result = utils.historical_swings(data, lvl_limit=3, last_known_peak_data=last_known_peak_data)
        expected = pd.DataFrame({
            'high': [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            'low': [0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
            'close': [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            'hi1': [None, None, None, None, None, None, 3.0, None, None],
            'lo1': [None, None, None, None, 0.0, None, None, None, None],
            'hi2': [None, None, None, None, None, None, None, None, None],
            'lo2': [None, None, None, None, None, None, None, None, None],
            'hi3': [None, None, None, None, None, None, None, None, None],
            'lo3': [None, None, None, None, None, None, None, None, None]
        }).astype('float64')
        pd.testing.assert_frame_equal(result, expected)

    def test_historical_swings_with_custom_column_names(self):
        data = pd.DataFrame({
            'high_price': [1.0, 2.0, 3.0, 2.0, 1.0],
            'low_price': [0.0, 1.0, 2.0, 1.0, 0.0],
            'close_price': [1.0, 2.0, 3.0, 2.0, 1.0]
        })
        result = utils.historical_swings(data, _h='high_price', _l='low_price', _c='close_price', lvl_limit=1)
        expected = pd.DataFrame({
            'high_price': [1.0, 2.0, 3.0, 2.0, 1.0],
            'low_price': [0.0, 1.0, 2.0, 1.0, 0.0],
            'close_price': [1.0, 2.0, 3.0, 2.0, 1.0],
            'hi1': [None, None, 3.0, None, None],
            'lo1': [None, None, None, None, None]
        }).astype('float64')
        pd.testing.assert_frame_equal(result, expected)


class TestInitCompare:
    """
    A collection of unit tests for the `init_compare` function in the `utils` module.
    """

    def test_init_compare_with_no_duplicates(self):
        """
        Test init_compare function with a pandas Series containing no duplicate swings.
        """
        swings = pd.Series([1, 2, 3, 2, 1])
        result = utils.init_compare(swings, skip_duplicate=False).astype('int64')
        expected = pd.DataFrame({
            'prior': [1, 2, 3],
            'sw': [2, 3, 2],
            'next': [3, 2, 1]
        },
        index=[1,2,3]).astype('int64')
        pd.testing.assert_frame_equal(result, expected)

    def test_init_compare_with_duplicates(self):
        """
        Test init_compare function with a pandas Series containing duplicate swings.
        """
        swings = pd.Series([1, 2, 2, 1])
        result = utils.init_compare(swings, skip_duplicate=True).astype('int64')
        expected = pd.DataFrame({
            'prior': [1],
            'sw': [2],
            'next': [1]
        },
        index=[2]).astype('int64')
        pd.testing.assert_frame_equal(result, expected)

    def test_init_compare_with_empty_series(self):
        """
        Test init_compare function with an empty pandas Series.
        """
        swings = pd.Series([])
        result = utils.init_compare(swings, skip_duplicate=False)
        assert result.empty

    def test_init_compare_with_single_swing(self):
        """
        Test init_compare function with a pandas Series containing a single swing.
        """
        swings = pd.Series([1])
        result = utils.init_compare(swings, skip_duplicate=False)
        assert result.empty

    def test_init_compare_with_two_swings(self):
        """
        Test init_compare function with a pandas Series containing two swings.
        """
        swings = pd.Series([1, 2])
        result = utils.init_compare(swings, skip_duplicate=False).astype('int64')
        expected = pd.DataFrame({
            'prior': [],
            'sw': [],
            'next': []
        }).astype('int64')
        pd.testing.assert_frame_equal(result, expected)
        

class TestInitSwings:   
    def test_init_swings(self):
        """
        Test init_swings function with a DataFrame containing one peak.

        The function creates a DataFrame with one peak and calls the init_swings function with the DataFrame and a limit of 3.
        The expected result is a tuple containing a DataFrame with the close price and a DataFrame with the peak data.
        The function uses pd.testing.assert_frame_equal to compare the expected result with the actual result.
        """
        data = pd.DataFrame({
            'close': [1.0, 2.0, 3.0, 2.0, 1.0],
            'hi1': [None, None, 3.0, None, None],
            'lo1': [None, None, None, None, None],
            'hi2': [None, None, None, None, None],
            'lo2': [None, None, None, None, None],
            'hi3': [None, None, None, None, None],
            'lo3': [None, None, None, None, None],
        })
        expected_px = pd.DataFrame({
            'close': [1.0, 2.0, 3.0, 2.0, 1.0]
        })
        expected_peak_table = pd.DataFrame({
            'start': [2],
            'end': [3],
            'type': [-1],
            'lvl': [1],
            'st_px': [3.0],
            'en_px': [2.0]
        })
        expected = (expected_px, expected_peak_table)
        result = utils.init_swings(data, lvl_limit=3)
        pd.testing.assert_frame_equal(result[0], expected_px)
        pd.testing.assert_frame_equal(result[1], expected_peak_table)

    def test_init_swings_last_known_peaks(self):
        """
        Test init_swings function with a DataFrame containing one peak.

        The function creates a DataFrame with one peak and calls the init_swings function with the DataFrame and a limit of 3.
        The expected result is a tuple containing a DataFrame with the close price and a DataFrame with the peak data.
        The function uses pd.testing.assert_frame_equal to compare the expected result with the actual result.
        """
        # load peak.pkl
        data = pd.read_pickle('.\\test\\peak.pkl').reset_index()
        # get last peak for each level and type
        last_peaks = data.groupby(['lvl', 'type']).last()
        # select data rows that have index equal to last peak index
        last_known_peak_data = data.loc[last_peaks.index.tolist()]
        print('l')
