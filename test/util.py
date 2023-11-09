import os

import pandas as pd
from sqlalchemy import create_engine, text
import dotenv
dotenv.load_dotenv()

def get_stock_data(_symbol, _interval, _neon_db_url):
    q = (
         "select {table}.*, stock.symbol, stock.is_relative "
         "from {table} "
         "left join stock on {table}.stock_id = stock.id "
         "where stock.symbol = '{symbol}' "
         "and stock.interval = '{interval}' "
         "{extra}"
    ).format
    
    _stock_data = pd.read_sql(q(table='stock_data', symbol=_symbol, interval=_interval, extra="order by stock_data.bar_number asc"), con=_neon_db_url)
    _regime_data = pd.read_sql(q(table='regime', symbol=_symbol, interval=_interval, extra="order by regime.start asc"), con=_neon_db_url)
    _peak_data = pd.read_sql(q(table='peak', symbol=_symbol, interval=_interval, extra="order by peak.start asc"), con=_neon_db_url)
    return _stock_data, _regime_data, _peak_data

def save_test_data():
    """
    create test data by freezing current SPY data as symbol TEST_DATA
    """
    engine = create_engine(os.environ.get('NEON_DB_CONSTR'), echo=True)
    stock_data, regime_data, peak_data = get_stock_data('SPY', '1d', engine)
    stock_data.to_pickle('test/stock_data.pkl')
    regime_data.to_pickle('test/regime.pkl')
    peak_data.to_pickle('test/peak.pkl')

if __name__ == '__main__':
    save_test_data()