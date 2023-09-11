import os

import pandas as pd
from sqlalchemy import create_engine, text
import dotenv
dotenv.load_dotenv()


def save_test_data():
    """
    create test data by freezing current SPY data as symbol TEST_DATA
    """
    engine = create_engine(os.environ.get('NEON_DB_CONSTR'), echo=True)
    base_query = 'SELECT * FROM {table} where {table}.symbol = \'{symbol}\' ORDER BY {table}.{bar} ASC'

    def add_test_data(table_name: str, symbol: str, bar_order: str):
        df = pd.read_sql(base_query.format(table=table_name, symbol=symbol, bar=bar_order), engine)
        df['symbol'] = 'TEST_DATA'
        df.to_pickle(f'{table_name}.pkl')

    # freeze new test data from SPY
    for table, order_col in [('stock_data', 'bar_number'), ('peak', 'start'), ('regime', 'start')]:
        add_test_data(table, 'SPY', order_col)

if __name__ == '__main__':
    save_test_data()