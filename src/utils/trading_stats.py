import numpy as np
import pandas as pd


def simple_log_returns(prices: pd.Series) -> pd.Series:
    """calculates log returns of a price series"""
    return np.log(prices / prices.shift(1))


def rolling_grit(cumul_returns, window):
    rolling_peak = cumul_returns.rolling(window).max()
    draw_down_squared = (cumul_returns - rolling_peak) ** 2
    ulcer = draw_down_squared.rolling(window).sum() ** 0.5
    grit = cumul_returns / ulcer
    return grit.replace([-np.inf, np.inf], np.NAN)


def expanding_grit(cumul_returns):
    tt_peak = cumul_returns.expanding().max()
    draw_down_squared = (cumul_returns - tt_peak) ** 2
    ulcer = draw_down_squared.expanding().sum() ** 0.5
    grit = cumul_returns / ulcer
    return grit.replace([-np.inf, np.inf], np.NAN)


def rolling_profits(returns, window):
    profit_roll = returns.copy()
    profit_roll[profit_roll < 0] = 0
    profit_roll_sum = profit_roll.rolling(window).sum().fillna(method="ffill")
    return profit_roll_sum


def rolling_losses(returns, window):
    loss_roll = returns.copy()
    loss_roll[loss_roll > 0] = 0
    loss_roll_sum = loss_roll.rolling(window).sum().fillna(method="ffill")
    return loss_roll_sum


def expanding_profits(returns):
    profit_roll = returns.copy()
    profit_roll[profit_roll < 0] = 0
    profit_roll_sum = profit_roll.expanding().sum().fillna(method="ffill")
    return profit_roll_sum


def expanding_losses(returns):
    loss_roll = returns.copy()
    loss_roll[loss_roll > 0] = 0
    loss_roll_sum = loss_roll.expanding().sum().fillna(method="ffill")
    return loss_roll_sum


def profit_ratio(profits, losses):
    pr = profits.fillna(method="ffill") / abs(losses.fillna(method="ffill"))
    return pr


def rolling_profit_ratio(returns, window):
    return profit_ratio(
        profits=rolling_profits(returns, window), losses=rolling_losses(returns, window)
    )


def expanding_profit_ratio(returns):
    return profit_ratio(
        profits=expanding_profits(returns), losses=expanding_losses(returns)
    )


def rolling_tail_ratio(cumul_returns, window, percentile, limit):
    left_tail = np.abs(cumul_returns.rolling(window).quantile(percentile))
    right_tail = cumul_returns.rolling(window).quantile(1 - percentile)
    np.seterr(all="ignore")
    tail = np.maximum(np.minimum(right_tail / left_tail, limit), -limit)
    return tail


def expanding_tail_ratio(cumul_returns, percentile, limit):
    left_tail = np.abs(cumul_returns.expanding().quantile(percentile))
    right_tail = cumul_returns.expanding().quantile(1 - percentile)
    np.seterr(all="ignore")
    tail = np.maximum(np.minimum(right_tail / left_tail, limit), -limit)
    return tail


def common_sense_ratio(pr, tr):
    return pr * tr


def expectancy(win_rate, avg_win, avg_loss):
    # win% * avg_win% - loss% * abs(avg_loss%)
    return win_rate * avg_win + (1 - win_rate) * avg_loss


def t_stat(signal_count, trading_edge):
    sqn = (signal_count**0.5) * trading_edge / trading_edge.std(ddof=0)
    return sqn


def t_stat_expanding(signal_count, edge):
    """"""
    sqn = (signal_count**0.5) * edge / edge.expanding().std(ddof=0)
    return sqn


def robustness_score(grit, csr, sqn):
    """
    clamp constituents of robustness score to >=0 to avoid positive scores when 2 values are negative
    exclude zeros when finding start date for rebase to avoid infinite score (divide by zero)
    """
    # TODO should it start at 1?
    _grit = grit.copy()
    _csr = csr.copy()
    _sqn = sqn.copy()
    # the score will be zero if on metric is negative
    _grit.loc[_grit < 0] = 0
    _csr.loc[_csr < 0] = 0
    _sqn.loc[_sqn < 0] = 0
    exclude_zeros = (_grit != 0) & (_csr != 0) & (_sqn != 0)
    try:
        start_date = max(
            _grit[pd.notnull(_grit) & exclude_zeros].index[0],
            _csr[pd.notnull(_csr) & exclude_zeros].index[0],
            _sqn[pd.notnull(_sqn) & exclude_zeros].index[0],
        )
    except IndexError:
        score = pd.Series(data=np.NaN, index=grit.index)
    else:
        score = (
            _grit * _csr * _sqn / (_grit[start_date] * _csr[start_date] * _sqn[start_date])
        )
    return score


def cumulative_returns_pct(returns, min_periods):
    return returns.expanding(min_periods=min_periods).sum().apply(np.exp) - 1


def pyramid(position, root=2):
    return 1 / (1 + position) ** (1 / root)


def kelly(win_rate, avg_win, avg_loss):
    """kelly position sizer, returns risk budget as percent"""
    return win_rate / np.abs(avg_loss) - (1 - win_rate) / avg_win


def kelly_fractional():
    pass
