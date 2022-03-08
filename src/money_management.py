import pd_accessors as pda
import pandas as pd
import numpy as np


def pyramid(position, root=2):
    return 1 / (1 + position) ** (1 / root)


def kelly(win_rate, avg_win, avg_loss):
    """kelly position sizer, returns risk budget as percent"""
    return win_rate / np.abs(avg_loss) - (1 - win_rate) / avg_win


def kelly_fractional():
    pass


def eqty_risk_shares(px, sl, eqty, risk, lot=None, fx=None):
    r = sl - px
    budget = eqty * risk

    if fx is not None and fx > 0:
        budget *= fx

    if lot is None:
        shares = budget // r
    else:
        shares = round(budget // (r * lot) * lot, 0)

    return shares


def concave(ddr, floor):
    """
    For demo purpose only
    """
    if floor == 0:
        concave_res = ddr
    else:
        concave_res = ddr**floor
    return concave_res


def convex(ddr, floor):
    """
    # obtuse
    obtuse = 1 - acute
    """
    if floor == 0:
        convex_res = ddr
    else:
        convex_res = ddr ** (1 / floor)
    return convex_res


def risk_appetite(eqty, tolerance, mn, mx, span, shape) -> pd.Series:
    """
    position sizer

    eqty: equity curve series
    tolerance: tolerance for drawdown (<0)
    mn: min risk
    mx: max risk
    span: exponential moving average to smooth the risk_appetite
    shape: convex (>45 deg diagonal) = 1, concave (<diagonal) = -1, else: simple risk_appetite
    """
    # drawdown rebased
    eqty = pd.Series(eqty)
    watermark = eqty.expanding().max()
    # all-time-high peak equity
    drawdown = eqty / watermark - 1
    # drawdown from peak
    ddr = 1 - np.minimum(drawdown / tolerance, 1)
    # drawdown rebased to tolerance from 0 to 1
    avg_ddr = ddr.ewm(span=span).mean()
    # span rebased drawdown

    # Shape of the curve
    if shape == 1:  #
        _power = mx / mn  # convex
    elif shape == -1:
        _power = mn / mx  # concave
    else:
        _power = 1  # raw, straight line
    ddr_power = avg_ddr**_power  # ddr

    # mn + adjusted delta
    risk = mn + (mx - mn) * ddr_power

    return risk


def test_risk_app():
    equity_curve = 25000
    tolerance = -0.1
    min_risk = -0.0025
    max_risk = -0.0075
    span = 5
    shape = 1

    convex_risk = risk_appetite(
        equity_curve, tolerance, min_risk, max_risk, span, shape
    )
    # convex_risk = -convex_risk * peak_equity


if __name__ == "__main__":
    print("d")
