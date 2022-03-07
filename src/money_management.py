import pd_accessors as pda
import pandas as pd
import numpy as np


def concave(ddr, floor):
    """
    For demo purpose only
    """
    if floor == 0:
        concave = ddr
    else:
        concave = ddr ** (floor)
    return concave


def convex(ddr, floor):
    """
    # obtuse
    obtuse = 1 - acute
    """
    if floor == 0:
        convex = ddr
    else:
        convex = ddr ** (1 / floor)
    return convex


def risk_appetite(eqty, tolerance, mn, mx, span, shape):
    """
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
    ddr_power = avg_ddr ** _power  # ddr

    # mn + adjusted delta
    risk = mn + (mx - mn) * ddr_power

    return risk

