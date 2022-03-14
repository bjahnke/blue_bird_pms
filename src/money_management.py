import pd_accessors as pda
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class TrailStop:
    """
    pos_price_col: price column to base trail stop movement off of
    neg_price_col: price column to check if stop was crossed
    cum_extreme: cummin/cummax, name of function to use to calculate trailing stop direction
    """

    neg_price_col: str
    pos_price_col: str
    cum_extreme: str
    dir: int

    def init_trail_stop(
        self, price: pd.DataFrame, initial_trail_price, entry_date, rg_end_date
    ) -> pd.Series:
        """
        :param rg_end_date:
        :param entry_date:
        :param price:
        :param initial_trail_price:
        :return:
        """
        entry_price = price.close.loc[entry_date]
        trail_pct_from_entry = (entry_price - initial_trail_price) / entry_price
        extremes = price.loc[entry_date:rg_end_date, self.pos_price_col]

        # when short, pct should be negative, pushing modifier above one
        trail_modifier = 1 - trail_pct_from_entry
        # trail stop reaction must be delayed one bar since same bar reaction cannot be determined
        trail_stop: pd.Series = (
            getattr(extremes, self.cum_extreme)() * trail_modifier
        ).shift(1)
        trail_stop.iat[0] = trail_stop.iat[1]

        return trail_stop

    def exit_signal(self, price: pd.DataFrame, trail_stop: pd.Series) -> pd.Series:
        """detect where price has crossed price"""
        return ((trail_stop - price[self.neg_price_col]) * self.dir) >= 0

    def target_exit_signal(self, price: pd.DataFrame, target_price) -> pd.Series:
        """detect where price has crossed price"""
        return ((target_price - price[self.pos_price_col]) * self.dir) <= 0

    def get_stop_price(
        self, price: pd.DataFrame, stop_date, offset_pct: float
    ) -> float:
        """calculate stop price given a date and percent to offset the stop point from the peak"""
        pct_from_peak = 1 - (offset_pct * self.dir)
        return price[self.neg_price_col].at[stop_date] * pct_from_peak

    def cap_trail_stop(self, trail_data: pd.Series, cap_price) -> pd.Series:
        """apply cap to trail stop"""
        trail_data.loc[((trail_data - cap_price) * self.dir) > 0] = cap_price
        return trail_data


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

    # cap share to less than equity x 2 (includes leverage)
    nominal_limit = eqty * 2
    nominal_value = abs(shares * px)
    exceed_limit = shares.loc[nominal_value > nominal_limit]
    if not exceed_limit.empty:
        shares.loc[exceed_limit.index] = (nominal_limit // nominal_value.loc[exceed_limit.index]) * exceed_limit

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


def test_eqty_risk():
    # px = 2000
    # sl = 2222
    px = 2222
    sl = 2000


    eqty = 100000
    risk = -0.005
    fx = 110
    lot = 100

    res = eqty_risk_shares(px, sl, eqty, risk, fx, lot)
    print('done')


if __name__ == '__main__':
    test_eqty_risk()


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
    print('d')


if __name__ == "__main__":
    test_risk_app()
