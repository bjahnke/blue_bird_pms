import src.position_calculation.utils as utils
import pytest


class TestTwoLegTradeEquation:
    """
    A collection of unit tests for the `TwoLegTradeEquation` class in the `utils` module.
    """

    def test_fraction(self):
        """
        Test the `_solve` method of the `TwoLegTradeEquation` class with the `Multiple` symbol.
        """
        result = utils.TwoLegTradeEquation.Solve.fraction(stop=3, cost=2, price=1)
        assert result == 1/2
        result = utils.TwoLegTradeEquation.Solve.fraction(stop=1, cost=2, price=3)
        assert result == 1/2

    def test_price(self):
        """
        Test the `_solve` method of the `TwoLegTradeEquation` class with the `Target` symbol.
        """
        result = utils.TwoLegTradeEquation.Solve.price(stop=5, cost=7, fraction=1/2)
        assert result == 9
        result = utils.TwoLegTradeEquation.Solve.price(stop=7, cost=5, fraction=1/2)
        assert result == 3


    def test_stop(self):
        """
        Test the `_solve` method of the `TwoLegTradeEquation` class with the `Stop` symbol.
        """
        result = utils.TwoLegTradeEquation.Solve.stop(cost=7, price=10, fraction=1/2)
        assert result == 4
        result = utils.TwoLegTradeEquation.Solve.stop(cost=10, price=7, fraction=1/2)
        assert result == 13

    def test_cost(self):
        """
        Test the `_solve` method of the `TwoLegTradeEquation` class with the `Entry` symbol.
        """
        result = utils.TwoLegTradeEquation.Solve.cost(stop=5, price=10, fraction=1/2)
        assert result == 7.5
        result = utils.TwoLegTradeEquation.Solve.cost(stop=10, price=5, fraction=1/2)
        assert result == 7.5


class TestPositionSize:
    @pytest.fixture
    def defaults(self):
        default_stop = 10
        default_cost = 20
        default_quantity = 100
        default_risk = 1000
        return default_stop, default_cost, default_quantity, default_risk

    def test_risk(self, defaults):
        stop, cost, quantity, risk = defaults
        result = utils.PositionSize.Solve.risk(stop, cost, quantity)
        assert result == risk

    def test_cost(self, defaults):
        stop, cost, quantity, risk = defaults
        assert utils.PositionSize.Solve.cost(stop, risk, quantity) == cost

    def test_stop(self, defaults):
        stop, cost, quantity, risk = defaults
        assert utils.PositionSize.Solve.stop(cost, risk, quantity) == stop

    def test_quantity(self, defaults):
        stop, cost, quantity, risk = defaults
        assert utils.PositionSize.Solve.quantity(stop, cost, risk) == quantity