from sympy import symbols, Eq, solve, Gt

class TwoLegTradeEquation:
    """
    Represents a two-leg trade equation, which is used to calculate the value of M in terms of T, S, E, and D.
    The equation is M = (T - S) / (E - S), subject to the requirements that (T - S) * D > 0 and (E - S) * D > 0.
    Multiple is the distance from the entry price to the target price as a multiple of the distance from the entry price to the stop price.
    Target is the target price
    Stop is the stop price
    Entry is the entry price
    Direction is the direction of the trade.
    """
    F, P, C, S = symbols('F P C S')
    equation = Eq((C - S), F * (P - S))
    solve_for_F = solve(equation, F)[0].subs
    solve_for_P = solve(equation, P)[0].subs
    solve_for_C = solve(equation, C)[0].subs
    solve_for_S = solve(equation, S)[0].subs
    
    class Solve:
        @classmethod
        def fraction(cls, stop, cost, price):
            values = {
                TwoLegTradeEquation.S: stop,
                TwoLegTradeEquation.C: cost,
                TwoLegTradeEquation.P: price,
            }
            return TwoLegTradeEquation.solve_for_F(values)
            
        
        @classmethod
        def price(cls, stop, cost, fraction):
            values = {
                TwoLegTradeEquation.S: stop,
                TwoLegTradeEquation.C: cost,
                TwoLegTradeEquation.F: fraction,
            }
            return TwoLegTradeEquation.solve_for_P(values)
        
        @classmethod
        def stop(cls, cost, price, fraction):
            values = {
                TwoLegTradeEquation.P: price,
                TwoLegTradeEquation.C: cost,
                TwoLegTradeEquation.F: fraction,
            }
            return TwoLegTradeEquation.solve_for_S(values)
        
        @classmethod
        def cost(cls, stop, price, fraction):
            values = {
                TwoLegTradeEquation.S: stop,
                TwoLegTradeEquation.P: price,
                TwoLegTradeEquation.F: fraction,
            }
            return TwoLegTradeEquation.solve_for_C(values)
        

class PositionSize:
    S, C, R, Q = symbols('S C R Q')
    equation = Eq(Q, R /(C - S))

    solve_for_Q = solve(equation, Q)[0].subs
    solve_for_R = solve(equation, R)[0].subs
    solve_for_C = solve(equation, C)[0].subs
    solve_for_S = solve(equation, S)[0].subs

    class Solve:
        @classmethod
        def risk(cls, stop, cost, quantity):
            values = {
                PositionSize.S: stop,
                PositionSize.C: cost,
                PositionSize.Q: quantity,
            }
            return PositionSize.solve_for_R(values)

        @classmethod
        def cost(cls, stop, risk, quantity):
            values = {
                PositionSize.S: stop,
                PositionSize.R: risk,
                PositionSize.Q: quantity,
            }
            return PositionSize.solve_for_C(values)

        @classmethod
        def stop(cls, cost, risk, quantity):
            values = {
                PositionSize.C: cost,
                PositionSize.R: risk,
                PositionSize.Q: quantity,
            }
            return PositionSize.solve_for_S(values)

        @classmethod
        def quantity(cls, stop, cost, risk):
            values = {
                PositionSize.S: stop,
                PositionSize.C: cost,
                PositionSize.R: risk,
            }
            return PositionSize.solve_for_Q(values)
    