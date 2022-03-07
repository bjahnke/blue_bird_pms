import pd_accessors as pda

def pyramid(position, root=2):
    return 1 / (1 + position) ** (1 / root)


def assign_pyramid_weight(df, regime_col, entry_count_col, regime_val=None):
    weights = []
    for regime_slice in pda.regime_slices(df, regime_col, regime_val):
        weights.append(pyramid(regime_slice[entry_count_col]))