def mmd2_u_stat_variance(K, inds=(0, 1)):
    """
    Estimate MMD variance with estimator from https://arxiv.org/abs/1906.02104.

    K should be a LazyKernel; we'll compare the parts in inds,
    default (0, 1) to use K.XX, K.XY, K.YY.
    """
    i, j = inds

    m = K.n(i)
    assert K.n(j) == m

    XX = K.matrix(i, i)
    XY = K.matrix(i, j)
    YY = K.matrix(j, j)

    mm = m * m
    mmm = mm * m
    m1 = m - 1
    m1_m1 = m1 * m1
    m1_m1_m1 = m1_m1 * m1
    m2 = m - 2
    mdown2 = m * m1
    mdown3 = mdown2 * m2
    mdown4 = mdown3 * (m - 3)
    twom3 = 2 * m - 3

    return (
        (4 / mdown4) * (XX.offdiag_sums_sq_sum() + YY.offdiag_sums_sq_sum())
        + (4 * (mm - m - 1) / (mmm * m1_m1))
        * (XY.row_sums_sq_sum() + XY.col_sums_sq_sum())
        - (8 / (mm * (mm - 3 * m + 2)))
        * (XX.offdiag_sums() @ XY.col_sums() + YY.offdiag_sums() @ XY.row_sums())
        + 8 / (mm * mdown3) * ((XX.offdiag_sum() + YY.offdiag_sum()) * XY.sum())
        - (2 * twom3 / (mdown2 * mdown4)) * (XX.offdiag_sum() + YY.offdiag_sum())
        - (4 * twom3 / (mmm * m1_m1_m1)) * XY.sum() ** 2
        - (2 / (m * (mmm - 6 * mm + 11 * m - 6)))
        * (XX.offdiag_sq_sum() + YY.offdiag_sq_sum())
        + (4 * m2 / (mm * m1_m1_m1)) * XY.sq_sum()
    )
