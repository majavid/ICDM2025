import numpy as np


def test_imports():
    import icdm2025  # noqa: F401
    import icdm2025.core.bnfit as bnfit  # noqa: F401
    import icdm2025.core.fit_on_source as fos  # noqa: F401
    import icdm2025.methods.ecme as ecme  # noqa: F401
    import icdm2025.methods.first_order_em as foem  # noqa: F401
    import icdm2025.methods.kiiveri_em as kiv  # noqa: F401
    import icdm2025.methods.px_em as pxem  # noqa: F401


def test_conditional_math_is_consistent():
    # Sigma = [[2,1],[1,2]] → coef for T|O where O={0}, T=1 is 0.5
    Sigma = np.array([[2.0, 1.0], [1.0, 2.0]])
    obs_idx = [0]
    t = 1
    Sigma_oo = Sigma[np.ix_(obs_idx, obs_idx)]
    Sigma_to = Sigma[np.ix_([t], obs_idx)]
    w = np.linalg.solve(Sigma_oo, Sigma_to.T).T  # 1×1
    assert np.allclose(w, [[0.5]], atol=1e-8)
