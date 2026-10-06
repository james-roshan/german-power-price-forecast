"""Model factories: LightGBM, linear and neural competitors, and the bounded search space."""
from __future__ import annotations

from collections.abc import Callable

from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import ParameterSampler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_PARAMS, SEED

SEARCH_SPACE = {
    "n_estimators": [150, 250, 400], "learning_rate": [0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63], "min_child_samples": [40, 100, 200],
    "reg_lambda": [1.0, 5.0, 15.0], "objective": ["regression_l1", "regression"],
    "window_days": [365, 730],
}


def make_lgbm() -> LGBMRegressor:
    """The original fixed-parameter LightGBM."""
    return LGBMRegressor(**MODEL_PARAMS)


def lgb_factory(params: dict) -> Callable[[], LGBMRegressor]:
    return lambda: LGBMRegressor(**params)


def _preprocess():
    # Imputation/scaling are learned inside each training window only.
    return [SimpleImputer(strategy="median", add_indicator=True), StandardScaler()]


def ridge_factory():
    return make_pipeline(*_preprocess(), Ridge(alpha=10.0))


def lasso_factory():
    """Pooled Lasso autoregression (LEAR-inspired; not the canonical 24 separate hourly models)."""
    return make_pipeline(*_preprocess(), Lasso(alpha=0.1, max_iter=5000, selection="cyclic"))


def mlp_factory():
    return TransformedTargetRegressor(
        regressor=make_pipeline(*_preprocess(), MLPRegressor(
            hidden_layer_sizes=(32, 16), alpha=0.01, max_iter=120, early_stopping=False,
            shuffle=False, random_state=SEED)),
        transformer=StandardScaler())


def candidate_specs(n_iter: int = 7, seed: int = SEED) -> dict[str, dict]:
    """Original parameters (trial 00) plus `n_iter` sampled configurations."""
    candidates = [dict(MODEL_PARAMS, window_days=730)]
    candidates += [dict(MODEL_PARAMS, **p) for p in ParameterSampler(SEARCH_SPACE, n_iter=n_iter, random_state=seed)]
    specs = {}
    for i, candidate in enumerate(candidates):
        params = candidate.copy()
        window = params.pop("window_days")
        specs[f"LightGBM trial {i:02d}"] = {"params": params, "window_days": window, "retrain_days": 7}
    return specs
