import numpy as np
import pytest

from src.losses import BCELoss, SquaredHingeLoss, WeightedBCELoss, build_loss
from src.metrics import auprc, auroc
from src.optimizers import build_optimizer
from src.regularizers import L1Reg, build_regularizer
from src.step_sizes import FixedLR, lipschitz_constant
from train import train_logreg


@pytest.mark.parametrize("loss", [BCELoss(), WeightedBCELoss(2, .5), SquaredHingeLoss()])
def test_loss_gradient_matches_finite_difference(loss):
    z = np.array([-.7, .2, 1.3]); y = np.array([0., 1., 1.]); eps = 1e-6
    numerical = np.array([(loss.value(z + eps * np.eye(3)[i], y) - loss.value(z - eps * np.eye(3)[i], y)) / (2 * eps) for i in range(3)])
    np.testing.assert_allclose(numerical, loss.grad(z, y) / len(y), rtol=1e-5, atol=1e-7)


def test_l1_prox_uses_threshold_directly():
    reg = L1Reg(.7)
    np.testing.assert_allclose(reg.prox(np.array([-2., .2, 3.]), .5), [-1.5, 0., 2.5])


def test_tie_aware_ranking_metrics():
    y = np.array([0, 1, 0, 1]); scores = np.ones(4)
    assert auroc(y, scores) == pytest.approx(.5)
    assert auprc(y, scores) == pytest.approx(.5)


def test_lipschitz_bounds():
    X = np.eye(3)
    assert lipschitz_constant(X, loss_name="squared_hinge") == pytest.approx(2 / 3)
    assert lipschitz_constant(X, reg_type="l2", lam=.2) == pytest.approx(1 / 12 + .2)


@pytest.mark.parametrize("optimizer,reg,step", [
    ("gd", "none", "fixed"), ("gd", "l1", "backtracking"),
    ("nag", "none", "fixed"), ("nag", "l1", "backtracking"),
    ("newton", "l2", "backtracking"), ("sgd", "l1", "diminishing")])
def test_training_smoke(optimizer, reg, step):
    rng = np.random.RandomState(3); X = rng.randn(80, 4); y = (X[:, 0] - .4 * X[:, 1] > 0).astype(float)
    regularizer = build_regularizer(reg, 1e-2)
    L = lipschitz_constant(X, reg=regularizer, lam=regularizer.lam)
    lr = FixedLR(.5, L)
    opt = build_optimizer(optimizer, lr=lr, backtracking=step == "backtracking",
                          schedule="diminishing" if step == "diminishing" else "fixed")
    result = train_logreg(X[:60], y[:60], X[60:], y[60:], build_loss("bce"), opt, regularizer,
                          n_epochs=12, batch_size=8, patience_inner=0, patience_outer=0)
    assert result["epochs_run"] == 12
    assert np.isfinite(result["history"]["train_loss"]).all()


def test_invalid_optimizer_combinations():
    with pytest.raises(ValueError): build_optimizer("sgd", backtracking=True)
