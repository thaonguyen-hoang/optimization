project on Optimization

- problem: binary classification - predict whether a patient has diabetes or not, based on medical data (have been proceesed and saved in /data/processed)

## main components

1. loss functions
- binary cross entropy (BCE)
- weighted binary cross entropy (weighted BCE)
- squared hinge
- focal

2. optimizers
- gradient descent (GD)
- stochastic gradient descent (SGD)
- accelerated gradient descent (nesterov/NAG)
- newton
- l-bfgs

3. regularization
- none
- L1 (lasso): using proximal gradient descent
- L2 (ridge)

4. hyperparam tuning
- learning rate/step size: (1) fixed and (2) backtracking, based on Lipschitz constant to choose the value/range of value for learning rate (sound more convincing to have theory backup)
- specific hyperparamer of each loss: w_{pos}, w_{neg} for weighted BCE, γ and α for focal
- λ for regularization

## note for running experiments:

- model: logistic regression
- each loss: experiment with all optimizers and regularization options and step size options
none/ridge: run with full optimizers, each with fixed step size và backtracking
- lasso: 
    + GD/NAG: each with fixed step size và backtracking
    + SGD: suitable only with fixed step size, but still run backtracking to prove that it does not make sense (and explain why)
    + not running with newton and l-bfgs
- stopping criteria: all running with a fixed num epochs and 
    + within one loss: early stop when objective function value between 2 steps no greater then the predefined tol_obj (for e.g. 10e-6)
    + between different losses: early stop using early stopping patience when testing on validation
- after all, run best combinations of loss + optimizer + regularization + hyperparam on the final test set (hold out from beginning) and mark common metrics (recall, precision, f1, auroc, auprc,...)
- plotting:
    + compare error/loss_value vs time/epoch per loss
    + hessian spectrum to highlight L2 effect

