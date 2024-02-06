from typing import Tuple

import numpy as np
from tqdm.autonotebook import tqdm

from sklearn.gaussian_process.kernels import Kernel


def mean_squared_error(model, inp, expected):
    predictions = model.predict(inp)
    return ((expected - predictions) ** 2).mean(axis=None)


def random_search(model_type, kernel: Kernel, bounds: Tuple[float, float],
                  X_train, y_train, X, y, size=25, loops=10,
                  return_loss: bool = False, pbar: bool = False):
    """Perform a random search to find optimal kernel hyperparameters

    Parameters
    ----------
    model_type: GaussianProcessRegressor
        Model to use when training.

    kernel: sklearn.gaussian_process.kernels.Kernel
        Kernel to optimise for use in the model.

    bounds: Tuple[float, float]
        Bounds of the hyperparameters of the kernel.

    X_train: np.ndarray
        Training input to the model.

    y_train: np.ndarray
        Training output to the model.

    X: np.ndarray
        Inputs to test loss for.

    y: np.ndarray
        Outputs to test loss for.

    size: int
        Number of samples per distribution per loop.

    loops: int
        Number of loops to run the random search for.

    return_loss: bool
        Return the loss aassociated with the kernel.

    pbar: bool
        Show a progress bar.

    Returns
    -------
    kernel: sklearn.gaussian_process.kernels.Kernel
        Kernel with optimised hyperparameters

    loss: np.float64
        Loss associated with the returned kernel.
        Only returned in `return_loss` is True.
    """
    param_limits = dict()
    for param, value in kernel.get_params().items():
        if type(value) == type(1.0):
            param_limits[param] = (1, 0.5)
        if type(value) == type((1, 1)):
            kernel.set_params(**{param: bounds})

    min_loss = float('inf')
    loss_prev_epoch = float('inf')
    size = size
    epoch_bar = tqdm(range(loops), leave=False) if pbar else range(loops)
    for j in epoch_bar:
        if pbar:
            epoch_bar.set_description(f'Current loss: {min_loss}')
        distributions = None
        params = []

        for param, value in kernel.get_params().items():
            if type(value) == type(1.0) or type(value) == np.float64:
                mean, std = param_limits[param]
                distribution = np.random.normal(loc=mean, scale=std, size=size)
                distribution = np.clip(distribution, bounds[0], bounds[1])
                params.append(param)
                if distributions is None:
                    distributions = distribution
                else:
                    distributions = np.vstack((distributions, distribution))

        XY = np.meshgrid(*tuple(distributions))

        mat = np.array(XY).transpose()

        coords = mat.shape[-1]

        mat = mat.reshape(-1, coords)

        for i in mat:
            parameters = {param: val for param, val in zip(params, i)}
            kernel.set_params(**parameters)
            test_model = model_type(kernel=kernel, optimizer=None)
            test_model.fit(X_train, y_train)
            error = mean_squared_error(test_model, X, y)
            if error < min_loss:
                min_loss = error
                stored_params = parameters
        for param, value in stored_params.items():
            if loss_prev_epoch == min_loss:
                param_limits[param] = (value, param_limits[param][1] * 1.5)
            else:
                param_limits[param] = (value, param_limits[param][1] / 1.5)
        kernel.set_params(**stored_params)
        loss_prev_epoch = min_loss
    to_return = kernel,
    if return_loss:
        to_return += min_loss,
    return to_return
