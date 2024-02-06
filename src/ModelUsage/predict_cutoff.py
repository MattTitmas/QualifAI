import random
import argparse
from typing import Dict

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import pandas as pd
import numpy as np

from src.ModelUsage import get_trained_model, convert_to_input, convert_to_training_data
from src.ModelUsage import random_search


def get_info(times_sorted, drivers_through,
             return_std: bool, return_con: bool):
    """Return the cutoff time and any extra associated information"""
    mean = np.mean(times_sorted[drivers_through - 1])
    std = np.std(times_sorted[drivers_through - 1])
    confidence_interval = mean - 1.96 * std, mean + 1.96 * std

    to_return = mean
    if return_std:
        if isinstance(to_return, np.float64):
            to_return = to_return, std
        else:
            to_return += std,
    if return_con:
        if isinstance(to_return, np.float64):
            to_return = to_return, confidence_interval
        else:
            to_return += confidence_interval,
    return to_return


def same_lap(model: GaussianProcessRegressor,
             drivers_to_check: pd.DataFrame,
             starting_lap: int,
             normalisation_constants: Dict[str, Dict[str, str]],
             return_std: bool = False, return_con: bool = False):
    check = drivers_to_check.copy()
    drivers_through = len(check) - 5
    check['LapsCompleted'] = starting_lap

    means, stds = model.predict(convert_to_input(check, normalisation_constants), return_std=True)

    means = means + check['ExpectedTime'].to_numpy().squeeze()
    values = None
    samples = 10000
    for mean, std in zip(means, stds):
        if values is None:
            values = np.random.normal(mean, std, samples)
        else:
            values = np.vstack((values, np.random.normal(mean, std, samples)))

    times_sorted = np.sort(np.reshape(values, (len(check), -1)), axis=0)

    return get_info(times_sorted, drivers_through, return_std, return_con)


def model_direct(model: GaussianProcessRegressor,
                 drivers_to_check: pd.DataFrame,
                 starting_lap: int,
                 normalisation_constants: Dict[str, Dict[str, str]],
                 return_std: bool = False, return_con: bool = False):
    """Sample the model directly to calculate predicted driver times"""
    check = drivers_to_check.copy()
    drivers_through = len(check) - 5

    repeat_factor = 1000
    laps = np.arange(drivers_through + 5, dtype=int)
    total_laps = np.tile(laps, repeat_factor).reshape(repeat_factor, -1)

    rng = np.random.default_rng()
    permuted_laps = rng.permuted(total_laps, axis=1) + starting_lap

    duplicated_data = pd.concat([check] * repeat_factor)
    duplicated_data['LapsCompleted'] = permuted_laps.reshape(-1)

    predictions = model.predict(convert_to_input(duplicated_data, normalisation_constants))
    expected_times = np.tile(check['ExpectedTime'].to_numpy().squeeze(), repeat_factor)
    predicted_time = expected_times + predictions

    times_sorted = np.sort(np.reshape(predicted_time, (drivers_through + 5, -1)), axis=0)

    return get_info(times_sorted, drivers_through, return_std, return_con)


def sample_model(model: GaussianProcessRegressor,
                 drivers_to_check: pd.DataFrame,
                 starting_lap: int,
                 normalisation_constants: Dict[str, Dict[str, str]],
                 return_std: bool = False, return_con: bool = False):
    check = drivers_to_check.copy()
    drivers_through = len(check) - 5
    laps = np.arange(drivers_through + 5, dtype=int)

    check = drivers_to_check.copy()
    check['LapsCompleted'] = starting_lap
    check = pd.concat([check] * (drivers_through + 5)).reset_index()
    check['LapsCompleted'] += laps.repeat(drivers_through + 5)

    means, stds = model.predict(convert_to_input(check, normalisation_constants), return_std=True)

    driver_probs = dict()
    for count, driver in enumerate(check['DriverName'].unique()):
        indices = check[check['DriverName'] == driver].index.tolist()
        predictions_for_driver = check['ExpectedTime'].to_numpy().squeeze()[count] + means[indices]
        driver_probs[driver] = {lap: (predictions_for_driver[lap], stds[indices][lap]) for lap in
                                range(len(predictions_for_driver))}

    cutoff_times = np.array([])
    samples = 100
    orders = 1000
    for order in range(orders):
        random.shuffle(laps)
        values = None
        for count, driver in enumerate(check['DriverName'].unique()):
            mean, std = driver_probs[driver][laps[count]]
            if values is None:
                values = np.random.normal(mean, std, samples)
            else:
                values = np.vstack((values, np.random.normal(mean, std, samples)))
        times_sorted = np.sort(values, axis=0)
        cutoff_times = np.concatenate([cutoff_times, times_sorted[drivers_through - 1]])

    mean = np.mean(cutoff_times)
    std = np.std(cutoff_times)

    confidence_interval = mean - 1.96 * std, mean + 1.96 * std
    to_return = mean,
    if return_std:
        to_return += std,
    if return_con:
        to_return += confidence_interval,
    return to_return


def call_model(csv_location: str, qualifying_session: str, event_date, tyres: str = 'Soft',
               train_to: int = -1, return_std: bool = False, return_con: bool = False, return_loss: bool = False,
               pbar: bool = True):
    """Output the predicted cutoff time for a given session and date

    Parameters
    ----------
    csv_location: str
        Location of the CSV file containing necessary data.

    qualifying_session: str
        The current qualifying session.

    event_date: str
        The date of the event we are checking.

    tyres: str, default='Soft'
        What tyre every driver will set another lap on.

    train_to: int, default=-1
        How far into the CSV to train to. -1 => Use all data

    return_std: bool, default=False
        Return the standard deviations associated with each probability

    return_con: bool, default=False
        Return the confidence associated with each probability .

    return_loss: bool, default=False
        Return the loss associated with the kernel.

    pbar: bool, default=False
        Use a progress bar when finding the kernel.

    Returns
    -------
    mean: np.float64
        Mean cutoff time prediction

    std: np.float64
        Standard deviation for the cutoff time prediction
        Only returned in `return_std` is True

    confidence_interval: Tuple[np.float64]
        Confidence interval for the cutoff time prediction
        Only returned in `return_con` is True

    loss: np.float64:
        Loss of the kernel produced via random search.
        Only returned if `return_loss` is True

    """

    data = pd.read_csv(csv_location)

    if train_to != -1:
        data = data[0:train_to]

    starting_lap = data.iloc[-1]['LapsCompleted'] + 1

    dataframe = data.copy()
    indices = dataframe.index[dataframe['Date'] == event_date].tolist()

    X_kernel = random.sample(indices, len(indices) // 2)

    dataframe, X_kernel = dataframe.drop(index=X_kernel), dataframe.loc[X_kernel]

    X_train_kernel, y_train_kernel = convert_to_training_data(X_kernel)
    X_train, y_train = convert_to_training_data(dataframe)

    kernel = RBF() + RBF() + ConstantKernel()
    new_kernel, loss = random_search(GaussianProcessRegressor, kernel, (1e-10, 1e10),
                                     X_train, y_train, X_train_kernel, y_train_kernel,
                                     size=3, loops=10, return_loss=True, pbar=pbar)

    model, normalisation_constants = get_trained_model(new_kernel, csv_location, data_cutoff=train_to,
                                                       return_constants=True)

    data = pd.read_csv(csv_location)
    data = data[(data['Date'] == event_date) & (data['SessionName'] == qualifying_session)]

    grouped_by_driver = data.groupby('DriverName')
    driver_fastest_laps = grouped_by_driver['LapTime'].min().sort_values()

    drivers_to_check = pd.DataFrame()
    expected_time = pd.DataFrame()

    for driver in driver_fastest_laps.index.values:
        drivers_to_check = pd.concat((drivers_to_check, data[data['DriverName'] == driver].iloc[:1]))
        expected_time = pd.concat((expected_time, data[data['DriverName'] == driver].iloc[:1]['ExpectedTime']))

    drivers_to_check = drivers_to_check.reset_index(drop=True)
    drivers_to_check['TyreCompound'] = tyres
    drivers_to_check['TyreUsage'] = 1

    info = model_direct(model,
                        drivers_to_check,
                        starting_lap,
                        normalisation_constants,
                        return_std=return_std,
                        return_con=return_con)
    if return_loss:
        return info, loss
    return info


def main(csv_location: str, qualifying_session: str, event_date, tyres: str = 'Soft',
         train_to: int = -1, return_std: bool = False, return_con: bool = False, return_loss: bool = False,
         pbar: bool = True):
    print(call_model(csv_location, qualifying_session, event_date, tyres,
                     train_to, return_std=return_std, return_con=return_con, return_loss=return_loss,
                     pbar=pbar))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--csv', required=True, type=str,
                        help='Location of the CSV file.')
    parser.add_argument('-s', '--session', required=True, type=str,
                        help='Current session.')
    parser.add_argument('-d', '--date', required=True, type=str,
                        help='Date of the current event (as CSV shows).')
    parser.add_argument('-t', '--tyres', required=False, type=str, default='Soft',
                        help='Most likely tyre choice for all drivers (default="Soft").')
    parser.add_argument('-tt', '--train_to', required=False, type=int, default=-1,
                        help='How many rows of the CSV to use as training data.')
    parser.add_argument('-rs', '--return_std', action='store_true',
                        help='Return a corresponding standard deviation.')
    parser.add_argument('-rc', '--return_conf', action='store_true',
                        help='Return a corresponding confidence interval.')
    parser.add_argument('-rl', '--return_loss', action='store_true',
                        help='Return the loss of the kernel.')
    parser.add_argument('-pb', '--progress_bar', action='store_true',
                        help='Show a progress bar.')
    args = parser.parse_args()

    main(args.csv, args.session, args.date, args.tyres,
         args.train_to, args.return_std, args.return_conf, args.return_loss,
         args.progress_bar)
