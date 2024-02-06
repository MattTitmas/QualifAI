import random
import argparse
from typing import List

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from src.ModelUsage import convert_to_training_data, convert_to_input, get_trained_model
from src.ModelUsage import random_search


def remain_in_garage(csv_location: str, qualifying_session: str, event_date: str,
                     wanted_drivers: List[str], tyres: str = 'Soft', train_to: int = -1,
                     return_std: bool = False, return_con: bool = False, return_loss: bool = False,
                     pbar: bool = False):
    """Output the probability of a list of drivers passing into the next session

    Parameters
    ----------
    csv_location: str
        Location of the CSV file containing necessary data.

    qualifying_session: str
        The current qualifying session.

    event_date: str
        The date of the event we are checking.

    wanted_drivers: List[str]
        List of drivers we want the probability of.

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
    driver_info: Dict[Tuple[np.flaot64, np.float64, Tuple[np.float64, np.float64]]]
        Dictionary of all information requested to be returned.

    loss: np.float64:
        Loss of the kernel produced via random search.
        Only returned if `return_loss` is True.

    """

    data = pd.read_csv(csv_location)

    if train_to != -1:
        data = data[0:train_to]

    starting_lap = data.iloc[-1]['LapsCompleted'] + 1

    lap_times_3_mins_to_go = data[(data['Date'] == event_date) & (data['SessionName'] == qualifying_session)]
    lap_times_3_mins_to_go = lap_times_3_mins_to_go.groupby('DriverName', group_keys=False) \
        .apply(lambda x: x[x['LapTime'] == x['LapTime'].min()]) \
        .sort_values('LapTime').reset_index(drop=True)
    lap_times_3_mins_to_go['LapsCompleted'] = starting_lap
    lap_times_3_mins_to_go['TyreUsage'] = 1
    lap_times_3_mins_to_go['TyreCompound'] = tyres

    cutoff = len(lap_times_3_mins_to_go) - 5

    training_data = data.copy()
    indices = data.index[data['Date'] == event_date].tolist()

    X_kernel = random.sample(indices, len(indices) // 2)

    training_data, X_kernel = training_data.drop(index=X_kernel), training_data.loc[X_kernel]

    X_train_kernel, y_train_kernel = convert_to_training_data(X_kernel)
    X_train, y_train = convert_to_training_data(training_data)

    kernel = RBF() + RBF() + ConstantKernel()
    new_kernel, loss = random_search(GaussianProcessRegressor, kernel, (1e-10, 1e10),
                                     X_train, y_train, X_train_kernel, y_train_kernel,
                                     size=3, loops=10, return_loss=True, pbar=pbar)

    model, normalisation_constants = get_trained_model(new_kernel, csv_location,
                                                       data_cutoff=train_to,
                                                       return_constants=True)

    return_dict = dict()
    for wanted_driver in wanted_drivers:
        if wanted_driver not in lap_times_3_mins_to_go['DriverName'].unique():
            continue
        idx = lap_times_3_mins_to_go.index[lap_times_3_mins_to_go['DriverName'] == wanted_driver][0]
        to_knockout = cutoff - idx
        driver_lap_time = lap_times_3_mins_to_go.iloc[idx]['LapTime']
        if idx >= cutoff:
            return_dict[wanted_driver] = 0.0, 0.0, driver_lap_time
            continue

        input_to_model = convert_to_input(lap_times_3_mins_to_go[idx + 1:], normalisation_constants)
        means, stds = model.predict(input_to_model, return_std=True)
        means += lap_times_3_mins_to_go[idx + 1:]['ExpectedTime'].to_numpy()

        values = None
        samples = 100000
        for mean, std in zip(means, stds):
            sample = np.random.normal(mean, std, samples)
            if values is None:
                values = sample
            else:
                values = np.vstack((values, sample))
        faster = values < driver_lap_time
        sum_of_cols = np.sum(faster, axis=0)
        mean = 1 - np.sum(sum_of_cols >= to_knockout) / samples
        std = np.std(sum_of_cols < to_knockout)
        add_to_dict = mean
        if return_std:
            if isinstance(add_to_dict, np.float64):
                add_to_dict = (add_to_dict, std)
            else:
                add_to_dict += (mean, std)
        if return_con:
            confidence_interval = (mean - 1.96 * std, mean + 1.96 * std)
            if isinstance(add_to_dict, np.float64):
                add_to_dict = (add_to_dict, confidence_interval)
            else:
                add_to_dict += (mean, confidence_interval)

        return_dict[wanted_driver] = add_to_dict

    if return_loss:
        return return_dict, loss
    return return_dict


def main(csv_location: str, qualifying_session: str, event_date: str,
         wanted_drivers: List[str], tyres: str = 'Soft', train_to: int = -1,
         return_std: bool = False, return_con: bool = False, return_loss: bool = False,
         pbar: bool = False):

    print(remain_in_garage(csv_location, qualifying_session, event_date, wanted_drivers,
                           tyres, train_to,
                           return_std, return_con, return_loss, pbar))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--csv', required=True, type=str,
                        help='Location of the CSV file.')
    parser.add_argument('-s', '--session', required=True, type=str,
                        help='Current session.')
    parser.add_argument('-d', '--date', required=True, type=str,
                        help='Date of the current event (as CSV shows).')
    parser.add_argument('--drivers', required=True, nargs='+',
                        help='Drivers to return results for.')
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
    main(args.csv, args.session, args.date, args.drivers, args.tyres,
         args.train_to, args.return_std, args.return_conf, args.return_loss,
         args.progress_bar)
