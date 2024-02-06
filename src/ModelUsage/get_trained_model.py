import argparse
from typing import Tuple, Union, Dict

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process.kernels import ConstantKernel

from src.ModelUsage.model_helper_functions import generate_one_hot_drivers, generate_one_hot_teams, \
    generate_one_hot_circuits


def normalise(column: pd.Series, return_constants: bool = False) \
        -> Union[pd.Series, Tuple[pd.Series, Dict[str, float]]]:
    """

    Parameters
    ----------
    column: pd.Series
        Column to be normalised.

    return_constants: bool, default=False
        Return the constants used in the normalisation.

    Returns
    -------
    normalised_column: pd.Series
        Normalised version of `column`.

    constants: Tuple[np.float64, np.float64]
        Constants used in the normalisation.
        Only returned if `return_constants` is True.
    """
    normalised_column = (column - column.mean()) / (column.std() if column.std() != 0 else 1)
    if return_constants:
        return normalised_column, {'mean': column.mean(), 'std': column.std()}
    return normalised_column


def get_model(kernel: kernels.Kernel) -> GaussianProcessRegressor:
    """Return a Gaussian Process Regressor with the given kernel"""
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None)
    return gpr


def normalise_dataframe(data: pd.DataFrame,
                        return_constants: bool = False):
    """Normalises all columns in a dataframe that need to be normalised

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame to be normalised - must be made from a CSV produced by DataExtractionProcess.generate_useful_CSV.py.

    return_constants: bool, default=False
        Returns the constants associated with the normalisation

    Returns
    -------
    normalised_version: pd.DataFrame
        DataFrame based on `data` that has normalisation applied to all necessary columns.

    constants: Dict[Str, Dict[Str, np.float64]]
        Dictionary of the normalisation constants used in normalising each column.
        Only returned if `return_constants` is True.
    """
    normalised_data = data.copy()

    tyres = ['Superhard', 'Hard', 'Medium', 'Soft', 'Supersoft', 'Ultrasoft', 'Hypersoft']
    if not return_constants:
        normalised_data['TyreUsage'] = normalise(data['TyreUsage'])
        normalised_data['LapsCompleted'] = normalise(data['LapsCompleted'])
        normalised_data['TyreCompound'] = normalise(data['TyreCompound'].apply(lambda x: tyres.index(x)))
        normalised_data['SessionName'] = normalise(data['SessionName'].apply(lambda x: int(x[1])))
        return normalised_data

    constants = dict()
    normalised_version, normalisation_constants = normalise(data['TyreUsage'], True)
    constants['TyreUsage'] = normalisation_constants
    normalised_data['TyreUsage'] = normalised_version

    normalised_version, normalisation_constants = normalise(data['LapsCompleted'], True)
    constants['LapsCompleted'] = normalisation_constants
    normalised_data['LapsCompleted'] = normalised_version

    normalised_version, normalisation_constants = normalise(data['TyreCompound'].apply(lambda x: tyres.index(x)), True)
    constants['TyreCompound'] = normalisation_constants
    normalised_data['TyreCompound'] = normalised_version

    normalised_version, normalisation_constants = normalise(data['SessionName'].apply(lambda x: int(x[1])), True)
    constants['SessionName'] = normalisation_constants
    normalised_data['SessionName'] = normalised_version

    return normalised_data, constants


def get_numpy_vector(data: pd.DataFrame):
    """Return the vector form of a dataframe that can be inputted into a model

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame to convert into a numpy vector.

    Returns
    -------
    stored_vectors: np.ndarray
        Vector format of the inputted DataFrame.

    """
    drivers = generate_one_hot_drivers(data.DriverName)
    teams = generate_one_hot_teams(data.Team)
    circuits = generate_one_hot_circuits(data.Circuit)

    stored_vectors = np.hstack((drivers, teams, circuits))

    stored_vectors = np.hstack((stored_vectors,
                                data['TyreUsage'].to_numpy().reshape((-1, 1))))

    stored_vectors = np.hstack((stored_vectors,
                                data['LapsCompleted'].to_numpy().reshape((-1, 1))))

    stored_vectors = np.hstack((stored_vectors,
                                data['TyreCompound'].to_numpy().reshape((-1, 1))))

    stored_vectors = np.hstack((stored_vectors,
                                data['SessionName'].to_numpy().reshape((-1, 1))))

    return stored_vectors


def convert_to_training_data(data: pd.DataFrame, return_constants: bool = False) \
        -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Dict[str, float]]]]:
    """Convert a dataframe to the associated input vector and output vector

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame of information to use when generating training data.

    return_constants: bool, default=False
        Return the normalisation constants associated with normalising the DataFrame

    Returns
    -------
    (stored_vector, output): Tuple[np.ndarray, np.ndarray]
        input for a model and the expected output for the given DataFrame

    constants: Dict[str, Dict[str, np.float64]]
        Constants used for normalisation.
        Only returned if `return_constants` is True.

    """
    constants = None

    if return_constants:
        normalised_data, constants = normalise_dataframe(data, True)
    else:
        normalised_data = normalise_dataframe(data)

    stored_vectors = get_numpy_vector(normalised_data)
    output = (data['LapTime'] - data['ExpectedTime']).to_numpy()

    if return_constants:
        return (stored_vectors, output), constants
    return stored_vectors, output


def convert_to_input(data: pd.DataFrame, normalisation_constants: Dict[str, Dict[str, str]]) -> np.ndarray:
    """Convert a given dataframe into a model input

    Parameters
    ----------
    data: pd.DataFrame
        data to be converted into a numpy vector.

    normalisation_constants: Dict[str, Dict[str, str]]
        constants to use when normalising the DataFrame.

    Returns
    -------
    numpy_vector: np.ndarray
        Numpy Vector representation of the DataFrame.
    """
    data_copy = data.copy()
    tyres = ['Superhard', 'Hard', 'Medium', 'Soft', 'Supersoft', 'Ultrasoft', 'Hypersoft']
    data_copy['SessionName'] = data['SessionName'].apply(lambda x: int(x[1]))
    data_copy['TyreCompound'] = data['TyreCompound'].apply(lambda x: tyres.index(x))
    for column in normalisation_constants:
        data_copy[column] = (data_copy[column] - normalisation_constants[column]['mean']) / \
                            (normalisation_constants[column]['std'] if normalisation_constants[column][
                                                                           'std'] != 0 else 1)
    return get_numpy_vector(data_copy)


def get_trained_model(kernel: kernels.Kernel, data_location: str,
                      data_cutoff: int = -1,
                      return_constants: bool = False):
    """Get a trained model

    Parameters
    ----------
    kernel: sklearn.gaussian_process.kernels.Kernel
        Kernel to use when training the model.

    data_location: str
        Location of the CSV file produceed by DataExtractionProcess.generate_useful_CSV.py.

    data_cutoff: int, default=-1
        Where in the CSV to stop getting training data from, -1 => to use the entire CSV.

    return_constants: bool, default=False
        Return constants associated with normalising the data.

    Returns
    -------
    gpr: GaussianProcessRegressor
        Trained gaussian process regressor
    constants: Dict[str, Dict[str, str]]
        Constants used to normalise data before training.
        Only returned if `return_constants` is True.

    """
    gpr = get_model(kernel)
    data = pd.read_csv(f'{data_location}')
    if data_cutoff != -1:
        data = data[:data_cutoff]

    constants = None
    if return_constants:
        (X, y), constants = convert_to_training_data(data.copy(), True)
    else:
        X, y = convert_to_training_data(data.copy())

    gpr.fit(X, y)

    if return_constants:
        return gpr, constants
    return gpr


def main(file_location: str):
    model, constants = get_trained_model(ConstantKernel(), file_location, True)
    print(f'Model: {model}')
    print(constants)
    print(f'Kernel: {model.kernel}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and train a model for a given CSV file.')
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='file to find training data.')
    args = parser.parse_args()
    main(args.file)
