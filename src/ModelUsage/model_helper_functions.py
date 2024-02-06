from typing import List, Callable, TypeVar
from functools import partial

import numpy as np


T = TypeVar('T')
def to_categorical(key: Callable, to_encode: List[T]) -> np.ndarray:
    """Return a one-hot encoding of a list"""
    return np.array([key(i) for i in to_encode], dtype=int)


def generate_one_hot_helper(item_list: List[T], item: T) -> np.ndarray:
    """
    Give the index of an item in a given list, assumes the item is in the list
    :param item_list: list of items
    :param item: item we want the index of
    :return: Index of {item} in {item_list}
    """
    info = np.zeros(len(item_list))
    info[item_list.index(item)] = 1
    return info


folder = f'Data\\model_data\\'

generate_one_hot_driver = partial(generate_one_hot_helper, open(f'{folder}drivers.txt').read().split('\n'))
generate_one_hot_team = partial(generate_one_hot_helper, open(f'{folder}teams.txt').read().split('\n'))
generate_one_hot_circuit = partial(generate_one_hot_helper, open(f'{folder}circuits.txt').read().split('\n'))


def generate_one_hot_drivers(drivers: List[str]):
    """one-hot encoding function for a list of drivers"""
    return to_categorical(generate_one_hot_driver, drivers)


def generate_one_hot_teams(teams: List[str]):
    """one-hot encoding function for a list of teams"""
    return to_categorical(generate_one_hot_team, teams)


def generate_one_hot_circuits(circuits: List[str]):
    """one-hot encoding function for a list of circuits"""
    return to_categorical(generate_one_hot_circuit, circuits)
