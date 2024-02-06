import numpy as np


def generate_driver_one_hot(driver_name: str) -> np.array:
    drivers = ['Nicholas Latifi', 'Antonio Giovinazzi', 'Kimi Raikkonen', 'George Russell', 'Lance Stroll',
               'Daniil Kvyat', 'Kevin Magnussen', 'Romain Grosjean', 'Lando Norris', 'Max Verstappen', 'Pierre Gasly',
               'Alexander Albon', 'Carlos Sainz Jr.', 'Sergio Perez', 'Sebastian Vettel', 'Charles Leclerc',
               'Valtteri Bottas',  'Lewis Hamilton', 'Esteban Ocon', 'Daniel Ricciardo']
    driver_location = drivers.index(driver_name) if driver_name in drivers else -1
    to_return = np.zeros(21)
    to_return[driver_location] = 1
    return to_return


def generate_teams_one_hot(team_name: str) -> np.array:
    teams = ['Williams', 'Alfa Romeo', 'Racing Point', 'AlphaTauri', 'Haas', 'McLaren', 'Red Bull', 'Ferrari',
             'Mercedes AMG', 'Renault']
    team_location = teams.index(team_name) if team_name in teams else -1
    to_return = np.zeros(11)
    to_return[team_location] = 1
    return to_return


def generate_circuits_one_hot(circuit_name: str) -> np.array:
    circuits = ['A1-Ring']
    team_location = circuits.index(circuit_name) if circuit_name in circuits else -1
    to_return = np.zeros(2)
    to_return[team_location] = 1
    return to_return

