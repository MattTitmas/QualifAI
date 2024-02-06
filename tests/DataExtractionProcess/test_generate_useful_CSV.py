import pytest
import pandas as pd

from src.DataExtractionProcess.generate_useful_CSV import filter_csv


@pytest.fixture
def dataframe():
    unfiltered = pd.read_csv('Data\\Updated_CSV_data\\2020.csv')
    filtered = filter_csv('Data\\Updated_CSV_data\\2020.csv', None)
    return filtered, unfiltered


class TestDataExtractionProcess:
    def test_filter_CSV_remove_wet_tyres(self, dataframe):
        # Assert that all running on wet tyres is removed
        filtered_csv, unfiltered_csv = dataframe
        print(filtered_csv['TyreCompound'].unique())
        tyre_intersection = {'X-Wet', 'Intermediate'}.intersection(set(filtered_csv['TyreCompound'].unique()))
        assert tyre_intersection == set()

    def test_filter_CSV_remove_all_corresponding_dates(self, dataframe):
        # Assert that all dates that have wet running are removed
        filtered_csv, unfiltered_csv = dataframe
        dates_with_wet = unfiltered_csv.groupby('Date').filter(lambda g: ((g.TyreCompound == 'X-Wet').any() or
                                                                     (g.TyreCompound == 'Intermediate').any()))
        # dates_with_wet['Date'].unique() == ['07_05', '07_12', '07_19', '11_15']
        date_intersection = set(dates_with_wet['Date'].unique()).intersection(set(filtered_csv['Date'].unique()))
        assert date_intersection == set()


    def test_filter_CSV_contain_only_qualifying_data(self, dataframe):
        # Assert that all non-qualifying sessions are removed
        filtered_csv, unfiltered_csv = dataframe
        assert sorted(filtered_csv['SessionName'].unique()) == ['Q1', 'Q2', 'Q3']

    def test_filter_CSV_contain_follows_107(self, dataframe):
        # Assert that all times > 1.07* the fastest are removed
        filtered_csv, unfiltered_csv = dataframe
        fastest_lap = filtered_csv.groupby('Date')['LapTime'].min()
        slowest_lap = filtered_csv.groupby('Date')['LapTime'].max()
        times_107 = fastest_lap.apply(lambda x: x * 1.07)
        for key in fastest_lap.index.values.tolist():
            assert slowest_lap[key] < times_107[key]

    def test_filter_CSV_contain_follows_plus_two(self, dataframe):
        # Assert that all times > 2 + the fastest are removed
        filtered_csv, unfiltered_csv = dataframe
        fastest_lap = filtered_csv.groupby(['Date', 'DriverName'])['LapTime'].min()
        slowest_lap = filtered_csv.groupby(['Date', 'DriverName'])['LapTime'].max()
        plus_two = fastest_lap.apply(lambda  x: x + 2)
        for key in fastest_lap.index.values.tolist():
            assert slowest_lap[key] < plus_two[key]

    def test_filter_CSV_doesnt_contain_unnecessary_columns(self, dataframe):
        # Assert that columns not used in generating numpy vector are removed
        filtered_csv, unfiltered_csv = dataframe
        columns = (filtered_csv.columns.tolist())
        assert 'HasLapTime' not in columns
        assert 'IsInLap' not in columns
        assert 'IsOutLap' not in columns
