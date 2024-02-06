import pytest
import pandas as pd

from src.DataExtractionProcess.order_laps_set import order_laps_set


@pytest.fixture
def dataframe():
    data = {
        'SessionName': ['1', '1', '1', '2', '2', '2', '3', '3', '3'],
        'TOD': [0, 1, 2, 0, 1, 2, 0, 1, 2]
    }
    return pd.DataFrame.from_dict(data), order_laps_set(pd.DataFrame.from_dict(data))


class TestDataExtractionProcess:
    def test_order_laps_set_cumcount(self, dataframe, tmp_path):
        # Assert that lap order increases by 1
        no_laps, ordered = dataframe
        grouped_by_session = ordered.groupby('SessionName')
        for b, g in grouped_by_session:
            sorted = g.sort_values('TOD')
            values = sorted['LapsCompleted'].values.tolist()
            for i in range(1, len(values)):
                assert values[i] == values[i-1] + 1

    def test_order_laps_set_reset_per_session(self, dataframe, tmp_path):
        # Assert that lap order resets to 0 per session
        no_laps, ordered = dataframe
        grouped_by_session = ordered.groupby('SessionName')
        for b, g in grouped_by_session:
            sorted = g.sort_values('TOD')
            values = sorted['LapsCompleted'].values.tolist()
            assert 0 in values
