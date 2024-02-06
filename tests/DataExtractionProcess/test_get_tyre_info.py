import pytest
import pandas as pd

from src.DataExtractionProcess.get_tyre_info import get_tyre_info


@pytest.fixture
def dataframe():
    data = {
        'TOD': [0, 1, 2, 0, 1, 2, 0, 1, 2],
        'TyreID': [0, 0, 0, 1, 1, 1, 2, 2, 2]
    }
    return pd.DataFrame.from_dict(data), get_tyre_info(pd.DataFrame.from_dict(data))


class TestDataExtractionProcess:
    def test_get_tyre_info_cumcount(self, dataframe, tmp_path):
        # Assert that tyre info increases by 1
        no_laps, ordered = dataframe
        grouped_by_session = ordered.groupby('TyreID')
        for b, g in grouped_by_session:
            sorted = g.sort_values('TOD')
            values = sorted['TyreUsage'].values.tolist()
            for i in range(1, len(values)):
                assert values[i] == values[i-1] + 1