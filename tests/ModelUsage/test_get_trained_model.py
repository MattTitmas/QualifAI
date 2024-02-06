import pytest
import pandas as pd
import numpy as np
import sklearn.gaussian_process
from sklearn.gaussian_process.kernels import RBF

from src.ModelUsage.get_trained_model import normalise, get_model, normalise_dataframe


@pytest.fixture
def normalisation():
    data = pd.DataFrame.from_dict({
        'ToNormalise': [1, 2, 3, 4, 5, 6, 7],
        'TyreCompound': ['Soft'] * 7,
        'TyreUsage': [1, 2, 3, 4, 5, 6, 7],
        'LapsCompleted': [1, 2, 3, 4, 5, 6, 7],
        'SessionName': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']
    })
    mean = np.mean([1, 2, 3, 4, 5, 6, 7])
    std = np.std([1, 2, 3, 4, 5, 6, 7], ddof=1)
    return data, mean, std


class TestDataExtractionProcess:
    def test_get_trained_model_normalise_columns(self, normalisation):
        data, mean, std = normalisation
        normalised = normalise(data['ToNormalise'])

        assert np.array_equal(normalised.to_numpy(), (np.array([1, 2, 3, 4, 5, 6, 7]) - mean) / std)

    def test_get_trained_model_normalise_returns_correct_constants(self, normalisation):
        data, mean, std = normalisation
        _, constants = normalise(data['ToNormalise'], return_constants=True)
        assert constants['mean'] == mean
        assert constants['std'] == std

    def test_get_trained_model_get_model_returns_GPR(self):
        model = get_model(RBF())
        assert isinstance(model, sklearn.gaussian_process.GaussianProcessRegressor)

    def test_get_trained_model_get_model_has_kernel(self):
        kernel = RBF(length_scale=3.14)
        model = get_model(kernel)
        assert model.kernel == kernel

    def test_normalise_dateframe_fix_tyrecompound(self, normalisation):
        data, mean, std = normalisation
        normalised = normalise_dataframe(data)
        assert len(normalised['TyreCompound'].unique()) == 1

    def test_normalise_dateframe_fix_sessionName(self, normalisation):
        data, mean, std = normalisation
        normalised = normalise_dataframe(data)
        assert np.array_equal(normalised['SessionName'].to_numpy(), (np.array([1, 2, 3, 4, 5, 6, 7]) - mean) / std)

    def test_normalise_dateframe_fix_tyreUsage(self, normalisation):
        data, mean, std = normalisation
        normalised = normalise_dataframe(data)
        assert np.array_equal(normalised['TyreUsage'].to_numpy(), (np.array([1, 2, 3, 4, 5, 6, 7]) - mean) / std)

    def test_normalise_dateframe_fix_lapsCompleted(self, normalisation):
        data, mean, std = normalisation
        normalised = normalise_dataframe(data)
        assert np.array_equal(normalised['LapsCompleted'].to_numpy(), (np.array([1, 2, 3, 4, 5, 6, 7]) - mean) / std)

    def test_normalise_dateframe_returns_constants(self, normalisation):
        data, mean, std = normalisation
        normalised, constants = normalise_dataframe(data, return_constants=True)
        for k, v in constants.items():
            if k == 'TyreCompound':
                assert v['mean'] == 3.0
                assert v['std'] == 0.0
            else:
                assert v['mean'] == mean
                assert v['std'] == std
