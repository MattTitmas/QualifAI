from functools import partial

import numpy as np
import pytest

from src.ModelUsage.model_helper_functions import generate_one_hot_helper, to_categorical


class TestDataExtractionProcess:
    def test_get_model_helper_functions_generate_one_hot_helper(self):
        result = generate_one_hot_helper([1, 2, 3, 4], 2)
        assert np.array_equal(result, np.array([0, 1, 0, 0]))

    def test_get_model_helper_to_categorical(self):
        temp = partial(generate_one_hot_helper, [1, 2, 3, 4])
        result = to_categorical(temp, [1, 2, 3, 4])
        assert np.array_equal(result, np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]]))

    def test_get_model_helper_to_categorical_raises_error_if_not_in_list(self):
        temp = partial(generate_one_hot_helper, [1, 2, 3, 4])
        with pytest.raises(ValueError):
            to_categorical(temp, [5])