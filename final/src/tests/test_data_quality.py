import json
import os
import pytest

DATA_PATH = "data/test_set.json"

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Test data file not found")
def test_data_quality():
    with open(DATA_PATH, "r") as f:
        dataset = json.load(f)

    assert isinstance(dataset, list), "Dataset must be a list"
    assert len(dataset) > 0, "Dataset must not be empty"

    for sample in dataset:
        assert isinstance(sample, dict), "Each sample must be a dict"
        assert "text" in sample, "Missing 'text' field"
        assert isinstance(sample["text"], str), "'text' must be a string"
        assert sample["text"].strip() != "", "Text must not be empty"