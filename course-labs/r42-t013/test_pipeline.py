import unittest
from copy import deepcopy
from pathlib import Path

from pipeline import (
    evaluate_predictions,
    load_fixture,
    majority_class,
    predict_baseline,
    run_baseline,
    validate_fixture,
    validate_records,
)


FIXTURE_PATH = Path(__file__).with_name("fixture.json")


class TestMajorityBaseline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture(FIXTURE_PATH)

    def test_majority_class_uses_training_labels(self):
        train_ids = self.fixture["splits"]["train"]
        records = {
            record["record_id"]: record
            for record in self.fixture["records"]
        }
        training_records = [records[record_id] for record_id in train_ids]

        self.assertEqual(majority_class(training_records), 0)

    def test_tie_chooses_smaller_label(self):
        training_records = [
            {"record_id": "a", "group_id": "A", "pixels": [[0] * 4] * 4, "label": 1},
            {"record_id": "b", "group_id": "B", "pixels": [[0] * 4] * 4, "label": 0},
        ]

        self.assertEqual(majority_class(training_records), 0)

    def test_predictions_use_same_majority_label(self):
        records = self.fixture["records"][:2]

        self.assertEqual(predict_baseline(1, records), [1, 1])

    def test_empty_training_records_are_rejected(self):
        with self.assertRaises(ValueError):
            majority_class([])

    def test_invalid_prediction_label_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_baseline(2, self.fixture["records"][:1])

    def test_accuracy_is_calculated_correctly(self):
        records = self.fixture["records"][:2]
        result = evaluate_predictions(records, [0, 1])

        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["accuracy"], 0.5)

    def test_pipeline_uses_training_split_for_majority(self):
        modified_fixture = deepcopy(self.fixture)

        for record in modified_fixture["records"]:
            if record["record_id"] == "img_004":
                record["label"] = 0
            elif record["record_id"] == "img_005":
                record["label"] = 1
            elif record["record_id"] == "img_006":
                record["label"] = 1

        result = run_baseline(modified_fixture)

        self.assertEqual(result["majority_label"], 0)

    def test_fixture_has_no_split_leakage(self):
        validate_fixture(self.fixture)

        group_splits = {}

        for split_name, record_ids in self.fixture["splits"].items():
            for record in self.fixture["records"]:
                if record["record_id"] in record_ids:
                    group_id = record["group_id"]
                    self.assertNotIn(group_id, group_splits)
                    group_splits[group_id] = split_name

    def test_duplicate_record_ids_are_rejected(self):
        records = deepcopy(self.fixture["records"])
        records[1]["record_id"] = records[0]["record_id"]

        with self.assertRaises(ValueError):
            validate_records(records)

    def test_invalid_pixel_values_are_rejected(self):
        records = deepcopy(self.fixture["records"])
        records[0]["pixels"][0][0] = 300

        with self.assertRaises(ValueError):
            validate_records(records)

    def test_invalid_labels_are_rejected(self):
        records = deepcopy(self.fixture["records"])
        records[0]["label"] = 2

        with self.assertRaises(ValueError):
            validate_records(records)

    def test_missing_record_assignment_is_rejected(self):
        fixture = deepcopy(self.fixture)
        fixture["splits"]["test"].remove("img_006")

        with self.assertRaises(ValueError):
            validate_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
