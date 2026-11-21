import json
from collections import Counter
from pathlib import Path


SPLIT_NAMES = ("train", "validation", "test")


def validate_records(records):
    """Validate the synthetic image records."""
    if not isinstance(records, list) or not records:
        raise ValueError("Records must be a non-empty list.")

    seen_ids = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be a dictionary.")

        required = {"record_id", "group_id", "pixels", "label"}
        missing = required - record.keys()

        if missing:
            raise ValueError(
                f"Record {index} is missing fields: {sorted(missing)}"
            )

        record_id = record["record_id"]
        group_id = record["group_id"]

        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"Record {index} has an invalid record_id.")

        if record_id in seen_ids:
            raise ValueError(f"Duplicate record_id: {record_id}")

        seen_ids.add(record_id)

        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"{record_id} has an invalid group_id.")

        label = record["label"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"{record_id} has an invalid label.")

        pixels = record["pixels"]
        if not isinstance(pixels, list) or len(pixels) != 4:
            raise ValueError(f"{record_id} must have 4 pixel rows.")

        for row in pixels:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError(f"{record_id} must have 4 pixels per row.")

            for pixel in row:
                if type(pixel) is not int or not 0 <= pixel <= 255:
                    raise ValueError(f"{record_id} has an invalid pixel.")

    return records


def validate_fixture(fixture):
    """Validate records and fixed split assignments."""
    if not isinstance(fixture, dict):
        raise ValueError("Fixture must be a dictionary.")

    if "records" not in fixture or "splits" not in fixture:
        raise ValueError("Fixture must contain records and splits.")

    records = validate_records(fixture["records"])
    splits = fixture["splits"]

    if not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES):
        raise ValueError(
            "Splits must contain train, validation, and test."
        )

    record_by_id = {
        record["record_id"]: record for record in records
    }

    assigned_ids = []

    for split_name in SPLIT_NAMES:
        record_ids = splits[split_name]

        if not isinstance(record_ids, list) or not record_ids:
            raise ValueError(f"{split_name} must contain record IDs.")

        for record_id in record_ids:
            if record_id not in record_by_id:
                raise ValueError(
                    f"Unknown record ID in {split_name}: {record_id}"
                )

        assigned_ids.extend(record_ids)

    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("A record appears in more than one split.")

    if set(assigned_ids) != set(record_by_id):
        raise ValueError("Every record must appear in exactly one split.")

    group_splits = {}

    for split_name in SPLIT_NAMES:
        for record_id in splits[split_name]:
            group_id = record_by_id[record_id]["group_id"]

            if group_id in group_splits and group_splits[group_id] != split_name:
                raise ValueError(
                    f"Group {group_id} appears in multiple splits."
                )

            group_splits[group_id] = split_name

    return fixture


def majority_class(training_records):
    """Learn the most frequent label from training records only."""
    validate_records(training_records)

    counts = Counter(record["label"] for record in training_records)

    # In a tie, choose the smaller label for deterministic behavior.
    return min(counts, key=lambda label: (-counts[label], label))


def predict_baseline(majority_label, records):
    """Predict the learned majority label for every supplied record."""
    if type(majority_label) is not int or majority_label not in (0, 1):
        raise ValueError("majority_label must be 0 or 1.")

    validate_records(records)
    return [majority_label] * len(records)


def evaluate_predictions(records, predictions):
    """Calculate accuracy without changing the predictions."""
    validate_records(records)

    if len(records) != len(predictions) or not predictions:
        raise ValueError("Records and predictions must have equal non-zero length.")

    correct = sum(
        record["label"] == prediction
        for record, prediction in zip(records, predictions)
    )

    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records),
    }


def run_baseline(fixture):
    """Train on the training split and evaluate held-out splits."""
    validate_fixture(fixture)

    record_by_id = {
        record["record_id"]: record for record in fixture["records"]
    }

    train_records = [
        record_by_id[record_id]
        for record_id in fixture["splits"]["train"]
    ]

    majority_label = majority_class(train_records)

    results = {
        "majority_label": majority_label,
        "training_label_counts": dict(
            sorted(Counter(
                record["label"] for record in train_records
            ).items())
        ),
        "evaluations": {},
    }

    for split_name in ("validation", "test"):
        split_records = [
            record_by_id[record_id]
            for record_id in fixture["splits"][split_name]
        ]

        predictions = predict_baseline(majority_label, split_records)

        results["evaluations"][split_name] = {
            "record_ids": [
                record["record_id"] for record in split_records
            ],
            "predictions": predictions,
            **evaluate_predictions(split_records, predictions),
        }

    return results


def load_fixture(path):
    """Load and validate a JSON fixture."""
    with Path(path).open("r", encoding="utf-8") as file:
        fixture = json.load(file)

    return validate_fixture(fixture)


if __name__ == "__main__":
    fixture_path = Path(__file__).with_name("fixture.json")
    fixture = load_fixture(fixture_path)
    results = run_baseline(fixture)
    print(json.dumps(results, indent=2))
