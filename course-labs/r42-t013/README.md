# R42-T013: Majority-Class Baseline

## Objective

Implement a simple majority-class baseline for the EMNIST Letters – CNN task.

This is a CPU smoke test, not an actual CNN training implementation.

## Dataset

The fixture contains six synthetic 4x4 grayscale images:
- Three black images with label 0.
- Three white-diagonal images with label 1.

This is simulated data, not the real EMNIST Letters dataset.

## Fixed Splits

| Split | Records | Labels |
|---|---|---|
| Train | img_001, img_002, img_003 | 0, 0, 1 |
| Validation | img_004 | 1 |
| Test | img_005, img_006 | 0, 1 |

Every record belongs to exactly one split, and groups do not overlap.

## Method

1. Validate the fixture.
2. Count labels using training records only.
3. Select the majority class.
4. Predict that class for validation and test records.
5. Calculate accuracy.

Ties are resolved by choosing the smaller label.

## Results

The training majority class is 0.

| Split | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Validation | 0 | 1 | 0% |
| Test | 1 | 2 | 50% |

These results apply only to the synthetic fixture.

## Run Pipeline

```bash
python3 course-labs/r42-t013/pipeline.py
