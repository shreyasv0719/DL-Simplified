# R42-T013 Evidence

## Environment
- Python: 3.14.6
- OS: Darwin 27.2.0 arm64
- Dependencies: Standard library only
- Execution: Offline, CPU-only

## Input
- Six synthetic 4x4 grayscale images.
- Three images labelled 0 and three labelled 1.
- Not the real EMNIST dataset.

## Expected and Actual Results
- Training labels: 0, 0, 1
- Majority class: 0
- Validation: 0/1 correct (0%)
- Test: 1/2 correct (50%)
- Unit tests: 12 passed, 0 failed.

## Commands Run
- `python3 course-labs/r42-t013/pipeline.py`
- `python3 -m unittest discover -s course-labs/r42-t013 -p "test_*.py" -v`

## Limitations
Results apply only to the synthetic fixture, not real-world EMNIST performance.
