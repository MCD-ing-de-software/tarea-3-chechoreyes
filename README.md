[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Ab2QrTYQ)
# Unit Testing Examples for Data Science Students

This small project contains two Python classes that are useful in data
science workflows and are well suited for practicing **unit testing**
with `pytest`.

## Project structure

- `src/data_cleaner.py`  
  Defines the `DataCleaner` class with methods for cleaning
  `pandas.DataFrame` objects: dropping rows with missing values,
  trimming whitespace in string columns, and removing outliers using
  the IQR rule.

- `src/statistics_utils.py`  
  Defines the `StatisticsUtils` class with small numerical helpers:
  moving average, z-score computation, and min–max scaling. All methods
  validate their inputs and raise clear exceptions for invalid cases.

- `tests/test_data_cleaner.py`  
  Unit tests for `DataCleaner` using `pytest`.

- `tests/test_statistics_utils.py`  
  Unit tests for `StatisticsUtils` using `pytest`.

## Installation

Create and activate a virtual environment (optional but recommended),
then install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the tests

From the root folder of the project, run:

```bash
pytest
```

This will discover and execute all tests under the `tests/` directory.
