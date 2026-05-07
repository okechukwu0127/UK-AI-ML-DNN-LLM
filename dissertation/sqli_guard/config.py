"""
Central config for the SQLi detection project.
Keep all paths and defaults in one place for clarity.
"""

from os import path

# paths
PROJECT_DIR = path.dirname(path.abspath(__file__))
DATASET_PATH = path.join(path.dirname(PROJECT_DIR), "rbsqli_dataset.csv")
DATA_DIR = path.join(PROJECT_DIR, "data")
MODEL_DIR = path.join(PROJECT_DIR, "models")
RESULTS_DIR = path.join(PROJECT_DIR, "results")

# dataset columns
TEXT_COLUMNS = [
    "sql_query",
    "sql_command",
    "target_table",
    "selected_columns",
    "comparison_operator",
    "logical_operator",
    "sql_comment_syntax",
    "injection_type",
]
LABEL_COLUMN = "vulnerability_status"
POS_LABEL = "Yes"

# data split
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# baseline model defaults
MAX_FEATURES = 200_000
NGRAM_RANGE = (1, 2)

# transformer defaults
TRANSFORMER_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256

"""
Notes
- DATASET_PATH points to the 2GB CSV in the dissertation root.
- Adjust TRANSFORMER_MODEL if you want a SQL specific transformer.
"""
