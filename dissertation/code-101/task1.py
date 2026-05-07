#============================================
# SQL INJECTION DETECTION SYSTEM
# Complete Solution with Machine Learning
# Author: MSc Artificial Intelligence Student
# Dataset: RbSQLi Dataset (Rule-Based SQL Injection)
# ============================================

# Suppress warnings for cleaner output
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================
# SECTION 1: IMPORTS (Following Task1_DNN_250460.py style)
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support
)


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# Deep Learning imports (following Task1_DNN_250460.py style)
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, GRU, Embedding
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# For API deployment
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Set random seeds for reproducibility (following Task1_DNN_250460.py)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 80)
print("SQL INJECTION DETECTION SYSTEM")
print("Using Machine Learning & Deep Learning Models")
print("=" * 80)


# ============================================
# SECTION 2: DATA LOADING AND EXPLORATION
# Following style from Task1_DNN_250460.py Question 1
# ============================================

print("\n" + "=" * 80)
print("SECTION 1: DATA LOADING AND EXPLORATION")
print("=" * 80)



def build_metric_summary(y_true, y_pred, y_score=None):
    """
    Build a consistent set of evaluation metrics for every model.

    This keeps the reporting format aligned across the classical machine
    learning models and the deep learning models, and adds weighted-average
    metrics that are useful for dissertation reporting.
    """
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, target_names=['Benign', 'Malicious'])
    }

    metrics['auc'] = roc_auc_score(y_true, y_score) if y_score is not None else None
    return metrics


def save_best_model_artifacts(model_to_save, model_type, best_model_result,
                              tfidf_vectorizer, num_scaler, feature_columns,
                              seq_tokenizer=None, max_sequence_length=None):
    """
    Save the best-performing model and all preprocessing objects required by
    the Flask middleware.

    The middleware must use the same TF-IDF vectorizer, numeric scaler,
    feature ordering, and sequence tokenizer that were used during training.
    Without this bundle the API would not be able to reproduce the training
    feature space correctly.
    """
    print("\nSaving best model artifacts for API middleware deployment...")

    artifact_bundle = {
        'model': model_to_save if model_type == 'ml' else None,
        'model_type': model_type,
        'model_name': best_model_result['model_name'],
        'threshold': 0.5,
        'tfidf_vectorizer': tfidf_vectorizer,
        'num_scaler': num_scaler,
        'feature_columns': feature_columns,
        'seq_tokenizer': seq_tokenizer if model_type in ['lstm', 'gru'] else None,
        'max_sequence_length': max_sequence_length,
        'metrics': {
            'accuracy': best_model_result['accuracy'],
            'precision': best_model_result['precision'],
            'recall': best_model_result['recall'],
            'f1_score': best_model_result['f1_score'],
            'weighted_precision': best_model_result.get('weighted_precision'),
            'weighted_recall': best_model_result.get('weighted_recall'),
            'weighted_f1': best_model_result.get('weighted_f1'),
            'auc': best_model_result.get('auc')
        }
    }

    if model_type in ['dnn', 'lstm', 'gru']:
        keras_model_path = f'best_sql_injection_{model_type}.keras'
        model_to_save.save(keras_model_path)
        artifact_bundle['keras_model_path'] = keras_model_path
        print(f"✓ Saved Keras model to: {keras_model_path}")

    bundle_path = 'best_sql_injection_model.pkl'
    joblib.dump(artifact_bundle, bundle_path)
    print(f"✓ Saved middleware bundle to: {bundle_path}")
    return bundle_path





# ============================================
# IMPROVED DATA LOADING FUNCTION
# Using real CSV data instead of hardcoded values
# ============================================

def load_real_rbsqli_dataset(filepath='rbsqli_dataset.csv', sample_size=None):
    """
    Load and process the real RbSQLi dataset from the CSV file.

    The RbSQLi dataset structure (from image.png):
    - sql_query: The actual SQL query string
    - injection_type: Type of injection attack (None_Type for benign)
    - vulnerability_status: 'Yes' for malicious, 'No' for benign
    - sql_command: SQL command type (SELECT, UPDATE, etc.)
    - target_table: Target database table
    - selected_columns: Columns being selected
    - comparison_operator: Operator used in WHERE clause
    - logical_operator: AND/OR/NOT operators
    - sql_comment_syntax: Comment syntax used (--, #, etc.)

    Reference: Data loading pattern from Task1_DNN_250460.py lines 44-62
    and 7CS033 PDF Task 2.1 (adult dataset loading)

    Args:
        filepath (str): Path to the RbSQLi dataset CSV file
        sample_size (int, optional): Number of rows to sample for development.
                                     If None, loads all data.

    Returns:
        pandas.DataFrame: Loaded and validated dataset
    """

    print("\n" + "=" * 60)
    print("LOADING REAL RbSQLi DATASET")
    print("=" * 60)

    try:
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Load CSV with proper data types for efficiency
        # Using dtype specification to speed up loading of large files
        dtypes = {
            'sql_query': 'string',
            'injection_type': 'string',
            'vulnerability_status': 'string',
            'sql_command': 'string',
            'target_table': 'string',
            'selected_columns': 'string',
            'comparison_operator': 'string',
            'logical_operator': 'string',
            'sql_comment_syntax': 'string'
        }

        # Load the dataset
        print(f"\nLoading dataset from: {filepath}")

        if sample_size:
            # Load only a sample for development/testing
            print(f"Loading sample of {sample_size:,} rows for development...")
            df = pd.read_csv(filepath, dtype=dtypes, nrows=sample_size, low_memory=False)
        else:
            # Load full dataset (caution: 10M+ rows)
            print("Loading full dataset (this may take several minutes)...")
            # Use chunks for large files to manage memory
            chunks = []
            chunk_size = 100000
            for chunk in pd.read_csv(filepath, dtype=dtypes, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                print(f"  Loaded {len(chunks) * chunk_size:,} rows...", end='\r')
            df = pd.concat(chunks, ignore_index=True)

        print(f"\n✓ Data loaded successfully!")
        print(f"  Total rows: {len(df):,}")
        print(f"  Total columns: {len(df.columns)}")

        # Display column information
        print("\nDataset Columns:")
        for col in df.columns:
            print(f"  - {col}: {df[col].dtype}")

        # Check for required columns
        required_columns = ['sql_query', 'vulnerability_status']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"\n⚠ Warning: Missing expected columns: {missing_columns}")
            print("  Available columns:", df.columns.tolist())

            # Try to identify alternative column names
            for col in df.columns:
                if 'query' in col.lower() or 'sql' in col.lower():
                    print(f"  Possible SQL query column: '{col}'")
                if 'vulnerable' in col.lower() or 'status' in col.lower() or 'label' in col.lower():
                    print(f"  Possible label column: '{col}'")

        # Display class distribution
        if 'vulnerability_status' in df.columns:
            print("\nClass Distribution (vulnerability_status):")
            class_counts = df['vulnerability_status'].value_counts()
            for label, count in class_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {label}: {count:,} ({percentage:.2f}%)")
        elif 'injection_type' in df.columns:
            print("\nInjection Type Distribution:")
            type_counts = df['injection_type'].value_counts()
            for inj_type, count in type_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {inj_type}: {count:,} ({percentage:.2f}%)")

            # Create vulnerability_status from injection_type
            print("\nCreating vulnerability_status from injection_type...")
            df['vulnerability_status'] = df['injection_type'].apply(
                lambda x: 'Yes' if x != 'None_Type' else 'No'
            )
            print("  ✓ Created vulnerability_status column")

        # Check for missing values
        print("\nMissing Values per Column:")
        missing_counts = df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count:,} ({count/len(df)*100:.2f}%)")

        # Display sample rows
        print("\nSample rows (first 5):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.head())

        return df

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure the RbSQLi dataset CSV file is in the correct location.")
        print("Expected path:", os.path.abspath(filepath))
        print("\nYou can download the dataset from: https://data.mendeley.com/datasets/xz4d5zj5yw/4")

        # Ask user if they want to create sample data for testing
        response = input("\nWould you like to create sample data for testing? (y/n): ")
        if response.lower() == 'y':
            return create_sample_data_for_testing()
        else:
            raise

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


def create_sample_data_for_testing(n_samples=10000):
    """
    Create sample data that mimics the structure of the real RbSQLi dataset.
    This is only used when the real CSV file is not available.

    This function generates data that follows the same patterns as the real dataset
    but is synthetic for development/testing purposes.

    Reference: Data structure from image.png and dataset documentation

    Args:
        n_samples (int): Number of samples to generate

    Returns:
        pandas.DataFrame: Synthetic dataset with realistic SQL injection patterns
    """
    print(f"\nCreating synthetic sample data with {n_samples:,} rows...")
    print("(This is for development only - use real CSV for production)")

    np.random.seed(SEED)

    # Define realistic SQL query patterns based on the RbSQLi dataset structure
    # These are generated programmatically to show patterns rather than hardcoded

    # Benign SQL operations (common legitimate queries)
    benign_templates = {
        'SELECT': [
            "SELECT {cols} FROM {table} WHERE {condition}",
            "SELECT {cols} FROM {table}",
            "SELECT COUNT(*) FROM {table} WHERE {condition}",
            "SELECT DISTINCT {col} FROM {table} ORDER BY {col}",
            "SELECT {cols} FROM {table} JOIN {table2} ON {join_condition}"
        ],
        'INSERT': [
            "INSERT INTO {table} ({cols}) VALUES ({values})",
            "INSERT INTO {table} VALUES ({values})"
        ],
        'UPDATE': [
            "UPDATE {table} SET {assignments} WHERE {condition}",
            "UPDATE {table} SET {assignment} WHERE id = {id}"
        ],
        'DELETE': [
            "DELETE FROM {table} WHERE {condition}",
            "DELETE FROM {table}"
        ],
        'CREATE': [
            "CREATE TABLE {table} ({schema})",
            "CREATE INDEX idx_{col} ON {table} ({col})"
        ],
        'ALTER': [
            "ALTER TABLE {table} ADD COLUMN {col} {datatype}",
            "ALTER TABLE {table} DROP COLUMN {col}"
        ]
    }

    # Malicious injection patterns for each attack type
    # Following the RbSQLi dataset's six attack types
    malicious_patterns = {
        'Union-based': [
            "SELECT {cols} FROM {table} WHERE {col} = {value} UNION SELECT {malicious_cols} FROM {malicious_table}",
            "SELECT {cols} FROM {table} WHERE {col} = '{value}' UNION ALL SELECT {malicious_cols} FROM {malicious_table}",
            "SELECT * FROM {table} WHERE {col} = {value} UNION SELECT 1,2,3,4,5,6,7,8,9,0"
        ],
        'Error-based': [
            "SELECT * FROM {table} WHERE {col} = {value} AND 1=CONVERT(int, @@version)",
            "SELECT {cols} FROM {table} WHERE {col} = '{value}' AND 1=CAST('a' AS INT)",
            "SELECT * FROM {table} WHERE {col} = {value} AND (SELECT COUNT(*) FROM {table2}) > 0"
        ],
        'Boolean-based': [
            "SELECT * FROM {table} WHERE {col} = {value} AND 1=1",
            "SELECT * FROM {table} WHERE {col} = '{value}' AND '1'='1'",
            "SELECT * FROM {table} WHERE {col} = {value} OR 1=1"
        ],
        'Time-based': [
            "SELECT * FROM {table} WHERE {col} = {value} AND SLEEP(5)",
            "SELECT * FROM {table} WHERE {col} = '{value}' AND BENCHMARK(10000000, MD5('a'))",
            "SELECT * FROM {table} WHERE {col} = {value} AND pg_sleep(10)"
        ],
        'Stack-queries-based': [
            "SELECT * FROM {table} WHERE {col} = {value}; DROP TABLE {table};",
            "SELECT * FROM {table} WHERE {col} = '{value}'; DELETE FROM {table2}; INSERT INTO {table3} VALUES('data');",
            "SELECT * FROM {table} WHERE {col} = {value}; UPDATE {table} SET {col} = 'hacked' WHERE 1=1;"
        ],
        'Meta-based': [
            "SELECT * FROM {table} WHERE {col} = {value} AND 1=0 UNION SELECT table_name FROM information_schema.tables",
            "SELECT * FROM {table} WHERE {col} = {value} AND (SELECT COUNT(*) FROM (SELECT 1 UNION SELECT 2) AS t) > 0",
            "SELECT * FROM {table} WHERE {col} = {value} AND 1=1 ORDER BY (SELECT NULL UNION SELECT NULL)"
        ]
    }

    # Helper function to generate a random benign query
    def generate_benign_query():
        command = np.random.choice(list(benign_templates.keys()))
        template = np.random.choice(benign_templates[command])

        # Common table names (following typical database schemas)
        tables = ['users', 'products', 'orders', 'payments', 'customers',
                  'employees', 'sessions', 'logs', 'transactions', 'inventory']
        columns = ['id', 'name', 'email', 'created_at', 'status', 'price',
                   'quantity', 'user_id', 'product_id', 'order_id']

        # Fill template placeholders
        query = template.format(
            cols=np.random.choice(columns, size=np.random.randint(1, 4), replace=False).tolist(),
            col=np.random.choice(columns),
            table=np.random.choice(tables),
            table2=np.random.choice(tables),
            condition=f"{np.random.choice(columns)} {np.random.choice(['=', '>', '<', 'LIKE'])} '{np.random.choice(['active', 'completed', 'pending'])}'",
            values=", ".join([str(np.random.randint(1, 1000)) for _ in range(np.random.randint(1, 4))]),
            assignments=f"{np.random.choice(columns)} = {np.random.randint(1, 100)}",
            assignment=f"{np.random.choice(columns)} = '{np.random.choice(['new', 'updated', 'modified'])}'",
            id=np.random.randint(1, 1000),
            schema=f"{np.random.choice(columns)} {np.random.choice(['INT', 'VARCHAR(255)', 'DATE', 'BOOLEAN'])}",
            datatype=np.random.choice(['INT', 'VARCHAR(255)', 'TEXT', 'TIMESTAMP']),
            join_condition=f"{np.random.choice(tables)}.{np.random.choice(columns)} = {np.random.choice(tables)}.{np.random.choice(columns)}"
        )

        return query, command

    # Helper function to generate a random malicious query
    def generate_malicious_query(attack_type):
        patterns = malicious_patterns.get(attack_type, malicious_patterns['Union-based'])
        template = np.random.choice(patterns)

        tables = ['users', 'admin', 'payments', 'credit_cards', 'passwords',
                  'login', 'credentials', 'user_data', 'secure_info', 'secrets']
        columns = ['id', 'username', 'password', 'email', 'credit_card',
                   'ssn', 'token', 'api_key', 'session_id', 'auth_token']

        query = template.format(
            cols=np.random.choice(columns, size=np.random.randint(1, 3), replace=False),
            col=np.random.choice(columns),
            table=np.random.choice(tables),
            table2=np.random.choice(tables),
            malicious_table=np.random.choice(['admin', 'users', 'passwords', 'credit_cards']),
            malicious_cols="username, password",
            value=np.random.choice(["1", "1' OR '1'='1", "'admin'", "NULL"]),
            value2=np.random.randint(1, 100)
        )

        return query

    # Generate data
    data = []
    n_benign = int(n_samples * 0.7)  # 70% benign
    n_malicious = n_samples - n_benign  # 30% malicious

    # Generate benign samples
    attack_types = list(malicious_patterns.keys())
    samples_per_attack = n_malicious // len(attack_types)

    for i in range(n_benign):
        query, sql_command = generate_benign_query()
        data.append({
            'sql_query': query,
            'injection_type': 'None_Type',
            'vulnerability_status': 'No',
            'sql_command': sql_command,
            'target_table': query.lower().split('from')[-1].split()[0] if 'from' in query.lower() else 'unknown',
            'selected_columns': '*',
            'comparison_operator': '=',
            'logical_operator': 'AND',
            'sql_comment_syntax': ''
        })

    # Generate malicious samples for each attack type
    for attack_type in attack_types:
        for i in range(samples_per_attack):
            query = generate_malicious_query(attack_type)
            data.append({
                'sql_query': query,
                'injection_type': attack_type,
                'vulnerability_status': 'Yes',
                'sql_command': query.split()[0].upper() if query.split() else 'SELECT',
                'target_table': query.lower().split('from')[-1].split()[0] if 'from' in query.lower() else 'unknown',
                'selected_columns': '*',
                'comparison_operator': '=',
                'logical_operator': 'AND',
                'sql_comment_syntax': '--' if '--' in query else ''
            })

    df = pd.DataFrame(data)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"\n✓ Synthetic dataset created successfully!")
    print(f"  Total rows: {len(df):,}")
    print(f"  Benign: {len(df[df['vulnerability_status'] == 'No']):,} ({(len(df[df['vulnerability_status'] == 'No'])/len(df))*100:.1f}%)")
    print(f"  Malicious: {len(df[df['vulnerability_status'] == 'Yes']):,} ({(len(df[df['vulnerability_status'] == 'Yes'])/len(df))*100:.1f}%)")
    print("\n  Attack type distribution:")
    for attack_type in attack_types:
        count = len(df[df['injection_type'] == attack_type])
        print(f"    {attack_type}: {count:,}")

    return df


def verify_dataset_integrity(df):
    """
    Verify the integrity of the loaded dataset.

    Checks:
    - No missing values in critical columns
    - Valid labels in target column
    - No duplicate SQL queries (if expected)
    - Distribution of classes is reasonable

    Args:
        df (pandas.DataFrame): Dataset to verify

    Returns:
        bool: True if dataset passes integrity checks
    """
    print("\n" + "=" * 60)
    print("VERIFYING DATASET INTEGRITY")
    print("=" * 60)

    issues_found = False

    # Check for required columns
    required_columns = ['sql_query', 'vulnerability_status']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"✗ Missing required columns: {missing_columns}")
        issues_found = True
    else:
        print("✓ Required columns present")

    # Check for missing values in critical columns
    if 'sql_query' in df.columns:
        missing_queries = df['sql_query'].isnull().sum()
        if missing_queries > 0:
            print(f"✗ {missing_queries:,} rows have missing SQL queries")
            issues_found = True
        else:
            print("✓ No missing SQL queries")

    # Check label distribution
    if 'vulnerability_status' in df.columns:
        valid_labels = ['Yes', 'No']
        invalid_labels = df[~df['vulnerability_status'].isin(valid_labels)]
        if len(invalid_labels) > 0:
            print(f"✗ Found {len(invalid_labels)} rows with invalid labels")
            print(f"  Invalid values: {invalid_labels['vulnerability_status'].unique()}")
            issues_found = True
        else:
            print("✓ All labels are valid")

    # Check for duplicate queries (might be intentional or error)
    if 'sql_query' in df.columns:
        duplicate_queries = df['sql_query'].duplicated().sum()
        if duplicate_queries > 0:
            print(f"⚠ Warning: {duplicate_queries:,} duplicate SQL queries found")
        else:
            print("✓ No duplicate SQL queries")

    # Check class balance
    if 'vulnerability_status' in df.columns:
        class_dist = df['vulnerability_status'].value_counts(normalize=True)
        if 'Yes' in class_dist and class_dist['Yes'] < 0.01:
            print(f"⚠ Warning: Very low malicious class proportion: {class_dist['Yes']*100:.4f}%")
            print("  This may cause model training issues. Consider class weighting.")
        elif 'Yes' in class_dist and class_dist['Yes'] > 0.5:
            print(f"⚠ Warning: Imbalanced dataset - {class_dist['Yes']*100:.1f}% malicious")

    if not issues_found:
        print("\n✓ Dataset passed all integrity checks!")
    else:
        print("\n⚠ Dataset has issues that need to be addressed before training.")

    return not issues_found



# ============================================
# MAIN EXECUTION WITH REAL DATA LOADING
# ============================================

print("\n" + "=" * 80)
print("SQL INJECTION DETECTION SYSTEM - REAL DATA MODE")
print("=" * 80)

# Configuration
DATA_FILE = './rbsqli_dataset_1k.csv'  # Update with actual file path
USE_SAMPLE_FOR_DEVELOPMENT = True   # Set to False to load full dataset
SAMPLE_SIZE = 500000                 # Number of rows to sample for development

# Load the real dataset
try:
    # Attempt to load real data
    df = load_real_rbsqli_dataset(
        filepath=DATA_FILE,
        sample_size=SAMPLE_SIZE if USE_SAMPLE_FOR_DEVELOPMENT else None
    )

    # Verify dataset integrity
    verify_dataset_integrity(df)

    # Display additional statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    if 'injection_type' in df.columns:
        print("\nInjection Type Breakdown:")
        type_counts = df['injection_type'].value_counts()
        for inj_type, count in type_counts.items():
            percentage = (count / len(df)) * 100
            bar_length = int(percentage / 2)  # Scale for display
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"  {inj_type:<20} {count:>8,} ({percentage:>5.1f}%) {bar}")

    if 'sql_command' in df.columns:
        print("\nSQL Command Distribution:")
        cmd_counts = df['sql_command'].value_counts().head(10)
        for cmd, count in cmd_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {cmd:<15} {count:>8,} ({percentage:>5.1f}%)")

    if 'target_table' in df.columns:
        print("\nMost Common Target Tables:")
        table_counts = df['target_table'].value_counts().head(10)
        for table, count in table_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {table:<20} {count:>8,} ({percentage:>5.1f}%)")

except FileNotFoundError:
    print("\n" + "=" * 60)
    print("REAL DATASET NOT FOUND - USING SYNTHETIC DATA")
    print("=" * 60)
    print("\nTo use the real RbSQLi dataset:")
    print("1. Download from: https://data.mendeley.com/datasets/xz4d5zj5yw/4")
    print("2. Place the CSV file in the current directory")
    print("3. Update the DATA_FILE variable with the correct filename")
    print("\nFor now, continuing with synthetic sample data...")

    df = create_sample_data_for_testing(n_samples=50000)

# Continue with preprocessing and model training as before...
print("\n✓ Data loading complete. Proceeding to preprocessing...")


# ============================================
# SECTION 3: DATA PREPROCESSING
# Following style from 7CS033_Assignment_Task_1 - 6.pdf (Task 2.1)
# ============================================

print("\n" + "=" * 80)
print("SECTION 2: DATA PREPROCESSING")
print("=" * 80)


def preprocess_sql_query(sql_query):
    """
    Clean and normalize SQL queries for feature extraction.

    Reference: Text cleaning from 7CS033 PDF Task 6 (lines 5-12)
    and preprocessing from Task1_DNN_250460.py

    This function:
    1. Converts to lowercase for consistency
    2. Removes extra whitespace
    3. Normalizes numeric values to 'N'
    4. Normalizes string literals to 'S'

    Args:
        sql_query (str): Raw SQL query string

    Returns:
        str: Cleaned and normalized SQL query
    """
    if pd.isna(sql_query):
        return ""

    query = str(sql_query).lower().strip()

    # Remove excessive whitespace
    query = ' '.join(query.split())

    # Normalize numeric values to 'N' (this helps detect patterns)
    import re
    query = re.sub(r'\b\d+\b', 'N', query)

    # Normalize string literals to 'S'
    query = re.sub(r"'[^']*'", "'S'", query)
    query = re.sub(r'"[^"]*"', '"S"', query)

    return query


def extract_sql_features(df):
    """
    Extract additional features from SQL queries for better detection.

    Reference: Feature extraction inspired by Task3_RNN_2504607.py data preparation

    This extracts features like:
    - Length of query
    - Number of special characters
    - Presence of UNION, OR, AND operators
    - Number of quotes, semicolons, etc.

    Args:
        df (pandas.DataFrame): DataFrame with 'sql_query' column

    Returns:
        pandas.DataFrame: DataFrame with additional engineered features
    """
    df = df.copy()

    # Clean the SQL queries
    df['clean_sql'] = df['sql_query'].apply(preprocess_sql_query)
    df['query_length'] = df['clean_sql'].apply(len)
    df['word_count'] = df['clean_sql'].apply(lambda x: len(x.split()))

    # SQL injection indicators
    sql_keywords = ['union', 'select', 'insert', 'update', 'delete', 'drop', 'create',
                    'alter', 'exec', 'execute', 'sleep', 'benchmark', 'pg_sleep',
                    'waitfor', 'delay', 'having', 'where', 'order by', 'group by']

    for keyword in sql_keywords:
        df[f'has_{keyword.replace(" ", "_")}'] = df['clean_sql'].apply(
            lambda x: 1 if keyword in str(x) else 0
        )

    # Special character counts (common in injections)
    special_chars = ["'", '"', ';', '--', '#', '/*', '*/', '||', '&&', '=', '>', '<']
    for char in special_chars:
        df[f'count_{char.replace("*", "star").replace("/", "slash")}'] = df['sql_query'].apply(
            lambda x: str(x).count(char)
        )

    # Heuristic features
    df['has_comment'] = df['sql_query'].apply(lambda x: 1 if any(c in str(x) for c in ['--', '#', '/*']) else 0)
    df['has_multiple_queries'] = df['sql_query'].apply(lambda x: 1 if str(x).count(';') > 1 else 0)
    df['has_union'] = df['clean_sql'].apply(lambda x: 1 if 'union' in str(x) else 0)
    df['has_sleep'] = df['clean_sql'].apply(lambda x: 1 if any(s in str(x) for s in ['sleep', 'benchmark', 'waitfor']) else 0)

    # Logical operator counts
    df['or_count'] = df['clean_sql'].apply(lambda x: str(x).count(' or '))
    df['and_count'] = df['clean_sql'].apply(lambda x: str(x).count(' and '))
    df['not_count'] = df['clean_sql'].apply(lambda x: str(x).count(' not '))

    # Quote balance (odd number of quotes might indicate injection attempt)
    df['single_quote_count'] = df['sql_query'].apply(lambda x: str(x).count("'"))
    df['double_quote_count'] = df['sql_query'].apply(lambda x: str(x).count('"'))
    df['unbalanced_quotes'] = ((df['single_quote_count'] % 2 == 1) | (df['double_quote_count'] % 2 == 1)).astype(int)

    return df

# Preprocess the data
print("\nPreprocessing SQL queries and extracting features...")
df_processed = extract_sql_features(df);


# Prepare target variable
if 'vulnerability_status' in df_processed.columns:
    # Encode target (Yes/No -> 1/0)
    df_processed['target'] = df_processed['vulnerability_status'].apply(lambda x: 1 if x == 'Yes' else 0)
    target_column = 'target'
else:
    target_column = 'vulnerability_status'

print(f"\n✓ Data preprocessing complete")
print(f"  Feature shape: {df_processed.shape}")

# Create train/validation/test split (following Task1_DNN_250460.py style)
print("\nCreating train/validation/test split...")

# First split: Separate test set (15%)
X_trainVal, X_test, y_trainVal, y_test = train_test_split(
    df_processed, df_processed[target_column] if target_column == 'target' else df_processed['vulnerability_status'],
    test_size=0.15,
    random_state=SEED,
    stratify=df_processed[target_column] if target_column == 'target' else df_processed['vulnerability_status']
)

# Second split: Separate validation set (17.6% of trainVal = 15% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainVal, y_trainVal,
    test_size=0.176,  # This gives ~15% validation, 70% training
    random_state=SEED,
    stratify=y_trainVal
)

print(f"\nSplit Statistics:")
print(f"  Train set: {X_train.shape[0]} samples")
print(f"  Validation set: {X_val.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")
print(f"  Train class distribution: {y_train.value_counts().to_dict() if hasattr(y_train, 'value_counts') else 'N/A'}")
print(f"  Val class distribution: {y_val.value_counts().to_dict() if hasattr(y_val, 'value_counts') else 'N/A'}")
print(f"  Test class distribution: {y_test.value_counts().to_dict() if hasattr(y_test, 'value_counts') else 'N/A'}")



# ============================================
# SECTION 4: FEATURE ENGINEERING FOR TEXT DATA
# Following style from Task3_RNN_2504607.py and 7CS033 PDF (Task 6)
# ============================================

print("\n" + "=" * 80)
print("SECTION 3: FEATURE ENGINEERING")
print("=" * 80)

def create_tfidf_features(df_train, df_val, df_test, text_column='clean_sql', max_features=5000):
    """
    Create TF-IDF features from SQL queries.

    Reference: CountVectorizer usage from 7CS033 PDF Task 6 (line 50-53)
    TF-IDF is better than simple count for text classification as it
    downscales common words and highlights distinctive patterns.

    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        df_test: Test DataFrame
        text_column: Column containing text to vectorize
        max_features: Maximum number of features to create

    Returns:
        tuple: (X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer)
    """
    print("\nCreating TF-IDF features from SQL queries...")

    # Initialize TF-IDF vectorizer with n-grams to capture patterns
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
        analyzer='char_wb',    # Character-based for SQL patterns
        strip_accents='unicode'
    )

    # Fit on training data only
    X_train_tfidf = tfidf.fit_transform(df_train[text_column].fillna(''))
    X_val_tfidf = tfidf.transform(df_val[text_column].fillna(''))
    X_test_tfidf = tfidf.transform(df_test[text_column].fillna(''))

    print(f"✓ TF-IDF features created: {X_train_tfidf.shape[1]} features")
    print(f"  Train shape: {X_train_tfidf.shape}")
    print(f"  Val shape: {X_val_tfidf.shape}")
    print(f"  Test shape: {X_test_tfidf.shape}")

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf


def create_numerical_features(df_train, df_val, df_test, exclude_columns=['sql_query', 'clean_sql', 'vulnerability_status', 'target', 'injection_type']):
    """
    Extract numerical features for machine learning models.

    Reference: Feature selection style from Task1_DNN_250460.py data preprocessing

    Args:
        df_train, df_val, df_test: DataFrames
        exclude_columns: Columns to exclude from features

    Returns:
        tuple: (X_train_num, X_val_num, X_test_num, feature_names)
    """
    print("\nExtracting numerical features...")

    # Identify numerical columns
    numeric_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude specified columns
    feature_columns = [col for col in numeric_columns if col not in exclude_columns]

    print(f"  Selected {len(feature_columns)} numerical features:")
    print(f"  {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}")

    # Extract features
    X_train_num = df_train[feature_columns].values
    X_val_num = df_val[feature_columns].values
    X_test_num = df_test[feature_columns].values

    # Standardize features (following Task1_DNN_250460.py)
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_val_num_scaled = scaler.transform(X_val_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    print(f"  Train shape: {X_train_num_scaled.shape}")
    print(f"  Val shape: {X_val_num_scaled.shape}")
    print(f"  Test shape: {X_test_num_scaled.shape}")

    return X_train_num_scaled, X_val_num_scaled, X_test_num_scaled, scaler, feature_columns

# Create feature sets
print("\n" + "-" * 50)
print("Creating feature sets for model training...")
print("-" * 50)




# TF-IDF features (for capturing SQL patterns)
X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vectorizer = create_tfidf_features(
    X_train, X_val, X_test, text_column='clean_sql', max_features=3000
)

# Numerical features (for heuristic indicators)
X_train_num, X_val_num, X_test_num, num_scaler, num_features = create_numerical_features(
    X_train, X_val, X_test
)

# Combine features (concatenate horizontally)
X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_num])
X_val_combined = np.hstack([X_val_tfidf.toarray(), X_val_num])
X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_num])

print(f"\n✓ Combined feature matrix:")
print(f"  Train: {X_train_combined.shape}")
print(f"  Validation: {X_val_combined.shape}")
print(f"  Test: {X_test_combined.shape}")




# ============================================
# SECTION 5: TRADITIONAL MACHINE LEARNING MODELS
# Following style from 7CS033 PDF Task 3.1 (lines 20-60)
# ============================================

print("\n" + "=" * 80)
print("SECTION 4: TRADITIONAL MACHINE LEARNING MODELS")
print("=" * 80)

class SQLInjectionDetector:
    """
    Wrapper class for SQL injection detection models.

    Reference: Model building style from Task1_DNN_250460.py (build_model function)
    and from 7CS033 PDF Task 3.1 (lines 15-25)

    This class provides a unified interface for training and evaluating
    multiple machine learning models for SQL injection detection.
    """

    def __init__(self):
        self.models = {}
        self.results = []

    def build_models(self):
        """
        Initialize multiple classifier models.

        Models included:
        - Random Forest: Ensemble method good for imbalanced data
        - XGBoost/Gradient Boosting: Powerful for structured data
        - Logistic Regression: Baseline linear classifier
        - Decision Tree: Interpretable model
        - SVM with RBF kernel: Good for high-dimensional spaces
        - KNN: Simple distance-based classifier
        - Naive Bayes: Fast probabilistic classifier
        """
        # Random Forest - best for imbalanced data
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
            class_weight='balanced'
        )

        # Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=SEED
        )

        # Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=SEED,
            class_weight='balanced',
            n_jobs=-1
        )

        # Decision Tree
        self.models['Decision Tree'] = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=5,
            random_state=SEED,
            class_weight='balanced'
        )

        # SVM with RBF kernel
        self.models['SVM'] = SVC(
            kernel='rbf',
            C=2.0,
            gamma='scale',
            random_state=SEED,
            probability=True,
            class_weight='balanced'
        )

        # KNN
        self.models['KNN'] = KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            n_jobs=-1
        )

        # Naive Bayes (requires dense arrays)
        self.models['Naive Bayes'] = GaussianNB()

        print(f"✓ Initialized {len(self.models)} models")

    def train_all(self, X_train, y_train, X_val, y_val):
        """
        Train all models and record their performance.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        print("\nTraining models...")
        print("-" * 50)

        for name, model in self.models.items():
            print(f"\n[{name}] Training...")

            # For Naive Bayes, ensure dense arrays
            if name == 'Naive Bayes' and hasattr(X_train, 'toarray'):
                X_train_dense = X_train.toarray()
                X_val_dense = X_val.toarray()
            else:
                X_train_dense = X_train
                X_val_dense = X_val

            try:
                model.fit(X_train_dense, y_train)

                # Evaluate on validation set
                val_pred = model.predict(X_val_dense)
                val_pred_proba = model.predict_proba(X_val_dense)[:, 1] if hasattr(model, 'predict_proba') else val_pred

                val_accuracy = accuracy_score(y_val, val_pred)
                val_precision = precision_score(y_val, val_pred, zero_division=0)
                val_recall = recall_score(y_val, val_pred, zero_division=0)
                val_f1 = f1_score(y_val, val_pred, zero_division=0)

                print(f"  Validation - Acc: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

                self.results.append({
                    'model_name': name,
                    'model': model,
                    'val_accuracy': val_accuracy,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1': val_f1
                })

            except Exception as e:
                print(f"  ✗ Error training {name}: {str(e)[:100]}")
                continue

        print("\n" + "=" * 50)
        print("Model training completed!")

    def evaluate_best_on_test(self, X_test, y_test):
        """
        Evaluate the best performing model on the test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            dict: Test performance metrics
        """
        # Find best model by validation F1 score (following Task1 style)
        best_result = max(self.results, key=lambda x: x['val_f1'])
        best_model = best_result['model']
        best_name = best_result['model_name']

        print(f"\n" + "=" * 50)
        print(f"BEST MODEL: {best_name}")
        print(f"Validation F1 Score: {best_result['val_f1']:.4f}")
        print("=" * 50)

        # Handle dense/sparse for best model
        if best_name == 'Naive Bayes' and hasattr(X_test, 'toarray'):
            X_test_dense = X_test.toarray()
        else:
            X_test_dense = X_test

        # Predict on test set
        y_pred = best_model.predict(X_test_dense)
        y_pred_proba = best_model.predict_proba(X_test_dense)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred

        # Calculate comprehensive metrics, including weighted averages
        test_results = build_metric_summary(y_test, y_pred, y_pred_proba if hasattr(best_model, 'predict_proba') else None)
        test_results['model_name'] = best_name

        print(f"\nTEST SET PERFORMANCE:")
        print(f"  Accuracy:  {test_results['accuracy']:.4f}")
        print(f"  Precision: {test_results['precision']:.4f}")
        print(f"  Recall:    {test_results['recall']:.4f}")
        print(f"  F1 Score:  {test_results['f1_score']:.4f}")
        print(f"  Weighted F1: {test_results['weighted_f1']:.4f}")
        if test_results.get('auc') is not None:
            print(f"  AUC:       {test_results['auc']:.4f}")

        return test_results, best_model

# Train traditional ML models
print("\n" + "=" * 50)
print("TRADITIONAL ML MODELS TRAINING")
print("=" * 50)





# Create detector instance
ml_detector = SQLInjectionDetector()
ml_detector.build_models()
ml_detector.train_all(X_train_combined, y_train, X_val_combined, y_val)

# Evaluate best model on test set
ml_test_results, best_ml_model = ml_detector.evaluate_best_on_test(X_test_combined, y_test)

# ============================================
# SECTION 6: DEEP NEURAL NETWORK MODEL
# Following style from Task1_DNN_250460.py (build_model function)
# ============================================

print("\n" + "=" * 80)
print("SECTION 5: DEEP NEURAL NETWORK MODEL")
print("=" * 80)

def build_dnn_model(input_dim, hidden_units=None, activation='relu', learning_rate=0.001,
                     use_dropout=True, use_bn=True, dropout_rate=0.3):
    """
    Build a Deep Neural Network for SQL injection detection.

    Reference: build_model function from Task1_DNN_250460.py (lines 95-168)

    Architecture design rationale:
    - Input layer: Matches feature dimension
    - Hidden layers: Progressive reduction in neurons
    - Batch Normalization: Stabilizes training, allows higher learning rates
    - Dropout: Prevents overfitting on limited data
    - Output layer: Sigmoid for binary classification

    Args:
        input_dim (int): Number of input features
        hidden_units (list): List of hidden layer sizes
        activation (str): Activation function for hidden layers
        learning_rate (float): Learning rate for optimizer
        use_dropout (bool): Whether to use dropout layers
        use_bn (bool): Whether to use batch normalization
        dropout_rate (float): Dropout rate if use_dropout is True

    Returns:
        tensorflow.keras.Model: Compiled Keras model
    """
    if hidden_units is None:
        hidden_units = [128, 64, 32, 16]  # Start with more capacity

    model = Sequential()

    # Input layer (explicit Input layer as in Task1 style)
    model.add(Input(shape=(input_dim,)))

    # First hidden layer
    model.add(Dense(hidden_units[0], activation=activation))
    if use_bn:
        model.add(BatchNormalization())
    if use_dropout:
        model.add(Dropout(dropout_rate))

    # Additional hidden layers
    for units in hidden_units[1:]:
        model.add(Dense(units, activation=activation))
        if use_bn:
            model.add(BatchNormalization())
        if use_dropout:
            model.add(Dropout(dropout_rate))

    # Output layer with sigmoid for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Use Adam optimizer with appropriate learning rate
    optimizer = Adam(learning_rate=learning_rate)

    # Compile with binary crossentropy (appropriate for binary classification)
    # Following Task1 style: include AUC metric
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

def train_dnn_model(X_train, y_train, X_val, y_val, input_dim, experiment_name="DNN Model"):
    """
    Train DNN model with early stopping and learning rate reduction.

    Reference: Training loop from Task1_DNN_250460.py (lines 124-145)

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_dim: Input dimension
        experiment_name: Name for this experiment

    Returns:
        tuple: (trained_model, training_history, test_predictions)
    """
    print(f"\n[{experiment_name}] Building DNN model...")

    # Build model with architecture optimized for SQL injection detection
    # Using moderate capacity to prevent overfitting
    model = build_dnn_model(
        input_dim=input_dim,
        hidden_units=[64, 32, 16],
        activation='relu',
        learning_rate=0.001,
        use_dropout=True,
        use_bn=True,
        dropout_rate=0.3
    )

    print(f"  Model summary:")
    model.summary()

    # Callbacks for training optimization (following Task1 style)
    early_stop = EarlyStopping(
        monitor='val_auc',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=5,
        mode='max',
        min_lr=1e-6,
        verbose=1
    )

    print(f"\n  Training DNN model...")

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate on validation set
    val_pred = model.predict(X_val, verbose=1).ravel()
    val_auc = roc_auc_score(y_val, val_pred)
    val_loss, val_acc, val_auc_metric = model.evaluate(X_val, y_val, verbose=1)

    print(f"\n  Validation Results:")
    print(f"    Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | AUC: {val_auc:.4f}")

    return model, history

# Prepare data for DNN (ensure dense for TF)
X_train_dense = X_train_combined.astype(np.float32)
X_val_dense = X_val_combined.astype(np.float32)
X_test_dense = X_test_combined.astype(np.float32)

# Train DNN model
dnn_model, dnn_history = train_dnn_model(
    X_train_dense, y_train.values,
    X_val_dense, y_val.values,
    input_dim=X_train_combined.shape[1],
    experiment_name="DNN for SQL Injection Detection"
)

# Evaluate DNN on test set
print("\n" + "=" * 50)
print("DNN MODEL TEST EVALUATION")
print("=" * 50)

dnn_test_pred = dnn_model.predict(X_test_dense, verbose=1).ravel()
dnn_test_binary = (dnn_test_pred > 0.5).astype(int)

dnn_results = build_metric_summary(y_test, dnn_test_binary, dnn_test_pred)
dnn_results['model_name'] = 'Deep Neural Network'

print(f"\nTEST SET PERFORMANCE:")
print(f"  Accuracy:  {dnn_results['accuracy']:.4f}")
print(f"  Precision: {dnn_results['precision']:.4f}")
print(f"  Recall:    {dnn_results['recall']:.4f}")
print(f"  F1 Score:  {dnn_results['f1_score']:.4f}")
print(f"  AUC:       {dnn_results['auc']:.4f}")

# ============================================
# SECTION 7: RNN/LSTM MODEL FOR SEQUENCE PATTERNS
# Following style from Task3_RNN_2504607.py (build_simple_lstm function)
# ============================================

print("\n" + "=" * 80)
print("SECTION 6: RNN/LSTM MODEL FOR SQL PATTERN SEQUENCE")
print("=" * 80)

def prepare_sequence_data(df_train, df_val, df_test, text_column='clean_sql', max_sequence_length=120, vocab_size=128):
    """
    Prepare sequential data for RNN/LSTM models.

    Reference: splitSequence function and data preparation from Task3_RNN_2504607.py (lines 22-44)

    SQL queries are sequential by nature, making them suitable for RNN/LSTM models.
    This function tokenizes the queries and pads them to uniform length.

    Args:
        df_train, df_val, df_test: DataFrames with text data
        text_column: Column containing cleaned SQL queries
        max_sequence_length: Maximum length of sequences (padding/truncation)
        vocab_size: Size of vocabulary for tokenization

    Returns:
        tuple: (X_train_seq, X_val_seq, X_test_seq, tokenizer)
    """
    print("\nPreparing sequential data for sequence (RNN/LSTM) models...")

    # Initialize tokenizer (character-level for SQL patterns)
    tokenizer = Tokenizer(num_words=vocab_size, char_level=True, oov_token='<OOV>')

    # Fit tokenizer on training data only
    train_texts = df_train[text_column].fillna('').astype(str).tolist()
    tokenizer.fit_on_texts(train_texts)

    # Convert texts to sequences
    X_train_seq = tokenizer.texts_to_sequences(train_texts)
    X_val_seq = tokenizer.texts_to_sequences(df_val[text_column].fillna('').astype(str).tolist())
    X_test_seq = tokenizer.texts_to_sequences(df_test[text_column].fillna('').astype(str).tolist())

    # Pad sequences to uniform length
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_sequence_length, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post', truncating='post')

    print(f"  Vocabulary size: {min(vocab_size, len(tokenizer.word_index) + 1)}")
    print(f"  Max sequence length: {max_sequence_length}")
    print(f"  Train shape: {X_train_pad.shape}")
    print(f"  Val shape: {X_val_pad.shape}")
    print(f"  Test shape: {X_test_pad.shape}")

    return X_train_pad, X_val_pad, X_test_pad, tokenizer

def build_lstm_model(input_shape, vocab_size=128, embedding_dim=32, lstm_units=32):
    """
    Build LSTM model for SQL injection detection.

    Reference: build_simple_lstm and build_deeper_lstm from Task3_RNN_2504607.py (lines 65-91)

    LSTM (Long Short-Term Memory) networks are well-suited for sequence data
    as they can capture long-range dependencies in the SQL query structure.

    Args:
        input_shape: Shape of input sequences (max_length,)
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embedding layer
        lstm_units: Number of LSTM units

    Returns:
        tensorflow.keras.Model: Compiled LSTM model
    """
    model = Sequential([
        # Embedding layer to convert token indices to dense vectors
        Embedding(vocab_size, embedding_dim, input_length=input_shape[0]),

        # First LSTM layer
        LSTM(lstm_units, activation='tanh', return_sequences=True),
        Dropout(0.2),

        # Second LSTM layer (following deeper_lstm pattern)
        LSTM(lstm_units // 2, activation='tanh'),
        Dropout(0.2),

        # Dense layers for classification
        Dense(16, activation='relu'),

        # Output layer
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

def build_gru_model(input_shape, vocab_size=128, embedding_dim=32, gru_units=32):
    """
    Build GRU model for SQL injection detection.

    Reference: build_gru function from Task3_RNN_2504607.py (lines 98-109)

    GRU (Gated Recurrent Unit) is similar to LSTM but with fewer parameters,
    making it faster to train while maintaining good performance.

    Args:
        input_shape: Shape of input sequences
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embedding layer
        gru_units: Number of GRU units

    Returns:
        tensorflow.keras.Model: Compiled GRU model
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_shape[0]),

        GRU(gru_units, activation='tanh', return_sequences=True),
        Dropout(0.2),

        GRU(gru_units // 2, activation='tanh'),
        Dropout(0.2),

        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

# Prepare sequence data
SEQUENCE_TRAIN_LIMIT = 12000
SEQUENCE_VAL_LIMIT = 2500
SEQUENCE_TEST_LIMIT = 2500
max_seq_length = 120
vocab_size = 128

X_train_seq_df = X_train.head(min(SEQUENCE_TRAIN_LIMIT, len(X_train))).copy()
X_val_seq_df = X_val.head(min(SEQUENCE_VAL_LIMIT, len(X_val))).copy()
X_test_seq_df = X_test.head(min(SEQUENCE_TEST_LIMIT, len(X_test))).copy()

y_train_seq = y_train.iloc[:len(X_train_seq_df)].copy()
y_val_seq = y_val.iloc[:len(X_val_seq_df)].copy()
y_test_seq = y_test.iloc[:len(X_test_seq_df)].copy()

print("\nSequence model subset sizes:")
print(f"  Train subset: {len(X_train_seq_df)}")
print(f"  Validation subset: {len(X_val_seq_df)}")
print(f"  Test subset: {len(X_test_seq_df)}")
print(f"  Max sequence length: {max_seq_length}")
print(f"  Character vocabulary cap: {vocab_size}")

X_train_seq, X_val_seq, X_test_seq, seq_tokenizer = prepare_sequence_data(
    X_train_seq_df, X_val_seq_df, X_test_seq_df,
    text_column='clean_sql',
    max_sequence_length=max_seq_length,
    vocab_size=vocab_size
)

# Train LSTM model
print("\n" + "-" * 50)
print("TRAINING LSTM MODEL")
print("-" * 50)

lstm_model = build_lstm_model(
    input_shape=(max_seq_length,),
    vocab_size=vocab_size,
    embedding_dim=32,
    lstm_units=32
)

print("\nLSTM Model Summary:")
lstm_model.summary()

# Callbacks for LSTM training (following Task3 style)
early_stop_lstm = EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True, mode='max', verbose=1)
reduce_lr_lstm = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, mode='max', min_lr=1e-6, verbose=1)

print("\nTraining LSTM model...")
lstm_history = lstm_model.fit(
    X_train_seq, y_train_seq.values,
    validation_data=(X_val_seq, y_val_seq.values),
    epochs=12,
    batch_size=128,
    callbacks=[early_stop_lstm, reduce_lr_lstm],
    verbose=1
)

# Train GRU model
print("\n" + "-" * 50)
print("TRAINING GRU MODEL")
print("-" * 50)

gru_model = build_gru_model(
    input_shape=(max_seq_length,),
    vocab_size=vocab_size,
    embedding_dim=32,
    gru_units=32
)

print("\nGRU Model Summary:")
gru_model.summary()

early_stop_gru = EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True, mode='max', verbose=1)
reduce_lr_gru = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, mode='max', min_lr=1e-6, verbose=1)

print("\nTraining GRU model...")
gru_history = gru_model.fit(
    X_train_seq, y_train_seq.values,
    validation_data=(X_val_seq, y_val_seq.values),
    epochs=12,
    batch_size=128,
    callbacks=[early_stop_gru, reduce_lr_gru],
    verbose=1
)

# Evaluate LSTM on test set
print("\n" + "=" * 50)
print("LSTM MODEL TEST EVALUATION")
print("=" * 50)

lstm_test_pred = lstm_model.predict(X_test_seq, verbose=0).ravel()
lstm_test_binary = (lstm_test_pred > 0.5).astype(int)

lstm_results = build_metric_summary(y_test_seq, lstm_test_binary, lstm_test_pred)
lstm_results['model_name'] = 'LSTM (Character-level)'

print(f"  Accuracy:  {lstm_results['accuracy']:.4f}")
print(f"  Precision: {lstm_results['precision']:.4f}")
print(f"  Recall:    {lstm_results['recall']:.4f}")
print(f"  F1 Score:  {lstm_results['f1_score']:.4f}")
print(f"  AUC:       {lstm_results['auc']:.4f}")

# Evaluate GRU on test set
print("\n" + "=" * 50)
print("GRU MODEL TEST EVALUATION")
print("=" * 50)

gru_test_pred = gru_model.predict(X_test_seq, verbose=0).ravel()
gru_test_binary = (gru_test_pred > 0.5).astype(int)

gru_results = build_metric_summary(y_test_seq, gru_test_binary, gru_test_pred)
gru_results['model_name'] = 'GRU (Character-level)'

print(f"  Accuracy:  {gru_results['accuracy']:.4f}")
print(f"  Precision: {gru_results['precision']:.4f}")
print(f"  Recall:    {gru_results['recall']:.4f}")
print(f"  F1 Score:  {gru_results['f1_score']:.4f}")
print(f"  AUC:       {gru_results['auc']:.4f}")



# ============================================
# SECTION 8: MODEL COMPARISON AND SELECTION
# Following style from Task1_DNN_250460.py Question 4
# ============================================

print("\n" + "=" * 80)
print("SECTION 7: MODEL COMPARISON AND SELECTION")
print("=" * 80)

# Combine all results
all_results = [
    ml_test_results,
    dnn_results,
    lstm_results,
    gru_results
]

print("\nAll Models Performance Summary:")
print("-" * 80)
print(f"{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'W-F1':<12} {'AUC':<12}")
print("-" * 80)

for result in all_results:
    print(f"{result['model_name']:<30} {result['accuracy']:<12.4f} {result['precision']:<12.4f} "
          f"{result['recall']:<12.4f} {result['f1_score']:<12.4f} {result.get('weighted_f1', 0):<12.4f} {result.get('auc', 0):<12.4f}")

# Select best model based on F1 Score (balances precision and recall)
best_model_result = max(all_results, key=lambda x: x['f1_score'])
print("\n" + "=" * 80)
print(f"SELECTED BEST MODEL: {best_model_result['model_name']}")
print(f"  F1 Score: {best_model_result['f1_score']:.4f}")
print(f"  Accuracy: {best_model_result['accuracy']:.4f}")
print(f"  Precision: {best_model_result['precision']:.4f}")
print(f"  Recall: {best_model_result['recall']:.4f}")
print(f"  Weighted F1: {best_model_result.get('weighted_f1', 0):.4f}")
if best_model_result.get('auc'):
    print(f"  AUC: {best_model_result['auc']:.4f}")
print("=" * 80)

# Save the best model for API deployment
# Determine which model object to save based on which performed best
if best_model_result['model_name'] == 'Deep Neural Network':
    best_model_to_save = dnn_model
    best_model_type = 'dnn'
elif best_model_result['model_name'] == 'LSTM (Character-level)':
    best_model_to_save = lstm_model
    best_model_type = 'lstm'
elif best_model_result['model_name'] == 'GRU (Character-level)':
    best_model_to_save = gru_model
    best_model_type = 'gru'
else:
    best_model_to_save = best_ml_model
    best_model_type = 'ml'

# Save the selected model and preprocessing objects for the middleware API.
# This is the critical deployment step that bridges the training script and
# the Flask middleware layer.
saved_bundle_path = save_best_model_artifacts(
    model_to_save=best_model_to_save,
    model_type=best_model_type,
    best_model_result=best_model_result,
    tfidf_vectorizer=tfidf_vectorizer,
    num_scaler=num_scaler,
    feature_columns=num_features,
    seq_tokenizer=seq_tokenizer if best_model_type in ['lstm', 'gru'] else None,
    max_sequence_length=max_seq_length if best_model_type in ['lstm', 'gru'] else None
)





# ============================================
# SECTION 9: VISUALIZATIONS
# Following style from Task1_DNN_250460.py Question 5
# ============================================

print("\n" + "=" * 80)
print("SECTION 8: GENERATING VISUALIZATIONS")
print("=" * 80)

# Create a figures directory
os.makedirs('sql_injection_plots', exist_ok=True)

# PLOT 1: Model Comparison Bar Chart (following Task1 style)

fig = plt.figure(figsize=(16, 10))

# Accuracy comparison
ax1 = plt.subplot(2, 2, 1)
names = [r['model_name'] for r in all_results]
accs = [r['accuracy'] for r in all_results]

# Compare using the model name instead of the whole result dictionary.
# The result dictionaries contain NumPy arrays such as confusion matrices,
# and direct dictionary equality can trigger ambiguous array comparisons.
best_model_name = best_model_result['model_name']
colors = ['green' if r['model_name'] == best_model_name else 'steelblue' for r in all_results]
bars = ax1.bar(range(len(names)), accs, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels([f"{i+1}" for i in range(len(names))], rotation=0)
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
ax1.set_xticklabels(names, rotation=30, ha='right')

# F1 Score comparison
ax2 = plt.subplot(2, 2, 2)
f1_scores = [r['f1_score'] for r in all_results]
bars = ax2.bar(range(len(names)), f1_scores, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
ax2.set_title('Model F1 Score Comparison', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(names)))
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)
for bar, f1 in zip(bars, f1_scores):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
             f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
ax2.set_xticklabels(names, rotation=30, ha='right')

# Precision vs Recall comparison
ax3 = plt.subplot(2, 2, 3)
precision_scores = [r['precision'] for r in all_results]
recall_scores = [r['recall'] for r in all_results]
x = np.arange(len(names))
width = 0.35
ax3.bar(x - width/2, precision_scores, width, label='Precision', alpha=0.7, color='skyblue', edgecolor='black')
ax3.bar(x + width/2, recall_scores, width, label='Recall', alpha=0.7, color='lightcoral', edgecolor='black')
ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
ax3.set_title('Precision vs Recall Comparison', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_ylim(0, 1)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_xticklabels(names, rotation=30, ha='right')

# AUC comparison (if available)
ax4 = plt.subplot(2, 2, 4)
auc_scores = [r.get('auc', 0) for r in all_results]
bars = ax4.bar(range(len(names)), auc_scores, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('AUC Score', fontsize=11, fontweight='bold')
ax4.set_title('Model AUC Comparison', fontsize=12, fontweight='bold')
ax4.set_xticks(range(len(names)))
ax4.set_ylim(0, 1)
ax4.grid(axis='y', alpha=0.3)
for bar, auc in zip(bars, auc_scores):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
             f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
ax4.set_xticklabels(names, rotation=30, ha='right')

plt.tight_layout()
plt.savefig('sql_injection_plots/01_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sql_injection_plots/01_model_comparison.png")
plt.close()

# PLOT 2: Confusion Matrix for Best Model (following Task1 style for best model)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
cm = best_model_result['confusion_matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'])
ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax.set_ylabel('True', fontsize=12, fontweight='bold')
ax.set_title(f'Confusion Matrix: {best_model_result["model_name"]}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sql_injection_plots/02_best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sql_injection_plots/02_best_model_confusion_matrix.png")
plt.close()

# PLOT 3: Training History for Best Deep Learning Model
if best_model_result['model_name'] in ['Deep Neural Network', 'LSTM (Character-level)', 'GRU (Character-level)']:

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Get the appropriate history
    if best_model_result['model_name'] == 'Deep Neural Network':
        history = dnn_history
    elif best_model_result['model_name'] == 'LSTM (Character-level)':
        history = lstm_history
    else:
        history = gru_history

    # Loss curves
    ax1.plot(history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(history.history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('Training vs Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # AUC curves
    if 'auc' in history.history and 'val_auc' in history.history:
        ax3.plot(history.history['auc'], 'b-', linewidth=2, label='Training AUC')
        ax3.plot(history.history['val_auc'], 'r-', linewidth=2, label='Validation AUC')
        ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax3.set_ylabel('AUC', fontsize=11, fontweight='bold')
        ax3.set_title('Training vs Validation AUC', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sql_injection_plots/03_best_model_training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: sql_injection_plots/03_best_model_training_history.png")
    plt.close()

# PLOT 4: ROC Curves for All Models
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Calculate and plot ROC curves for models that have probability predictions
model_predictions = {
    'DNN': {
        'preds': dnn_test_pred if 'dnn_test_pred' in dir() else None,
        'y_true': y_test
    },
    'LSTM': {
        'preds': lstm_test_pred if 'lstm_test_pred' in dir() else None,
        'y_true': y_test_seq if 'y_test_seq' in dir() else y_test
    },
    'GRU': {
        'preds': gru_test_pred if 'gru_test_pred' in dir() else None,
        'y_true': y_test_seq if 'y_test_seq' in dir() else y_test
    }
}

colors = ['blue', 'green', 'red', 'orange']
color_idx = 0

for model_name, model_data in model_predictions.items():
    preds = model_data['preds']
    y_true = model_data['y_true']

    if preds is not None:
        fpr, tpr, _ = roc_curve(y_true, preds)
        auc_score = roc_auc_score(y_true, preds)
        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc_score:.4f})', color=colors[color_idx])
        color_idx += 1

# For ML model if it has predict_proba
if hasattr(best_ml_model, 'predict_proba'):
    ml_pred_proba = best_ml_model.predict_proba(X_test_combined)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, ml_pred_proba)
    auc_score = roc_auc_score(y_test, ml_pred_proba)
    ax.plot(fpr, tpr, linewidth=2, label=f'ML Model (AUC={auc_score:.4f})', color=colors[color_idx])

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves for SQL Injection Detection Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sql_injection_plots/04_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sql_injection_plots/04_roc_curves.png")
plt.close()

print("\n✓ All visualizations saved to 'sql_injection_plots/' directory")

# ============================================
# SECTION 10: FLASK API FOR SQL INJECTION DETECTION
# ============================================
