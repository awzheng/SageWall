"""
SageWall ETL Lambda Function.

This serverless function performs Extract-Transform-Load (ETL) operations
on raw NSL-KDD network logs, preparing them for SageMaker XGBoost training.

Architecture Context:
    S3 Raw Bucket → [THIS LAMBDA] → S3 Processed Bucket → SageMaker Training

Trigger:
    Automatically invoked by S3 PUT events when new files land in the raw bucket.

Key Transformations:
    1. Schema enforcement (headerless CSV → named columns)
    2. Target engineering (multi-class labels → binary classification)
    3. Feature encoding (categorical → one-hot numeric)
    4. SageMaker formatting (target column first, no headers)

Runtime Configuration:
    - Python 3.11
    - Memory: 1024 MB (vertical scaling for pandas operations)
    - Timeout: 3 minutes
    - Layer: AWSSDKPandas (provides awswrangler + pandas)

Author: Andrew Zheng
"""

import awswrangler as wr
import pandas as pd
import urllib.parse

# =============================================================================
# SCHEMA CONFIGURATION
# =============================================================================

# NSL-KDD dataset ships without headers — we must enforce schema manually
# These 42 columns represent network connection features + labels
# Reference: https://www.unb.ca/cic/datasets/nsl.html
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Categorical features requiring one-hot encoding
# XGBoost cannot process string values — must convert to numeric indicator columns
CATEGORICAL_COLUMNS = ['protocol_type', 'service', 'flag']

# S3 bucket configuration — data flows from raw → processed
RAW_BUCKET = 'sagewall-raw-zheng-1b'
PROCESSED_BUCKET = 'sagewall-processed-zheng-1b'


def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point for S3-triggered ETL processing.

    This function transforms raw NSL-KDD network logs into a format
    compatible with SageMaker's built-in XGBoost algorithm.

    Args:
        event: S3 event payload containing bucket name and object key.
               Structure: {'Records': [{'s3': {'bucket': {...}, 'object': {...}}}]}
        context: Lambda runtime context (unused but required by AWS).

    Returns:
        dict: Response with status code and processing metadata.
              Example: {'statusCode': 200, 'body': {'records_processed': 125973}}

    Processing Pipeline:
        1. EXTRACT: Read headerless CSV from S3 using awswrangler
        2. TRANSFORM:
           - Add column names (schema enforcement)
           - Drop 'difficulty' column (not useful for binary classification)
           - Convert multi-class 'label' to binary 'attack' (0/1)
           - Move target to column 0 (SageMaker XGBoost requirement)
           - One-hot encode categorical features
           - Cast all values to float (XGBoost numeric requirement)
        3. LOAD: Write to processed bucket (no headers, no index)

    Raises:
        Exception: Propagates any S3 read/write or pandas processing errors
                   to Lambda for CloudWatch logging and retry handling.
    """
    # ==========================================================================
    # EXTRACT: Parse S3 event and read raw data
    # ==========================================================================

    # S3 events contain an array of records; we process one file per invocation
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']

    # URL-decode the key: S3 event encodes special characters (spaces → %20)
    key = urllib.parse.unquote_plus(record['s3']['object']['key'])

    print(f"Processing file: s3://{bucket}/{key}")

    # Construct full S3 URIs for awswrangler
    input_path = f"s3://{bucket}/{key}"
    output_path = f"s3://{PROCESSED_BUCKET}/{key}"

    # Read CSV via awswrangler — handles S3 authentication automatically
    # header=None because NSL-KDD ships without column headers
    df = wr.s3.read_csv(input_path, header=None)
    print(f"Read {len(df)} records from raw bucket")

    # ==========================================================================
    # TRANSFORM: Prepare data for SageMaker XGBoost
    # ==========================================================================

    # Step 1: Enforce schema by assigning column names
    df.columns = COLUMN_NAMES

    # Step 2: Drop 'difficulty' — this is a dataset-specific score, not a feature
    # Keeping it would leak information about attack complexity into the model
    df = df.drop(columns=['difficulty'])

    # Step 3: Target Engineering — convert multi-class to binary classification
    # Original labels: 'normal', 'dos', 'probe', 'r2l', 'u2r'
    # Binary target: 0 = normal, 1 = any attack type
    df['attack'] = (df['label'] != 'normal').astype(int)

    # Remove original label column now that we have binary target
    df = df.drop(columns=['label'])

    # Step 4: Move target to first column — CRITICAL for SageMaker XGBoost
    # SageMaker's built-in XGBoost expects: [target, feature1, feature2, ...]
    # Without this, the model trains on wrong columns and produces garbage
    attack_col = df.pop('attack')
    df.insert(0, 'attack', attack_col)

    # Step 5: One-hot encode categorical features
    # dtype=int prevents pandas from outputting Boolean (True/False)
    # which XGBoost cannot process — it requires numeric values only
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, dtype=int)

    # Step 6: Final type cast — ensure ALL values are numeric floats
    # This catches any edge cases where strings or objects might remain
    # XGBoost will crash with cryptic errors if non-numeric data slips through
    df = df.astype(float)

    print(f"Processed dataframe shape: {df.shape}")
    print(f"Columns: {list(df.columns[:10])}... (showing first 10)")

    # ==========================================================================
    # LOAD: Write processed data to S3
    # ==========================================================================

    # Write to processed bucket using awswrangler
    # index=False: Don't include row numbers as a column
    # header=False: SageMaker XGBoost expects no header row
    wr.s3.to_csv(df, output_path, index=False, header=False)

    print(f"Successfully wrote processed data to: {output_path}")

    # Return success response with metadata for monitoring
    return {
        'statusCode': 200,
        'body': {
            'message': 'Preprocessing complete',
            'input': input_path,
            'output': output_path,
            'records_processed': len(df),
            'features': df.shape[1]
        }
    }
