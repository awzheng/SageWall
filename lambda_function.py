"""
SageWall Preprocessing Lambda Function
Transforms raw NSL-KDD network logs for SageMaker XGBoost training.
"""

import awswrangler as wr
import pandas as pd
import urllib.parse

# NSL-KDD column headers (dataset has no headers)
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

# Categorical columns for one-hot encoding
CATEGORICAL_COLUMNS = ['protocol_type', 'service', 'flag']

# S3 bucket configuration
RAW_BUCKET = 'sagewall-raw-zheng-1b'
PROCESSED_BUCKET = 'sagewall-processed-zheng-1b'


def lambda_handler(event, context):
    """
    Lambda handler triggered by S3 PUT events on the raw bucket.
    Preprocesses NSL-KDD data for SageMaker XGBoost training.
    """
    # Extract bucket and key from S3 event
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(record['s3']['object']['key'])
    
    print(f"Processing file: s3://{bucket}/{key}")
    
    # Construct S3 paths
    input_path = f"s3://{bucket}/{key}"
    output_path = f"s3://{PROCESSED_BUCKET}/{key}"
    
    # 1. Read CSV without headers
    df = wr.s3.read_csv(input_path, header=None)
    print(f"Read {len(df)} records from raw bucket")
    
    # 2. Add column headers
    df.columns = COLUMN_NAMES
    
    # 3. Drop the difficulty column
    df = df.drop(columns=['difficulty'])
    
    # 4. Target Engineering
    # Create binary attack column: 0 = normal, 1 = attack
    df['attack'] = (df['label'] != 'normal').astype(int)
    
    # Drop the original label column
    df = df.drop(columns=['label'])
    
    # Move attack column to first position (SageMaker XGBoost requirement)
    attack_col = df.pop('attack')
    df.insert(0, 'attack', attack_col)
    
    # 5. One-hot encode categorical features (dtype=int to avoid Boolean values)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, dtype=int)
    
    # Safety check: ensure all values are numeric (SageMaker XGBoost requirement)
    df = df.astype(float)
    
    print(f"Processed dataframe shape: {df.shape}")
    print(f"Columns: {list(df.columns[:10])}... (showing first 10)")
    
    # 6. Write to processed bucket (no index, no header for SageMaker)
    wr.s3.to_csv(df, output_path, index=False, header=False)
    
    print(f"Successfully wrote processed data to: {output_path}")
    
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

