```eraser.io
title SageWall - ML Training (Write) Pipeline
// Data Sources
Network Logs [shape: oval, icon: file-text, color: blue, label: "NSL-KDD\nDataset"]
// Storage Layer
S3 Raw [icon: aws-simple-storage-service, label: "S3 Bucket\n(Raw)"]
S3 Processed [icon: aws-simple-storage-service, label: "S3 Bucket\n(Processed)"]
S3 Models [icon: aws-simple-storage-service, label: "S3 Bucket\n(Model Artifacts)"]
// Compute Layer
Lambda [icon: aws-lambda, label: "Lambda\n(ETL Pre-processing)"]
SageMaker Training [icon: aws-sagemaker, label: "SageMaker\nTraining"]
// Supporting Services
IAM [icon: aws-iam, label: "IAM\nRoles"]
CloudWatch [icon: aws-cloudwatch, label: "CloudWatch\nLogs"]
// Main Training Flow
Network Logs > S3 Raw: Upload raw data
S3 Raw > Lambda: S3 PUT Event trigger
Lambda > S3 Processed: Cleaned CSV
S3 Processed > SageMaker Training: 80/20 train/val split
SageMaker Training > S3 Models: Save model.tar.gz
// IAM Permissions
IAM > Lambda: Lambda execution role
IAM > SageMaker Training: SageMaker execution role
// Monitoring
Lambda > CloudWatch: Execution logs
SageMaker Training > CloudWatch: Training metrics
```