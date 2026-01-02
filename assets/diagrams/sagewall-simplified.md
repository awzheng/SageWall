```eraser.io
title SageWall Simplified AWS Pipeline

Network Logs [shape: oval, icon: file-text, color: blue]
Lambda [icon: aws-lambda]
S3 raw [icon: aws-simple-storage-service]
S3 processed [icon: aws-simple-storage-service]
Sagemaker[icon: aws-sagemaker]
IAM[icon: aws-iam]
Cloudwatch[icon: aws-cloudwatch]

Network Logs > S3 raw
S3 raw > Lambda
Lambda > S3 processed
S3 processed > Sagemaker
Sagemaker <> IAM
Sagemaker <> Cloudwatch
```