```eraser.io
title SageWall - Real-Time Inference (Read) Pipeline
// User Layer
User [icon: user, label: "User"]
// Frontend
Streamlit [icon: monitor, label: "Streamlit Frontend"]
// Inference Layer
Endpoint [icon: aws-sagemaker, label: "SageMaker Endpoint"]
// Storage
S3 Models [icon: aws-simple-storage-service, label: "S3 Bucket\n(Model Artifacts)"]
// Supporting Services
IAM [icon: aws-iam, label: "IAM\nRoles"]
CloudWatch [icon: aws-cloudwatch, label: "CloudWatch\nLogs"]
// Result Display
// User Interaction Flow
User > Streamlit: Enter packet CSV
Streamlit > Endpoint: invoke_endpoint()\nvia boto3
Endpoint > Streamlit: Return score (0.0-1.0)
// Model Loading (One-Time on Deployment)
S3 Models > Endpoint: Load model on\nendpoint creation
// IAM Permissions
IAM > Endpoint: SageMaker execution role
// Monitoring
Endpoint > CloudWatch: Invocation logs\n+ latency metrics
Streamlit > User: Return score
```