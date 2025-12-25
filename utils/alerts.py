"""
SageWall Alerting Module.

This module provides real-time threat notification capabilities via AWS SNS.
When the ML model detects a high-confidence attack (>90%), this module
pushes an alert to subscribed endpoints (email, SMS, Slack, etc.).

Architecture Context:
    SageMaker Endpoint → This Module → AWS SNS → Email/SMS/Webhook

Dependencies:
    - boto3: AWS SDK for Python (SNS client)
    - Active SNS Topic with confirmed subscriptions

Author: Andrew Zheng
"""

import boto3


def send_alert(score: float, topic_arn: str) -> bool:
    """
    Publish a threat alert to AWS SNS if confidence exceeds threshold.

    This function acts as a "circuit breaker" — it only fires when the
    threat score is critically high (>0.90), preventing alert fatigue
    from low-confidence detections.

    Args:
        score: ML model's prediction confidence (0.0 = normal, 1.0 = attack).
               Only scores > 0.90 trigger an alert.
        topic_arn: The Amazon Resource Name of the SNS topic.
                   Format: arn:aws:sns:<region>:<account-id>:<topic-name>

    Returns:
        bool: True if alert was successfully published, False if skipped
              (score too low) or if the publish operation failed.

    Example:
        >>> send_alert(0.99, "arn:aws:sns:us-east-1:123456789:sagewall-alerts")
        🚨 Alert sent! MessageId: abc123...
        True

    Note:
        This function is designed to fail gracefully. If AWS credentials
        are missing or the topic doesn't exist, it logs a warning but
        does NOT raise an exception — ensuring the main application
        continues running.
    """
    # Threshold gate: Avoid alert fatigue by only notifying on high-confidence threats
    # 0.90 chosen as balance between catching real attacks and reducing false positives
    if score <= 0.90:
        return False

    try:
        # Initialize SNS client for AWS Simple Notification Service
        # SNS provides pub/sub messaging to fan out alerts to multiple subscribers
        sns = boto3.client('sns')

        # Publish message to SNS topic
        # All subscribers (email, SMS, Lambda, HTTP endpoints) receive this notification
        response = sns.publish(
            TopicArn=topic_arn,
            Subject='🚨 SAGEWALL ALERT',
            Message=f'Critical Threat Detected! Confidence: {score:.4f}'
        )

        print(f"🚨 Alert sent! MessageId: {response['MessageId']}")
        return True

    except Exception as e:
        # Graceful degradation: Log the failure but don't crash the inference pipeline
        # Production systems should send this to CloudWatch Logs for monitoring
        print(f"⚠️ WARNING: Failed to send SNS alert - {type(e).__name__}: {e}")
        return False
