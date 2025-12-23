"""
SageWall Streamlit Frontend.

This module provides an interactive web interface for the SageWall
Intrusion Detection System. It allows users to submit network packet
features and receive real-time threat classifications from a deployed
SageMaker XGBoost model.

Architecture Context:
    User → [THIS APP] → SageMaker Runtime API → XGBoost Endpoint → Prediction

Key Features:
    - Real-time inference via SageMaker endpoint
    - Visual threat/safe classification display
    - Configurable detection threshold
    - Educational documentation for beginners

Tech Stack:
    - Streamlit: Web framework for data apps
    - boto3: AWS SDK for SageMaker Runtime invocation
    - Base64: SVG icon encoding for inline display

Usage:
    streamlit run app.py --server.headless true

Author: Andrew Zheng
"""

import streamlit as st

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# Must be called first — Streamlit requires this before any other st.* calls
st.set_page_config(
    page_title="SageWall IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# BRANDING ASSETS
# =============================================================================
# Inline SVG icons avoid external file dependencies and load instantly
# Design: Four rectangular segments with angular cutouts (Sage Barrier Wall motif)

ICON_BLACK = '''
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 50" fill="#000000">
  <!-- Block 1: Left - angular cuts on right side -->
  <polygon points="0,0 40,0 40,12 32,20 40,28 32,36 40,50 0,50"/>
  <!-- Block 2: Center-left - angular cuts both sides -->
  <polygon points="52,0 88,0 88,18 82,25 88,32 88,50 52,50 52,32 58,25 52,18"/>
  <!-- Block 3: Center-right - angular cuts both sides -->
  <polygon points="100,0 136,0 136,18 130,25 136,32 136,50 100,50 100,32 106,25 100,18"/>
  <!-- Block 4: Right - angular cuts on left side -->
  <polygon points="148,0 188,0 188,50 148,50 148,36 156,28 148,20 156,12 148,0"/>
</svg>
'''

ICON_WHITE = '''
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 50" fill="#FFFFFF">
  <!-- Block 1: Left - angular cuts on right side -->
  <polygon points="0,0 40,0 40,12 32,20 40,28 32,36 40,50 0,50"/>
  <!-- Block 2: Center-left - angular cuts both sides -->
  <polygon points="52,0 88,0 88,18 82,25 88,32 88,50 52,50 52,32 58,25 52,18"/>
  <!-- Block 3: Center-right - angular cuts both sides -->
  <polygon points="100,0 136,0 136,18 130,25 136,32 136,50 100,50 100,32 106,25 100,18"/>
  <!-- Block 4: Right - angular cuts on left side -->
  <polygon points="148,0 188,0 188,50 148,50 148,36 156,28 148,20 156,12 148,0"/>
</svg>
'''

import base64


def get_icon_base64(svg_content: str) -> str:
    """
    Encode SVG markup as Base64 for inline HTML embedding.

    Streamlit's st.markdown() with unsafe_allow_html requires images
    to be either external URLs or data URIs. This function converts
    raw SVG strings to data URI format for zero-latency icon display.

    Args:
        svg_content: Raw SVG XML string containing the icon markup.

    Returns:
        str: Base64-encoded string suitable for use in:
             <img src="data:image/svg+xml;base64,{returned_value}"/>

    Example:
        >>> b64 = get_icon_base64('<svg>...</svg>')
        >>> html = f'<img src="data:image/svg+xml;base64,{b64}"/>'
    """
    return base64.b64encode(svg_content.encode()).decode()


# =============================================================================
# CUSTOM STYLING
# =============================================================================
# Inject CSS to override Streamlit's default styling
# Provides visual polish: gradient backgrounds, rounded corners, hover effects
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 10px;
    }
    
    .icon-container {
        width: 60px;
        height: 25px;
    }
    
    /* Result boxes — visual feedback for threat classification */
    .threat-box {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        border-radius: 12px;
        padding: 25px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.4);
    }
    
    .safe-box {
        background: linear-gradient(135deg, #00c853 0%, #009624 100%);
        border-radius: 12px;
        padding: 25px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.4);
    }
    
    .confidence-score {
        font-size: 48px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .result-label {
        font-size: 24px;
        font-weight: 600;
        letter-spacing: 2px;
    }
    
    /* Sidebar styling */
    .sidebar-icon {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    /* Input area — monospace font for CSV data readability */
    .stTextArea textarea {
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 12px;
    }
    
    /* Primary action button — gradient with hover animation */
    .stButton > button {
        width: 100%;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR — Configuration Panel
# =============================================================================
with st.sidebar:
    # Display white icon on dark sidebar background
    icon_white_b64 = get_icon_base64(ICON_WHITE)
    st.markdown(f'''
    <div class="sidebar-icon">
        <img src="data:image/svg+xml;base64,{icon_white_b64}" width="140" alt="SageWall Icon"/>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    # SageMaker endpoint name — user must paste their deployed endpoint
    endpoint_name = st.text_input(
        "SageMaker Endpoint Name",
        value="sagemaker-xgboost-YYYY-MM-DD-HH-MM-SS-XXX",
        help="The specific URL address of the active SageMaker Inference Server. This allows the app to connect to the ephemeral 'Brain' hosted on AWS."
    )
    
    # AWS region selector — must match where endpoint is deployed
    aws_region = st.selectbox(
        "AWS Region",
        options=["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Algorithm:** XGBoost
    - **Dataset:** NSL-KDD
    - **Accuracy:** 99.9%
    - **Latency:** <100ms
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Threshold")
    
    # Detection threshold — controls sensitivity vs. false positive tradeoff
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Sensitivity Control. Lowering this makes the AI more paranoid (catching more attacks but risking false alarms). Raising it makes it stricter."
    )

# =============================================================================
# MAIN UI — Header and Input Section
# =============================================================================

# Header with icon and title
icon_black_b64 = get_icon_base64(ICON_BLACK)
col1, col2 = st.columns([1, 11])
with col1:
    st.markdown(f'''
    <img src="data:image/svg+xml;base64,{icon_black_b64}" width="100" alt="SageWall"/>
    ''', unsafe_allow_html=True)
with col2:
    st.title("SageWall: AI Intrusion Detection System")

st.markdown("**Real-time Network Traffic Analysis using AWS SageMaker**")
st.markdown("---")

# Sample attack vector for immediate testing
# This represents a processed NSL-KDD record with one-hot encoded features
# The specific pattern triggers high attack confidence in the trained model
SAMPLE_ATTACK_VECTOR = """0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0"""

# Input section
st.markdown("### 📡 Packet Data Input")
st.markdown("Paste the comma-separated feature vector from your processed network log:")

packet_data = st.text_area(
    "Feature Vector (CSV format)",
    value=SAMPLE_ATTACK_VECTOR,
    height=120,
    help="The 'DNA' of a network packet. This raw string contains numerical features (like Duration, Protocol, Byte Count) extracted from network logs. The AI analyzes this mathematical pattern to detect anomalies."
)

# Centered action button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    scan_button = st.button("🔍 Scan Packet", use_container_width=True)

# =============================================================================
# EDUCATIONAL SECTION — How It Works
# =============================================================================
with st.expander("ℹ️ How SageWall Works & Why It Matters"):
    st.markdown("""
    ### 🔴 The Problem: Signature-Based Detection is Failing
    
    Traditional firewalls and antivirus systems rely on **"Signatures"** — predefined rules that match 
    known attack patterns. While effective against documented threats, they are **completely blind** 
    to new, unknown attacks called **Zero-Day Exploits**.
    
    > *"You can't block what you've never seen before."*
    
    ---
    
    ### 🟢 The SageWall Solution: Behavioral Machine Learning
    
    SageWall takes a fundamentally different approach using **XGBoost**, a gradient-boosted decision 
    tree algorithm. Instead of memorizing attack signatures, it learns the *statistical behavior* 
    of normal network traffic.
    
    **How it works:**
    1. **Training Phase:** The model analyzes 125,000+ labeled network packets from the NSL-KDD dataset
    2. **Pattern Learning:** It identifies mathematical relationships between features (duration, bytes, protocols)
    3. **Anomaly Detection:** New packets that deviate from "normal" patterns are flagged as potential threats
    
    This allows SageWall to detect **attacks it has never seen before** — based purely on mathematical anomalies.
    
    ---
    
    ### 🏗️ The Architecture: Fully Serverless on AWS
    
    ```
    📁 Raw Logs (S3)  →  ⚡ ETL Pipeline (Lambda)  →  🧠 ML Model (SageMaker)  →  📊 This App
    ```
    
    | Component | Service | Purpose |
    |-----------|---------|---------|
    | **Storage** | S3 | Raw network log ingestion |
    | **ETL** | Lambda | Preprocessing & feature engineering |
    | **Training** | SageMaker | XGBoost model training |
    | **Inference** | SageMaker Endpoint | Real-time predictions (<100ms) |
    | **Frontend** | Streamlit | Interactive demo interface |
    
    This **decoupled architecture** separates the heavy mathematical computation (SageMaker) from 
    the lightweight user interface, enabling infinite scalability and cost optimization.
    
    ---
    
    ### 📈 Key Metrics
    
    | Metric | Value |
    |--------|-------|
    | Model Accuracy | **99.9%** |
    | Inference Latency | **<100ms** |
    | Training Dataset | **125,973 packets** |
    | Attack Types | DoS, Probe, R2L, U2R |
    """)

# =============================================================================
# EDUCATIONAL SECTION — How to Use (Beginner Guide)
# =============================================================================
with st.expander("🚀 How to Use This Website (Step-by-Step Guide)"):
    st.markdown("""
    ### 👋 Welcome! Here's how to get SageWall running in 5 minutes.
    
    This guide assumes you're **completely new to AWS**. Don't worry — we'll walk through everything!
    
    ---
    
    ## Step 1: Set Up AWS Credentials (One-Time Setup)
    
    Before SageWall can talk to AWS, your computer needs permission. Here's how:
    
    **Option A: AWS CLI (Recommended)**
    ```bash
    # 1. Install AWS CLI
    # Mac: brew install awscli
    # Windows: Download from aws.amazon.com/cli
    
    # 2. Configure your credentials
    aws configure
    ```
    
    When prompted, enter:
    - **AWS Access Key ID**: Get this from AWS Console → IAM → Users → Security Credentials
    - **AWS Secret Access Key**: Same location (save it — you won't see it again!)
    - **Default region**: `us-east-1` (or your preferred region)
    - **Output format**: `json`
    
    **Option B: Environment Variables**
    ```bash
    export AWS_ACCESS_KEY_ID="your-access-key"
    export AWS_SECRET_ACCESS_KEY="your-secret-key"
    export AWS_DEFAULT_REGION="us-east-1"
    ```
    
    ---
    
    ## Step 2: Deploy the SageMaker Endpoint (The AI Brain)
    
    The "Endpoint" is the live AI model running on AWS servers. Here's how to deploy it:
    
    1. **Open the Training Notebook**
       - File: `SageWall_Training.ipynb`
       - Run in: AWS SageMaker Studio, Jupyter, or VS Code
    
    2. **Run Cells 0 and 1**
       - Cell 0: Trains the XGBoost model (~3-5 minutes)
       - Cell 1: Deploys the endpoint (~5-8 minutes)
    
    3. **Copy the Endpoint Name**
       - Look for output like: `Endpoint name: sagemaker-xgboost-2025-12-21-17-42-13-588`
       - Copy this entire string!
    
    ⚠️ **Important:** The endpoint costs ~$0.05/hour while running. Remember to delete it when done!
    
    ---
    
    ## Step 3: Connect This App to Your Endpoint
    
    1. **Paste the Endpoint Name**
       - Look at the sidebar on the left → "SageMaker Endpoint Name"
       - Replace the placeholder with your copied endpoint name
    
    2. **Verify the Region**
       - Make sure "AWS Region" matches where you deployed (usually `us-east-1`)
    
    ---
    
    ## Step 4: Scan Your First Packet! 🎉
    
    1. The input box already has a **sample attack vector** (a malicious packet pattern)
    2. Click **"🔍 Scan Packet"**
    3. Watch the AI analyze it in real-time!
    
    **Expected Result:** You should see a red **"🚨 THREAT DETECTED"** box with ~99% confidence.
    
    ---
    
    ## Step 5: Try Different Packets
    
    Want to test normal traffic? Replace the input with:
    ```
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ```
    This should show **"✅ TRAFFIC NORMAL"**.
    
    ---
    
    ## Step 6: Clean Up (Save Money!)
    
    When you're done testing, **delete the endpoint** to stop billing:
    
    **Option A: Run Cell 3 in the notebook**
    ```python
    xgb_predictor.delete_endpoint()
    ```
    
    **Option B: AWS Console**
    1. Go to: AWS Console → SageMaker → Inference → Endpoints
    2. Select your endpoint → Actions → Delete
    
    ---
    
    ## 🆘 Troubleshooting
    
    | Error | Solution |
    |-------|----------|
    | "Endpoint not found" | Double-check the endpoint name, or redeploy via the notebook |
    | "Access Denied" | Run `aws configure` and verify your IAM permissions |
    | "Connection timeout" | Check your region setting matches where endpoint is deployed |
    | "Model error" | Ensure input is comma-separated numbers (no text or headers) |
    
    ---
    
    ## 💡 Pro Tips
    
    - **Adjust Sensitivity:** Use the threshold slider to tune detection aggressiveness
    - **Batch Testing:** For production, use the SageMaker API directly via boto3
    - **Cost Savings:** Use `ml.t2.medium` for testing, scale up for production
    """)

st.markdown("---")

# =============================================================================
# INFERENCE LOGIC — SageMaker Integration
# =============================================================================


def invoke_sagemaker_endpoint(endpoint_name: str, payload: str, region: str) -> float:
    """
    Send feature vector to SageMaker and retrieve threat prediction.

    This function establishes a connection to the SageMaker Runtime API
    and invokes the deployed XGBoost model endpoint for real-time inference.

    Args:
        endpoint_name: The unique identifier of the deployed SageMaker endpoint.
                       Format: 'sagemaker-xgboost-YYYY-MM-DD-HH-MM-SS-XXX'
        payload: Comma-separated feature values as a string.
                 Must match the schema expected by the trained model
                 (121 features after one-hot encoding).
        region: AWS region where the endpoint is deployed (e.g., 'us-east-1').

    Returns:
        float: Prediction score between 0.0 and 1.0.
               - Values close to 0.0 indicate normal traffic
               - Values close to 1.0 indicate attack traffic

    Raises:
        botocore.exceptions.ClientError: If endpoint doesn't exist or
                                         AWS credentials are invalid.
        ValueError: If the model returns non-numeric output.

    Note:
        boto3 is imported inside this function to avoid startup permission
        issues on macOS when the module loads before credentials are available.
    """
    # Deferred import: Prevents macOS sandbox permission errors at module load time
    # boto3 attempts to read SSL certificates on import, which can fail in restricted environments
    import boto3

    # Initialize SageMaker Runtime client for the specified region
    # SageMaker Runtime is specifically for inference (not training/management)
    runtime = boto3.client('sagemaker-runtime', region_name=region)

    # Invoke the endpoint with CSV payload
    # ContentType tells SageMaker how to deserialize the input
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=payload.strip()
    )

    # Parse response — XGBoost returns prediction as plain text float
    result = response['Body'].read().decode('utf-8')
    return float(result.strip())


# =============================================================================
# EVENT HANDLER — Scan Button Click
# =============================================================================
if scan_button:
    # Input validation — prevent API calls with placeholder or empty data
    if endpoint_name == "sagemaker-xgboost-YYYY-MM-DD-HH-MM-SS-XXX":
        st.warning("⚠️ Please enter your actual SageMaker endpoint name in the sidebar.")
    elif not packet_data.strip():
        st.warning("⚠️ Please enter packet data to analyze.")
    else:
        try:
            # Show loading spinner during API call (typically <100ms)
            with st.spinner("🔄 Analyzing packet with SageMaker..."):
                score = invoke_sagemaker_endpoint(
                    endpoint_name=endpoint_name,
                    payload=packet_data,
                    region=aws_region
                )

            # Display results based on threshold comparison
            st.markdown("### 📋 Analysis Results")

            if score > threshold:
                # THREAT DETECTED — Red gradient card with high confidence display
                st.markdown(f"""
                <div class="threat-box">
                    <div class="result-label">🚨 THREAT DETECTED</div>
                    <div class="confidence-score">{score:.1%}</div>
                    <div>Confidence Score</div>
                    <br/>
                    <div style="font-size: 14px; opacity: 0.9;">
                        Malicious network activity identified. Recommend immediate investigation.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.error(f"**Alert:** High-confidence threat detected with score {score:.4f}")

            else:
                # TRAFFIC NORMAL — Green gradient card with safety confidence
                st.markdown(f"""
                <div class="safe-box">
                    <div class="result-label">✅ TRAFFIC NORMAL</div>
                    <div class="confidence-score">{1-score:.1%}</div>
                    <div>Safety Confidence</div>
                    <br/>
                    <div style="font-size: 14px; opacity: 0.9;">
                        No malicious patterns detected. Traffic appears legitimate.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.success(f"**Status:** Normal traffic (threat score: {score:.4f})")

            # Collapsible raw response for debugging
            with st.expander("🔬 Raw Model Response"):
                st.code(f"Prediction Score: {score}", language="text")

        except Exception as e:
            # Graceful error handling with helpful diagnostics
            error_msg = str(e)
            error_type = type(e).__name__

            # Detect AWS-specific errors for targeted help messages
            if 'endpoint' in error_msg.lower() or 'sagemaker' in error_msg.lower() or 'aws' in error_msg.lower() or 'botocore' in error_type.lower():
                st.error(f"""
                **❌ AWS Error**
                
                Could not connect to SageMaker endpoint.
                
                **Details:** {error_type}: {error_msg}
                
                **Possible causes:**
                - Endpoint name is incorrect
                - Endpoint is not deployed/active
                - AWS credentials not configured
                - Insufficient IAM permissions
                """)
            else:
                st.error(f"""
                **❌ Error**
                
                An unexpected error occurred.
                
                **Details:** {error_type}: {error_msg}
                """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    <strong>SageWall</strong> | AI-Powered Intrusion Detection System<br/>
    Built with AWS SageMaker & XGBoost | © 2025
</div>
""", unsafe_allow_html=True)
