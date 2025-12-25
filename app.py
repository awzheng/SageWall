"""
SageWall Streamlit Frontend.

-------------------------------------------------------------------------------
1.0: WHAT IS THIS FILE?
-------------------------------------------------------------------------------
This is the "Brain" and "Face" of the SageWall web application.
- It defines the UI (User Interface) you see in the browser.
- It handles the logic for sending network packet data to the AWS cloud.
- It interprets the AI's response (Threat vs. Safe) and displays it.

-------------------------------------------------------------------------------
2.0: HOW IS IT STRUCTURED?
-------------------------------------------------------------------------------
The code is organized into clear sections:
1. IMPORTS & SETUP: Loading necessary tools (libraries).
2. CONFIGURATION: Setting page title, layout, and global options.
3. THEME & STYLING: Defining colors, fonts, and dark mode.
4. HELPER FUNCTIONS: Small tools for handling icons and API calls.
5. MAIN UI: Building the visible parts (Header, Inputs, Buttons).
6. LOGIC: What happens when you click "Scan".
7. EDUCATION: The help sections at the bottom.

-------------------------------------------------------------------------------
3.0: KEY CONCEPTS FOR BEGINNERS
-------------------------------------------------------------------------------
- Streamlit (st): A Python library that turns scripts into web apps instantly.
- Boto3: The official AWS SDK for Python. It allows Python to talk to AWS.
- SageMaker: AWS's machine learning platform where our "Brain" runs.
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
# We import 'streamlit' as 'st'. This is standard convention.
# Everywhere you see 'st.something()', we are creating a UI element.
import streamlit as st
import base64  # Used to encode image data (icons) so they can live directly in this file\
from utils import alerts

# =============================================================================
# SECTION 2: HELPER FUNCTIONS & ASSETS
# =============================================================================

# Define SVG Icon as a multiline string
# This draws the geometric "Wall" logo
ICON_JADE = '''
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 50" fill="#4ECDC4">
  <polygon points="0,0 40,0 40,12 32,20 40,28 32,36 40,50 0,50"/>
  <polygon points="52,0 88,0 88,18 82,25 88,32 88,50 52,50 52,32 58,25 52,18"/>
  <polygon points="100,0 136,0 136,18 130,25 136,32 136,50 100,50 100,32 106,25 100,18"/>
  <polygon points="148,0 188,0 188,50 148,50 148,36 156,28 148,20 156,12 148,0"/>
</svg>
'''

def get_icon_base64(svg_content: str) -> str:
    """
    Helper function to prepare SVGs for display.
    WHY? Browsers can't display raw SVG strings in <img> tags easily.
    We convert it to 'base64' which acts like a filename but contains the whole image.
    """
    return base64.b64encode(svg_content.encode()).decode()

# Convert our icon
icon_b64 = get_icon_base64(ICON_JADE)

# =============================================================================
# SECTION 3: PAGE CONFIGURATION
# =============================================================================
# This must be the very first Streamlit command. It sets the browser tab title and favicon.
st.set_page_config(
    page_title="SageWall IDS",  # Title shown in browser tab
    page_icon= ICON_JADE,             # Favicon shown in browser tab
    layout="centered",          # Centers the content in the middle of the screen
    initial_sidebar_state="collapsed" # Hides the side menu by default
)

# -----------------------------------------------------------------------------
# QUICK FIX: Hide the default "Deploy" button
# -----------------------------------------------------------------------------
# Streamlit adds a "Deploy" button by default. We use custom HTML/CSS to hide it
# because we want a clean look for our users.
st.markdown(
    """
    <style>
    .stAppDeployButton {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True # Required to allow injecting raw CSS/HTML
)

# =============================================================================
# SECTION 4: THEME & COLOR PALETTE
# =============================================================================
# Here we define our "Jade & Cyan" color scheme.
# Changing these hex codes will update the app's look globally.

COLORS = {
    'light': {
        'bg': '#FFFFFF',
        'bg_secondary': "#EFEFEF",
        'text': '#2D3436',
        'text_secondary': '#636E72',
        'border': "#2D3132",
        'card_bg': '#F0F9F8',
    },
    'dark': {
        'bg': "#0e1117",  # Deep charcoal for dark mode background
        'bg_secondary': "#262730",
        'text': '#FAFAFA', # White text
        'text_secondary': '#D3D3D3',
        'border': '#4A5568',
        'card_bg': "#1C1D1D",
    },
    'accent': {
        'jade': '#4ECDC4',     # Primary brand color
        'jade_light': '#70C1B3',
        'cyan': '#5BC0EB',
        'gradient_start': '#70C1B3',
        'gradient_end': '#5BC0EB',
    }
}

# -----------------------------------------------------------------------------
# TOGGLE SWITCH
# -----------------------------------------------------------------------------
# We use columns to position the toggle on the right side.
# [8, 2] means the first column takes 80% width (spacer), second takes 20% (toggle).
col_spacer, col_toggle = st.columns([8, 2])
with col_toggle:
    # st.toggle returns True if ON, False if OFF.
    dark_mode = st.toggle("Dark Mode", value=False)

# Select the active theme based on the toggle position
theme = 'dark' if dark_mode else 'light'
c = COLORS[theme]
accent = COLORS['accent']

# =============================================================================
# SECTION 5: CSS STYLING
# =============================================================================
# Streamlit allows minimal styling by default. To make our app look professional
# and custom (e.g., custom fonts, gradients), we inject raw CSS code.

st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        font-family: sans-serif !important;
    }}
    
    /* 2. BACKGROUND: Set global background color based on theme selection */
    .stApp {{
        background-color: {c['bg']};
        color: {c['text']};
    }}
    
    /* 3. HEADERS: Force headers to match text color */
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp span, .stApp label {{
        color: {c['text']};
    }}
    
    /* 4. TITLE GRADIENT: Special CSS to make the title text multi-colored */
    .header-title {{
        margin: 0;
        font-size: 32px; /* Smaller title */
        font-weight: 700;
        /* Why we use 'background-clip': It clips the background color TO the text shape */
        background: {accent['jade']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .header-subtitle {{
        color: {c['text_secondary']};
        font-size: 16px; /* Smaller subtitle */
    }}
    
    .header-container {{
        text-align: center;
        margin-bottom: 30px;
    }}

    /* 5. RESULT CARDS: Styling for the result boxes */
    
    /* RED BOX (THREAT) */
    .threat-box {{
        background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
        border-radius: 12px;
        padding: 25px;
        color: white !important;
        text-align: center;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
    }}
    
    /* GREEN BOX (SAFE) */
    .safe-box {{
        background: linear-gradient(135deg, {accent['jade']} 0%, {accent['jade_light']} 100%);
        border-radius: 12px;
        padding: 25px;
        color: white !important; /* Force white text even in light mode */
        text-align: center;
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);
    }}
    
    /* 6. INPUT ELEMENTS: Making text boxes and sliders look native */
    .stTextArea textarea {{
        font-family: 'Monaco', 'Menlo', monospace; /* Monospace for data */
        font-size: 10px;
        background-color: {c['bg_secondary']};
        color: {c['text']};
        border-color: {c['border']};
    }}
    
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
        background-color: {c['bg_secondary']};
        color: {c['text']};
        border-color: {c['border']};
    }}

    /* Fix for Buttons in Dark Mode */
    .stButton > button {{
        background-color: {c['bg_secondary']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
    }}
    .stButton > button:hover {{
        border-color: {accent['jade']} !important;
        color: {accent['jade']} !important;
    }}

    /* Fix for Expander Headers (Info Buttons) in Dark Mode */
    /* Fix for Expander Headers (Info Buttons) in Dark Mode */
    [data-testid="stExpander"] {{
        background-color: {c['bg_secondary']} !important;
        border: 1px solid {c['border']} !important;
        color: {c['text']} !important;
    }}
    
    /* Target the summary (header) specifically to prevent white background on open */
    [data-testid="stExpander"] summary {{
        background-color: {c['bg_secondary']} !important;
        color: {c['text']} !important;
    }}
    
    [data-testid="stExpander"] summary:hover {{
        color: {accent['jade']} !important;
    }}
    
    [data-testid="stExpander"] summary svg {{
        fill: {c['text']} !important;
    }}
    
    /* Ensure content inside uses correct background */
    [data-testid="stExpander"] > div {{
        background-color: {c['bg_secondary']} !important;
        color: {c['text']} !important;
    }}
    
    /* 7. CLEANUP: Hiding Streamlit clutter */
    [data-testid="stSidebar"] {{ display: none; }} /* Hide empty sidebar */
    hr {{ border-color: {c['border']}; }} /* Theme-aware divider lines */
    
    /* Fix table visibility issues in different themes */
    table, th, td {{
        color: {c['text']} !important;
        border-color: {c['border']} !important;
        background-color: transparent !important;
    }}
    th {{ background-color: {c['bg_secondary']} !important; }}
    
</style>
""", unsafe_allow_html=True)



# =============================================================================
# SECTION 6: BUILDING THE UI (User Interface)
# =============================================================================

# 6.1 HEADER
# We use HTML string + unsafe_allow_html to create a layout that standard
# Streamlit markdown cannot do (centering an image + text tightly).
st.markdown(f'''
<div class="header-container">
    <img src="data:image/svg+xml;base64,{icon_b64}" width="120" alt="SageWall Icon" style="margin-bottom: 10px;"/>
    <h1 class="header-title">SageWall</h1>
    <p class="header-subtitle">Cloud/ML Intrusion Detection System</p>
</div>
''', unsafe_allow_html=True)

# 6.2 CONFIGURATION INPUTS
st.markdown("### Configuration")

# Create two columns: Left (Endpoint - wider), Right (Region - narrower)
col1, col2 = st.columns([2, 1])

with col1:
    endpoint_name = st.text_input(
        "SageMaker Endpoint Name",
        value="sagemaker-xgboost-YYYY-MM-DD-HH-MM-SS-XXX",
        help="Copy this from your AWS Notebook. It's the ID of the running AI."
    )

with col2:
    aws_region = st.selectbox(
        "AWS Region",
        options=["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"],
        # Default index 0 is 'us-east-1'
        help="The physical location where your AWS server is running."
    )

# 6.3 THRESHOLD SLIDER
# This controls how strict the AI is.
threshold = st.slider(
    "Detection Threshold",
    min_value=0.1, max_value=0.9, value=0.5, step=0.1,
    help="Low (0.1) = Paranoid (Flags everything). High (0.9) = Relaxed (Flags only obvious threats)."
)

# 6.4 DATA INPUT
# A sample malicious packet to help users test immediately.
# This is a CSV string of feature values.
SAMPLE_ATTACK_VECTOR = """0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0"""

st.markdown("### Packet Analysis")
packet_data = st.text_area(
    "Feature Vector (CSV format)",
    value=SAMPLE_ATTACK_VECTOR,
    height=100,
    help="Paste your network log data here."
)

# 6.5 ACTION BUTTON
# 'use_container_width=True' makes the button stretch to full width
scan_button = st.button("🔍 Scan Packet", use_container_width=True, key="scan_packet_btn")


# =============================================================================
# SECTION 7: CORE LOGIC & AWS COMMUNICATION
# =============================================================================

def invoke_sagemaker_endpoint(endpoint_name: str, payload: str, region: str) -> float:
    """
    The function that talks to AWS.
    
    Workflow:
    1. Import boto3 (AWS tool).
    2. Connect to SageMaker in the specific region.
    3. Send the 'payload' (packet data).
    4. Receive the score (0.0 to 1.0).
    """
    # IMPORT NOTE: We import 'boto3' inside the function instead of at the top.
    # Why? On some Mac setups, importing it at the top causes crash loops due to
    # SSL certificate loading issues before the app fully starts. This is a safety fix.
    import boto3

    # Create the client connection
    runtime = boto3.client('sagemaker-runtime', region_name=region)

    # Send the data!
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',  # Telling AWS we are sending CSV text
        Body=payload.strip()     # .strip() removes accidental whitespace
    )

    # Decode the response
    # The response comes back as bytes, we decode to string, then convert to float number.
    result = response['Body'].read().decode('utf-8')
    return float(result.strip())


# =============================================================================
# SECTION 8: EXECUTION (When Button is Clicked)
# =============================================================================

if scan_button:
    # 8.1 VALIDATION (Check if input is valid before calling AWS)
    if endpoint_name == "sagemaker-xgboost-YYYY-MM-DD-HH-MM-SS-XXX":
        st.warning("⚠️ Please set valid endpoint name.")
    elif not packet_data.strip():
        st.warning("⚠️ The packet data is empty.")
    else:
        try:
            # 8.2 API CALL WITH SPINNER
            with st.spinner("🔄 Analyzing packet pattern..."):
                score = invoke_sagemaker_endpoint(
                    endpoint_name=endpoint_name,
                    payload=packet_data,
                    region=aws_region
                )

            # 8.3 DISPLAY RESULTS
            st.markdown("### Analysis Results")

            # Logic: If score is higher than threshold -> It's a THREAT.
            if score > threshold:
                # THREAT DETECTED
                st.markdown(f"""
                <div class="threat-box">
                    <div class="result-label">🚨 THREAT DETECTED</div>
                    <div class="confidence-score">{score:.1%}</div> <!-- Formats 0.992 as 99.2% -->
                    <div>Confidence Score</div>
                    <br/>
                    <div style="font-size: 14px; opacity: 0.9;">
                        Malicious network activity identified.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.error(f"High-confidence threat detected (Score: {score:.4f})")

            else:
                # SAFE TRAFFIC
                # Confidence is the inverse (1.0 - threat_score)
                # If threat is 0.05 (5%), Safe confidence is 0.95 (95%).
                st.markdown(f"""
                <div class="safe-box">
                    <div class="result-label">✅ TRAFFIC NORMAL</div>
                    <div class="confidence-score">{1-score:.1%}</div>
                    <div>Safety Confidence</div>
                    <br/>
                    <div style="font-size: 14px; opacity: 0.9;">
                        Traffic appears legitimate.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.success(f"Status: Normal (Threat Score: {score:.4f})")

            # Debug View
            with st.expander("🔬 View Raw Model Response"):
                st.code(f"Prediction Score: {score}", language="text")

        except Exception as e:
            # 8.4 ERROR HANDLING
            # If something breaks (wrong endpoint name, no internet), we catch it here.
            error_msg = str(e)
            if 'endpoint' in error_msg.lower():
                st.error("❌ Error: Endpoint not found. Ensure that the endpoint name is correct.")
            else:
                st.error(f"❌ Error: {error_msg}")

# =============================================================================
# SECTION 9: FOOTER & EDUCATION
# =============================================================================
st.markdown("---")

# Stats Cards
cols = st.columns(4)
stats = [
    ("Algorithm", "XGBoost"),
    ("Accuracy", "99.9%"),
    ("Latency", "<100ms"),
    ("Samples", "125K+")
]

# Loop through stats to create cards (Efficient coding!)
for i, (val, label) in enumerate(stats):
    with cols[i]:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Educational Expanders
with st.expander("ℹ️ How SageWall Works"):
    with open("docs/how_sagewall_works.md", "r") as f:
        st.markdown(f.read())

with st.expander("📚 Beginner's Guide"):
    with open("docs/beginners_guide.md", "r") as f:
        st.markdown(f.read())

# Footer
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px; margin-top: 50px;">
    <strong>SageWall</strong> | AI-Powered Intrusion Detection System | © 2025
</div>
""", unsafe_allow_html=True)
