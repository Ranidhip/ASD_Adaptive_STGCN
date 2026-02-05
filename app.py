import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# --- MUST BE THE FIRST COMMAND ---
st.set_page_config(
    page_title="ASD Adaptive ST-GCN Interface",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- IMPORT MODEL AFTER PAGE CONFIG ---
from model import AdaptiveSTGCN

# --- STYLE & CSS ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .st-emotion-cache-1v0mbdj { box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 20px; border-radius: 10px; background-color: white; }
    .metric-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }
    .metric-label { font-size: 1.1rem; font-weight: 600; color: #555; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #333; }
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-mid { color: #ffc107; font-weight: bold; }
    .risk-high { color: #dc3545; font-weight: bold; }
    .report-header { text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "mmasd_v1_frozen.npz"
MODEL_PATH = "trained_models/best_model.pth"
SKELETON_IMAGE_PATH = "skeleton_diagram.png" # Make sure you have this image

JOINTS = [
    "Base", "Spine", "Neck", "Head", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand", "L_Hip", "L_Knee", "L_Ankle",
    "L_Foot", "R_Hip", "R_Knee", "R_Ankle", "R_Foot", "Spine_Chest", "Spine_Mid",
    "L_HandTip", "L_Thumb", "R_HandTip"
]

# --- LOADERS ---
@st.cache_resource
def load_model():
    model = AdaptiveSTGCN().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.warning(f"⚠️ Model file '{MODEL_PATH}' not found. Using random weights.")
    model.eval()
    return model

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None, None
    data = np.load(DATA_PATH)
    return data['X_test'], data['Y_test']

# --- MAIN APP ---
# Load Resources
model = load_model()
X_test, Y_test = load_data()

if X_test is None:
    st.error(f"❌ Data file '{DATA_PATH}' not found. Please add it to the project folder.")
    st.stop()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    st.subheader("📁 Data Source")
    # In a real app, this would process the uploaded file
    uploaded_file = st.file_uploader("Upload a Skeleton .CSV file", type=["csv"])
    if uploaded_file is not None:
        st.info("File uploaded! (Using demo data for now as processing logic is not attached).")

    st.subheader("👤 Patient Selection (Demo Data)")
    sample_id = st.number_input("Select Patient ID from Test Set", min_value=0, max_value=len(X_test)-1, value=0, help="Choose a patient from the pre-loaded test dataset.")
    
    st.markdown("---")
    st.subheader("ℹ️ System Info")
    st.caption(f"**Model:** Adaptive ST-GCN")
    st.caption(f"**Device:** {str(DEVICE).upper()}")
    st.caption(f"**Status:** Ready for Inference")
    
    st.markdown("---")
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.session_state['run_analysis'] = True

# --- INFERENCE LOGIC ---
input_tensor = torch.tensor(X_test[sample_id]).unsqueeze(0).float().to(DEVICE)
input_tensor.requires_grad = True

outputs = model(input_tensor)
probs = F.softmax(outputs, dim=1)
asd_score = probs[0, 1].item()
pred_class = torch.argmax(probs, dim=1).item()
true_label = Y_test[sample_id]

if asd_score < 0.4:
    severity, color_class = "Low Risk / Typical", "risk-low"
    color_hex = "#28a745"
elif 0.4 <= asd_score < 0.7:
    severity, color_class = "Moderate Risk", "risk-mid"
    color_hex = "#ffc107"
else:
    severity, color_class = "High Risk (ASD)", "risk-high"
    color_hex = "#dc3545"

# --- MAIN CONTENT AREA ---
st.title("🧠 Adaptive ST-GCN for ASD Screening")
st.markdown("### Intelligent Movement Analysis & Risk Assessment Dashboard")

# Create Tabs for better organization
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔎 Explainability (XAI)", "📄 Clinical Report"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.markdown("#### Patient Overview & Analysis Results")
    
    # Top Row of Metrics (Card Style)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-label">Patient ID</div><div class="metric-value">#{}</div></div>'.format(sample_id), unsafe_allow_html=True)
    with col2:
        pred_text = "ASD" if pred_class == 1 else "Typical"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Model Prediction</div><div class="metric-value">{pred_text}</div></div>', unsafe_allow_html=True)
    with col3:
        true_text = "ASD" if true_label == 1 else "Typical"
        st.markdown(f'<div class="metric-card"><div class="metric-label">True Label</div><div class="metric-value">{true_text}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence Score</div><div class="metric-value">{asd_score:.1%}</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")

    # Main Result Section
    result_col1, result_col2 = st.columns([2, 1])
    with result_col1:
        st.subheader("Assessment Outcome")
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; border-left: 8px solid {color_hex}; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h2 style="color: {color_hex}; margin-top: 0;">{severity}</h2>
                <p style="font-size: 1.1rem;">The model has analyzed the skeletal movement patterns and determined a risk probability of <b>{asd_score:.2%}</b>.</p>
            </div>
            """, unsafe_allow_html=True)
        st.write("")
        st.progress(asd_score)
        st.caption("Probability Scale: 0% (Typical) to 100% (High Risk ASD)")

    with result_col2:
        st.subheader("Patient Details (Placeholder)")
        st.info("Create a 'metadata.csv' file to link IDs to real patient info.")
        with st.container(border=True):
            st.text_input("Name", value="John Doe", disabled=True)
            st.text_input("Age", value="7 years", disabled=True)
            st.text_input("Gender", value="Male", disabled=True)


# --- TAB 2: EXPLAINABILITY ---
with tab2:
    st.markdown("#### Model Decision Explanation (Gradient-Based Saliency)")
    st.write("This analysis highlights which body joints contributed most significantly to the model's prediction. Joints with higher bars and red color were more influential.")
    
    # Calculate Gradients
    score = outputs[0, 1]
    score.backward()
    grads = input_tensor.grad.cpu().numpy()[0]
    importance = np.sum(np.abs(grads), axis=(0, 1))
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-9)
    
    xai_col1, xai_col2 = st.columns([3, 2])
    
    with xai_col1:
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = ['#dc3545' if x > 0.4 else '#6c757d' for x in importance]
        sns.barplot(x=np.arange(25), y=importance, palette=bar_colors, ax=ax)
        ax.set_xticks(np.arange(25))
        ax.set_xticklabels(JOINTS, rotation=90, fontsize=10)
        ax.set_ylabel("Relative Importance Score", fontsize=12)
        ax.set_title(f"Joint Importance Profile for Patient #{sample_id}", fontsize=14, fontweight='bold')
        sns.despine()
        st.pyplot(fig)

    with xai_col2:
        st.write("##### Key Influential Joints")
        # Find top 3 most important joints
        top_indices = np.argsort(importance)[-3:][::-1]
        for i, idx in enumerate(top_indices):
            st.markdown(f"**{i+1}. {JOINTS[idx]}** (Score: {importance[idx]:.2f})")

        st.write("")
        st.write("##### Reference Skeleton")
        if os.path.exists(SKELETON_IMAGE_PATH):
             st.image(SKELETON_IMAGE_PATH, caption="Skeletal Joint Map", use_column_width=True)
        else:
             st.warning("⚠️ 'skeleton_diagram.png' not found. Please add it to the project folder for reference.")

# --- TAB 3: CLINICAL REPORT ---
with tab3:
    st.markdown('<div class="report-header"><h1>ASD Screening Clinical Report</h1></div>', unsafe_allow_html=True)
    st.markdown("---")
    
    rep_col1, rep_col2 = st.columns(2)
    with rep_col1:
        st.write("**Patient ID:**", f"#{sample_id}")
        st.write("**Date of Analysis:**", "October 26, 2023")
        st.write("**Referring Physician:**", "Dr. S. Silva")
    with rep_col2:
        st.write("**Analysis Method:**", "Adaptive ST-GCN (Deep Learning)")
        st.write("**Model Version:**", "v1.0 (Research Prototype)")
    
    st.markdown("### Assessment Summary")
    st.markdown(f"""
    Based on the analysis of the provided skeletal motion data, the system has classified the movement patterns with the following results:
    
    * **Primary Classification:** **{pred_text}**
    * **Risk Severity Assessment:** <span class="{color_class}">{severity}</span>
    * **Model Confidence:** **{asd_score:.2%}**
    
    This automated screening result suggests that the patient's motor patterns are most consistent with the **{severity}** group. This report is generated by an AI system and should be used as a supporting tool for clinical diagnosis by a qualified professional.
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.write("© 2023 University Project. For research use only.")
    
    st.write("")
    col_print, _ = st.columns([1, 4])
    with col_print:
        st.button("🖨️ Print Report", use_container_width=True, help="This would open a print dialog in a real application.")