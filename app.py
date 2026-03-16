# =====================================================
# IMPORTS
# =====================================================
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
import tempfile
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from fpdf import FPDF
import os
import hashlib
import qrcode
from io import BytesIO
import urllib.parse
import sqlite3
from cryptography.fernet import Fernet
import time 
from utilis.database import get_training_patients, get_test_patients, save_test_patient
from utilis.pdf_generator import generate_patient_pdf
from utilis.database import create_tables
from datetime import datetime
import urllib.parse
import streamlit as st
import tempfile
import base64
from utilis.database import (
    create_tables,
    register_user,
    login_user,
    reset_user_password
)
from utilis.database import save_test_patient
create_tables()

# Splash Screen
splash = st.empty()

splash.markdown("""
# 🏥 BreastCare AI
### Oncology Intelligence System
Loading...
""")

time.sleep(2)

splash.empty()

# =============================
# ENCRYPTION SETUP
# =============================

key = Fernet.generate_key()
cipher = Fernet(key)

# =============================
# DATABASE SETUP
# =============================

conn = sqlite3.connect("patients.db", check_same_thread=False)
cursor = conn.cursor()
# ====================================
# AUTHENTICATION FUNCTIONS
# ====================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_doctor(name, email, phone, password):
    try:
        hashed = hash_password(password)

        cursor.execute("""
        INSERT INTO doctors (name, email, phone, password)
        VALUES (?, ?, ?, ?)
        """, (name, email, phone, hashed))

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False


def login_doctor(identifier, password):

    hashed = hash_password(password)

    cursor.execute("""
    SELECT * FROM doctors 
    WHERE (email=? OR phone=?) AND password=?
    """, (identifier, identifier, hashed))

    doctor = cursor.fetchone()

    return doctor


def reset_password(identifier, new_password):

    hashed = hash_password(new_password)

    cursor.execute("""
    UPDATE doctors
    SET password=?
    WHERE email=? OR phone=?
    """, (hashed, identifier, identifier))

    conn.commit()

cursor.execute("""
CREATE TABLE IF NOT EXISTS doctors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    phone TEXT UNIQUE,
    password TEXT
)
""")

conn.commit()

def save_patient(name, data):

    encrypted = cipher.encrypt(data.encode())

    cursor.execute(
        "INSERT INTO patients (name, encrypted_data) VALUES (?, ?)",
        (name, encrypted)
    )
    conn.commit()
# Hide Streamlit default UI
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Breast Cancer Intelligence System",
    page_icon="🏥",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
for key in ["diagnosis","probability","final_risk",
            "confidence","test_time","heatmap_path",
            "stage","aggressiveness","subtype",
            "node_risk","referral","screening",
            "biopsy","future_risk","explanation"]:
    if key not in st.session_state:
        st.session_state[key] = None
         
  #Login system with language selection
language = st.sidebar.selectbox(
    "Select Language",
    ["English", "French", "Swahili"]
)

if language == "Swahili":
    login_text = "Ingia"
    register_text = "Jisajili"
elif language == "French":
    login_text = "Connexion"
    register_text = "Créer un compte"
else:
    login_text = "Login"
    register_text = "Create Account"
# =============================
# AUTHENTICATION SYSTEM
# =============================

auth_menu = ["Login", "Register", "Forgot Password"]

auth_choice = st.sidebar.selectbox(
    "Doctor Access",
    auth_menu
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# =============================
# LOGIN
# =============================

if auth_choice == "Login":

    st.title("Doctor Login")

    email_phone = st.text_input("Email or Phone Number")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        hashed = hash_password(password)

        doctor = login_user(email_phone, hashed)

        if doctor:

            st.session_state.logged_in = True
            st.session_state.doctor_name = doctor[1]

            st.success(f"Welcome Dr. {doctor[1]}")
            st.rerun()

        else:
            st.error("Invalid credentials")

# =============================
# REGISTER DOCTOR
# =============================

elif auth_choice == "Register":

    st.title("Create Doctor Account")

    name_first = st.text_input("Full Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    password = st.text_input("Password", type="password")

    if st.button("Create Account"):

        hashed = hash_password(password)

        success = register_user(name_first,email,phone,hashed)

        if success:
            st.success("Account created successfully")
        else:
            st.error("Doctor already exists")

# =============================
# RESET PASSWORD
# =============================

elif auth_choice == "Forgot Password":

    st.title("Reset Password")

    email = st.text_input("Enter Email or Phone")
    new_password = st.text_input("New Password", type="password")

    if st.button("Reset Password"):

        hashed = hash_password(new_password)

        reset_user_password(email,hashed)

        st.success("Password reset successfully")
#Login protection
if not st.session_state.logged_in:
    st.stop()
# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/breast_model.h5", compile=False)

model = load_model()

# Initialize session variables
if "final_risk" not in st.session_state:
    st.session_state.final_risk = 0

if "stage" not in st.session_state:
    st.session_state.stage = "Unknown"

if "subtype" not in st.session_state:
    st.session_state.subtype = "Unknown"

if "agg" not in st.session_state:
    st.session_state.agg = 0
# =====================================================
# HEADER
# =====================================================
st.title("Breast Cancer Diagnostic Intelligence System")
st.caption("AI-Assisted Clinical Decision Support Prototype – Not a substitute for pathology.")
menu = ["Diagnosis", "Training Patients", "Tested Patients"]

choice = st.sidebar.selectbox(
    "Navigation Menu",
    menu
)
# =====================================================
#Dashboard menu

if choice == "Training Patients":

    st.title("Training Dataset Patients")

    df = get_training_patients()

    if df.empty:
        st.warning("No training data found")
    else:
        st.dataframe(df)

#Tested Patients history section

elif choice == "Tested Patients":

    st.title("Tested Patients History")

    df = get_test_patients()

    search = st.text_input("Search Patient")

    if search:
        df = df[df["Name"].str.contains(search)]

    if df.empty:
        st.warning("No tested patients yet")

    else:

        st.dataframe(df)

        st.metric("Total Patients Tested", len(df))

        benign = len(df[df["Prediction"] == "Benign"])
        malignant = len(df[df["Prediction"] == "Malignant"])

        st.metric("Benign Cases", benign)
        st.metric("Malignant Cases", malignant)

        import os

        for index, row in df.iterrows():

            pdf_path = f"reports/patient_reports/{row['Name']}_report.pdf"

            if os.path.exists(pdf_path):

                with open(pdf_path, "rb") as f:

                    st.download_button(
                        label=f"Download {row['Name']} Report",
                        data=f,
                        file_name=f"{row['Name']}_report.pdf",
                        mime="application/pdf"
                    )
# CLINICAL INPUTS
# PATIENT INFORMATION
# =====================================================
name = st.text_input("Patient Name")
st.header("Patient Clinical Information")

colA, colB = st.columns(2)

with colA:
    age = st.number_input("Age", 0, 120, key="age")
    bmi = st.number_input("BMI", 10.0, 60.0, key="bmi")

with colB:
    family_history = st.selectbox(
        "Family History of Cancer",
        ["No", "Yes"],
        key="family_history"
    )

    hormone_therapy = st.selectbox(
        "Hormone Therapy",
        ["No", "Yes"],
        key="hormone_therapy"
    )

    smoking = st.selectbox(
        "Smoking History",
        ["No", "Former", "Current"],
        key="smoking"
    )

    alcohol = st.selectbox(
        "Alcohol Intake",
        ["No", "Occasional", "Regular"],
        key="alcohol"
    )

exercise = st.selectbox(
    "Exercise Level",
    ["None", "Occasional", "Regular"],
    key="exercise"
)

symptoms = st.multiselect(
    "Symptoms",
    ["Lump", "Pain", "Nipple Discharge", "Skin Changes"],
    key="symptoms"
)

breast_density = st.selectbox(
    "Breast Density",
    [
        "A (Almost entirely fatty)",
        "B (Scattered fibroglandular)",
        "C (Heterogeneously dense)",
        "D (Extremely dense)"
    ],
    key="breast_density"
)

breast_feeding = st.selectbox(
    "Breastfeeding History",
    ["No", "Yes"],
    key="breast_feeding"
)

parity = st.number_input(
    "Number of Full-Term Pregnancies",
    min_value=0,
    max_value=20,
    key="parity"
)

st.divider()
# =====================================================
# BIOMARKERS
# =====================================================
st.header("🧬 Biomarkers & Hormonal Profiling")

er = st.selectbox("ER Status", ["Positive", "Negative"], key="er")
pr = st.selectbox("PR Status", ["Positive", "Negative"], key="pr")
her2 = st.selectbox("HER2 Status", ["Positive", "Negative"], key="her2")
brca1 = st.selectbox("BRCA1 Mutation", ["Positive", "Negative"], key="brca1")

ca15_3 = st.number_input("CA 15-3 (U/mL)", min_value=0.0, key="ca15_3")
lymph_nodes = st.number_input("Lymph Nodes (0-12)", min_value=0, max_value=12, key="lymph_nodes")
tumor_size = st.number_input("Tumor Size (cm)", min_value=0.0, key="tumor_size")

st.divider()

# =====================================================
# IMAGE UPLOAD
# =====================================================
uploaded_file = st.file_uploader("Upload Mammogram/ultrasound",type=["jpg","png","jpeg"])

img_pil = None
if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    st.image(img_pil, width="stretch")
    
    
    st.divider()
st.subheader("Doctor Authentication")

doctor_signature = st.file_uploader(
    "Upload Doctor Signature (PNG)",
    type=["png"]
)

# =====================================================
# PREPROCESS
# =====================================================
def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    return np.expand_dims(img,0)

# =====================================================
# SAFE PREDICTION EXTRACTOR
# =====================================================
def extract_malignancy_probability(pred):

    # Multi-output safe handling
    if isinstance(pred,(list,tuple)):
        main_output = pred[0]
    else:
        main_output = pred

    main_output = np.array(main_output).astype("float32")

    if main_output.ndim == 2:
        return float(main_output[0][0])
    elif main_output.ndim == 1:
        return float(main_output[0])
    else:
        return float(main_output.flatten()[0])

#Saving Patients Automatically After Test
from datetime import datetime

prediction = "Malignant"
probability = 0.87

stage = "Stage II"
subtype = "HER2+"
aggressiveness = 0.72

date = datetime.now().strftime("%Y-%m-%d")

save_test_patient(
    name,
    prediction,
    probability,
    stage,
    subtype,
    aggressiveness
)

generate_patient_pdf(name, age, prediction, probability, date)  
 
# =====================================================
# SAFE GRAD-CAM
# =====================================================
def generate_gradcam(model,img_array):

    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer,tf.keras.layers.Conv2D):
            last_conv = layer
            break

    if last_conv is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv.output,model.output]
    )

    with tf.GradientTape() as tape:
        conv_output,pred = grad_model(img_array)

        # Multi-output safe
        if isinstance(pred,(list,tuple)):
            pred = pred[0]

        loss = pred[:,0]

    grads = tape.gradient(loss,conv_output)
    pooled_grads = tf.reduce_mean(grads,axis=(0,1,2))
    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[...,tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap,0)
    heatmap /= (tf.reduce_max(heatmap)+1e-8)

    return heatmap.numpy()

# =====================================================
# CLINICAL ENGINES
# =====================================================
def molecular_subtype():
    if er=="Positive" and her2=="Negative":
        return "Luminal A"
    if er=="Positive" and her2=="Positive":
        return "Luminal B"
    if er=="Negative" and her2=="Positive":
        return "HER2-Enriched"
    if er=="Negative" and pr=="Negative" and her2=="Negative":
        return "Triple Negative"
    return "Indeterminate"

def staging():
    if tumor_size<=2 and lymph_nodes==0:
        return "Stage I"
    if 2<tumor_size<=5 or 1<=lymph_nodes<=3:
        return "Stage II"
    if tumor_size>5 or 4<=lymph_nodes<=9:
        return "Stage III"
    if lymph_nodes>=10:
        return "Stage IIIC"
    return "Stage Undetermined"

def aggressiveness_score(imaging_prob):
    size_factor = min(tumor_size/5,1)
    node_factor = min(lymph_nodes/10,1)
    her2_factor = 1 if her2=="Positive" else 0
    er_inverse = 1 if er=="Negative" else 0

    score = (0.30*imaging_prob +
             0.20*size_factor +
             0.20*node_factor +
             0.15*her2_factor +
             0.15*er_inverse)
    return min(score,1)

def node_risk_estimation(imaging_prob,agg):
    risk = (0.4*(tumor_size/5) +
            0.3*imaging_prob +
            0.3*agg)
    return min(risk,1)

def future_cancer_risk():
    score=0
    if age>50: score+=0.2
    if family_history=="Yes": score+=0.25
    if breast_density in ["C","D"]: score+=0.2
    if bmi>30: score+=0.15
    if hormone_therapy=="Yes": score+=0.2
    return min(score,1)

def biopsy_recommendation():
    if tumor_size<1:
        return "Core Needle Biopsy"
    if breast_density in ["C","D"]:
        return "Stereotactic Biopsy"
    if tumor_size>3:
        return "Excisional Biopsy"
    return "Ultrasound-Guided Biopsy"
def generate_verification_qr():

    unique_string = f"""
    {st.session_state.diagnosis}
    {st.session_state.final_risk}
    {st.session_state.stage}
    {datetime.now()}
    """

    verification_hash = hashlib.sha256(
        unique_string.encode()
    ).hexdigest()

    qr = qrcode.make(verification_hash)

    buffer = BytesIO()
    qr.save(buffer, format="PNG")

    return buffer.getvalue(), verification_hash

# =====================================================
# RUN AI
# =====================================================

if st.button("Run Full Oncology Assessment"):

    if not img_pil:
        st.error("Please upload mammogram.")

    else:

        img_array = preprocess(img_pil)
        pred = model.predict(img_array)

        imaging_prob = extract_malignancy_probability(pred)

        clinical_score = (
            (0.1 if age > 50 else 0) +
            (0.2 if family_history == "Yes" else 0) +
            (0.2 if tumor_size > 2 else 0)
        )

        final_risk = 0.7 * imaging_prob + 0.3 * clinical_score
        confidence = final_risk * 0.91

        diagnosis = "Malignant Suspicion" if final_risk > 0.5 else "Likely Benign"

        prediction = diagnosis
        probability = final_risk

        subtype = molecular_subtype()
        stage = staging()

        agg = aggressiveness_score(imaging_prob)

        node_risk = node_risk_estimation(imaging_prob, agg)

        referral = "Urgent Suspected Cancer Referral" \
            if final_risk > 0.75 or agg > 0.7 or stage.startswith("Stage III") \
            else "Routine Diagnostic Referral"

        if diagnosis == "Likely Benign":
            future = future_cancer_risk()
            screening = "Routine or Periodic Cancer Screening" if future > 0.3 else "Every 2 Years"
        else:
            future = None
            screening = "Oncology-Guided Treatment and Surveillance Protocol"

        biopsy = biopsy_recommendation()

        explanation = f"""
Imaging risk: {imaging_prob:.2f}
Stage: {stage}
Subtype: {subtype}
Aggressiveness: {agg:.2f}
Node involvement risk: {node_risk:.2f}
Referral decision based on fused risk model.
"""

        # ✅ SAVE RESULTS IN SESSION STATE
        st.session_state.final_risk = final_risk
        st.session_state.stage = stage
        st.session_state.subtype = subtype
        st.session_state.agg = agg
        st.session_state.node_risk = node_risk
        st.session_state.referral = referral
        st.session_state.biopsy = biopsy
        st.session_state.screening = screening
        st.session_state.future = future
        st.session_state.explanation = explanation
        st.session_state.diagnosis = diagnosis

        # =========================
        # SAVE PATIENT
        # =========================

        patient_data = {
            "Name": name,
            "Age": age,
            "Tumor Size": tumor_size,
            "Prediction": prediction,
            "Probability": probability,
            "Stage": stage,
            "Subtype": subtype,
            "Aggressiveness": aggressiveness
        }

        save_test_patient(
            name,
            prediction,
            probability,
            stage,
            subtype,
            aggressiveness
        )

        pdf_file = generate_patient_pdf(
            name,
            age,
            prediction,
            probability,
            date
        )

        st.success("Patient saved successfully")

        with open(pdf_file, "rb") as f:
            st.download_button(
                "Download Patient Report",
                f,
                file_name=pdf_file
            )

# =====================================================
# DISPLAY METRICS (SAFE VERSION)
# =====================================================

# Safely retrieve values from session state
# =====================================================
risk = st.session_state.get("final_risk", 0) or 0
stage = st.session_state.get("stage", "Unknown")
subtype = st.session_state.get("subtype", "Unknown")
agg = st.session_state.get("agg", 0) or 0

node_risk = st.session_state.get("node_risk", 0) or 0
referral = st.session_state.get("referral", "Not determined")
biopsy = st.session_state.get("biopsy", "Not determined")
screening = st.session_state.get("screening", "Not determined")

future = st.session_state.get("future", None)
explanation = st.session_state.get("explanation", "")

# =====================================================
# DISPLAY METRICS DASHBOARD
# =====================================================

col1, col2 = st.columns(2)
with col1:
    st.metric("Final Risk", f"{risk*100:.1f}%")

with col2:
    st.metric("Stage", stage)
col3, col4 = st.columns(2)

with col3:
    st.metric("Subtype", subtype)

with col4:
    st.metric("Aggressiveness", f"{agg*100:.1f}%")

# Second row
col5, col6 = st.columns(2)

with col5:
    st.metric("Node Risk", f"{node_risk*100:.1f}%")

with col6:
    st.metric("Referral", referral)
col7, col8 = st.columns(2)

with col7:
    st.metric("Biopsy", biopsy)

with col8:
    st.metric("Screening", screening)

# Future risk prediction
if future is not None:
    st.metric("Future 5-Year Risk", f"{future*100:.1f}%")

# AI explanation
if explanation:
    st.info(explanation)
    
# =====================================================
# Grad-CAM Visualization
# =====================================================
# Ensure image exists
# =====================================================
# Grad-CAM Visualization
# =====================================================

overlay = None
heatmap = None

if "img_pil" in locals() and img_pil is not None:

    # Convert image to array
    img_array = np.array(img_pil)

    # Resize to model input size
    img_resized = cv2.resize(img_array, (224, 224))

    # Normalize
    img_array = img_resized / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Generate Grad-CAM heatmap
    heatmap = generate_gradcam(model, img_array)

    if heatmap is not None:

        heatmap = cv2.resize(heatmap, (img_pil.width, img_pil.height))
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(
            np.array(img_pil),
            0.6,
            heatmap,
            0.4,
            0
        )

        st.subheader("Grad-CAM Heatmap (Suspicious Regions)")
        st.image(overlay, channels="BGR", width="stretch")

        # Save overlay for PDF
        heatmap_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        cv2.imwrite(heatmap_path, overlay)

        st.session_state.heatmap_path = heatmap_path

        # AI explanation
        explanation = (
            "The highlighted red/yellow regions represent areas "
            "where the model detected features strongly associated "
            "with malignant patterns. Cooler regions indicate lower "
            "influence on the model's decision."
        )

        st.session_state.heatmap_explanation = explanation
        st.info(explanation)

    else:
        st.warning("Grad-CAM could not be generated.")

else:
    st.warning("Upload an image to generate Grad-CAM.")


#Whatsapp sharingoverlay = overlay_heatmap(img_pil, heatmap)
# =====================================================
# Grad-CAM Visualization
# =====================================================
def overlay_heatmap(img_pil, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image
    """

    try:
        img = np.array(img_pil)

        # Resize heatmap to image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Normalize heatmap
        heatmap = np.uint8(255 * heatmap)

        # Apply color map
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay heatmap on image
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

        return overlay

    except Exception as e:
        print("Heatmap overlay error:", e)
        return None
if heatmap is not None and img_pil is not None:

    overlay = overlay_heatmap(img_pil, heatmap)

    if overlay is not None:

        st.subheader("Grad-CAM Heatmap (Suspicious Regions)")
        st.image(overlay, channels="BGR", width=400)

        # Save overlay for PDF
        heatmap_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        cv2.imwrite(heatmap_path, overlay)

        st.session_state.heatmap_path = heatmap_path

        # AI explanation
        explanation = (
            "The highlighted red/yellow regions represent areas "
            "where the model detected features strongly associated "
            "with malignant patterns. Cooler regions indicate lower "
            "influence on the model's decision."
        )

        st.session_state.heatmap_explanation = explanation
        st.info(explanation)

    else:
        st.warning("Heatmap overlay could not be generated.")

else:
    st.warning("Grad-CAM could not be generated.")  

   
#PDF REPORT GENERATOR 
def safe_pdf_text(text):
    if text is None:
        return ""
    return str(text).encode("latin-1", "replace").decode("latin-1")
def generate_clinical_pdf():

    diagnosis = st.session_state.get("diagnosis", "N/A")
    stage = st.session_state.get("stage", "N/A")
    subtype = st.session_state.get("subtype", "N/A")
    aggressiveness = st.session_state.get("agg", 0)
    lymph_risk = st.session_state.get("node_risk", "N/A")
    referral = st.session_state.get("referral", "N/A")
    screening = st.session_state.get("screening", "N/A")
    biopsy = st.session_state.get("biopsy", "N/A")
    explanation = st.session_state.get("heatmap_explanation", "")

    heatmap_path = st.session_state.get("heatmap_path", None)

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0,10,"BreastCancer Intelligence Clinical Report", ln=True)

    pdf.set_font("Arial","",12)
    pdf.cell(0,10,f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}", ln=True)

    pdf.ln(5)

    pdf.cell(0,10,safe_pdf_text(f"Malignancy Status: {diagnosis}"), ln=True)
    pdf.cell(0,10,safe_pdf_text(f"Diagnostic Stage: {stage}"), ln=True)
    pdf.cell(0,10,safe_pdf_text(f"Molecular Subtype: {subtype}"), ln=True)
    pdf.cell(0,10,safe_pdf_text(f"Tumor Aggressiveness Score: {aggressiveness:.2f}"), ln=True)
    pdf.cell(0,10,safe_pdf_text(f"Risk of Lymph Node Involvement: {lymph_risk}"), ln=True)
    
    pdf.ln(5)
    
    pdf.cell(0,10,safe_pdf_text(f"Referral Status: {referral}"), ln=True)
    pdf.cell(0,10,safe_pdf_text(f"Recommended Screening Frequency: {screening}"), ln=True)
    pdf.cell(0,10,safe_pdf_text(f"Recommended Biopsy Type: {biopsy}"), ln=True)
    
    pdf.ln(8)

    # ==============================
    # EMBED GRAD-CAM IMAGE
    # ==============================

    if heatmap_path and os.path.exists(heatmap_path):

        pdf.set_font("Arial","B",14)
        pdf.cell(0,10,"Grad-CAM Explainability Heatmap", ln=True)

        pdf.ln(3)

        pdf.image(
            heatmap_path,
            x=15,
            w=180
        )

        pdf.ln(5)

        pdf.set_font("Arial","",11)
        pdf.multi_cell(
            0,
            8,
            explanation
        )

    pdf.ln(10)

    pdf.set_font("Arial","I",9)
    pdf.multi_cell(
        0,
        8,
        "AI-generated report for clinical decision support. "
        "Our model is aimed in enhancing patients accuracy in Breast Cancer diagnosis."
    )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    pdf.output(temp_file.name)

    return temp_file.name
    
#DOWNLOAD PDF REPORT
def download_pdf():

    pdf_path = generate_clinical_pdf()

    with open(pdf_path, "rb") as file:
        st.download_button(
            label="Download Full Clinical Report",
            data=file,
            file_name="BreastCare_AI_Report.pdf",
            mime="application/pdf"
        )
         
#whatsapp sharing
def generate_whatsapp_link():

    diagnosis = st.session_state.get("diagnosis","N/A")
    stage = st.session_state.get("stage","N/A")
    subtype = st.session_state.get("subtype","N/A")
    aggressiveness = st.session_state.get("aggressiveness",0)
    lymph_risk = st.session_state.get("lymph_risk","N/A")

    referral = st.session_state.get("referral_status","N/A")
    screening = st.session_state.get("screening_frequency","N/A")
    biopsy = st.session_state.get("biopsy_type","N/A")
    next_steps = st.session_state.get("next_steps","N/A")

    message = f"""
BreastCare AI Clinical Report

Malignancy Status: {diagnosis}
Stage: {stage}
Molecular Subtype: {subtype}
Tumor Aggressiveness Score: {aggressiveness}

Lymph Node Risk: {lymph_risk}

Referral Status: {referral}
Screening Frequency: {screening}
Recommended Biopsy: {biopsy}

Next Steps:
{next_steps}

Generated {datetime.now().strftime('%d-%m-%Y %H:%M')}
"""

    encoded = urllib.parse.quote(message)

    return f"https://wa.me/?text={encoded}"
#Email sharing
def generate_email_link():

    subject = "BreastCare AI Clinical Report"

    diagnosis = st.session_state.get("diagnosis","N/A")
    stage = st.session_state.get("stage","N/A")
    subtype = st.session_state.get("subtype","N/A")
    aggressiveness = st.session_state.get("aggressiveness",0)

    body = f"""
BreastCare AI Clinical Report

Malignancy Status: {diagnosis}
Diagnostic Stage: {stage}
Molecular Subtype: {subtype}
Aggressiveness Score: {aggressiveness}

Please see attached clinical evaluation.

Generated {datetime.now().strftime('%d-%m-%Y %H:%M')}
"""

    return f"mailto:?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
#QR Code Verification System
# =====================================================
def generate_verification_qr():

    diagnosis = st.session_state.get("diagnosis", "Unknown")
    risk = st.session_state.get("final_risk", 0)
    stage = st.session_state.get("stage", "Unknown")

    unique_string = f"""
{diagnosis}
{risk}
{stage}
{datetime.now()}
"""

    verification_hash = hashlib.sha256(unique_string.encode()).hexdigest()

    qr = qrcode.make(verification_hash)

    buffer = BytesIO()
    qr.save(buffer, format="PNG")

    return buffer.getvalue(), verification_hash

# =====================================================
# Display QR Code
# =====================================================

qr_image, verification_hash = generate_verification_qr()

st.subheader("Report Verification QR")
st.image(qr_image, width=200)
st.caption(f"Verification ID: {verification_hash[:12]}")
# =====================================================
# CLINICAL REPORT DOWNLOAD & SHARING
# =====================================================

st.divider()
st.subheader("Clinical Report")

download_pdf()

st.subheader("Share Report")

col1, col2 = st.columns(2)

with col1:
    st.link_button(
        "Share via WhatsApp",
        generate_whatsapp_link()
    )

with col2:
    st.link_button(
        "Share via Email",
        generate_email_link()
    )

# =====================================================
# Save Case Section
# =====================================================

st.subheader("Save Case")

patient_name = st.text_input(
    "Patient Name",
    key="patient_name_input"
)

if st.button("Save Case", key="save_patient_button"):

    diagnosis = st.session_state.get("diagnosis", "Unknown")
    risk = st.session_state.get("final_risk", 0)

    # Use entered name or fallback
    name_to_save = patient_name if patient_name else "Auto Patient"

    save_patient(
        name_to_save,
        f"{diagnosis}, Risk: {risk}"
    )

    st.success("Case saved securely.")
    
# =============================
# SIDEBAR MENU
# =============================
menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Dashboard",
        "Diagnosis",
        "Training Patients",
        "Tested Patients"
    ]
)

choice = st.sidebar.selectbox("Menu", menu)

st.sidebar.success(
    f"Logged in as Dr. {st.session_state.get('doctor_name','Unknown')}"
)


#logged doctor Display
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# =============================
# TRAINING PATIENTS
# =============================

# ==========================================
# TRAINING PATIENTS PAGE
# ==========================================
elif menu == "Training Patients":

    st.title("Training Dataset Patients")

    df = get_training_patients()

    if not df.empty:

        df_display = df.rename(columns={
            "name": "Patient Image",
            "image_path": "Image Path",
            "diagnosis": "Malignancy Label"
        })

        st.dataframe(df_display, use_container_width=True)

        st.info(f"Total Patients Used For Training: {len(df_display)}")

    else:
        st.warning("No training data found.")


# =============================
# TESTED PATIENTS
# =============================

elif choice == "Tested Patients":

    st.title("Tested Patients History")

    df = get_test_patients()

    search = st.text_input("Search Patient")

    if search:
        df = df[df["name"].str.contains(search, case=False)]

    if df.empty:
        st.warning("No tested patients yet")

    else:

        st.dataframe(df)

        st.metric("Total Patients Tested", len(df))

        benign = len(df[df["prediction"] == "Benign"])
        malignant = len(df[df["prediction"] == "Malignant"])

        st.metric("Benign Cases", benign)
        st.metric("Malignant Cases", malignant)

        for index, row in df.iterrows():

            pdf_path = f"reports/patient_reports/{row['name']}_report.pdf"

            if os.path.exists(pdf_path):

                with open(pdf_path, "rb") as f:

                    st.download_button(
                        label=f"Download {row['name']} Report",
                        data=f,
                        file_name=f"{row['name']}_report.pdf",
                        mime="application/pdf"
                    )
if "final_risk" in locals():
    st.session_state.final_risk = final_risk
    st.session_state.stage = stage
    st.session_state.subtype = subtype
    st.session_state.agg = agg