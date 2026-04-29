import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import pandas as pd
from PIL import Image
import datetime

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Plant Disease AI", page_icon="🌿", layout="wide")

def local_css():
    css = """
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
<style>
* { font-family: 'Outfit', sans-serif; }
.main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
.stApp { background: #f8fafc; }
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
.sidebar-history { background: rgba(46, 125, 50, 0.05); border-radius: 12px; padding: 12px; margin-bottom: 10px; border-left: 4px solid #2e7d32; transition: transform 0.2s ease; }
.sidebar-history:hover { transform: translateX(5px); background: rgba(46, 125, 50, 0.1); }
.stButton>button { border-radius: 12px; height: 3.5em; background: linear-gradient(90deg, #2e7d32 0%, #43a047 100%); color: white; font-weight: 600; border: none; box-shadow: 0 4px 12px rgba(46, 125, 50, 0.2); transition: all 0.3s ease; }
.stButton>button:hover { box-shadow: 0 6px 16px rgba(46, 125, 50, 0.3); transform: translateY(-2px); }
.diagnosis-card { padding: 30px; border-radius: 24px; background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.3); box-shadow: 0 20px 40px rgba(0,0,0,0.05); margin-bottom: 25px; animation: fadeIn 0.6s ease-out; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
.status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 600; margin-bottom: 10px; }
.status-healthy { background: #dcfce7; color: #166534; }
.status-diseased { background: #fee2e2; color: #991b1b; }
.doctor-card { background: white; padding: 20px; border-radius: 16px; border: 1px solid #f1f5f9; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); transition: all 0.3s ease; }
.doctor-card:hover { box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); border-color: #2e7d32; }
</style>
"""
    st.markdown(css, unsafe_allow_html=True)

local_css()

# --- CACHE DATA & MODEL ---
@st.cache_resource
def load_assets():
    model = load_model("plant_disease_model.h5")
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    class_names = [None] * len(class_indices)
    for name, index in class_indices.items():
        class_names[index] = name
    return model, class_names


model, class_names = load_assets()

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- DATA ---
disease_info = {
    "Pepper__bell___Bacterial_spot": """
        **Symptoms:** Small, water-soaked spots on leaves that turn brown and papery.
        **Management:** 
        ➡ Use certified, disease-free seeds.
        ➡ Apply copper-based bactericides every 7-10 days.
        ➡ Avoid overhead irrigation to reduce leaf wetness.
    """,
    "Pepper__bell___healthy": "✅ **Condition:** Healthy. Your plant shows no signs of disease. Maintain consistent watering and nutrition.",
    "Potato___Early_blight": """
        **Symptoms:** Dark, concentric rings (target-like) on older leaves.
        **Management:**
        ➡ Use fungicides like mancozeb or chlorothalonil.
        ➡ Remove and destroy infected plant debris.
        ➡ Practice 3-year crop rotation with non-solanaceous crops.
    """,
    "Potato___healthy": "✅ **Condition:** Healthy. Keep monitoring for any changes in leaf color or texture.",
    "Potato___Late_blight": """
        **Symptoms:** Large, irregular water-soaked patches on leaves that quickly turn brown and rot.
        **Management:**
        ➡ Apply metalaxyl-based fungicides immediately.
        ➡ Remove and bury affected parts.
        ➡ Ensure good field drainage and avoid high humidity.
    """,
    "Tomato__Target_Spot": """
        **Symptoms:** Circular brown spots with a yellow halo and characteristic target-like rings.
        **Management:**
        ➡ Use chlorothalonil or mancozeb sprays.
        ➡ Avoid water splash on leaves by using drip irrigation.
        ➡ Improve air circulation between plants.
    """,
    "Tomato__Tomato_mosaic_virus": """
        **Symptoms:** Mottling, mosaic patterns, and curling of leaves.
        **Management:**
        ➡ Destroy infected plants immediately; there is no cure.
        ➡ Disinfect tools with a 10% bleach solution.
        ➡ Avoid tobacco exposure as it can carry the virus.
    """,
    "Tomato__Tomato_YellowLeaf__Curl_Virus": """
        **Symptoms:** Severe stunting, yellowing of leaf margins, and upward leaf curling.
        **Management:**
        ➡ Remove infected plants and control whitefly populations.
        ➡ Use virus-resistant seed varieties.
        ➡ Use reflective mulches to deter whiteflies.
    """,
    "Tomato_Bacterial_spot": """
        **Symptoms:** Small, dark, greasy spots on leaves and fruit.
        **Management:**
        ➡ Use copper-based sprays early in the season.
        ➡ Avoid handling plants when foliage is wet.
        ➡ Practice strict sanitation and crop rotation.
    """,
    "Tomato_Early_blight": """
        **Symptoms:** "Bull's-eye" patterned spots on lower leaves.
        **Management:**
        ➡ Apply chlorothalonil-based fungicides.
        ➡ Remove lower infected leaves to prevent upward spread.
        ➡ Improve plant spacing for better airflow.
    """,
    "Tomato_healthy": "✅ **Condition:** Healthy. Ensure proper support (staking) and consistent moisture.",
    "Tomato_Late_blight": """
        **Symptoms:** Dark, water-soaked areas on leaves that rapidly expand and kill the plant.
        **Management:**
        ➡ Use copper or metalaxyl fungicides preventatively.
        ➡ Remove and destroy infected leaves immediately.
        ➡ Avoid wet foliage overnight; water early in the morning.
    """,
    "Tomato_Leaf_Mold": """
        **Symptoms:** Pale green or yellow spots on upper leaf surfaces; olive-green mold underneath.
        **Management:**
        ➡ Use sulfur-based fungicides.
        ➡ Improve air circulation in greenhouses.
        ➡ Keep relative humidity below 85%.
    """,
    "Tomato_Septoria_leaf_spot": """
        **Symptoms:** Numerous small, circular spots with dark borders and gray centers.
        **Management:**
        ➡ Use neem oil or liquid copper fungicides.
        ➡ Remove and destroy spotted leaves.
        ➡ Avoid overhead watering and keep soil mulched.
    """,
    "Tomato_Spider_mites_Two_spotted_spider_mite": """
        **Symptoms:** Fine stippling (yellow dots) on leaves and delicate webbing.
        **Management:**
        ➡ Use neem oil, insecticidal soap, or predatory mites.
        ➡ Keep humidity high around plants.
        ➡ Blast mites off leaves with a strong stream of water.
    """
}

# --- SIDEBAR ---
with st.sidebar:
    try:
        st.image("assets/banner.png", width="stretch")
    except:
        st.title("🌿 PlantAI Pro")
    
    st.markdown("### 🕒 Recent Scans")
    if not st.session_state.history:
        st.info("No scans yet.")
    else:
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"""
            <div class='sidebar-history'>
                <div style="font-weight: 600; color: #1e293b;">{item['name']}</div>
                <div style="font-size: 0.8em; color: #64748b; margin-top: 4px;">
                    <span>🕒 {item['time']}</span> | <span style="color: #2e7d32;">🎯 {item['conf']:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 🌡️ Environment Dashboard")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Temp", "28°C", "1.2°C")
    with col_b:
        st.metric("Humidity", "65%", "-5%")
    
    st.markdown("""
    <div style="background: #fff4e5; padding: 12px; border-radius: 10px; border-left: 4px solid #ed8936; font-size: 0.85em;">
        ⚠️ <b>Risk Alert:</b> High humidity detected. Fungal growth risk is elevated for <b>Potato</b> and <b>Tomato</b> crops.
    </div>
    """, unsafe_allow_html=True)

# --- MAIN UI ---
st.title("🌿 Plant Disease Detection Pro")
st.write("Upload a leaf image or take a photo to detect plant diseases with high-precision AI.")

# Top Dashboard (Environment)
env_col1, env_col2, env_col3 = st.columns([1, 1, 2])
with env_col1:
    st.metric("Temperature", "28°C", "1.2°C")
with env_col2:
    st.metric("Humidity", "65%", "-5%")
with env_col3:
    st.markdown("""
    <div style="background: #fff4e5; padding: 10px; border-radius: 10px; border-left: 4px solid #ed8936; font-size: 0.8em;">
        ⚠️ <b>Risk Alert:</b> High humidity detected. Fungal growth risk is elevated.
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📁 Upload leaf image", type=["jpg", "jpeg", "png"])
    camera_photo = st.camera_input("📸 Take a photo")

image_input = camera_photo if camera_photo else uploaded_file

with col2:
    if image_input is not None:
        st.image(image_input, caption="🖼️ Analysis Target", width="stretch")
        
        # Prediction
        with st.spinner("Analyzing leaf patterns..."):
            img = Image.open(image_input).resize((224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = prediction[0][predicted_index] * 100
            
            pretty_name = predicted_class.replace("___", " - ").replace("__", " ").replace("_", " ").title()
            pretty_name = " ".join(pretty_name.split()) # Remove extra spaces

            # Store in history if it's a new scan
            scan_time = datetime.datetime.now().strftime("%H:%M:%S")
            if not st.session_state.history or st.session_state.history[-1]['name'] != pretty_name:
                st.session_state.history.append({"name": pretty_name, "conf": confidence, "time": scan_time})

        # Display Results
        status_class = "status-healthy" if "healthy" in predicted_class.lower() else "status-diseased"
        status_text = "HEALTHY" if "healthy" in predicted_class.lower() else "DISEASE DETECTED"
        
        st.markdown(f"""
        <div class="diagnosis-card">
            <div class="status-badge {status_class}">{status_text}</div>
            <h2 style="margin: 0; color: #1e293b;">{pretty_name}</h2>
            <p style="color: #64748b; margin-bottom: 20px;">AI Analysis Report</p>
            
            <div style="background: #f8fafc; padding: 15px; border-radius: 12px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: 600; color: #475569;">Confidence Score</span>
                    <span style="font-weight: 600; color: #2e7d32;">{confidence:.1f}%</span>
                </div>
                <div style="height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden;">
                    <div style="width: {confidence}%; height: 100%; background: linear-gradient(90deg, #2e7d32, #43a047);"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if confidence < 60:
            st.warning("⚠️ **Low Confidence Scan:** The AI suggests caution. For more accurate results, ensure the leaf is well-lit, centered, and against a plain background.")

        # Show treatment
        if predicted_class in disease_info:
            with st.expander("🩺 Treatment & Expert Recommendations", expanded=True):
                st.markdown(f"""
                <div style="padding: 10px; line-height: 1.6;">
                    {disease_info[predicted_class]}
                </div>
                """, unsafe_allow_html=True)
                
                st.info("💡 **Pro Tip:** Prune infected leaves early to prevent the spread to healthy parts of the plant.")
                
                # Report Download
                report_text = f"DIAGNOSIS REPORT\nPlant: {pretty_name}\nConfidence: {confidence:.2f}%\nScan Time: {scan_time}\n\nRecommended Actions:\n{disease_info[predicted_class]}"
                st.download_button(
                    label="📥 Download Diagnosis Report",
                    data=report_text,
                    file_name=f"diagnosis_{scan_time.replace(':', '-')}.txt",
                    mime="text/plain"
                )
        else:
            st.warning("⚠️ Detailed treatment protocol not found in local database for this specific strain.")

st.divider()

# --- DOCTOR SECTION ---
st.subheader("👨‍⚕️ Find Nearby Plant Health Experts")
try:
    doctor_df = pd.read_csv("plant_doctors.csv")
    locations = sorted(doctor_df["location"].unique())
    selected_loc = st.selectbox("📍 Select your city/town:", locations)

    if selected_loc:
        matches = doctor_df[doctor_df["location"] == selected_loc]
        doc_cols = st.columns(3)
        for i, (_, row) in enumerate(matches.iterrows()):
            with doc_cols[i % 3]:
                st.markdown(f"""
                <div class="doctor-card">
                    <div style="font-size: 1.1em; font-weight: 600; color: #1e293b; margin-bottom: 5px;">🧑‍🔬 {row['name']}</div>
                    <div style="font-size: 0.9em; color: #64748b; margin-bottom: 12px;">{row['designation']}</div>
                    <div style="background: #f1f5f9; padding: 8px; border-radius: 8px; font-size: 0.85em; color: #2e7d32; font-weight: 600;">
                        📞 {row['contact']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Could not load doctor directory. Error: {e}")
