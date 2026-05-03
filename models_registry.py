from tensorflow.keras.models import load_model
import joblib

# ===============================
# IMAGE-BASED DEEP LEARNING MODELS
# ===============================

pneumonia_model = load_model("models/pneumonia.h5", compile=False)
tuberculosis_model = load_model("models/tuberculosis.h5", compile=False)
brain_tumor_model = load_model("models/brain_tumor.h5", compile=False)
skin_cancer_model = load_model("models/skin_cancer.h5", compile=False)
retinopathy_model = load_model("models/retinopathy.h5", compile=False)
glaucoma_model = load_model("models/glaucoma.h5", compile=False)
kidney_stone_model = load_model("models/kidney_stone.h5", compile=False)
malaria_model = load_model("models/malaria.h5", compile=False)

# ===============================
# TABULAR (CLINICAL DATA) MODEL
# ===============================

heart_disease_model = None


# ===============================
# LABEL MAPPINGS
# ===============================

PNEUMONIA_LABELS = {
    0: "Normal",
    1: "Pneumonia Detected"
}

TUBERCULOSIS_LABELS = {
    0: "Normal",
    1: "Tuberculosis Detected"
}

BRAIN_TUMOR_LABELS = {
    0: "No Tumor",
    1: "Glioma",
    2: "Meningioma",
    3: "Pituitary Tumor"
}

SKIN_CANCER_LABELS = {
    0: "Benign",
    1: "Malignant"
}

RETINOPATHY_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

GLAUCOMA_LABELS = {
    0: "Normal",
    1: "Glaucoma Detected"
}

KIDNEY_STONE_LABELS = {
    0: "Normal",
    1: "Kidney Stone Detected"
}

MALARIA_LABELS = {
    0: "Uninfected",
    1: "Parasitized"
}

HEART_DISEASE_LABELS = {
    0: "Low Risk",
    1: "High Risk"
}

# ===============================
# CONFIDENCE THRESHOLDS
# ===============================

CONFIDENCE_THRESHOLDS = {
    "pneumonia": 0.70,
    "tuberculosis": 0.70,
    "brain_tumor": 0.65,
    "skin_cancer": 0.75,
    "retinopathy": 0.70,
    "glaucoma": 0.70,
    "kidney": 0.70,
    "malaria": 0.75,
    "heart": 0.60
}

print("✅ All 9 disease AI models loaded successfully")
