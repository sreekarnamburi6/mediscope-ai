from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image, ImageStat

# ---------------- IMPORT HEART LOGIC ----------------
from modules.heart import predict_heart_disease

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

print("🚀 MediScope AI app.py loaded successfully")

# ================= LOAD MODELS =================
pneumonia_model = tf.keras.models.load_model("models/pneumonia.h5", compile=False)
tb_model = tf.keras.models.load_model("models/tuberculosis.h5", compile=False)
brain_model = tf.keras.models.load_model("models/brain_tumor.h5", compile=False)
skin_model = tf.keras.models.load_model("models/skin_cancer.h5", compile=False)
retina_model = tf.keras.models.load_model("models/retinopathy.h5", compile=False)
glaucoma_model = tf.keras.models.load_model("models/glaucoma.h5", compile=False)
kidney_model = tf.keras.models.load_model("models/kidney_stone.h5", compile=False)
malaria_model = tf.keras.models.load_model("models/malaria.h5", compile=False)

# ================= CONFIG =================
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ================= UTILITIES =================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(file, size=(224, 224)):
    file.seek(0)
    img = Image.open(file).convert("RGB")

    if img.size[0] < 200 or img.size[1] < 200:
        return None, img

    img_resized = img.resize(size)
    img_arr = np.array(img_resized) / 255.0
    return np.expand_dims(img_arr, axis=0), img


# ================= IMAGE TYPE VALIDATORS =================
def is_grayscale_like(img):
    arr = np.array(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return np.mean(np.abs(r - g)) < 12 and np.mean(np.abs(g - b)) < 12


def has_xray_contrast(img):
    gray = img.convert("L")
    return ImageStat.Stat(gray).stddev[0] > 18


def is_mri_like(img):
    return is_grayscale_like(img)


def is_fundus_like(img):
    arr = np.array(img)
    return np.mean(arr[:, :, 0]) > np.mean(arr[:, :, 2])


def is_skin_like(img):
    arr = np.array(img)
    return np.std(arr) > 40


def is_ultrasound_like(img):
    gray = img.convert("L")
    return ImageStat.Stat(gray).stddev[0] < 35


def is_blood_smear_like(img):
    arr = np.array(img)
    return np.mean(arr[:, :, 0]) > np.mean(arr[:, :, 1])


def risk_from_confidence(conf):
    if conf < 40:
        return "low"
    elif conf < 70:
        return "medium"
    return "high"


def render_analysis(**ctx):
    return render_template("ai_analysis.html", **ctx)


def reject(ctx, msg):
    ctx["result"] = msg
    ctx["confidence"] = 0
    ctx["risk"] = "low"
    return render_analysis(**ctx)


# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")


# ================= PNEUMONIA =================
@app.route("/pneumonia", methods=["GET", "POST"])
def pneumonia():
    ctx = dict(title="Pneumonia Diagnosis", icon="🫁", action="/pneumonia",
               upload_hint="Upload Chest X-ray only", result=None, confidence=0, risk="low")

    if request.method == "POST":
        file = request.files.get("file")
        if not file or not allowed_file(file.filename):
            return reject(ctx, "❌ Invalid file")

        image, raw = preprocess_image(file)
        if image is None or not is_grayscale_like(raw) or not has_xray_contrast(raw):
            return reject(ctx, "❌ Chest X-ray required")

        pred = pneumonia_model.predict(image)[0][0]
        ctx["confidence"] = int(pred * 100)
        ctx["risk"] = risk_from_confidence(ctx["confidence"])
        ctx["result"] = "Pneumonia Detected ❌" if pred > 0.5 else "Normal ✅"

    return render_analysis(**ctx)


# ================= TUBERCULOSIS =================
@app.route("/tuberculosis", methods=["GET", "POST"])
def tuberculosis():
    ctx = dict(title="Tuberculosis Detection", icon="🫁", action="/tuberculosis",
               upload_hint="Upload Chest X-ray only", result=None, confidence=0, risk="low")

    if request.method == "POST":
        image, raw = preprocess_image(request.files["file"])
        if image is None or not is_grayscale_like(raw) or not has_xray_contrast(raw):
            return reject(ctx, "❌ Chest X-ray required")

        pred = tb_model.predict(image)[0][0]
        ctx["confidence"] = int(pred * 100)
        ctx["risk"] = risk_from_confidence(ctx["confidence"])
        ctx["result"] = "TB Detected ❌" if pred > 0.5 else "No TB ✅"

    return render_analysis(**ctx)


# ================= BRAIN =================
@app.route("/brain", methods=["GET", "POST"])
def brain():
    ctx = dict(title="Brain Tumor Detection", icon="🧠", action="/brain",
               upload_hint="Upload Brain MRI only", result=None, confidence=0, risk="low")

    if request.method == "POST":
        image, raw = preprocess_image(request.files["file"])
        if image is None or not is_mri_like(raw):
            return reject(ctx, "❌ Brain MRI required")

        preds = brain_model.predict(image)[0]
        cls = np.argmax(preds)
        ctx["confidence"] = int(preds[cls] * 100)
        ctx["risk"] = risk_from_confidence(ctx["confidence"])
        ctx["result"] = "Tumor Detected ❌" if cls != 0 else "No Tumor ✅"

    return render_analysis(**ctx)


# ================= SKIN =================
@app.route("/skin", methods=["GET", "POST"])
def skin():
    ctx = dict(title="Skin Cancer Detection", icon="🧬", action="/skin",
               upload_hint="Upload skin lesion image", result=None, confidence=0, risk="low")

    if request.method == "POST":
        image, raw = preprocess_image(request.files["file"])
        if image is None or not is_skin_like(raw):
            return reject(ctx, "❌ Skin lesion image required")

        preds = skin_model.predict(image)[0]
        cls = np.argmax(preds)
        ctx["confidence"] = int(preds[cls] * 100)
        ctx["risk"] = risk_from_confidence(ctx["confidence"])
        ctx["result"] = "Cancer Detected ❌" if cls != 0 else "Benign ✅"

    return render_analysis(**ctx)


# ================= RETINA =================
@app.route("/retina", methods=["GET", "POST"])
def retina():
    ctx = dict(title="Diabetic Retinopathy", icon="👁️", action="/retina",
               upload_hint="Upload retinal fundus image", result=None, confidence=0, risk="low")

    if request.method == "POST":
        image, raw = preprocess_image(request.files["file"])
        if image is None or not is_fundus_like(raw):
            return reject(ctx, "❌ Fundus image required")

        preds = retina_model.predict(image)[0]
        cls = np.argmax(preds)
        ctx["confidence"] = int(preds[cls] * 100)
        ctx["risk"] = risk_from_confidence(ctx["confidence"])
        ctx["result"] = "Retinopathy Detected ❌" if cls != 0 else "Healthy Retina ✅"

    return render_analysis(**ctx)


# ================= GLAUCOMA =================
@app.route("/glaucoma", methods=["GET", "POST"])
def glaucoma():
    ctx = dict(title="Glaucoma Detection", icon="👁️", action="/glaucoma",
               upload_hint="Upload retinal fundus image", result=None, confidence=0, risk="low")

    if request.method == "POST":
        image, raw = preprocess_image(request.files["file"])
        if image is None or not is_fundus_like(raw):
            return reject(ctx, "❌ Fundus image required")

        pred = glaucoma_model.predict(image)[0][0]
        ctx["confidence"] = int(pred * 100)
        ctx["risk"] = risk_from_confidence(ctx["confidence"])
        ctx["result"] = "Glaucoma Detected ❌" if pred > 0.5 else "No Glaucoma ✅"

    return render_analysis(**ctx)


# ================= KIDNEY =================
@app.route("/kidney", methods=["GET", "POST"])
def kidney():
    ctx = dict(title="Kidney Stone Detection", icon="🩺", action="/kidney",
               upload_hint="Upload ultrasound image", result=None, confidence=0, risk="low")

    if request.method == "POST":
        image, raw = preprocess_image(request.files["file"])
        if image is None or not is_ultrasound_like(raw):
            return reject(ctx, "❌ Ultrasound image required")

        pred = kidney_model.predict(image)[0][0]
        ctx["confidence"] = int(pred * 100)
        ctx["risk"] = risk_from_confidence(ctx["confidence"])
        ctx["result"] = "Stone Detected ❌" if pred > 0.5 else "No Stone ✅"

    return render_analysis(**ctx)


# ================= MALARIA =================
@app.route("/malaria", methods=["GET", "POST"])
def malaria():
    ctx = dict(title="Malaria Detection", icon="🦠", action="/malaria",
               upload_hint="Upload blood smear image", result=None, confidence=0, risk="low")

    if request.method == "POST":
        image, raw = preprocess_image(request.files["file"])
        if image is None or not is_blood_smear_like(raw):
            return reject(ctx, "❌ Blood smear image required")

        pred = malaria_model.predict(image)[0][0]
        ctx["confidence"] = int(pred * 100)
        ctx["risk"] = risk_from_confidence(ctx["confidence"])
        ctx["result"] = "Malaria Detected ❌" if pred > 0.5 else "No Malaria ✅"

    return render_analysis(**ctx)


# ================= HEART =================
@app.route("/heart", methods=["GET", "POST"])
def heart():
    ctx = dict(title="Heart Disease Prediction", icon="❤️", action="/heart",
               upload_hint="Enter clinical parameters", result=None,
               confidence=0, risk="low", explanation="")

    if request.method == "POST":
        result = predict_heart_disease(request.form)
        ctx["result"] = result["prediction"]
        ctx["confidence"] = int(result["confidence"])
        ctx["risk"] = result["risk_level"].lower()
        ctx["explanation"] = result["explanation"]

    return render_analysis(**ctx)


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
