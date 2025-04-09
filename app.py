import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from skimage.color import rgb2gray

label_map = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

@st.cache_resource
def load_model():
    model = joblib.load("model/optimized_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

def extract_features(img):
    img = cv2.resize(img, (128, 128))
    gray = rgb2gray(img)
    hog_features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()
    hist_h /= hist_h.sum() if hist_h.sum() != 0 else 1
    hist_s /= hist_s.sum() if hist_s.sum() != 0 else 1
    hist_v /= hist_v.sum() if hist_v.sum() != 0 else 1
    return np.hstack([hog_features, hist_h, hist_s, hist_v])

def detect_and_classify_objects(img, model, scaler):
    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Basic preprocessing to find contours
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Ignore very small regions (likely noise)
        if w < 30 or h < 30:
            continue

        # Extract ROI and classify
        roi = img[y:y+h, x:x+w]
        features = extract_features(roi)
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        label = label_map[pred]

        # Draw bounding box and label
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(output_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return output_img

# Streamlit UI
st.title("Garbage Object Detection and Classification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    model, scaler = load_model()
    result_img = detect_and_classify_objects(img, model, scaler)

    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Classified Image with Bounding Boxes", use_column_width=True)
