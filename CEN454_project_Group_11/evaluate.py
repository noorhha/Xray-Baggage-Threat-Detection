import os
import cv2
import joblib
import numpy as np
import pandas as pd

from ultralytics import YOLO
from skimage.feature import hog, local_binary_pattern


# The instructor will place the hidden test images in this folder.
TEST_DIR = "./Test_data"

# Relative paths so the project can run on any computer after unzipping.
YOLO_MODEL_PATH = "./best.pt"
WEIGHTED_MODEL_PATH = "./weighted_ensemble_classifier.pkl"
WEIGHTED_SCALER_PATH = "./weighted_ensemble_scaler.pkl"

IMG_SIZE = (128, 128)

CONF_THRESHOLD = 0.25
LOW_CONF_THRESHOLD = 0.05

CLASS_NAMES = {
    0: "gun",
    1: "knife",
    2: "shuriken"
}


def normalize_label(label):
    """Keep the same model decision, but format labels consistently."""
    label = str(label).strip().lower()
    if label == "gun":
        return "gun"
    if label == "knife":
        return "knife"
    if label == "safe":
        return "safe"
    if label == "shuriken":
        return "shuriken"
    return label


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


def extract_features(gray):
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10)
    )
    lbp_hist = lbp_hist.astype("float")
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)

    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.array([np.sum(edges > 0) / edges.size])

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    angle = np.arctan2(sobel_y, sobel_x)

    angle_hist, _ = np.histogram(
        angle.ravel(),
        bins=16,
        range=(-np.pi, np.pi),
        weights=magnitude.ravel()
    )

    angle_hist = angle_hist.astype("float")
    angle_hist = angle_hist / (angle_hist.sum() + 1e-7)

    features = np.hstack([
        hog_features,
        lbp_hist,
        hist,
        edge_density,
        angle_hist
    ])

    return features


def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, x2, y2


def main():
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError("Test_data folder was not found. Please place test images in ./Test_data/")

    yolo_model = YOLO(YOLO_MODEL_PATH)
    classifier = joblib.load(WEIGHTED_MODEL_PATH)
    scaler = joblib.load(WEIGHTED_SCALER_PATH)

    image_files = sorted([
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    prediction_rows = []
    localization_rows = []

    os.makedirs("prediction_visuals", exist_ok=True)

    for filename in image_files:
        image_path = os.path.join(TEST_DIR, filename)
        image = cv2.imread(image_path)

        if image is None:
            print("Could not read:", filename)
            continue

        # ---------------------------
        # 1. Weighted ensemble classification
        # ---------------------------
        gray = preprocess_image(image_path)
        features = extract_features(gray)
        features_scaled = scaler.transform([features])

        predicted_label = normalize_label(classifier.predict(features_scaled)[0])

        # ---------------------------
        # 2. YOLO detection/localization
        # ---------------------------
        results = yolo_model(image_path, conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        yolo_label = "safe"
        confidence = ""
        x1 = y1 = x2 = y2 = ""

        if (boxes is None or len(boxes) == 0) and predicted_label != "safe":
            results = yolo_model(image_path, conf=LOW_CONF_THRESHOLD, verbose=False)
            boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            best_index = boxes.conf.argmax()
            best_box = boxes[best_index]

            class_id = int(best_box.cls[0].cpu().numpy())
            confidence = float(best_box.conf[0].cpu().numpy())

            yolo_label = CLASS_NAMES.get(class_id, str(class_id))

            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            height, width = image.shape[:2]
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, width, height)

        # Required classification CSV format
        prediction_rows.append({
            "ImageName": filename,
            "PredictedLabel": predicted_label
        })

        # Required localization CSV format: include only detected threat boxes.
        if x1 != "" and yolo_label != "safe":
            localization_rows.append({
                "ImageName": filename,
                "Label": yolo_label,
                "X_min": int(x1),
                "Y_min": int(y1),
                "X_max": int(x2),
                "Y_max": int(y2)
            })

        # Optional visual output for checking predictions.
        image_box = image.copy()

        if x1 != "":
            cv2.rectangle(image_box, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                image_box,
                f"Final: {predicted_label} | YOLO: {yolo_label} {confidence:.2f}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        else:
            cv2.putText(
                image_box,
                f"Final: {predicted_label} | no YOLO box",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        output_path = os.path.join("prediction_visuals", filename)
        cv2.imwrite(output_path, image_box)

    # Save outputs using the exact file names and headers required by the submission guide.
    pd.DataFrame(prediction_rows, columns=["ImageName", "PredictedLabel"]).to_csv(
        "predictions.csv", index=False
    )

    pd.DataFrame(
        localization_rows,
        columns=["ImageName", "Label", "X_min", "Y_min", "X_max", "Y_max"]
    ).to_csv("localization.csv", index=False)

    print("Done.")
    print("Saved predictions.csv")
    print("Saved localization.csv")
    print("Saved visual results in prediction_visuals folder")


if __name__ == "__main__":
    main()
