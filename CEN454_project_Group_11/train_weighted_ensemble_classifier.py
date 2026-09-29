import os
import cv2
import numpy as np
import joblib

from skimage.feature import hog, local_binary_pattern

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier


# For honest testing use TrainData_80.
# For final doctor submission, change this to "TrainData".
DATASET_DIR = "TrainData"

CLASSES = {
    "safe": "safe",
    "GUN": "gun",
    "knife": "knife",
    "shuriken": "shuriken"
}

IMG_SIZE = (128, 128)


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Noise reduction
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


def extract_features(gray):
    # HOG shape/orientation features
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    # LBP texture features
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10)
    )
    lbp_hist = lbp_hist.astype("float")
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)

    # Intensity histogram
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Canny edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.array([np.sum(edges > 0) / edges.size])

    # Sobel orientation histogram
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


# =========================
# Load dataset
# =========================
X = []
y = []

print("Loading dataset and extracting selected features...\n")

for folder_name, label in CLASSES.items():
    folder_path = os.path.join(DATASET_DIR, folder_name)

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Reading {folder_name}: {len(files)} images")

    for filename in files:
        image_path = os.path.join(folder_path, filename)

        gray = preprocess_image(image_path)
        features = extract_features(gray)

        X.append(features)
        y.append(label)


X = np.array(X)
y = np.array(y)

print("\nTotal images:", len(X))
print("Feature vector size:", X.shape[1])
print("Labels:", np.unique(y, return_counts=True))


X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


best_model = None
best_score = -1
best_info = None


# Try different gun weights because many errors are gun -> knife
gun_weights = [1.0, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]

for gun_weight in gun_weights:
    print("\n" + "=" * 60)
    print(f"Testing gun weight = {gun_weight}")

    # IMPORTANT:
    # VotingClassifier internally converts class labels into numbers.
    # Alphabetical order is:
    # 0 = gun
    # 1 = knife
    # 2 = safe
    # 3 = shuriken
    class_weights = {
        0: gun_weight,  # gun
        1: 1.0,         # knife
        2: 2.0,         # safe
        3: 2.0          # shuriken
    }

    svm_rbf = SVC(
        kernel="rbf",
        C=5,
        gamma="scale",
        class_weight=class_weights,
        probability=True
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )

    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )

    ensemble = VotingClassifier(
        estimators=[
            ("svm_rbf", svm_rbf),
            ("extra_trees", extra_trees),
            ("random_forest", random_forest),
            ("knn", knn)
        ],
        voting="soft",
        weights=[2, 2, 1, 1]
    )

    ensemble.fit(X_train_scaled, y_train)
    y_pred = ensemble.predict(X_val_scaled)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")

    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=["safe", "gun", "knife", "shuriken"]
    )

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Confusion matrix labels: safe, gun, knife, shuriken")
    print(cm)

    # Balance accuracy and macro F1
    score = acc + macro_f1

    if score > best_score:
        best_score = score
        best_model = ensemble
        best_info = {
            "gun_weight": gun_weight,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "confusion_matrix": cm
        }


print("\n\nBEST WEIGHTED ENSEMBLE")
print("=" * 60)
print("Gun weight:", best_info["gun_weight"])
print("Accuracy:", best_info["accuracy"])
print("Macro F1:", best_info["macro_f1"])
print("Confusion matrix labels: safe, gun, knife, shuriken")
print(best_info["confusion_matrix"])

y_best_pred = best_model.predict(X_val_scaled)

print("\nClassification report:\n")
print(classification_report(y_val, y_best_pred))

joblib.dump(scaler, "weighted_ensemble_scaler.pkl")
joblib.dump(best_model, "weighted_ensemble_classifier.pkl")

with open("weighted_ensemble_info.txt", "w") as f:
    f.write(f"Best gun weight: {best_info['gun_weight']}\n")
    f.write(f"Accuracy: {best_info['accuracy']}\n")
    f.write(f"Macro F1: {best_info['macro_f1']}\n")
    f.write("Confusion matrix labels: safe, gun, knife, shuriken\n")
    f.write(str(best_info["confusion_matrix"]))

print("\nSaved:")
print("weighted_ensemble_scaler.pkl")
print("weighted_ensemble_classifier.pkl")
print("weighted_ensemble_info.txt")