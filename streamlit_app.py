# ============================================================
# PORK vs NON-PORK IMAGE CLASSIFICATION SYSTEM
# Streamlit + TensorFlow + MobileNetV2
#
# Original folders:
#   meat_dataset/train/pok/
#   meat_dataset/train/beef/
#   meat_dataset/train/horse/
#
# Binary labels:
#   0 = PORK
#   1 = NON-PORK (beef + horse)
# ============================================================

import os
import io
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# ------------------------- SETTINGS -------------------------

DATA_DIR = "meat_dataset/train"
MODEL_PATH = "pork_classifier.keras"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
LEARNING_RATE = 0.0001
SEED = 123
THRESHOLD = 0.50
PORK_FOLDER_NAMES = {"pork", "pok"}
AUTOTUNE = tf.data.AUTOTUNE

# ----------------------- PAGE CONFIG ------------------------

st.set_page_config(
    page_title="Pork vs Non-Pork Classifier",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- STYLE --------------------------

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.main-title {font-size: 2rem; font-weight: 800; text-align:center;}
.sub-title {text-align:center; color:gray; margin-bottom:1.5rem;}
.result-pork {
    background:#e5f5e9; color:#146c35; padding:1.2rem;
    border-radius:12px; text-align:center; font-size:2rem;
    font-weight:800; border:1px solid #a8d5b5;
}
.result-nonpork {
    background:#e6f1ff; color:#084298; padding:1.2rem;
    border-radius:12px; text-align:center; font-size:2rem;
    font-weight:800; border:1px solid #b6d4fe;
}
.small-muted {color:gray; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------- SESSION STATE ----------------------

defaults = {
    "training_history": None,
    "training_completed": False,
    "last_evaluation": None,
    "prediction_history": [],
    "training_error": None
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ------------------------- HELPERS --------------------------

def get_dataset_class_names(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}. "
            "Create meat_dataset/train and place the three class folders inside it."
        )

    names = sorted([
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    ])

    normalized = [name.strip().lower() for name in names]
    pork_matches = [i for i, name in enumerate(normalized)
                    if name in PORK_FOLDER_NAMES]

    if len(names) != 3:
        raise ValueError(
            f"Expected exactly 3 class folders (pok/pork, beef, horse), "
            f"but found {len(names)}: {names}"
        )

    if len(pork_matches) != 1:
        raise ValueError(
            f"Could not uniquely identify the pork folder. Found: {names}. "
            "Rename the pork folder to 'pok' or 'pork'."
        )

    return names, pork_matches[0]


def load_datasets():
    class_names, pork_index = get_dataset_class_names(DATA_DIR)

    train_raw = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.20,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True
    )

    val_raw = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.20,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    # Confirm Keras directory order matches the folder names.
    class_names = train_raw.class_names
    normalized = [name.strip().lower() for name in class_names]
    pork_matches = [i for i, name in enumerate(normalized)
                    if name in PORK_FOLDER_NAMES]

    if len(pork_matches) != 1:
        raise ValueError(f"Could not identify pork class in {class_names}")

    pork_index = pork_matches[0]

    def convert_labels(images, labels):
        binary_labels = tf.where(
            tf.equal(labels, pork_index), 0, 1
        )
        return images, tf.cast(binary_labels, tf.float32)

    train_ds = train_raw.map(
        convert_labels, num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    val_ds = val_raw.map(
        convert_labels, num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names, pork_index


def build_model():
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10)
    ], name="data_augmentation")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model


def predict_pil_image(model, pil_image):
    image = pil_image.convert("RGB").resize(IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)

    non_pork_probability = float(model.predict(array, verbose=0)[0][0])
    pork_probability = 1.0 - non_pork_probability

    if non_pork_probability >= THRESHOLD:
        label = "NON-PORK"
        confidence = non_pork_probability
    else:
        label = "PORK"
        confidence = pork_probability

    return {
        "label": label,
        "confidence": confidence,
        "pork_probability": pork_probability,
        "non_pork_probability": non_pork_probability
    }


@st.cache_resource
def load_saved_model(model_path, modified_time):
    # modified_time invalidates the cache when the model file changes.
    del modified_time
    return tf.keras.models.load_model(model_path)


def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_saved_model(MODEL_PATH, os.path.getmtime(MODEL_PATH))


def collect_evaluation(model, dataset):
    y_true, y_prob = [], []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0).reshape(-1)
        y_prob.extend(probabilities.tolist())
        y_true.extend(labels.numpy().astype(int).tolist())

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    return y_true, y_prob, y_pred


# ---------------------------- HEADER ------------------------

st.markdown(
    '<div class="main-title">🥩 PORK vs NON-PORK CLASSIFICATION SYSTEM</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">MobileNetV2 · Transfer Learning · Binary Image Classification</div>',
    unsafe_allow_html=True
)

# --------------------------- SIDEBAR ------------------------

with st.sidebar:
    st.title("🥩 Meat Classifier")
    st.caption("Navigation")
    page = st.radio(
        "Select module",
        [
            "Dashboard",
            "Model Training",
            "Performance Visualization",
            "Test Meat Image",
            "Model Evaluation"
        ]
    )
    st.divider()
    st.markdown("**Classification mapping**")
    st.markdown("- `pok` / `pork` → PORK")
    st.markdown("- `beef` → NON-PORK")
    st.markdown("- `horse` → NON-PORK")
    st.divider()
    st.caption("Model: MobileNetV2")
    st.caption("Input: 224 × 224 RGB")

# -------------------------- DASHBOARD -----------------------

if page == "Dashboard":
    st.subheader("System Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Original classes", "3")
    c2.metric("Output classes", "2")
    c3.metric("Image input", "224 × 224")

    st.markdown("### Classification workflow")
    left, right = st.columns(2)
    with left:
        st.success("🐷 PORK")
        st.write("Images from the `pok` or `pork` folder.")
    with right:
        st.info("🥩 NON-PORK")
        st.write("Images from the `beef` and `horse` folders.")

    st.markdown("### Dataset folder")
    st.code(
        "PorkNPork/\n"
        "├── streamlit_app.py\n"
        "├── pork_classifier.keras  (created after training)\n"
        "└── meat_dataset/\n"
        "    └── train/\n"
        "        ├── pok/\n"
        "        ├── beef/\n"
        "        └── horse/",
        language="text"
    )

    if os.path.isdir(DATA_DIR):
        try:
            folder_names, pork_index = get_dataset_class_names(DATA_DIR)
            st.success(f"Dataset folders detected: {', '.join(folder_names)}")
            st.write(f"Pork folder detected: **{folder_names[pork_index]}**")
            counts = {}
            for folder in folder_names:
                folder_path = os.path.join(DATA_DIR, folder)
                counts[folder] = sum(
                    1 for f in os.listdir(folder_path)
                    if os.path.isfile(os.path.join(folder_path, f))
                    and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
                )
            st.markdown("### Image count by folder")
            st.bar_chart(counts)
            st.dataframe(
                [{"Folder": name,
                  "Binary category": "PORK" if i == pork_index else "NON-PORK",
                  "Images": counts[name]}
                 for i, name in enumerate(folder_names)],
                use_container_width=True,
                hide_index=True
            )
        except Exception as exc:
            st.warning(str(exc))
    else:
        st.warning(f"Dataset directory not found: `{DATA_DIR}`")

    if os.path.exists(MODEL_PATH):
        st.success("A trained model file is available.")
    else:
        st.warning("No trained model found yet. Open Model Training to train it.")

# ------------------------- MODEL TRAINING -------------------

elif page == "Model Training":
    st.subheader("Model Training")
    st.write(
        "Train the binary classifier using your three original folders. "
        "Pork is label 0; beef and horse are label 1."
    )

    epochs = st.slider("Number of training epochs", 1, 100, DEFAULT_EPOCHS)
    start_training = st.button(
        "🚀 Start Training",
        type="primary",
        use_container_width=True
    )

    if start_training:
        try:
            with st.spinner("Loading and preparing dataset..."):
                train_ds, val_ds, class_names, pork_index = load_datasets()

            st.write("**Detected folders:**", ", ".join(class_names))
            st.write(
                f"**Mapping:** {class_names[pork_index]} = PORK; "
                "remaining folders = NON-PORK"
            )

            model = build_model()
            progress_bar = st.progress(0)
            status_text = st.empty()
            chart_placeholder = st.empty()
            epoch_logs = []

            class StreamlitProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    row = {"epoch": epoch + 1}
                    row.update({
                        key: float(value)
                        for key, value in logs.items()
                        if value is not None
                    })
                    epoch_logs.append(row)
                    progress_bar.progress(
                        min((epoch + 1) / epochs, 1.0)
                    )
                    status_text.write(
                        f"Epoch {epoch + 1}/{epochs} · "
                        f"Training accuracy: {logs.get('accuracy', 0):.4f} · "
                        f"Validation accuracy: {logs.get('val_accuracy', 0):.4f}"
                    )
                    if len(epoch_logs) > 0:
                        chart_placeholder.line_chart(
                            {
                                "Training accuracy": [
                                    r.get("accuracy", np.nan) for r in epoch_logs
                                ],
                                "Validation accuracy": [
                                    r.get("val_accuracy", np.nan) for r in epoch_logs
                                ]
                            }
                        )

            callbacks_list = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=7,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=3,
                    min_lr=0.000001,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    MODEL_PATH,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1
                ),
                StreamlitProgressCallback()
            ]

            with st.spinner("Training MobileNetV2. Please wait..."):
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    callbacks=callbacks_list,
                    verbose=0
                )

            # Load the best checkpoint, not simply the last epoch.
            best_model = tf.keras.models.load_model(MODEL_PATH)
            best_model.save(MODEL_PATH)
            load_saved_model.clear()

            st.session_state.training_history = history.history
            st.session_state.training_completed = True
            st.session_state.training_error = None

            st.success("Training completed. Best model saved successfully.")
            st.metric(
                "Best validation accuracy",
                f"{max(history.history.get('val_accuracy', [0])) * 100:.2f}%"
            )
            st.info(f"Model saved as `{MODEL_PATH}`")

        except Exception as exc:
            st.session_state.training_error = str(exc)
            st.error(f"Training failed: {exc}")

    if st.session_state.training_history:
        hist = st.session_state.training_history
        st.markdown("### Training history")
        if "accuracy" in hist and "val_accuracy" in hist:
            st.line_chart({
                "Training accuracy": hist["accuracy"],
                "Validation accuracy": hist["val_accuracy"]
            })
        if "loss" in hist and "val_loss" in hist:
            st.line_chart({
                "Training loss": hist["loss"],
                "Validation loss": hist["val_loss"]
            })

# -------------------- PERFORMANCE VISUALIZATION -------------

elif page == "Performance Visualization":
    st.subheader("Training Performance Visualization")
    hist = st.session_state.training_history

    if not hist:
        st.info("Train the model first. Training accuracy and loss graphs will appear here.")
    else:
        epochs_range = range(1, len(hist.get("accuracy", [])) + 1)
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(epochs_range, hist.get("accuracy", []), label="Training")
            ax.plot(epochs_range, hist.get("val_accuracy", []), label="Validation")
            ax.set_title("Model Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(epochs_range, hist.get("loss", []), label="Training")
            ax.plot(epochs_range, hist.get("val_loss", []), label="Validation")
            ax.set_title("Model Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        summary = {
            "Metric": ["Final training accuracy", "Final validation accuracy",
                       "Final training loss", "Final validation loss"],
            "Value": [
                hist.get("accuracy", [0])[-1],
                hist.get("val_accuracy", [0])[-1],
                hist.get("loss", [0])[-1],
                hist.get("val_loss", [0])[-1]
            ]
        }
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.caption(
            "These graphs show the training run stored in this Streamlit session. "
            "They are not automatically reconstructed from an older model file."
        )

# ------------------------- IMAGE TESTING --------------------

elif page == "Test Meat Image":
    st.subheader("Test an Image")
    st.write("Upload a meat image to obtain a PORK or NON-PORK prediction.")

    model = get_model()
    if model is None:
        st.warning("No trained model found. Train the model first.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a meat image",
        type=["jpg", "jpeg", "png"],
        key="meat_image_upload"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded image", width=450)

            if st.button("🔍 Classify Image", type="primary",
                         use_container_width=True):
                with st.spinner("Analyzing image..."):
                    result = predict_pil_image(model, image)

                st.markdown("### Classification result")
                if result["label"] == "PORK":
                    st.markdown(
                        '<div class="result-pork">🐷 PORK</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-nonpork">🥩 NON-PORK</div>',
                        unsafe_allow_html=True
                    )

                st.metric("Prediction confidence", f"{result['confidence'] * 100:.2f}%")
                st.progress(float(result["confidence"]))

                with st.expander("View probability details"):
                    st.write(f"PORK probability: {result['pork_probability'] * 100:.2f}%")
                    st.write(f"NON-PORK probability: {result['non_pork_probability'] * 100:.2f}%")

                st.session_state.prediction_history.insert(0, {
                    "Filename": uploaded_file.name,
                    "Prediction": result["label"],
                    "Confidence": f"{result['confidence'] * 100:.2f}%"
                })
                st.success("Image testing completed.")

        except Exception as exc:
            st.error(f"Could not process image: {exc}")

    if st.session_state.prediction_history:
        st.markdown("### Recent predictions")
        st.dataframe(
            st.session_state.prediction_history,
            use_container_width=True,
            hide_index=True
        )

# ------------------------ MODEL EVALUATION ------------------

elif page == "Model Evaluation":
    st.subheader("Model Evaluation")
    st.write(
        "Evaluate the saved model using the validation split from the dataset. "
        "The validation data is not an independent external test set."
    )

    model = get_model()
    if model is None:
        st.warning("No trained model found. Train the model first.")
        st.stop()

    if st.button("📊 Run Evaluation", type="primary"):
        try:
            with st.spinner("Preparing validation dataset and evaluating..."):
                _, val_ds, class_names, pork_index = load_datasets()
                y_true, y_prob, y_pred = collect_evaluation(model, val_ds)

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                try:
                    auc = roc_auc_score(y_true, y_prob)
                except ValueError:
                    auc = None

                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                report = classification_report(
                    y_true, y_pred,
                    labels=[0, 1],
                    target_names=["PORK", "NON-PORK"],
                    output_dict=True,
                    zero_division=0
                )

                st.session_state.last_evaluation = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "auc": auc,
                    "cm": cm.tolist(),
                    "report": report
                }

        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    evaluation = st.session_state.last_evaluation
    if evaluation:
        st.markdown("### Evaluation metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{evaluation['accuracy'] * 100:.2f}%")
        m2.metric("Precision", f"{evaluation['precision'] * 100:.2f}%")
        m3.metric("Recall", f"{evaluation['recall'] * 100:.2f}%")
        m4.metric("F1-score", f"{evaluation['f1'] * 100:.2f}%")
        m5.metric(
            "AUC",
            "N/A" if evaluation["auc"] is None
            else f"{evaluation['auc']:.4f}"
        )

        st.markdown("### Confusion matrix")
        cm = np.asarray(evaluation["cm"])
        fig, ax = plt.subplots(figsize=(6, 4))
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["PORK", "NON-PORK"]
        )
        display.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title("Validation Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Classification report")
        report_rows = []
        for label in ["PORK", "NON-PORK"]:
            row = evaluation["report"][label]
            report_rows.append({
                "Class": label,
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1-score": row["f1-score"],
                "Support": int(row["support"])
            })
        st.dataframe(report_rows, use_container_width=True, hide_index=True)

        csv_text = (
            "Actual label,Predicted label\n" +
            "\n".join(
                f"{'NON-PORK' if actual else 'PORK'},"
                f"{'NON-PORK' if predicted else 'PORK'}"
                for actual, predicted in zip(
                    # Recompute only if user requests a fresh evaluation.
                    [], []
                )
            )
        )
        st.caption(
            "Positive class for precision/recall/F1 is NON-PORK (label 1). "
            "The confusion matrix uses PORK=0 and NON-PORK=1."
        )

# ----------------------------- FOOTER -----------------------

st.divider()
st.caption(
    "Pork vs Non-Pork Image Classification System | "
    "TensorFlow · MobileNetV2 · Streamlit"
)
