import gradio as gr
import numpy as np
import cv2
import mediapipe as mp
import joblib
import os

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "./Pose_Estimation/workspace"

trained = open(os.path.join(MODEL_PATH, "Best_model.txt")).read().strip()
model_pipeline = joblib.load(f'./Pose_Estimation/models/{trained}.pkl')

scaler = model_pipeline[0]
model = model_pipeline[1]

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# IMAGE mode (NO timestamp needed)
image_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# VIDEO mode (timestamp required)
video_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# -----------------------------
# CONSTANTS
# -----------------------------
LM_SIZE = 21 * 2
TIME_STEP = 33000  # microseconds

# -----------------------------
# SAFE IMAGE CONVERSION
# -----------------------------
def safe_convert(image):
    if image is None:
        return None

    frame = np.array(image)

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame

# -----------------------------
# FEATURE EXTRACTION (IMAGE)
# -----------------------------
def extract_features_image(frame, landmarker):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]

    lm_list = []
    for lm in hand:
        lm_list.extend([lm.x, lm.y])

    return np.array(lm_list, dtype=np.float32)

# -----------------------------
# FEATURE EXTRACTION (VIDEO)
# -----------------------------
def extract_features_video(frame, landmarker, timestamp):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(mp_image, int(timestamp))

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]

    lm_list = []
    for lm in hand:
        lm_list.extend([lm.x, lm.y])

    return np.array(lm_list, dtype=np.float32)

# -----------------------------
# IMAGE PREDICTION
# -----------------------------
def predict_image(image):

    frame = safe_convert(image)

    if frame is None:
        return "No image"

    # NEW landmarker (IMAGE MODE)
    landmarker = HandLandmarker.create_from_options(image_options)

    features = extract_features_image(frame, landmarker)

    if features is None:
        return "No hand detected"

    features = scaler.transform([features])
    pred = model.predict(features)[0]

    return f"Prediction: {pred}"

# -----------------------------
# VIDEO PREDICTION
# -----------------------------
def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    # NEW landmarker (VIDEO MODE)
    landmarker = HandLandmarker.create_from_options(video_options)

    preds = []
    timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        features = extract_features_video(frame, landmarker, timestamp)
        timestamp += TIME_STEP

        if features is None:
            continue

        features = scaler.transform([features])
        pred = model.predict(features)[0]

        preds.append(pred)

    cap.release()

    if len(preds) == 0:
        return "No hand detected"

    final = max(set(preds), key=preds.count)

    return f"Final Prediction: {final}"

# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks() as app:

    gr.Markdown("# 🤟 Hand Sign Recognition System")

    with gr.Tab("Image"):
        img = gr.Image(type="pil")
        out1 = gr.Textbox()
        btn1 = gr.Button("Predict")

        btn1.click(predict_image, inputs=img, outputs=out1)

    with gr.Tab("Video Upload"):
        vid = gr.Video()
        out2 = gr.Textbox()
        btn2 = gr.Button("Predict")

        btn2.click(predict_video, inputs=vid, outputs=out2)

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.launch()