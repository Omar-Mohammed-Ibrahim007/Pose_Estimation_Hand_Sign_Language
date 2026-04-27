import gradio as gr
import numpy as np
import cv2
import mediapipe as mp
import joblib
import os

# -----------------------------
# LOAD MODELS
# -----------------------------
MODEL_PATH = "./Pose_Estimation/workspace"

trained = open(os.path.join(MODEL_PATH, "Best_model.txt")).read().strip()

model_pipeline = joblib.load(f'./Pose_Estimation/models/{trained}.pkl')
model = model_pipeline[1]

scaler =  model_pipeline[0]

# -----------------------------
# MEDIAPIPE TASKS API
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# -----------------------------
# CONSTANTS
# -----------------------------
LM_SIZE = 21 * 3
FEATURE_SIZE = LM_SIZE
TIME_STEP = 33  # microseconds

# -----------------------------
# FEATURE EXTRACTION (FIXED)
# -----------------------------
def extract_features_from_frame(frame, landmarker, timestamp):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(mp_image, int(timestamp))

    # ❗ HARD FIX: reject empty frames completely
    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]

    lm_list = []
    for lm in hand:
        lm_list.extend([lm.x, lm.y, lm.z])

    return np.array(lm_list, dtype=np.float32)


# -----------------------------
# IMAGE PREDICTION
# -----------------------------
def predict_image(image):

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    landmarker = HandLandmarker.create_from_options(options)

    features = extract_features_from_frame(frame, landmarker, 0)

    if features is None:
        return "No hand detected"

    features = scaler.transform([features])
    pred = model.predict(features)[0]

    return f"Prediction: {pred}"


# -----------------------------
# VIDEO PREDICTION (FIXED LOGIC)
# -----------------------------
def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)
    landmarker = HandLandmarker.create_from_options(options)

    preds = []
    timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        features = extract_features_from_frame(frame, landmarker, timestamp)
        timestamp += TIME_STEP

        # ❗ skip invalid frames (VERY IMPORTANT FIX)
        if features is None:
            continue

        # scale only valid data
        features = scaler.transform([features])
        pred = model.predict(features)[0]

        preds.append(pred)

    cap.release()

    if len(preds) == 0:
        return "No hand detected"

    # stable majority vote
    final = max(set(preds), key=preds.count)

    return f"Final Prediction: {final}"

# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks() as app:

    gr.Markdown("# 🤟 Hand Sign Recognition System (FIXED PIPELINE)")

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