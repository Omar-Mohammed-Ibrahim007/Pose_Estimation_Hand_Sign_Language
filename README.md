<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Hand Sign Recognition System</title>
</head>

<body>

<style>


    body {
        margin: 0;
        font-family: "Segoe UI", Arial, sans-serif;
        background: #0b1220;
        color: #e5e7eb;
    }

    header {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-bottom: 1px solid #334155;
    }

    h1 {
        color: #38bdf8;
        margin-bottom: 10px;
        font-size: 32px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 16px;
    }

    .badges img {
        margin: 5px;
    }

    .container {
        max-width: 1000px;
        margin: auto;
        padding: 30px 20px;
    }

    .card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    }

    h2 {
        color: #38bdf8;
        margin-top: 0;
    }

    p, li {
        color: #cbd5e1;
        line-height: 1.7;
    }

    code, pre {
        background: #0f172a;
        padding: 12px;
        border-radius: 8px;
        color: #facc15;
        overflow-x: auto;
        display: block;
    }

    .tag {
        display: inline-block;
        background: #1e293b;
        color: #cbd5e1;
        padding: 6px 10px;
        margin: 4px;
        border-radius: 6px;
        font-size: 13px;
        border: 1px solid #334155;
    }

    ul {
        padding-left: 20px;
    }
</style>



<header>
    <h1>Hand Sign Recognition System</h1>
    <div class="subtitle">
        MediaPipe • Machine Learning • OpenCV • Gradio
    </div>

   ![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge)
   ![OpenCV](https://img.shields.io/badge/OpenCV-CV-green?style=for-the-badge)
   ![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange?style=for-the-badge)
   ![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-red?style=for-the-badge)

</header>

<div class="container">

<!-- PROJECT OVERVIEW -->
<div class="card">
    <h2>Project Overview</h2>
    <p>
        A real-time hand sign recognition system using <b>MediaPipe Hands</b>,
        extracting 21 key landmarks and applying multiple machine learning models
        for classification. Supports image, video, and real-time inference.
    </p>
</div>

<!-- FEATURES -->
<div class="card">
    <h2>Features</h2>
    <ul>
        <li>Real-time hand detection using MediaPipe</li>
        <li>Image-based prediction</li>
        <li>Video-based classification</li>
        <li>Stable prediction using majority voting</li>
        <li>ML pipeline with feature scaling</li>
    </ul>
</div>

<!-- TECH STACK -->
<div class="card">
    <h2>Tech Stack</h2>

    <span class="tag">Python</span>
    <span class="tag">OpenCV</span>
    <span class="tag">MediaPipe</span>
    <span class="tag">Scikit-Learn</span>
    <span class="tag">Gradio</span>
    <span class="tag">NumPy</span>
    <span class="tag">Joblib</span>
</div>

<!-- STRUCTURE -->
<div class="card">
    <h2>Project Structure</h2>

<pre>
Pose_Estimation/
 ├── workspace/
 │   ├── Best_model.txt
 │   ├── scaler.pkl
 │
 ├── models/
 │   ├── *.pkl
 │
hand_landmarker.task
app.py
hand_sign_frame_dataset.csv
Dataset/
 ├── *.mp4
README.md
</pre>
</div>

<!-- ML PIPELINE -->
<div class="card">
    <h2>Machine Learning Pipeline</h2>
    <ul>
        <li>MediaPipe extracts 21 hand landmarks</li>
        <li>Each landmark → (x, y) → 42 features</li>
        <li>Features normalized using StandardScaler</li>
        <li>Models: SVM, Random Forest, MLP, LDA, etc.</li>
    </ul>
</div>

<!-- RUN -->
<div class="card">
    <h2>Run Project</h2>

<pre>python app.py</pre>
</div>

<!-- AUTHOR -->
<div class="card">
    <h2>Author</h2>
    <p><b>Omar Mohammed</b></p>
    <p>Computer Vision & Machine Learning Project</p>
</div>

<!-- NOTES -->
<div class="card">
    <h2>Notes</h2>
    <ul>
        <li>MediaPipe VIDEO mode requires increasing timestamps</li>
        <li>Dataset imbalance may affect accuracy</li>
        <li>LDA gave best generalization performance</li>
    </ul>
</div>

</div>

</body>
</html>