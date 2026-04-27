<!DOCTYPE html>

<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Hand Sign Recognition System</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            margin: 0;
            padding: 0;
        }

    header {
            background: #1e293b;
            padding: 20px;
            text-align: center;
        }

    h1 {
            color: #38bdf8;
        }

    .container {
            padding: 20px;
            max-width: 1000px;
            margin: auto;
        }

    .card {
            background: #1e293b;
            padding: 15px;
            margin: 15px 0;
            border-radius: 10px;
        }

    code, pre {
            background: #0f172a;
            padding: 10px;
            border-radius: 5px;
            color: #facc15;
            display: block;
            overflow-x: auto;
        }

    ul {
            line-height: 1.8;
        }

    .tag {
            display: inline-block;
            background: #334155;
            padding: 5px 10px;
            margin: 3px;
            border-radius: 5px;
            font-size: 13px;
        }

    .badges img {
            margin: 3px;
        }</style>

</head>

<body>

<header>
    <h1>Hand Sign Recognition System</h1>
    <p>MediaPipe + Machine Learning + OpenCV + Gradio</p>

    <div class="badges">
        <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge">
        <img src="https://img.shields.io/badge/OpenCV-CV-green?style=for-the-badge">
        <img src="https://img.shields.io/badge/MediaPipe-Hands-orange?style=for-the-badge">
        <img src="https://img.shields.io/badge/ML-Scikit--Learn-red?style=for-the-badge">
    </div>
</header>

<div class="container">

    <div class="card">
        <h2>Project Overview</h2>
        <p>
             This project is a real-time hand sign recognition system using <b>MediaPipe Hands</b>,
            feature extraction, and multiple machine learning classifiers.
            It supports image, video, and real-time prediction with Gradio.
        </p>
    </div>
    <div class="card">
        <h2> Features</h2>
        <ul>
            <li>✔ Image prediction</li>
            <li>✔ Video prediction</li>
            <li>✔ Real-time hand tracking</li>
            <li>✔ Majority voting for stability</li>
        </ul>
    </div>
    <div class="card">
        <h2>Tech Stack</h2>
        <span class="tag">Python
        <span class="tag">OpenCV
        <span class="tag">MediaPipe
        <span class="tag">Scikit-Learn
        <span class="tag">Gradio
        <span class="tag">NumPy
        <span class="tag">Joblib
    </div>

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
Dataset/
 ├── *.mp4
README.md
</pre>

    </div>
    

    <div class="card">
        <h2>Machine Learning Pipeline</h2>
        <ul>
            <li>MediaPipe extracts 21 hand landmarks</li>
            <li>Each landmark → (x, y) → 42 features</li>
            <li>Features scaled using StandardScaler</li>
            <li>Models: SVM, RF, MLP, LDA, ..etc</li>
        </ul>
    </div>

    <div class="card">
        <h2>Run Project</h2>
        <pre><code>python app.py</code></pre>
    </div>

  <div class="card">
        <h2> Author</h2>
        <p>Omar Mohammed</p>
        <p>Computer Vision & Machine Learning Project</p>
    </div>

    <div class="card">
        <h2> Notes</h2>
        <ul>
            <li>MediaPipe VIDEO mode needs increasing timestamps</li>
            <li>Dataset imbalance affects accuracy</li>
            <li>LDA performed best overall</li>
        </ul>
    </div>
</div>


</body>
</html><!DOCTYPE html>
