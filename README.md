# Deepfake Video & Audio Detector

A prototype application for experimenting with deepfake video and audio analysis using Python, PyTorch, Librosa, OpenCV, Flask, and a React/TypeScript frontend.

## Current status

This repository is a **prototype**, not a validated production deepfake detector.

- The audio pipeline contains a CNN architecture and performs inference **only when trained model weights are supplied**.
- The application no longer fabricates random REAL/FAKE predictions when weights are missing.
- The frontend API URL is configurable through `VITE_API_BASE_URL`.
- The video API currently reports a clear error when the `simple_deepfake_detector` module is not present rather than silently pretending detection is available.
- No accuracy, precision, recall, or F1 score is claimed here because reproducible trained-model evaluation artifacts are not currently included.

## Architecture

```text
React / TypeScript UI
        |
        v
     Flask API
     /       \
 Video       Audio
   |            |
Detector     Mel-spectrogram
module          |
               CNN
                |
          REAL / FAKE
```

## Audio model

The audio model uses:

- Mel-spectrogram preprocessing
- 3 convolutional blocks
- Batch normalization
- Max pooling
- Fully connected classification layers

Expected checkpoint:

```text
models/audio_deepfake_detector.pt
```

or set:

```text
AUDIO_DEEPFAKE_MODEL_PATH=/path/to/model.pt
```

The checkpoint should contain a PyTorch state dictionary compatible with `AudioDeepfakeDetector`.

## Frontend configuration

Set the backend URL with:

```env
VITE_API_BASE_URL=http://localhost:5000
```

The previous hard-coded LAN address has been removed.

## Running the Flask backend

Install the Python dependencies required by the backend, then run:

```bash
python app.py
```

The API starts on:

```text
http://localhost:5000
```

## API

- `GET /` — API status
- `POST /api/upload/video` — upload a video for analysis
- `POST /api/upload/audio` — upload an audio file
- `GET /api/task/<task_id>` — check processing status

## Important limitation

A deepfake detector should not be evaluated from confidence values alone. A proper version of this project should include:

1. A documented training dataset.
2. A reproducible training pipeline.
3. Held-out test data.
4. Accuracy, precision, recall, F1, and ROC-AUC where appropriate.
5. Saved model weights and their version.
6. Tests covering preprocessing and inference.

Until those artifacts are available, results should be treated as experimental.

## Ethical use

Use this project for research, education, and controlled testing. Detection results should not be treated as definitive proof that media is authentic or manipulated.

## License

This project is licensed under the MIT License.
