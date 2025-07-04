Here's a polished and informative README for your repository:

---

# PCB Defect Classification with LLM Captioning

A lightweight pipeline combining computer vision with large language model (LLM) captioning for visual inspection of PCB defects.

## 🚀 Pipeline Overview

1. **Defect Detection**
   Uses a YOLO-based model (`test_trained_model`) to detect the most likely PCB defect class and output a confidence score.

2. **LLM Captioning**
   Passes the predicted defect class and score to an LLM (via `generate_response_openai`) that generates a concise, factual caption describing the defect.

---
### 🧩 Project Structure
pcb-defect_classification_with-llm_captioning/
├── pipeline.py              # Orchestrates classification + captioning
├── LLM.py                   # Handles LLM API interaction
├── yolo_pred.py             # YOLOv5 classification model loading and inference
├── test_images/             # Sample input images (add your own)
├── runs/
│   └── classify/
│       └── pcb_defects_v1/
│           └── weights/
│               └── best.pt  # Trained YOLO model weights
├── requirements.txt         # Python dependencies (if added)
└── README.md                # Project documentation

---

## 🧩 File Structure

* `pipeline.py`
  Main integration script. Defines `get_result(image_path)` that:

  1. Runs YOLO-based defect classification
  2. Sends prediction + confidence to an LLM for captioning
  3. Returns the generated descriptive caption

* `LLM.py`
  Handles LLM API calls (e.g., to OpenAI). Must define `generate_response_openai(image_path, prompt)` returning model response.

* `yolo_pred.py`
  Contains `test_trained_model(model_path, image_path)` which loads a YOLO model and predicts the defect class + confidence.

---

## 🔧 Usage

```python
from pipeline import get_result

caption = get_result("path/to/pcb_sample.jpg")
print(caption)
```

This outputs a descriptive caption of the defect detected in the given PCB image.

---

## 🛠️ Setup & Requirements

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Ensure YOLO model weights are present at:

   ```
   /runs/classify/pcb_defects_v1/weights/best.pt
   ```
3. Configure access to your chosen LLM (e.g., set OpenAI API key).

---

## ☑️ Customization

* **Switch LLMs or APIs**
  Modify `generate_response_openai()` in `LLM.py` to connect to GPT-4, Claude, or your preferred LLM backend.

* **Use Different Detection Models**
  Replace the model path and detection logic in `yolo_pred.py` to switch models/datasets.

* **Modify Prompt Style**
  Tweak the prompt template in `pipeline.py` to generate different caption tones or lengths.

---

