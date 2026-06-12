# 🅿️ Parking Space Occupancy Detector

A production-ready computer vision system that detects and classifies parking spaces
as **free**, **occupied**, or **partially occupied** using a fine-tuned YOLOv8 model
and a Streamlit web interface.

---

## 🌍 Real-World Use Cases

- Smart city parking management systems
- Shopping mall / airport parking guidance
- Automated parking fee calculation
- Real-time availability dashboards for drivers

---

## 🛠️ Tech Stack

| Component    | Library                  |
|--------------|--------------------------|
| Detection    | Ultralytics YOLOv8       |
| Image proc.  | OpenCV, NumPy, Pillow    |
| Web UI       | Streamlit                |
| Config       | PyYAML                   |
| Deep learning| PyTorch                  |

---

## ⚙️ Installation

```bash
# 1. (Recommended) Create a conda environment
conda create -n parking_detector python=3.10 -y
conda activate parking_detector

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Prepare dataset
Place your Roboflow / YOLO-format dataset inside the `dataset/` folder:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Step 2 — Train the model
```bash
python train.py
```
This will:
- Read `dataset/data.yaml` for class configuration
- Fine-tune `yolov8n.pt` on your parking dataset (50 epochs)
- Auto-save the best weights to `models/best.pt`

### Step 3 — Run the app
```bash
streamlit run app.py
```
Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 📁 Project Structure

```
parking_occupancy_detector/
├── app.py                   ← Streamlit web app
├── train.py                 ← YOLOv8 training script
├── requirements.txt
├── README.md
├── models/
│   └── best.pt              ← saved after training
├── dataset/
│   ├── train/
│   ├── valid/
│   └── data.yaml
└── utils/
    └── detection_utils.py   ← load_model, run_detection, draw_results, get_summary
```

---

## 🏷️ Class Descriptions

| Class ID | Name                  | Color  | Description                              |
|----------|-----------------------|--------|------------------------------------------|
| 0        | free                  | 🟢 Green  | Parking space is empty and available   |
| 1        | occupied              | 🔴 Red    | Parking space is fully occupied        |
| 2        | partially_occupied    | 🟡 Yellow | Space is partially blocked             |

Color logic uses **flexible string matching** — class names like `empty`, `free`,
`partial`, `partly-occupied` etc. all work correctly.

---

## 🔧 Common Issues & Fixes

### "Model not found" error
```
Model file not found at 'models/best.pt'
```
→ Run `python train.py` first. Training must complete before the app can run.

### "0 detections" on upload
→ Lower the **Confidence Threshold** slider in the sidebar (try 0.20–0.30).
→ Make sure your image shows parking spaces from an aerial or angled view.

### CUDA out of memory during training
→ Open `train.py` and reduce `BATCH = 16` to `BATCH = 8` or `BATCH = 4`.

### Training is slow
→ The script auto-selects GPU if available. Install CUDA-enabled PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### App shows COCO classes (person, car, …) instead of parking classes
→ The model in `models/best.pt` is a pretrained base model, not fine-tuned yet.
→ Run `python train.py` to produce a properly fine-tuned `best.pt`.

---

## 📦 Dataset Credit

Dataset prepared using [Roboflow](https://roboflow.com/) in YOLOv8 format.
Classes: `free`, `occupied`, `partially_occupied`.
