<h1 align="center">MaGRoad-Prompt</h1>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Frontend: React](https://img.shields.io/badge/Frontend-React-blue.svg)](https://reactjs.org/)
[![Backend: FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

</div>

<div align="center">

[**[ArXiv Paper]**](#) | [**[Automated Extraction]**](../MaGRoad-main/README.md) | [**[WildRoad Dataset]**](#)

</div>

## 📖 Introduction

**MaGRoad-Prompt** is the **first interactive road network extraction algorithm and tool**, designed to bridge the gap between automated extraction and human expertise. It serves as an Interactive Road Network Annotation Tool based on MaGRoad, providing a user-friendly application for generating precise road masks and graph structures via interactive prompting.

<div align="center">
  <img src="assets/arc.png" width="100%" alt="MaGRoad-Prompt Architecture"/>
  <br/>
  <em>Figure 1: The architecture of the interactive algorithm based on MaGRoad. If you are interested in MaGRoad, please refer to the <a href="https://github.com/xiaofei-guan/MaGRoad">MaGRoad</a> code.</em>
</div>

## 💻 Interactive Annotation Tool

Our web-based application transforms the annotation workflow. Below is the main interface of the tool:

<div align="center">
  <img src="assets/ui.png" width="100%" alt="Annotation Tool UI"/>
</div>

### Efficient Workflow vs. Manual Annotation

Compared to traditional manual plotting (e.g., QGIS), MaGRoad-Prompt significantly reduces effort by intelligently inferring path connectivity from sparse user clicks.

| **Manual Annotation (QGIS)** | **Interactive Annotation (Ours)** |
|:----------------------------:|:---------------------------------:|
| <img src="assets/qgis_manual.gif" width="100%" alt="Manual"/> | <img src="assets/interactive.gif" width="100%" alt="Interactive"/> |

### Diverse Scenarios

**Satellite Imagery**: The model handles complex large-scale road networks effectively.

<div align="center">
  <img src="assets/sat2.gif" width="49%" alt="Satellite Demo 1"/>
  <img src="assets/sat4.gif" width="49%" alt="Satellite Demo 2"/>
</div>

**UAV Generalization**: Although trained only on satellite imagery, the model demonstrates strong generalization capabilities on UAV images.

<div align="center">
  <img src="assets/uav1.gif" width="32%" alt="UAV Demo 1"/>
  <img src="assets/uav2.gif" width="32%" alt="UAV Demo 2"/>
  <img src="assets/uav3.gif" width="32%" alt="UAV Demo 3"/>
</div>

---

## 🚀 Usage

Follow these steps to set up and run the annotation tool.

### 1. Environment Setup

```bash
conda create -n magroad-prompt python=3.8
conda activate magroad-prompt
```

**Backend (Algorithm)**
```bash
cd app/backend
pip install -r requirements.txt
```

**Frontend (Interface)**
Ensure Node.js is installed.
```bash
cd app/frontend
npm install # npm==10.8.2
```

### 2. Model Preparation

Download the provided open-source checkpoint from [**here**](https://drive.google.com/drive/folders/1Z9lJIhmhaQBLOzN2WlfwUYLHPUaFH4EZ?usp=sharing) and place it under `./app/backend/app/models/weights`:

```bash
mkdir -p app/backend/app/models/weights
# Save the downloaded checkpoint here
```


### 3. Start the Application

**Start Backend**
```bash
cd app/backend
uvicorn main:app --reload --port 8000
```

**Start Frontend**
```bash
cd app/frontend
npm run dev
```

Open your browser (typically `http://localhost:5173`) to start using the tool.

---

## 🧠 Interactive Algorithm (Training)

If you wish to train the interactive algorithm on your own data or the WildRoad dataset, follow the steps below.

### 1. Data Preparation
Generate masks for the WildRoad dataset:
```bash
python wildroad/generate_masks_for_wild_road.py --input_dir ./wildroad/wild_road --output_dir ./wildroad/wild_road_mask
```

### 2. Model Preparation

Download the **ViT-B** checkpoint from the [SAM repository](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it under `ckpt`:

```bash
mkdir -p ckpt
# Save sam_vit_b_01ec64.pth here
```

### 3. Training
Train the model to understand path connectivity from sparse prompts.
```bash
python train.py --config=config/toponet_vitb_1024_wild_road.yaml
```

### 4. Testing & Configuration Update
Evaluate the model to find the optimal thresholds.
```bash
python test.py --config=config/toponet_vitb_1024_wild_road.yaml --checkpoint=<your_ckpt_path>
```

**Integrating Custom Models into the Tool:**
After training, update the backend configuration to use your new model:
1.  Copy your checkpoint to `app/backend/app/models/weights/`.
2.  Edit `app/backend/app/models/road_extraction_algorithm/config_road_extraction.yaml`.
3.  Update `ROAD_EXTRACTION_MODEL_CKPT_PATH` with your checkpoint filename.
4.  Update `ITSC_THRESHOLD`, `ROAD_THRESHOLD`, and `TOPO_THRESHOLD` with the values obtained from the test step.

## 📍 Citation

If you use MaGRoad-Prompt or the WildRoad dataset in your research, please cite:

```bibtex
@article{magroad2025,
  title={Beyond Endpoints: Path-Centric Reasoning for Vectorized Off-Road Network Extraction},
  author={Guan, Wenfei and Mei, Jilin and Shen, Tong and Wu, Xumin and Wang, Shuo and Min, Cheng and Hu, Yu},
  journal={arXiv preprint},
  year={2025}
}
```
