# Mini Deep Learning Workflow

This is a simple project, that replicates the usual workflow of setting up, training, and performing inference on deep learning models using PyTorch.

## Project structure
```
mini-dl-workflow/
├── config/
│   ├── config.yaml          # Configuration file for training
├── data/                    # Dataset directory
├── notebooks/               # Jupyter notebooks (optional)
├── runs/                    # Model training runs & logs
├── scripts/                 # Executable scripts
│   ├── collect_images.py    # Capture dataset images
│   ├── infer.py             # Inference script
│   ├── train.py             # Training script
│   ├── vis_fmaps.py         # Visualizing feature maps
├── src/
│   ├── mdlw/                # Core framework
│   │   ├── utils/           # Utility functions
│   │   ├── augment.py       # Augmentations and transformations
│   │   ├── dataset.py       # Dataset management
│   │   ├── engine.py        # Training, validation, exporting
│   │   ├── inference.py     # Model inference handling
│   │   ├── model.py         # Model definitions
├── download_dataset.py      # Dataset download script
├── README.md
├── setup.py                 # Installation setup script
```

---

## Setup

### **1. Download the project**
#### **Linux / Mac**
```bash
curl -L -o project.zip https://github.com/ugniusalekna/mini-dl-workflow/archive/refs/heads/main.zip
unzip project.zip && rm project.zip
mv mini-dl-workflow-main your_project_name
```
#### **Windows**
```powershell
Invoke-WebRequest -Uri https://github.com/ugniusalekna/mini-dl-workflow/archive/refs/heads/main.zip -OutFile "./project.zip"
Expand-Archive -Path project.zip -DestinationPath .
mv mini-dl-workflow-main your_project_name

# If received 'script execution is disabled' warning, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# and try downloading the project again.
```

### **2️. Install project, dependencies & PyTorch**
```bash
cd your_project_name
python -m venv .venv
```
#### **Activate virtual environment**
- **Linux / Mac**:
  ```bash
  source .venv/bin/activate
  ```
- **Windows**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

### **Install project & dependencies**

- **Project with required dependencies**:
  ```bash
  pip install -e .
  ```

- **Manual PyTorch install (only required for Linux / Windows users)**:
    - **Linux / Windows (CUDA GPU)**:
        ```bash
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
        ```
        > **Note:** This installs PyTorch with CUDA support, but **does not install the full CUDA toolkit** required for development. If CUDA is not installed on your system, download it from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and ensure it matches the version of PyTorch you are installing.
        > To check if PyTorch detects CUDA, run:
        > ```bash
        > python -c "import torch; print(torch.cuda.is_available())"
        > ```
        > If this returns `False`, ensure CUDA and NVIDIA drivers are correctly installed.
    - **Linux / Windows (CPU-only)**:
      ```bash
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      ```
    
    - **If your system is not listed, refer to the official PyTorch installation guide:**
      [PyTorch Get Started](https://pytorch.org/get-started/)


#### **Verify installation**
- **Verify PyTorch installation**:
  ```bash
  python -c "import torch; print(torch.__version__)"
  ```
- **Verify project installation**:
  ```bash
  python -c "import sys; print('\n'.join(sys.path))"
  ```
  Ensure that:
  ```
  /path/to/project/your_project_name/src
  ```
  is listed in the output paths.

> **Note:** ONNX will not be installed for Python 3.13+ as there are no available wheels on PyPI.

---

## Running scripts

### **Training a model**

```bash
python scripts/train.py
```

To monitor training, open a **separate terminal session** and run:

```bash
tensorboard --logdir=./runs
```

Then, navigate to `http://localhost:6006` to visualize logs.

### **Performing inference**

Inference can be performed using either an **ONNX (.onnx) or PyTorch (.pt) model** by modifying the `--model_path` argument accordingly.

```bash
python scripts/infer.py --model_path ./runs/run_1/best_model.onnx --config_path ./runs/run_1/args.yaml
```

or

```bash
python scripts/infer.py --model_path ./runs/run_1/best_model.pt --config_path ./runs/run_1/args.yaml
```

To use a **drawing canvas** instead of a webcam, set `--mode` argument to `draw`:

```bash
python scripts/infer.py --model_path ./runs/run_1/best_model.onnx --config_path ./runs/run_1/args.yaml --mode draw
```

#### **Key controls in inference mode**

- **For webcam inference:**
  - 'q': Quit
- **For drawing mode:**
  - 'q': Quit
  - 'c': Clear Canvas
  - 'w/s': Adjust Brush Size
  - 'e': Toggle Eraser

### **Visualizing model feature maps**

**Note:** Only PyTorch (.pt) models can be used for feature map visualization.

```bash
python scripts/vis_fmaps.py --model_path ./runs/run_1/best_model.pt --config_path ./runs/run_1/args.yaml
```

#### **Feature map controls**

- 'j' / 'l': Navigate through model layers
- 'k': Toggle activation visualization
- 'space': Freeze visualization
- 'q': Quit

### **Collecting custom dataset**

For example, to create a dataset for recognizing hand signs such as **rock**, **paper**, and **scissors**, you may capture images as such:
1. Run the script with a specified label and dataset directory:
   ```bash
   python scripts/collect_images.py --output_dir ./data/RockPaperScissors --label rock
   ```
   - `--output_dir` specifies where the dataset will be stored.
   - `--label` is the category name (e.g., 'rock', 'paper', 'scissors').
2. The webcam window will open, allowing you to capture images.
3. Press the corresponding keys to interact:
- 'c': Capture an image
- 'q': Quit data collection

---