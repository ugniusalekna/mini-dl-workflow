import os
import sys
from setuptools import setup, find_packages

python_version = sys.version_info[:2]

is_windows = sys.platform.startswith("win")
is_linux = sys.platform.startswith("linux")
is_mac = sys.platform.startswith("darwin")

install_requires = [
    "inquirer",
    "ipykernel",
    "ipython",
    "matplotlib",
    "numpy",
    "opencv-python",
    "pillow",
    "PyYAML",
    "requests",
    "tensorboard",
    "tqdm",
]

if python_version < (3, 13):
    install_requires.extend(["onnx", "onnxruntime"])

if is_mac:
    install_requires.extend(["torch", "torchvision"])

extras_require = {
    "jupyter": [
        "jupyter",
        "notebook",
        "jupyterlab",
    ]
}

setup(
    name=os.path.basename(os.path.abspath(os.path.dirname(__file__))),
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    extras_require=extras_require,
)


print("\nTorch Installation Instructions:")
if is_linux or is_windows:
    print("  CUDA (NVIDIA GPU): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
    print("  CPU Only: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

elif not (is_linux or is_windows or is_mac):
    print("  Unknown system detected. Install PyTorch manually from https://pytorch.org/get-started/")

print("\nAfter installation, verify PyTorch with:")
print("  python -c \"import torch; print(torch.__version__)\"")