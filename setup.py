import os
from setuptools import setup, find_packages

setup(
    name=os.path.basename(os.path.abspath(os.path.dirname(__file__))),
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "inquirer",
        "ipykernel",
        "ipython",
        "matplotlib",
        "numpy",
        "onnx",
        "onnxruntime",
        "opencv-python",
        "pillow",
        "PyYAML",
        "requests",
        "tensorboard",
        "torch",
        "torchvision",
        "tqdm",
    ],
    extras_require={
        "jupyter": [
            "jupyter",
            "notebook",
            "jupyterlab",
        ]
    },
)