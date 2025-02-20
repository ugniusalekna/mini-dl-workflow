import os, sys

_IS_COLAB = "COLAB_GPU" in os.environ or "COLAB_JUPYTER_IP" in os.environ

if _IS_COLAB:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)