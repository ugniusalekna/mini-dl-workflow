## Changes since the lecture

Here are the updates I made after the lecture:

- Modified function `read_image` under `mdlw/utils/data.py`
  - Added grayscale image handling in function; 
  - Before, all images were converted to 3-channel RGB, even if they were grayscale (like in `FER2013` e.g.), now grayscale images stay mono-channel (H × W × 1). I've made this change, so that I could keep the original shape of the image and then track back that the model was trained on such images, in order to automatically use either RGB or grayscale input when doing real time inference (since before, all images on inference were converted to RGB, even if they were trained on grayscale).

- Thus, added `input_channels` argument to `ImageClassifier` in `mdlw/model.py`
  - Now you can specify the input channels of the image (`1` for grayscale, `3` for RGB, etc.).

- Also, added `input channels` arg to `config/config.yaml`

- Changed all args in `config/config.yaml` from CAPS to lower
  - Thus, changed all calls of args in `scripts/train.py` (and maybe other scripts) from e.g. `cfg.DATA_DIR` to `cfg.data_dir`

- Renamed `scripts/test.py` to `scripts/infer.py` -- I think this better represents the file, since no actual test are done, only inference

- Also, i think its nicer to always be `cd`'ed in main project dir, e.g. `mini-dl-workflow` in my case, and then run scripts as `python scripts/train.py`. Then, all paths should be changed to be relative to the main dir, not the scripts dir. So i changed `data_dir` and `log_dir` in `config/config.yaml` (from `../` to `./`), and changed the path in `scripts/train.py` (`cfg = load_cfg(path='./config/config.yaml')`).

- in `ImageDataset` in `mdlw/dataset.py` i've added `self.transform = transform if transform else T.ToTensor()`, (line 49) so that when no transform is provided, transformation to tensor was still done (otherwise error was thrown)

- in `Trainer` in `mdlw/engine.py` added `self.scheduler.step()` under the condition if scheduler is provided. scheduler has to be stepped similar to optimizer, it adaptivelly changes the learning rate during training, this may improve training (check https://pytorch.org/docs/stable/optim.html or ask chatgpt about it).

- `mdlw/inference.py` was basically rewritten: since onnx was not available for some, i've added a base class `BaseInferenceModel` and then defined two classes for inference -- `TorchInferenceModel` and `ONNXInferenceModel`. This way, now inference (using infer.py now, or test.py before) can be made by providing either `.pt` file (for PyTorch models) or `.onnx` file (for ONNX if it works for you).

- In `scripts/train.py`: defined `scheduler`; added `writer.close()` (line 100, forgot to do that in lecture while coding); added onnx export only if onnx is actually available; changed fstring float formatting in set_postfix_str (line 98)

- I've also made some changes to `mdlw/utils/fmaps.py` so just copy paste the whole code

- Modified `mdlw/__init__.py`, it checks if onnx is installed before executing any scripts.

- Made many changes in `scripts/vis_fmaps.py` so copy paste the code

- Also made many changes in `scripts/infer.py` so yet again copy paste the code

- modified `show_image_grid` function in `mdlw/utils/visualize.py`, since it did not display the img grid correctly for some `num_images` numbers. 

- Modified `setup.py` quite a lot, added auto check if python version is < 13.x to skip onnx install and did some more changes about the install of torch lib. But if all works well for you, you may not modify yours.

- I think thats all, but i may have skipped something, so inspect the code if interested or if any errors occur.

> NOTE: I have this script `scripts/collect_images.py`, made it to collect images with a webcam, in order to train a model on your own data. Feel free to try it out, i've tried to train a classifier to recognize rock paper scissors.