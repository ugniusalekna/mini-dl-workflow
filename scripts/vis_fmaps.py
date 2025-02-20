import warnings
warnings.simplefilter("ignore", FutureWarning)
import argparse
import torch
import cv2 as cv
import numpy as np

from mdlw.utils.capture import video_capture, crop_square, draw_text
from mdlw.utils.data import make_class_map, reverse_class_map
from mdlw.utils.fmaps import update_hooks, build_grid, preprocess, build_fc_composite
from mdlw.utils.misc import load_cfg, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--config_path', type=str, default="../configs/config.yaml")
    p.add_argument('--mode', type=str, default='stream', choices=['stream', 'draw'])
    return p.parse_args()


def init_model(args):
    cfg = load_cfg(path=args.config_path)
    device = get_device()
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model.eval()
    class_map = make_class_map(cfg.data_dir)
    reversed_map = reverse_class_map(class_map)
    conv_layers = [name for name, _ in model.named_modules() if name.startswith("bn")]
    fc_layers   = [name for name, _ in model.named_modules() if name.startswith("fc")]
    layers = conv_layers + ['fc_all']
    activation = {}
    update_hooks(model, activation, [layers[0]])
    return cfg, device, model, reversed_map, layers, fc_layers, activation


def process_frame(image, cfg, device, model, reversed_map, layers, fc_layers, activation, current_idx, use_act, flip=True):
    proc_img = crop_square(image)
    if flip:
        proc_img = cv.flip(proc_img, 1)
        
    tensor = preprocess(proc_img, imgsz=cfg.image_size, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    class_idx = np.argmax(probs)
    pred, prob = reversed_map[class_idx], probs[class_idx]
    
    if layers[current_idx] == 'fc_all':
        vis = build_fc_composite(activation, fc_layers, gridsz=proc_img.shape[:2], use_act=use_act)
        draw_text(vis, text="FC layers" + ('+act' if use_act else ''), pos=(10, 30), font_scale=1.0)
    else:
        feat = activation[layers[current_idx]].squeeze(0)
        grid = cv.resize(build_grid(feat, use_act=use_act), (proc_img.shape[1], proc_img.shape[0]),
                         interpolation=cv.INTER_NEAREST)
        vis = cv.applyColorMap(grid, cv.COLORMAP_VIRIDIS)
        draw_text(vis, text=layers[current_idx] + ('+act' if use_act else ''), pos=(10, 30), font_scale=1.0)
    
    combined = np.concatenate([proc_img, vis], axis=1)
    return combined, (pred, prob)


def run_stream_mode(cfg, device, model, reversed_map, layers, fc_layers, activation):
    current_idx = 0
    use_act = False
    paused = False
    with video_capture(0) as cap:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
            output, (pred, prob) = process_frame(frame, cfg, device, model, reversed_map,
                                                   layers, fc_layers, activation, current_idx, use_act)
            draw_text(output, text="'j','l' change layers, 'k' toggle act, 'space' pause, 'q' quit.", font_scale=1.0)
            draw_text(output, text=f"Prediction: {pred}; Probability: {prob:.2f}", font_scale=1.0, pos=(10, 80))
            cv.imshow("Feature visualization", output)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                current_idx = (current_idx + 1) % len(layers)
                hook_layers = fc_layers if layers[current_idx] == 'fc_all' else [layers[current_idx]]
                update_hooks(model, activation, hook_layers)
            elif key == ord('j'):
                current_idx = (current_idx - 1) % len(layers)
                hook_layers = fc_layers if layers[current_idx] == 'fc_all' else [layers[current_idx]]
                update_hooks(model, activation, hook_layers)
            elif key == ord(' '):
                paused = not paused
            elif key == ord('k'):
                use_act = not use_act


def run_draw_mode(cfg, device, model, reversed_map, layers, fc_layers, activation, canvas_size=1024):
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    drawing = False
    last_point = None
    brush_size = 30
    erase_mode = False
    current_idx = 0
    use_act = False

    def draw_callback(event, x, y, flags, param):
        nonlocal drawing, last_point, canvas, brush_size, erase_mode
        color = (0, 0, 0) if erase_mode else (255, 255, 255)
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
            cv.circle(canvas, (x, y), brush_size, color, -1)
        elif event == cv.EVENT_MOUSEMOVE and drawing:
            cv.line(canvas, last_point, (x, y), color, thickness=brush_size * 2)
            last_point = (x, y)
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False

    window_name = "Feature visualization"
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, draw_callback)

    while True:
        frame = canvas.copy()
        output, (pred, prob) = process_frame(frame, cfg, device, model, reversed_map,
                                             layers, fc_layers, activation, current_idx, use_act, flip=False)
        draw_text(output, text="'j','l' change layers, 'k' toggle act, 'space' pause, 'q' quit;", font_scale=1.0)
        draw_text(output, text="'c': clear; 'w/s': change brush size; 'e': toggle erase;", font_scale=1.0, pos=(10, 80))
        draw_text(output, text=f"Prediction: {pred}; Probability: {prob:.2f}", font_scale=1.0, pos=(10, 130))
        draw_text(output, text=f"Brush size: {brush_size}; Eraser: {'ON' if erase_mode else 'OFF'}", font_scale=1.0, pos=(10, 180))
        cv.imshow(window_name, output)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
        elif key == ord('w'):
            brush_size = min(brush_size + 1, 99)
        elif key == ord('s'):
            brush_size = max(1, brush_size - 1)
        elif key == ord('e'):
            erase_mode = not erase_mode
        elif key == ord('l'):
            current_idx = (current_idx + 1) % len(layers)
            hook_layers = fc_layers if layers[current_idx] == 'fc_all' else [layers[current_idx]]
            update_hooks(model, activation, hook_layers)
        elif key == ord('j'):
            current_idx = (current_idx - 1) % len(layers)
            hook_layers = fc_layers if layers[current_idx] == 'fc_all' else [layers[current_idx]]
            update_hooks(model, activation, hook_layers)
        elif key == ord('k'):
            use_act = not use_act
    cv.destroyAllWindows()


def main():
    args = parse_args()
    cfg, device, model, reversed_map, layers, fc_layers, activation = init_model(args)
    if args.mode == 'stream':
        run_stream_mode(cfg, device, model, reversed_map, layers, fc_layers, activation)
    elif args.mode == 'draw':
        run_draw_mode(cfg, device, model, reversed_map, layers, fc_layers, activation)


if __name__ == '__main__':
    main()