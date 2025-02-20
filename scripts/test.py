import argparse
import cv2 as cv
import numpy as np

from mdlw.inference import InferenceModel
from mdlw.utils.data import make_class_map
from mdlw.utils.capture import video_capture, crop_square, draw_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx_path', type=str, required=True)
    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--mode', type=str, default='stream', choices=['stream', 'draw'])
    return p.parse_args()


def run_stream_mode(model):
    with video_capture() as cap:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame = crop_square(frame)
            display_frame = cv.flip(frame.copy(), 1)
            
            pred, prob = model(frame, return_prob=True)
            draw_text(display_frame, text="Press 'q' to quit", font_scale=1.0, pos=(10, 40))
            draw_text(display_frame, text=f"Prediction: {pred}; Probability: {prob:.2f}",
                      font_scale=1.0, pos=(10, 80))
            cv.imshow('Inference', display_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break


def run_draw_mode(model, canvas_size=1024):
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    drawing = False
    last_point = None
    brush_size = 30
    erase_mode = False

    def draw_callback(event, x, y, flags, param):
        nonlocal drawing, last_point, canvas, brush_size, erase_mode
        color = (0, 0, 0) if erase_mode else (255, 255, 255)
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
            cv.circle(canvas, (x, y), brush_size, color, -1)
        elif event == cv.EVENT_MOUSEMOVE and drawing:
            cv.line(canvas, last_point, (x, y), color, thickness=brush_size*2)
            last_point = (x, y)
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False

    window_name = "Inference"
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, draw_callback)

    while True:
        frame = canvas.copy()
        pred, prob = model(frame, return_prob=True)
        draw_text(frame, text="Press 'q' to quit; 'c' to clear", font_scale=1.0, pos=(10, 30))
        draw_text(frame, text=f"Prediction: {pred}; Probability: {prob:.2f}", font_scale=1.0, pos=(10, 80))
        draw_text(frame, text=f"Brush size: {brush_size}; Eraser: {'ON' if erase_mode else 'OFF'}", font_scale=1.0, pos=(10, 130))
        cv.imshow(window_name, frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
        elif key == ord('w'):
            brush_size += 1
        elif key == ord('s'):
            brush_size = max(1, brush_size - 1)
        elif key == ord('e'):
            erase_mode = not erase_mode
    cv.destroyAllWindows()


def main():
    args = parse_args()
    class_map = make_class_map(args.data_dir)
    model = InferenceModel(model_path=args.onnx_path, class_map=class_map)
    
    if args.mode == 'stream':
        run_stream_mode(model)
    elif args.mode == 'draw':
        run_draw_mode(model)


if __name__ == '__main__':
    main()