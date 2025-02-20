import os
import argparse
import cv2 as cv

from mdlw.utils.capture import video_capture, crop_square, draw_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', type=str, default='./data')
    p.add_argument('--label', type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    input_src = 0
    resolution = 640

    label_dir = os.path.join(args.output_dir, args.label)
    os.makedirs(label_dir, exist_ok=True)
    
    with video_capture(input_src) as cap:
        img_count = len(os.listdir(label_dir))

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame = crop_square(frame)
            frame = cv.resize(frame, (resolution, resolution))

            display_frame = cv.flip(frame.copy(), 1)
            draw_text(display_frame, text="Press 'c' to capture, 'q' to quit")
            cv.imshow('Data capture', display_frame)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('c'):
                fname = os.path.join(label_dir, f"{args.label}_{img_count:05d}.jpg")
                cv.imwrite(fname, frame)
                print(f"Captured: {fname}")
                img_count += 1
            elif key == ord('q'):
                break


if __name__ == '__main__':
    main()