import cv2 as cv
from contextlib import contextmanager


@contextmanager
def video_capture(device_id=0):
    cap = cv.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError("Can't access the webcam")
    try:
        yield cap
    finally:
        cap.release()
        cv.destroyAllWindows()


def crop_square(image):
    h, w, _ = image.shape
    min_dim = min(h, w)
    x, y = (w - min_dim) // 2, (h - min_dim) // 2
    return image[y:y + min_dim, x:x + min_dim]


def draw_text(frame, text='', pos=(10, 30), font=cv.FONT_HERSHEY_SIMPLEX, font_scale=0.5, 
              color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0), alpha=0.5):
    if not text:
        return
    (text_w, text_h), baseline = cv.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    
    overlay = frame.copy()
    cv.rectangle(overlay, (x - 5, y - text_h - 5), (x + text_w + 5, y + baseline), bg_color, -1)
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv.putText(frame, text, pos, font, font_scale, color, thickness, cv.LINE_AA)