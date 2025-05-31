# recognize.py (수정된 최종 버전)
import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='korean')

def is_vertical(box):
    box = np.array(box)
    width = np.linalg.norm(box[0] - box[1])
    height = np.linalg.norm(box[1] - box[2])
    return height > width

def ocr_crop(image, box):
    pts = np.array(box).astype(int)
    x_min, y_min = pts[:,0].min(), pts[:,1].min()
    x_max, y_max = pts[:,0].max(), pts[:,1].max()
    roi = image[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        return ''

    roi_rotated = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    result = ocr.ocr(roi_rotated, cls=True)
    if not result or not result[0]:
        return ''

    text = ''.join([line[1][0] for line in result[0]])
    return text

def preprocess_and_recognize(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    ocr_result = ocr.ocr(image, cls=True)

    full_text = ''
    visualized_image = image.copy()

    for line in ocr_result[0]:
        box, (text, conf) = line
        pts = np.array(box).astype(int)

        if is_vertical(box):
            text = ocr_crop(image, box)
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        full_text += text
        cv2.polylines(visualized_image, [pts], isClosed=True, color=color, thickness=2)

    return full_text, visualized_image
