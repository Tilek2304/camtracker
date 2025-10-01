import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 30  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 0)  # black

def visualize(image, detection_result) -> tuple[np.ndarray, list[int]]:
    cord = [0, 0]
    count = 0
    
    for detection in detection_result.detections:
        # Рисуем рамку
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        
        cv2.rectangle(image, start_point, end_point, (0, 165, 255), 3)
        
        # Рисуем центральную точку лица
        center_x = bbox.origin_x + bbox.width // 2
        center_y = bbox.origin_y + bbox.height // 2
        cv2.circle(image, (center_x, center_y), 4, (0, 255, 0), -1)
        
        cord[0] += center_x
        cord[1] += center_y
        count += 1
        
    if count > 0:
        # Усредняем координаты, если найдено несколько лиц
        cord[0] //= count
        cord[1] //= count
        # Рисуем общую центральную точку (цель)
        cv2.circle(image, tuple(cord), 4, (255, 0, 0), -1)
        
    return image, cord