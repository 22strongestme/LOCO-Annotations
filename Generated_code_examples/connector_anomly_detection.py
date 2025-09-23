import json
import numpy as np
import cv2
import os
from typing import List, Dict, Union, Tuple, Optional
from scipy.spatial.distance import euclidean

class Image:
    def __init__(self, image_path):
        self.image_path = image_path
        self.height = None
        self.width = None
        self.instances = self.find()
        self.locate()
        self.size()
        self.image_text = self.read_image_text()

    def read_image_text(self):
        json_file = os.path.splitext(self.image_path)[0] + '.json'
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data.get("image_text", "") 

    def find(self):
        json_file = os.path.splitext(self.image_path)[0] + '.json'
        if not os.path.exists(json_file):
            return []
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.height = data['imageHeight']
        self.width = data['imageWidth']
        instances = []
        for obj in data['shapes']:
            label = obj['label']
            shape_type = obj['shape_type']
            points = np.array(obj['points']).astype(np.int32)
            mask = np.zeros((self.height, self.width), dtype=np.uint8)

            if shape_type == 'polygon':
                cv2.fillPoly(mask, [points], color=1)
            elif shape_type == 'rectangle':
                top_left = tuple(points[0])
                bottom_right = tuple(points[1])
                cv2.rectangle(mask, top_left, bottom_right, color=1, thickness=-1)
            elif shape_type == 'circle':

                x1, y1 = points[0]
                x2, y2 = points[1]
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                radius = int(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / 2)
                cv2.circle(mask, center, radius, color=1, thickness=-1)

            instance = {'label': label, 'mask': mask, 'imageHeight': self.height, 'imageWidth': self.width}
            instances.append(instance)
        return instances

    def locate(self):
        for instance in self.instances:
            mask = instance['mask']

            y, x = np.where(mask == 1)

            if y.size and x.size:
                x_min, y_min = np.min(x), np.min(y)
                x_max, y_max = np.max(x), np.max(y)

                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                instance['bounding_box'] = (x_min, y_min, x_max, y_max)
                instance['center'] = (x_center, y_center)

                points = np.array(list(zip(x, y)))
                rect = cv2.minAreaRect(points)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                x_sorted = box[np.argsort(box[:, 0]), :]
                y_sorted = box[np.argsort(box[:, 1]), :]

                left_endpoint = ((x_sorted[0, 0] + x_sorted[1, 0]) / 2, (x_sorted[0, 1] + x_sorted[1, 1]) / 2)
                right_endpoint = ((x_sorted[2, 0] + x_sorted[3, 0]) / 2, (x_sorted[2, 1] + x_sorted[3, 1]) / 2)
                top_endpoint = ((y_sorted[0, 0] + y_sorted[1, 0]) / 2, (y_sorted[0, 1] + y_sorted[1, 1]) / 2)
                bottom_endpoint = ((y_sorted[2, 0] + y_sorted[3, 0]) / 2, (y_sorted[2, 1] + y_sorted[3, 1]) / 2)

                instance['left_endpoint'] = left_endpoint
                instance['right_endpoint'] = right_endpoint
                instance['top_endpoint'] = top_endpoint
                instance['bottom_endpoint'] = bottom_endpoint

                instance['minRect'] = box
        return self.instances

    def size(self):
        for instance in self.instances:
            mask = instance['mask']

            y, x = np.where(mask == 1)
            points = np.array(list(zip(x, y)))
            x, y, w, h = cv2.boundingRect(points)

            horizontal = w
            vertical = h


            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            width = int(rect[1][0])
            length = int(rect[1][1])

            if width > length:
                width, length = length, width

            box_area = width * length

            mask_area = np.sum(mask)

            aspect_ratio = width / length if length != 0 else 0

            instance['minRect'] = box
            instance['width'] = width
            instance['length'] = length
            instance['box_area'] = box_area
            instance['mask_area'] = mask_area
            instance['aspect_ratio'] = aspect_ratio
            instance['horizontal'] = horizontal
            instance['vertical'] = vertical
        return self.instances

    def count(self, position=None, min_size=None, max_size=None, label=None, min_area=None, max_area=None):

        count = 0

        for instance in self.instances:

            image_width = instance['imageWidth']
            image_height = instance['imageHeight']

            if label and instance['label'] != label:
                continue

            if position and image_width:
                x_min, y_min, x_max, y_max = instance['bounding_box']
                if position == 'left' and x_max > image_width / 2:
                    continue
                elif position == 'right' and x_min < image_width / 2:
                    continue
                elif position == 'top' and y_max > image_height / 2:
                    continue
                elif position == 'bottom' and y_min < image_height / 2:
                    continue

            width, length, area = instance['width'], instance['length'], instance['mask_area']
            if min_size and (width < min_size[0] or length < min_size[1]):
                continue
            if max_size and (width > max_size[0] or length > max_size[1]):
                continue
            if min_area and area < min_area:
                continue
            if max_area and area > max_area:
                continue

            count += 1
        return count

    def determine_color(self, label):
        target_instance = None
        for instance in self.instances:
            if instance['label'] == label:
                target_instance = instance
                break
        if not target_instance:
            print(f"No instance found with label: {label}")
            return
        image = cv2.imread(self.image_path)
        mask = target_instance['mask']

        masked_image = cv2.bitwise_and(image, image, mask=mask)
        pixels = masked_image[mask == 1]

        pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

        hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)

        color_ranges = {
            'yellow-sticker': ([20, 100, 100], [40, 255, 255]),
            'red': ([0, 50, 50], [10, 255, 255]),
            'white': ([0, 0, 180], [180, 40, 255]),
            'yellow': ([20, 50, 50], [40, 255, 255]),
            'blue': ([100, 50, 50], [140, 255, 255]) 
        }

        color_percentages = {}

        for color_name, color_range in color_ranges.items():
            mask_color = cv2.inRange(hsv_pixels.reshape(-1, 1, 3), np.array(color_range[0]).reshape(1, 3),
                                     np.array(color_range[1]).reshape(1, 3))
            percentage = np.sum(mask_color) / 255.0 / len(hsv_pixels)
            color_percentages[color_name] = percentage


        std_dev = np.std(pixels, axis=1)
        white_pixels = pixels[np.where(std_dev < 20)] 

        white_percentage = len(white_pixels) / len(pixels)

        if instance['label'] == 'juice':

            if color_percentages.get('yellow', 0) > 0.8:
                main_color = 'yellow'
            elif color_percentages.get('red', 0) >= 0.3:
                main_color = 'red'
            elif white_percentage >= 0.3:
                main_color = 'white'
            else:
                main_color = 'unknown'

        if instance['label'] == 'cable':

            if color_percentages.get('yellow', 0) >= 0.5:
                main_color = 'yellow'
            elif color_percentages.get('red', 0) >= 0.5:
                main_color = 'red'
            elif color_percentages.get('blue', 0) >= 0.5:
                main_color = 'blue'
            else:
                main_color = 'unknown'


        target_instance['color'] = main_color

        return target_instance

def analyze_image(image_path: str) -> Tuple[bool, str]:
    # Initialize Image object
    image = Image(image_path)

    # Locate Objects
    instances = image.locate()

    # Object Presence Check
    cables = [instance for instance in instances if instance['label'] == 'cable']
    connectors = [instance for instance in instances if instance['label'] == 'connector']
    if len(cables) != 1 or len(connectors) < 1:
        return True, "Incorrect number of cables or connectors."

    # Measure Cable Length
    cable = cables[0]
    image.size()
    cable_length = cable['length']
    if not 500 <= cable_length <= 1300:
        return True, f"Cable length is not within the required range: {cable_length}px"

    # Check Connector Positions
    connectors_sorted_by_y = sorted(connectors, key=lambda x: x['center'][1])
    left_connectors = [c for c in connectors_sorted_by_y if c['center'][0] < image.width / 2]
    right_connectors = [c for c in connectors_sorted_by_y if c['center'][0] >= image.width / 2]
    if len(left_connectors) != len(right_connectors):
        return True, "Unequal number of connectors on each end of the cable."

    # Determine Cable Color
    cable_color = image.determine_color('cable')['color']
    connector_count = len(left_connectors)  # Assuming equal number of connectors on each side
    expected_color = {2: 'yellow', 3: 'blue', 5: 'red'}.get(connector_count)
    if cable_color != expected_color:
        return True, f"Cable color does not match the expected color for {connector_count} connectors."

    # Check Cable-Connector Consistency
    left_attachment_point = cable['left_endpoint']
    right_attachment_point = cable['right_endpoint']
    closest_left_connector = min(left_connectors,
                                 key=lambda c: np.linalg.norm(np.array(c['center']) - np.array(left_attachment_point)))
    closest_right_connector = min(right_connectors, key=lambda c: np.linalg.norm(
        np.array(c['center']) - np.array(right_attachment_point)))
    if closest_left_connector != left_connectors[0] or closest_right_connector != right_connectors[0]:
        return True, "Inconsistent attachment of cable to connectors."

    # If all checks pass
    return False, "No abnormalities found."


if __name__ == '__main__':
    directory = "logical_anomalies/"
    results_file = "analysis_results.txt"


    with open(results_file, 'w') as result_file:
        for filename in os.listdir(directory):
            if filename.endswith('.png'): 
                image_path = os.path.join(directory, filename)
                image = Image(image_path)
                result, message = analyze_image(image_path)
                if result:
                    print("abnormal",message)
                else:
                    print("normal",message)


                result_text = "abnormal" if result else "normal"
                result_file.write(f"{filename}: {result_text}, {message}\n")
