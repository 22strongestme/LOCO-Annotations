import json
import numpy as np
import cv2
import os
from typing import List, Dict, Union, Tuple, Optional

class Image:
    def __init__(self, image_path):
        self.image_path = image_path
        self.instances = self.find()
        self.height = None
        self.width = None
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
        return self.instances

    def size(self):
        for instance in self.instances:
            mask = instance['mask']
            y, x = np.where(mask == 1)
            points = np.array(list(zip(x, y)))

            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            width = int(rect[1][0])
            height = int(rect[1][1])

            if width > height:
                width, height = height, width

            box_area = width * height

            mask_area = np.sum(mask)

            aspect_ratio = width / height if height != 0 else 0

            instance['minRect'] = box
            instance['width'] = width
            instance['height'] = height
            instance['box_area'] = box_area
            instance['mask_area'] = mask_area
            instance['aspect_ratio'] = aspect_ratio
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

            width, height, area = instance['width'], instance['height'], instance['mask_area']
            if min_size and (width < min_size[0] or height < min_size[1]):
                continue
            if max_size and (width > max_size[0] or height > max_size[1]):
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



def analyze_image_abnormality(image_path):
    # Step 1: Initialization
    img_instance = Image(image_path)
    img_instance.locate()
    img_instance.size()

    abnormalities = []

    # Step 2: Object Presence
    required_labels = ["bottle", "juice", "image_label", "text_label"]
    fruit_patterns = ["banana", "orange", "cherry"]
    present_labels = [instance['label'] for instance in img_instance.instances]

    for label in required_labels:
        if label not in present_labels:
            abnormalities.append(f"Missing {label}")
    if not any(fruit_pattern in present_labels for fruit_pattern in fruit_patterns):
        abnormalities.append("Missing fruit pattern in the image")

    # Helper function to get instance by label
    def get_instance_by_label(label):
        for instance in img_instance.instances:
            if instance['label'] == label:
                return instance
        return None

    # Step 3: Object Properties
    juice_instance = get_instance_by_label('juice')
    if juice_instance:
        if not 1000 <= juice_instance['height'] <= 1100:
            abnormalities.append("Juice height is out of range")

    # Step 4: Object Positions
    image_label_instance = get_instance_by_label('image_label')
    bottle_instance = get_instance_by_label('bottle')
    text_label_instance = get_instance_by_label('text_label')
    fruit_pattern_instance = None
    for fruit_pattern in fruit_patterns:
        if get_instance_by_label(fruit_pattern):
            fruit_pattern_instance = get_instance_by_label(fruit_pattern)
            break

    if image_label_instance and bottle_instance:
        distance = np.linalg.norm(np.array(image_label_instance['center']) - np.array(bottle_instance['center']))
        if distance >= 100 or abs(image_label_instance['center'][0] - bottle_instance['center'][0]) >= 25:
            abnormalities.append("Improper position of image_label with respect to bottle")

    if fruit_pattern_instance and image_label_instance:
        distance = np.linalg.norm(np.array(fruit_pattern_instance['center']) - np.array(image_label_instance['center']))
        if distance > 25:
            abnormalities.append("Improper position of fruit pattern with respect to image_label")

    if text_label_instance and bottle_instance:
        x_diff = abs(text_label_instance['center'][0] - bottle_instance['center'][0])
        y_diff = text_label_instance['center'][1] - bottle_instance['center'][1]
        if x_diff >= 25 or not 430 <= y_diff <= 530:
            abnormalities.append("Improper position of text_label with respect to bottle")

    # Step 5: Color Consistency
    if juice_instance:
        juice_color = img_instance.determine_color('juice')['color']
        fruit_pattern_label = fruit_pattern_instance['label'] if fruit_pattern_instance else None
        color_mapping = {'orange': 'yellow', 'cherry': 'red', 'banana': 'white'}
        if fruit_pattern_label and juice_color != color_mapping[fruit_pattern_label]:
            abnormalities.append(f"Juice color does not match with {fruit_pattern_label}")

    # Step 6: Compile Results
    is_abnormal = len(abnormalities) > 0
    reasons = " and ".join(abnormalities)

    return is_abnormal, reasons



if __name__ == '__main__':

    directory = "logical_anomalies/"
    results_file = "results.txt"
    with  open(results_file, 'w') as result_file:
        for filename in os.listdir(directory):
            if filename.endswith('.png'):
                image_path = os.path.join(directory, filename)
                image = Image(image_path)
                result, message = analyze_image_abnormality(image_path)
                if result:
                    print("normal",message)
                else:
                    print("abnormal",message)
                result_text = "normal" if result else "abnormal"
                result_file.write(f"{filename}: {result_text}, {message}\n")
