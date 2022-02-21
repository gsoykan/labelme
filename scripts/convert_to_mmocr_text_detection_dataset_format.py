import json
import os
import glob
import random
from math import floor

import cv2
import numpy as np
from shapely.geometry import Polygon, box
import shutil

img_folder = "/home/gsoykan20/Desktop/datasets/comics_speech_bubble_dataset/raw_images"
raw_annotations_path = "/home/gsoykan20/Desktop/datasets/comics_speech_bubble_dataset/raw_images"


def delete_contents_of_folder(folder_path):
    files = glob.glob(f'{folder_path}/*')
    for f in files:
        os.remove(f)


def search_files(extension='.ttf', folder='H:\\'):
    files = []
    for r, d, f in os.walk(folder):
        for file in f:
            if file.endswith(extension):
                files.append(r + "/" + file)
    return files


def read_or_get_image(img, read_rgb: bool = False):
    img_str = ""
    if not isinstance(img, (np.ndarray, str)):
        raise AssertionError('Images must be strings or numpy arrays')

    if isinstance(img, str):
        img_str = img
        img = cv2.imread(img)

    if img is None:
        raise AssertionError('Image could not be read: ' + img_str)

    if read_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def create_instances_json(json_files, is_train: bool):
    json_data = {}
    json_data["images"] = []
    json_data["categories"] = [{"id": 1, "name": "text"}]
    json_data["annotations"] = []
    detection_id_count = 0
    for json_file in json_files:
        img_alias = json_file.split("/")[-1].split(".")[0]
        img_path = os.path.join(img_folder, f"{img_alias}.jpg")
        cv_img = read_or_get_image(img_path)
        img_height, img_width, img_channels = cv_img.shape
        img_id = int(img_alias)
        f = open(json_file)
        raw_annotation_json = json.load(f)
        f.close()
        shapes = raw_annotation_json["shapes"]
        file_dir = "train" if is_train else "test"
        json_data["images"].append({
            "file_name": f"{file_dir}/{img_alias}.jpg",
            "height": img_height,
            "width": img_width,
            "segm_file": None,
            "id": img_id
        })

        for shape in shapes:
            points = shape["points"]
            shape_type = shape["shape_type"]
            polygon = None
            # polygon.bounds (minx, miny, maxx, maxy)
            if shape_type == "rectangle":
                x_1 = points[0][0]
                y_1 = points[0][1]
                x_2 = points[1][0]
                y_2 = points[1][1]
                polygon = box(min(x_1, x_2), min(y_1, y_2), max(x_1, x_2), max(y_1, y_2))
            elif shape_type == "polygon":
                polygon = Polygon(points)
            else:
                raise Exception("unknown shape_type")

            minx, miny, maxx, maxy = polygon.bounds
            segm = []
            for coord in polygon.exterior.coords:
                segm.append(floor(coord[0]))
                segm.append(floor(coord[1]))
            segm = segm[:-2]

            json_data["annotations"].append({
                "iscrowd": 0,
                "category_id": 1,
                "bbox": [minx, miny, maxx - minx, maxy - miny],
                "area": polygon.area,
                "segmentation": [segm],
                "image_id": img_id,
                "id": detection_id_count
            })
            detection_id_count += 1

    return json_data


if __name__ == '__main__':
    raw_annotations_path = "/home/gsoykan20/Desktop/datasets/comics_speech_bubble_dataset/raw_images"
    json_files = search_files(".json", raw_annotations_path)
    random.seed(10)
    random.shuffle(json_files)
    test_size = 20
    train_json = json_files[:-test_size]
    test_json = json_files[-test_size:]
    delete_contents_of_folder(
        "/home/gsoykan20/Desktop/self_development/mmocr/tests/data/comics_speech_bubble_dataset/test/imgs/test")
    delete_contents_of_folder(
        "/home/gsoykan20/Desktop/self_development/mmocr/tests/data/comics_speech_bubble_dataset/train/imgs/train")
    for t in test_json:
        head, tail = os.path.split(t)
        img_path = os.path.join(img_folder, tail.split(".")[0] + ".jpg")
        shutil.copy2(img_path,
                     "/home/gsoykan20/Desktop/self_development/mmocr/tests/data/comics_speech_bubble_dataset/test/imgs/test")
    for t in train_json:
        head, tail = os.path.split(t)
        img_path = os.path.join(img_folder, tail.split(".")[0] + ".jpg")
        shutil.copy2(img_path,
                     "/home/gsoykan20/Desktop/self_development/mmocr/tests/data/comics_speech_bubble_dataset/train/imgs/train")
    train_json = create_instances_json(train_json, True)
    test_json = create_instances_json(test_json, False)
    with open('train_dataset.json', 'w') as outfile:
        json.dump(train_json, outfile)
    with open('test_dataset.json', 'w') as outfile:
        json.dump(test_json, outfile)
    shutil.copy2('test_dataset.json',
                 "/home/gsoykan20/Desktop/self_development/mmocr/tests/data/comics_speech_bubble_dataset/test/instances_test.json")
    shutil.copy2('train_dataset.json',
                 "/home/gsoykan20/Desktop/self_development/mmocr/tests/data/comics_speech_bubble_dataset/train/instances_train.json")
