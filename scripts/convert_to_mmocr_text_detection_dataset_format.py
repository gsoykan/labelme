import json
import os
import glob
import random
from math import floor

import cv2
import numpy as np
from shapely.geometry import Polygon, box
import shutil
from dotenv import load_dotenv

load_dotenv()


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


def create_instances_json(json_files, is_train: bool, is_val: bool = False):
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
        img_id = img_alias
        f = open(json_file)
        raw_annotation_json = json.load(f)
        f.close()
        shapes = raw_annotation_json["shapes"]
        file_dir = "train"
        if not is_train and is_val:
            file_dir = "test"
        elif not is_train and not is_val:
            file_dir = "test"
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
    raw_annotations_path = os.environ.get('text_detection_raw_annotations_path')
    img_folder = os.environ.get('text_detection_img_folder')
    text_det_test_dataset = os.environ.get('text_det_test_dataset')
    text_det_train_dataset = os.environ.get('text_det_train_dataset')
    text_det_val_dataset = os.environ.get('text_det_val_dataset')
    text_det_test_dataset_label = os.environ.get('text_det_test_dataset_label')
    text_det_train_dataset_label = os.environ.get('text_det_train_dataset_label')
    text_det_val_dataset_label = os.environ.get('text_det_val_dataset_label')
    json_files = search_files(".json", raw_annotations_path)
    random.seed(10)
    random.shuffle(json_files)
    test_size = 50
    val_size = 50
    train_json = json_files[:-(test_size + val_size)]
    val_json = json_files[-(test_size + val_size):-test_size]
    test_json = json_files[-test_size:]
    for dataset_location in [text_det_test_dataset, text_det_train_dataset, text_det_val_dataset]:
        delete_contents_of_folder(dataset_location)
    for t in test_json:
        head, tail = os.path.split(t)
        img_path = os.path.join(img_folder, tail.split(".")[0] + ".jpg")
        shutil.copy2(img_path,
                     text_det_test_dataset)
    for t in val_json:
        head, tail = os.path.split(t)
        img_path = os.path.join(img_folder, tail.split(".")[0] + ".jpg")
        shutil.copy2(img_path,
                     text_det_val_dataset)
    for t in train_json:
        head, tail = os.path.split(t)
        img_path = os.path.join(img_folder, tail.split(".")[0] + ".jpg")
        shutil.copy2(img_path,
                     text_det_train_dataset)
    train_json = create_instances_json(train_json, True)
    test_json = create_instances_json(test_json, False)
    val_json = create_instances_json(val_json, False, True)
    with open('train_dataset.json', 'w') as outfile:
        json.dump(train_json, outfile)
    with open('test_dataset.json', 'w') as outfile:
        json.dump(test_json, outfile)
    with open('val_dataset.json', 'w') as outfile:
        json.dump(val_json, outfile)
    shutil.copy2('test_dataset.json',
                 text_det_test_dataset_label)
    shutil.copy2('val_dataset.json',
                 text_det_val_dataset_label)
    shutil.copy2('train_dataset.json',
                 text_det_train_dataset_label)
