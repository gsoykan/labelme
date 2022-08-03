import json
import os
import glob
import random
from math import floor

import cv2
import numpy as np
from shapely.geometry import Polygon, box
import shutil
from PIL import Image
import uuid

img_folder = "/home/gsoykan20/Desktop/datasets/comics_speech_bubble_dataset/raw_images_with_recognition_jsons"
raw_annotations_path = "/home/gsoykan20/Desktop/datasets/comics_speech_bubble_dataset/raw_images_with_recognition_jsons"

dataset_train_img_folder = "/home/gsoykan20/Desktop/self_development/mmocr/tests/data/ocr_comics_speech_bubble_dataset/train/imgs"
dataset_train_label_txt_path = "/home/gsoykan20/Desktop/self_development/mmocr/tests/data/ocr_comics_speech_bubble_dataset/train/label.txt"

dataset_test_img_folder = "/home/gsoykan20/Desktop/self_development/mmocr/tests/data/ocr_comics_speech_bubble_dataset/test/imgs"
dataset_test_label_txt_path = "/home/gsoykan20/Desktop/self_development/mmocr/tests/data/ocr_comics_speech_bubble_dataset/test/label.txt"

dataset_val_img_folder = "/home/gsoykan20/Desktop/self_development/mmocr/tests/data/ocr_comics_speech_bubble_dataset/val/imgs"
dataset_val_label_txt_path = "/home/gsoykan20/Desktop/self_development/mmocr/tests/data/ocr_comics_speech_bubble_dataset/val/label.txt"

def delete_contents_of_folder(folder_path):
    files = glob.glob(f'{folder_path}/*')
    for f in files:
        os.remove(f)


def delete_txt_content(txt_file_path):
    file = open(txt_file_path, "r+")
    file.truncate(0)
    file.close()


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


def crop_img(img,
             bb,
             img_name_to_save=None):
    bb_img = img[max(int(bb[1]), 0): int(bb[3]), max(int(bb[0]), 0):int(bb[2])]
    # This might throw "ValueError: tile cannot extend outside image"
    # possible lead: https://discuss.pytorch.org/t/valueerror-tile-cannot-extend-outside-image-pls-help/98729
    img = Image.fromarray(bb_img)
    if img_name_to_save is not None:
        img.save(img_name_to_save)
    return img


def create_dataset(json_files, is_train, is_val=False):
    dataset = []
    for json_file in json_files:
        img_alias = json_file.split("/")[-1].split(".")[0]
        img_path = os.path.join(img_folder, f"{img_alias}.jpg")
        cv_img = read_or_get_image(img_path, read_rgb=True)
        f = open(json_file)
        raw_annotation_json = json.load(f)
        f.close()
        shapes = raw_annotation_json["shapes"]
        if len(shapes) == 0:
            continue
        for shape in shapes:
            points = shape["points"]
            text_annotation = shape["label"]
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

            img_alias_for_dataset = str(uuid.uuid4()) + ".jpg"

            img_folder_to_save = None
            if is_train:
                img_folder_to_save = dataset_train_img_folder
            elif is_val:
                img_folder_to_save = dataset_val_img_folder
            else:
                img_folder_to_save = dataset_test_img_folder

            minx, miny, maxx, maxy = polygon.bounds
            crop_img(cv_img,
                     (minx, miny, maxx, maxy),
                     os.path.join(img_folder_to_save,
                                  img_alias_for_dataset))

            dataset.append((img_alias_for_dataset, text_annotation))
    return dataset


def save_dataset(dataset, is_train, is_val=False):
    dataset = list(map(lambda x: " ".join(x) + "\n", dataset))
    dataset[-1] = dataset[-1][:-2]
    label_path = None
    if is_train:
        label_path = dataset_train_label_txt_path
    elif is_val:
        label_path = dataset_val_label_txt_path
    else:
        label_path = dataset_test_label_txt_path
    with open(label_path, 'w') as f:
        f.writelines(dataset)


if __name__ == '__main__':
    json_files = search_files(".json", raw_annotations_path)
    random.seed(10)
    random.shuffle(json_files)
    test_size = 50
    val_size = 50
    train_json = json_files[:-(test_size + val_size)]
    val_json = json_files[-(test_size + val_size):-test_size]
    test_json = json_files[-test_size:]
    delete_contents_of_folder(dataset_train_img_folder)
    delete_txt_content(dataset_train_label_txt_path)
    delete_contents_of_folder(dataset_test_img_folder)
    delete_txt_content(dataset_test_label_txt_path)
    delete_contents_of_folder(dataset_val_img_folder)
    delete_txt_content(dataset_val_label_txt_path)
    train_dataset = create_dataset(train_json, True)
    test_dataset = create_dataset(test_json, False)
    val_dataset = create_dataset(val_json, False, is_val=True)
    save_dataset(train_dataset, True)
    save_dataset(test_dataset, False)
    save_dataset(val_dataset, False, is_val=True)
