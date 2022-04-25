import pandas as pd
# from sklearn.model_selection import train_test_split
import os
import cv2
from skimage import io, img_as_uint
import json
from ast import literal_eval
from pathlib import Path


# def train_val(files_path, images_path):
#     """
#     file_path: annotation file path
#     image_path: image file path
#     """
#     file = pd.read_csv(files_path)
#     filename_list = []
#     x1_list = []
#     y1_list = []
#     x2_list = []
#     y2_list = []
#     category_list = []
#     for filename, x1, y1, x2, y2, category in zip(file.filename, file.X1, file.Y1, file.X3, file.Y3, file.type):
#         if x1 < x2 and y1 < y2:
#             image_path = os.path.join(images_path, filename)
#             filename_list.append(image_path)
#             x1_list.append(x1)
#             x2_list.append(x2)
#             y1_list.append(y1)
#             y2_list.append(y2)
#             category_list.append(category)
#
#     anno = pd.DataFrame()
#     anno["filename"] = filename_list
#     anno["X1"] = x1_list
#     anno["Y1"] = y1_list
#     anno["X2"] = x2_list
#     anno["Y2"] = y2_list
#     anno["type"] = category_list
#
#     anno.to_csv("train.csv", index=None)
#     train_annotation, val_annotation = train_test_split(anno, test_size=0.15)
#     train_annotation.to_csv("train_annotation.csv", index=None)
#     val_annotation.to_csv("val_annotation.csv", index=None)
#     print("Done!")


def json_convert(csv_path, json_file):
    """
    csv file convert to json file of coco dataset
    :param csv_path: path
    :param json_file: path
    :return: none
    """
    start_bbox_id = 1
    categories = {'AcrimSat': 1, 'Aquarius': 2, 'Aura': 3, 'Calipso': 4, 'Cloudsat': 5, 'CubeSat': 6,
                  'Debris': 7, 'Jason': 8, 'Sentinel-6': 9, 'Terra': 10, 'TRMM': 11}

    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

    root_dir = csv_path[:-11]
    train_or_val = root_dir.split('/')[-1]

    bnd_id = start_bbox_id
    labels = pd.read_csv(csv_path)
    images_num = len(labels)  # 文件数
    for idx in range(images_num):
        image_id = idx + 1
        sat_name = labels.iloc[idx]['class']
        img_name = labels.iloc[idx]['image']
        img_name = sat_name + "_" + img_name.split("_")[1] + "_img.png"
        img_path = f'{root_dir}/{img_name}'
        # image = io.imread(img_path)

        height, width = 1024, 1024
        image = {"file_name": img_path, "height": height, "width": width, "id": image_id}
        json_dict["images"].append(image)

        category_id = categories[sat_name]

        bbox = labels.iloc[idx]['bbox']
        bbox = literal_eval(bbox)

        xmin, ymin, xmax, ymax = bbox
        # xmin = float(xmin)
        # ymin = float(ymin)
        # xmax = float(xmax)
        # ymax = float(ymax)

        o_width = abs(xmax - xmin)
        o_height = abs(ymax - ymin)
        area = o_height * o_width
        anno = {"area": area, "iscrowd": 0, "image_id": image_id, "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id, "id": bnd_id, "ignore": 0, "segmentation": []}
        json_dict["annotations"].append(anno)
        bnd_id += 1

        cat = {"supercategory": "none", "id": category_id, "name": sat_name}
        if cat not in json_dict["categories"]:
            json_dict["categories"].append(cat)

        json_fp = open(json_file, "w")
        json_str = json.dumps(json_dict, indent=4)
        json_fp.write(json_str)
        json_fp.close()
        print(f"{train_or_val} {image_id} Done!")


def txt_convert(csv_path, txt_root):
    """
    csv file convert to json file of coco dataset
    :param csv_path: path
    :param json_file: path
    :return: none
    """

    categories = {'AcrimSat': 1, 'Aquarius': 2, 'Aura': 3, 'Calipso': 4, 'Cloudsat': 5, 'CubeSat': 6,
                  'Debris': 7, 'Jason': 8, 'Sentinel-6': 9, 'Terra': 10, 'TRMM': 11}

    labels = pd.read_csv(csv_path)
    images_num = len(labels)  # 文件数
    for idx in range(images_num):
        image_id = idx + 1
        sat_name = labels.iloc[idx]['class']
        img_name = labels.iloc[idx]['image']
        index = img_name.split('_')[1]

        new_label_file = os.path.join(txt_root, sat_name + '_' + index + '_img.txt')

        if os.path.exists(new_label_file):
            pass
        else:
            category_id = categories[sat_name] - 1

            bbox = labels.iloc[idx]['bbox']
            bbox = literal_eval(bbox)

            ymin, xmin, ymax, xmax = bbox
            # xmin, ymin, xmax, ymax = bbox
            xmin = float(xmin)
            ymin = float(ymin)
            xmax = float(xmax)
            ymax = float(ymax)

            image_width = 1024
            image_height = 1024

            x = (xmin + xmax) / 2 / image_width
            y = (ymin + ymax) / 2 / image_height
            w = (xmax - xmin) / image_width
            h = (ymax - ymin) / image_height

            fp = open(new_label_file, mode="w", encoding="utf-8")
            file_str = str(category_id) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) \
                           + ' ' + str(round(h, 6))

            fp.write(file_str)
            fp.close()

            print(f"image {image_id} Done!")


if __name__ == "__main__":
    # csv_path = "./spark_dataset/train_labels.csv"
    # txt_root = "./spark_dataset/labels/train"
    # os.makedirs(txt_root)
    # txt_convert(csv_path, txt_root)
    #
    # csv_path = "./spark_dataset/validate_labels.csv"
    # txt_root = "./spark_dataset/labels/val"
    # os.makedirs(txt_root)
    # txt_convert(csv_path, txt_root)

    csv_path = "./datasets/SPARK/validate_labels.csv"
    json_file = "./datasets/SPARK/annotations/validate_labels.json"
    json_convert(csv_path, json_file)
