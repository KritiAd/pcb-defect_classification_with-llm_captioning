import os
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import random
import shutil
from PIL import Image
import base64
from io import BytesIO



def parse_xml(file_path, record):
  tree = ET.parse(file_path)
  root = tree.getroot()
  filename = root.find("filename").text

  for obj in root.findall("object"):


    name = obj.find("name").text
    bndbox = obj.find("bndbox")
    xmin = bndbox.find("xmin").text
    ymin = bndbox.find("ymin").text
    xmax = bndbox.find("xmax").text
    ymax = bndbox.find("ymax").text

    dict_xml = {
        "filename" : filename,
        "name": name,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax
    }

    record.append(dict_xml)

def convert_annotations_to_df(xml_dir):
    all_data = []
    xml_dir = Path(xml_dir)

    for xml_folder in xml_dir.iterdir():
        for xml_file in xml_folder.iterdir():
            parse_xml(xml_file, all_data)

    df_xml = pd.DataFrame(all_data)

    return df_xml

def  train_test_split(root_img_directory):
  
  for defect_docs in root_img_directory.iterdir():
    if not os.path.exists(f"test/{defect_docs.name}"):
        os.mkdir(f"test/{defect_docs.name}")

    image_count = len(os.listdir(defect_docs))
    len_train = int(0.8*image_count)
    len_test = int (0.2* image_count)

    randomly_selected_files = random.sample(os.listdir(defect_docs), len_test)

    for fileR in randomly_selected_files:
        src = os.path.join(defect_docs, fileR)
        dest = os.path.join("test", defect_docs.name, fileR)
        shutil.move(src, dest)
    
    for file_train in defect_docs:
        src = os.path.join(defect_docs, file_train)
        dest = os.path.join("train", defect_docs.name, file_train)
        shutil.move(src, dest)

    #print(f"Split for folder {defect_docs.name} is complete. {len_test} images in the respective test folder and {len_train} in the train folder")

def resize_and_convert_to_base64(image_path, max_width=512):
    img = Image.open(image_path)
    if img.width > max_width:
        ratio = max_width / float(img.width)
        new_height = int(float(img.height) * ratio)
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="WEBP", quality=70)
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/webp;base64,{base64_img}"
