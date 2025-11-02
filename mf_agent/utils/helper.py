#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/19 18:23
# @Author  : lizimo@nuist.edu.cn
# @File    : helper.py
# @Description:
import requests
from pathlib import Path
import uuid
import io
import os
from PIL import Image
import base64
import re
from hashlib import md5

def compute_content_hash(content, prefix: str = "es") -> str:
    clean_content = content.strip().lower()
    hash_val = md5(clean_content.encode("utf-8")).hexdigest()
    return f"{prefix}{hash_val}"


def new_doc_id(prefix: str, original_path: str) -> str:
    return f"{prefix}_{Path(original_path).stem}_{uuid.uuid4().hex[:16]}"

def new_doc_id_from_title(title: str) -> str:
    return f"{title}_{uuid.uuid4().hex[:16]}"

def save_image_pil(img: Image.Image, output_assets_dir: Path, doc_id: str, idx: int, ext="png") -> str:
    out_dir = output_assets_dir / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"img_{idx}.{ext}"
    img.save(path)
    return str(path)

def ocr_image(pil_img: Image.Image, lang: str = "chi_sim+eng", image_type: str = "chart") -> str:
    """
    Default OCR via pytesseract. Replace or extend this with cloud OCR SDKs
    if higher accuracy is required.
    chart and flowchart -> 图像理解 返回文本
    promo and table_image -> 图像OCR 返回OCR文本
    """
    assert image_type in ["chart", "flowchart", "promo", "table_image"], "Invalid image type"
    ocr_server_url = "http://172.16.107.15:6006/models/ocr"
    mlm_server_url = "http://172.16.107.15:6006/models/qwen"
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    if image_type in ["chart", "flowchart"]:
        # TODO 走多模态图像理解pipeline
        prompt = """
        # 你是一位图片理解分析师，你的任务是根据给定的图片完成以下任务：
        1. 识别图片中的关键信息和数据，用自然语言描述。
        2. 识别图片中的标题，如果没有则输出'N'。
        # 输出JSON格式：{"output": ..., "title": ...}"""
        payload = {"image": img_b64, "prompt": prompt}
        response = requests.post(mlm_server_url, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return data["output"]
            else:
                raise RuntimeError(f"Qwen-VL Error: {data.get('message')}")
        else:
            raise RuntimeError(f"HTTP Error: {response.status_code}")
    elif image_type in ["promo", "table_image"]:
        # TODO 走图像OCR pipeline
        payload = {"image": img_b64}
        response = requests.post(ocr_server_url, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return data["output"]
            else:
                raise RuntimeError(f"OCR Error: {data.get('message')}")
        else:
            raise RuntimeError(f"HTTP Error: {response.status_code}")
    else:
        raise ValueError(f"{image_type} is a invalid image type")

def image_classify(image_input: str, api_url: str = "http://localhost:8000/predict"):
    # TODO : 图像分类 采用yolo
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"local image file is not exit ：{image_input}")
        with Image.open(image_input).convert("RGB") as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
        filename = os.path.basename(image_input)
    elif isinstance(image_input, Image.Image):
        img_byte_arr = io.BytesIO()
        image_input.convert("RGB").save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        filename = "pil_image.png"
    else:
        raise TypeError("image_input must is local file or PIL.Image Object")
    files = {
        "file": (filename, img_byte_arr, "image/png")
    }
    try:
        response = requests.post(
            url=api_url,
            files=files,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        if result["status"] == "success":
            return result["prediction"]["class"]
        else:
            raise RuntimeError(f"API error：{result['message']}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"request API error：{str(e)}")
    except Exception as e:
        raise RuntimeError(f"Call interface exception：{str(e)}")



def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    def replace_cn_number(match):
        cn_num = match.group()
        try:
            return str(cn2an(cn_num, "smart"))
        except Exception:
            return cn_num
    text = re.sub(r"[零一二三四五六七八九十百千万亿点两]+", replace_cn_number, text)
    text = re.sub(r"[\s　]+", " ", text).strip()
    text = re.sub(r"[^\u4e00-\u9fa5a-z0-9\s,，。.!?%/:-]", "", text)
    return text

if __name__ == '__main__':
    res = clean_text("金融命名体识别是ner，但是NER。一百万，一个人有两个人")
    print(res)