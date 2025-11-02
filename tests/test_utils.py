#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/19 19:55
# @Author  : lizimo@nuist.edu.cn
# @File    : test_utils.py
# @Description:
import unittest
import os
import subprocess
import io
import sys
import time
import requests
from PIL import Image
import base64

class UtilsTest(unittest.TestCase):
    def setUp(self):
        """测试初始化"""
        print("elian: lizimo@nuist.edu.cn")

    def wait_for_server(self, url="http://localhost:8000/docs", timeout=20):
        print("Waiting for YOLO service to start...")
        for _ in range(timeout):
            try:
                r = requests.get(url, timeout=1)
                if r.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("timeout")

    def test_image_classify(self):
        """tester image_classify"""
        from src.utils.helper import image_classify
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        script_path = os.path.join(project_root, "src", "utils", "yolo", "inference.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"don't find the inference.py file: {script_path}")
        command = [sys.executable, script_path]
        print(f"Starting YOLO service: {script_path}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        try:
            self.wait_for_server()
            test_img = r"G:\项目成果打包\金融多模态知识库构建与复杂问答检索算法\AI-Claude\data\test\b23.jpg"
            result = image_classify(test_img)
            print("res", result)
            self.assertIn("status", result)
            self.assertEqual(result["status"], "success")

        finally:
            print("stoping YOLO service")
            process.terminate()

    def test_ocr_serve(self):
        """test the ocr_serve function"""
        # command: python -m unittest tests.test_utils.UtilsTest.test_ocr_serve
        ocr_server_url = "http://172.16.107.15:6006/models/ocr"
        mlm_server_url = "http://172.16.107.15:6006/models/qwen"
        image_path = r"G:\项目成果打包\金融多模态知识库构建与复杂问答检索算法\AI-Claude\data\test\BCE5DDFA2DFE4CB382B18F3DEDFBBA530.png"
        img = Image.open(image_path)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        prompt = """
        # 你是一位图片理解分析师，你的任务是根据给定的图片完成以下任务：
        1. 识别图片中的关键信息和数据，用自然语言描述。
        2. 识别图片中的标题，如果没有则输出'N'。
        # 输出JSON格式：{"output": ..., "title": ...}"""
        payload = {"image": img_b64}
        response = requests.post(ocr_server_url, json=payload, timeout=60)
        data = response.json()
        print(data)
if __name__ == "__main__":
    unittest.main()