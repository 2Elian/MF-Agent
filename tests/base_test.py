#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/19 20:01
# @Author  : lizimo@nuist.edu.cn
# @File    : base_test.py
# @Description:
import unittest
import os


class BaseTest(unittest.TestCase):
    def setUp(self):
        """测试初始化"""
        print("elian: lizimo@nuist.edu.cn")

    def test_image_classify(self):
        """tester image_classify"""
        # command: python -m unittest tests.test_document_parser.TestDocumentParser.test_image_classify
        pass
if __name__ == "__main__":
    unittest.main()