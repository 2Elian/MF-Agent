#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/19 19:55
# @Author  : lizimo@nuist.edu.cn
# @File    : test_parser.py
# @Description:
import unittest
from pathlib import Path
from src.document_parser.mf_parser import HeadParser

class TestDocumentParser(unittest.TestCase):
    def setUp(self):
        """测试初始化"""
        print("elian: lizimo@nuist.edu.cn")
        self.document = HeadParser(Path(r"G:\项目成果打包\金融多模态知识库构建与复杂问答检索算法\AI-Claude\data\test\assets"))

    def test_docx_parser(self):
        """tester docx parser"""
        # command: python -m unittest tests.test_parser.TestDocumentParser.test_docx_parser
        res = self.document.parse_docx(r"G:\项目成果打包\金融多模态知识库构建与复杂问答检索算法\AI-Claude\data\test\丰收e网企业网上银行使用手册.docx")
        print(res)

    def test_pdf_parser(self):
        """tester docx parser"""
        # command: python -m unittest tests.test_parser.TestDocumentParser.test_pdf_parser
        res = self.document.parse_pdf(r"G:\项目成果打包\金融多模态知识库构建与复杂问答检索算法\AI-Claude\data\test\上海浦东发展银行 B2B 网上支付操作手册.pdf")
        print(res)


if __name__ == "__main__":
    unittest.main()