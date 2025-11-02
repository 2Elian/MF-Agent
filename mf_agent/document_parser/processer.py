#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/20 20:05
# @Author  : lizimo@nuist.edu.cn
# @File    : main.py
# @Description:
import os
from pathlib import Path
from tqdm import tqdm
from src.document_parser.mf_parser import HeadParser

if __name__ == "__main__":
    output_dir = r""
    parser = HeadParser(output_assets_dir=Path(output_dir))
    markdown_path = r""
    markdown_path2 = r""
    for mk_filename in tqdm(os.listdir(markdown_path)):
        mk_filename_clean = mk_filename.strip()
        input_file_path = Path(markdown_path) / mk_filename / f"{mk_filename_clean}.md"
        if not input_file_path.exists():
            print(f"文件不存在: {input_file_path}")
            continue
        parser.parse_md_01(input_file_path)

    for mk_filename in tqdm(os.listdir(markdown_path2)):
        mk_filename_clean = mk_filename.strip()
        input_file_path = Path(markdown_path2) /  mk_filename / "auto" / f"{mk_filename_clean}.md"
        if not input_file_path.exists():
            print(f"文件不存在: {input_file_path}")
            continue
        parser.parse_md_02(input_file_path)

