#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/19 18:14
# @Author  : lizimo@nuist.edu.cn
# @File    : main_parser.py
# @Description:
import logging
import os
import io
import json
from typing import Dict, Any
import re
from pathlib import Path

from src.utils.helper import new_doc_id,new_doc_id_from_title


class HeadParser:
    def __init__(self, output_assets_dir: Path):
        self.output_assets_dir = output_assets_dir
        self._parser_make_dir(self.output_assets_dir)
        self.logger = logging.getLogger(__name__)
        self.logger.info("@Elian正在求职!!\n 我感兴趣的方向是大模型应用开发/大模型强化学习算法 \n 如果您有合适的岗位愿意给我一个面试的机会的话 请联系邮箱:lizimo@nusit.edu.cn")

    def _parser_make_dir(self, dir_name: Path):
        dir_name.mkdir(parents=True, exist_ok=True)

    def clean_title(self, title: str) -> str:
        title = re.sub(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ]", "", title)
        title = re.sub(r"\d+", "", title)
        title = re.sub(r"[一二三四五六七八九十零]", "", title)
        title = title.replace("、", "").replace(".", "")
        return title.strip()
    def _save_json(self, res):
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(i) for i in obj]
            else:
                return obj

        res = convert_paths(res)
        output_path = self.output_assets_dir / f"{res['doc_id']}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        self.logger.info(f"JSON saved successfully: {output_path}")

    def parse_md_01(self, path: str) -> Dict[str, Any]:
        """
        # 大标题
        **int.int** 知识点标题
        适配比较老的解析方法
        """
        self.logger.warning("This method has been deprecated. Please use parser_md_02 method for the latest documentation.")
        title = os.path.splitext(os.path.basename(path))[0]
        title = title.replace(" ", "").replace("　", "")
        doc_id = new_doc_id_from_title(title)
        content_blocks = []

        with open(path, "r", encoding="utf-8") as f:
            md_lines = f.readlines()

        current_big_title = None
        current_subsection = None

        for line in md_lines:
            line = line.rstrip()

            if not line.strip():
                continue

            # 匹配 ## -> 大标题
            big_title_match = re.match(r"^#{2}\s*(?:\*\*)?(.+?)(?:\*\*)?\s*$", line) # 大标题在md中以##形式出现
            if big_title_match:
                if current_subsection is not None and current_subsection["content"]:
                    current_subsection["content"] = "\n".join(current_subsection["content"])
                    content_blocks.append(current_subsection)
                    current_subsection = None
                current_big_title = self.clean_title(big_title_match.group(1).strip())
                continue

            # 匹配 **数字.数字 标题内容** -> 当前标题
            sub_title_match = re.match(r"^\s*\*\*(\d+\.\d+)\s*(.*?)\*\*\s*$", line) # 小标题可以以**int.int or int.int.int**形式出现
            if sub_title_match:
                if current_subsection is not None and current_subsection["content"]:
                    current_subsection["content"] = "\n".join(current_subsection["content"])
                    content_blocks.append(current_subsection)
                subsection_title = self.clean_title(sub_title_match.group(2).strip())
                knowledge_id = f"{doc_id}_sec{len(content_blocks) + 1}"
                current_subsection = {
                    "knowledge_id": knowledge_id, # 知识点id 便于后续做倒排索引标记
                    "meta_data": {
                        "file_path": path, # 当前知识点属于哪个文档
                        "file_name": title, # 当前知识点所属文档的标题
                        "big_title": current_big_title,
                        "mid_title": " ",
                        "current_title": subsection_title
                    },
                    "content": []
                }
                continue
            # 匹配 **数字.数字.数字 标题内容** -> 当前标题
            number_title_match = re.match(r"^(\d+(?:\.\d+)*)[\.\s]*(.+?)[:：]?\s*$", line)
            if number_title_match:
                if current_subsection is not None and current_subsection["content"]:
                    current_subsection["content"] = "\n".join(current_subsection["content"])
                    content_blocks.append(current_subsection)

                subsection_title = self.clean_title(number_title_match.group(2).strip())
                knowledge_id = f"{doc_id}_sec{len(content_blocks) + 1}"
                current_subsection = {
                    "knowledge_id": knowledge_id,
                    "meta_data": {
                        "file_path": path,
                        "file_name": title,
                        "big_title": current_big_title,
                        "mid_title": " ",
                        "current_title": subsection_title
                    },
                    "content": []
                }
                continue

            # 解决无小标题现象
            if current_big_title and current_subsection is None:
                knowledge_id = f"{doc_id}_sec{len(content_blocks) + 1}"
                current_subsection = {
                    "knowledge_id": knowledge_id,
                    "meta_data": {
                        "file_path": path,
                        "file_name": title,
                        "big_title": current_big_title,
                        "mid_title": " ",
                        "current_title": " "
                    },
                    "content": []
                }

            # 追加内容到当前小节
            if current_subsection is not None:
                current_subsection["content"].append(line)
        if current_subsection is not None and current_subsection["content"]:
            current_subsection["content"] = "\n".join(current_subsection["content"])
            content_blocks.append(current_subsection)

        out = {
            "doc_id": doc_id,
            "file_path": path,
            "file_type": "md",
            "title": title,
            "metadata": {},
            "content_blocks": content_blocks
        }
        self._save_json(out)
        return out

    def parse_md_02(self, path: str) -> Dict[str, Any]:
        """
        Markdown结构：
        # 大标题
        ## 中标题
        ### 知识点最小单元标题
        """
        title = os.path.splitext(os.path.basename(path))[0]
        title = title.replace(" ", "").replace("　", "")
        doc_id = new_doc_id_from_title(title)
        content_blocks = []

        with open(path, "r", encoding="utf-8") as f:
            md_lines = f.readlines()

        current_big_title = None
        current_mid_title = None
        current_subsection = None

        for line in md_lines:
            line = line.rstrip()

            if not line.strip():
                continue

            # # -> 大标题
            big_title_match = re.match(r"^#\s+(.+)$", line)
            if big_title_match:
                if current_subsection is not None and current_subsection["content"]:
                    current_subsection["content"] = "\n".join(current_subsection["content"])
                    content_blocks.append(current_subsection)
                    current_subsection = None

                current_big_title = self.clean_title(big_title_match.group(1).strip())
                current_mid_title = None
                continue

            # ## -> 中标题
            mid_title_match = re.match(r"^##\s+(.+)$", line)
            if mid_title_match:
                if current_subsection is not None and current_subsection["content"]:
                    current_subsection["content"] = "\n".join(current_subsection["content"])
                    content_blocks.append(current_subsection)
                    current_subsection = None

                current_mid_title = self.clean_title(mid_title_match.group(1).strip())
                continue

            # ### -> 知识点最小单元标题
            sub_title_match = re.match(r"^###\s+(.+)$", line)
            if sub_title_match:
                if current_subsection is not None and current_subsection["content"]:
                    current_subsection["content"] = "\n".join(current_subsection["content"])
                    content_blocks.append(current_subsection)

                subsection_title = self.clean_title(sub_title_match.group(1).strip())
                knowledge_id = f"{doc_id}_sec{len(content_blocks) + 1}"
                current_subsection = {
                    "knowledge_id": knowledge_id,
                    "meta_data": {
                        "file_path": path,
                        "file_name": title,
                        "big_title": current_big_title,
                        "mid_title": current_mid_title,
                        "current_title": subsection_title
                    },
                    "content": []
                }
                continue

            # 解决无小标题现象
            if (current_big_title or current_mid_title) and current_subsection is None:
                knowledge_id = f"{doc_id}_sec{len(content_blocks) + 1}"
                current_subsection = {
                    "knowledge_id": knowledge_id,
                    "meta_data": {
                        "file_path": path,
                        "file_name": title,
                        "big_title": current_big_title,
                        "mid_title": current_mid_title,
                        "current_title": " "
                    },
                    "content": []
                }
            # 处理无中标题现象
            if current_big_title and current_mid_title is None:
                knowledge_id = f"{doc_id}_sec{len(content_blocks) + 1}"
                current_subsection = {
                    "knowledge_id": knowledge_id,
                    "meta_data": {
                        "file_path": path,
                        "file_name": title,
                        "big_title": current_big_title,
                        "mid_title": " ",
                        "current_title": " "
                    },
                    "content": []
                }

            if current_subsection is not None:
                current_subsection["content"].append(line)
        if current_subsection is not None and current_subsection["content"]:
            current_subsection["content"] = "\n".join(current_subsection["content"])
            content_blocks.append(current_subsection)

        out = {
            "doc_id": doc_id,
            "file_path": path,
            "file_type": "md",
            "title": title,
            "metadata": {},
            "content_blocks": content_blocks
        }
        self._save_json(out)
        return out
