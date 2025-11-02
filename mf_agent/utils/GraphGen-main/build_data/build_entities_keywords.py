"""
###为每个document构建一个json文件, json文件中每个元素是当前document下的知识块###
"""

import asyncio
import os
import json
from hashlib import md5
from graphgen.models import OpenAIClient, Tokenizer
from graphgen.bases.datatypes import Chunk
from graphgen.models import LightRAGKGBuilder


def compute_content_hash(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


async def single_knowledge(knowledge_chunk: Chunk):
    tokenizer_instance = Tokenizer(
        model_name="/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-30b-a3b"
    )
    synthesizer_llm_client = OpenAIClient(
        model_name="/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-30b-a3b",
        api_key="dummy",
        base_url="http://172.16.107.15:23333/v1",
        tokenizer=tokenizer_instance,
    )
    kg_builder = LightRAGKGBuilder(llm_client=synthesizer_llm_client, max_loop=3)
    result = await kg_builder.extract_knowledge_chunk(knowledge_chunk)
    return result


def parse_result(result_text):
    entities = []
    keywords = []
    records = result_text.strip().split("##")
    for rec in records:
        rec = rec.strip()
        if not rec:
            continue
        rec = rec.strip("()")
        parts = [p.strip('"') for p in rec.split("<|>")]
        if parts[0] == "entity":
            entities.append({"name": parts[1], "type": parts[2], "desc": parts[3]})
        elif parts[0] == "content_keywords":
            keywords = [k.strip() for k in parts[1].split(",")]
    return entities, keywords


async def process_all_files():
    DATA_PATH = "/data1/nuist_llm/MF-Agent/mf_agent/data/rawDocuments/middle_station"
    SAVE_PATH = "/data1/nuist_llm/MF-Agent/mf_agent/data/knowledge_parsed"
    os.makedirs(SAVE_PATH, exist_ok=True)

    for json_file in os.listdir(DATA_PATH):
        if not json_file.endswith(".json"):
            continue
        print(f"Processing {json_file}")
        with open(os.path.join(DATA_PATH, json_file), "r", encoding="utf-8") as f:
            all_data = json.load(f)

        file_name = all_data["title"]
        data = all_data["content_blocks"]
        save_list = []

        for da in data:
            knowledge_id = da["knowledge_id"]
            meta_data = da["meta_data"]
            hash_id = compute_content_hash(knowledge_id)
            content = (
                f"{meta_data['big_title']}\n{meta_data['mid_title']}\n"
                f"{meta_data['current_title']}\n{da['content']}"
            )
            knowledge_chunk = Chunk(id=hash_id, content=content, metadata=meta_data)

            result = await single_knowledge(knowledge_chunk)
            entities, keywords = parse_result(result)

            save_list.append({
                "id": hash_id,
                "content": content,
                "entities": entities,
                "keywords": keywords
            })

        save_file = os.path.join(SAVE_PATH, f"{file_name}.json")
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(save_list, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {save_file}")


if __name__ == "__main__":
    asyncio.run(process_all_files())
