"""
###为每个document构建一个graph###
"""
import asyncio
import os
from typing import List
import json
from hashlib import md5
from graphgen.models import OpenAIClient, Tokenizer
from graphgen.bases.datatypes import Chunk
from graphgen.models import LightRAGKGBuilder
from graphgen.operators.build_kg import build_kg
from graphgen.models import (
    JsonKVStorage,
    JsonListStorage,
    NetworkXStorage,
    OpenAIClient,
    Tokenizer
)
def compute_content_hash(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


async def build_graph(knowledge_chunk: List[Chunk], namespace: str):
    graph_storage: NetworkXStorage = NetworkXStorage(
    "/data1/nuist_llm/MF-Agent/mf_agent/data/graph", namespace=namespace
)
    tokenizer_instance = Tokenizer(
        model_name="/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-30b-a3b"
    )
    synthesizer_llm_client = OpenAIClient(
        model_name="/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-30b-a3b",
        api_key="dummy",
        base_url="http://172.16.107.15:23333/v1",
        tokenizer=tokenizer_instance,
    )
    result = await build_kg(synthesizer_llm_client, graph_storage, knowledge_chunk, namespace=namespace)
    return result



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
        
        chunks = []
        for da in data:
            hash_id = compute_content_hash(da["knowledge_id"])
            meta_data = da["meta_data"]
            content = (
                f"{meta_data['big_title']}\n{meta_data['mid_title']}\n"
                f"{meta_data['current_title']}\n{da['content']}"
            )
            chunks.append(
                Chunk(id=hash_id, content=content)
            )
        await build_graph(chunks, file_name)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(process_all_files())
