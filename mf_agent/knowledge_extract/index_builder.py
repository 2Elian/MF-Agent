#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/22 21:23
# @Author  : lizimo@nuist.edu.cn
# @File    : build.py
# @Description: index_builder
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import json
import logging
import os
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)
import networkx as nx
from neo4j import GraphDatabase

from mf_agent.utils.helper import compute_content_hash
from mf_agent.knowledge_extract.base import BaseIndexBuilder

class EmbeddingModel:
    def __init__(self, model_name: str = ""):
        if model_name == "":
            raise NotImplemented("Please set model_name as your embedding model path")
        self.model = SentenceTransformer(model_name, device="cpu")
    def encode(self, text: str):
        return self.model.encode(text, batch_size=64, normalize_embeddings=True).tolist()

class InvertedIndexBuilder(BaseIndexBuilder):
    def __init__(self, es_client: Elasticsearch, logger: logging.Logger):
        super().__init__(logger)
        self.es = es_client

    def build(self, knowledge_units_path: str):
        self.logger.info("Start building inverted index (Elasticsearch).")
        for filename in os.listdir(knowledge_units_path):
            file_path = os.path.join(knowledge_units_path, filename)
            index_name = compute_content_hash(os.path.splitext(filename)[0], prefix="es").strip()
            if not self.es.indices.exists(index=index_name):
                self.es.indices.create(index=index_name, body={
                        "mappings": {
                            "properties": {
                                "id": {"type": "keyword"},
                                "content": {"type": "text"},
                                "entities": {
                                    "type": "nested",
                                    "properties": {
                                        "name": {"type": "keyword"},
                                        "type": {"type": "keyword"},
                                        "desc": {"type": "text"}
                                    }
                                },
                                "keywords": {"type": "keyword"}
                            }
                        }
                })
                self.logger.info(f"Created Elasticsearch index: {index_name}")
            else:
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                knowledge_units = json.load(f)

            for unit in knowledge_units:
                kb_id = unit.get("id") # hash id
                content = unit.get("content", "")
                entities = unit.get("entities", [])
                self.es.index(index=index_name, id=kb_id, body={
                    "id": kb_id,
                    "content": content,
                    "entities": entities,
                    "keywords": [str(k).strip() for k in unit.get("keywords", [])]
                })

        self.logger.info("Elasticsearch inverted index build completed.")

class VectorIndexBuilder(BaseIndexBuilder):
    def __init__(self, milvus_host: str, milvus_port: str, collection_name: str, embedding_model: EmbeddingModel, logger: logging.Logger, create_vec : bool = False):
        super().__init__(logger)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        connections.connect("default", host=milvus_host, port=milvus_port)
        self.fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=1024),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=12304)
        ]
        self.schema = CollectionSchema(fields=self.fields, description="Knowledge embeddings collection")
        if create_vec:
            self.collection = Collection(name=collection_name, schema=self.schema)
            self.logger.info(f"Created new Milvus collection: {collection_name}")

    def build(self, knowledge_units_path: str):
        all_ids, all_embeddings, all_contents = [], [], []

        self.logger.info("Start building vector index (Milvus).")

        for filename in os.listdir(knowledge_units_path):
            file_path = os.path.join(knowledge_units_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                knowledge_units = json.load(f)

            for unit in knowledge_units:
                kb_id = unit.get("id") # hash id
                content = unit.get("content", "")
                max_bytes = 12304
                content = content.encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore')
                embedding = self.embedding_model.encode(content)

                all_ids.append(kb_id)
                all_embeddings.append(embedding)
                all_contents.append(content)
        
        self.collection.insert([all_ids, all_embeddings, all_contents])

        self.collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
        )

        self.collection.load()
        self.logger.info(f"Inserted {len(all_ids)} embeddings into Milvus collection '{self.collection_name}'.")

class GraphIndexBuilder(BaseIndexBuilder):
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, logger: logging.Logger = None):
        super().__init__(logger)
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.logger.info(f"Connected to Neo4j at {neo4j_uri}")

    def _import_graphml_file(self, file_path: str):
        G = nx.read_graphml(file_path)
        with self.driver.session() as session:
            for node_id, node_data in G.nodes(data=True):
                props = ", ".join([f"{k}: ${k}" for k in node_data.keys()])
                cypher = f"MERGE (n:Node {{id: $id}}) SET n += {{{props}}}"
                params = {"id": node_id, **node_data}
                session.run(cypher, params)
            for u, v, edge_data in G.edges(data=True):
                props = ", ".join([f"{k}: ${k}" for k in edge_data.keys()])
                cypher = (
                    "MATCH (a:Node {id: $u}), (b:Node {id: $v}) "
                    "MERGE (a)-[r:RELATED]->(b) "
                    "SET r += {" + props + "}"
                )
                params = {"u": u, "v": v, **edge_data}
                session.run(cypher, params)
        self.logger.info(f"Imported GraphML file: {file_path}")

    def build(self, graphml_dir: str):
        files = [f for f in os.listdir(graphml_dir) if f.endswith(".graphml")]
        if not files:
            self.logger.warning(f"No GraphML files found in {graphml_dir}")
            return
        self.logger.info(f"Found {len(files)} GraphML files in {graphml_dir}. Starting import...")
        for file_name in files:
            file_path = os.path.join(graphml_dir, file_name)
            self._import_graphml_file(file_path)
        self.logger.info("All GraphML files imported successfully.")

    def close(self):
        self.driver.close()
        self.logger.info("Neo4j driver closed.")

class KnowledgeBase:
    def __init__(self, embedding_model_name="", create_vec: bool = False):
        self.logger = logging.getLogger(name="KnowledgeBase")
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
        self.es = Elasticsearch("http://your_ip:9200")
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.inverted_index_builder = InvertedIndexBuilder(self.es, self.logger)
        self.vector_index_builder = VectorIndexBuilder(
            milvus_host="your_ip",
            milvus_port="19530",
            collection_name="knowledge_embeddings",
            embedding_model=self.embedding_model,
            logger=self.logger,
            create_vec = create_vec
        )
        self.graph_index_builder = GraphIndexBuilder(
            neo4j_uri="bolt://your_ip:7687",
            neo4j_user="neo4j",
            neo4j_password="MyStrongPassword123",
            logger=self.logger
        )

    def build_indexes(self, knowledge_units_path: str, knowledge_graph_path: str = None):
        self.logger.info("Start Building Inverted Index")
        self.inverted_index_builder.build(knowledge_units_path)
        self.logger.info("Start Building Embedding Index")
        self.vector_index_builder.build(knowledge_units_path)
        self.logger.info("Start Building graph Index")
        self.graph_index_builder.build(knowledge_graph_path)
        self.graph_index_builder.close()
        self.logger.info("Congratulations on completing this task!")


if __name__ == "__main__":
    kb = KnowledgeBase(embedding_model_name="your_embedding_path")
    kb.build_indexes(knowledge_units_path="/MF-Agent/mf_agent/data/knowledge_unit", knowledge_graph_path="/MF-Agent/mf_agent/data/graph")