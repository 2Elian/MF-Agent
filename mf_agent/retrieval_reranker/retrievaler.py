#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/22 22:13
# @Author  : lizimo@nuist.edu.cn
# @File    : retrievaler.py
# @Description: 检索器 + 负样本构造器
import os
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Union
import torch
import logging
from elasticsearch import Elasticsearch
from pymilvus import Collection
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util

from mf_agent.models.bases import Intention, EsReturn, DetailReturn, MultiIntentReturn, MultiHopReturn, ReasoningReturn
from mf_agent.knowledge_extract.index_builder import KnowledgeBase
from mf_agent.models.bases import Intention
from mf_agent.utils.helper import compute_content_hash

class MultiPolicyRetrievaler:
    def __init__(self, knowledge_base, embedding_model, batch_size=4, device="cuda", graph_raw_path: str = "/data/graph"):
        self.kb = knowledge_base
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.device = device
        self.es = knowledge_base.es
        self.milvus_collection = Collection("knowledge_embeddings")
        self.graph_driver = knowledge_base.graph_index_builder.driver
        self.logger = logging.getLogger("MF-Agent- MultiPolicyRetrievaler")
        self.graph_raw_path = graph_raw_path

    # Distribute based on intent type
    def retrieve_knowledge(self, intention: Intention) -> Union[List[DetailReturn], List[MultiIntentReturn], List[MultiHopReturn], List[ReasoningReturn]]:
        question_type = intention.query_cls
        question = intention.query
        keywords = intention.keywords
        entities = intention.entities
        document_title = intention.document_title
        range_title = compute_content_hash(document_title)

        if question_type == "detail":
            return self._retrieve_detail(query=question, keywords=keywords, entities=entities, range_title=range_title)
        elif question_type == "multi_intent":
            return self._retrieve_multi_intent(questions=question, keywords=keywords, entities=entities, range_title=range_title)
        elif question_type == "multi_hop":
            return self._retrieve_multi_hop(question, entities, range_title)
        elif question_type == "reasoning":
            return self._retrieve_reasoning(question)
        else:
            raise ValueError(f"query_cls:{question_type} not implemented")

    def _retrieve_detail(self, query, keywords, entities, range_title, top_k=10) -> List[DetailReturn]:
        es_results, len_hit = self.es_retrievaler(query, keywords, entities, range_title)
        if not es_results or len_hit==0:
            return []
        top_k = min(top_k, len_hit)
        sorted_results = self.keyword_rerank(es_results, keywords)
        top_docs = sorted_results[:top_k]
        final_results = self.embedding_retrievaler(query, top_docs)
        if len(final_results)>3:
            return final_results[:3]
        else:
            return final_results

    def _retrieve_multi_intent(self, questions: List[str], keywords: List[List[str]], entities: List[List[str]], range_title):
        # TODO
        results = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, question in enumerate(questions):
                futures.append(executor.submit(
                    self._retrieve_detail,
                    query=question,
                    keywords=keywords[i] if keywords else [],
                    entities=entities[i] if entities else [],
                    range_title=range_title
                ))
            for future in futures:
                results.extend(future.result())
        return self.embedding_retrievaler_multi(questions, results)

    def _retrieve_multi_hop(self, question, entity, range_title, max_hops=2, top_k=5):
        # TODO
        # step1 -> 找到图的位置
        graph_path = os.path.join(self.graph_raw_path, range_title)
        # step2 -> 实例化图检索器
        retriever = GraphRetriever(graph_path, self.embedding_model)
        # 返回结果
        docs = retriever.knowledge_graph_multi_hop_retrieval(question, entity, max_hops=max_hops, top_k=top_k)
        return docs

    def _retrieve_reasoning(self, question):
        # TODO 细节检索器+LLM推理
        return 1


    def es_retrievaler(self, query, keywords, entities, range_title, size=20) -> Tuple[List[EsReturn], int]:
        should_clauses = []
        # entity strict match
        for ent in entities:
            should_clauses.append({
                "nested": {
                    "path": "entities",
                    "query": {
                        "term": {"entities.name": ent}
                    }
                }
            })
        
        # entity ambiguous match
        for ent in entities:
            should_clauses.append({
                "nested": {
                    "path": "entities", 
                    "query": {
                        "wildcard": {
                            "entities.name": f"*{ent}*"
                        }
                    }
                }
            })
        should_clauses.append({
            "match": {
                "content": {
                    "query": " ".join(entities + keywords),
                    "operator": "or"
                }
            }
        })

        body = {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            },
            "_source": ["id", "content", "keywords"],
            "size": size
        }

        try:
            res = self.es.search(index=range_title.lower(), body=body)
        except Exception as e:
            self.logger.warning(f"Index {range_title.lower()} not found: {e}")
            return []

        hits = res["hits"]["hits"]
        self.logger.info(f"ES hits: {len(hits)}")
        docs = [
            EsReturn(
                id=hit["_id"],
                content=hit["_source"]["content"],
                keywords=hit["_source"].get("keywords", []),
                score=hit["_score"]
            )
            for hit in hits
        ]

        return docs, len(hits)

    def keyword_rerank(self, es_results, keywords):
        """
        三个维度进行keywords排序
        """
        if not keywords or not es_results:
            return es_results
        
        scored_docs = []
        
        for doc in es_results:
            # 维度1: 关键词匹配分数
            keyword_match_score = self._calculate_keyword_match_score(doc, keywords)
            # 维度2: 正文匹配分数
            content_relevance_score = self._calculate_content_relevance(doc, keywords)
            # 维度3: 关键词所在正文出现位置分数
            position_score = self._calculate_position_score(doc, keywords)
            combined_score = (
                doc.score * 1 +  
                keyword_match_score * 4 + 
                content_relevance_score * 2 + 
                position_score * 0.5 
            )
            
            scored_docs.append((doc, combined_score))
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs]
    
    def _calculate_keyword_match_score(self, doc, keywords):
        """计算关键词精确匹配得分"""
        if not doc.keywords:
            return 0
        
        score = 0
        doc_keywords_lower = [kw.lower() for kw in doc.keywords]
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # 精确匹配
            if keyword_lower in doc_keywords_lower:
                score += 3
            
            # 部分匹配
            for doc_kw in doc_keywords_lower:
                if keyword_lower in doc_kw or doc_kw in keyword_lower:
                    score += 1
                    break
        
        return min(score, 10)

    def _calculate_content_relevance(self, doc, keywords):
        """计算内容相关性得分"""
        if not doc.content:
            return 0
        
        score = 0
        content_lower = doc.content.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # 关键词在内容中出现的频率
            count = content_lower.count(keyword_lower)
            score += min(count, 3)
        
        return min(score, 15)  # 限制最大得分

    def _calculate_position_score(self, doc, keywords):
        """计算位置得分（关键词出现在开头得分更高）"""
        if not doc.content:
            return 0
        
        score = 0
        content_lower = doc.content.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            position = content_lower.find(keyword_lower)
            
            if position != -1:
                # 位置越靠前得分越高
                if position < 50:  # 前50个字符
                    score += 2
                elif position < 200:  # 前200个字符
                    score += 1
        return score

    def embedding_retrievaler(self, query: str, top_docs: List[EsReturn]) -> List[DetailReturn]:
        if not top_docs:
            return []
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            search_params = {
                "metric_type": "IP",  # 内积相似度
                "params": {"nprobe": 10} # 在10个聚类里面搜索
            }
            doc_ids = [doc.id for doc in top_docs]

            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=len(doc_ids),  # 返回所有文档
                expr=f'id in {doc_ids}',  # 只搜索ES结果中的文档
                output_fields=["id", "content"]
            )
            id_to_similarity = {}
            for hits in results:
                for hit in hits:
                    id_to_similarity[hit.entity.get('id')] = hit.score
            
            # 重新排序文档
            reranked_docs = []
            for doc in top_docs:
                similarity_score = id_to_similarity.get(doc.id, 0)
                # 创建新的EsReturn对象，包含向量相似度得分
                reranked_doc = DetailReturn(
                    id=doc.id,
                    content=doc.content,
                    score=similarity_score,
                    original_score=doc.score,
                    vector_score=similarity_score
                )
                reranked_docs.append(reranked_doc)
            reranked_docs.sort(key=lambda x: x.vector_score, reverse=True)
            self.logger.info(f"Vector reranking completed. Top similarity: {reranked_docs[0].vector_score if reranked_docs else 0}")
            
            return reranked_docs
            
        except Exception as e:
            self.logger.error(f"Vector retrieval failed: {e}")
            # 如果向量检索失败，返回原始排序结果
            return top_docs

    def embedding_retrievaler_multi(self, queries, documents):
        """多意图 query 的 Embedding 合并打分"""
        doc_contents = [doc["content"] for doc in documents]
        doc_embeddings = self._batch_encode(doc_contents, self.embedding_model)
        all_query_embeddings = [self._batch_encode([q], self.embedding_model)[0] for q in queries]

        final_scores = []
        for doc_emb in doc_embeddings:
            score = max(torch.nn.functional.cosine_similarity(
                torch.tensor([doc_emb]), torch.tensor(all_query_embeddings), dim=1
            )).item()
            final_scores.append(score)

        for doc, score in zip(documents, final_scores):
            doc["embedding_score"] = score
        return sorted(documents, key=lambda x: x["embedding_score"], reverse=True)

    def _batch_encode(self, texts, model):
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            with torch.no_grad():
                emb = model.encode(batch, device=self.device, normalize_embeddings=True)
            all_embeddings.extend(emb)
        return all_embeddings

    def knowledge_graph_retrievaler(self, question):
        """Neo4j 知识图谱检索"""
        query_text = f"MATCH (n) WHERE n.name CONTAINS '{question}' RETURN n.id AS id, n.name AS name, n.desc AS desc LIMIT 10"
        with self.graph_driver.session() as session:
            result = session.run(query_text)
            records = result.data()

        docs = [
            {
                "id": rec["id"],
                "content": f"{rec.get('name', '')} - {rec.get('desc', '')}"
            }
            for rec in records
        ]
        return docs

    def get_range_title(self, document_title):
        return document_title.lower() if document_title else "default"


class GraphRetriever:
    def __init__(self, graphml_path: str,
                 embedding_model):
        assert os.path.exists(graphml_path), f"GraphML 文件不存在: {graphml_path}"
        self.graph = nx.read_graphml(graphml_path)
        self.model = embedding_model
        print(f"已加载图谱：{graphml_path}")
        print(f"包含 {self.graph.number_of_nodes()} 个节点，{self.graph.number_of_edges()} 条边")

    def get_node_text(self, node_id: str) -> str:
        node_data = self.graph.nodes[node_id]
        name = node_data.get("name", "")
        desc = node_data.get("desc", "")
        label = node_data.get("label", "")
        return f"{name} {label} {desc}".strip()

    def find_entity_nodes(self, entities: list[str]) -> list[str]:
        matched_nodes = []
        for node_id, data in self.graph.nodes(data=True):
            text = self.get_node_text(node_id)
            for ent in entities:
                if ent in text:
                    matched_nodes.append(node_id)
        return list(set(matched_nodes))

    def multi_hop_expand(self, start_nodes: list[str], max_hops: int = 2) -> list[str]:
        visited = set(start_nodes)
        current_layer = set(start_nodes)

        for _ in range(max_hops):
            next_layer = set()
            for node in current_layer:
                neighbors = list(self.graph.neighbors(node))
                for n in neighbors:
                    if n not in visited:
                        next_layer.add(n)
            visited |= next_layer
            current_layer = next_layer
        return list(visited)

    def rank_nodes_by_semantic(self, query: str, candidate_nodes: list[str], top_k: int = 10):
        texts = [self.get_node_text(nid) for nid in candidate_nodes]
        query_emb = self.model.encode(query, normalize_embeddings=True)
        node_embs = self.model.encode(texts, normalize_embeddings=True)
        scores = util.dot_score(query_emb, node_embs)[0].cpu().tolist()

        ranked = sorted(zip(candidate_nodes, texts, scores), key=lambda x: x[2], reverse=True)[:top_k]
        return [{"id": n, "content": t, "score": round(s, 4)} for n, t, s in ranked]

    def knowledge_graph_multi_hop_retrieval(
            self, query: str, entities: list[str], max_hops: int = 2, top_k: int = 5
    ):
        start_nodes = self.find_entity_nodes(entities)
        if not start_nodes:
            print("没有命中实体节点，尝试全图语义检索")
            all_nodes = list(self.graph.nodes())
            return self.rank_nodes_by_semantic(query, all_nodes, top_k=top_k)

        candidate_nodes = self.multi_hop_expand(start_nodes, max_hops=max_hops)

        ranked_results = self.rank_nodes_by_semantic(query, candidate_nodes, top_k=top_k)
        return ranked_results

    def knowledge_graph_retrieval(self, query: str, top_k: int = 5):
        node_texts = []
        node_ids = list(self.graph.nodes())

        for node_id in node_ids:
            text = self.get_node_text(node_id)
            if text.strip():
                node_texts.append(text)
            else:
                node_texts.append(node_id)

        query_emb = self.model.encode(query, normalize_embeddings=True)
        node_embs = self.model.encode(node_texts, normalize_embeddings=True)
        scores = util.dot_score(query_emb, node_embs)[0].cpu().tolist()

        ranked = sorted(zip(node_ids, node_texts, scores), key=lambda x: x[2], reverse=True)[:top_k]
        results = [{"id": node_id, "content": text, "score": round(score, 4)} for node_id, text, score in ranked]
        return results

if __name__ == "__main__":
    kb = KnowledgeBase(embedding_model_name="your_embedding_model")
    retrievaler = MultiPolicyRetrievaler(kb, kb.embedding_model)
    # example
    intention = Intention(
        id = 1,
        query_cls="detail",
        query="中国银行B2B开通流程中，需要的材料有什么？",
        keywords=["中国银行", "B2B", "开通流程", "材料"],
        entities=["中国银行", "B2B"],
        document_title="中国银行B2B开通流程"
    )
    results = retrievaler.retrieve_knowledge(intention)
    print(results)