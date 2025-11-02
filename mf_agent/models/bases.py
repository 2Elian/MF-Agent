#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/27 21:35
# @Author  : lizimo@nuist.edu.cn
# @File    : bases.py
# @Description:
import math
from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class Intention:
    id: int
    query_cls: str
    document_title: str
    query: List[Union[str, List[str]]] = field(default_factory=list)
    keywords: List[Union[str, List[str]]] = field(default_factory=list)
    entities: List[Union[str, List[str]]] = field(default_factory=list)


@dataclass
class Question:
    id: int
    query: str

@dataclass
class EsReturn:
    id: str
    content: str
    keywords: List[str]
    score: float

@dataclass
class DetailReturn:
    id: str
    content: str
    source: float
    original_score: float
    vector_score: float

@dataclass
class MultiIntentReturn:
    id: str
    content: str
    source: float
    original_score: float
    vector_score: float


@dataclass
class MultiHopReturn:
    id: str
    content: str
    source: float
    original_score: float
    vector_score: float

@dataclass
class ReasoningReturn:
    id: str
    content: str
    source: float
    original_score: float
    vector_score: float