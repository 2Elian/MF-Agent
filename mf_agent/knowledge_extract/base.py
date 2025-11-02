from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
import os
import json

class BaseIndexBuilder(ABC):
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @abstractmethod
    def build(self, knowledge_units_path: str) -> None:
        raise NotImplementedError