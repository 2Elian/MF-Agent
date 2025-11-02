import re
import json
from typing import List, Dict, Any, Tuple

class MfAgentIntentionRecognitionReward:
    """
    期望模型能够输出的统一格式：
    <question_type> cls str </question_type> and cls in [detail|multi_intent|multi_hop|reasoning]
    <document_name>str</document_name>
    <ner_entities>List</ner_entities>
    if cls == detail:
        <keywords> </keywords>
    elif cls == multi_intent:
        <num> </num>
        <information>
        List[Dict]{
                "question": "拆解后的问题1",
                "entity": ["实体A", "实体B"],
                "keywords": ["关键词+实体内容"]}
        </information>
    elif cls == multi_hop:
        <step_num> </step_num>
        <information>
        List[Dict]{
                "question": "拆解后的问题1",
                "entity": ["实体A", "实体B"],
                "keywords": ["关键词+实体内容"]}
        </information>
    elif cls == reasoning:
        <step_num> </step_num>
        <information>
        List[Dict]{
                "question": "拆解后的问题1",
                "entity": ["实体A", "实体B"],
                "keywords": ["关键词+实体内容"]}
        </information>
    然后解析成json       
    {
        "question_type": "detail|multi_intent|multi_hop|reasoning",
        "position_info": {
            "document_name": "文档名称",
            "ner_entities": ["实体1", "实体2", "..."] # 这个是NER模型实现给好的 让RLHF之后仍输出这个格式是为了让模型只限于我们给定的实体里面思考 不能脱离这些实体思考
        },
        "processing_info": {
            "keywords": ["关键词1", "关键词2"]
            },
        "multi_intent_info": {
        "num": 2,
        "information": [
            {
            "question": "拆解后的问题1",
            "entity": ["实体A", "实体B"],
            "detail": ["关键词+实体内容"]
            }
        ]
        },
        "multi_hop_info": {
        "step_num": 3,
        "information": [
            {
            "step": 1,
            "detail": ["关键词+实体内容"]
            }
        ]
        },
        "reasoning_info": {
        "step_num": 2,
        "information": [
            {
            "step": 1,
            "detail": ["关键词+实体内容"]
            }
        ],
        "finally": "最终推理问题"
        }
    }
    }
    
    """
    VALID_QUESTION_TYPES = ["detail", "multi_intent", "multi_hop", "reasoning"]
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
    def parse_model_output(self, generated_text: str) -> Dict[str, Any]:
        try:
            parsed_data = {}
            question_type_match = re.search(r'<question_type>(.*?)</question_type>', generated_text, re.DOTALL)
            if question_type_match:
                parsed_data['question_type'] = question_type_match.group(1).strip()
            else:
                parsed_data['question_type'] = "unknown"

            doc_match = re.search(r'<document_name>(.*?)</document_name>', generated_text, re.DOTALL)
            if doc_match:
                parsed_data['document_name'] = doc_match.group(1).strip()
            ner_match = re.search(r'<ner_entities>(.*?)</ner_entities>', generated_text, re.DOTALL)
            if ner_match:
                try:
                    parsed_data['ner_entities'] = json.loads(ner_match.group(1).strip())
                except:
                    parsed_data['ner_entities'] = []

            if parsed_data['question_type'] == 'detail':
                self._parse_detail_info(generated_text, parsed_data)
            elif parsed_data['question_type'] == 'multi_intent':
                self._parse_multi_intent_info(generated_text, parsed_data)
            elif parsed_data['question_type'] == 'multi_hop':
                self._parse_multi_hop_info(generated_text, parsed_data)
            elif parsed_data['question_type'] == 'reasoning':
                self._parse_reasoning_info(generated_text, parsed_data)
            
            return parsed_data
            
        except Exception as e:
            return {
                'question_type': 'unknown',
                'document_name': '',
                'ner_entities': [],
                'parse_error': str(e)
            }
    
    def _parse_detail_info(self, text: str, parsed_data: Dict):
        keywords_match = re.search(r'<keywords>(.*?)</keywords>', text, re.DOTALL)
        if keywords_match:
            try:
                parsed_data['keywords'] = json.loads(keywords_match.group(1).strip())
            except:
                parsed_data['keywords'] = []
    
    def _parse_multi_intent_info(self, text: str, parsed_data: Dict):
        # 提取num
        num_match = re.search(r'<num>(\d+)</num>', text)
        if num_match:
            parsed_data['num'] = int(num_match.group(1))
        info_match = re.search(r'<information>(.*?)</information>', text, re.DOTALL)
        if info_match:
            try:
                parsed_data['information'] = json.loads(info_match.group(1).strip())
            except:
                parsed_data['information'] = []
    
    def _parse_multi_hop_info(self, text: str, parsed_data: Dict):
        step_num_match = re.search(r'<step_num>(\d+)</step_num>', text)
        if step_num_match:
            parsed_data['step_num'] = int(step_num_match.group(1))
        info_match = re.search(r'<information>(.*?)</information>', text, re.DOTALL)
        if info_match:
            try:
                parsed_data['information'] = json.loads(info_match.group(1).strip())
            except:
                parsed_data['information'] = []
    
    def _parse_reasoning_info(self, text: str, parsed_data: Dict):
        step_num_match = re.search(r'<step_num>(\d+)</step_num>', text)
        if step_num_match:
            parsed_data['step_num'] = int(step_num_match.group(1))
        
        info_match = re.search(r'<information>(.*?)</information>', text, re.DOTALL)
        if info_match:
            try:
                parsed_data['information'] = json.loads(info_match.group(1).strip())
            except:
                parsed_data['information'] = []
        finally_match = re.search(r'<finally>(.*?)</finally>', text, re.DOTALL)
        if finally_match:
            parsed_data['finally'] = finally_match.group(1).strip()
    
    def calculate_reward(self, generated_text: str, ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        generated_output = self.parse_model_output(generated_text)
        rewards = {}
        discount_factor = 1.0  # 奖励折扣因子
        
        # 1. 类型分类奖励
        type_reward, type_discount = self._calculate_type_reward(
            generated_output.get('question_type', 'unknown'),
            ground_truth.get('question_type')
        )
        rewards['type_reward'] = type_reward
        discount_factor = type_discount
        
        # 2. 文档名称奖励
        rewards['document_reward'] = self._calculate_document_reward(
            generated_output.get('document_name', ''),
            ground_truth.get('document_name', '')
        ) * discount_factor
        
        # 3. 实体一致性奖励
        entity_reward = self._calculate_entity_consistency(
            generated_output.get('ner_entities', []),
            generated_output
        )
        rewards['entity_reward'] = entity_reward * discount_factor
        
        # 4. 类型特定的奖励
        specific_reward = self._calculate_specific_reward(generated_output, ground_truth)
        rewards['specific_reward'] = specific_reward * discount_factor
        
        # 5. 格式完整性奖励
        format_reward = self._calculate_format_reward(generated_output)
        rewards['format_reward'] = format_reward * discount_factor
        
        total_reward = sum(rewards.values())
        
        return total_reward, rewards
    
    def _calculate_type_reward(self, generated_type: str, ground_truth_type: str) -> Tuple[float, float]:
        """
        计算类型奖励和折扣因子
        返回: (奖励分数, 折扣因子)
        """
        # 检查是否在预设列表中
        if generated_type not in self.VALID_QUESTION_TYPES:
            # 不在预设列表中，奖励大幅打折
            return -1.0, 0.3  # 负奖励 + 70%折扣
        
        # 在预设列表中
        if generated_type == ground_truth_type:
            return 1.0, 1.0  # 正确，满分 + 无折扣
        else:
            return 0.2, 0.6  # 错误，低分 + 40%折扣
    
    def _calculate_document_reward(self, generated_doc: str, ground_truth_doc: str) -> float:
        """计算文档名称奖励"""
        if generated_doc == ground_truth_doc:
            return 1.0
        elif generated_doc and ground_truth_doc and generated_doc in ground_truth_doc:
            return 0.7  # 部分匹配
        else:
            return 0.0
    
    def _calculate_entity_consistency(self, input_entities: List[str], generated_output: Dict) -> float:
        """实体一致性检查"""
        all_output_entities = self._extract_all_entities(generated_output)
        
        if not all_output_entities:
            return -0.5  # 没有输出实体
        
        # 检查输出实体是否都在输入NER中
        valid_entities = [entity for entity in all_output_entities if entity in input_entities]
        
        if not valid_entities:
            return -1.0  # 所有实体都不在输入中
        
        consistency_ratio = len(valid_entities) / len(all_output_entities)
        return consistency_ratio
    
    def _extract_all_entities(self, generated_output: Dict) -> List[str]:
        """从生成输出中提取所有实体"""
        entities = set()
        
        # 从ner_entities字段提取
        entities.update(generated_output.get('ner_entities', []))
        
        # 从information字段提取实体
        if 'information' in generated_output:
            for item in generated_output['information']:
                if 'entity' in item and isinstance(item['entity'], list):
                    entities.update(item['entity'])
        
        return list(entities)
    
    def _calculate_specific_reward(self, generated_output: Dict, ground_truth: Dict) -> float:
        """计算类型特定的奖励"""
        question_type = generated_output.get('question_type')
        
        if question_type == 'detail':
            return self._calculate_detail_reward(generated_output, ground_truth)
        elif question_type == 'multi_intent':
            return self._calculate_multi_intent_reward(generated_output, ground_truth)
        elif question_type == 'multi_hop':
            return self._calculate_multi_hop_reward(generated_output, ground_truth)
        elif question_type == 'reasoning':
            return self._calculate_reasoning_reward(generated_output, ground_truth)
        else:
            return 0.0
    
    def _calculate_detail_reward(self, generated: Dict, ground_truth: Dict) -> float:
        """计算detail类型的奖励"""
        reward = 0.0
        
        # 检查keywords
        gen_keywords = set(generated.get('keywords', []))
        gt_keywords = set(ground_truth.get('keywords', []))
        
        if gen_keywords == gt_keywords:
            reward += 1.0
        elif gen_keywords and gt_keywords:
            # 计算Jaccard相似度
            intersection = len(gen_keywords & gt_keywords)
            union = len(gen_keywords | gt_keywords)
            reward += intersection / union if union > 0 else 0
        
        return reward
    
    def _calculate_multi_intent_reward(self, generated: Dict, ground_truth: Dict) -> float:
        """计算multi_intent类型的奖励"""
        reward = 0.0
        
        # 检查数量一致性
        gen_num = generated.get('num', 0)
        gen_info = generated.get('information', [])
        gt_num = ground_truth.get('num', 0)
        gt_info = ground_truth.get('information', [])
        
        if gen_num == len(gen_info):
            reward += 0.3
        if gen_num == gt_num:
            reward += 0.3
        
        # 检查信息质量
        if gen_info and gt_info:
            # 简化的信息匹配度计算
            min_len = min(len(gen_info), len(gt_info))
            for i in range(min_len):
                if gen_info[i].get('question') == gt_info[i].get('question'):
                    reward += 0.4 / len(gt_info)
        
        return min(reward, 1.0)
    
    def _calculate_multi_hop_reward(self, generated: Dict, ground_truth: Dict) -> float:
        """计算multi_hop类型的奖励"""
        # 实现类似的逻辑
        return 0.5  # 简化实现
    
    def _calculate_reasoning_reward(self, generated: Dict, ground_truth: Dict) -> float:
        """计算reasoning类型的奖励"""
        # 实现类似的逻辑
        return 0.5  # 简化实现
    
    def _calculate_format_reward(self, generated_output: Dict) -> float:
        """计算格式完整性奖励"""
        reward = 0.0
        
        # 检查必需字段
        required_fields = ['question_type', 'document_name', 'ner_entities']
        for field in required_fields:
            if field in generated_output and generated_output[field]:
                reward += 0.2
        
        # 检查类型特定字段
        question_type = generated_output.get('question_type')
        if question_type == 'detail' and 'keywords' in generated_output:
            reward += 0.4
        elif question_type in ['multi_intent', 'multi_hop', 'reasoning']:
            if 'information' in generated_output:
                reward += 0.4
        
        return min(reward, 1.0)