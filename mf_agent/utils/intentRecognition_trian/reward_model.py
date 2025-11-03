import re
import json
from typing import List, Dict, Any, Tuple

class MfAgentIntentionRecognitionReward:
    VALID_QUESTION_TYPES = ["detail", "multi_intent", "multi_hop", "reasoning"]

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
    def calculate_reward(self, generated_text: str, ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        result = self.parse_model_output(generated_text)
        rewards = {
            "type_reward": 0, # cls correct score
            "title_reward": 0,
            "entity_reward": 0,
            "keyword_reward": 0,
            "special_token_reward": 0
        }
        # step1 -> type reward
        q_type = result.get("question_type", "unknown")
        gt_type = ground_truth.get("question_type")

        if q_type == "unknown":
            return 0, {"error": "missing question_type"}
        if q_type not in self.VALID_QUESTION_TYPES:
            rewards["type_reward"] = -1
            return sum(rewards.values()), rewards
        if q_type == gt_type:
            rewards["type_reward"] = 1
        else:
            rewards["type_reward"] = 0.1  # partial match

        # step2 -> title reward
        title_match = re.search(r'<|title_name_start|>(.*?)<|title_name_end|>', generated_text, re.DOTALL)
        if title_match:
            gen_title = title_match.group(1).strip()
            if gen_title and ground_truth.get("title_name"):
                rewards["title_reward"] = 1 if gen_title == ground_truth["title_name"] else 0.8
        else:
            return 0, {"error": "missing title"}

        # steo3 -> entity and keywords extract
        gen_entities = set(result.get("ner_entities", []))
        gt_entities = set(ground_truth.get("ner_entities", []))
        if not gen_entities:
            rewards["entity_reward"] = 0
        else:
            inter = len(gen_entities & gt_entities)
            union = len(gen_entities | gt_entities)
            rewards["entity_reward"] = inter / union if union else 0

        if q_type == "detail":
            gen_kw = set(result.get("keywords", []))
            gt_kw = set(ground_truth.get("keywords", []))
            if gen_kw:
                inter = len(gen_kw & gt_kw)
                union = len(gen_kw | gt_kw)
                rewards["keyword_reward"] = inter / union if union else 0
            else:
                rewards["keyword_reward"] = 0

        if q_type in ["multi_intent", "multi_hop", "reasoning"]:
            rewards["special_token_reward"] = self._check_special_tokens(result)

        total = sum(rewards.values())
        return total, rewards

    def _check_special_tokens(self, parsed_data: Dict[str, Any]) -> float:
        # TODO @zimo: 对 multi_intent / multi_hop / reasoning 的 special token 检查
        score = 0
        num = parsed_data.get("num") or parsed_data.get("step_num")
        info = parsed_data.get("information")

        if num and not info:
            return -1
        if num and isinstance(info, list) and len(info) != num:
            return -2

        if isinstance(info, list):
            score += 1  # 有效的信息结构
        return score

    def parse_model_output(self, generated_text: str) -> Dict[str, Any]:
        parsed = {}
        try:
            # question_type
            match_q = re.search(r"<|question_type_start|>(.*?)<|question_type_end|>", generated_text, re.DOTALL)
            parsed["question_type"] = match_q.group(1).strip() if match_q else "unknown"

            # document name
            match_doc = re.search(r"<|title_name_start|>(.*?)<|title_name_end|>", generated_text, re.DOTALL)
            parsed["document_name"] = match_doc.group(1).strip() if match_doc else ""

            # entities
            match_ner = re.search(r"<|ner_entities_start|>(.*?)<|ner_entities_end|>", generated_text, re.DOTALL)
            if match_ner:
                try:
                    parsed["ner_entities"] = json.loads(match_ner.group(1).strip())
                except:
                    parsed["ner_entities"] = []
            else:
                parsed["ner_entities"] = []

            # detail keywords
            match_kw = re.search(r"<|keywords+_start|>(.*?)<|keywords_end|>", generated_text, re.DOTALL)
            if match_kw:
                try:
                    parsed["keywords"] = json.loads(match_kw.group(1).strip())
                except:
                    parsed["keywords"] = []

            # multi_intent / multi_hop / reasoning
            match_info = re.search(r"<|information_start|>(.*?)<|information_end|>", generated_text, re.DOTALL)
            if match_info:
                try:
                    parsed["information"] = json.loads(match_info.group(1).strip())
                except:
                    parsed["information"] = []

            num_match = re.search(r"<|num_start|>(\d+)<|num_end|>", generated_text)
            step_match = re.search(r"<|step_num_start|>(\d+)<|step_num_end|>", generated_text)
            if num_match:
                parsed["num"] = int(num_match.group(1))
            if step_match:
                parsed["step_num"] = int(step_match.group(1))

        except Exception as e:
            parsed["parse_error"] = str(e)
        return parsed
