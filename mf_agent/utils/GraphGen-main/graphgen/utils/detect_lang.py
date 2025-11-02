def detect_main_language(text):
    """
    识别文本的主要语言

    :param text:
    :return:
    """
    assert isinstance(text, str)
    def is_chinese_char(char):
        return '\u4e00' <= char <= '\u9fff'

    def is_english_char(char):
        return char.isascii() and char.isalpha()

    # 去除空格和标点符号
    text = ''.join(char for char in text if char.strip())

    chinese_count = sum(1 for char in text if is_chinese_char(char))
    english_count = sum(1 for char in text if is_english_char(char))

    total = chinese_count + english_count
    if total == 0:
        return 'en'

    chinese_ratio = chinese_count / total

    if chinese_ratio >= 0.5:
        return 'zh'
    return 'en'

def detect_if_chinese(text):
    """
    判断文本是否包含有中文

    :param text:
    :return:
    """

    assert isinstance(text, str)
    return any('\u4e00' <= char <= '\u9fff' for char in text) # \u4e00是中文unicode编码的第一个字符 \u9fff是最后一个 any函数的意思是：循环里面只要有一个True 那么就返回True

def save_extraction_to_json(chunk_id: str, content: str, final_result: str, json_path: str):
    import json
    import re
    entries = [x.strip() for x in final_result.split("##") if x.strip()]
    entities = []
    pattern = r'\("([^"]+)"<\|>"([^"]+)"<\|>"([^"]+)"<\|>"([^"]+)"\)'
    for entry in entries:
        match = re.search(pattern, entry)
        if not match:
            continue
        entry_type, name, cls, summary = match.groups()
        if entry_type == "entity":
            entities.append({
                "EntityName": name,
                "EntityClass": cls,
                "EntitySummary": summary
            })
    
    data = {
        "chunk_id": chunk_id,
        "content": content,
        "entity": entities
    }
    
    with open(json_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")