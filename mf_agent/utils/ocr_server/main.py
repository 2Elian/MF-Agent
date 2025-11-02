# ocr server
from pathlib import Path
from paddleocr import PaddleOCRVL
from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import numpy as np
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
app = Flask(__name__)
print("load paddle ocr vl model")
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://172.16.107.15:8080/v1"
)
print("🔹 Loading Qwen3-VL-4B-Instruct model on GPU3 ...")
model_name = "/data1/nuist_llm/TrainLLM/qwen-vl/ckpt"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)
@app.route("/models/ocr", methods=["POST"])
def ocr_endpoint():
    try:
        data = request.json
        img_b64 = data.get("image")
        image_bytes = base64.b64decode(img_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        result = pipeline.predict(input=image_np)

        # 安全地提取内容
        if not result:
            return jsonify({"status": "error", "message": "Empty OCR result."}), 200

        # 兼容不同格式
        parsing_list = None
        if isinstance(result, list) and len(result) > 0:
            parsing_list = result[0].get("parsing_res_list", [])
        elif isinstance(result, dict):
            parsing_list = result.get("parsing_res_list", [])
        else:
            parsing_list = []

        if not parsing_list:
            return jsonify({"status": "error", "message": "No parsing results found."}), 200

        def extract_content(item):
            if hasattr(item, 'content'):
                return item.content
            elif isinstance(item, dict):
                return item.get('content', '')
            return str(item)

        all_contents = [extract_content(item) for item in parsing_list]
        combined_content = '\n'.join(all_contents)

        return jsonify({
            "status": "success",
            "output": combined_content
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/models/qwen", methods=["POST"])
def infer_qwen_vl():
    """
    输入：
        {
            "image": "base64_string",
            "prompt": "Describe the chart..."
        }
    输出：
        {
            "status": "success",
            "text": "图像描述内容"
        }
    """
    try:
        data = request.get_json()
        img_b64 = data.get("image", None)
        prompt = data.get("prompt", "Describe this image.")

        if img_b64 is None:
            return jsonify({"status": "error", "message": "Missing 'image' field."}), 400
        image_bytes = base64.b64decode(img_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return jsonify({"status": "success", "text": output_text})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6006)