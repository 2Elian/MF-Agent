from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as T
from PIL import Image
import io
import traceback
from ultralytics import YOLO
app = FastAPI(title="YOLO图像分类接口", description="用于预测图片类型:chart/flowchart/promo/table_image")
MODEL_PATH = r"G:\项目成果打包\金融多模态知识库构建与复杂问答检索算法\AI-Claude\src\utils\yolo\runs\classify\train\weights\best.pt"
CLASS_NAMES = ["chart", "flowchart", "promo", "table_image"]
model = YOLO(MODEL_PATH)
print(f"YOLO分类信息：{model.names}")
def preprocess_image(image: Image.Image, imgsz=224):
    """将输入图片转为模型可接受的格式"""
    transforms = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(0), std=torch.tensor(1))
    ])
    img_tensor = transforms(image).unsqueeze(0)  # shape: [1, 3, 224, 224]
    return img_tensor

@app.post("/predict", summary="预测图片类型")
async def predict_image(file: UploadFile = File(..., description="image")):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB").copy()

        results = model.predict(image, imgsz=224, verbose=False)
        pred_probs = results[0].probs  # Probs对象
        pred_class_idx = pred_probs.top1
        pred_class_name = CLASS_NAMES[pred_class_idx]
        pred_score = float(pred_probs.top1conf)
        all_classes = {name: float(pred_probs.data[i]) for i, name in enumerate(CLASS_NAMES)}

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "prediction": {
                "class": pred_class_name,
                "confidence": round(pred_score, 4),
                "all_classes": all_classes
            }
        })
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)