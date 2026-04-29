from typing import *
import os
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        model_name = os.environ.get("TRELLIS_REMBG_MODEL", model_name)
        self.backend = "onnx" if self._is_onnx_model(model_name) else "torch"
        if self.backend == "onnx":
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            model_path = model_name
            if not os.path.exists(model_path):
                repo_id, filename = model_name.split(":", 1)
                model_path = hf_hub_download(repo_id, filename)
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self.session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
        else:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _is_onnx_model(model_name: str) -> bool:
        if model_name.lower().endswith(".onnx"):
            return True
        return ":" in model_name and not (len(model_name) > 1 and model_name[1] == ":")
    
    def to(self, device: str):
        if self.backend == "torch":
            self.model.to(device)

    def cuda(self):
        if self.backend == "torch":
            self.model.cuda()

    def cpu(self):
        if self.backend == "torch":
            self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image.convert("RGB")).unsqueeze(0)
        if self.backend == "onnx":
            pred = self.session.run(None, {self.input_name: input_images.numpy()})[0]
            pred = np.clip(pred[0, 0], 0.0, 1.0)
            pred_pil = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
        else:
            input_images = input_images.to("cuda")
            with torch.no_grad():
                preds = self.model(input_images)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    
