import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

# Choose CPU or GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the CLIP model & processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def multimodal_prediction(text: str, image_path: str):
    """
    Returns:
      - risk_score (float): a toy score computed from the mean of the combined embedding
      - embedding (np.ndarray): 1×(dim_image+dim_text) feature vector
    """
    # Load & preprocess
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)

    # Extract features
    with torch.no_grad():
        image_feats = model.get_image_features(pixel_values=inputs.pixel_values)
        text_feats  = model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

    # Normalize
    image_feats = image_feats / image_feats.norm(p=2, dim=-1, keepdim=True)
    text_feats  = text_feats  / text_feats.norm(p=2, dim=-1, keepdim=True)

    # Concatenate into one vector
    embedding = torch.cat([image_feats, text_feats], dim=1).cpu().numpy()  # shape (1, 1024)
    risk_score = float(np.mean(embedding))

    return risk_score, embedding