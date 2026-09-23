"""
EP39 Challenge — CLIP Zero-Shot Image Classification
pip install transformers Pillow torch
"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from pathlib import Path

# 1. Load model + processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Your 5 class descriptions (prompt engineering matters!)
labels = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a car",
    "a photo of a tree",
    "a photo of an airplane",
]

# 3. Put 10 of your own images in a folder called "my_images"
image_dir = Path("my_images")
image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

if not image_paths:
    print("Put at least one image in the 'my_images' folder!")
    exit()

print(f"Found {len(image_paths)} images. Running zero-shot classification...\n")

correct = 0
for img_path in image_paths:
    image = Image.open(img_path).convert("RGB")

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image   # shape [1, 5]
        probs = logits_per_image.softmax(dim=1)

    pred_idx = probs.argmax(dim=1).item()
    predicted = labels[pred_idx]
    confidence = probs[0, pred_idx].item()

    print(f"{img_path.name:30s} → {predicted:25s} ({confidence*100:.1f}%)")
    # Optional: keep a ground-truth label in the filename to auto-calculate accuracy

print("\nDone! Report your accuracy and best prompt template in the comments.")
