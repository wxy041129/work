import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import clip

class MultimodalDataset(Dataset):
    def __init__(self, guid_label_list, data_dir, preprocess, modality='both'):
        self.guid_label_list = guid_label_list  # [(guid, label), ...]
        self.data_dir = data_dir
        self.preprocess = preprocess
        self.modality = modality
        self.label_map = {'positive': 0, 'neutral': 1, 'negative': 2}

    def __len__(self):
        return len(self.guid_label_list)

    def __getitem__(self, idx):
        guid, label_str = self.guid_label_list[idx]
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        txt_path = os.path.join(self.data_dir, f"{guid}.txt")

        # Load text
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception:
            text = ""  # fallback to empty string

        # Tokenize using CLIP's built-in tokenizer
        tokens = clip.tokenize(text, truncate=True).squeeze(0)  # [77]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=0)  # black image

        image = self.preprocess(image)
        label = self.label_map.get(label_str, -1)

        return {
            'guid': guid,
            'image': image,
            'text': tokens,
            'label': label
        }