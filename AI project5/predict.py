# predict.py

import torch
from torch.utils.data import DataLoader
import clip
from dataset import MultimodalDataset
from model import MultimodalClassifier
from utils import load_test_guids  # ← 改这里

TEST_FILE = "test_without_label.txt"
DATA_DIR = "data"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FUSION_TYPE = "concat"
MODEL_PATH = f"best_model_{FUSION_TYPE}.pth"

def predict():
    _, preprocess = clip.load("ViT-B/32", device=DEVICE)

    # 只加载 guid，不依赖 label
    test_guids = load_test_guids(TEST_FILE)
    # 构造 [(guid, 'dummy')] 列表，label 不会被使用
    test_data = [(guid, 'positive') for guid in test_guids]  # label 随便填，反正不用

    test_dataset = MultimodalDataset(test_data, DATA_DIR, preprocess)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = MultimodalClassifier(fusion_type=FUSION_TYPE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    idx_to_label = {0: 'positive', 1: 'neutral', 2: 'negative'}
    results = []

    with torch.no_grad():
        for batch in test_loader:
            image = batch['image'].to(DEVICE)
            text = batch['text'].to(DEVICE)
            output = model(image, text)
            preds = output.argmax(dim=1)
            for i, guid in enumerate(batch['guid']):
                results.append((guid, idx_to_label[preds[i].item()]))

    with open("prediction.txt", "w", encoding="utf-8") as f:
        for guid, label in results:
            f.write(f"{guid} {label}\n")
    print(f"✅ 预测完成！共 {len(results)} 条结果已保存到 prediction.txt")

if __name__ == "__main__":
    predict()