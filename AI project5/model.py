import torch
import torch.nn as nn
import clip

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes=3, fusion_type='concat', freeze_clip=True):
        super().__init__()
        self.fusion_type = fusion_type
        self.clip_model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
        
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        embed_dim = self.clip_model.ln_final.weight.shape[0]  # 512

        if fusion_type == 'concat':
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        elif fusion_type == 'text_only':
            self.classifier = nn.Linear(embed_dim, num_classes)
        elif fusion_type == 'image_only':
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            raise ValueError("fusion_type must be 'concat', 'text_only', or 'image_only'")

    def forward(self, image, text):
        image_features = self.clip_model.encode_image(image)
        text_features = self.clip_model.encode_text(text)

        # L2 normalize (as in original CLIP)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.fusion_type == 'concat':
            fused = torch.cat([image_features, text_features], dim=1)
            return self.classifier(fused)
        elif self.fusion_type == 'text_only':
            return self.classifier(text_features)
        elif self.fusion_type == 'image_only':
            return self.classifier(image_features)