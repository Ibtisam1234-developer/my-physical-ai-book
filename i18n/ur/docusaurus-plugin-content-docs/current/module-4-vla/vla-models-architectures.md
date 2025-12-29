---
slug: /module-4-vla/vla-models-architectures
title: "VLA Models and Architectures"
hide_table_of_contents: false
---

# VLA Models and Architectures

VLA models کی architecture اور implementation patterns کو سمجھنا۔

## Model Architectures

### CLIP-based Models

```python
import torch
from transformers import CLIPProcessor, CLIPModel

class VLA_CLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.action_head = torch.nn.Linear(512, 7)  # 7 DOF actions

    def forward(self, image, text):
        outputs = self.clip(image_input=image, input_ids=text)
        image_features = outputs.image_embeds
        action = self.action_head(image_features)
        return action
```

### RT-2 Style Models

```python
class RT2Model(torch.nn.Module):
    def __init__(self, vision_encoder, language_model, action_head):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.action_head = action_head

    def forward(self, image, instruction):
        vision_features = self.vision_encoder(image)
        combined = self.fuse(vision_features, instruction)
        action = self.action_head(combined)
        return action
```

## Training Strategies

### Imitation Learning

```python
def train_imitation(model, demonstrations, optimizer):
    model.train()
    total_loss = 0

    for image, text, action in demonstrations:
        optimizer.zero_grad()
        predicted_action = model(image, text)
        loss = F.mse_loss(predicted_action, action)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(demonstrations)
```

## اگلے steps

[llm-task-planning.md](./llm-task-planning.md) پڑھیں۔
