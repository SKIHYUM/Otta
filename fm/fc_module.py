# fc_module.py
import numpy as np
import torch
from einops import rearrange
from torch.nn import CosineSimilarity
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import cv2

class FrameConsistency:
    RETURN_TYPE = ['pt', 'np', 'float']

    def __init__(self, version="openai/clip-vit-base-patch32", device="cuda", mini_bsz=64, return_type='pt'):
        self.device = device
        assert return_type in self.RETURN_TYPE
        self.return_type = return_type

    
        self.processor = CLIPImageProcessor.from_pretrained("./clip-vit-base-patch32")
        self.model     = CLIPVisionModelWithProjection.from_pretrained("./clip-vit-base-patch32").to(self.device)

        self.cosine = CosineSimilarity(dim=1).to(self.device)
        self.mini_bsz = mini_bsz
        self.freeze()

    def freeze(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, frames, step=1):
      
        num = len(frames)
        assert num > step, "Frame > step"

        embeds = []
        bs = min(self.mini_bsz, num)
        for i in range(0, num, bs):
            batch = frames[i:i+bs]
            inputs = self.processor(batch, return_tensors='pt').to(self.device)
            out = self.model(**inputs)
            embeds.append(out.image_embeds)
        embeds = torch.cat(embeds, dim=0)  # (N, D)

       
        sim = self.cosine(embeds[:-step], embeds[step:])  # (N-step,)
        score = sim.mean().cpu()

        if self.return_type == 'pt':
            return score
        elif self.return_type == 'np':
            return score.numpy()
        else:
            return float(score)

def load_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        # BGR->RGB
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames