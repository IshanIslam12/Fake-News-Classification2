import os
import re
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel


# ==============================
# 1. Custom BERT model (MATCHES YOUR COLAB CHECKPOINT)
# ==============================
class BertForFakeNews(nn.Module):
    """
    Custom BERT-based classifier.

    Head structure MUST MATCH your training checkpoint:
        classifier.0 = Linear(hidden -> head_hidden)
        classifier.1 = ReLU()
        classifier.2 = Dropout()
        classifier.3 = Linear(head_hidden -> num_labels)
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        head_hidden: int = 256,
        head_dropout: float = 0.1,
    ):
        super().__init__()

        # BERT backbone
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Classification head (IMPORTANT: Indexes must match checkpoint keys)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),   # classifier.0 (weights exist)
            nn.ReLU(),                             # classifier.1
            nn.Dropout(head_dropout),              # classifier.2
            nn.Linear(head_hidden, num_labels),    # classifier.3 (weights exist)
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls_repr)
        return logits


# ==============================
# 2. Paths & device
# ==============================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "fake_news_model")
CKPT_PATH = os.path.join(MODEL_DIR, "model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# 3. Load tokenizer
# ==============================
print(f"Loading tokenizer from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)


# ==============================
# 4. Load checkpoint and rebuild model
# ==============================
print(f"Loading checkpoint from {CKPT_PATH} on {DEVICE}...")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

model_name = ckpt.get("model_name")
num_labels = ckpt.get("num_labels", 2)
head_hidden = ckpt.get("head_hidden", 256)
head_dropout = ckpt.get("head_dropout", 0.1)

print(
    f"Rebuilding BertForFakeNews("
    f"model_name={model_name}, num_labels={num_labels}, "
    f"head_hidden={head_hidden}, head_dropout={head_dropout})"
)

model = BertForFakeNews(
    model_name=model_name,
    num_labels=num_labels,
    head_hidden=head_hidden,
    head_dropout=head_dropout,
)

# Load STATE DICT (should now match perfectly)
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE)
model.eval()

print(f"✅ Model loaded on {DEVICE}")


# ==============================
# 5. FastAPI app + CORS
# ==============================
app = FastAPI(title="Fake News Classifier API (Custom BERT)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# 6. Request schema
# ==============================
class InputData(BaseModel):
    title: Optional[str] = ""
    text: Optional[str] = ""


# ==============================
# 7. Simple text cleaner
# ==============================
LINKS_RE = re.compile(r"http\S+|www\S+", re.IGNORECASE)
HTML_RE = re.compile(r"<.*?>", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = LINKS_RE.sub(" ", s)
    s = HTML_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


# ==============================
# 8. Inference helper
# ==============================
def predict_fake_news(text: str):
    cleaned = clean_text(text)

    enc = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc)

    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    prob_real = float(probs[0])
    prob_fake = float(probs[1])

    pred_idx = int(np.argmax(probs))
    label = "REAL" if pred_idx == 0 else "FAKE"
    confidence = max(prob_real, prob_fake)

    return label, prob_real, prob_fake, confidence


# ==============================
# 9. Routes
# ==============================
@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Fake News Custom BERT API is running",
        "model_name": model_name,
        "device": str(DEVICE),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: InputData):
    merged = f"{data.title or ''} {data.text or ''}".strip()
    if not merged:
        return {"ok": False, "error": "Please provide a title or text."}

    label, prob_real, prob_fake, confidence = predict_fake_news(merged)

    pct_real = round(prob_real * 100)
    pct_fake = round(prob_fake * 100)

    result_text = f"{pct_real}% REAL" if label == "REAL" else f"{pct_fake}% FAKE"

    return {
        "ok": True,
        "label": label,
        "label_id": 0 if label == "REAL" else 1,
        "confidence": float(confidence),
        "score_real": float(prob_real),
        "score_fake": float(prob_fake),
        "percentage_real": pct_real,
        "percentage_fake": pct_fake,
        "result_text": result_text,
    }


# Run via:
#   uvicorn model_api:app --reload --port 8000
