
import os
from typing import Optional, Tuple

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

APP_TITLE = "ChestAI — Pneumonia Classifier"

# --------------------
# Helpers
# --------------------
def infer_arch_from_name(name: str) -> str:
    n = name.lower()
    if "resnet50" in n:
        return "resnet50"
    if "resnet" in n:
        return "resnet18"
    if "densenet" in n:
        return "densenet121"
    if "alexnet" in n:
        return "alexnet"
    return "resnet18"  # sensible default

def build_backbone(arch: str, num_classes: int = 1) -> nn.Module:
    if arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    if arch == "alexnet":
        m = models.alexnet(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
        return m
    # fallback
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def is_state_dict(obj) -> bool:
    return isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys())

def guess_num_classes_from_state_dict(sd: dict) -> Optional[int]:
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and k.endswith(".weight"):
            if any(h in k for h in ["fc.weight", "classifier.weight", "classifier.6.weight"]):
                return v.shape[0]
    return 1

@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> Tuple[nn.Module, str]:
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)
    arch = infer_arch_from_name(os.path.basename(model_path))

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        model.eval()
        model.to(device)
        return model, arch

    if is_state_dict(checkpoint):
        # unwrap common nesting keys
        for key in ["state_dict", "model_state_dict", "net", "model"]:
            if key in checkpoint and is_state_dict(checkpoint[key]):
                checkpoint = checkpoint[key]
                break
        num_classes = guess_num_classes_from_state_dict(checkpoint)
        model = build_backbone(arch, num_classes=num_classes)
        # strip dataparallel prefixes
        new_sd = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        model.to(device)
        return model, arch

    raise RuntimeError("Unsupported checkpoint format. Provide a full PyTorch model or a state_dict.")

def default_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def predict(model: nn.Module, img: Image.Image, threshold: float = 0.5) -> Tuple[str, float]:
    tfm = default_transforms(224)
    x = tfm(img.convert("RGB")).unsqueeze(0)  # [1,3,H,W]
    with torch.no_grad():
        logits = model(x)
        # handle shapes: [1], [1,1], [1,2]
        if logits.ndim == 2 and logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[0]
            pneu_prob = float(probs[1])
        else:
            # binary logit
            pneu_prob = float(torch.sigmoid(logits.view(-1)[0]))
    label = "Pneumonia: Positive" if pneu_prob >= threshold else "Pneumonia: Negative"
    return label, pneu_prob

def list_model_files() -> list:
    exts = (".pt", ".pth", ".pkl", ".bin")
    return [f for f in os.listdir(".") if f.lower().endswith(exts)]

# --------------------
# UI
# --------------------
st.set_page_config(page_title="ChestAI", page_icon="🫁", layout="centered")
st.title(APP_TITLE)
st.caption("Upload a chest X-ray to get a binary prediction.")

models_found = list_model_files()
if not models_found:
    st.warning("No model checkpoint found in the current directory. "
               "Place your PyTorch model file (e.g., ChestAI_final_project_resnet_gan.pkl) next to this app.")
else:
    default_model = None
    # Prefer a likely file name if present
    for prefer in ["ChestAI_final_project_resnet_gan.pkl", "ChestAI_final_project_densenet_gan.pkl",
                   "ChestAI_final_project_alexnet.pkl", "model.pkl", "model.pt"]:
        if prefer in models_found:
            default_model = prefer
            break
    model_name = st.selectbox("Select model file", models_found, index=models_found.index(default_model) if default_model in models_found else 0)
    with st.spinner("Loading model..."):
        try:
            model, arch = load_model(model_name)
            st.success(f"Model loaded ({arch}).")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

uploaded = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

# A simple "Predict" button as asked
if st.button("Predict", type="primary", disabled=uploaded is None):
    if uploaded is None:
        st.warning("Please upload an image first.")
    else:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_column_width=True)
        with st.spinner("Running inference..."):
            try:
                label, p = predict(model, img, threshold=0.5)
                st.subheader(label)
                st.write(f"Confidence (Pneumonia): {p:.3f}")
            except Exception as e:
                st.error(f"Inference failed: {e}")

st.markdown("---")
st.caption("ChestAI: simple Streamlit app for pneumonia detection. Move your model file next to this script and run `streamlit run app.py`.")
