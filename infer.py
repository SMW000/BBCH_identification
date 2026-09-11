"""
infer.py - BBCH phenological stage identification inference script.

Loads the released checkpoint (best_model.pth) and predicts the BBCH
phenological stage for one RGB + MS (G/NIR/R/RE) image pair.

Usage:
    python infer.py \
        --checkpoint checkpoints/EfficientFormer_RGBMS_Aux_CSoP2/best_model.pth \
        --rgb  path/to/rgb.jpg \
        --g    path/to/G.tif   --nir path/to/NIR.tif \
        --r    path/to/R.tif   --re  path/to/RE.tif

The model architecture is imported from train.py (the exact training script
that produced the checkpoint); training code only runs under __main__ there.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import tifffile
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import DualStreamFusionNetWithAux_CSoP  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MS_CHANNELS = ['G', 'NIR', 'R', 'RE']


def load_model(checkpoint_path, device):
    """Instantiate the model from the training script and load the checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    cfg = ckpt['config']
    num_classes = len(ckpt['class_to_idx'])
    model = DualStreamFusionNetWithAux_CSoP(
        num_classes=num_classes,
        backbone_feat_dim=176,
        image_size=cfg.get('IMAGE_SIZE', 224),
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model, ckpt


def preprocess_rgb(rgb_path, size):
    """RGB: resize -> tensor -> ImageNet normalization (same as training)."""
    img = Image.open(rgb_path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tf(img)


def read_ms_channel(path, size):
    """Read one MS channel (same as _read_tiff in train.py): min-max to 0-1."""
    try:
        with tifffile.TiffFile(path) as tif:
            img = tif.asarray().astype(np.float32)
    except Exception:
        with Image.open(path) as im:
            img = np.array(im, dtype=np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = np.zeros_like(img)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img


def preprocess_ms(channel_paths, mean, std, size):
    """MS: stack 4 channels -> bilinear resize -> (x - mean) / std (same as training)."""
    channels = [read_ms_channel(p, size) for p in channel_paths]
    t = torch.from_numpy(np.stack(channels, axis=0)).float()  # (4, H, W)
    t = F.interpolate(t.unsqueeze(0), size=(size, size),
                      mode='bilinear', align_corners=False).squeeze(0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
    return (t - mean_t) / std_t


def predict(model, ckpt, rgb_path, ms_paths, device):
    size = ckpt['config'].get('IMAGE_SIZE', 224)
    rgb = preprocess_rgb(rgb_path, size).unsqueeze(0).to(device)
    ms = preprocess_ms(ms_paths, ckpt['ms_mean'], ckpt['ms_std'], size).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(rgb, ms)
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs))
    idx_to_class = {v: k for k, v in ckpt['class_to_idx'].items()}
    return idx_to_class[idx], float(probs[idx]), probs


def main():
    parser = argparse.ArgumentParser(description='BBCH phenological stage inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to best_model.pth')
    parser.add_argument('--rgb', type=str, required=True, help='RGB image path')
    parser.add_argument('--g', type=str, required=True, help='MS G channel (tif)')
    parser.add_argument('--nir', type=str, required=True, help='MS NIR channel (tif)')
    parser.add_argument('--r', type=str, required=True, help='MS R channel (tif)')
    parser.add_argument('--re', type=str, required=True, help='MS RE channel (tif)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt = load_model(args.checkpoint, device)
    ms_paths = [args.g, args.nir, args.r, args.re]
    cls, prob, probs = predict(model, ckpt, args.rgb, ms_paths, device)

    print(f'Predicted BBCH stage : {cls}')
    print(f'Confidence           : {prob:.4f}')
    print('Full distribution (top-5):')
    idx_to_class = {v: k for k, v in ckpt['class_to_idx'].items()}
    for i in torch.topk(probs, 5).indices.tolist():
        print(f'  {idx_to_class[i]}: {probs[i]:.4f}')


if __name__ == '__main__':
    main()
