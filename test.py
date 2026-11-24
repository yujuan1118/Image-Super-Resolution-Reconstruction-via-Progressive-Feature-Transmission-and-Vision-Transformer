# 即時測試專用 Cell（已加入顯存清理與切塊推論，避免全圖 Attention 爆顯存）
%matplotlib inline

import os, gc, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = '/kaggle/input/notebooks/ishanmanglik/div2k'

def release_old_training_memory():
    # 嘗試釋放前面訓練 Cell 可能殘留的物件
    for var_name in ['model', 'train_dataset', 'val_dataset', 'train_loader', 'val_loader', 'opt', 'scheduler', 'scaler', 'pbar']:
        if var_name in globals():
            try:
                del globals()[var_name]
            except Exception:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

class PFA_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, prev_attn=None):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(x_flat).reshape(B, H * W, 3, 4, C // 4).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = q.float(), k.float()
        attn = (q @ k.transpose(-2, -1)) * ((C // 4) ** -0.5)
        if prev_attn is not None:
            attn = attn + 0.5 * prev_attn
        attn_out = attn.softmax(dim=-1).type_as(v)
        x_out = (attn_out @ v).transpose(1, 2).reshape(B, H * W, C)
        x_out = self.proj(x_out).transpose(1, 2).reshape(B, C, H, W)
        return x + x_out, attn

class PFTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.head = nn.Conv2d(3, 64, 3, padding=1)
        self.layers = nn.ModuleList([PFA_Block(64) for _ in range(3)])
        self.upsample_feat = nn.Sequential(
            nn.Conv2d(64, 64 * 16, 3, padding=1),
            nn.PixelShuffle(4)
        )
        self.tail = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        base = self.upsample(x)
        feat = self.head(x)
        attn = None
        for layer in self.layers:
            feat, attn = layer(feat, attn)
        res_feat = self.upsample_feat(feat)
        res = self.tail(res_feat)
        return torch.clamp(base + res, 0.0, 1.0)

def predict_with_tiles(model, lr_img, scale=4, tile_size=64, overlap=8):
    """將輸入低解析度影像切塊推論並拼接，避免整張圖做 Attention 造成顯存不足"""
    _, C, H, W = lr_img.shape
    stride = max(1, tile_size - overlap)
    output = torch.zeros((1, C, H * scale, W * scale), device=lr_img.device, dtype=torch.float32)
    weight = torch.zeros((1, C, H * scale, W * scale), device=lr_img.device, dtype=torch.float32)

    for y in range(0, H, stride):
        y_end = min(y + tile_size, H)
        y_start = max(0, y_end - tile_size)
        for x in range(0, W, stride):
            x_end = min(x + tile_size, W)
            x_start = max(0, x_end - tile_size)

            tile_lr = lr_img[:, :, y_start:y_end, x_start:x_end]
            with torch.amp.autocast('cuda', enabled=(lr_img.device.type == 'cuda')):
                tile_sr = model(tile_lr).float()

            oy_s, oy_e = y_start * scale, y_end * scale
            ox_s, ox_e = x_start * scale, x_end * scale
            output[:, :, oy_s:oy_e, ox_s:ox_e] += tile_sr
            weight[:, :, oy_s:oy_e, ox_s:ox_e] += 1.0

            del tile_lr, tile_sr

    return torch.clamp(output / torch.clamp(weight, min=1.0), 0.0, 1.0)

class ValOnlyDataset(Dataset):
    def __init__(self, base_path):
        self.x = np.load(os.path.join(base_path, 'val_tensor_x.npy'))
        self.y = np.load(os.path.join(base_path, 'val_tensor_y.npy'))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        lr_np = self.x[i][:, :, ::-1].copy()
        hr_np = self.y[i][:, :, ::-1].copy()
        lr = torch.from_numpy(lr_np).float()
        hr = torch.from_numpy(hr_np).float()
        if lr.shape[-1] == 3: lr = lr.permute(2, 0, 1)
        if hr.shape[-1] == 3: hr = hr.permute(2, 0, 1)
        if lr_np.max() > 1.5: lr = lr / 255.0
        if hr_np.max() > 1.5: hr = hr / 255.0
        return lr, hr

def find_weight():
    candidates = ['/kaggle/working/best_pftnet.pth', 'best_pftnet.pth']
    for c in candidates:
        if os.path.exists(c):
            return c
    for root, _, files in os.walk('/kaggle/input'):
        if 'best_pftnet.pth' in files:
            return os.path.join(root, 'best_pftnet.pth')
    return None

def safe_ssim(hr_np, pred_np):
    h, w = hr_np.shape[:2]
    min_side = min(h, w)
    if min_side < 3:
        return 0.0
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    return ssim_metric(hr_np, pred_np, data_range=1.0, channel_axis=-1, win_size=win_size)

def run_live_test(num_samples=4, tile_size=64, overlap=8):
    release_old_training_memory()

    weight_path = find_weight()
    if weight_path is None:
        print("❌ 找不到 best_pftnet.pth！請先確認權重檔已存出，或上傳到 Kaggle Input。")
        return

    print(f"✅ 載入權重：{weight_path}")
    val_ds = ValOnlyDataset(DATA_PATH)
    model = PFTNet().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    indices = random.sample(range(len(val_ds)), min(num_samples, len(val_ds)))
    print(f"🎲 亂數抽樣索引：{indices}")

    fig, axes = plt.subplots(len(indices), 2, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for row_idx, img_idx in enumerate(indices):
            lr_t, hr_t = val_ds[img_idx]
            lr_in = lr_t.unsqueeze(0).to(device)

            img_sr = predict_with_tiles(model, lr_in, scale=4, tile_size=tile_size, overlap=overlap)
            img_bi = F.interpolate(lr_in, scale_factor=4, mode='bicubic', align_corners=False).float()

            img_hr_np = hr_t.permute(1, 2, 0).cpu().numpy()
            img_bi_np = np.clip(img_bi[0].permute(1, 2, 0).cpu().numpy(), 0, 1)
            img_sr_np = np.clip(img_sr[0].permute(1, 2, 0).cpu().numpy(), 0, 1)

            bi_p = psnr_metric(img_hr_np, img_bi_np, data_range=1.0)
            bi_s = safe_ssim(img_hr_np, img_bi_np)
            sr_p = psnr_metric(img_hr_np, img_sr_np, data_range=1.0)
            sr_s = safe_ssim(img_hr_np, img_sr_np)

            axes[row_idx, 0].imshow(img_bi_np)
            axes[row_idx, 0].set_title(f"Val #{img_idx} [Bicubic]\nPSNR: {bi_p:.2f}dB | SSIM: {bi_s:.4f}")
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(img_sr_np)
            axes[row_idx, 1].set_title(f"Val #{img_idx} [PFTNet]\nPSNR: {sr_p:.2f}dB | SSIM: {sr_s:.4f}")
            axes[row_idx, 1].axis("off")

            del lr_in, img_sr, img_bi
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

run_live_test(num_samples=4, tile_size=64, overlap=8)
