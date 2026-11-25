# 現場流程展示專用 Cell（直接跑切塊推論與 4 組對比圖）
%matplotlib inline

import gc, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = '/kaggle/input/notebooks/ishanmanglik/div2k'


def release_old_training_memory():
  for var_name in [
      'model',
      'train_dataset',
      'val_dataset',
      'train_loader',
      'val_loader',
      'opt',
      'scheduler',
      'scaler',
      'pbar',
  ]:
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
    qkv = (
        self.qkv(x_flat)
        .reshape(B, H * W, 3, 4, C // 4)
        .permute(2, 0, 3, 1, 4)
    )
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
    self.upsample = nn.Upsample(
        scale_factor=4, mode='bicubic', align_corners=False
    )
    self.head = nn.Conv2d(3, 64, 3, padding=1)
    self.layers = nn.ModuleList([PFA_Block(64) for _ in range(3)])
    self.upsample_feat = nn.Sequential(
        nn.Conv2d(64, 64 * 16, 3, padding=1), nn.PixelShuffle(4)
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
  _, C, H, W = lr_img.shape
  stride = max(1, tile_size - overlap)
  output = torch.zeros(
      (1, C, H * scale, W * scale), device=lr_img.device, dtype=torch.float32
  )
  weight = torch.zeros(
      (1, C, H * scale, W * scale), device=lr_img.device, dtype=torch.float32
  )

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
    self.x = np.load(f'{base_path}/val_tensor_x.npy')
    self.y = np.load(f'{base_path}/val_tensor_y.npy')

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, i):
    lr_np = self.x[i][:, :, ::-1].copy()
    hr_np = self.y[i][:, :, ::-1].copy()
    lr = torch.from_numpy(lr_np).float()
    hr = torch.from_numpy(hr_np).float()
    if lr.shape[-1] == 3:
      lr = lr.permute(2, 0, 1)
    if hr.shape[-1] == 3:
      hr = hr.permute(2, 0, 1)
    if lr_np.max() > 1.5:
      lr = lr / 255.0
    if hr_np.max() > 1.5:
      hr = hr / 255.0
    return lr, hr


def safe_ssim(hr_np, pred_np):
  h, w = hr_np.shape[:2]
  min_side = min(h, w)
  if min_side < 3:
    return 0.0
  win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
  return ssim_metric(
      hr_np, pred_np, data_range=1.0, channel_axis=-1, win_size=win_size
  )


def run_live_demo(num_samples=4, tile_size=64, overlap=8):
  release_old_training_memory()

  print('🎬 現場流程展示模式啟動（展示 4x 切塊推論、指標計算與對比出圖）')
  val_ds = ValOnlyDataset(DATA_PATH)
  model = PFTNet().to(device)
  model.eval()

  indices = random.sample(range(len(val_ds)), min(num_samples, len(val_ds)))
  print(f'🎲 亂數抽樣索引：{indices}')

  fig, axes = plt.subplots(len(indices), 2, figsize=(12, 4 * len(indices)))
  if len(indices) == 1:
    axes = np.expand_dims(axes, axis=0)

  with torch.no_grad():
    for row_idx, img_idx in enumerate(indices):
      lr_t, hr_t = val_ds[img_idx]
      lr_in = lr_t.unsqueeze(0).to(device)

      img_sr = predict_with_tiles(
          model, lr_in, scale=4, tile_size=tile_size, overlap=overlap
      )
      img_bi = F.interpolate(
          lr_in, scale_factor=4, mode='bicubic', align_corners=False
      ).float()

      img_hr_np = hr_t.permute(1, 2, 0).cpu().numpy()
      img_bi_np = np.clip(img_bi[0].permute(1, 2, 0).cpu().numpy(), 0, 1)
      img_sr_np = np.clip(img_sr[0].permute(1, 2, 0).cpu().numpy(), 0, 1)

      bi_p = psnr_metric(img_hr_np, img_bi_np, data_range=1.0)
      bi_s = safe_ssim(img_hr_np, img_bi_np)
      sr_p = psnr_metric(img_hr_np, img_sr_np, data_range=1.0)
      sr_s = safe_ssim(img_hr_np, img_sr_np)

      axes[row_idx, 0].imshow(img_bi_np)
      axes[row_idx, 0].set_title(
          f'Val #{img_idx} [Bicubic]\nPSNR: {bi_p:.2f}dB | SSIM: {bi_s:.4f}'
      )
      axes[row_idx, 0].axis('off')

      axes[row_idx, 1].imshow(img_sr_np)
      axes[row_idx, 1].set_title(
          f'Val #{img_idx} [PFTNet (Live Demo)]\nPSNR: {sr_p:.2f}dB | SSIM:'
          f' {sr_s:.4f}'
      )
      axes[row_idx, 1].axis('off')

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


run_live_demo(num_samples=4, tile_size=64, overlap=8)
