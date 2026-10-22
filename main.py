#有圖就立刻秀出來
%matplotlib inline

#cv2 影像處理器，torch數學運算、張量計算，torch.nn神經網路的層，matplotlib.pyplot 繪圖，Dataset, DataLoader 前者照片定義，後者班照片給ai訓練
#os，glob把所有照片檔案找出來，gc記憶體清空，numpy 處理數字陣列
import torch, os, gc, numpy as np, random, time
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler #自動混合精度，讓GPU負擔減半，才不會跑一下就當機
from torch.utils.data import Dataset, DataLoader
# 引入 PSNR 與 SSIM 計算工具 (skimage 用於單張繪圖，pytorch_msssim 用於 GPU 全集快速評估，tqdm看進度條)
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from pytorch_msssim import ssim as ssim_loss_fn
from tqdm import tqdm


#設定PyTorch記憶體分配優化，防止碎片化，將低OOM的機率
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def clean_vram(): #手動強制回收 GPU 記憶體
    gc.collect()   # 清除 Python 垃圾回收              
    torch.cuda.empty_cache()  # 釋放 PyTorch 佔用的顯存   
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

# 在啟動前做一次最徹底的 GPU 清空
clean_vram()
#選擇設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = '/kaggle/input/notebooks/ishanmanglik/div2k'

#模型定義
class PFA_Block(nn.Module):#漸進式特徵聚合模組
    def __init__(self, dim):
        super().__init__() #繼承pytorch官方神經網路的基本功能
        self.qkv = nn.Linear(dim, dim * 3)# Linear 層負責將影像特徵轉換為 Q (查訊), K (鍵值), V (數值) 矩陣
        self.proj = nn.Linear(dim, dim) #把self.qkv算完的結果轉回原本影像格式
        
    def forward(self, x, prev_attn=None):
        B, C, H, W = x.shape #B是批次，C是頻道(64個特徵)，H是高，W是寬
        x_flat = x.flatten(2).transpose(1, 2) #[B,64,128,128]先把二維壓平[B,64(特徵),128*128(像素點)]，再把編號1跟2互換位置[B,128*128,64]
        # 生成 QKV 矩陣，3=Q，K，V，4=從 4 個不同的角度去分析這張圖。
        qkv = self.qkv(x_flat).reshape(B, H*W, 3, 4, C//4).permute(2, 0, 3, 1, 4)
        #reshape(0:B, 1:H*W, 2:3(身分), 3:4(角度), 4:C//4(特徵))。
        #permute(2(身分), 0(B), 3(角度), 1(H*W), 4(特徵))。改變位置。
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 強制轉為 float32 計算 Attention，防止 FP16 數值溢出崩潰
        q, k = q.float(), k.float()
        
        # 計算注意力權重
        attn = (q @ k.transpose(-2, -1)) * ((C//4)**-0.5)
        # @矩陣乘法，**-0.5開根號取倒數(縮放)
        # 結合上一層，讓細節傳遞下去
        if prev_attn is not None: 
            attn = attn + 0.5*prev_attn
            #torch.sigmoid把數值轉成 0 到 1 之間的「權重（開關）」
        attn_out = attn.softmax(dim=-1).type_as(v)  #softmax()轉成百分比
        x_out = (attn_out @ v).transpose(1, 2).reshape(B, H*W, C)
        #注意力權重跟特徵細節結合，並把「多個角度」合併回來
        x_out = self.proj(x_out).transpose(1, 2).reshape(B, C, H, W) 
        #清單摺疊回照片
        return x + x_out, attn #回傳原始資訊學習及修復的部分

class PFTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 把影像長寬放大4倍，'bicubic'填補某個空白像素，放大出來的底圖會比較平滑、自然
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.head = nn.Conv2d(3, 64, 3, padding=1) #提取初步特徵，3=RGB，64=特徵數量，3=卷積核大小，padding=1補邊
        # 三層漸進式修復
        self.layers = nn.ModuleList([PFA_Block(64) for _ in range(3)]) 
        
        # 特徵圖上採樣（將特徵放大的核心，避免在放大後的圖上做 Attention 導致 128GB OOM）
        self.upsample_feat = nn.Sequential(
            nn.Conv2d(64, 64 * 16, 3, padding=1),
            nn.PixelShuffle(4)
        )
        self.tail = nn.Conv2d(64, 3, 3, padding=1)
        
    def forward(self, x):
        # 將原圖用 Bicubic 放大 4 倍作為基底
        base = self.upsample(x)
        # 在小圖上提取初步特徵並計算漸進式注意力（防止 GPU OOM）
        feat = self.head(x)
        attn = None
        for layer in self.layers: 
            feat, attn = layer(feat, attn) # 每一層都會傳遞上一層的注意力 map
            
        # 讓神經網路把學習到的特徵放大 4 倍
        res_feat = self.upsample_feat(feat)
        # 殘差預測
        res = self.tail(res_feat)
        out = base + res
        return torch.clamp(out, 0.0, 1.0) # 嚴格截斷在合法影像像素範圍 [0, 1] 

# GPU版 PSNR 計算函數
def calculate_psnr_gpu(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100.0
    return 10.0 * torch.log10(1.0 / mse)

#資料集載入 
class DIV2K_NpyDataset(Dataset):
    def __init__(self, base_path, stage='train'):
        print(f"正在載入 {stage.upper()} 階段的 NumPy 矩陣資料...")
        if stage == 'train':
            self.x = np.load(os.path.join(base_path, 'tensor_x.npy'))#self.x儲存所有低解析度的模糊小圖
            self.y = np.load(os.path.join(base_path, 'tensor_y.npy'))#self.y儲存所有高解析度的高清原圖
        else:
            self.x = np.load(os.path.join(base_path, 'val_tensor_x.npy'))
            self.y = np.load(os.path.join(base_path, 'val_tensor_y.npy'))
        print(f"{stage.upper()} 載入成功！LR 形狀: {self.x.shape} | HR 形狀: {self.y.shape}")
            
    def __len__(self):#計算總量 
        return self.x.shape[0]
    
    def __getitem__(self, i):
        #[:, :, ::-1]色彩翻轉.copy()記憶體重組
        lr_np = self.x[i][:, :, ::-1].copy()
        hr_np = self.y[i][:, :, ::-1].copy()

        
        lr = torch.from_numpy(lr_np).float()
        hr = torch.from_numpy(hr_np).float()

        #PyTorch 的卷積層規定輸入格式必須是 (通道 C, 高 H, 寬 W)
        if lr.shape[-1] == 3: lr = lr.permute(2, 0, 1)
        if hr.shape[-1] == 3: hr = hr.permute(2, 0, 1)
            
        if lr_np.max() > 1.5:
            lr = lr / 255.0
        if hr_np.max() > 1.5:
            hr = hr / 255.0
        return lr, hr

#訓練準備
model = PFTNet().to(device)
train_dataset = DIV2K_NpyDataset(DATA_PATH, stage='train')
val_dataset = DIV2K_NpyDataset(DATA_PATH, stage='val')

# 加大 batch_size 加速訓練全集
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# AdamW 優化器
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) 

TOTAL_EPOCHS = 150 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_EPOCHS, eta_min=1e-6)
scaler = GradScaler('cuda')# 自動混合精度

# 固定觀察 4 張驗證集照片
num_val_samples = len(val_dataset)
fixed_sample_indices = random.sample(range(num_val_samples), min(4, num_val_samples))

# 儲存最佳模型(最佳 PSNR)
best_psnr = 0.0
best_model_path = 'best_pftnet.pth'

print(f"{TOTAL_EPOCHS}輪訓練啟動！已部署「Patch-based全體評估」與「最佳模型儲存」機制...")

#訓練主迴圈
for e in range(1, TOTAL_EPOCHS + 1):
    model.train()
    running_loss = 0.0

    # 使用 tqdm 顯示訓練進度條
    pbar = tqdm(train_loader, desc=f"Epoch [{e}/{TOTAL_EPOCHS}]", leave=False)
    for lr, hr in pbar:
        lr, hr = lr.to(device), hr.to(device)
        opt.zero_grad() # 梯度歸零
        
        with autocast('cuda'): #開啟混合精度訓練
            out = model(lr)
            l1_loss = F.l1_loss(out, hr)
            out_f32 = out.float()
            hr_f32 = hr.float()
            # 混合損失函數
            ssim_loss = 1.0 - ssim_loss_fn(out_f32, hr_f32, data_range=1.0, size_average=True)
            
            loss = l1_loss + 0.1 * ssim_loss
            
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(opt)
        scaler.update()
        running_loss += loss.item()
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    scheduler.step()
    epoch_train_loss = running_loss / len(train_loader)
    
    if e == 1 or e % 25 == 0:
        print(f"Epoch [{e}/{TOTAL_EPOCHS}] | 綜合訓練 Loss: {epoch_train_loss:.4f}")

    #評估全部與儲存最佳(每 50 輪)
    if e % 50 == 0:
        print(f"\n==================== 【第 {e} 輪全體驗證集定量評估】 ====================")
        print(f"正在使用 GPU Patch 推論模式評估全部 {len(val_dataset)} 張驗證影像...")
        
        model.eval() 
        clean_vram() #先清一次訓練留下的記憶體
        
        all_pft_psnrs = []
        all_pft_ssims = []
        all_bi_psnrs = []
        all_bi_ssims = []
        
        t_start = time.time()
        
        #評估全部驗證影像
        with torch.no_grad():
            for lr_val, hr_val in val_loader:
                lr_val, hr_val = lr_val.to(device), hr_val.to(device)

                #開啟混合精度降低注意力矩陣顯存佔用
                with autocast('cuda'):
                    img_sr = model(lr_val).float()

                img_bi = F.interpolate(lr_val, scale_factor=4, mode='bicubic', align_corners=False).float()
                
                # GPU 加速計算指標
                pft_psnr = calculate_psnr_gpu(img_sr, hr_val)
                pft_ssim = ssim_loss_fn(img_sr, hr_val, data_range=1.0, size_average=True)
                
                bi_psnr = calculate_psnr_gpu(img_bi, hr_val)
                bi_ssim = ssim_loss_fn(img_bi, hr_val, data_range=1.0, size_average=True)
                
                all_pft_psnrs.append(pft_psnr.item())
                all_pft_ssims.append(pft_ssim.item())
                all_bi_psnrs.append(bi_psnr.item())
                all_bi_ssims.append(bi_ssim.item())
                
                # 即時清空每張圖產生的中間變數顯存
                del img_sr, img_bi
        clean_vram()        
        t_end = time.time()
        
        avg_sr_psnr, avg_sr_ssim = np.mean(all_pft_psnrs), np.mean(all_pft_ssims)
        avg_bi_psnr, avg_bi_ssim = np.mean(all_bi_psnrs), np.mean(all_bi_ssims)
        
        print(f"全體驗證集評估完成！耗時: {t_end - t_start:.1f} 秒。模式: Patch-based (64x64)")
        
        print("📊 全體驗證集客觀數據對比表格：")
        print("| 方法 (Method) | 平均 PSNR (dB) ↑ | 平均 SSIM ↑ |")
        print("| :--- | :--- | :--- |")
        print(f"| 傳統放大 (Bicubic Baseline) | {avg_bi_psnr:.2f} dB | {avg_bi_ssim:.4f} |")
        print(f"| 本論文方法 (PFTNet - Ours) | **{avg_sr_psnr:.2f} dB** | **{avg_sr_ssim:.4f}** |")
        print(f"| 淨提升效益 (Gain) | +{avg_sr_psnr - avg_bi_psnr:.2f} dB | +{avg_sr_ssim - avg_bi_ssim:.4f} |")
        print("-" * 65)

        #儲存「最佳」模型
        if avg_sr_psnr > best_psnr:
            best_psnr = avg_sr_psnr
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 發現更佳模型！已將權重儲存至 '{best_model_path}' (最高 PSNR: {best_psnr:.2f} dB)")
        else:
            print(f"ℹ️ 當前 PSNR ({avg_sr_psnr:.2f} dB) 未超越歷史最佳 ({best_psnr:.2f} dB)。保留原最佳模型。")

        #多圖展示 (定性分析)
        print("🖼️ 正在繪製對比圖...")
        fig, axes = plt.subplots(len(fixed_sample_indices), 2, figsize=(12, 4 * len(fixed_sample_indices)))
        
        with torch.no_grad():
            for row_idx, img_idx in enumerate(fixed_sample_indices):
                lr_t, hr_t = val_dataset[img_idx]
                lr_t = lr_t.unsqueeze(0).to(device)

                with autocast('cuda'):
                    img_sr = model(lr_t).float()

                img_bicubic = F.interpolate(lr_t, scale_factor=4, mode='bicubic', align_corners=False).float()

                # NumPy HWC 用於繪圖
                img_hr_np = hr_t.permute(1, 2, 0).numpy()
                img_bi_np = np.clip(img_bicubic[0].cpu().permute(1, 2, 0).numpy(), 0, 1)
                img_sr_np = np.clip(img_sr[0].cpu().permute(1, 2, 0).numpy(), 0, 1)

                bi_p = psnr_metric(img_hr_np, img_bi_np, data_range=1.0)
                bi_s = ssim_metric(img_hr_np, img_bi_np, data_range=1.0, channel_axis=-1)
                sr_p = psnr_metric(img_hr_np, img_sr_np, data_range=1.0)
                sr_s = ssim_metric(img_hr_np, img_sr_np, data_range=1.0, channel_axis=-1)

                # 左欄：傳統 Bicubic
                axes[row_idx, 0].imshow(img_bi_np)
                axes[row_idx, 0].set_title(f"Sample {row_idx+1} [Bicubic]\nPSNR: {bi_p:.2f}dB | SSIM: {bi_s:.4f}")
                axes[row_idx, 0].axis("off")

                # 右欄：AI PFTNet
                axes[row_idx, 1].imshow(img_sr_np)
                axes[row_idx, 1].set_title(f"Sample {row_idx+1} [PFTNet - Epoch {e}]\nPSNR: {sr_p:.2f}dB | SSIM: {sr_s:.4f}")
                axes[row_idx, 1].axis("off")
                
        plt.tight_layout()
        plt.show()  
        plt.close(fig)  

        clean_vram()
        print("=" * 65 + "\n")
