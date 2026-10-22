# Image-Super-Resolution-Reconstruction-via-Progressive-Feature-Transmission-and-Vision-Transformer
```bash
# 1. 複製專案
git clone https://github.com/yujuan1118/PFTNet-Super-Resolution.git
cd PFTNet-Super-Resolution

# 2. 安裝相依套件
pip install -r requirements.txt

PFTNet-Super-Resolution/
├── dataset/             # 訓練與驗證集 (DIV2K / Flickr2K / 驗證影像)
├── models/              # PFTNet 網路架構與 PFA 模組定義
├── weights/             # 訓練權重存放區 (包含 best_pftnet.pth)
├── train.py             # 模型訓練主程式 (含 Patch-based 全體評估機制)
├── test.py              # 單張/批次超解析度推論與視覺化程式
├── requirements.txt     # 專案相依套件列表
└── README.md            # 專案說明文件

# 開始訓練 (150 epochs)
python train.py --epochs 150 --batch_size 32

# 執行模型評估與推論
python test.py --weights weights/best_pftnet.pth
