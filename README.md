# Image-Super-Resolution-Reconstruction-via-Progressive-Feature-Transmission-and-Vision-Transformer

### 專題檔案說明
* `main.py`：包含 PFTNet 模型架構、PFA 漸進式注意力模組、DIV2K 資料集載入、150 輪訓練主迴圈，以及每 50 輪全體驗證集定量評估與對比圖繪製之完整程式碼。
### 專題簡報
* [點此查看專題簡報][(AI Ethical Scenarios Presentation in Black Colourful Modern Lined Style.pdf)](https://github.com/yujuan1118/Image-Super-Resolution-Reconstruction-via-Progressive-Feature-Transmission-and-Vision-Transformer/blob/main/AI%20Ethical%20Scenarios%20Presentation%20in%20Black%20Colourful%20Modern%20Lined%20Style.pdf)

###  執行方式
```bash
# 安裝相依套件
pip install torch torchvision numpy matplotlib skimage pytorch-msssim tqdm

# 執行模型訓練與全體驗證集評估
python main.py
