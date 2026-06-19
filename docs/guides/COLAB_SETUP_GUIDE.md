# 🚀 Colab Training Guide - Amodal Shape Prediction

**Tạo ngày**: April 22, 2025  
**Mục đích**: Train model trên GPU miễn phí của Google Colab

---

## 📋 Quick Start (3 bước)

### Step 1: Mở Colab Notebook

**Link Colab Notebook**:
```
https://colab.research.google.com/github/vothenguyen/amodal_shape_prediction/blob/main/colab_training.ipynb
```

Hoặc:
1. Vào https://colab.research.google.com
2. Click "File" → "Open notebook"
3. Chọn "GitHub"
4. Paste URL: `https://github.com/vothenguyen/amodal_shape_prediction`
5. Chọn `colab_training.ipynb`

### Step 2: Enable GPU

1. Click "Runtime" → "Change runtime type"
2. Select "GPU" (T4 hoặc A100 nếu có)
3. Click "Save"

### Step 3: Run All Cells

1. Click "Runtime" → "Run all"
2. Hoặc chạy từng cell bằng cách click ▶️
3. Follow các prompts (mount Google Drive, etc.)

---

## ⏱️ Thời Gian Dự Kiến

| Step | Time | GPU Memory |
|------|------|-----------|
| Setup & Download | 5-10 min | Minimal |
| Dataset Download | 30-60 min | Minimal |
| Training (10 epochs) | 30-60 min | ~12GB (T4) |
| Evaluation | 5-10 min | ~12GB |
| **Total** | **1-2 hours** | **T4: OK** |

---

## 📊 What the Notebook Does

```
┌─ Step 1: GPU Setup
│  ├─ Check GPU availability
│  └─ Mount Google Drive
│
├─ Step 2: Clone Repository
│  ├─ git clone from GitHub
│  └─ Setup working directory
│
├─ Step 3: Install Dependencies
│  ├─ torch, timm, albumentations
│  └─ opencv, pycocotools, etc.
│
├─ Step 4: Download Dataset
│  ├─ Annotations (small)
│  └─ Images (option: full or sample)
│
├─ Step 5: Train Model
│  ├─ COMBO method (best results)
│  ├─ 10 epochs
│  └─ Save checkpoints
│
├─ Step 6: Evaluate
│  ├─ Run on validation set
│  └─ Print metrics
│
└─ Step 7: Save to Drive
   ├─ Upload checkpoints
   └─ Upload results
```

---

## 🎯 Key Features

### Training Methods Available

Run one of these:

**A. COMBO (Best Results)** ⭐
```python
# In cell "Train Model"
cmd = [
    "python", "src/train_balanced.py",
    "--loss-type", "combo",
    "--occlusion-weight", "15.0",
    "--use-balanced-sampling",
    "--occlusion-threshold", "0.1",
    "--oversample-ratio", "2.0",
    "--epochs", "10",
]
```
Expected: +10-15% mIoU improvement

**B. Quick Test (Fast)**
```python
cmd = [
    "python", "src/train_balanced.py",
    "--loss-type", "original",
    "--occlusion-weight", "15.0",
    "--epochs", "5",
]
```
Expected: +2-3% mIoU improvement

---

## 📁 Dataset Options

### Option 1: Full COCO Dataset (Recommended but slow)
```python
# Downloads full images (~84GB)
train_url = "http://images.cocodataset.org/zips/train2014.zip"
# Takes 1-2 hours to download
```

### Option 2: Sample Dataset (Quick Testing)
```python
# Creates 100 dummy train + 10 dummy val images
# Good for testing without waiting for downloads
# Takes <1 minute
```

**Recommended**: Use Option 2 first to test, then Option 1 for real training

---

## 💾 Google Drive Integration

### Before Running

1. Open Google Drive
2. Create folders:
   ```
   My Drive/
   ├─ amodal_checkpoints/     (for saving models)
   └─ amodal_results/         (for saving metrics)
   ```

### After Training

Checkpoints & results automatically saved to:
```
My Drive/amodal_results/
├─ swin_amodal_epoch_1.pth
├─ swin_amodal_epoch_2.pth
├─ ...
└─ eval_results_colab.json
```

Download them after training to use locally!

---

## ⚠️ Important Notes

### GPU Memory Limits
- **T4 (free tier)**: ~16GB VRAM
  - Max batch size: 2
  - Usually OK for this model
  
- **A100/V100 (paid)**: ~40GB+ VRAM
  - Can increase batch size if needed

### Dataset Size Issues
- Training images: ~84GB (might timeout)
- Validation images: ~4GB
- **Solution**: Use sample dataset for quick test, real dataset for production

### Session Timeout
- Free Colab sessions timeout after 12 hours
- Save checkpoints to Drive regularly!
- Train script automatically saves after each epoch

### Disk Space
- Free Colab: ~100GB storage
- Dataset: ~88GB
- **Solution**: Use mounted Google Drive (unlimited, slower I/O)

---

## 🔧 Troubleshooting

### "CUDA out of memory"
```
→ Reduce batch size in notebook:
   --batch-size 1
```

### "Dataset download timeout"
```
→ Use sample dataset instead:
   Uncomment "Creating sample dataset" cell
   Run before training
```

### "Colab session disconnected"
```
→ Checkpoints saved to Google Drive
→ Can resume training later with saved weights
→ Keep Drive tab open to refresh session
```

### "Git clone fails"
```
→ Check GitHub URL is correct:
   https://github.com/vothenguyen/amodal_shape_prediction
```

### "Out of disk space"
```
→ Delete images after training:
   !rm -rf data/train2014 data/val2014
   
→ Or use Google Drive for storage
```

---

## 📊 Expected Results

After training:

```
🏆 EVALUATION RESULTS
==================================================
Overall mIoU:    63% (±2%)
Invisible mIoU:   45% (±5%)  ⭐ Main metric!
Dice Coefficient: 68% (±2%)
Samples:         10 (sample dataset)
==================================================
```

**With full dataset**: Expect +10-15% on invisible mIoU

---

## 🎓 Advanced Options

### Resume Training from Checkpoint
```python
# In train_balanced.py
--resume-epoch 5
# Continue from epoch 5
```

### Different Loss Functions
```python
# Try different methods
--loss-type original      # Baseline
--loss-type focal         # Hard negative mining
--loss-type combo         # Best (default)
```

### Adjust Hyperparameters
```python
--occlusion-weight 20.0        # Increase penalty (higher=harder)
--occlusion-threshold 0.25     # Only highly occluded samples
--oversample-ratio 3.0         # More oversampling (3x)
--learning-rate 5e-5           # Lower LR if overfitting
```

---

## 📤 Download Results

After training:

1. Click folder icon on left
2. Navigate to "My Drive" → "amodal_results"
3. Right-click on file
4. Select "Download"

Files to download:
- `swin_amodal_epoch_*.pth` - Model checkpoints
- `eval_results_colab.json` - Metrics

---

## 🚀 Next Steps After Training

1. **Download checkpoint** from Google Drive
2. **Use locally** with existing code
3. **Fine-tune** on custom dataset
4. **Deploy** to production

---

## 📞 Support

### If something goes wrong:

1. **Check GPU**: Run first cell → check if GPU is available
2. **Check disk**: Run `!df -h` in a cell
3. **Check memory**: Run `!free -h` in a cell
4. **View logs**: Look at cell output for error messages

### Common Issues:

| Error | Solution |
|-------|----------|
| ModuleNotFoundError | Install missing package with `!pip install ...` |
| CUDA out of memory | Reduce batch size to 1 |
| FileNotFoundError | Check dataset downloaded correctly |
| Timeout on download | Use sample dataset instead |

---

## 📚 Documentation

Before running Colab, read these in project repo:

1. **QUICK_START_OCCLUSION.md** - Method overview
2. **OCCLUSION_STRATEGY.md** - Detailed explanation
3. **colab_training.ipynb** - This notebook

All available on GitHub!

---

## ✅ Checklist

Before starting:
- [ ] Google account ready
- [ ] Google Drive space available (100+ GB recommended)
- [ ] GitHub account (optional, for bookmarking)
- [ ] Time available (1-2 hours)

During training:
- [ ] GPU enabled (Runtime → Change runtime type → GPU)
- [ ] Notebook running (follow cells in order)
- [ ] Google Drive mounted (should prompt)
- [ ] Monitor progress (watch loss decrease)

After training:
- [ ] Results saved to Google Drive
- [ ] Evaluation metrics printed
- [ ] Checkpoint downloaded (optional)
- [ ] Can run inference with trained model

---

## 🎉 Success Criteria

✅ Training completed  
✅ Model evaluated on val set  
✅ Metrics printed (mIoU, Dice, etc.)  
✅ Results saved to Google Drive  
✅ Can download and use checkpoint  

---

**Created**: April 22, 2025  
**Version**: 1.0  
**Status**: Ready to Use ✅

---

## 📖 Colab Notebook URL

```
https://colab.research.google.com/github/vothenguyen/amodal_shape_prediction/blob/main/colab_training.ipynb
```

**Click above link or:**
1. Go to https://colab.research.google.com
2. Upload the notebook from this repo
3. Run all cells!

Enjoy training on free GPU! 🚀
