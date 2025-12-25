# 数据集下载快速指南

## 🎯 推荐下载顺序

### 第1步：MOT17（最简单，推荐首选）
- 大小: 5.5 GB
- 下载方式: 官网直接下载
- 时间: 约30分钟-2小时（取决于网速）

**下载链接**: https://motchallenge.net/data/MOT17/

**操作步骤**:
1. 访问上述链接
2. 点击 "Download" 按钮
3. 下载 `MOT17.zip`
4. 将zip文件放到 `data/` 目录
5. 运行: `python download_datasets.py --check`

### 第2步：MOT20（中等难度）
- 大小: 13 GB
- 下载方式: 官网直接下载
- 时间: 约1-3小时

**下载链接**: https://motchallenge.net/data/MOT20/

### 第3步：BDD100K（需要注册）
- 大小: 30 GB（跟踪子集）
- 下载方式: 注册后下载
- 时间: 注册审核1-2天 + 下载2-5小时

**注册链接**: https://bdd-data.berkeley.edu/

### 第4步：TAO（可选，非常大）
- 大小: 500 GB
- 建议: 先完成前3个数据集的实验

---

## 📥 快速命令

```bash
# 1. 检查当前状态
python download_datasets.py --check

# 2. 创建目录结构
python download_datasets.py --dataset all

# 3. 手动下载数据集后，再次检查
python download_datasets.py --check
```

---

## 💡 实用建议

1. **先下载MOT17**: 最小、最简单、最常用
2. **使用下载工具**: IDM、FDM等可以断点续传
3. **检查磁盘空间**: 至少预留50 GB
4. **分批下载**: 不要一次下载所有数据集

---

详细说明请查看: `DATASET_DOWNLOAD_GUIDE.md`
