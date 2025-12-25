"""
多目标跟踪数据集下载脚本

支持数据集：
- MOT17: 多目标跟踪基准（~5.5 GB）
- MOT20: 高密度场景跟踪（~13 GB）
- BDD100K: 自动驾驶场景（~100 GB，可选择子集）
- TAO: 开放词汇跟踪（需要注册）

使用方法：
    python download_datasets.py --dataset mot17
    python download_datasets.py --dataset all
"""

import os
import argparse
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """下载文件并显示进度条"""
    print(f"正在下载: {url}")
    print(f"保存到: {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    print(f"✓ 下载完成: {output_path}")


def extract_zip(zip_path, extract_to):
    """解压ZIP文件"""
    print(f"正在解压: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✓ 解压完成: {extract_to}")


def extract_tar(tar_path, extract_to):
    """解压TAR文件"""
    print(f"正在解压: {tar_path}")
    with tarfile.open(tar_path, 'r:*') as tar_ref:
        tar_ref.extractall(extract_to)
    print(f"✓ 解压完成: {extract_to}")


def download_mot17(data_dir):
    """
    下载MOT17数据集
    
    数据集信息：
    - 大小: ~5.5 GB
    - 训练集: 7个序列（5,316帧）
    - 测试集: 7个序列（5,919帧）
    - 类别: 行人
    """
    print("\n" + "="*60)
    print("下载 MOT17 数据集")
    print("="*60)
    
    mot17_dir = os.path.join(data_dir, 'MOT17')
    os.makedirs(mot17_dir, exist_ok=True)
    
    # MOT17下载链接
    urls = {
        'train': 'https://motchallenge.net/data/MOT17.zip',
    }
    
    print("\n⚠️  注意：MOT17数据集需要从官网手动下载")
    print("请访问: https://motchallenge.net/data/MOT17/")
    print("下载后将MOT17.zip放到data/目录下")
    print(f"目标路径: {mot17_dir}")
    
    # 检查是否已下载
    zip_path = os.path.join(data_dir, 'MOT17.zip')
    if os.path.exists(zip_path):
        print(f"\n✓ 找到MOT17.zip，开始解压...")
        extract_zip(zip_path, data_dir)
        print(f"\n✓ MOT17数据集准备完成: {mot17_dir}")
    else:
        print(f"\n✗ 未找到MOT17.zip")
        print("请手动下载后重新运行此脚本")
    
    return mot17_dir


def download_mot20(data_dir):
    """
    下载MOT20数据集
    
    数据集信息：
    - 大小: ~13 GB
    - 训练集: 4个序列（8,931帧）
    - 测试集: 4个序列（4,479帧）
    - 类别: 行人（高密度场景）
    """
    print("\n" + "="*60)
    print("下载 MOT20 数据集")
    print("="*60)
    
    mot20_dir = os.path.join(data_dir, 'MOT20')
    os.makedirs(mot20_dir, exist_ok=True)
    
    print("\n⚠️  注意：MOT20数据集需要从官网手动下载")
    print("请访问: https://motchallenge.net/data/MOT20/")
    print("下载后将MOT20.zip放到data/目录下")
    print(f"目标路径: {mot20_dir}")
    
    # 检查是否已下载
    zip_path = os.path.join(data_dir, 'MOT20.zip')
    if os.path.exists(zip_path):
        print(f"\n✓ 找到MOT20.zip，开始解压...")
        extract_zip(zip_path, data_dir)
        print(f"\n✓ MOT20数据集准备完成: {mot20_dir}")
    else:
        print(f"\n✗ 未找到MOT20.zip")
        print("请手动下载后重新运行此脚本")
    
    return mot20_dir


def download_bdd100k(data_dir, subset='tracking'):
    """
    下载BDD100K数据集
    
    数据集信息：
    - 完整数据集: ~100 GB
    - 跟踪子集: ~30 GB
    - 类别: 8类（car, pedestrian, truck, bus, train, motorcycle, bicycle, rider）
    """
    print("\n" + "="*60)
    print("下载 BDD100K 数据集")
    print("="*60)
    
    bdd100k_dir = os.path.join(data_dir, 'BDD100K')
    os.makedirs(bdd100k_dir, exist_ok=True)
    
    print("\n⚠️  注意：BDD100K数据集需要注册后下载")
    print("步骤：")
    print("1. 访问: https://bdd-data.berkeley.edu/")
    print("2. 注册账号并登录")
    print("3. 下载 'MOT 2020 Tracking' 数据集")
    print("4. 将下载的文件放到data/BDD100K/目录下")
    print(f"目标路径: {bdd100k_dir}")
    
    print("\n推荐下载文件：")
    print("- bdd100k_images_track_train.zip (~20 GB)")
    print("- bdd100k_images_track_val.zip (~5 GB)")
    print("- bdd100k_labels_release.zip (~500 MB)")
    
    return bdd100k_dir


def download_tao(data_dir):
    """
    下载TAO数据集
    
    数据集信息：
    - 大小: ~500 GB（完整）
    - 视频数: 2,907个序列
    - 类别: 833个类别
    """
    print("\n" + "="*60)
    print("下载 TAO 数据集")
    print("="*60)
    
    tao_dir = os.path.join(data_dir, 'TAO')
    os.makedirs(tao_dir, exist_ok=True)
    
    print("\n⚠️  注意：TAO数据集非常大（~500 GB）")
    print("步骤：")
    print("1. 访问: https://taodataset.org/")
    print("2. 阅读数据集说明")
    print("3. 使用官方下载脚本")
    print("4. 建议先下载小规模子集进行测试")
    print(f"目标路径: {tao_dir}")
    
    print("\n官方下载命令：")
    print("git clone https://github.com/TAO-Dataset/tao.git")
    print("cd tao")
    print("python scripts/download/download_annotations.py")
    
    return tao_dir


def create_dataset_structure(data_dir):
    """创建数据集目录结构"""
    datasets = ['MOT17', 'MOT20', 'BDD100K', 'TAO']
    
    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"\n✓ 数据集目录结构已创建: {data_dir}")


def check_dataset_status(data_dir):
    """检查数据集下载状态"""
    print("\n" + "="*60)
    print("数据集状态检查")
    print("="*60)
    
    datasets = {
        'MOT17': ['train', 'test'],
        'MOT20': ['train', 'test'],
        'BDD100K': ['images', 'labels'],
        'TAO': ['frames', 'annotations']
    }
    
    for dataset, subdirs in datasets.items():
        dataset_dir = os.path.join(data_dir, dataset)
        if os.path.exists(dataset_dir):
            has_data = any(os.path.exists(os.path.join(dataset_dir, subdir)) for subdir in subdirs)
            status = "✓ 已下载" if has_data else "✗ 未下载"
            print(f"{dataset:15s} {status}")
        else:
            print(f"{dataset:15s} ✗ 目录不存在")


def main():
    parser = argparse.ArgumentParser(description='下载多目标跟踪数据集')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['mot17', 'mot20', 'bdd100k', 'tao', 'all'],
                        help='要下载的数据集')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据集保存目录')
    parser.add_argument('--check', action='store_true',
                        help='仅检查数据集状态')
    
    args = parser.parse_args()
    
    # 创建数据目录
    os.makedirs(args.data_dir, exist_ok=True)
    
    # 检查状态
    if args.check:
        check_dataset_status(args.data_dir)
        return
    
    print("\n" + "="*60)
    print("多目标跟踪数据集下载工具")
    print("="*60)
    print(f"数据保存目录: {args.data_dir}")
    
    # 创建目录结构
    create_dataset_structure(args.data_dir)
    
    # 下载数据集
    if args.dataset in ['mot17', 'all']:
        download_mot17(args.data_dir)
    
    if args.dataset in ['mot20', 'all']:
        download_mot20(args.data_dir)
    
    if args.dataset in ['bdd100k', 'all']:
        download_bdd100k(args.data_dir)
    
    if args.dataset in ['tao', 'all']:
        download_tao(args.data_dir)
    
    # 最终状态检查
    print("\n" + "="*60)
    print("下载完成！")
    print("="*60)
    check_dataset_status(args.data_dir)
    
    print("\n下一步：")
    print("1. 手动下载需要注册的数据集")
    print("2. 运行 python download_datasets.py --check 检查状态")
    print("3. 运行跟踪实验: python main.py --dataset mot17")


if __name__ == '__main__':
    main()
