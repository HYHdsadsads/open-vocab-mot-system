"""
MOT17数据集直接下载脚本

这个脚本会尝试直接从MOT Challenge服务器下载MOT17数据集
"""

import os
import requests
from tqdm import tqdm


def download_file(url, filename):
    """下载文件并显示进度条"""
    print(f"\n正在下载: {url}")
    print(f"保存到: {filename}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    print(f"✓ 下载完成: {filename}")


def main():
    # 创建data目录
    os.makedirs('./data', exist_ok=True)
    
    print("="*60)
    print("MOT17数据集下载")
    print("="*60)
    
    # MOT17下载链接（尝试多个镜像）
    urls = [
        "https://motchallenge.net/data/MOT17.zip",
        "https://motchallenge.net/data/MOT17Labels.zip",  # 仅标注
    ]
    
    print("\n⚠️  注意：")
    print("1. MOT17数据集约5.5 GB，下载需要时间")
    print("2. 如果下载失败，请手动访问: https://motchallenge.net/data/MOT17/")
    print("3. 下载后将MOT17.zip放到data/目录")
    
    choice = input("\n是否开始下载？(y/n): ")
    
    if choice.lower() == 'y':
        try:
            download_file(urls[0], './data/MOT17.zip')
            print("\n✓ MOT17数据集下载完成！")
            print("下一步: 运行 python download_datasets.py --check")
        except Exception as e:
            print(f"\n✗ 下载失败: {e}")
            print("请手动下载: https://motchallenge.net/data/MOT17/")
    else:
        print("\n已取消下载")
        print("请手动下载: https://motchallenge.net/data/MOT17/")


if __name__ == '__main__':
    main()
