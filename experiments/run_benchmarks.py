"""在公开数据集上运行对比实验"""
import os
os.environ["OMP_NUM_THREADS"] = "1"  # 解决KMeans内存泄漏警告
from pipeline.system import OpenVocabMOTSystem
from config import DATASET_CONFIG
import pandas as pd
import json



def run_benchmarks():
    # 定义对比模型配置
    model_configs = {
        "baseline": {  # 无优化的基线模型

            "dynamic_weight": False,
            "semantic_evolution": False
        },
        "our_model": {  # 本文提出的模型
            "dynamic_weight": True,
            "semantic_evolution": True
        }
    }

    # 定义待验证的数据集
    datasets = ["MOT17", "MOT20", "LVIS"]
    results = {}

    for dataset in datasets:
        results[dataset] = {}
        # 获取数据集类别（已知+零样本）
        known_classes = DATASET_CONFIG["datasets"][dataset].get("classes", [])
        unknown_classes = DATASET_CONFIG["datasets"][dataset].get("unknown_classes", [])

        for model_name, config in model_configs.items():
            # 初始化系统
            mot_system = OpenVocabMOTSystem(known_classes)
            # 配置模型（启用/禁用优化模块）
            mot_system.dictionary.dynamic_weight = config["dynamic_weight"]
            mot_system.zero_shot_handler.semantic_evolution = config["semantic_evolution"]

            # 运行评估
            print(f"=== 评估 {model_name} 在 {dataset} 上 ===")
            metrics = mot_system.evaluate_on_dataset(dataset, split="test")
            results[dataset][model_name] = metrics

    # 保存结果
    import json
    with open("./experiments/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("实验完成，结果已保存")

"""在公开数据集上运行对比实验（仅测试MOT17）"""
import os
os.environ["OMP_NUM_THREADS"] = "1"  # 解决KMeans内存泄漏警告
from pipeline.system import OpenVocabMOTSystem
from config import DATASET_CONFIG
import pandas as pd
import json


def run_mot17_benchmark():
    # 定义对比模型配置
    model_configs = {
        "baseline": {  # 无优化的基线模型
            "dynamic_weight": False,
            "semantic_evolution": False
        },
        "our_model": {  # 本文提出的模型
            "dynamic_weight": True,
            "semantic_evolution": True
        }
    }

    # 仅测试MOT17数据集
    datasets = ["MOT17"]
    results = {}

    for dataset in datasets:
        results[dataset] = {}
        # 获取数据集类别（已知+零样本）
        known_classes = DATASET_CONFIG["datasets"][dataset].get("classes", [])
        unknown_classes = DATASET_CONFIG["datasets"][dataset].get("unknown_classes", [])

        for model_name, config in model_configs.items():
            # 初始化系统
            mot_system = OpenVocabMOTSystem(known_classes)
            # 配置模型（启用/禁用优化模块）
            mot_system.dictionary.dynamic_weight = config["dynamic_weight"]
            mot_system.zero_shot_handler.semantic_evolution = config["semantic_evolution"]

            # 运行评估（仅测试集）
            print(f"=== 评估 {model_name} 在 {dataset} 测试集上 ===")
            metrics = mot_system.evaluate_on_dataset(dataset, split="test")
            results[dataset][model_name] = metrics

        # 保存结果
        output_dir = "./experiments"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "mot17_benchmark_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print("MOT17实验完成，结果已保存")

        # 打印MOT17的准确率指标（修复KeyError）
        rows = []
        for dataset in results:
            for model in results[dataset]:
                # 从评估结果中提取小写的mota和idf1（无average层级）
                mota = results[dataset][model].get("mota", 0.0)  # 使用get避免键不存在报错
                idf1 = results[dataset][model].get("idf1", 0.0)
                rows.append({
                    "数据集": dataset,
                    "模型": model,
                    "MOTA": round(mota, 3),  # 保留3位小数
                    "IDF1": round(idf1, 3)
                })
        df = pd.DataFrame(rows)
        print("\nMOT17测试集准确率结果：")
        print(df.pivot(index="数据集", columns="模型", values=["MOTA", "IDF1"]))

if __name__ == "__main__":
    run_mot17_benchmark()
    # run_benchmarks()
    # results = json.load(open("./experiments/benchmark_results.json", "r"))
    # rows = []
    # for dataset in results:
    #     for model in results[dataset]:
    #         if dataset in ["MOT17", "MOT20"]:
    #             rows.append({
    #                 "数据集": dataset,
    #                 "模型": model,
    #                 "MOTA": results[dataset][model]["average"]["MOTA"],
    #                 "IDF1": results[dataset][model]["average"]["IDF1"]
    #             })
    #         else:
    #             rows.append({
    #                 "数据集": dataset,
    #                 "模型": model,
    #                 "准确率": results[dataset][model]["accuracy"],
    #                 "mAP": results[dataset][model]["mAP"]
    #             })
    #
    # df = pd.DataFrame(rows)
    # print(df.pivot(index="数据集", columns="模型", values=["MOTA", "IDF1", "准确率", "mAP"]))
    #

