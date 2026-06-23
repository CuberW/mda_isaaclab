import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path.home() / "trashbot_ws"
DEFAULT_LOG_DIR = ROOT / "data" / "logs"
DEFAULT_INPUT = DEFAULT_LOG_DIR / "yolo_task_execution_log.json"
DEFAULT_OUTPUT_DIR = DEFAULT_LOG_DIR


CATEGORY_CN = {
    "recyclable": "可回收垃圾",
    "kitchen": "厨余垃圾",
    "hazardous": "有害垃圾",
    "other": "其他垃圾",
    "unknown": "未知类别",
}

BIN_CN = {
    "bin_recyclable_blue": "蓝色可回收垃圾桶",
    "bin_kitchen_green": "绿色厨余垃圾桶",
    "bin_hazardous_red": "红色有害垃圾桶",
    "bin_other_gray": "灰色其他垃圾桶",
    "unknown": "未知垃圾桶",
}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_execution_log(data):
    if "execution_log" in data:
        return data["execution_log"]

    if "records" in data:
        return data["records"]

    if "tasks" in data:
        return data["tasks"]

    raise ValueError("Input JSON must contain execution_log, records, or tasks.")


def compute_final_metrics(execution_log):
    n = len(execution_log)

    if n == 0:
        return {
            "num_tasks": 0,
            "valid_planning_rate": 0.0,
            "grasp_success_rate": 0.0,
            "placement_success_rate": 0.0,
            "target_bin_accuracy": 0.0,
            "end_to_end_success_rate": 0.0,
            "average_confidence": 0.0,
            "average_duration_sec": 0.0,
            "category_count": {},
            "target_bin_count": {},
        }

    planned = 0
    grasp_success = 0
    placement_success = 0
    target_bin_correct = 0
    end_to_end_success = 0

    category_count = {}
    target_bin_count = {}

    confidence_sum = 0.0
    duration_sum = 0.0

    for item in execution_log:
        category = item.get("garbage_category", item.get("category", "unknown"))
        target_bin = item.get("target_bin", "unknown")

        category_count[category] = category_count.get(category, 0) + 1
        target_bin_count[target_bin] = target_bin_count.get(target_bin, 0) + 1

        if item.get("planning_status", "planned") == "planned":
            planned += 1

        if item.get("grasp_success", False):
            grasp_success += 1

        if item.get("placement_success", False):
            placement_success += 1

        if item.get("target_bin_correct", False):
            target_bin_correct += 1

        if item.get("end_to_end_success", False):
            end_to_end_success += 1

        confidence_sum += float(item.get("confidence", 0.0))
        duration_sum += float(item.get("duration_sec", 0.0))

    return {
        "num_tasks": n,
        "valid_planning_rate": round(planned / n, 4),
        "grasp_success_rate": round(grasp_success / n, 4),
        "placement_success_rate": round(placement_success / n, 4),
        "target_bin_accuracy": round(target_bin_correct / n, 4),
        "end_to_end_success_rate": round(end_to_end_success / n, 4),
        "average_confidence": round(confidence_sum / n, 4),
        "average_duration_sec": round(duration_sum / n, 4),
        "category_count": category_count,
        "target_bin_count": target_bin_count,
    }


def save_formal_task_table(path: Path, execution_log):
    """
    正式结果表：只展示垃圾大类，不展示 YOLO 小类。
    """
    fieldnames = [
        "序号",
        "任务ID",
        "垃圾大类",
        "置信度",
        "目标垃圾桶",
        "规划状态",
        "抓取成功",
        "投放成功",
        "目标桶正确",
        "端到端成功",
        "耗时_s",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(execution_log, start=1):
            category = item.get("garbage_category", item.get("category", "unknown"))
            target_bin = item.get("target_bin", "unknown")

            row = {
                "序号": idx,
                "任务ID": item.get("task_id", f"task_{idx:03d}"),
                "垃圾大类": item.get("garbage_category_cn", CATEGORY_CN.get(category, category)),
                "置信度": item.get("confidence", 0.0),
                "目标垃圾桶": item.get("target_bin_cn", BIN_CN.get(target_bin, target_bin)),
                "规划状态": item.get("planning_status", "planned"),
                "抓取成功": item.get("grasp_success", False),
                "投放成功": item.get("placement_success", False),
                "目标桶正确": item.get("target_bin_correct", False),
                "端到端成功": item.get("end_to_end_success", False),
                "耗时_s": item.get("duration_sec", 0.0),
            }

            writer.writerow(row)


def save_debug_mapping_table(path: Path, execution_log):
    """
    调试映射表：保留 YOLO 小类，用于说明小类到四类垃圾的映射。
    报告正文一般不用这张表，调试报告可以用。
    """
    fieldnames = [
        "序号",
        "任务ID",
        "YOLO小类",
        "垃圾大类",
        "置信度",
        "目标垃圾桶",
        "动作编号",
        "端到端成功",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(execution_log, start=1):
            category = item.get("garbage_category", item.get("category", "unknown"))
            target_bin = item.get("target_bin", "unknown")

            row = {
                "序号": idx,
                "任务ID": item.get("task_id", f"task_{idx:03d}"),
                "YOLO小类": item.get("raw_class_name", item.get("class_name", "unknown")),
                "垃圾大类": item.get("garbage_category_cn", CATEGORY_CN.get(category, category)),
                "置信度": item.get("confidence", 0.0),
                "目标垃圾桶": item.get("target_bin_cn", BIN_CN.get(target_bin, target_bin)),
                "动作编号": item.get("action_code", ""),
                "端到端成功": item.get("end_to_end_success", False),
            }

            writer.writerow(row)


def plot_metric_bar(path: Path, metrics):
    metric_names = [
        "valid_planning_rate",
        "grasp_success_rate",
        "placement_success_rate",
        "target_bin_accuracy",
        "end_to_end_success_rate",
    ]

    metric_labels = [
        "Planning",
        "Grasp",
        "Placement",
        "Target Bin",
        "End-to-End",
    ]

    values = [metrics.get(name, 0.0) for name in metric_names]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(metric_labels, values)

    plt.ylim(0, 1.1)
    plt.ylabel("Rate")
    plt.title("TrashBot Task Execution Metrics")

    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        plt.text(x, y + 0.02, f"{value:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_category_count(path: Path, metrics):
    category_count = metrics.get("category_count", {})

    labels = []
    values = []

    order = ["hazardous", "kitchen", "recyclable", "other", "unknown"]

    for key in order:
        if key in category_count:
            labels.append(CATEGORY_CN.get(key, key))
            values.append(category_count[key])

    for key, value in category_count.items():
        if key not in order:
            labels.append(CATEGORY_CN.get(key, key))
            values.append(value)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)

    plt.ylabel("Count")
    plt.title("Detected Tasks by Garbage Category")

    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        plt.text(x, y + 0.05, str(value), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def build_markdown_summary(metrics):
    lines = []
    lines.append("# TrashBot 实验结果汇总")
    lines.append("")
    lines.append("## 1. 任务规模")
    lines.append("")
    lines.append(f"- 任务数量：{metrics['num_tasks']}")
    lines.append(f"- 平均识别置信度：{metrics['average_confidence']:.4f}")
    lines.append(f"- 平均任务耗时：{metrics['average_duration_sec']:.2f} s")
    lines.append("")
    lines.append("## 2. 闭环指标")
    lines.append("")
    lines.append(f"- 任务规划成功率：{metrics['valid_planning_rate']:.2%}")
    lines.append(f"- 抓取成功率：{metrics['grasp_success_rate']:.2%}")
    lines.append(f"- 投放成功率：{metrics['placement_success_rate']:.2%}")
    lines.append(f"- 目标垃圾桶匹配准确率：{metrics['target_bin_accuracy']:.2%}")
    lines.append(f"- 端到端任务成功率：{metrics['end_to_end_success_rate']:.2%}")
    lines.append("")
    lines.append("## 3. 各类垃圾数量")
    lines.append("")

    for category, count in metrics.get("category_count", {}).items():
        lines.append(f"- {CATEGORY_CN.get(category, category)}：{count}")

    lines.append("")
    lines.append("## 4. 说明")
    lines.append("")
    lines.append(
        "系统底层保留 YOLO11 实例分割模型输出的小类标签，用于调试和结果追溯；"
        "在任务规划和展示阶段，将小类统一映射为可回收垃圾、厨余垃圾、有害垃圾和其他垃圾四类，"
        "并根据垃圾大类选择目标垃圾桶。Isaac Sim 中的机器人执行为逻辑抓取与投放动画，"
        "用于验证感知、规划、控制和评估模块的闭环流程。"
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(input_path)
    execution_log = extract_execution_log(data)
    metrics = compute_final_metrics(execution_log)

    metrics_json = output_dir / "final_metrics.json"
    formal_csv = output_dir / "final_task_table.csv"
    debug_csv = output_dir / "final_debug_mapping_table.csv"
    metric_bar_png = output_dir / "final_metrics_bar_chart.png"
    category_png = output_dir / "final_category_count_chart.png"
    summary_md = output_dir / "final_result_summary.md"

    save_json(metrics_json, metrics)
    save_formal_task_table(formal_csv, execution_log)
    save_debug_mapping_table(debug_csv, execution_log)
    plot_metric_bar(metric_bar_png, metrics)
    plot_category_count(category_png, metrics)

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(build_markdown_summary(metrics))

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[SAVED] {metrics_json}")
    print(f"[SAVED] {formal_csv}")
    print(f"[SAVED] {debug_csv}")
    print(f"[SAVED] {metric_bar_png}")
    print(f"[SAVED] {category_png}")
    print(f"[SAVED] {summary_md}")


if __name__ == "__main__":
    main()