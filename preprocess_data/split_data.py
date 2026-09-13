import argparse
import json
import random
from pathlib import Path


def split_jsonl(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """将 .jsonl 文件按比例划分为 train 集与 dev 集。

    Args:
        input_path: 输入的 .jsonl 文件路径。
        output_dir: 输出目录，train.jsonl 与 dev.jsonl 将写入该目录。
        train_ratio: 训练集占比，取值范围 (0, 1)。
        seed: 随机种子，保证划分可复现。
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio 必须位于 (0, 1) 区间内")

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(records)

    split_idx = int(len(records) * train_ratio)
    train_records = records[:split_idx]
    dev_records = records[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with dev_path.open("w", encoding="utf-8") as f:
        for rec in dev_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"共读取 {len(records)} 条数据")
    print(f"训练集 {len(train_records)} 条 -> {train_path}")
    print(f"开发集 {len(dev_records)} 条 -> {dev_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 .jsonl 文件按比例划分为 train 集与 dev 集。"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="输入 .jsonl 文件路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="输出目录，默认当前目录。",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集占比，默认 0.8。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42。",
    )
    args = parser.parse_args()

    split_jsonl(args.input, args.output_dir, args.train_ratio, args.seed)


if __name__ == "__main__":
    main()