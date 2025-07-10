#!/usr/bin/env python3
"""
Plot six action curves from a CSV.
Usage:
    python plot_actions.py /path/to/actions.csv
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    # -------- 参数检查 --------
    if len(sys.argv) < 2:
        print("Usage: python plot_actions.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        sys.exit(f"File not found: {csv_path}")

    # -------- 读取 CSV --------
    # 兼容制表符 / 逗号分隔：engine='python' 自动检测
    df = pd.read_csv(csv_path, sep=None, engine="python")

    # 取 action_1 .. action_6（过滤掉 action_0）
    cols = [f"action_{i}" for i in range(1, 7)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns in CSV: {missing}")

    # -------- 逐列画图 --------
    for col in cols:
        plt.figure()                 # 每个指标单独一张图
        plt.plot(df.index, df[col])  # 默认颜色
        plt.title(f"{col} over Time")
        plt.xlabel("Row Index")
        plt.ylabel(col)
        plt.grid(True)
        plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
