"""核心方法分析: 容量-预算协同设计 (Capacity-Budget Co-Design)

命题: 给定标注预算 k, 存在最优读出容量 c*(k), 且随预算变化:
  小预算 -> 小容量 (linear, 0† 逆转)
  大预算 -> 大容量 (30K)
给出 c*(k) 轨迹, 若 0t/200/1000/ALL 的 argmax 确实随 k 单调增大 ->
  规律成立, 可作为可操作设计规则 (给定预算选容量)。

数据: phase_diagram (3 readout × {0t,200,1000,ALL}, 3 seed mean)
输出: 表 + 轨迹 + 交叉验证 (留一个预算点, 预测最优容量)。
"""
import json
import sys
import io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
OUT = "D:/lunwen/outputs"
d = json.load(open(f"{OUT}/phase_diagram.json"))
s = d["summary"]
archs = ["linear", "4K", "30K"]
budgets = ["0t", "200", "1000", "ALL"]
kmap = {"0t": 0, "200": 200, "1000": 1000, "ALL": 14731}

print("=== 3x4 mIoU 矩阵 (3 seed mean) ===")
rows = {}
for b in budgets:
    row = []
    for a in archs:
        row.append(s[a][b]["miou"]["mean"])
        # 也收集标准差用于 argmax 区分度
    rows[b] = row
print(f"{'budget':>8s} {'linear':>8s} {'4K':>8s} {'30K':>8s}   argmax")
for b in budgets:
    r = rows[b]
    am = archs[int(np.argmax(r))]
    print(f"{kmap[b]:>8d} {r[0]:>8.4f} {r[1]:>8.4f} {r[2]:>8.4f}   {am}")

print("\n=== argmax 轨迹 (c*(k)) ===")
seq = []
for b in budgets:
    r = rows[b]
    am = np.argmax(r)
    seq.append(am)
    print(f"k={kmap[b]:>6d}: argmax={archs[am]} (mIoU={r[am]:.4f})")

monotone = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
print(f"\n单调递增 (0=linear, 1=4K, 2=30K): {seq} -> {'√ 单调' if monotone else '× 非单调'}")

# 交叉验证: 留一个预算点, 用其余点看 argmax 是否可预测
print("\n=== 留一法 (LOO) 交叉验证 ===")
for hold in range(len(budgets)):
    # 用其余 3 个预算点拟合每容量 mIoU ~ logk 线性, 预测 hold
    ks = np.array([kmap[b] for b in budgets if b != budgets[hold]], float)
    preds = []
    for a in archs:
        vs = np.array([s[a][b]["miou"]["mean"] for b in budgets if b != budgets[hold]])
        beta, b0 = np.polyfit(np.log10(np.maximum(ks, 1e-6)), vs, 1)
        pk = kmap[budgets[hold]]
        preds.append(beta * np.log10(max(pk, 1e-6)) + b0)
    # 预测 argmax
    pred_am = archs[int(np.argmax(preds))]
    true_am = archs[int(np.argmax(rows[budgets[hold]]))]
    ok = "√" if pred_am == true_am else "×"
    print(f"hold={budgets[hold]:>5s}: 预测 argmax={pred_am:>6s} (pred {np.round(preds,3)}) "
          f"vs 真={true_am:>6s} {ok}")

# 关键: 检测维度的 argmax (检测几乎 flat -> 容量预算选择主要由定位驱动)
print("\n=== 检测 AUC 矩阵 (验证: 容量选择不影响检测) ===")
for b in budgets:
    r = [s[a][b]["det"]["mean"] for a in archs]
    print(f"k={kmap[b]:>6d}: detection {np.round(r, 4)}")

with open(OUT + "/capacity_analysis.json", "w", encoding="utf-8") as f:
    json.dump({"rows": {b: [float(x) for x in rows[b]] for b in budgets},
               "argmax_seq": [int(x) for x in seq],
               "monotone": bool(monotone)}, f, indent=1)
print("\nsaved: capacity_analysis.json")
