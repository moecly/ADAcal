#!/usr/bin/env python3
import sys

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                index = int(parts[0])
                voltage = float(parts[1])
                data.append((index, voltage))
    return data

def linear_regression(data):
    n = len(data)
    if n < 2:
        return None, None, None

    sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = 0
    for x, y in data:
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x
        sum_y2 += y * y

    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return None, None, None

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in data)
    ss_tot = sum_y2 - sum_y * sum_y / n
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return slope, intercept, r_squared

def draw_unicode_chart(data, width=80, height=30):
    if not data:
        print("No data to display")
        return

    voltages = [v for _, v in data]
    min_v = min(voltages)
    max_v = max(voltages)
    v_range = max_v - min_v
    if v_range == 0:
        v_range = 1

    indices = [i for i, _ in data]
    min_i = min(indices)
    max_i = max(indices)
    i_range = max_i - min_i
    if i_range == 0:
        i_range = 1

    step = width // 10

    print(f"┌{'─' * (width + 2)}┐")
    print(f"│{' ' * (width + 2)}│")
    print(f"│  Voltage Chart (Range: {min_v:.3f}V - {max_v:.3f}V){' ' * (width - 50)}│")
    print(f"│{' ' * (width + 2)}│")
    print(f"│{'─' * step}┼{'-' * (width - step)}{'│'}")

    grid = [[' ' for _ in range(width)] for _ in range(height)]

    for idx, volt in data:
        x = int((idx - min_i) / i_range * (width - 1))
        y = int((volt - min_v) / v_range * (height - 1))
        y = height - 1 - y
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = '●'

    for row in grid:
        print("│" + ''.join(row) + "│")

    print(f"└{'─' * width}┘")
    print(f"   Index: {min_i} -> {max_i}")

def analyze_linearity(data, slope, intercept, r_squared):
    if slope is None:
        print("\n⚠ 无法进行线性分析（数据点不足）")
        return

    print("\n" + "=" * 50)
    print("│         Linear Regression Analysis          │")
    print("=" * 50)
    print(f"│  Slope (k):     {slope:>10.6f} V/index      │")
    print(f"│  Intercept (b):{intercept:>10.6f} V          │")
    print(f"│  R² value:     {r_squared:>10.6f}             │")
    print("=" * 50)

    if r_squared >= 0.99:
        print("│  Result: ★★★ Highly Linear             │")
    elif r_squared >= 0.95:
        print("│  Result: ★★  Good Linear Fit             │")
    elif r_squared >= 0.80:
        print("│  Result: ★   Moderate Linear            │")
    else:
        print("│  Result: ✗ Poor Linear Fit              │")
    print("=" * 50)

    print(f"\n  Formula: V = {slope:.6f} × Index + {intercept:.6f}")
    print(f"  at Index={data[4][0]}: Measured={data[4][1]:.3f}V, Predicted={slope*data[4][0]+intercept:.3f}V")

def main():
    filepath = 'da.txt'
    data = load_data(filepath)
    if not data:
        print(f"No data loaded from {filepath}")
        sys.exit(1)

    slope, intercept, r_squared = linear_regression(data)
    draw_unicode_chart(data)
    analyze_linearity(data, slope, intercept, r_squared)

if __name__ == '__main__':
    main()