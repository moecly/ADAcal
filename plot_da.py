#!/usr/bin/env python3
import sys

def load_da_data(filepath):
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

def polynomial_fit(x_data, y_data, degree=2):
    n = len(x_data)
    x = x_data.copy()
    y = y_data.copy()

    if degree == 2:
        sum_x = sum(x)
        sum_x2 = sum(xi * xi for xi in x)
        sum_x3 = sum(xi * xi * xi for xi in x)
        sum_x4 = sum(xi * xi * xi * xi for xi in x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2y = sum(x[i] * x[i] * y[i] for i in range(n))

        A = [
            [n, sum_x, sum_x2],
            [sum_x, sum_x2, sum_x3],
            [sum_x2, sum_x3, sum_x4]
        ]
        B = [sum_y, sum_xy, sum_x2y]

        coeff = solve_linear(A, B)
        return (coeff[2], coeff[1], coeff[0])

    elif degree == 1:
        return linear_regression_coef(x_data, y_data)

    return None

def solve_linear(A, B):
    n = len(A)
    for i in range(n):
        max_row = i
        for r in range(i + 1, n):
            if abs(A[r][i]) > abs(A[max_row][i]):
                max_row = r
        A[i], A[max_row] = A[max_row], A[i]
        B[i], B[max_row] = B[max_row], B[i]

        div = A[i][i]
        for c in range(i, n):
            A[i][c] /= div
        B[i] /= div

        for r in range(n):
            if r != i:
                factor = A[r][i]
                for c in range(i, n):
                    A[r][c] -= factor * A[i][c]
                B[r] -= factor * B[i]

    return B

def linear_regression_coef(x_data, y_data):
    n = len(x_data)
    sum_x = sum_y = sum_xy = sum_x2 = 0
    for i in range(n):
        x = x_data[i]
        y = y_data[i]
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0, 0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept

def linear_regression(data):
    n = len(data)
    if n < 2:
        return None, None, None

    x_data = [d[0] for d in data]
    y_data = [d[1] for d in data]

    slope, intercept = linear_regression_coef(x_data, y_data)

    sum_res = sum((y_data[i] - (slope * x_data[i] + intercept)) ** 2 for i in range(n))
    sum_y2 = sum(y * y for y in y_data)
    sum_y = sum(y_data)
    ss_tot = sum_y2 - sum_y * sum_y / n
    r_squared = 1 - (sum_res / ss_tot) if ss_tot != 0 else 0

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
    print(f"│  Voltage Chart (Range: {min_v:.3f}V - {max_v:.3f}V){' ' * (width - 43)}│")
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
    print(f"│  Slope (k):     {slope:>12.10f}    │")
    print(f"│  Intercept (b):{intercept:>12.6f}V         │")
    print(f"│  R² value:      {r_squared:>12.8f}             │")
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

def calculate_errors(data):
    errors = []
    for idx, volt in data:
        if idx == 0:
            continue
        theory = idx * 0.000153
        if theory > 0:
            error = (volt - theory) / theory * 100
            errors.append((idx, volt, theory, error))
    return errors

def polynomial_compensate(x, a2, a1, a0):
    return a2 * x * x + a1 * x + a0

def polynomial_analysis_v(data, a2, a1, a0):
    compensated = []
    for i in range(len(data)):
        idx = data[i][0]
        voltMeas = data[i][1]
        voltCorr = polynomial_compensate(idx, a2, a1, a0)
        compensated.append((idx, voltMeas, voltCorr))

    errors = []
    for idx, voltMeas, voltCorr in compensated:
        if idx > 0:
            theory = idx * 0.000153
            error = (voltCorr - theory) / theory * 100
            errors.append(error)

    max_err = max(errors)
    min_err = min(errors)
    avg_err = sum(errors) / len(errors)

    print("\n" + "=" * 60)
    print("│        Polynomial Compensation Analysis (y=ax²+bx+c)        │")
    print("=" * 60)
    print(f"│  Coefficients:                                            │")
    print(f"│    a = {a2:>15.10e}                             │")
    print(f"│    b = {a1:>15.10e}                             │")
    print(f"│    c = {a0:>15.10e}                             │")
    print("=" * 60)
    print(f"│  Max Error:  {max_err:>10.4f}%                            │")
    print(f"│  Min Error:  {min_err:>10.4f}%                            │")
    print(f"│  Avg Error:  {avg_err:>10.4f}%                            │")
    print(f"│  Error Range:{max_err - min_err:.4f}%                          │")
    print("=" * 60)

    if max(abs(max_err), abs(min_err)) <= 0.01:
        print("│  Result: ★★★★★ Excellent (Error < 0.01%)         │")
    elif max(abs(max_err), abs(min_err)) <= 0.05:
        print("│  Result: ★★★★ Great (Error < 0.05%)              │")
    elif max(abs(max_err), abs(min_err)) <= 0.1:
        print("│  Result: ★★★ Good (Error < 0.1%)                │")
    elif max(abs(max_err), abs(min_err)) <= 0.5:
        print("│  Result: ★★ Acceptable (Error < 0.5%)            │")
    else:
        print("│  Result: ✗ Needs Improvement                    │")
    print("=" * 60)

def compare_results_v(data, errors_raw, a2, a1, a0):
    print("\n" + "=" * 70)
    print("│              Error Comparison (Raw vs Compensation)              │")
    print("=" * 70)
    print(f"│ {'Index':>8} │ {'V_Meas':>8} │ {'Raw_Err%':>10} │ {'V_Corr':>8} │ {'Corr_Err%':>10} │")
    print("-" * 70)

    sample_indices = [i for i in range(0, len(data), len(data)//8)]

    for i in sample_indices:
        idx, voltMeas = data[i]
        if idx == 0:
            continue
        theory = idx * 0.000153
        raw_err = (voltMeas - theory) / theory * 100
        voltCorr = polynomial_compensate(idx, a2, a1, a0)
        corr_err = (voltCorr - theory) / theory * 100
        print(f"│ {idx:>8} │ {voltMeas:>8.3f} │ {raw_err:>10.4f} │ {voltCorr:>8.3f} │ {corr_err:>10.4f} │")

    print("=" * 70)

def analyze_errors(data, errors):
    error_vals = [e[3] for e in errors]
    if not error_vals:
        return {'max': (0, 0, 0, 0), 'min': (0, 0, 0, 0), 'avg': 0}

    max_error = max(error_vals)
    min_error = min(error_vals)
    avg_error = sum(error_vals) / len(error_vals)

    max_idx = error_vals.index(max_error)
    min_idx = error_vals.index(min_error)

    return {
        'max': errors[max_idx],
        'min': errors[min_idx],
        'avg': avg_error,
        'all': errors
    }

def polynomial_analysis(data, a2, a1, a0):
    compensated = []
    for i in range(len(data)):
        idx = data[i][0]
        voltMeas = data[i][1]
        voltCorr = polynomial_compensate(voltMeas, a2, a1, a0)
        compensated.append((idx, voltMeas, voltCorr))

    errors = []
    for idx, voltMeas, voltCorr in compensated:
        if idx > 0:
            theory = idx * 0.000153
            error = (voltCorr - theory) / theory * 100
            errors.append(error)

    max_err = max(errors)
    min_err = min(errors)
    avg_err = sum(errors) / len(errors)

    print("\n" + "=" * 60)
    print("│        Polynomial Compensation Analysis (y=ax²+bx+c)        │")
    print("=" * 60)
    print(f"│  Coefficients:                                            │")
    print(f"│    a = {a2:>15.10e}                             │")
    print(f"│    b = {a1:>15.10e}                             │")
    print(f"│    c = {a0:>15.10e}                             │")
    print("=" * 60)
    print(f"│  Max Error:  {max_err:>10.4f}%                            │")
    print(f"│  Min Error:  {min_err:>10.4f}%                            │")
    print(f"│  Avg Error:  {avg_err:>10.4f}%                            │")
    print(f"│  Error Range:{max_err - min_err:.4f}%                          │")
    print("=" * 60)

    if max(abs(max_err), abs(min_err)) <= 0.01:
        print("│  Result: ★★★★★ Excellent (Error < 0.01%)         │")
    elif max(abs(max_err), abs(min_err)) <= 0.05:
        print("│  Result: ★★★★ Great (Error < 0.05%)              │")
    elif max(abs(max_err), abs(min_err)) <= 0.1:
        print("│  Result: ★★★ Good (Error < 0.1%)                │")
    elif max(abs(max_err), abs(min_err)) <= 0.5:
        print("│  Result: ★★ Acceptable (Error < 0.5%)            │")
    else:
        print("│  Result: ✗ Needs Improvement                    │")
    print("=" * 60)

def compare_results(data, errors_raw, a2, a1, a0):
    print("\n" + "=" * 70)
    print("│              Error Comparison (Raw vs Compensation)              │")
    print("=" * 70)
    print(f"│ {'Index':>8} │ {'V_Meas':>8} │ {'Raw_Err%':>10} │ {'V_Corr':>8} │ {'Corr_Err%':>10} │")
    print("-" * 70)

    sample_indices = [i for i in range(0, len(data), len(data)//8)]

    for i in sample_indices:
        idx, voltMeas = data[i]
        if idx == 0:
            continue
        theory = idx * 0.000153
        raw_err = (voltMeas - theory) / theory * 100
        voltCorr = polynomial_compensate(voltMeas, a2, a1, a0)
        corr_err = (voltCorr - theory) / theory * 100
        print(f"│ {idx:>8} │ {voltMeas:>8.3f} │ {raw_err:>10.4f} │ {voltCorr:>8.3f} │ {corr_err:>10.4f} │")

    print("=" * 70)

def main():
    filepath = 'da.txt'
    data = load_da_data(filepath)
    if not data:
        print(f"No data loaded from {filepath}")
        sys.exit(1)

    data = sorted(data, key=lambda x: x[0])

    print(f"Loaded {len(data)} data points")

    slope, intercept, r_squared = linear_regression(data)
    draw_unicode_chart(data)
    analyze_linearity(data, slope, intercept, r_squared)

    errors = calculate_errors(data)
    err_stats = analyze_errors(data, errors)

    print("\n" + "=" * 50)
    print("│            Raw Error Analysis                    │")
    print("=" * 50)
    print(f"│  Max Error:  {err_stats['max'][3]:>10.4f}%  @Index={err_stats['max'][0]:>5}  │")
    print(f"│  Min Error:  {err_stats['min'][3]:>10.4f}%  @Index={err_stats['min'][0]:>5}  │")
    print(f"│  Avg Error:  {err_stats['avg']:>10.4f}%                   │")
    print("=" * 50)

    print("\n>>> Calculating Polynomial Fit...")

    x_data = [d[0] for d in data]
    y_data = [d[1] for d in data]

    a2, a1, a0 = polynomial_fit(x_data, y_data, degree=2)

    polynomial_analysis_v(data, a2, a1, a0)
    compare_results_v(data, errors, a2, a1, a0)

if __name__ == '__main__':
    main()