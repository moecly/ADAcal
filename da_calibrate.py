#!/usr/bin/env python3
import sys
import argparse
import math

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

    if degree == 3:
        sum_x = sum(x)
        sum_x2 = sum(xi * xi for xi in x)
        sum_x3 = sum(xi * xi * xi for xi in x)
        sum_x4 = sum(xi * xi * xi * xi for xi in x)
        sum_x5 = sum(xi * xi * xi * xi * xi for xi in x)
        sum_x6 = sum(xi * xi * xi * xi * xi * xi for xi in x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2y = sum(x[i] * x[i] * y[i] for i in range(n))
        sum_x3y = sum(x[i] * x[i] * x[i] * y[i] for i in range(n))

        A = [
            [n, sum_x, sum_x2, sum_x3],
            [sum_x, sum_x2, sum_x3, sum_x4],
            [sum_x2, sum_x3, sum_x4, sum_x5],
            [sum_x3, sum_x4, sum_x5, sum_x6]
        ]
        B = [sum_y, sum_xy, sum_x2y, sum_x3y]

        coeff = solve_linear(A, B)
        return (coeff[3], coeff[2], coeff[1], coeff[0])

    elif degree == 2:
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

def calculate_aic_bic(x_data, y_data, degree):
    n = len(x_data)
    if degree == 1:
        k = 2
    elif degree == 2:
        k = 3
    elif degree == 3:
        k = 4
    else:
        k = degree + 1

    if degree == 1:
        b, a0 = polynomial_fit(x_data, y_data, degree=1)
        y_pred = [b * x + a0 for x in x_data]
    elif degree == 2:
        a2, a1, a0 = polynomial_fit(x_data, y_data, degree=2)
        y_pred = [a2 * x * x + a1 * x + a0 for x in x_data]
    elif degree == 3:
        a3, a2, a1, a0 = polynomial_fit(x_data, y_data, degree=3)
        y_pred = [a3 * x * x * x + a2 * x * x + a1 * x + a0 for x in x_data]
    else:
        return None, None

    rss = sum((y_data[i] - y_pred[i]) ** 2 for i in range(n))
    if rss <= 0:
        rss = 1e-10

    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)

    return aic, bic

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

def polynomial_compensate_v1(x, b, a0):
    return b * x + a0

def polynomial_analysis_v1(data, b, a0):
    compensated = []
    for idx, voltMeas in data:
        voltCorr = polynomial_compensate_v1(idx, b, a0)
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
    print("│        Polynomial Compensation (y=bx+c)                       │")
    print("=" * 60)
    print(f"│  Coefficient:                                         │")
    print(f"│    b = {b:>15.10e}                             │")
    print(f"│    c = {a0:>15.10e}                             │")
    print("=" * 60)
    print(f"│  Max Error:  {max_err:>10.4f}%                            │")
    print(f"│  Min Error:  {min_err:>10.4f}%                            │")
    print(f"│  Avg Error:  {avg_err:>10.4f}%                            │")
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

def compare_results_v1(data, errors_raw, b, a0):
    print("\n" + "=" * 70)
    print("│              Error Comparison (Raw vs Degree-1 Fit)            │")
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
        voltCorr = polynomial_compensate_v1(idx, b, a0)
        corr_err = (voltCorr - theory) / theory * 100
        print(f"│ {idx:>8} │ {voltMeas:>8.3f} │ {raw_err:>10.4f} │ {voltCorr:>8.3f} │ {corr_err:>10.4f} │")

    print("=" * 70)

def polynomial_compensate_v3(x, a3, a2, a1, a0):
    return a3 * x * x * x + a2 * x * x + a1 * x + a0

def polynomial_analysis_v3(data, a3, a2, a1, a0):
    compensated = []
    for idx, voltMeas in data:
        voltCorr = polynomial_compensate_v3(idx, a3, a2, a1, a0)
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
    print("│     Polynomial Compensation (y=ax³+bx²+cx+d)                     │")
    print("=" * 60)
    print(f"│  Coefficients:                                        │")
    print(f"│    a = {a3:>15.10e}                             │")
    print(f"│    b = {a2:>15.10e}                             │")
    print(f"│    c = {a1:>15.10e}                             │")
    print(f"│    d = {a0:>15.10e}                             │")
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

def compare_results_v3(data, errors_raw, a3, a2, a1, a0):
    print("\n" + "=" * 70)
    print("│              Error Comparison (Raw vs Degree-3 Fit)           │")
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
        voltCorr = polynomial_compensate_v3(idx, a3, a2, a1, a0)
        corr_err = (voltCorr - theory) / theory * 100
        print(f"│ {idx:>8} │ {voltMeas:>8.3f} │ {raw_err:>10.4f} │ {voltCorr:>8.3f} │ {corr_err:>10.4f} │")

    print("=" * 70)

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

def generate_c_code(name, degree, coefficients, max_error, min_val=0, max_val=65535):
    if degree == 1:
        b, a0 = coefficients
        func_body = f"    if (x == 0) return 0.0f;\n    float result = {b:.10e} * (float)x + {a0:.10e};"
    elif degree == 2:
        a2, a1, a0 = coefficients
        func_body = f"    if (x == 0) return 0.0f;\n    float xf = (float)x;\n    float result = {a2:.10e} * xf * xf + {a1:.10e} * xf + {a0:.10e};"
    elif degree == 3:
        a3, a2, a1, a0 = coefficients
        func_body = f"    if (x == 0) return 0.0f;\n    float xf = (float)x;\n    float result = {a3:.10e} * xf * xf * xf + {a2:.10e} * xf * xf + {a1:.10e} * xf + {a0:.10e};"

    saturation = f"    if (result > {max_val:.1f}f) result = {max_val:.1f}f;\n    if (result < {min_val:.1f}f) result = {min_val:.1f}f;\n    return result;"

    guard_name = name.upper().replace(" ", "_").replace("-", "_")

    print(f"\n// =======================================================================")
    print(f"// {name} Compensation Function (degree={degree}, Max Error: {max_error:.4f}%)")
    print(f"// Input:  DA theoretical value (0-65535)")
    print(f"// Output: Compensated DA value (float)")
    print(f"// =======================================================================")
    print(f"")
    print(f"#ifndef {guard_name}_H")
    print(f"#define {guard_name}_H")
    print(f"")
    print(f"#ifdef __cplusplus")
    print(f"extern \"C\" {{")
    print(f"#endif")
    print(f"")
    print(f"// Compensate DA theoretical value to actual output")
    print(f"// Input:  uint16_t da_theoretical - Theoretical DA value")
    print(f"// Output: float - Compensated DA value")
    print(f"extern float {name.replace(' ', '_').replace('-', '_')}_compensate(uint16_t x);")
    print(f"")
    print(f"#ifdef __cplusplus")
    print(f"}}")
    print(f"#endif")
    print(f"")
    print(f"#endif // {guard_name}_H")
    print(f"")
    print(f"// =======================================================================")
    print(f"// Implementation")
    print(f"// =======================================================================")
    print(f"")
    print(f"#ifdef {guard_name}_IMPLEMENTATION")
    print(f"")
    print(f"float {name.replace(' ', '_').replace('-', '_')}_compensate(uint16_t x) {{")
    print(f"{func_body}")
    print(f"{saturation}")
    print(f"}}")
    print(f"")
    print(f"#endif // {guard_name}_IMPLEMENTATION")
    print(f"")

def main():
    parser = argparse.ArgumentParser(description='DA Voltage Analysis Tool')
    parser.add_argument('-d', '--degree', type=int, default=None, help='Polynomial degree (1, 2, or 3). Auto-select if not specified.')
    parser.add_argument('--auto', action='store_true', help='Auto-select best polynomial degree using AIC/BIC')
    parser.add_argument('--show-all', action='store_true', help='Show results for all polynomial degrees')
    parser.add_argument('--output-c', action='store_true', help='Output C code for integration')
    args = parser.parse_args()

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

    x_data = [d[0] for d in data]
    y_data = [d[1] for d in data]

    if args.show_all or args.auto:
        print("\n" + "=" * 60)
        print("│       Comparing Polynomial Degrees (1/2/3)                        │")
        print("=" * 60)
        print(f"│ {'Degree':^8} │ {'AIC':^14} │ {'BIC':^14} │ {'MaxErr%':^10} │")
        print("-" * 60)

        results = []
        for deg in [1, 2, 3]:
            try:
                aic, bic = calculate_aic_bic(x_data, y_data, deg)
                if deg == 1:
                    b, a0 = polynomial_fit(x_data, y_data, degree=1)
                    y_pred = [b * x + a0 for x in x_data]
                elif deg == 2:
                    a2, a1, a0 = polynomial_fit(x_data, y_data, degree=2)
                    y_pred = [a2 * x * x + a1 * x + a0 for x in x_data]
                else:
                    a3, a2, a1, a0 = polynomial_fit(x_data, y_data, degree=3)
                    y_pred = [a3 * x * x * x + a2 * x * x + a1 * x + a0 for x in x_data]

                curr_errors = []
                for i, (idx, volt) in enumerate(data):
                    if idx > 0:
                        theory = idx * 0.000153
                        err = abs((y_pred[i] - theory) / theory * 100)
                        curr_errors.append(err)
                max_err = max(curr_errors) if curr_errors else 0
                results.append((deg, aic, bic, max_err))
                print(f"│ {deg:^8} │ {aic:^14.4f} │ {bic:^14.4f} │ {max_err:^10.4f} │")
            except Exception as e:
                print(f"│ {deg:^8} │ Failed: {str(e)[:20]:^20} │")

        print("=" * 60)

        if args.auto:
            best_aic = min(results, key=lambda x: x[1])
            best_err = min(results, key=lambda x: x[3])
            print(f"\n>>> AIC Best: degree={best_aic[0]} (AIC={best_aic[1]:.4f})")
            print(f">>> Error Best: degree={best_err[0]} (MaxErr={best_err[3]:.4f}%)")

            if best_aic[0] == best_err[0]:
                auto_degree = best_aic[0]
                print(f">>> Auto-selected: degree={auto_degree} (both AIC and error best)")
            else:
                auto_degree = best_err[0]
                print(f">>> Auto-selected: degree={auto_degree} (lowest error wins)")

            args.degree = auto_degree

    if args.degree is None:
        args.degree = 2

    print(f"\n>>> Calculating Polynomial Fit (degree={args.degree})...")

    max_err = 0
    if args.degree == 1:
        b, a0 = polynomial_fit(x_data, y_data, degree=1)
        coefficients = (b, a0)
        polynomial_analysis_v1(data, b, a0)
        compare_results_v1(data, errors, b, a0)
        curr_errors = []
        for i, (idx, volt) in enumerate(data):
            if idx > 0:
                theory = idx * 0.000153
                pred = b * idx + a0
                err = abs((pred - theory) / theory * 100)
                curr_errors.append(err)
        max_err = max(curr_errors) if curr_errors else 0
    elif args.degree == 2:
        a2, a1, a0 = polynomial_fit(x_data, y_data, degree=2)
        coefficients = (a2, a1, a0)
        polynomial_analysis_v(data, a2, a1, a0)
        compare_results_v(data, errors, a2, a1, a0)
        curr_errors = []
        for i, (idx, volt) in enumerate(data):
            if idx > 0:
                theory = idx * 0.000153
                pred = a2 * idx * idx + a1 * idx + a0
                err = abs((pred - theory) / theory * 100)
                curr_errors.append(err)
        max_err = max(curr_errors) if curr_errors else 0
    elif args.degree == 3:
        a3, a2, a1, a0 = polynomial_fit(x_data, y_data, degree=3)
        coefficients = (a3, a2, a1, a0)
        polynomial_analysis_v3(data, a3, a2, a1, a0)
        compare_results_v3(data, errors, a3, a2, a1, a0)
        curr_errors = []
        for i, (idx, volt) in enumerate(data):
            if idx > 0:
                theory = idx * 0.000153
                pred = a3 * idx * idx * idx + a2 * idx * idx + a1 * idx + a0
                err = abs((pred - theory) / theory * 100)
                curr_errors.append(err)
        max_err = max(curr_errors) if curr_errors else 0

    if args.output_c:
        generate_c_code("DA", args.degree, coefficients, max_err)

if __name__ == '__main__':
    main()