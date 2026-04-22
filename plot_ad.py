#!/usr/bin/env python3
import sys

def load_ad_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('[Info]'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                ad_value = int(parts[1])
                index = int(parts[2])
                data.append((index, ad_value))
    return data

def polynomial_fit(x_data, y_data, degree=3):
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

        c = solve_linear(A, B)
        return (c[2], c[1], c[0])

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

def calculate_errors(data):
    errors = []
    for da_set, ad_meas in data:
        if da_set == 0:
            continue
        error = (ad_meas - da_set) / da_set * 100
        errors.append((da_set, ad_meas, error))
    return errors

def polynomial_compensate(x, a2, a1, a0):
    return a2 * x * x + a1 * x + a0

def polynomial_compensate_3(x, a3, a2, a1, a0):
    return a3 * x * x * x + a2 * x * x + a1 * x + a0

def polynomial_analysis_3(data, a3, a2, a1, a0):
    x_data = [d[1] for d in data]
    compensated = []
    for i in range(len(data)):
        da_set = data[i][0]
        ad_meas = data[i][1]
        ad_corr = polynomial_compensate_3(ad_meas, a3, a2, a1, a0)
        compensated.append((da_set, ad_meas, ad_corr))

    errors = []
    for da_set, ad_meas, ad_corr in compensated:
        if da_set > 0:
            error = (ad_corr - da_set) / da_set * 100
            errors.append(error)

    max_err = max(errors)
    min_err = min(errors)
    avg_err = sum(errors) / len(errors)

    print("\n" + "=" * 60)
    print("│        Polynomial Compensation (y=ax³+bx²+cx+d)             │")
    print("=" * 60)
    print(f"│  Coefficients:                                            │")
    print(f"│    a = {a3:>15.10e}                             │")
    print(f"│    b = {a2:>15.10e}                             │")
    print(f"│    c = {a1:>15.10e}                             │")
    print(f"│    d = {a0:>15.10e}                             │")
    print("=" * 60)
    print(f"│  Max Error:  {max_err:>10.4f}%                            │")
    print(f"│  Min Error:  {min_err:>10.4f}%                            │")
    print(f"│  Avg Error:  {avg_err:>10.4f}%                            │")
    print(f"│  Error Range: {max_err - min_err:.4f}%                         │")
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

def compare_results_3(data, errors, a3, a2, a1, a0):
    print("\n" + "=" * 70)
    print("│              Error Comparison (Raw vs Degree-3 Fit)           │")
    print("=" * 70)
    print(f"│ {'DA_Set':>8} │ {'AD_Raw':>8} │ {'Raw_Err%':>10} │ {'AD_Corr':>8} │ {'Corr_Err%':>10} │")
    print("-" * 70)

    sample_points = [0, len(data)//6, len(data)//3, len(data)//2, 2*len(data)//3, 5*len(data)//6, len(data)-1]
    sample_points = [i for i in sample_points if i < len(data)]

    for i in sample_points:
        da_set, ad_meas = data[i]
        if da_set == 0:
            continue
        raw_err = (ad_meas - da_set) / da_set * 100
        ad_corr = polynomial_compensate_3(ad_meas, a3, a2, a1, a0)
        corr_err = (ad_corr - da_set) / da_set * 100
        print(f"│ {da_set:>8} │ {ad_meas:>8} │ {raw_err:>10.4f} │ {ad_corr:>8.0f} │ {corr_err:>10.4f} │")

    print("=" * 70)

def analyze_errors(data, errors):
    error_vals = [e[2] for e in errors]
    max_error = max(error_vals)
    min_error = min(error_vals)
    avg_error = sum(error_vals) / len(error_vals)

    max_idx = error_vals.index(max_error)
    min_idx = error_vals.index(min_error)

    return {
        'max': (errors[max_idx][0], errors[max_idx][1], max_error),
        'min': (errors[min_idx][0], errors[min_idx][1], min_error),
        'avg': avg_error,
        'all': errors
    }

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
    print(f"│  AD Value Chart (Range: {min_v:.0f} - {max_v:.0f}){' ' * (width - 35)}│")
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
    print(f"│  Slope (k):     {slope:>12.8f}     │")
    print(f"│  Intercept (b):{intercept:>12.3f}          │")
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

    print(f"\n  Formula: AD = {slope:.8f} × Index + {intercept:.3f}")
    print(f"  at Index={data[5][0]}: Measured={data[5][1]:.0f}, Predicted={slope*data[5][0]+intercept:.0f}")

def polynomial_analysis(data, a, b, c):
    x_data = [d[0] for d in data]
    y_data = [d[1] for d in data]

    compensated = []
    for i in range(len(data)):
        da_set = x_data[i]
        ad_meas = y_data[i]
        ad_corr = polynomial_compensate(ad_meas, a, b, c)
        compensated.append((da_set, ad_meas, ad_corr))

    errors = []
    for da_set, ad_meas, ad_corr in compensated:
        if da_set > 0:
            error = (ad_corr - da_set) / da_set * 100
            errors.append(error)

    max_err = max(errors)
    min_err = min(errors)
    avg_err = sum(errors) / len(errors)

    print("\n" + "=" * 60)
    print("│        Polynomial Compensation Analysis (y=ax²+bx+c)        │")
    print("=" * 60)
    print(f"│  Coefficients:                                            │")
    print(f"│    a = {a:>15.10e}                             │")
    print(f"│    b = {b:>15.10e}                             │")
    print(f"│    c = {c:>15.10e}                             │")
    print("=" * 60)
    print(f"│  Compensation Formula:                                     │")
    print(f"│    DA_corr = a × AD² + b × AD + c                       │")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("│             Compensation Error Analysis              │")
    print("=" * 60)
    print(f"│  Max Error:  {max_err:>10.4f}%                            │")
    print(f"│  Min Error:  {min_err:>10.4f}%                            │")
    print(f"│  Avg Error:  {avg_err:>10.4f}%                            │")
    print(f"│  Error Range: {max_err - min_err:.4f}%                         │")
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

def compare_results(data, errors_raw, a, b, c):
    print("\n" + "=" * 70)
    print("│              Error Comparison (Raw vs Compensation)              │")
    print("=" * 70)
    print(f"│ {'DA_Set':>8} │ {'AD_Raw':>8} │ {'Raw_Err%':>10} │ {'AD_Corr':>8} │ {'Corr_Err%':>10} │")
    print("-" * 70)

    sample_points = [0, len(data)//6, len(data)//3, len(data)//2, 2*len(data)//3, 5*len(data)//6, len(data)-1]
    sample_points = [i for i in sample_points if i < len(data)]

    for i in sample_points:
        da_set, ad_meas = data[i]
        if da_set == 0:
            continue
        raw_err = (ad_meas - da_set) / da_set * 100
        ad_corr = polynomial_compensate(ad_meas, a, b, c)
        corr_err = (ad_corr - da_set) / da_set * 100
        print(f"│ {da_set:>8} │ {ad_meas:>8} │ {raw_err:>10.4f} │ {ad_corr:>8.0f} │ {corr_err:>10.4f} │")

    print("=" * 70)

def main():
    filepath = 'ad.txt'
    data = load_ad_data(filepath)
    if not data:
        print(f"No data loaded from {filepath}")
        sys.exit(1)

    data = sorted(data, key=lambda x: x[0])

    print(f"Loaded {len(data)} data points")

    errors = calculate_errors(data)
    err_stats = analyze_errors(data, errors)

    print("\n" + "=" * 50)
    print("│            Raw Error Analysis                    │")
    print("=" * 50)
    print(f"│  Max Error:  {err_stats['max'][2]:>10.4f}%  @DA={err_stats['max'][0]:>5}  │")
    print(f"│  Min Error:  {err_stats['min'][2]:>10.4f}%  @DA={err_stats['min'][0]:>5}  │")
    print(f"│  Avg Error:  {err_stats['avg']:>10.4f}%                   │")
    print("=" * 50)

    print("\n>>> Filtering outliers and recalculating...")

    filtered_data = []
    err_threshold = 3.0
    for da, ad in data:
        if da > 500:
            err = abs((ad - da) / da * 100)
            if err < err_threshold:
                filtered_data.append((da, ad))

    print(f"Filtered {len(data) - len(filtered_data)} outliers, {len(filtered_data)} points for fit")

    x_data = [d[1] for d in filtered_data]
    y_data = [d[0] for d in filtered_data]

    x_all = [d[1] for d in data]
    y_all = [d[0] for d in data]

    print("\n" + "=" * 60)
    print("│  Segmented Analysis (Different compensation for different ranges)        │")
    print("=" * 60)

    thresholds = [2000, 5000, 10000, 30000]
    for thresh in thresholds:
        low_data = [(d[0], d[1]) for d in filtered_data if d[0] <= thresh]
        high_data = [(d[0], d[1]) for d in filtered_data if d[0] > thresh]

        if len(low_data) >= 3:
            x_low = [d[1] for d in low_data]
            y_low = [d[0] for d in low_data]
            a2_l, a1_l, a0_l = polynomial_fit(x_low, y_low, degree=2)

            max_err = 0
            for da, ad in data:
                if da > 0:
                    da_corr = polynomial_compensate(ad, a2_l, a1_l, a0_l)
                    err = abs((da_corr - da) / da * 100)
                    if err > max_err:
                        max_err = err

            print(f"│  Range 0-{thresh:5d}: a={a2_l:.2e}, b={a1_l:.4f}, c={a0_l:.2f} │")
            print(f"│               MaxErr: {max_err:.4f}%                         │")

    print("=" * 60)

    print("\n>>> Recommended: Use lookup table for precision...")
    print(">>> Generating LUT (Lookup Table) for 0-65535...")

    x_full = [d[1] for d in data]
    y_full = [d[0] for d in data]
    a2_f, a1_f, a0_f = polynomial_fit(x_full, y_full, degree=2)

    lut_errors = []
    for da, ad in data:
        if da > 500:
            da_corr = polynomial_compensate(ad, a2_f, a1_f, a0_f)
            err = (da_corr - da) / da * 100
            lut_errors.append((da, ad, da_corr, err))

    max_err_lut = max(abs(e[3]) for e in lut_errors)
    avg_err_lut = sum(e[3] for e in lut_errors) / len(lut_errors)

    print("\n" + "=" * 60)
    print("│           Polynomial Fit Results Summary                        │")
    print("=" * 60)
    print(f"│  Formula: DA_corr = a×AD² + b×AD + c                       │")
    print(f"│    a = {a2_f:.10e}                             │")
    print(f"│    b = {a1_f:.10f}                             │")
    print(f"│    c = {a0_f:.10f}                             │")
    print("=" * 60)
    print(f"│  Max Error:  {max_err_lut:>10.4f}%                            │")
    print(f"│  Avg Error:  {avg_err_lut:>10.4f}%                            │")
    print("=" * 60)

    if max_err_lut <= 0.01:
        print("│  Result: ★★★★★ Excellent (<0.01%)                 │")
    elif max_err_lut <= 0.05:
        print("│  Result: ★★★★ Great (<0.05%)                     │")
    elif max_err_lut <= 0.1:
        print("│  Result: ★★★ Good (<0.1%)                       │")
    elif max_err_lut <= 0.5:
        print("│  Result: ★★ Acceptable (<0.5%)                 │")
    else:
        print("│  Result: ✗ Needs Improvement                    │")
    print("=" * 60)

    print("\n>>> Sample Compensated Values:")
    for da, ad, dac, err in lut_errors[::100]:
        print(f"  DA_Set={da:5d}: AD={ad:5d} -> DA_corr={dac:7.1f}, err={err:+.4f}%")

if __name__ == '__main__':
    main()