#!/usr/bin/env python3
import os
import re
import math
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def transform_channel(ch_name):
    """
    Convert a channel name from 'adc_bN_chMM' to 'NMM'.
    If MM is a single digit, prepend '0' so it becomes 'N0M'.

    Examples:
        'adc_b1_ch2'  -> '102'
        'adc_b1_ch01' -> '101'
        'adc_b12_ch9' -> '1209'
    """
    if ch_name.startswith("adc_b"):
        ch_name = ch_name[len("adc_b"):]
    if "_ch" in ch_name:
        parts = ch_name.split("_ch")
        if len(parts) == 2:
            part1 = parts[0]
            try:
                part2_int = int(parts[1])
                if part2_int < 10:
                    part2 = f"0{part2_int}"
                else:
                    part2 = str(part2_int)
                return part1 + part2
            except ValueError:
                return parts[0] + parts[1]
    return ch_name

def create_plot(pivoted, output_file, title_suffix=""):
    """
    Create a grid of subplots (8 columns per row) for each channel.
    X-axis tick labels are rotated 90 degrees and y-axis is fixed from 0.5 to 1.5.
    """
    channels = pivoted.columns
    n_channels = len(channels)
    n_cols = 8
    n_rows = math.ceil(n_channels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), sharex=False)
    axes = axes.flatten()

    for i, ch in enumerate(channels):
        ax = axes[i]
        ax.plot(pivoted.index, pivoted[ch], marker='o', linestyle='-')
        ax.set_title(ch)
        ax.grid(True)
        ax.set_ylim(0.5, 2.0)
        # Rotate x-axis labels 90 degrees
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
            tick.set_ha('center')
    
    # Hide any unused subplots
    for j in range(n_channels, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Calibration Monitor" + title_suffix, y=0.98, fontsize=16)
    plt.savefig(output_file, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")
    plt.close(fig)

def main():
    # 두 개의 디스크 경로를 리스트로 지정합니다.
    DATA_DIRS = [
        '/media/disk_k/30t-DATA/csv/phase0/non-validated/',
        '/media/disk_l/30t-DATA/csv/phase0/non-validated/',
        '/media/disk_l/30t-DATA/csv/phase1/non-validated/'
    ]
    OUTPUT_DIR = 'diagnostics/30t'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # bnl30t_spe_fit_results_YYMMDD.csv 또는 ..._adjusted.csv 패턴에 맞는 파일을 찾습니다.
    pattern = re.compile(r"bnl30t_spe_fit_results_(\d{6})(?:_adjusted)?\.csv$")
    #pattern = re.compile(r"bnl30t_injection_spe_fit_(\d{6})_1(?:_adjusted)?\.csv$")
    
    # 모든 디렉토리에서 파일을 읽어 레코드로 저장합니다.
    records = []
    for DATA_DIR in DATA_DIRS:
        if not os.path.exists(DATA_DIR):
            print(f"Directory not found: {DATA_DIR}")
            continue

        all_files = sorted(
            f for f in os.listdir(DATA_DIR)
            if pattern.match(f) and os.path.isfile(os.path.join(DATA_DIR, f))
        )
        for filename in all_files:
            match = pattern.match(filename)
            if not match:
                continue
            
            date_str = match.group(1)
            dt = datetime.strptime(date_str, "%y%m%d")
            
            csv_path = os.path.join(DATA_DIR, filename)
            try:
                df = pd.read_csv(csv_path, usecols=["ch_name", "spe_mean"])
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue

            for _, row in df.iterrows():
                channel = transform_channel(row["ch_name"])
                records.append((dt, channel, row["spe_mean"]))
    
    if not records:
        print("No valid CSV data found. Exiting.")
        return

    # DataFrame을 만들고 피벗 테이블 형식으로 변환합니다.
    df_all = pd.DataFrame(records, columns=["date", "channel", "spe_mean"])
    pivoted = df_all.pivot_table(index="date", columns="channel", values="spe_mean")
    pivoted.sort_index(inplace=True)

    # 전체 데이터를 플로팅합니다.
    output_file_all = os.path.join(OUTPUT_DIR, 'Datamonitor.png')
    create_plot(pivoted, output_file_all)

    # 3월 10일 이후 데이터만 필터링하여 플로팅합니다.
    filter_date = datetime(2025, 3, 10)
    pivoted_filtered = pivoted[pivoted.index >= filter_date]
    if pivoted_filtered.empty:
        print("No data from March 10 onward.")
    else:
        output_file_filtered = os.path.join(OUTPUT_DIR, 'Datamonitor_fromMar10.png')
        create_plot(pivoted_filtered, output_file_filtered, title_suffix=" (From March 10)")

if __name__ == "__main__":
    main()
