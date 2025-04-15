import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

def transform_label(label):
    """
    Transform a label from 'adc_bN_chMM' format to 'NMM'.
    If MM is a single digit, prepend '0' so it becomes 'N0M'.

    Examples:
        'adc_b1_ch2'  -> '102'
        'adc_b1_ch01' -> '101'
    """
    # Remove the "adc_b" prefix if present
    if label.startswith("adc_b"):
        label = label[len("adc_b"):]
    # Split on "_ch" and handle the parts
    if "_ch" in label:
        parts = label.split("_ch")
        if len(parts) == 2:
            # The first part can be taken as is (e.g., '1', '12', etc.)
            part1 = parts[0]
            # The second part may need a leading zero if it's a single digit
            try:
                part2_int = int(parts[1])  # convert to int
                if part2_int < 10:
                    part2 = f"0{part2_int}"
                else:
                    part2 = str(part2_int)
                return part1 + part2
            except ValueError:
                # If it can't be converted to int, just return concatenation
                return parts[0] + parts[1]
    return label

def main(date1, date2):
    DATA_DIR = '/media/disk_k/30t-DATA/csv/phase0/non-validated/'
    
    # Build file paths
    file1 = os.path.join(DATA_DIR, f"bnl30t_spe_fit_results_{date1}.csv")
    file2 = os.path.join(DATA_DIR, f"bnl30t_spe_fit_results_{date2}.csv")
    
    # Check if both files exist
    if not os.path.isfile(file1):
        print(f"File not found: {file1}")
        return
    if not os.path.isfile(file2):
        print(f"File not found: {file2}")
        return

    try:
        # Read only 'ch_name' and 'spe_mean' columns from each CSV file
        df1 = pd.read_csv(file1, usecols=["ch_name", "spe_mean"])
        df2 = pd.read_csv(file2, usecols=["ch_name", "spe_mean"])
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Merge the two dataframes on 'ch_name'
    merged = pd.merge(df1, df2, on="ch_name", suffixes=('_1', '_2'))

    # Calculate the difference in spe_mean (first minus second)
    merged['spe_diff'] = merged['spe_mean_1'] - merged['spe_mean_2']

    # Generate transformed labels for the x-axis
    merged['ch_label'] = merged['ch_name'].apply(transform_label)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(merged['ch_name'], merged['spe_diff'], marker='o', linestyle='-')
    plt.xlabel("Channel")
    plt.ylabel("spe_mean Difference")
    plt.title(f"spe_mean Difference between {date1} and {date2}")
    plt.grid(True)
    
    # Replace x-axis ticks with transformed labels and rotate them 90 degrees
    plt.xticks(merged['ch_name'], merged['ch_label'], rotation=90, ha='center')
    
    # Adjust layout to prevent label overlap
    plt.tight_layout()

    # Save plot
    output_dir = 'diagnostics/30t'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'DataCompare_{date1}&{date2}.png')
    plt.savefig(output_file)
    print(f"Plot saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python DataComparism.py YYMMDD YYMMDD")
        sys.exit(1)
    
    date1 = sys.argv[1]
    date2 = sys.argv[2]
    main(date1, date2)
