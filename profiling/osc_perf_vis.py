import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap

# --- Load Data ---
# User provided file path
csv_file = '/fs/ess/PAS2699/nitrogen/data/uas/2023/processing/logs_perf/aggregated_metrics_2023.csv'
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: The CSV file was not found at {csv_file}")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# --- Helper Functions for Data Cleaning ---
def convert_wall_clock_to_seconds(time_str):
    """Converts HH:MM:SS string to total seconds."""
    if pd.isna(time_str) or time_str in ['N/A', 'N/A_empty', 'N/A_parse_error', 'Error_seff', 'Error_script']:
        return np.nan
    try:
        parts = list(map(int, str(time_str).split(':')))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2: # MM:SS
            return parts[0] * 60 + parts[1]
        elif len(parts) == 1: # SS
            return parts[0]
    except ValueError:
        return np.nan
    return np.nan

def convert_memory_to_gb(mem_str):
    """Converts memory string (e.g., "30.58 GB", "90.40 MB") to GB."""
    if pd.isna(mem_str) or not isinstance(mem_str, str) or mem_str in ['N/A', 'N/A_empty', 'N/A_parse_error', 'Error_seff', 'Error_script']:
        return np.nan
    mem_str_lower = mem_str.lower()
    val_str = ''.join(filter(lambda x: x.isdigit() or x == '.', mem_str_lower.split()[0]))
    try:
        val = float(val_str)
        if 'gb' in mem_str_lower:
            return val
        elif 'mb' in mem_str_lower:
            return val / 1024
        elif 'kb' in mem_str_lower:
            return val / (1024 * 1024)
        return val # Assuming GB if no unit and parsable
    except ValueError:
        return np.nan

def convert_percentage_to_float(percent_str):
    """Converts percentage string (e.g., "75.47%") to float (e.g., 0.7547)."""
    if pd.isna(percent_str):
        return np.nan
    if isinstance(percent_str, (int, float)):
        return float(percent_str) / 100 if abs(float(percent_str)) > 1 else float(percent_str)
    s = str(percent_str).replace('%', '').strip()
    try:
        val = float(s)
        return val / 100
    except ValueError:
        return np.nan

def extract_location_sensor(job_name):
    """Extracts location and sensor from job_name (e.g., western_rgb_20230612)."""
    if pd.isna(job_name):
        return "Unknown", "Unknown"
    parts = str(job_name).split('_')
    if len(parts) >= 2:
        location = parts[0].capitalize()
        sensor = parts[1].upper()
        if sensor not in ["RGB", "MS"]:
            sensor = "OTHER"
        return location, sensor
    return "Unknown", "Unknown"

# --- Apply Data Cleaning and Preprocessing ---
df['Wall-Clock Time (s)'] = df['Wall-Clock Time'].apply(convert_wall_clock_to_seconds)
df['CPU Efficiency (float)'] = df['CPU Efficiency (%)'].apply(convert_percentage_to_float)
df['Memory Utilized (GB)'] = df['Memory Utilized'].apply(convert_memory_to_gb)
df['num_cores_numeric'] = pd.to_numeric(df['num_cores'], errors='coerce')
df[['location', 'sensor']] = df['job_name'].apply(lambda x: pd.Series(extract_location_sensor(x)))

# --- Calculation of Derived and Normalized Metrics ---
# This section is organized to ensure all columns are created before use.

# 1. Create time columns in different units
df['execution_time_min'] = df['execution_time'] / 60
df['Wall-Clock Time (min)'] = df['Wall-Clock Time (s)'] / 60
df['Wall-Clock Time (hr)'] = df['Wall-Clock Time (s)'] / 3600

# 2. Impute area and calculate area in hectares
df['area_m2_numeric'] = pd.to_numeric(df['area_m2'], errors='coerce')
df['imputed_area_m2'] = df.groupby('job_name')['area_m2_numeric'].transform('max')
df['area_ha'] = df['imputed_area_m2'] / 10000

# 3. Calculate normalized metrics
df['time_per_ha_min'] = df['execution_time_min'] / df['area_ha']
df['cpu_core_hours'] = df['num_cores_numeric'] * df['Wall-Clock Time (hr)']
df['core_hours_per_ha'] = df['cpu_core_hours'] / df['area_ha']


# --- Define mapping for full processing step names ---
step_name_map = {
    'om': 'Orthomosaic Creation',
    'omalign': 'Orthomosaic Alignment',
    'plottile': 'Plot Tile Creation',
    'dgr': 'Direct Georeferencing',
    'ir': 'Image Registration',
    'maptile': 'Map Tile Creation'
}
step_order_short = ['om', 'omalign', 'plottile', 'dgr', 'ir', 'maptile']
step_order_full = [step_name_map[step] for step in step_order_short]


# --- Set plotting style for ACM quality ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 12


# --- Chart 1: Scatter plot for "om" processing_step, split by sensor ---
df_om_full = df[df['processing_step'] == 'om'].copy()
df_om_full.dropna(subset=['area_ha', 'Wall-Clock Time (min)', 'sensor'], inplace=True)

df_om_rgb = df_om_full[df_om_full['sensor'] == 'RGB']
df_om_ms = df_om_full[df_om_full['sensor'] == 'MS']

fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), sharey=True)

if not df_om_rgb.empty:
    sns.scatterplot(data=df_om_rgb, x='area_ha', y='Wall-Clock Time (min)', ax=axes1[0], s=50, alpha=0.7, legend=False)
    axes1[0].set_title('RGB Sensor', fontsize=16)
    axes1[0].set_xlabel('Area Processed (hectares)', fontsize=16, labelpad=15)
    axes1[0].set_ylabel('Processing Time (minutes)', fontsize=16)
    axes1[0].tick_params(axis='both', labelsize=12)
else:
    axes1[0].set_title('RGB Sensor (No Data)')
    axes1[0].text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=axes1[0].transAxes)

if not df_om_ms.empty:
    sns.scatterplot(data=df_om_ms, x='area_ha', y='Wall-Clock Time (min)', ax=axes1[1], color='orange', s=50, alpha=0.7, legend=False)
    axes1[1].set_title('Multispectral Sensor', fontsize=16)
    axes1[1].set_xlabel('Area Processed (hectares)', fontsize=16, labelpad=15)
    axes1[1].set_ylabel('')
    axes1[1].tick_params(axis='both', labelsize=12)
else:
    axes1[1].set_title('MS Sensor (No Data)')
    axes1[1].text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=axes1[1].transAxes)

if not df_om_full.empty:
    axes1[0].set_xlim(left=0, right=20)
    axes1[1].set_xlim(left=0, right=20)
    axes1[0].set_ylim(bottom=0, top=250)
    axes1[1].set_ylim(bottom=0, top=250)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('chart1_om_area_vs_time_by_sensor.png')
print("Chart 1 (om_area_vs_time_by_sensor.png) generated.")
plt.close(fig1)


# --- Chart 2: Normalized Processing Time by Sensor ---
df_chart2 = df.copy()
df_chart2.dropna(subset=['time_per_ha_min', 'sensor', 'processing_step'], inplace=True)

avg_rate = df_chart2.groupby(['sensor', 'processing_step'])['time_per_ha_min'].mean().unstack(fill_value=0)

if not avg_rate.empty:
    avg_rate.rename(columns=step_name_map, inplace=True)
    avg_rate.sort_index(ascending=False, inplace=True)
    ordered_cols_for_chart2 = [step for step in step_order_full if step in avg_rate.columns]
    avg_rate = avg_rate[ordered_cols_for_chart2]
    
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    avg_rate.plot(kind='bar', stacked=True, colormap='tab10', ax=ax2, width=0.6)
    
    ax2.set_title('Normalized Mean Processing Time by Sensor', fontsize=16)
    ax2.set_ylabel('Mean Processing Time (minutes per hectare)', fontsize=12)
    ax2.set_xlabel('Sensor Type', fontsize=12, labelpad=10)
    ax2.legend(title='Processing Step', loc='upper right')
    
    current_labels = [item.get_text() for item in ax2.get_xticklabels()]
    new_labels = ['Multispectral' if label == 'MS' else label for label in current_labels]
    ax2.set_xticklabels(new_labels, rotation=0, ha='center', fontsize=12)
    
    totals = avg_rate.sum(axis=1)
    y_offset = max(totals) * 0.02
    for i, total in enumerate(totals):
        ax2.text(i, total + y_offset, f'{total:.1f}', ha='center', va='bottom', fontsize=10, color='black')

    ax2.set_ylim(bottom=0, top=ax2.get_ylim()[1] * 1.1)
    
    plt.tight_layout()
    plt.savefig('chart2_normalized_time_by_sensor.png')
    print("Chart 2 (chart2_normalized_time_by_sensor.png) generated.")
    plt.close(fig2)
else:
    print("No data available for Chart 2 (Normalized Processing Time).")


# --- Chart 3: Bar chart for average values by processing_step ---
df_chart3_metrics = df.groupby('processing_step').agg(
    avg_cpu_efficiency=('CPU Efficiency (float)', 'mean'),
    avg_mem_utilized_gb=('Memory Utilized (GB)', 'mean'),
    avg_num_cores=('num_cores_numeric', 'mean'),
    avg_core_hours_per_ha=('core_hours_per_ha', 'mean')
).reset_index()

df_chart3_metrics['processing_step'] = df_chart3_metrics['processing_step'].map(step_name_map)
df_chart3_metrics['processing_step'] = pd.Categorical(df_chart3_metrics['processing_step'], categories=step_order_full, ordered=True)
df_chart3_metrics.sort_values('processing_step', inplace=True)
df_chart3_metrics.dropna(subset=['processing_step'], inplace=True) 

ordered_steps_present_for_chart3 = [s for s in step_order_full if s in df_chart3_metrics['processing_step'].unique()]
wrapped_labels = [textwrap.fill(label, width=15) for label in ordered_steps_present_for_chart3]

if not df_chart3_metrics.empty and ordered_steps_present_for_chart3:
    fig3, axes3 = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True)
    
    palette = sns.color_palette('tab10', n_colors=len(ordered_steps_present_for_chart3))

    # Function to add labels to bars
    def add_bar_labels(ax, as_integer=False):
        for p in ax.patches:
            height = p.get_height()
            if pd.isna(height): continue
            
            if as_integer:
                label = f'{height:.0f}'
            # For CPU Efficiency plot which has a fixed 0-1 scale
            elif ax.get_ylim()[1] <= 1.0:
                 label = f'{height:.2f}'
            else: # For other plots like Memory and Core-Hours/Ha
                 label = f'{height:.1f}'

            ax.annotate(label,
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 5), textcoords='offset points',
                        fontsize=7, color='black')
        # Adjust y-limit to make space for labels, but not for the fixed 0-1 plot
        if ax.get_ylim()[1] > 1:
            ax.set_ylim(top=ax.get_ylim()[1] * 1.18)

    # CPU Efficiency
    ax = axes3[0,0]
    sns.barplot(data=df_chart3_metrics, x='processing_step', y='avg_cpu_efficiency', ax=ax, hue='processing_step', palette=palette, order=ordered_steps_present_for_chart3, dodge=False, legend=False)
    ax.set_title('Mean CPU Utilization')
    ax.set_xlabel('')
    ax.set_ylabel('CPU Efficiency (Ratio 0-1)')
    ax.set_ylim(0, 1)
    add_bar_labels(ax)

    # Memory Utilized
    ax = axes3[0,1]
    sns.barplot(data=df_chart3_metrics, x='processing_step', y='avg_mem_utilized_gb', ax=ax, hue='processing_step', palette=palette, order=ordered_steps_present_for_chart3, dodge=False, legend=False)
    ax.set_title('Average Memory Utilized')
    ax.set_xlabel('')
    ax.set_ylabel('Memory Utilized (GB)')
    ax.set_ylim(bottom=0)
    add_bar_labels(ax)

    # Num Cores
    ax = axes3[1,0]
    sns.barplot(data=df_chart3_metrics, x='processing_step', y='avg_num_cores', ax=ax, hue='processing_step', palette=palette, order=ordered_steps_present_for_chart3, dodge=False, legend=False)
    ax.set_title('Mean Number of CPU Cores Allocated')
    ax.set_xlabel('Processing Step', labelpad=15)
    ax.set_ylabel('Number of Cores')
    ax.set_xticklabels(wrapped_labels, rotation=0, ha='center')
    ax.set_ylim(bottom=0)
    add_bar_labels(ax, as_integer=True)

    # Core-Hours per Hectare
    ax = axes3[1,1]
    sns.barplot(data=df_chart3_metrics, x='processing_step', y='avg_core_hours_per_ha', ax=ax, hue='processing_step', palette=palette, order=ordered_steps_present_for_chart3, dodge=False, legend=False)
    ax.set_title('Mean CPU Core-Hours per Hectare')
    ax.set_xlabel('Processing Step', labelpad=15)
    ax.set_ylabel('Core-Hours per Hectare')
    ax.set_xticklabels(wrapped_labels, rotation=0, ha='center')
    ax.set_ylim(bottom=0)
    add_bar_labels(ax)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig('chart3_avg_metrics_by_step_ordered.png')
    print("Chart 3 (avg_metrics_by_step_ordered.png) generated.")
    plt.close(fig3)
else:
    print("No data available for Chart 3 (average metrics by step) or no steps match the defined order.")

print("\nAll requested charts have been generated and saved as PNG files.")
print("Please check the generated PNG files in the script's directory.")