import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap

# --- Load Data ---
# User provided file path
csv_file = '/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_perf/aggregated_metrics_2025.csv'
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: The CSV file was not found at {csv_file}")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# --- Helper Functions for Data Cleaning (Unchanged) ---
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

def convert_size_to_gb(size_str):
    """Converts human-readable size string (e.g., "130G", "241M", "512") to GB."""
    if pd.isna(size_str):
        return np.nan
    size_str = str(size_str).strip().upper()

    if size_str.isnumeric():
        return float(size_str) / (1024**3) # Assume bytes if no unit

    unit = size_str[-1]
    if unit in ['G', 'M', 'K']:
        val_str = size_str[:-1]
    else:
        val_str = size_str
        unit = 'B'

    try:
        val = float(val_str)
        if unit == 'G':
            return val
        elif unit == 'M':
            return val / 1024
        elif unit == 'K':
            return val / (1024**2)
        else: # Bytes
            return val / (1024**3)
    except (ValueError, TypeError):
        return np.nan

def extract_location_sensor(job_name):
    """Extracts location and sensor from job_name (e.g., western_bftb_rgb_20230612)."""
    if pd.isna(job_name):
        return "Unknown", "Unknown"
    
    name_lower = str(job_name).lower()
    parts = name_lower.split('_')
    location = parts[0].capitalize()
    
    if 'rgb' in parts:
        sensor = 'RGB'
    elif 'ms' in parts:
        sensor = 'MS' # Changed from Multispectral for consistency
    else:
        sensor = 'OTHER'
        
    return location, sensor

# --- Apply Data Cleaning and Preprocessing (Unchanged) ---
df['Wall-Clock Time (s)'] = df['Wall-Clock Time'].apply(convert_wall_clock_to_seconds)
df['CPU Efficiency (float)'] = df['CPU Efficiency (%)'].apply(convert_percentage_to_float)
df['Memory Utilized (GB)'] = df['Memory Utilized'].apply(convert_memory_to_gb)
df['output_size_gb'] = df['output_size'].apply(convert_size_to_gb)
df['num_cores_numeric'] = pd.to_numeric(df['num_cores'], errors='coerce')
df[['location', 'sensor']] = df['job_name'].apply(lambda x: pd.Series(extract_location_sensor(x)))

df['execution_time_min'] = df['execution_time'] / 60
df['Wall-Clock Time (min)'] = df['Wall-Clock Time (s)'] / 60
df['Wall-Clock Time (hr)'] = df['Wall-Clock Time (s)'] / 3600

df['area_m2_numeric'] = pd.to_numeric(df['area_m2'], errors='coerce')
df['imputed_area_m2'] = df.groupby('job_name')['area_m2_numeric'].transform('max')
df['area_ha'] = df['imputed_area_m2'] / 10000

df['time_per_ha_min'] = df['execution_time_min'] / df['area_ha']
df['cpu_core_hours'] = df['num_cores_numeric'] * df['Wall-Clock Time (hr)']
df['core_hours_per_ha'] = df['cpu_core_hours'] / df['area_ha']
df['output_gb_per_ha'] = df['output_size_gb'] / df['area_ha']

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

# ==============================================================================
# --- NEW: Define Color Palette based on your workflow diagram ---
# ==============================================================================
step_color_map = {
    'Orthomosaic Creation': '#006699',   # Dark Blue
    'Orthomosaic Alignment': '#99ccff', # Light Blue
    'Plot Tile Creation': '#ff9900',      # Orange
    'Direct Georeferencing': '#ffcc00',   # Yellow
    'Image Registration': '#339933',      # Dark Green
    'Map Tile Creation': '#ff99cc'       # Pink
}
# Create a list of colors in the correct order for plotting
color_palette_ordered = [step_color_map[step] for step in step_order_full]

# --- Set plotting style ---
sns.set_theme(style="white", rc={'axes.axisbelow': True})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12

# ==============================================================================
# --- Chart Generation Sections (Updated with new color palette) ---
# ==============================================================================

# --- Chart 1: Scatter Plot of Area vs. Processing Time by Sensor ---
print("Generating Chart 1 (Area vs. Time Scatter Plot)...")

# 1. Prepare data specifically for this chart
df_chart1 = df.copy()
df_chart1 = df_chart1[df_chart1['processing_step'] == 'om']
df_chart1 = df_chart1[df_chart1['sensor'].isin(['RGB', 'MS'])]
df_chart1.dropna(subset=['area_ha', 'execution_time_min'], inplace=True)


# ==============================================================================
# --- NEW: Create a new column with the full sensor names for plotting ---
# ==============================================================================
# Use a dictionary to map the short names to the full names
sensor_name_map = {
    'RGB': 'RGB',
    'MS': 'Multispectral'
}
df_chart1['sensor_full_name'] = df_chart1['sensor'].map(sensor_name_map)


if not df_chart1.empty:
    # 2. Create the plot using the new 'sensor_full_name' column for faceting
    g = sns.relplot(
        data=df_chart1,
        x='area_ha',
        y='execution_time_min',
        col='sensor_full_name', # *** CHANGED: Use the new column for the plot columns
        hue='sensor',         # Keep using original 'sensor' column for color if you want
        legend=False,         # Removes the legend as requested previously
        kind='scatter',
        s=60,
        alpha=0.8,
        edgecolor='w',
        linewidth=0.5,
        height=5,
        aspect=1.2
    )

    # 3. Customize the plot aesthetics
    # *** CHANGED: Update the title template to just use the new full name ***
    g.set_titles("{col_name} Sensor", size=16)
    
    # Set axis labels for the entire figure
    g.set_axis_labels("Area Processed (hectares)", "Processing Time (minutes)", size=14)
    
    # Adjust tick label sizes
    g.tick_params(axis='both', which='major', labelsize=12)

    # ==============================================================================
    # --- NEW: Add this loop to turn on horizontal grid lines for each subplot ---
    # ==============================================================================
    for ax in g.axes.flat:
        ax.yaxis.grid(True, linestyle='-', alpha=0.7)

    # Set consistent axis limits
    max_x = df_chart1['area_ha'].max()
    max_y = df_chart1['execution_time_min'].max()
    g.set(xlim=(0, max_x * 1.1), ylim=(0, max_y * 1.1))

    # Fine-tune layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save the figure
    plt.savefig('chart1_om_area_vs_time_by_sensor.png')
    print("Chart 1 (chart1_om_area_vs_time_by_sensor.png) generated.")
    plt.close(g.fig)
else:
    print("No data available to generate Chart 1.")




# --- Chart 2 (Previously): Normalized Mean Processing Time by Sensor ---
print("Generating Chart 2 (Normalized Time)...")
df_chart2 = df.copy()
df_chart2 = df_chart2[df_chart2['sensor'].isin(['RGB', 'MS'])]
df_chart2.replace([np.inf, -np.inf], np.nan, inplace=True)
df_chart2.dropna(subset=['time_per_ha_min', 'processing_step', 'sensor'], inplace=True)
df_chart2 = df_chart2[df_chart2['processing_step'].isin(step_name_map.keys())]

if not df_chart2.empty:
    avg_rate = df_chart2.groupby(['sensor', 'processing_step'])['time_per_ha_min'].mean().unstack(fill_value=0)
    avg_rate.rename(columns=step_name_map, inplace=True)
    ordered_cols_for_chart2 = [step for step in step_order_full if step in avg_rate.columns]
    avg_rate = avg_rate[ordered_cols_for_chart2]
    
    desired_order = ['RGB', 'MS']
    existing_sensors_in_order = [s for s in desired_order if s in avg_rate.index]
    avg_rate = avg_rate.reindex(existing_sensors_in_order)
    
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    # *** CHANGED: Use the defined colors ***
    avg_rate.plot(kind='bar', stacked=True, ax=ax2, width=0.6, color=color_palette_ordered)

    ax2.yaxis.grid(True, linestyle='-', alpha=0.7)

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

    ax2.set_ylim(bottom=0, top=ax2.get_ylim()[1] * 1.15)
    plt.tight_layout()
    plt.savefig('chart2_normalized_time_by_sensor.png')
    print("Chart 2 (chart2_normalized_time_by_sensor.png) generated.")
    plt.close(fig2)
else:
    print("No data available for Chart 2.")


# --- Chart 3: Mean Performance Metrics by Step ---
print("Generating Chart 3 (Performance Metrics)...")
df_chart3 = df.copy()
df_chart3.replace([np.inf, -np.inf], np.nan, inplace=True)
df_chart3['processing_step_full'] = df_chart3['processing_step'].map(step_name_map)
df_chart3.dropna(subset=['processing_step_full'], inplace=True)

# Define the color palette from your workflow diagram
# Assuming step_color_map and step_order_full are defined as before
color_palette_ordered = [step_color_map.get(step, '#cccccc') for step in step_order_full]

fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))
fig3.suptitle('Mean Performance Metrics for Each Processing Step', fontsize=16, y=1.02)
axes3 = axes3.flatten()

# --- NEW: Create wrapped labels for the x-axis ---
# Adjust the 'width' parameter as needed to get the best look
wrapped_labels = [textwrap.fill(label, width=12, break_long_words=False) for label in step_order_full]


# --- Subplot Generation (no changes here) ---
# Subplot 1: CPU Utilization
sns.barplot(data=df_chart3, x='processing_step_full', y='CPU Efficiency (float)', order=step_order_full, ax=axes3[0], palette=color_palette_ordered, errorbar=None)
axes3[0].set_title('Mean CPU Utilization', fontsize=16)
axes3[0].set_ylabel('CPU Efficiency (Ratio 0-1)', fontsize=14)

# Subplot 2: Memory Utilized
sns.barplot(data=df_chart3, x='processing_step_full', y='Memory Utilized (GB)', order=step_order_full, ax=axes3[1], palette=color_palette_ordered, errorbar=None)
axes3[1].set_title('Mean Memory Utilized', fontsize=16)
axes3[1].set_ylabel('Memory Utilized (GB)', fontsize=14)

# Subplot 3: Number of Cores
sns.barplot(data=df_chart3, x='processing_step_full', y='num_cores_numeric', order=step_order_full, ax=axes3[2], palette=color_palette_ordered, errorbar=None)
axes3[2].set_title('Mean Number of CPU Cores Allocated', fontsize=16)
axes3[2].set_ylabel('Number of Cores', fontsize=14)

# Subplot 4: Core-Hours per Hectare
sns.barplot(data=df_chart3, x='processing_step_full', y='core_hours_per_ha', order=step_order_full, ax=axes3[3], palette=color_palette_ordered, errorbar=None)
axes3[3].set_title('Mean CPU Core-Hours per Hectare', fontsize=16)
axes3[3].set_ylabel('Core-Hours per Hectare', fontsize=14)

# =================================================================================
# --- NEW: More Robust Loop to format all subplots with requested label changes ---
# =================================================================================
for i, ax in enumerate(axes3):
    ax.yaxis.grid(True, linestyle='-', alpha=0.7)
    # Add data labels to each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
    
    # Adjust y-axis limit to give space for labels
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.15)

    # --- Apply specific formatting based on subplot row ---
    if i < 2:  # Top row (subplots 0 and 1)
        # Remove the x-axis title and clear the tick labels
        ax.set_xlabel('')
        ax.set_xticklabels([])
    else:  # Bottom row (subplots 2 and 3)
        # More robust way to set wrapped labels for categorical plots
        ax.set_xticks(ticks=np.arange(len(wrapped_labels))) # Set tick positions
        ax.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=12)
        ax.set_xlabel('Processing Step', labelpad=10, fontsize=14)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# I've appended _v3 to the filename to avoid overwriting your previous attempts
plt.savefig('chart3_avg_metrics_by_step_ordered_v3.png')
print("Chart 3 (chart3_avg_metrics_by_step_ordered_v3.png) generated.")
plt.close(fig3)

# --- Chart 4: STACKED Bar Chart for Normalized Output Size (GB/Hectare) ---
print("Generating Chart 4 (Stacked, Ordered Output Size)...")

# Make a copy to work with
df_chart4 = df.copy()

# ==============================================================================
# --- Process and prepare the 'input' data rows ---
# ==============================================================================

# Create a map of job_name to area in hectares from the processed steps
# This allows us to assign an area to the input data rows
area_map = df_chart4[df_chart4['job_id'] != 'input'].dropna(
    subset=['job_name', 'area_ha']
).drop_duplicates(
    subset='job_name'
).set_index('job_name')['area_ha']

# Isolate the input data rows
df_input = df_chart4[df_chart4['job_id'] == 'input'].copy()

# Prepare the input data to be added to the chart
if not df_input.empty:
    df_input['processing_step'] = 'input' # Use the short name for mapping
    # Assuming the size '2.5G' is in the 'output_size' column from your CSV
    df_input['output_size_gb'] = df_input['output_size'].apply(convert_size_to_gb)
    df_input['area_ha'] = df_input['job_name'].map(area_map)
    df_input['output_gb_per_ha'] = df_input['output_size_gb'] / df_input['area_ha']

    # Combine the processed input data with the output data from other steps
    df_output = df_chart4[df_chart4['job_id'] != 'input']
    df_combined = pd.concat([df_output, df_input], ignore_index=True)
else:
    # If no input rows were found, just use the original dataframe
    df_combined = df_chart4
    print("Warning: No 'input' data rows found in CSV for Chart 4.")


# ==============================================================================
# --- Update step order, name map, and color map to include 'Input Data' ---
# ==============================================================================
step_name_map['input'] = 'Input Data'
# *** Key Change for Stacking Order: Add 'Input Data' to the beginning of the order ***
# This places it at the bottom of the stacked bar chart.
step_order_full_with_input = ['Input Data'] + step_order_full

# Add a color for 'Input Data'
step_color_map['Input Data'] = '#808080'  # Assign a neutral Gray color

# Create a new color palette list that includes the new step
color_palette_ordered_with_input = [step_color_map.get(step, '#cccccc') for step in step_order_full_with_input]


# --- Generate the plot using the combined data ---
df_plot_data = df_combined.copy()
df_plot_data = df_plot_data[df_plot_data['sensor'].isin(['RGB', 'MS'])]
df_plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
df_plot_data.dropna(subset=['output_gb_per_ha', 'processing_step', 'sensor'], inplace=True)
df_plot_data = df_plot_data[df_plot_data['processing_step'].isin(step_name_map.keys())]

if not df_plot_data.empty:
    # Group by sensor and step, calculate mean, and pivot steps to columns
    avg_rate_gb_ha = df_plot_data.groupby(['sensor', 'processing_step'])['output_gb_per_ha'].mean().unstack(fill_value=0)
    avg_rate_gb_ha.rename(columns=step_name_map, inplace=True)
    
    # *** CHANGED: Ensure a consistent order for the stacks using the new order list ***
    ordered_cols_for_chart4 = [step for step in step_order_full_with_input if step in avg_rate_gb_ha.columns]
    avg_rate_gb_ha = avg_rate_gb_ha[ordered_cols_for_chart4]

    # Set the desired order for the x-axis
    desired_order = ['RGB', 'MS']
    existing_sensors_in_order = [s for s in desired_order if s in avg_rate_gb_ha.index]
    avg_rate_gb_ha = avg_rate_gb_ha.reindex(existing_sensors_in_order)

    # Create the plot
    fig4, ax4 = plt.subplots(figsize=(8, 7))
    # *** CHANGED: Use the new color palette with the 'Input Data' color ***
    avg_rate_gb_ha.plot(kind='bar', stacked=True, ax=ax4, width=0.6, color=color_palette_ordered_with_input)

    ax4.yaxis.grid(True, linestyle='-', alpha=0.7)

    # --- Customize the plot (no changes from here down) ---
    ax4.set_title('Normalized Mean Output Size by Sensor', fontsize=16)
    ax4.set_ylabel('Total Mean Output Size (GB per Hectare)', fontsize=12)
    ax4.set_xlabel('Sensor Type', fontsize=12, labelpad=10)
    ax4.legend(title='Processing Step', loc='upper right')

    current_labels = [item.get_text() for item in ax4.get_xticklabels()]
    new_labels = ['Multispectral' if label == 'MS' else label for label in current_labels]
    ax4.set_xticklabels(new_labels, rotation=0, ha='center', fontsize=12)

    totals = avg_rate_gb_ha.sum(axis=1)
    y_offset = max(totals) * 0.02
    for i, total in enumerate(totals):
        ax4.text(i, total + y_offset, f'{total:.2f}', ha='center', va='bottom', fontsize=10, color='black')

    ax4.set_ylim(bottom=0, top=ax4.get_ylim()[1] * 1.15)
    plt.tight_layout()
    plt.savefig('chart4_stacked_output_size_with_input.png')
    print("Chart 4 (chart4_stacked_output_size_with_input.png) generated.")
    plt.close(fig4)
else:
    print("No data available for Chart 4 after processing.")

print("\nScript finished.")