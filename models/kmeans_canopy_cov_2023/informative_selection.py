import pandas as pd
import re

file_path = "color_based.csv"
df = pd.read_csv(file_path)

target_labels = {
    0: "VE", 1: "VC", 2: "V1", 3: "V2", 4: "V3", 5: "V4", 6: "V5",
    7: "V6", 8: "V7", 9: "V8", 10: "Vn", 11: "R1", 12: "R2",
    13: "R3", 14: "R4", 15: "R5", 16: "R6", 17: "R7"
}

criteria = {
    "VE": (24.93, 51.19, 48.81, 75.07),
    "VC": (27.92, 52.48, 47.52, 72.08),
    "V1": (33.15, 56.03, 43.97, 66.85),
    "V2": (35.31, 61.34, 38.66, 64.69),
    "V3": (58.60, 68.08, 31.92, 41.40),
    "V4": (58.41, 76.59, 23.41, 41.59),
    "V5": (63.20, 85.81, 14.19, 36.80),
    "V6": (66.44, 86.18, 13.82, 33.56),
    "V7": (68.67, 89.53, 10.47, 31.33),
    "V8": (82.60, 98.92, 1.08, 17.40),
    "Vn": (91.79, 99.70, 0.30, 8.21),
    "R1": (85.74, 99.91, 0.09, 14.26),
    "R2": (95.78, 99.84, 0.16, 4.22),
    "R3": (91.32, 99.89, 0.11, 8.68),
    "R4": (94.90, 99.69, 0.31, 5.10),
    "R5": (74.16, 97.49, 2.51, 25.84),
    "R6": (90.45, 99.98, 0.02, 9.55),
    "R7": (92.21, 99.53, 0.47, 7.79)
}

pattern = r'\/([^\/]+_\d{8}_sony_\d+)_'
df['Filename_Base'] = df['Filename'].apply(lambda x: re.search(pattern, x).group(1) if re.search(pattern, x) else '')

selected_filenames = []

for index, row in df.iterrows():
    target_label = row['Target']
    green_percentage = row['Green_Percentage']
    brown_percentage = row['Brown_Percentage']
    filename = row['Filename']

    target_str = target_labels[target_label]

    green_min, green_max, brown_min, brown_max = criteria[target_str]
    print(target_str, green_min, green_max, brown_min, brown_max)
    if (
            (green_min > green_percentage or green_percentage > green_max) and
            (brown_min > brown_percentage or brown_percentage > brown_max)
    ):
        selected_filenames.append(f"/fs/ess/PAS0272/chaeun/combined/train/{target_label + 1:02d}/" + filename)

output_csv_path = 'color_noise.csv'
pd.DataFrame(selected_filenames, columns=['Filename']).to_csv(output_csv_path, index=False)

print(f"Selected filenames saved to {output_csv_path}")
