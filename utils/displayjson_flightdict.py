import json
import pandas as pd
import subprocess
import argparse

def create_flight_status_html(df, html_path):
    """
    Generates a styled HTML report from a pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the flight status data.
        html_path (str): The path for the output HTML file.
    """
    # --- Start HTML Generation with modern CSS styling ---
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Flight Status Report</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                margin: 20px;
                background-color: #f4f4f9;
                color: #333;
            }
            h1 {
                color: #444;
                text-align: center;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                box-shadow: 0 2px 15px rgba(0,0,0,0.1);
                background-color: #fff;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px 15px;
                text-align: left;
                font-size: 14px;
            }
            th {
                background-color: #4a5568; /* Tailwind's gray-700 */
                color: #ffffff;
                font-weight: bold;
                position: sticky;
                top: 0;
            }
            tr:nth-child(even) {
                background-color: #f8f8f8;
            }
            tr:hover {
                background-color: #f1f1f1;
            }
            /* Status-specific cell styles */
            .cell-validated {
                background-color: #c6f6d5; /* Green */
                color: #2f855a;
                font-weight: 500;
            }
            .cell-complete {
                background-color: #e2e8f0; /* Gray */
                color: #4a5568;
                font-weight: 500;
            }
            .cell-error {
                background-color: #fed7d7; /* Red */
                color: #c53030;
                font-weight: 500;
            }
            .cell-not_ready {
                background-color: #fefcbf; /* Yellow */
                color: #b7791f;
                font-weight: 500;
            }
            .cell-na {
                background-color: #e2e8f0; /* Gray */
                color: #718096;
            }
        </style>
    </head>
    <body>
        <h1>Flight Status Report</h1>
        <table>
    """

    # --- Create Table Header ---
    header_row = "<tr>"
    for col_name in df.columns:
        header_row += f"<th>{col_name.replace('_', ' ').title()}</th>"
    header_row += "</tr>"
    html_content += header_row

    # --- Create Table Body ---
    # Map status values to CSS class names
    status_to_class = {
        'validated': 'cell-validated',
        'complete': 'cell-complete',
        'error': 'cell-error',
        'not_ready': 'cell-not_ready',
        'N/A': 'cell-na'
    }

    for index, row_data in df.iterrows():
        row_html = "<tr>"
        for col_name in df.columns:
            value = row_data[col_name]
            # Get the appropriate CSS class for the status, default to 'cell-na'
            cell_class = status_to_class.get(value, 'cell-na')
            row_html += f'<td class="{cell_class}">{value}</td>'
        row_html += "</tr>"
        html_content += row_html

    # --- Close HTML ---
    html_content += """
        </table>
    </body>
    </html>
    """

    # --- Write to file ---
    try:
        with open(html_path, 'w') as f:
            f.write(html_content)
        print(f"Successfully generated HTML report at: {html_path}")
    except Exception as e:
        print(f"Could not write HTML file. Error: {e}")


def process_flight_data(file_path: str, output_html_file: str):
    """
    Loads flight data from JSON and generates a styled HTML report.

    Args:
        file_path: The path to the JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            flight_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return

    records = []
    # This complex iteration handles the nested structure of the input JSON
    for top_level_key, sites in flight_data.items():
        for site_name, types in sites.items():
            for image_type, dates in types.items():
                for date, steps in dates.items():
                    record = {
                        "Site": site_name,
                        "Type": image_type,
                        "Date": date,
                    }
                    for step_name, details in steps.items():
                        if isinstance(details, dict) and 'status' in details:
                            record[step_name] = details['status']
                    records.append(record)

    if not records:
        print("No records to display.")
        return

    df = pd.DataFrame(records)

    # Order the step columns numerically
    step_columns = [col for col in df.columns if col.startswith('step')]
    step_columns.sort(key=lambda x: int(x.replace('step', '')))
    
    # Define the final column order
    id_columns = ["Site", "Type", "Date"]
    ordered_columns = id_columns + step_columns
    
    # Reorder DataFrame and fill missing values
    df = df[ordered_columns].fillna('N/A')

    # Generate the HTML file
    #output_html_file = '/fs/ess/PAS2699/nitrogen/data/uas/published/status/report_flight_status.html'
    create_flight_status_html(df, output_html_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process flight data and generate HTML report.')
    parser.add_argument('--json_file', type=str, required=True, help='Path to the JSON file containing flight data.')
    parser.add_argument('--output_html_file', type=str, required=True, help='Path to the output HTML file.')
    args = parser.parse_args()

    json_file = args.json_file
    output_html_file = args.output_html_file

    # Assumes the JSON file is in a specific path.
    # You may need to change this path to match the location of your file.
    #json_file = '/users/PAS2312/lwaltz/code/uas_orchestration/execution/flight_dict.json'
    process_flight_data(json_file, output_html_file)
