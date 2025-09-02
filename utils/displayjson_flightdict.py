import json
import pandas as pd
import subprocess
import argparse

def create_flight_status_html(df, html_path):
    """
    Generates a styled HTML report from a pandas DataFrame with row numbers and sticky filters.

    Args:
        df (pd.DataFrame): DataFrame containing the flight status data.
        html_path (str): The path for the output HTML file.
    """
    # --- Get unique values for filters ---
    site_options = sorted(df['Site'].unique())
    type_options = sorted(df['Type'].unique())
    
    status_columns = [col for col in df.columns if col.startswith('step')]
    all_statuses = pd.unique(df[status_columns].values.ravel('K'))
    status_options = sorted([s for s in all_statuses if s != 'N/A' and s])


    # --- Start HTML Generation with modern CSS styling and filters ---
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Flight Status Report</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                margin: 0; 
                background-color: #f4f4f9;
                color: #333;
            }}
            h1 {{
                color: #444;
                text-align: center;
                padding-top: 20px;
            }}
            .filter-container {{
                display: flex;
                justify-content: center;
                gap: 20px;
                padding: 15px;
                background-color: #ffffff;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                position: sticky;
                top: 0;
                z-index: 10;
            }}
            .filter-container select {{
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }}
            .table-wrapper {{
                padding: 0 20px 20px 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                box-shadow: 0 2px 15px rgba(0,0,0,0.1);
                background-color: #fff;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px 15px;
                text-align: left;
                font-size: 14px;
            }}
            /* --- MODIFICATION START: Style for the new row number column --- */
            th:first-child, td:first-child {{
                text-align: center;
                width: 50px; /* Give it a fixed width */
            }}
            /* --- MODIFICATION END --- */
            th {{
                background-color: #4a5568;
                color: #ffffff;
                font-weight: bold;
                position: sticky;
                top: 68px; 
            }}
            tr:nth-child(even) {{
                background-color: #f8f8f8;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            /* Status-specific cell styles */
            .cell-complete {{ background-color: #c6f6d5; color: #2f855a; font-weight: 500; }}
            .cell-validated {{ background-color: #fefcbf; color: #b7791f; font-weight: 500; }}
            .cell-not_ready {{ background-color: #feebc8; color: #9c4221; font-weight: 500; }}
            .cell-not_validated {{ background-color: #bee3f8; color: #2c5282; font-weight: 500; }}
            .cell-error {{ background-color: #fed7d7; color: #c53030; font-weight: 500; }}
            .cell-na {{ background-color: #e2e8f0; color: #718096; }}
        </style>
    </head>
    <body>
        <div class="filter-container">
            <div>
                <label for="siteFilter">Site:</label>
                <select id="siteFilter" onchange="filterTable()">
                    <option value="all">All Sites</option>
                    {''.join([f'<option value="{site}">{site}</option>' for site in site_options])}
                </select>
            </div>
            <div>
                <label for="typeFilter">Type:</label>
                <select id="typeFilter" onchange="filterTable()">
                    <option value="all">All Types</option>
                    {''.join([f'<option value="{img_type}">{img_type}</option>' for img_type in type_options])}
                </select>
            </div>
            <div>
                <label for="statusFilter">Status:</label>
                <select id="statusFilter" onchange="filterTable()">
                    <option value="all">Any Status</option>
                    {''.join([f'<option value="{status}">{status.replace("_", " ").title()}</option>' for status in status_options])}
                </select>
            </div>
        </div>
        
        <div class="table-wrapper">
            <h1>Flight Status Report</h1>
            <table id="statusTable">
    """

    # --- Create Table Header ---
    html_content += "<thead>"
    header_row = "<tr>"
    # --- MODIFICATION START: Add header for the number column ---
    header_row += "<th>#</th>"
    # --- MODIFICATION END ---
    for col_name in df.columns:
        header_row += f"<th>{col_name.replace('_', ' ').title()}</th>"
    header_row += "</tr>"
    html_content += header_row
    html_content += "</thead>"

    # --- Create Table Body ---
    html_content += "<tbody>"
    for index, row_data in df.iterrows():
        row_html = "<tr>"
        # --- MODIFICATION START: Add cell with the row number (index + 1) ---
        row_html += f"<td>{index + 1}</td>"
        # --- MODIFICATION END ---
        for col_name in df.columns:
            value = str(row_data[col_name])
            
            cell_class = ''
            if 'error' in value.lower():
                cell_class = 'cell-error'
            elif value == 'complete':
                cell_class = 'cell-complete'
            elif value == 'validated':
                cell_class = 'cell-validated'
            elif value == 'not_ready':
                cell_class = 'cell-not_ready'
            elif value == 'not_validated':
                cell_class = 'cell-not_validated'
            else:
                cell_class = 'cell-na'
            
            row_html += f'<td class="{cell_class}">{value}</td>'
        row_html += "</tr>"
        html_content += row_html
    html_content += "</tbody>"

    # --- Close HTML and Add JavaScript ---
    html_content += """
            </table>
        </div>

        <script>
            function filterTable() {
                const siteFilter = document.getElementById('siteFilter').value;
                const typeFilter = document.getElementById('typeFilter').value;
                const statusFilter = document.getElementById('statusFilter').value;
                const table = document.getElementById('statusTable');
                const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');

                for (let i = 0; i < rows.length; i++) {
                    const row = rows[i];
                    // --- MODIFICATION START: Adjust cell indices for filtering ---
                    const siteCell = row.cells[1]; // Site is now the 2nd column
                    const typeCell = row.cells[2]; // Type is now the 3rd column
                    const stepCells = Array.from(row.cells).slice(4); // Steps start from the 5th column
                    // --- MODIFICATION END ---

                    const siteMatch = (siteFilter === 'all' || siteCell.textContent.trim() === siteFilter);
                    const typeMatch = (typeFilter === 'all' || typeCell.textContent.trim() === typeFilter);

                    let statusMatch = (statusFilter === 'all');
                    if (!statusMatch) {
                        for (const cell of stepCells) {
                            if (cell.textContent.trim().toLowerCase().includes(statusFilter.toLowerCase())) {
                                statusMatch = true;
                                break;
                            }
                        }
                    }

                    if (siteMatch && typeMatch && statusMatch) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                }
            }
        </script>

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
    create_flight_status_html(df, output_html_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process flight data and generate HTML report.')
    parser.add_argument('--json_file', type=str, required=True, help='Path to the JSON file containing flight data.')
    parser.add_argument('--output_html_file', type=str, required=True, help='Path to the output HTML file.')
    args = parser.parse_args()

    json_file = args.json_file
    output_html_file = args.output_html_file

    process_flight_data(json_file, output_html_file)