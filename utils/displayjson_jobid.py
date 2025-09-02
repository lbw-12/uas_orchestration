import json
import subprocess
import argparse

def create_html_report(json_path, html_path):
    """
    Reads a JSON file with job statuses and generates a colored HTML table.

    Args:
        json_path (str): The path to the input JSON file.
        html_path (str): The path for the output HTML file.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}.")
        return

    # --- Start HTML Generation ---
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Job Status Report</title>
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
                vertical-align: top;
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
            .cell-green {
                background-color: #c6f6d5; /* Tailwind's green-200 */
                color: #2f855a; /* Tailwind's green-800 */
                font-weight: 500;
            }
            .cell-gray {
                background-color: #e2e8f0; /* Tailwind's gray-300 */
                color: #718096; /* Tailwind's gray-600 */
            }
            .dependency-text {
                color: #4a5568; /* Muted gray text */
                font-size: 12px;   /* Smaller font size */
                font-style: italic;/* Italicized text */
                font-weight: 400;
            }
            hr {
                border: none;
                border-top: 1px solid #ccc;
                margin: 8px 0;
            }
        </style>
    </head>
    <body>
        <h1>Job Status Report</h1>
        <table>
    """

    # --- Find all unique step keys (ignoring _dep keys) and sort them ---
    all_steps = set()
    for user, types in data.items():
        for type_name, dates in types.items():
            for date, steps in dates.items():
                for key in steps.keys():
                    if not key.endswith('_dep'):
                        all_steps.add(key)

    # Sort keys like 'step1', 'step2', 'step10' correctly
    sorted_steps = sorted(list(all_steps), key=lambda x: int(x.replace('step', '')))

    # --- Create Table Header ---
    header_row = "<tr><th>User</th><th>Type</th><th>Date</th>"
    for step in sorted_steps:
        header_row += f"<th>{step.capitalize()}</th>"
    header_row += "</tr>"
    html_content += header_row

    # --- Create Table Body ---
    for user, types in data.items():
        for type_name, dates in types.items():
            for date, steps in dates.items():
                row = f"<tr><td>{user}</td><td>{type_name}</td><td>{date}</td>"
                for step_key in sorted_steps:
                    step_value = steps.get(step_key)
                    dep_key = f"{step_key}_dep"
                    dep_value = steps.get(dep_key)

                    cell_class = "cell-gray"
                    cell_content = ""

                    if step_value is not None:
                        cell_content = str(step_value)
                        cell_class = "cell-green"
                    else:
                        cell_content = "NULL"

                    if dep_value is not None:
                        # Split dependencies by colon
                        dep_list = str(dep_value).split(':')
                        
                        # Format each dependency with the new style and prepended text
                        formatted_deps_list = []
                        for dep in dep_list:
                            # Wrap in a span with the new class and prepend text
                            formatted_deps_list.append(f'<span class="dependency-text">dependency: {dep.strip()}</span>')
                            
                        # Join them with <br> for HTML newlines
                        formatted_deps = '<br>'.join(formatted_deps_list)
                        
                        # Add a horizontal rule and the formatted dependencies
                        cell_content += f"<hr>{formatted_deps}"

                    row += f'<td class="{cell_class}">{cell_content}</td>'
                row += "</tr>"
                html_content += row

    # --- Close HTML ---
    html_content += """
        </table>
    </body>
    </html>
    """

    # --- Write to network location ---
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"Successfully generated HTML report at: {html_path}")


# --- Main execution ---
if __name__ == "__main__":
    # Use the provided 'job_id.json' from the user upload

    parser = argparse.ArgumentParser(description='Process flight data and generate HTML report.')
    parser.add_argument('--json_file', type=str, required=True, help='Path to the JSON file containing flight data.')
    parser.add_argument('--output_html_file', type=str, required=True, help='Path to the output HTML file.')
    args = parser.parse_args()

    json_file = args.json_file
    output_html_file = args.output_html_file

    create_html_report(json_file, output_html_file)