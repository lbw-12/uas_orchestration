import pandas as pd
import subprocess
import io
import argparse # Import the argparse module

def get_job_metrics(job_id):
    """
    Retrieves CPU efficiency, memory utilized, wall-clock time, memory efficiency,
    and job state for a given job ID using the 'seff' command at OSC.

    Args:
        job_id (str): The job ID.

    Returns:
        dict: A dictionary containing 'CPU Efficiency', 'Memory Utilized',
              'Wall-Clock Time', 'Memory Efficiency', and 'State'.
              Returns None if metrics can't be fetched.
    """
    try:
        cmd = ["seff", str(job_id)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout

        metrics = {
            "CPU Efficiency": "N/A",
            "Memory Utilized": "N/A",
            "Wall-Clock Time": "N/A",
            "Memory Efficiency": "N/A",
            "State": "N/A" # Initialize State
        }

        for line_raw in output.splitlines(): # Renamed to line_raw to avoid confusion
            line = line_raw.strip() # Ensure no leading/trailing whitespace

            if "State:" in line and not "efficiency" in line.lower() and not "core-walltime" in line.lower(): # Make sure it's the Job State line
                try:
                    # Example: "State: COMPLETED (exit code 0)"
                    prefix = "State:"
                    value_str = line.split(prefix, 1)[1].strip()
                    print(f'job state: {value_str}')
                    if value_str:
                        metrics["State"] = value_str
                    else:
                        metrics["State"] = "N/A_empty"
                except IndexError:
                    metrics["State"] = "N/A_parse_error"
            elif "CPU Utilized:" in line and "CPU Efficiency:" in line: # Newer seff combines these
                 try:
                    parts = line.split()
                    efficiency_index = parts.index("Efficiency:") + 1
                    metrics["CPU Efficiency"] = parts[efficiency_index]
                 except (ValueError, IndexError):
                    pass # Keep N/A if parsing fails
            elif "CPU Efficiency:" in line and "CPU Utilized:" not in line:
                try:
                    # Example: "CPU Efficiency:          64.42% of 01:02:08 core-walltime"
                    prefix = "CPU Efficiency:"
                    value_part = line.split(prefix, 1)[1].strip()
                    metrics["CPU Efficiency"] = value_part.split()[0]
                except (IndexError, ValueError):
                    pass # Keep N/A
            elif "Job Wall-clock time:" in line:
                try:
                    prefix = "Job Wall-clock time:"
                    value_str = line.split(prefix, 1)[1].strip()
                    if value_str:
                        metrics["Wall-Clock Time"] = value_str
                    else:
                        metrics["Wall-Clock Time"] = "N/A_empty"
                except IndexError:
                    metrics["Wall-Clock Time"] = "N/A_parse_error"
            elif "Memory Utilized:" in line:
                try:
                    prefix = "Memory Utilized:"
                    value_str = line.split(prefix, 1)[1].strip()
                    if value_str:
                        metrics["Memory Utilized"] = value_str
                    else:
                        metrics["Memory Utilized"] = "N/A_empty"
                except IndexError:
                    metrics["Memory Utilized"] = "N/A_parse_error"
            elif "Memory Efficiency:" in line:
                try:
                    prefix = "Memory Efficiency:"
                    value_part = line.split(prefix, 1)[1].strip()
                    metrics["Memory Efficiency"] = value_part.split()[0]
                except (IndexError, ValueError):
                    metrics["Memory Efficiency"] = "N/A_parse_error"
        return metrics

    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not retrieve metrics for job {job_id}. `seff` error: {e.stderr.strip()}")
        return {
            "CPU Efficiency": "Error_seff",
            "Memory Utilized": "Error_seff",
            "Wall-Clock Time": "Error_seff",
            "Memory Efficiency": "Error_seff",
            "State": "Error_seff" # Add State for this error case
        }
    except FileNotFoundError:
        print("Error: 'seff' command not found. Make sure you are running this on a system where 'seff' is available (like an OSC login node).")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while fetching metrics for job {job_id}: {e}")
        return {
            "CPU Efficiency": "Error_script",
            "Memory Utilized": "Error_script",
            "Wall-Clock Time": "Error_script",
            "Memory Efficiency": "Error_script",
            "State": "Error_script" # Add State for this error case
        }

def main():
    """
    Main function to read CSV, get job metrics, and write an updated CSV,
    taking file paths and column name as command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Query OSC job metrics (using seff) and append them to a CSV.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("output_csv", help="Path for the new CSV file with added metrics.")
    parser.add_argument("job_id_column", help="Name of the column in the input CSV containing the Job IDs.")
    parser.add_argument(
        "--skip_header_check",
        action="store_true",
        help="Skip checking if the job ID column exists in the header (use with caution)."
    )
    args = parser.parse_args()

    input_csv_path = args.input_csv
    output_csv_path = args.output_csv
    job_id_column_name = args.job_id_column

    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV '{input_csv_path}': {e}")
        return

    if not args.skip_header_check and job_id_column_name not in df.columns:
        print(f"Error: Column '{job_id_column_name}' not found in the CSV: {input_csv_path}")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    # Prepare lists to store new metrics
    cpu_efficiencies = []
    mem_utilized_list = []
    wall_clock_times = []
    mem_efficiencies = []
    job_states = [] # New list for job states

    total_jobs = len(df)
    if total_jobs == 0:
        print("Input CSV is empty. Nothing to process.")
        # Create an empty output file with new headers if no jobs
        df["CPU Efficiency (%)"] = []
        df["Memory Utilized"] = []
        df["Memory Efficiency (%)"] = []
        df["Wall-Clock Time"] = []
        df["Job State"] = [] # Add Job State header for empty CSV
        try:
            df.to_csv(output_csv_path, index=False)
            print(f"Empty input CSV. Created output file with new headers: {output_csv_path}")
        except Exception as e:
            print(f"Error writing empty output CSV: {e}")
        return

    print(f"\nProcessing {total_jobs} jobs from '{input_csv_path}'...\n")

    for index, row in df.iterrows():
        try:
            job_id = row[job_id_column_name]
        except KeyError:
            print(f"Error: Job ID column '{job_id_column_name}' not found in row {index + 1}. Skipping this row.")
            cpu_efficiencies.append("Error_column_missing")
            mem_utilized_list.append("Error_column_missing")
            wall_clock_times.append("Error_column_missing")
            mem_efficiencies.append("Error_column_missing")
            job_states.append("Error_column_missing") # Add for state
            continue

        print(f"[{index + 1}/{total_jobs}] Querying metrics for job ID: {job_id}...")
        metrics = get_job_metrics(job_id) # This now returns a dict even on error

        # 'metrics' should always be a dictionary based on get_job_metrics structure
        cpu_efficiencies.append(metrics.get("CPU Efficiency", "N/A_fallback"))
        mem_utilized_list.append(metrics.get("Memory Utilized", "N/A_fallback"))
        wall_clock_times.append(metrics.get("Wall-Clock Time", "N/A_fallback"))
        mem_efficiencies.append(metrics.get("Memory Efficiency", "N/A_fallback"))
        job_states.append(metrics.get("State", "N/A_fallback")) # Get the state

    # Add new columns to the DataFrame
    df["CPU Efficiency (%)"] = cpu_efficiencies
    df["Memory Utilized"] = mem_utilized_list
    df["Memory Efficiency (%)"] = mem_efficiencies
    df["Wall-Clock Time"] = wall_clock_times
    df["Job State"] = job_states # Add the new Job State column

    try:
        df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully processed all jobs. Output saved to: {output_csv_path} 🎉")
    except Exception as e:
        print(f"Error writing updated CSV to '{output_csv_path}': {e}")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print("Script terminated because 'seff' command was not found.")
    except Exception as e:
        print(f"An unexpected error occurred in the main execution: {e}")