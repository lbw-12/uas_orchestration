import os
import time
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import subprocess
import argparse
import sys
import orchestrate_funcs as om_funcs

def get_git_root():
    try:
        # Run the Git command and decode the result
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                       stderr=subprocess.STDOUT).strip().decode("utf-8")
        return root
    except subprocess.CalledProcessError:
        raise Exception("This directory is not inside a Git repository or Git is not installed.")

class FolderStabilityHandler(FileSystemEventHandler):
    def __init__(self, watch_dir, stability_interval=10):
        self.watch_dir = watch_dir
        self.stability_interval = stability_interval
        self.last_event_time = time.time()
        # Cooldown timer
        self.last_run_time = 0
        self.initial_run_done = False # Flag for the first automatic run

    def on_any_event(self, event):
        # Update the last event time whenever any event occurs
        self.last_event_time = time.time()


def trigger_job_when_stable(handler, orchestrate_script, config_file, cooldown_period):
    # Perform the initial run automatically
    if not handler.initial_run_done:
        print("Performing initial automatic job run...")
        working_dir = os.getcwd()
        python_executable = sys.executable  # This gets the path to the current running Python

        result = subprocess.run([python_executable, orchestrate_script,  "--config_file", config_file],
                                cwd=working_dir,
                                env=os.environ.copy(),
                                capture_output=True,
                                text=True)

        if result.returncode == 0:
            output = result.stdout.strip()
            print(f"Initial Run Success: {output}")
            handler.last_run_time = time.time()
        else:
            print(f"Initial Run Error: {result.stderr.strip()}")

        handler.initial_run_done = True
        handler.last_event_time = time.time()
        print("Initial job run complete. Now monitoring for stability.")

    while True:
        time_since_last_event = time.time() - handler.last_event_time
        time_since_last_run = time.time() - handler.last_run_time

        is_stable = time_since_last_event > handler.stability_interval
        is_cooled_down = time_since_last_run > cooldown_period
        if is_stable and is_cooled_down:
            print("Folder is stable and cooldown period has passed. Triggering uas_orchestrate.py...")
            working_dir = os.getcwd()
            python_executable = sys.executable  # This gets the path to the current running Python

            result = subprocess.run([python_executable, orchestrate_script,  "--config_file", config_file],
                                    cwd=working_dir,
                                    env=os.environ.copy(),
                                    capture_output=True,
                                    text=True)

            if result.returncode == 0:
                output = result.stdout.strip()
                print(f"Stability Run Success: {output}")
                handler.last_run_time = time.time()

            else:
                print(f"Stability Run Error: {result.stderr.strip()}")

            # Reset the event time to prevent immediate re-triggering
            handler.last_event_time = time.time()

        time.sleep(5)
                
if __name__ == "__main__":

    # Setup argparse
    parser = argparse.ArgumentParser(description="Process UAS data using a config file.")
    parser.add_argument("--config_file", required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    # Load and process configuration
    config = om_funcs.load_config(args.config_file)

    # Extract values from config file
    watch_dir = os.path.join(config['base_folder'],'flights')

    git_root = get_git_root()

    print(f'watch dir: {watch_dir}')
    
    orchestrate_script = os.path.join(git_root, config['uas_pipeline']['orchestrate'])

    stability_interval = 3600       # 1 hour of  inactivity required for stability.
    cooldown_period = 43200        # 12 hours of cooldown period

    event_handler = FolderStabilityHandler(watch_dir, stability_interval)
    observer = PollingObserver()
    observer.schedule(event_handler, watch_dir, recursive=True)
    observer.start()
    try:
        # Continuously check for folder stability.
        trigger_job_when_stable(event_handler, orchestrate_script, args.config_file, cooldown_period)
    except KeyboardInterrupt:
        print("Stopping observer...")
    finally:
        observer.stop()
        observer.join()