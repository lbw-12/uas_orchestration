import os
import pathlib
import argparse
import subprocess
from jinja2 import Environment, FileSystemLoader
import pandas as pd
from shapely.geometry import Point
import re
import shutil
import json
import orchestrate_funcs as om_funcs
import time
from pathlib import Path

def get_git_root():
    try:
        # Run the Git command and decode the result
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                       stderr=subprocess.STDOUT).strip().decode("utf-8")
        return root
    except subprocess.CalledProcessError:
        raise Exception("This directory is not inside a Git repository or Git is not installed.")

# This creates the individual shell scripts to generate an orthomosaic for a flight, sensor, and date
def generate_shell_script(flight_dict, config, uas_pipeline):
    # Extract values from config file
    git_root = get_git_root()
    base_folder = config['base_folder']
    om_dir = os.path.join(base_folder, config['om_folder'])
    om_aligned_folder = os.path.join(base_folder, config['om_aligned_folder'])
    plottiles_folder = os.path.join(base_folder, config['plottiles_folder'])
    processing_dir = os.path.join(base_folder, config['processing_folder'])
    om_plot_dict = config['plot_shapefiles']
    shapefiles_alignment_path = os.path.join(base_folder, config['shapefiles_alignment'])
    maptiles_path = config['maptiles_folder']
    logdir_perf = os.path.join(processing_dir,"logs_perf")
    pub_folder_dict = config['publishing_folder']
    os.makedirs(logdir_perf, exist_ok = True)

    # Get the absolute path to the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up Jinja2 to look for templates in the same directory
    env = Environment(loader=FileSystemLoader(git_root))

    # Only generate scripts for the steps in our filtered pipeline
    for step in uas_pipeline:
        count = 0
        print(f'path to template: {uas_pipeline[step]["shell_script_template"]}')
        template = env.get_template(uas_pipeline[step]['shell_script_template'])
        log_folder = os.path.join(processing_dir, uas_pipeline[step]['log_folder'])
        os.makedirs(log_folder, exist_ok = True)
        for flight, value in flight_dict.items():
            for om, value2 in value.items():
                #print(f'flight: {flight}, om: {om}')
                for sensor_type, value3 in value2.items():
                    # print(f'sensor_type: {sensor_type}')
                    for date, value4 in value3.items():
                        # print(f'date: {date}')
                        if sensor_type in uas_pipeline[step]['sensor']:
                            if step in flight_dict[flight][om][sensor_type][date]:
                                if flight_dict[flight][om][sensor_type][date][step]['status'] in ['validated', 'not_ready']:

                                    input_path = value4[step]['input_path'] # was img_dir and folder_path
                                    processing_subdir = pathlib.Path(f'{processing_dir}/{om}_{sensor_type}_{date}')

                                    # Generate unique parameters for plot tile creation shell script
                                    plotimage_source = "om"
                                    corn_relpath = config.get('plot_shapefiles', {}).get(om, {}).get('corn')
                                    soy_relpath = config.get('plot_shapefiles', {}).get(om, {}).get('soy')
                                    corn_path = os.path.join(base_folder, corn_relpath) if corn_relpath else ''
                                    soy_path = os.path.join(base_folder, soy_relpath) if soy_relpath else ''
                                    
                                    output_folder_plottiles_template = config['uas_pipeline']['step3']['output_folder']
                                    output_folder_plottiles = output_folder_plottiles_template.format(om=om, sensor_type=sensor_type, date=date, source = "om")
                                    output_path_plottiles = os.path.join(base_folder, output_folder_plottiles)

                                    # Generate unique parameters for Image Registration shell script
                                    # Make a tag to put in batch script name, to identify invalid jobs
                                    scriptnametag=''

                                    # Create path variables for IR process
                                    output_path_geo_template = os.path.join(base_folder, config['uas_pipeline']['step4']['output_folder'])
                                    output_path_geo = output_path_geo_template.format(om=om, sensor_type=sensor_type, date=date)

                                    output_path_ir_template = os.path.join(base_folder, config['uas_pipeline']['step5']['output_folder'])
                                    output_path_ir = output_path_ir_template.format(om=om, sensor_type=sensor_type, date=date)

                                    ortho_path = os.path.join(f'{om_aligned_folder}',f'{om}_{sensor_type}_{date}_aligned.tif')

                                    # Create path variables for plot patches
                                    output_path_plot_patches_template = os.path.join(base_folder, config['uas_pipeline']['step6']['output_folder'])
                                    output_path_plot_patches = output_path_plot_patches_template.format(om=om, source="om", sensor_type=sensor_type, date=date)

                                    job_title = f'{om}_{sensor_type}_{date}'

                                    # Generate parameters for getting patches from rgb plot tiles
                                    ptp_input_dir = os.path.join(base_folder, config['plottiles_folder'], 'plot_tiles_rgb_om/')
                                    ptp_output_dir = os.path.join(base_folder, config['patches_folder'])
                                    os.makedirs(ptp_output_dir, exist_ok=True)

                                    # Generate parameters for growth stage, canopy cover, and spectral reflectance inference shell scripts
                                    gs_input_dir = os.path.join(base_folder, output_path_plot_patches)
                                    cc_input_dir = os.path.join(base_folder, config['plottiles_folder'], 'plot_tiles_rgb_om/')
                                    sr_input_dir = os.path.join(base_folder, config['plottiles_folder'], 'plot_tiles_multispectral_om/')

                                    model_output_dir = os.path.join(base_folder, config['uas_pipeline']['step9']['output_folder'])
                                    os.makedirs(model_output_dir, exist_ok=True)
                                    model_path = uas_pipeline[step]['model_path']


                                    output_path_gs_template = os.path.join(base_folder, config['uas_pipeline']['step7']['output_folder'], config['uas_pipeline']['step7']['output_file'])
                                    output_path_gs = output_path_gs_template.format(om=om, source="om", date=date)
                                    output_path_cc_json_template = os.path.join(base_folder, config['uas_pipeline']['step8']['output_folder'], config['uas_pipeline']['step8']['output_file'])
                                    output_path_cc_json = output_path_cc_json_template.format(om=om, source="om", date=date)
                                    output_path_sr_template = os.path.join(base_folder, config['uas_pipeline']['step9']['output_folder'], config['uas_pipeline']['step9']['output_file'])
                                    output_path_sr = os.path.join(base_folder, config['uas_pipeline']['step9']['output_folder'])
                                    output_path_sr_json = output_path_sr_template.format(om=om, source="om", date=date) 
                                    output_folder_geojson_template = os.path.join(base_folder, config['uas_pipeline']['step10']['output_folder'], config['uas_pipeline']['step10']['output_file'])
                                    output_folder_geojson = output_folder_geojson_template.format(year = '2025',om=om, date=date)

                                    geojson_inference_dir = model_output_dir

                                    # Generate parameters for publishing shell script
                                    pub_folder = os.path.join(pub_folder_dict[om], om, date)

                                    # This dictionary is a superset of all the parameters for all the steps
                                    data_step = {
                                        "job_title": job_title,     #common
                                        "log_dir": log_folder,     #common
                                        "sensor_type": sensor_type, #common
                                        "logdir_perf": logdir_perf, #common
                                        "date": date,               #common
                                        "python_script": uas_pipeline[step]['python_script'], #common
                                        "om_aligned_folder": om_aligned_folder, #step 2
                                        "om": om,  # step 1,2
                                        "processing_dir": processing_dir, #step 1
                                        "om_dir": om_dir, #step 1
                                        "input_path": input_path, #step 1
                                        "processing_subdir": processing_subdir, #step 1
                                        "script_dir": script_dir, #step 1
                                        "script_name": f'single_job_{sensor_type}.sh', #step 1
                                        "shapefiles_alignment_path": shapefiles_alignment_path, #step 2
                                        "om_folder": om_dir, #step 2
                                        "plotimage_source": plotimage_source, #step 3
                                        "shapefile_path_corn": corn_path, #step 3
                                        "shapefile_path_soy": soy_path, #step 3
                                        "output_path_plottiles": output_path_plottiles, #step 3
                                        "output_path_geo": output_path_geo,
                                        "output_path_ir": output_path_ir,
                                        "ortho_path": ortho_path,
                                        "output_path_plot_patches": output_path_plot_patches,
                                        "maptiles_dir": pub_folder,
                                        "ptp_input_dir": ptp_input_dir,
                                        "ptp_output_dir": ptp_output_dir,
                                        "gs_input_dir": gs_input_dir,
                                        "cc_input_dir": output_path_plottiles,
                                        "sr_input_dir": output_path_plottiles,
                                        "model_output_dir": model_output_dir,
                                        "model_path": model_path,
                                        "geojson_inference_dir": geojson_inference_dir,
                                        "output_path_gs": output_path_gs,
                                        "output_path_cc_json": output_path_cc_json,
                                        "output_path_sr": output_path_sr,
                                        "output_path_sr_json": output_path_sr_json,
                                        "output_folder_geojson": output_folder_geojson
                                    }

                                    output = template.render(data_step)
                                    if not os.path.exists(os.path.join(git_root, uas_pipeline[step]['shell_script_folder'])):
                                        os.makedirs(os.path.join(git_root, uas_pipeline[step]['shell_script_folder']))
                                    f_path = f"{os.path.join(git_root, uas_pipeline[step]['shell_script_folder'])}/{job_title}{scriptnametag}.sh"
                                    with open(f_path, 'w') as f:
                                        f.write(output)
                                        #print(f'Generated: {f_path}')

                                    os.chmod(f_path, 0o770)
                                    count += 1

    print(f'Generated {count} shell scripts for step: {step}')

def submit_job(script_path, working_dir, dependency=None, dry_run = False):
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return None if not dry_run else "dry_run_job_id"

    command = ["sbatch"]
    dep_str = None

    if dependency:
        # Handle both single dependency and list of dependencies
        if isinstance(dependency, list):
            # Filter out any None values from the list before joining
            valid_dependencies = [str(d) for d in dependency if d is not None]
            if valid_dependencies:
                dep_str = ':'.join(valid_dependencies)
        else:
            # If it's not a list, it's a single dependency
            dep_str = str(dependency)

        # Only add the --dependency flag if we have a valid dependency string
        if dep_str:
            command.append(f"--dependency=afterok:{dep_str}")

    command.append(script_path)

    # Print the command to show what would be run
    print(f"{'Dry run:' if dry_run else 'Submitting:'} {' '.join(command)}")

    if dry_run:
        # Check if the counter exists on the function object, if not, initialize it
        if not hasattr(submit_job, "dry_run_counter"):
            submit_job.dry_run_counter = 0
        
        # Increment the counter and format the dummy job ID
        submit_job.dry_run_counter += 1
        dummy_job_id = f"dr_{submit_job.dry_run_counter:04d}"
        print(f"Success: Submitted batch job {dummy_job_id}") # Mimic sbatch output
        return dummy_job_id, dep_str

    result = subprocess.run(
        command,
        cwd=working_dir,
        env=os.environ.copy(),
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        output = result.stdout.strip()
        print(f"Success: {output}")
        match = re.search(r'Submitted batch job (\d+)', output)
        job_id = match.group(1) if match else None
        return job_id, dep_str if job_id else None
    else:
        print(f"Error submitting {script_path}: {result.stderr.strip()}")
        return None

# Generates individual sbatch commands for all dates (where an existing orthomosaic is not already present) for a given list of 
# locations and sensor types. If no list of locations or sensor types is provided, all will be used.
def run_sbatch(flight_dict, working_dir, uas_pipeline, dry_run = False, location_filter = None, date_range = None):
    git_root = get_git_root()
    job_id = {}

    for location in flight_dict.keys():
        for om in flight_dict[location].keys():
            job_id[om] = {}
            for sensor_type in flight_dict[location][om].keys():
                    job_id[om][sensor_type] = {}
                    # Get sorted list of dates for this om/sensor combination
                    dates = sorted(flight_dict[location][om][sensor_type].keys())
                    for i, date in enumerate(dates):
                        # Initialize job_id dictionary for all steps
                        job_id[om][sensor_type][date] = {key: None for step in uas_pipeline for key in (step, f'{step}_dep')}
                        for step in uas_pipeline:
                            if sensor_type in uas_pipeline[step]['sensor']:
                                if step in flight_dict[location][om][sensor_type][date]:
                                    if flight_dict[location][om][sensor_type][date][step]['status'] in {'validated'}:
                                        print(f'step: {step}')
                                        script_name = f"{om}_{sensor_type}_{date}.sh"
                                        script_path = os.path.join(git_root, uas_pipeline[step]['shell_script_folder'], script_name)
                                        
                                        dependencies = []
                                        # Handle step dependencies
                                        for dependency in uas_pipeline[step]['step_dependency']:
                                            # Only check dependencies that are in our filtered pipeline
                                            if dependency in uas_pipeline:
                                                if job_id[om][sensor_type][date][dependency] is not None:
                                                    dependencies.append(job_id[om][sensor_type][date][dependency])
                                        
                                        # Handle date dependencies
                                        date_dep = uas_pipeline[step]['date_dependency']
                                        if date_dep not in ['previous', 'none']:
                                            raise ValueError(f"Invalid date dependency: {date_dep}")
                                            
                                        if date_dep == 'previous' and i > 0:
                                            # Get the job ID from the previous date's run of this step
                                            prev_date = dates[i-1]
                                            if job_id[om][sensor_type][prev_date][step] is not None:
                                                dependencies.append(job_id[om][sensor_type][prev_date][step])
                                        print(f'dependencies: {dependencies}')

                                        job_id[om][sensor_type][date][step], job_id[om][sensor_type][date][f'{step}_dep'] = submit_job(script_path, working_dir, dependencies, dry_run = dry_run)
                                        if not job_id[om][sensor_type][date][step]:
                                            continue
    return job_id

if __name__ == '__main__':
    # Setup argparse
    parser = argparse.ArgumentParser(description="Process UAS data using a config file.")
    parser.add_argument("--config_file", required=True, help="Path to the YAML config file")
    parser.add_argument("--dry_run", action="store_true", help="Dry run the script")
    parser.add_argument("--date_range", nargs=2, help="Date range to process (YYYYMMDD YYYYMMDD)")
    parser.add_argument("--steps", nargs='+', help="List of steps to run (e.g., step1 step2 step3)")
    parser.add_argument("--flight", help="Flight to run")
    args = parser.parse_args()

    # Load and process configuration
    config = om_funcs.load_config(args.config_file)

    if args.steps:
        if "step1" not in args.steps:
            # Add it temporarily to the beginning of the steps list
            args.steps.insert(0, "step1")
            remove_step1 = True
        else:
            remove_step1 = False
    else:
        remove_step1 = False

    print(f'remove_step1: {remove_step1}')

    # Extract values from config file
    base_folder = config['base_folder']
    om_folder = os.path.join(base_folder, config['om_folder'])
    sensor_dict = config['sensor_dict']
    flight_config_dict = config['flight_list']
    uas_pipeline = config['uas_pipeline']
    dry_run = args.dry_run
    date_range = args.date_range
    flight_filter = args.flight
    # Filter pipeline steps if specific steps are requested
    if args.steps:
        filtered_pipeline = {step: uas_pipeline[step] for step in args.steps if step in uas_pipeline}
        if not filtered_pipeline:
            raise ValueError(f"No valid steps found in {args.steps}. Valid steps are: {list(uas_pipeline.keys())}")
        uas_pipeline = filtered_pipeline

    print(f'uas_pipeline keys in main: {uas_pipeline.keys()}')
    cp0 = time.time()
    working_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.realpath(__file__))

    #make the dictionary of flights and the folders/orthomosaics remaining
    flight_dict = om_funcs.make_dict_bystep(config, uas_pipeline)
    cp1 = time.time()
    print(f'time to create flight_dict: {cp1 - cp0:.2f} seconds')

    print('-' * 100)
    print(f'created flight_dict, now creating omspecific outputfolders')
    cp2 = time.time()
    om_funcs.create_omspecific_output_folders(flight_dict, flight_config_dict, rerun_boundary = False)
    cp3 = time.time()
    print(f'time to create omspecific outputfolders: {cp3 - cp2:.2f} seconds')
    if remove_step1:
        # Remove step1 from the flight_dict
        for flight, value in flight_dict.items():
            for om, value2 in value.items():
                for sensor_type, value3 in value2.items():
                    for date, value4 in value3.items():
                        flight_dict[flight][om][sensor_type][date].pop('step1', None)
        # Remove step1 from the uas_pipeline
        uas_pipeline.pop('step1', None)


    print('-' * 100)
    print(f'filtering flight dictionary')

    flight_dict = om_funcs.filter_flight_dict(flight_dict, flight_filter = flight_filter, date_range = date_range)
    cp4 = time.time()
    print(f'time to filter flight dictionary: {cp4 - cp3:.2f} seconds')
    print('-' * 100)
    print(f'generating shell scripts')

    generate_shell_script(flight_dict, config, uas_pipeline)
    cp5 = time.time()
    print(f'time to generate shell scripts: {cp5 - cp4:.2f} seconds')
    print('-' * 100)
    print(f'running sbatch jobs')

    job_id = run_sbatch(flight_dict, working_dir, uas_pipeline, dry_run = dry_run)
    cp6 = time.time()
    print(f'time to run sbatch jobs: {cp6 - cp5:.2f} seconds')
    print('-' * 100)
    print(f'total time: {cp6 - cp0:.2f} seconds')
    print('-' * 100)

    git_root = get_git_root()

    print(f'git_root: {git_root}')


    # Only export if no filters are placed on flight, steps, or date range
    if not args.flight and not args.steps and not args.date_range:
        # Export job_id to a json file
        with open(os.path.join(git_root, 'profiling/job_id.json'), 'w') as f:

            print(f'os.path.join(git_root, "profiling/job_id.json"): {os.path.join(git_root, "profiling/job_id.json")}')
            json.dump(job_id, f, indent=4)

        # Export flight_dict to a json file
        with open(os.path.join(git_root, 'profiling/flight_dict.json'), 'w') as f:
            print(f'os.path.join(git_root, "profiling/flight_dict.json"): {os.path.join(git_root, "profiling/flight_dict.json")}')
            json.dump(flight_dict, f, indent=4)
    else:
        # Export job_id to a json file
        with open(os.path.join(git_root, '/profiling/job_id_filtered.json'), 'w') as f:
            json.dump(job_id, f, indent=4)

        # Export flight_dict to a json file
        with open(os.path.join(git_root, '/profiling/flight_dict_filtered.json'), 'w') as f:
            json.dump(flight_dict, f, indent=4)
