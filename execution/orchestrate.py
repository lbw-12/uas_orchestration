import os
import pathlib
import argparse
import subprocess
from jinja2 import Environment, FileSystemLoader
import pandas as pd
from shapely.geometry import Point
import re
import shutil

import src.om_funcs as om_funcs

def get_git_root():
    try:
        # Run the Git command and decode the result
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                       stderr=subprocess.STDOUT).strip().decode("utf-8")
        return root
    except subprocess.CalledProcessError:
        raise Exception("This directory is not inside a Git repository or Git is not installed.")


def step1_find(base_folder):

    flights_folder = os.path.join(base_folder, 'flights')
    folders = om_funcs.list_folders_two_levels_deep(flights_folder)
    print(len(folders))
    print(folders[0])

    # Gets folders with valid folders where there is a valid geotagged OUTPUT folder
    valid_folders, invalid_folders = om_funcs.geotagged_folders(folders)

    print(f'number of valid folders: {len(valid_folders)}')
    print(f'number of invalid folders: {len(invalid_folders)}')

    for folder in sorted(invalid_folders):
        folder_only = folder.split('/')[-2]
        print(f'invalid folders: {folder_only}')

    return valid_folders

def step2_find(base_folder, config):
    step2_jobs = []
    om_dir = os.path.join(base_folder, config['om_folder'])
    om_aligned_dir = os.path.join(base_folder, config['om_aligned_folder'])
    
    for om in os.listdir(om_dir):
        # replace .tif with _aligned.tif
        om_aligned = om.replace('.tif', '_aligned.tif')
        if not os.path.exists(os.path.join(om_aligned_dir, om_aligned)):
            step2_jobs.append(om)
    for job in step2_jobs:
        print(f'step2_job: {job}')
    print(f'step2_jobs: {len(step2_jobs)}')
    return step2_jobs

def step3_find(base_folder, config, om_list):
    step3_jobs = []
    om_aligned_dir = os.path.join(base_folder, config['om_aligned_folder'])
    plottiles_base_dir = os.path.join(base_folder, config['plottiles_folder'])
    
    for om_sensor_date in os.listdir(om_aligned_dir):
        om_sensor_date = om_sensor_date.replace('.tif', '')
        for om in om_list:
            if om in om_sensor_date:
                om_location = om
        for sensor_type in config['sensor_dict'].keys():
            if sensor_type in om_sensor_date:
                om_sensor_type = sensor_type
        # Search om_sensor_date for 8 digit date
        om_date = re.search(r'\d{8}', om_sensor_date).group()

        print(f'step3_sensor_type: {om_sensor_type}')
        print(f'step3_location: {om_location}')
        print(f'step3_date: {om_date}')

        plottiles_dir = os.path.join(plottiles_base_dir, f'plot_tiles_{om_sensor_type}_om')

        plottiles_files = []
        for file in sorted(os.listdir(plottiles_dir)):
            if file.endswith('.tif'):
                if om_sensor_type in file and om_location in file and om_date in file:
                    plottiles_files.append(file)
        if len(plottiles_files) < 50:
            step3_jobs.append(om)

    for job in step3_jobs:
        print(f'step3_job: {job}')
    print(f'step3_jobs: {len(step3_jobs)}')
    return step3_jobs

def step4_5_find(config, flight_dict):
    base_folder = config['base_folder']
    output_path_geo_template = os.path.join(base_folder, config['uas_pipeline']['step4']['output_folder'])
    output_path_ir_template = os.path.join(base_folder, config['uas_pipeline']['step5']['output_folder'])

    step4_jobs = []
    step5_jobs = []
    for location, value in flight_dict.items():
        for om, value2 in value.items():
            for sensor_type, value3 in value2.items():
                for date, folder_path in value3.items():
                    if sensor_type in config['uas_pipeline']['step4']['sensor']:
                        output_path_geo = output_path_geo_template.format(om=om, sensor_type=sensor_type, date=date)
                        output_path_ir = output_path_ir_template.format(om=om, sensor_type=sensor_type, date=date)
                        if not os.path.exists(output_path_geo):
                            step4_jobs.append(f'{om}_{sensor_type}_{date}')
                        elif os.path.exists(output_path_geo):
                            # Count the number of .tif files in the output_path_geo
                            num_tif_files = len([f for f in os.listdir(output_path_geo) if f.endswith('.tif')])
                            # Count the number of .tif or .jpg files in the folder_path
                            num_tif_jpg_files = len([f for f in os.listdir(folder_path) if f.endswith('.tif') or f.endswith('.jpg')])
                            if num_tif_jpg_files - num_tif_files > 25:
                                step4_jobs.append(f'{om}_{sensor_type}_{date}')
                        if not os.path.exists(output_path_ir):
                            step5_jobs.append(f'{om}_{sensor_type}_{date}')
                        elif os.path.exists(output_path_ir):
                            # Count the number of .tif files in the output_path_ir
                            num_tif_files = len([f for f in os.listdir(output_path_ir) if f.endswith('.tif')])
                            if num_tif_files < 10:
                                step5_jobs.append(f'{om}_{sensor_type}_{date}')

    for job in step4_jobs:
        print(f'step4_job: {job}')
    print(f'step4_jobs: {len(step4_jobs)}')
    for job in step5_jobs:
        print(f'step5_job: {job}')
    print(f'step5_jobs: {len(step5_jobs)}')
    return step4_jobs, step5_jobs




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
    env = Environment(loader=FileSystemLoader(script_dir))

    # Only generate scripts for the steps in our filtered pipeline
    for step in uas_pipeline:
        count = 0
        template = env.get_template(uas_pipeline[step]['shell_script_template'])
        log_folder = os.path.join(processing_dir, uas_pipeline[step]['log_folder'])
        os.makedirs(log_folder, exist_ok = True)
        for flight, value in flight_dict.items():
            for om, value2 in value.items():
                #print(f'flight: {flight}, om: {om}')
                for sensor_type, value3 in value2.items():
                    # print(f'sensor_type: {sensor_type}')
                    for date, folder_path in value3.items():
                        # print(f'date: {date}')
                        if sensor_type in uas_pipeline[step]['sensor']:
                            # Generate parameters for othomsosaic creation shell script
                            if os.path.exists(f'{folder_path}OUTPUT_{om}'):
                                img_dir = f'{folder_path}OUTPUT_{om}/'
                            else:
                                img_dir = f'{folder_path}OUTPUT/'

                            processing_subdir = pathlib.Path(f'{processing_dir}/{om}_{sensor_type}_{date}')
                            #print(processing_subdir)

                            # Generate unique parameters for plot tile creation shell script
                            plotimage_source = "om"
                            corn_relpath = config.get('plot_shapefiles', {}).get(om, {}).get('corn')
                            soy_relpath = config.get('plot_shapefiles', {}).get(om, {}).get('soy')
                            corn_path = os.path.join(base_folder, corn_relpath) if corn_relpath else ''
                            soy_path = os.path.join(base_folder, soy_relpath) if soy_relpath else ''
                            

                            output_folder_plottiles_template = config['uas_pipeline']['step3']['output_folder']
                            output_folder_plottiles = output_folder_plottiles_template.format(om=om, sensor_type=sensor_type, date=date, source = "om")
                            output_path_plottiles = os.path.join(base_folder, output_folder_plottiles)


                            # Generate unique parameters for DGR creation shell script
                            flight_dir_dgr = "/".join(folder_path.split("/")[:-3])

                            if "rgb" in sensor_type:
                                allotted_time = len(os.listdir(img_dir))
                            else:
                                allotted_time = len(os.listdir(img_dir))
                            allotted_time_h = int(allotted_time // 60)
                            allotted_time_m = int(allotted_time % 60)
                            flight_name = flight_dir_dgr.split("/")[-2]

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

                            # Generate unique parameters for Maptiles creation shell script
                            maptiles_dir = pathlib.Path(f'{maptiles_path}/{om}/{date}')
                            maptiles_base_dir = pathlib.Path(config["maptiles_base_folder"])

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
                            output_path_gs = output_path_gs_template.format(om=om, sensor_type=sensor_type, date=date)
                            output_path_cc_json_template = os.path.join(base_folder, config['uas_pipeline']['step8']['output_folder'], config['uas_pipeline']['step8']['output_file'])
                            output_path_cc_json = output_path_cc_json_template.format(om=om, sensor_type=sensor_type, date=date)
                            output_path_sr_template = os.path.join(base_folder, config['uas_pipeline']['step9']['output_folder'], config['uas_pipeline']['step9']['output_file'])
                            output_path_sr = os.path.join(base_folder, config['uas_pipeline']['step9']['output_folder'])
                            output_path_sr_json = output_path_sr_template.format(om=om, sensor_type=sensor_type, date=date) 
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
                                "img_dir": img_dir, #step 1
                                "processing_subdir": processing_subdir, #step 1
                                "script_dir": script_dir, #step 1
                                "script_name": f'single_job_{sensor_type}.sh', #step 1
                                "shapefiles_alignment_path": shapefiles_alignment_path, #step 2
                                "om_folder": om_dir, #step 2
                                "plotimage_source": plotimage_source, #step 3
                                "shapefile_path_corn": corn_path, #step 3
                                "shapefile_path_soy": soy_path, #step 3
                                "output_path_plottiles": output_path_plottiles, #step 3
                                "flight_dir": flight_dir_dgr, #step 4 This could use the folder_path if the script was updated
                                "time": f'{allotted_time_h}:{allotted_time_m}:00', # Step 4
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
                                print(f'Generated: {f_path}')

                            os.chmod(f_path, 0o770)
                            count += 1

        print(f'Generated {count} shell scripts for step: {step}')

def submit_job(script_path, working_dir, dependency=None, dry_run = False):
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return None if not dry_run else "dry_run_job_id"

    command = ["sbatch"]
    if dependency:
        # Handle both single dependency and list of dependencies
        if isinstance(dependency, list):
            # Join multiple dependencies with ':'
            dep_str = ':'.join(str(d) for d in dependency if d is not None)
            if dep_str:  # Only add dependency if we have valid dependencies
                command.append(f"--dependency=afterok:{dep_str}")
        else:
            command.append(f"--dependency=afterok:{dependency}")
    command.append(script_path)

    # Print the command to show what would be run
    print(f"{'Dry run:' if dry_run else 'Submitting:'} {' '.join(command)}")

    # If dry_run is enabled, skip actually running the command
    if dry_run:
        return "dry_run_job_id"

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
        return match.group(1) if match else None
    else:
        print(f"Error submitting {script_path}: {result.stderr.strip()}")
        return None

# Generates individual sbatch commands for all dates (where an existing orthomosaic is not already present) for a given list of 
# locations and sensor types. If no list of locations or sensor types is provided, all will be used.
def run_sbatch(flight_dict, working_dir, uas_pipeline, dry_run = False):
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
                        job_id[om][sensor_type][date] = {step: None for step in uas_pipeline}
                        for step in uas_pipeline:
                            print(f'step: {step}')
                            if sensor_type in uas_pipeline[step]['sensor']:
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

                                job_id[om][sensor_type][date][step] = submit_job(script_path, working_dir, dependencies, dry_run = dry_run)
                                if not job_id[om][sensor_type][date][step]:
                                    continue

if __name__ == '__main__':
    # Setup argparse
    parser = argparse.ArgumentParser(description="Process UAS data using a config file.")
    parser.add_argument("--config_file", required=True, help="Path to the YAML config file")
    parser.add_argument("--dry_run", action="store_true", help="Dry run the script")
    parser.add_argument("--regen_shell_scripts", action="store_true", help="Regenerate all possible shell scripts")
    parser.add_argument("--date_range", nargs=2, help="Date range to process (YYYYMMDD YYYYMMDD)")
    parser.add_argument("--steps", nargs='+', help="List of steps to run (e.g., step1 step2 step3)")
    parser.add_argument("--location", help="Location to run")
    args = parser.parse_args()

    # Load and process configuration
    config = om_funcs.load_config(args.config_file)

    # Extract values from config file
    base_folder = config['base_folder']
    om_folder = os.path.join(base_folder, config['om_folder'])
    sensor_dict = config['sensor_dict']
    flight_config_dict = config['flight_list']
    uas_pipeline = config['uas_pipeline']
    dry_run = args.dry_run
    regen_shell_scripts = args.regen_shell_scripts
    # If regenerating shell scripts, we need to run in dry_run mode to not overwrite existing data outputs.
    #if regen_shell_scripts:
    #    dry_run = True

    om_list = []
    for location in flight_config_dict.keys():
        for om in flight_config_dict[location].keys():
            om_list.append(om)
    print(f'om_list: {om_list}')

    
    date_range = args.date_range
    location_filter = args.location
    # Filter pipeline steps if specific steps are requested
    if args.steps:
        filtered_pipeline = {step: uas_pipeline[step] for step in args.steps if step in uas_pipeline}
        if not filtered_pipeline:
            raise ValueError(f"No valid steps found in {args.steps}. Valid steps are: {list(uas_pipeline.keys())}")
        uas_pipeline = filtered_pipeline

    print(f'uas_pipeline keys in main: {uas_pipeline.keys()}')

    working_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Run the processing - find the folders that have the right data structure/output folder
    valid_folders = step1_find(base_folder)

    step2_jobs = step2_find(base_folder, config)

    step3_jobs = step3_find(base_folder, config, om_list)


    #make the dictionary of flights and the folders/orthomosaics remaining
   
    flight_dict, count, folders_not_matched, folders_multiple, om_remaining, folders_remaining = om_funcs.make_dict(valid_folders, flight_config_dict, sensor_dict, om_folder, regen_shell_scripts, date_range, location_filter = location_filter)
    
    step4_jobs, step5_jobs = step4_5_find(config, flight_dict)




    om_funcs.create_omspecific_output_folders(flight_dict, flight_config_dict, rerun_boundary = False)

    generate_shell_script(flight_dict, config, uas_pipeline)


    run_sbatch(flight_dict, working_dir, uas_pipeline, dry_run = dry_run)

