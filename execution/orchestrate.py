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
import pathlib
import boto3
import os
import pprint # Used for pretty-printing the job parameters in dry runs
from google.cloud import batch_v1
import datetime


def get_project_root():
    """
    Returns the absolute root path of the project, adapting to whether
    it's running in a Docker container or a local Git repository.
    """
    # Check if a known container directory exists. This is a simple and
    # reliable way to detect if we are inside the container.
    container_root = "/app"
    if os.path.exists(container_root):
        return container_root
    else:
        # If not in a container, fall back to the git command for the local/HPC environment.
        try:
            root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                           stderr=subprocess.STDOUT).strip().decode("utf-8")
            return root
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise Exception("Could not determine project root. Not in a container and not in a Git repository.")


def json_path_converter(o):
    """
    A helper function to tell the JSON encoder how to handle
    pathlib.Path objects. It converts them to strings.
    """
    if isinstance(o, pathlib.Path):
        # If the object is a Path, convert it to its string representation
        return str(o)

    # For any other type, let the default encoder raise the error
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

# This creates the individual shell scripts to generate an orthomosaic for a flight, sensor, and date
def prepare_jobs(flight_dict, filtered_flight_dict, config, uas_pipeline, platform):
    # Extract values from config file
    project_root = get_project_root()
    base_folder = config['base_folder']
    om_dir = os.path.join(base_folder, config['om_folder'])
    om_aligned_folder = os.path.join(base_folder, config['om_aligned_folder'])
    processing_dir = os.path.join(base_folder, config['processing_folder'])
    shapefiles_alignment_path = os.path.join(base_folder, config['shapefiles_alignment'])
    logdir_perf = os.path.join(processing_dir,"logs_perf")
    pub_folder_dict = config['publishing_folder']
    os.makedirs(logdir_perf, exist_ok = True)

    gcp_tasks = []

    # Get the absolute path to the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up Jinja2 to look for templates in the same directory
    env = Environment(loader=FileSystemLoader(project_root))

    # Only generate scripts for the steps in our filtered pipeline
    for step in uas_pipeline:
        count = 0
        print(f'path to template: {uas_pipeline[step]["shell_script_template"]}')
        template = env.get_template(uas_pipeline[step]['shell_script_template'])
        log_folder = os.path.join(processing_dir, uas_pipeline[step]['log_folder'])
        resources = uas_pipeline[step].get('resources', {})
        os.makedirs(log_folder, exist_ok = True)
        for flight, value in filtered_flight_dict.items():
            for om, value2 in value.items():
                #print(f'flight: {flight}, om: {om}')
                for sensor_type, value3 in value2.items():
                    # print(f'sensor_type: {sensor_type}')
                    for date, value4 in value3.items():
                        # print(f'date: {date}')
                        if sensor_type in uas_pipeline[step]['sensor']:
                            if step in filtered_flight_dict[flight][om][sensor_type][date]:
                                if filtered_flight_dict[flight][om][sensor_type][date][step]['status'] in ['validated', 'not_ready']:

                                    input_path = flight_dict[flight][om][sensor_type][date][step]['input_path'] # was img_dir and folder_path
                                    csv_folder_path = flight_dict[flight][om][sensor_type][date]['step1']['input_path']
                                    processing_subdir = pathlib.Path(f'{processing_dir}/{om}_{sensor_type}_{date}')

                                    # Generate unique parameters for plot tile creation shell scripts for om and ir
                                    if 'source' in uas_pipeline[step]:
                                        plotimage_source = uas_pipeline[step]['source']
                                        if 'plottile' in uas_pipeline[step]['shell_script_template']:
                                            output_folder_plottiles_template = config['uas_pipeline'][step]['output_folder']
                                            output_folder_plottiles = output_folder_plottiles_template.format(om=om, sensor_type=sensor_type, date=date, source = plotimage_source)
                                            output_path_plottiles = os.path.join(base_folder, output_folder_plottiles)
                                        else:
                                            output_path_plottiles = 'na'
                                    else:
                                        plotimage_source = 'na'
                                        output_path_plottiles = 'na'

                                    #corn_relpath = config.get('plot_shapefiles', {}).get(om, {}).get('corn')
                                    #soy_relpath = config.get('plot_shapefiles', {}).get(om, {}).get('soy')
                                    #corn_path = os.path.join(base_folder, corn_relpath) if corn_relpath else ''
                                    #soy_path = os.path.join(base_folder, soy_relpath) if soy_relpath else ''

                                    plot_shapefiles = config.get('plot_shapefiles', {}).get(om, {})
                                    keys = list(plot_shapefiles.keys())

                                    plottiles_relpath1 = plot_shapefiles.get(keys[0]) if len(keys) > 0 else None
                                    plottiles_relpath2 = plot_shapefiles.get(keys[1]) if len(keys) > 1 else None

                                    crop1 = keys[0] if len(keys) > 0 else None
                                    crop2 = keys[1] if len(keys) > 1 else None

                                    shapefile_path1 = os.path.join(base_folder, plottiles_relpath1) if plottiles_relpath1 else ''
                                    shapefile_path2 = os.path.join(base_folder, plottiles_relpath2) if plottiles_relpath2 else ''


                                    # Generate unique parameters for Image Registration shell script
                                    # Make a tag to put in batch script name, to identify invalid jobs
                                    scriptnametag=''

                                    # Create path variables for DGR and IR process only if step4 and step5 exist in uas_pipeline
                                    if 'step4' in uas_pipeline and 'step5' in uas_pipeline:

                                        output_path_geo_template = os.path.join(base_folder, config['uas_pipeline']['step4']['output_folder'])
                                        output_path_geo = output_path_geo_template.format(om=om, sensor_type=sensor_type, date=date)
                                        elevation_base_folder = config['elevation_base_folder']
                                        elevation_flight_folders = config['elevation_flight_folders']
                                        elevation_folder = os.path.join(elevation_base_folder, elevation_flight_folders[flight]['folder'])
                                        geoid_height = elevation_flight_folders[flight]['geoid_height']
                                        output_path_ir_template = os.path.join(base_folder, config['uas_pipeline']['step5']['output_folder'])
                                        output_path_ir = output_path_ir_template.format(om=om, sensor_type=sensor_type, date=date)

                                        ortho_path = os.path.join(f'{om_aligned_folder}',f'{om}_{sensor_type}_{date}_aligned.tif')

                                    else:
                                        output_path_geo = 'na'
                                        output_path_ir = 'na'
                                        elevation_folder = 'na'
                                        geoid_height = 'na'
                                        ortho_path = 'na'

                                    #Step7: Create path variables for om plot patches
                                    if 'plot_to_patch' in uas_pipeline[step]['shell_script_template']:
                                        output_path_plot_patches_template = os.path.join(base_folder, config['uas_pipeline'][step]['output_folder'])
                                        output_path_plot_patches = output_path_plot_patches_template.format(om=om, source=plotimage_source, sensor_type=sensor_type, date=date)
                                    else:
                                        output_path_plot_patches = 'na'

                                    job_title = f'{om}_{sensor_type}_{date}'

                                    # Generate parameters for growth stage, canopy cover, and spectral reflectance inference shell scripts
                                    #gs_input_dir = os.path.join(base_folder, output_path_plot_patches)
                                    #cc_input_dir = os.path.join(base_folder, config['plottiles_folder'], 'plot_tiles_rgb_om/')
                                    #sr_input_dir = os.path.join(base_folder, config['plottiles_folder'], 'plot_tiles_multispectral_om/')

                                    #model_output_dir = os.path.join(base_folder, config['uas_pipeline']['step9']['output_folder'])
                                    #os.makedirs(model_output_dir, exist_ok=True) 
                                    model_path = uas_pipeline[step]['model_path']

                                    if 'inf_gs' in uas_pipeline[step]['shell_script_template']:
                                        output_path_gs_template = os.path.join(base_folder, config['uas_pipeline'][step]['output_folder'], config['uas_pipeline'][step]['output_file'])
                                        output_path_gs = output_path_gs_template.format(om=om, source=plotimage_source, date=date)
                                        step_dependency = uas_pipeline[step]['step_dependency'][0]
                                        gs_input_dir_template = os.path.join(base_folder, uas_pipeline[step_dependency]['output_folder'])
                                        gs_input_dir = gs_input_dir_template.format(om=om, sensor_type=sensor_type, date=date, source = plotimage_source)
                                    else:
                                        gs_input_dir = 'na'
                                        output_path_gs = 'na'


                                    if 'inf_cc' in uas_pipeline[step]['shell_script_template']:
                                        output_path_cc_json_template = os.path.join(base_folder, config['uas_pipeline'][step]['output_folder'], config['uas_pipeline'][step]['output_file'])
                                        output_path_cc_json = output_path_cc_json_template.format(om=om, source=plotimage_source, date=date)
                                        step_dependency = uas_pipeline[step]['step_dependency'][0]
                                        cc_input_dir_template = os.path.join(base_folder, uas_pipeline[step_dependency]['output_folder'])
                                        cc_input_dir = cc_input_dir_template.format(om=om, sensor_type=sensor_type, date=date, source = plotimage_source)
                                    else:
                                        cc_input_dir = 'na'
                                        output_path_cc_json = 'na'

                                    if 'inf_sr' in uas_pipeline[step]['shell_script_template']:
                                        output_path_sr_template = os.path.join(base_folder, config['uas_pipeline']['step11']['output_folder'], config['uas_pipeline']['step11']['output_file'])
                                        output_path_sr = os.path.join(base_folder, config['uas_pipeline']['step9']['output_folder'])
                                        output_path_sr_json = output_path_sr_template.format(om=om, source=plotimage_source, date=date)
                                        step_dependency = uas_pipeline[step]['step_dependency'][0]
                                        sr_input_dir_template = os.path.join(base_folder, uas_pipeline[step_dependency]['output_folder'])
                                        sr_input_dir = sr_input_dir_template.format(om=om, sensor_type=sensor_type, date=date, source = plotimage_source)
                                    else:
                                        sr_input_dir = 'na'
                                        output_path_sr = 'na'
                                        output_path_sr_json = 'na'

                                    output_folder_geojson_template = os.path.join(base_folder, config['uas_pipeline']['step14']['output_folder'], config['uas_pipeline']['step14']['output_file'])
                                    output_folder_geojson = output_folder_geojson_template.format(year = '2025',om=om, date=date)

                                    #IR:  Generate parameters for growth stage, canopy cover, and spectral reflectance inference shell scripts 
                                    #cc_input_dir_ir = os.path.join(base_folder, config['plottiles_folder'], 'plot_tiles_rgb_ir/')

                                    #model_output_dir = os.path.join(base_folder, config['uas_pipeline']['step12']['output_folder'])
                                    #os.makedirs(model_output_dir, exist_ok=True) 


                                    #output_path_gs_template_ir = os.path.join(base_folder, config['uas_pipeline']['step12']['output_folder'], config['uas_pipeline']['step12']['output_file'])
                                    #output_path_gs_ir = output_path_gs_template.format(om=om, source="ir", date=date)
                                    #output_path_cc_json_template_ir = os.path.join(base_folder, config['uas_pipeline']['step13']['output_folder'], config['uas_pipeline']['step13']['output_file'])
                                    #output_path_cc_json_ir = output_path_cc_json_template.format(om=om, source="ir", date=date)
                                    
                                    if 'step9' in uas_pipeline:
                                        json_inference_dir = uas_pipeline['step9']['output_folder']
                                    else:
                                        json_inference_dir = 'na'

                                    output_folder = os.path.join(base_folder, config['uas_pipeline'][step]['output_folder'])

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
                                        "output_folder": output_folder, #common
                                        "resources": resources, #common
                                        "om_aligned_folder": om_aligned_folder, #step 2
                                        "om": om,  # step 1,2
                                        "processing_dir": processing_dir, #step 1
                                        "om_dir": om_dir, #step 1
                                        "input_path": input_path, #common
                                        "processing_subdir": processing_subdir, #step 1
                                        "script_dir": script_dir, #step 1
                                        "script_name": f'single_job_{sensor_type}.sh', #step 1
                                        "shapefiles_alignment_path": shapefiles_alignment_path, #step 2
                                        "om_folder": om_dir, #step 2
                                        "plotimage_source": plotimage_source, #step 3
                                        "shapefile_path1": shapefile_path1, #step 3
                                        "shapefile_path2": shapefile_path2, #step 3
                                        "output_path_plottiles": output_path_plottiles, #step 3
                                        "crop1": crop1,
                                        "crop2": crop2,
                                        "output_path_geo": output_path_geo,
                                        "output_path_ir": output_path_ir,
                                        "elevation_folder": elevation_folder,
                                        "geoid_height": geoid_height,
                                        "ortho_path": ortho_path,
                                        "csv_folder_path": csv_folder_path,
                                        "output_path_plot_patches": output_path_plot_patches,
                                        "maptiles_dir": pub_folder,
                                        "gs_input_dir": gs_input_dir,
                                        "cc_input_dir": cc_input_dir,
                                        "sr_input_dir": sr_input_dir,
                                        "model_path": model_path,
                                        "json_inference_dir": json_inference_dir,
                                        "output_path_gs": output_path_gs,
                                        "output_path_cc_json": output_path_cc_json,
                                        "output_path_sr": output_path_sr,
                                        "output_path_sr_json": output_path_sr_json,
                                        "output_folder_geojson": output_folder_geojson
                                    }

                                    filtered_flight_dict[flight][om][sensor_type][date][step]['data'] = data_step


                                    if platform == 'slurm':
                                        output = template.render(data_step)
                                        if not os.path.exists(os.path.join(project_root, uas_pipeline[step]['shell_script_folder'])):
                                            os.makedirs(os.path.join(project_root, uas_pipeline[step]['shell_script_folder']))
                                        f_path = f"{os.path.join(project_root, uas_pipeline[step]['shell_script_folder'])}/{job_title}{scriptnametag}.sh"
                                        with open(f_path, 'w') as f:
                                            f.write(output)
                                            #print(f'Generated: {f_path}')

                                        os.chmod(f_path, 0o770)
                                        count += 1

                                    elif platform == 'gcp':

                                        # --- GCP Action: Create and collect a task dictionary ---
                                        env_vars = {key: str(value) for key, value in data_step.items()}
                                        worker_script = uas_pipeline[step]['worker_script']
                                        requires_gpu = uas_pipeline[step].get('resources', {}).get('gpu', False)
                                        
                                        task = {
                                            'id': data_step['job_title'] + "_" + step,
                                            'script_to_run': worker_script,
                                            'gpu': requires_gpu,
                                            'env_vars': env_vars,
                                            'dependencies': [] # Dependencies will be calculated later
                                        }
                                        gcp_tasks.append(task)

        print(f'Generated {count} shell scripts for step: {step}')

    if platform == 'gcp':
        print(f'Collected {len(gcp_tasks)} GCP tasks')
        # Note: Actual submission happens later in submit_gcp_dag()
        return gcp_tasks
    elif platform == 'slurm':
        return None

def submit_slurm_job(script_path, working_dir, dependency=None, dry_run = False):
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
        if not hasattr(submit_slurm_job, "dry_run_counter"):
            submit_slurm_job.dry_run_counter = 0
        
        # Increment the counter and format the dummy job ID
        submit_slurm_job.dry_run_counter += 1
        dummy_job_id = f"dr_{submit_slurm_job.dry_run_counter:04d}"
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
    
def submit_aws_job(batch_client, job_name, job_queue, job_definition, command, dependency=None, dry_run=False):
    """
    Submits a job to AWS Batch and returns the job ID.

    Args:
        batch_client: An initialized boto3 client for the 'batch' service.
        job_name (str): The name to give the AWS Batch job.
        job_queue (str): The name of the AWS Batch job queue.
        job_definition (str): The name of the AWS Batch job definition.
        command (list): The command to run in the container (e.g., ['bash', 'script.sh']).
        dependency (str or list, optional): A single job ID or a list of job IDs this job depends on.
        dry_run (bool): If True, prints the job parameters without submitting.

    Returns:
        str: The job ID if submission is successful, otherwise None.
    """
    # Base parameters for the AWS Batch job submission
    job_params = {
        'jobName': job_name,
        'jobQueue': job_queue,
        'jobDefinition': job_definition,
        'containerOverrides': {
            'command': command
        }
    }

    # --- Dependency Logic ---
    # Translate the Slurm dependency format to the AWS Batch format
    if dependency:
        # Ensure dependency is a list
        if not isinstance(dependency, list):
            dependency = [dependency]
        
        # Filter out any None or invalid values
        valid_dependencies = [str(d) for d in dependency if d]
        
        if valid_dependencies:
            # AWS expects a list of dictionaries
            job_params['dependsOn'] = [{'jobId': dep_id, 'type': 'N_TO_N'} for dep_id in valid_dependencies]

    # --- Dry Run Logic ---
    if dry_run:
        print("--- Dry Run: Job Parameters ---")
        pprint.pprint(job_params)
        
        # Mimic returning a dummy job ID for chaining in dry runs
        if not hasattr(submit_aws_job, "dry_run_counter"):
            submit_aws_job.dry_run_counter = 0
        submit_aws_job.dry_run_counter += 1
        dummy_job_id = f"dr_{submit_aws_job.dry_run_counter:04d}"
        print(f"--- Would return job ID: {dummy_job_id} ---\n")
        return dummy_job_id

    # --- Real Submission Logic ---
    try:
        print(f"Submitting job '{job_name}'...")
        response = batch_client.submit_job(**job_params)
        job_id = response['jobId']
        print(f"✅ Success! Submitted job '{job_name}' with ID: {job_id}\n")
        return job_id
    except Exception as e:
        print(f"❌ Error submitting job '{job_name}': {e}\n")
        return None
    
def build_gcp_dag(data_step,script_path, dependencies, uas_pipeline, step):
    """
    Collects the information for a single task in the GCP DAG.
    Returns a dictionary representing the task, not a real job ID.
    """
    # Convert all values in the data_step dict to strings for the environment
    env_vars = {key: str(value) for key, value in data_step.items()}

    task_info = {
        'id': data_step['job_title'],
        'script_to_run': 'execution/' + uas_pipeline[step]['worker_script'],
        'dependencies': dependencies,
        'gpu': data_step['resources'].get('gpu', False),
        'env_vars': env_vars # Use the new, clean dictionary
    }
    
    # We return the task_id to be stored for subsequent dependency lookups
    # and the full task_info dictionary itself.
    return data_step['job_title'], task_info

def submit_gcp_dag(job_id_dict, config, dry_run=False):
    """
    Parses the collected job data, builds the full DAG, and submits it to GCP.
    """
    tasks = []
    task_map = {}  # Maps a task's unique ID to its index in the 'tasks' list

    # --- 1. Flatten the nested dictionary into a simple list of tasks ---
    # First pass: Collect all unique tasks and map their IDs to an index.
    for om in job_id_dict.values():
        for sensor in om.values():
            for date in sensor.values():
                for step, task_info in date.items():
                    # Ensure it's a valid task dictionary and not a dependency placeholder
                    if task_info and isinstance(task_info, dict):
                        task_id = task_info.get('id')
                        if task_id and task_id not in task_map:
                            task_map[task_id] = len(tasks)
                            tasks.append(task_info)

    if not tasks:
        print("No tasks to submit.")
        return

    # --- 2. Build the dependency map using indices ---
    # Second pass: Create a dictionary mapping an index to a list of its prerequisite indices.
    dependencies = {}
    for i, task in enumerate(tasks):
        dep_indices = []
        for dep_id in task['dependencies']:
            if dep_id in task_map:
                dep_indices.append(task_map[dep_id])
        dependencies[i] = dep_indices

    # --- 3. Render the Jinja2 Template ---
    project_root = get_project_root() # Assuming you have this helper function
    template_loader = FileSystemLoader(searchpath=os.path.join(project_root, "templates"))
    template_env = Environment(loader=template_loader)
    template = template_env.get_template("gcp_job.json.j2")
    
    service_account = config['project_settings']['gcp_service_account']
    machine_type = config['project_settings']['gcp_machine_type']

    json_payload_str = template.render(
        tasks=tasks,
        dependencies=dependencies,
        service_account_email=service_account,
        machine_type=machine_type
    )

    if dry_run:
        print("--- Dry Run: GCP Job JSON Payload ---")
        print(json_payload_str)
        return "dry_run_job_name"

    # --- 4. Submit the Job to the GCP API ---
    client = batch_v1.BatchServiceClient()
    job_payload = batch_v1.Job.from_json(json_payload_str)
    
    # Create a unique job ID for this run
    job_name = "uas-workflow-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parent = f"projects/{config['project_settings']['gcp_project_id']}/locations/us-central1"
    
    try:
        print(f"Submitting GCP Batch job '{job_name}'...")
        request = batch_v1.CreateJobRequest(parent=parent, job_id=job_name, job=job_payload)
        response = client.create_job(request=request)
        print(f"✅ Success! Submitted job with name: {response.name}")
        return response.name
    except Exception as e:
        print(f"❌ Error submitting GCP Batch job: {e}")
        return None

# Generates individual sbatch commands for all dates for a given list of 
# locations and sensor types. If no list of locations or sensor types is provided, all will be used.
def run_batch(flight_dict, working_dir, uas_pipeline, platform,dry_run = False, location_filter = None, date_range = None):
    project_root = get_project_root()
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
                                    if flight_dict[location][om][sensor_type][date][step]['status'] in {'validated', 'not_ready'}:
                                        print(f'step: {step}')
                                        script_name = f"{om}_{sensor_type}_{date}.sh"
                                        script_path = os.path.join(project_root, uas_pipeline[step]['shell_script_folder'], script_name)
                                        
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

                                        if platform == 'gcp':
                                            job_id[om][sensor_type][date][step], job_id[om][sensor_type][date][f'{step}_dep'] = build_gcp_dag(data_step, script_path, dependencies, uas_pipeline, step)
                                        elif platform == 'slurm':
                                            job_id[om][sensor_type][date][step], job_id[om][sensor_type][date][f'{step}_dep'] = submit_slurm_job(script_path, working_dir, dependencies, dry_run = dry_run)
                                        elif platform == 'aws':
                                            job_id[om][sensor_type][date][step], job_id[om][sensor_type][date][f'{step}_dep'] = submit_aws_job(batch_client, script_path, working_dir, dependencies, dry_run = dry_run)
                                        else:
                                            raise ValueError(f"Invalid platform: {platform}")

                                        if not job_id[om][sensor_type][date][step]:
                                            continue
    if platform == 'gcp':
        # Submit the GCP DAG
        print(f'submitting GCP DAG')
        submit_gcp_dag(job_id, config, dry_run = dry_run)
    return job_id

if __name__ == '__main__':
    # Setup argparse
    parser = argparse.ArgumentParser(description="Process UAS data using a config file.")
    parser.add_argument("--config_file", required=True, help="Path to the YAML config file")
    parser.add_argument("--platform", required=True, help = "slurm, gcp, or aws")
    parser.add_argument("--dry_run", action="store_true", help="Dry run the script")
    parser.add_argument("--date_range", nargs=2, help="Date range to process (YYYYMMDD YYYYMMDD)")
    parser.add_argument("--steps", nargs='+', help="List of steps to run (e.g., step1 step2 step3)")
    parser.add_argument("--flight", help="Flight to run")
    parser.add_argument("--file_age", type=int, default=3600, help="File age in seconds")
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
    date_range = args.date_range
    flight_filter = args.flight
    file_age = args.file_age
    platform = args.platform
    # Filter pipeline steps if specific steps are requested

    cp0 = time.time()
    working_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.realpath(__file__))

    #make the dictionary of flights and the folders/orthomosaics remaining
    flight_dict = om_funcs.make_dict_bystep(config, uas_pipeline, confirm_action = False, file_age = file_age)
    cp1 = time.time()
    print(f'time to create flight_dict: {cp1 - cp0:.2f} seconds')

    print('-' * 100)
    print(f'created flight_dict, now creating omspecific outputfolders')
    cp2 = time.time()
    om_funcs.create_omspecific_output_folders(flight_dict, flight_config_dict, rerun_boundary = False, confirm_action = False)
    cp3 = time.time()
    print(f'time to create omspecific outputfolders: {cp3 - cp2:.2f} seconds')


    print('-' * 100)
    print(f'filtering flight dictionary')

    flight_dict = om_funcs.filter_flight_dict(flight_dict, flight_filter = flight_filter, date_range = date_range)

    flight_dict = om_funcs.validate_flight_dict(flight_dict, config)

    project_root = get_project_root()
    print(f'project_root: {project_root}')

    # Export flight_dict to a json file before filtering
    with open(os.path.join(project_root, 'profiling/flight_dict.json'), 'w') as f:
        json.dump(flight_dict, f, indent=4, default=json_path_converter)


    # 4. FILTER BY STEP (THE FINAL ACTION)
    # Now that validation is done, filter both the dict and pipeline 
    # down to only what the user wants to generate scripts for.
    if args.steps:
        print(f"--- Filtering for requested steps: {args.steps} ---")
        
        # Filter the pipeline
        final_pipeline = {step: uas_pipeline[step] for step in args.steps if step in uas_pipeline}
        if not final_pipeline:
            raise ValueError(f"No valid steps found in {args.steps}.")
        
        # Filter the flight_dict to match the final pipeline
        filtered_flight_dict = {}
        for flight, flight_data in flight_dict.items():
            filtered_flight_dict[flight] = {}
            for om, om_data in flight_data.items():
                filtered_flight_dict[flight][om] = {}
                for sensor, sensor_data in om_data.items():
                    filtered_flight_dict[flight][om][sensor] = {}
                    for date, date_data in sensor_data.items():
                        # Keep only the steps that are in the final pipeline
                        filtered_steps = {step: data for step, data in date_data.items() if step in final_pipeline}
                        if filtered_steps:
                             filtered_flight_dict[flight][om][sensor][date] = filtered_steps

        #uas_pipeline = final_pipeline # Now overwrite the main pipeline variable

    else:
        filtered_flight_dict = flight_dict
        final_pipeline = uas_pipeline

    cp4 = time.time()
    print(f'time to filter flight dictionary: {cp4 - cp3:.2f} seconds')
    print('-' * 100)
    print(f'generating shell scripts')

    if platform == 'slurm':
        prepare_jobs(flight_dict, filtered_flight_dict, config, uas_pipeline, platform) #Use the full uas pipeline to generate the shell scripts

        cp5 = time.time()
        print(f'time to generate shell scripts: {cp5 - cp4:.2f} seconds')
        print('-' * 100)
        print(f'running sbatch jobs')

        job_id = run_batch(flight_dict, working_dir, final_pipeline, platform, dry_run = dry_run) #Use the final pipeline to run the sbatch jobs

    elif platform == 'gcp':
        gcp_tasks = prepare_jobs(flight_dict, filtered_flight_dict, config, final_pipeline, platform)
        #export gcp_tasks to a json file
        with open(os.path.join(project_root, 'profiling/gcp_tasks.json'), 'w') as f:
            json.dump(gcp_tasks, f, indent=4, default=json_path_converter)
        # Then you calculate dependencies and submit the single DAG
        submit_gcp_dag(gcp_tasks, config, dry_run=dry_run)
    cp6 = time.time()
    print(f'time to run sbatch jobs: {cp6 - cp5:.2f} seconds')
    print('-' * 100)
    print(f'total time: {cp6 - cp0:.2f} seconds')
    print('-' * 100)

    # Cycle through all of the job ids for steps 9, 10, 11 to generate an sbatch job for the process_geojson.sh script

    afterok_jobs = []
    for om in job_id.keys():
        for sensor_type in job_id[om].keys():
            for date in job_id[om][sensor_type].keys():
                if job_id[om][sensor_type][date]['step9'] is not None:
                    afterok_jobs.append(job_id[om][sensor_type][date]['step9'])
                if job_id[om][sensor_type][date]['step10'] is not None:
                    afterok_jobs.append(job_id[om][sensor_type][date]['step10'])
                if job_id[om][sensor_type][date]['step11'] is not None:
                    afterok_jobs.append(job_id[om][sensor_type][date]['step11'])

    if len(afterok_jobs) > 0:
        print(f'submitting process_geojson.sh job with dependencies: {afterok_jobs}')
        #process_geojson_job_id, process_geojson_dep = submit_slurm_job(os.path.join(project_root, 'shell_scripts/process_geojson_2025_poc.sh'), working_dir, afterok_jobs, dry_run = dry_run)


    # Export job_id to a json file
    with open(os.path.join(project_root, 'profiling/job_id.json'), 'w') as f:
        json.dump(job_id, f, indent=4, default=json_path_converter)
