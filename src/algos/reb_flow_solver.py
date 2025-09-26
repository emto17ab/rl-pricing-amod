"""
Minimal Rebalancing Cost 
------------------------
This file contains the specifications for the Min Reb Cost problem.
"""
import os
import math
import subprocess
import time
from collections import defaultdict
from src.misc.utils import mat2str


def solveRebFlow(env, res_path, desiredAcc, CPLEXPATH, directory):
    t = env.time
    accRLTuple = [(n, int(desiredAcc[n])) for n in desiredAcc]
    accTuple = [(n, int(env.acc[n][t+1])) for n in env.acc]
    edgeAttr = [(i, j, env.G.edges[i, j]['time']) for i, j in env.G.edges]
    modPath = os.getcwd().replace('\\', '/')+'/src/cplex_mod/'
    OPTPath = os.getcwd().replace('\\', '/') + '/' + directory + '/cplex_logs/rebalancing/'+res_path + '/'
    
    # Ensure directory creation is atomic and handle race conditions
    try:
        os.makedirs(OPTPath, exist_ok=True)
        # Double-check the directory exists and is writable
        if not os.path.exists(OPTPath):
            raise OSError(f"Failed to create directory: {OPTPath}")
        if not os.access(OPTPath, os.W_OK):
            raise PermissionError(f"No write permission for directory: {OPTPath}")
    except Exception as e:
        print(f"Error creating output directory {OPTPath}: {e}")
        raise e
        
    datafile = OPTPath + f'data_{t}.dat'
    resfile = OPTPath + f'res_{t}.dat'
    # Write data file with error handling
    try:
        with open(datafile, 'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
            file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
            file.write('accRLTuple='+mat2str(accRLTuple)+';\r\n')
        
        # Verify the file was written correctly
        if not os.path.exists(datafile) or os.path.getsize(datafile) == 0:
            raise IOError(f"Failed to write data file or file is empty: {datafile}")
            
    except Exception as e:
        print(f"Error writing data file {datafile}: {e}")
        raise e
    modfile = modPath+'minRebDistRebOnly.mod'
    if CPLEXPATH is None:
        CPLEXPATH = "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
    out_file = OPTPath + f'out_{t}.dat'
    
    # Add error handling and better debugging with retry logic
    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay
    
    for attempt in range(max_retries + 1):
        try:
            # Ensure data and model files exist and are readable
            if not os.path.exists(datafile):
                raise FileNotFoundError(f"Data file not found: {datafile}")
            if not os.path.exists(modfile):
                raise FileNotFoundError(f"Model file not found: {modfile}")
            if not os.access(datafile, os.R_OK):
                raise PermissionError(f"Cannot read data file: {datafile}")
            if not os.access(modfile, os.R_OK):
                raise PermissionError(f"Cannot read model file: {modfile}")
            
            with open(out_file, 'w') as output_f:
                subprocess.check_call(
                    [CPLEXPATH+"oplrun", modfile, datafile], 
                    stdout=output_f, 
                    stderr=subprocess.STDOUT,  # Capture stderr in the output file
                    env=my_env,
                    timeout=60  # Add timeout to prevent hanging
                )
            output_f.close()
            break  # Success, exit retry loop
            
        except subprocess.CalledProcessError as e:
            if attempt < max_retries:
                print(f"CPLEX attempt {attempt + 1} failed with return code {e.returncode}, retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            
            # Final attempt failed, provide detailed error information
            print(f"CPLEX Error: Command failed with return code {e.returncode} after {max_retries + 1} attempts")
            print(f"Command: {e.cmd}")
            print(f"CPLEX path: {CPLEXPATH}")
            print(f"Data file: {datafile}")
            print(f"Mod file: {modfile}")
            print(f"Output file: {out_file}")
            
            # Try to read what was written to the output file for debugging
            if os.path.exists(out_file):
                try:
                    with open(out_file, 'r') as f:
                        error_output = f.read()
                    print(f"CPLEX output/error:\n{error_output}")
                except Exception as read_e:
                    print(f"Could not read output file: {read_e}")
            
            # Re-raise the original exception
            raise e
            
        except subprocess.TimeoutExpired as e:
            if attempt < max_retries:
                print(f"CPLEX attempt {attempt + 1} timed out, retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            print(f"CPLEX Error: Command timed out after {max_retries + 1} attempts")
            raise e
            
        except (FileNotFoundError, PermissionError) as e:
            if attempt < max_retries:
                print(f"File access error on attempt {attempt + 1}: {e}, retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            print(f"File access error: {e}")
            raise e
            
        except Exception as e:
            if attempt < max_retries:
                print(f"Unexpected error on attempt {attempt + 1}: {e}, retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            print(f"Unexpected error in solveRebFlow: {e}")
            print(f"CPLEX path: {CPLEXPATH}")
            print(f"Data file exists: {os.path.exists(datafile)}")
            print(f"Mod file exists: {os.path.exists(modfile)}")
            raise e

    # 3. collect results from file
    flow = defaultdict(float)
    with open(resfile, 'r', encoding="utf8") as file:
        for row in file:
            item = row.strip().strip(';').split('=')
            if item[0] == 'flow':
                values = item[1].strip(')]').strip('[(').split(')(')
                for v in values:
                    if len(v) == 0:
                        continue
                    i, j, f = v.split(',')
                    flow[int(i), int(j)] = float(f)
    action = [flow[i, j] for i, j in env.edges]
    return action
