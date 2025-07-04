# PowerShell script for running Dakota job

# Load necessary modules (equivalent of `module load`)
# Adjust as necessary based on your system's environment module setup
$env:PATH += ";C:\Path\To\Python\bin"
$env:VIRTUAL_ENV = "D:\dakotaEnv"
. "$env:VIRTUAL_ENV\Scripts\activate"

$install_dir = "D:\Dakota"  # Replace with the actual installation directory
$dakota_path = "$install_dir\share\dakota\Python"

# Add to the current session
$env:PYTHONPATH = "$dakota_path;$env:PYTHONPATH"

# Define job parameters
$jobName = "dakota"
$nodes = 39
$cpusPerTask = 2
$timeLimit = "2:00:00"
$account = "proj85"

# Submit the job using `mpiexec` (or `Start-Job` for simpler parallel tasks)
python driver.py
