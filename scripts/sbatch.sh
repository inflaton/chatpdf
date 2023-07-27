#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=10           # Number of CPU to request for the job
#SBATCH --mem=64GB                   # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=02-00:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=%u.%j.out          # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory
#SBATCH --requeue                   # Remove if you are not want the workload scheduler to requeue your job after preemption
#SBATCH --constraint=a40            # This tells the workload scheduler to provision you a40 nodes 

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

# ================ Account parameters ================

# Description			| Value
# ---------------------------------------------
# Account name			| zhaoxiaresearch
# List of Assigned Partition	| researchlong researchshort 
# List of Assigned QOS		| research-1-qos
# ---------------------------------------------


#SBATCH --partition=researchshort                 # The partition you've been assigned
#SBATCH --account=zhaoxiaresearch   # The account you've been assigned (normally student)
#SBATCH --qos=research-1-qos       # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=dh.huang.2023@engd.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=testLlama2     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
module load Anaconda3/2022.05
module load CUDA/11.8.0

# Do not remove this line even if you have executed conda init
eval "$(conda shell.bash hook)"

# Create a virtual environment can be commented off if you already have a virtual environment
#conda create -n chatpdf

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
conda activate chatpdf

# If you require any packages, install it before the srun job submission.
#conda install pytorch torchvision torchaudio -c pytorch

# Submit your job to the cluster
srun --gres=gpu:1 $HOME/code/chat-with-pci-dss-v4/scripts/llama-2.sh
