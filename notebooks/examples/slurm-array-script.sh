#!/bin/bash
#SBATCH --job-name adult-dp-pvi1
#SBATCH --account=project_2003275
#SBATCH --array=0-1
#SBATCH -J adult-dp-pvi1
#SBATCH -o log/array_job_%A_%a.out
#SBATCH -e log/array_job_%A_%a.err
#SBATCH --time 0-16:00:00
##SBATCH -p gputest
##SBATCH -p test
#SBATCH --cpus-per-task=1
#SBATCH -p small
##SBATCH -p longrun
##SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=8GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mixheikk@gmail.com


echo "== Job ID: ${SLURM_JOBID}, Task ID: ${SLURM_ARRAY_TASK_ID}, Node ID: ${SLURM_NODEID}, local ID: ${SLURM_LOCALID}, node name: ${SLURMD_NODENAME}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"

echo "current host"
hostname

module purge

###### kale #####
## could also load these manually before submitting so will be inherited
#module load Python/3.7.2-GCCcore-8.2.0
#module load cuDNN/7.6.2.24-CUDA-10.1.243

#source /proj/mixheikk/python372/bin/activate

# set XLA flag pointing to CUDA dir for jax/jaxlib:
#export XLA_FLAGS="--xla_gpu_cuda_data_dir=/appl/opt/CUDA/10.1.243"


###### puhti #####
#conda init
#source ~/.bashrc

#conda activate dp-pvi

#module load python-data/3.7.6-1 # could leave this out with virtualenv
#module load  gcc/8.3.0
#module load cuda/10.1.168
#module load cudnn/7.6.1.34-10.1

module load pytorch/1.6

source /projappl/project_2003275/dp-pvi/py376/bin/activate


#export XLA_FLAGS="--xla_gpu_cuda_data_dir=/appl/spack/install-tree/gcc-8.3.0/cuda-10.1.168-mrdepn"

cd /scratch/project_2003275/dp-pvi/pvi/notebooks/adult_examples
##################
# test that jax is working
#srun python jax-test.py

# individual run with default params for testing that everything works
#srun python parallel_dp_pvi.py

# NOTE: this has been removed for now
# try hyperparam optimisation
#srun python run_ax_experiment.py

sleep "${SLURM_ARRAY_TASK_ID}"

# run sacred with default configs
#srun python sacred-test.py with "job_id=${SLURM_ARRAY_TASK_ID}"

# run specific sacred config:
#srun python dp_logistic_regression.py
srun python adult_1sacred-test.py with "job_id=${SLURM_ARRAY_TASK_ID}"
#srun python sacred-test.py with "unbalanced_test_config" "job_id=${SLURM_ARRAY_TASK_ID}"
#srun python sacred-test.py with "unbalanced_50clients_test_config" "job_id=${SLURM_ARRAY_TASK_ID}"

# try giving some time for all results to log before killing
sleep 30

echo "All done."

