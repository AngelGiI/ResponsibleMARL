#!/bin/bash
#set job requirements
#SBATCH --job-name="smaac_seq"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --output=final2_param_case14_dppo_m3_%j.out

# function to handle the SIGTERM signal
function handle_interrupt {
    echo "Caught SIGTERM signal, copying output directory from scratch to home..."
    cp -r "$TMPDIR"/evds_output_dir/result $HOME/smaac/
    exit 1
}

# register the signal handler
trap handle_interrupt TERM

echo "Activate envirnonment"
source activate l2rpn2023_env
export PYTHONPATH=$PYTHONPATH:$PWD

#Create output directory on scratch
mkdir "$TMPDIR"/evds_output_dir
srun cp -r $HOME/learning-2-run-a-power-network/SMAAC/old_data "$TMPDIR"/evds_output_dir/old_data

echo "Run code:"
time srun python -u SMAAC/param_tuning.py -d "$TMPDIR"/evds_output_dir -ns 50_000 -ev 1000 -n final2_param_case14 -c 14 -a "dppo" -s 3 -m 3 -nl 3 -tr 30 -nj 2 -st 8 -ws 3 -to 45
echo "Done"

#Copy output directory from scratch to home
echo "copy output to home dir"
srun cp -r "$TMPDIR"/evds_output_dir/result $HOME/smaac/
