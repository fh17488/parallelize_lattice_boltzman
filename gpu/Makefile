# Makefile
# Load modules:
# module use /mnt/storage/scratch/jp8463/modules/modulefiles
# module load clang-ykt/2017-07-24
# module load libs/cudnn/9.0-cuda-9.0: for profiler

EXE=d2q9-bgk
USER=fh17488

CUDA_PATH=/mnt/storage/easybuild/software/CUDA/8.0.44
CC=clang
CFLAGS= -std=c11 -Wall -O3 -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=$(CUDA_PATH) -o d2q9-bgk 
LIBS = -lm

OUTPUT_FILE=$(EXE).out
FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat
#REF_FINAL_STATE_FILE=check/128x256.final_state.dat
#REF_AV_VELS_FILE=check/128x256.av_vels.dat
#REF_FINAL_STATE_FILE=check/256x256.final_state.dat
#REF_AV_VELS_FILE=check/256x256.av_vels.dat
#REF_FINAL_STATE_FILE=check/1024x1024.final_state.dat
#REF_AV_VELS_FILE=check/1024x1024.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

purge:
	rm -f $(EXE) $(OUTPUT_FILE) final*.dat av*dat

rungpu:
	sbatch job_submit_to_gpu

status:
	squeue -u $(USER)

print: 
	less $(OUTPUT_FILE)

