# Makefile

EXE=d2q9-bgk
USER=fh17488

#CC=gcc
#CC=icc
CC=mpiicc
CFLAGS= -std=c11 -Wall -O2 -no-prec-div -ansi-alias -xCORE-AVX2 -ipo -qopenmp -qopenmp-simd
#CFLAGS= -std=c11 -Wall -O2 -no-prec-div -ansi-alias -xCORE-AVX2 -no-ipo -qopenmp -qopt-report=5 -g -Bdynamic
LIBS = -lm

OUTPUT_FILE=$(EXE).out
FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
#REF_FINAL_STATE_FILE=check/128x128.final_state.dat
#REF_AV_VELS_FILE=check/128x128.av_vels.dat
#REF_FINAL_STATE_FILE=check/128x256.final_state.dat
#REF_AV_VELS_FILE=check/128x256.av_vels.dat
#REF_FINAL_STATE_FILE=check/256x256.final_state.dat
#REF_AV_VELS_FILE=check/256x256.av_vels.dat
REF_FINAL_STATE_FILE=check/1024x1024.final_state.dat
REF_AV_VELS_FILE=check/1024x1024.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c 
	$(CC) $(CFLAGS) $^ $(LIBS) -o d2q9-bgk

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)

rmout:
	rm *.out
purge:
	rm -f $(EXE) $(FINAL_STATE_FILE) $(AV_VELS_FILE) $(OUTPUT_FILE) *.optrpt

runn01:
	sbatch submit_job_n01
	
runn05:
	sbatch submit_job_n05
	
runn10:
	sbatch submit_job_n10
	
runn15:
	sbatch submit_job_n15
	
runn20:
	sbatch submit_job_n20
	
runn25:
	sbatch submit_job_n25
	
runn28:
	sbatch submit_job_n28
	
runn42:
	sbatch submit_job_n42
	
runn56:
	sbatch submit_job_n56
	
runn74:
	sbatch submit_job_n74
	
runn84:
	sbatch submit_job_n84
	
runn96:
	sbatch submit_job_n96
	
runn112:
	sbatch submit_job_n112

status:
	squeue -u $(USER)

print: 
	less $(OUTPUT_FILE)

advisor1:
	mpirun -n 1 advixe-cl -collect survey --project-dir ./report ./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat : -np 1 ./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat

advisor2:
	mpirun -n 1 advixe-cl -collect tripcounts -flop --project-dir ./report ./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat : -np 1 ./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat
