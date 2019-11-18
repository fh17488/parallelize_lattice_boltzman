/*
Submission: Flat MPI
Name: Faraz Haider
User id: fh17488
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdint.h>
#include <omp.h>  
#include "mpi.h"

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define MASTER_RANK		0
//#define DEBUG

typedef struct
{
	int size;
	int rank;
	int rank_ahead;
	int rank_before;
	int tag;
	MPI_Status status;
	int thread_limit;
} t_rank_metadata;

typedef struct
{
	int global_rows;
	int global_cols;
	int grid_extent;
	int buffer_extent;
	int extra_rows;
	int local_rows;
	int first_row;
	int last_row;
	int first_row_neighbour;
	int last_row_neighbour;
	int first_col;
	int last_col;
	int first_row_start_idx;
	int last_row_start_idx;
	int global_non_obstacle_cells;
} t_grid_meta_data;

/* struct to hold the parameter values */
typedef struct
{
	int    nx;            /* no. of cells in x-direction */
	int    ny;            /* no. of cells in y-direction */
	int    maxIters;      /* no. of iterations */
	int    reynolds_dim;  /* dimension for Reynolds number */
	float density;       /* density per link */
	float accel;         /* density redistribution */
	float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
	float speeds[NSPEEDS];
} t_speed;

typedef struct
{
	float* cell_speed_0_ve;
	float* cell_speed_1_ve;
	float* cell_speed_2_ve;
	float* cell_speed_3_ve;
	float* cell_speed_4_ve;
	float* cell_speed_5_ve;
	float* cell_speed_6_ve;
	float* cell_speed_7_ve;
	float* cell_speed_8_ve;

	float* tmp_cell_speed_0_ve;
	float* tmp_cell_speed_1_ve;
	float* tmp_cell_speed_2_ve;
	float* tmp_cell_speed_3_ve;
	float* tmp_cell_speed_4_ve;
	float* tmp_cell_speed_5_ve;
	float* tmp_cell_speed_6_ve;
	float* tmp_cell_speed_7_ve;
	float* tmp_cell_speed_8_ve;

	int32_t* obstacles;    /* grid indicating which cells are blocked */

	float* partialcellbuf;
} t_buffer_ptrs;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(int* argc, char*** argv, t_rank_metadata* RANK, t_grid_meta_data* GRID,
	const char* paramfile, const char* obstaclefile,
	t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
	int** obstacles_ptr, float** av_vels_ptr, t_buffer_ptrs* BUFFER);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(t_rank_metadata* RANK, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int accelerate_flow(t_rank_metadata* RANK, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, int* obstacles);
int propagate(t_rank_metadata* RANK, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, t_speed* tmp_cells);
int rebound(t_rank_metadata* RANK, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int collision(t_rank_metadata* RANK, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
	int** obstacles_ptr, float** av_vels_ptr, t_buffer_ptrs* BUFFER);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

void initialize_MPI(int* argc, char*** argv);
void initialize_RANK(t_rank_metadata* RANK);
void initialize_GRID(t_rank_metadata* RANK, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param* params, int** obstacles_ptr, float w0, float w1, float w2);
void initialize_BUFFER(t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, float w0, float w1, float w2);
void finalize_MPI();
int calc_rank_non_obstacle_cells(t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param* params, int** obstacles_ptr);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
	t_rank_metadata RANK;
	t_grid_meta_data GRID;
	t_buffer_ptrs BUFFER;
	char*    paramfile = NULL;    /* name of the input parameter file */
	char*    obstaclefile = NULL; /* name of a the input obstacle file */
	t_param  params;              /* struct to hold parameter values */
	t_speed* cells = NULL;    /* grid containing fluid densities */
	t_speed* tmp_cells = NULL;    /* scratch space */
	int*     obstacles = NULL;    /* grid indicating which cells are blocked */
	float* av_vels = NULL;     /* a record of the av. velocity computed for each timestep */
	struct timeval timstr;        /* structure to hold elapsed time */
	struct rusage ru;             /* structure to hold CPU time--system and user */
	double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
	double rank_time = 0;
	double cumulative_time = 0;
	double rank_usrtime = 0;		/* floating point number to record elapsed user CPU time */
	double cumulative_usrtime = 0;
	double rank_systime = 0;		/* floating point number to record elapsed system CPU time */
	double cumulative_systime = 0;
	float reynolds_num = 0;
	int message_length = 0;
	int first_row_for_global_gather = 0;

	/* parse the command line */
	if (argc != 3)
	{
		usage(argv[0]);
	}
	else
	{
		paramfile = argv[1];
		obstaclefile = argv[2];
	}

	/* initialise our data structures and load values from file */
	initialise(&argc, &argv, &RANK, &GRID, paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &BUFFER);
	MPI_Barrier(MPI_COMM_WORLD);
	/* iterate for maxIters timesteps */
	gettimeofday(&timstr, NULL);
	tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
	for (int tt = 0; tt < params.maxIters; tt++)
	{
		float rank_av_vel = 0;
		float iteration_av_vel = 0;
		accelerate_flow(&RANK, &GRID, &BUFFER, params, cells, obstacles);
		propagate(&RANK, &GRID, &BUFFER, params, cells, tmp_cells);
		rebound(&RANK, &GRID, &BUFFER, params, cells, tmp_cells, obstacles);
		collision(&RANK, &GRID, &BUFFER, params, cells, tmp_cells, obstacles);
		rank_av_vel = av_velocity(&GRID, &BUFFER, params, cells, obstacles);
		MPI_Reduce(&rank_av_vel, &iteration_av_vel, 1, MPI_FLOAT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
		av_vels[tt] = iteration_av_vel;
		MPI_Barrier(MPI_COMM_WORLD);
		RANK.tag = 4;
		MPI_Sendrecv(&BUFFER.cell_speed_4_ve[GRID.first_col + GRID.first_row * GRID.global_cols], GRID.global_rows, MPI_FLOAT, RANK.rank_before, RANK.tag,
			&BUFFER.cell_speed_4_ve[GRID.first_col + GRID.last_row_neighbour * GRID.global_cols], GRID.global_rows, MPI_FLOAT, RANK.rank_ahead, RANK.tag,
			MPI_COMM_WORLD, &RANK.status);
		RANK.tag = 7;
		MPI_Sendrecv(&BUFFER.cell_speed_7_ve[GRID.first_col + GRID.first_row * GRID.global_cols], GRID.global_rows, MPI_FLOAT, RANK.rank_before, RANK.tag,
			&BUFFER.cell_speed_7_ve[GRID.first_col + GRID.last_row_neighbour * GRID.global_cols], GRID.global_rows, MPI_FLOAT, RANK.rank_ahead, RANK.tag,
			MPI_COMM_WORLD, &RANK.status);
		RANK.tag = 8;
		MPI_Sendrecv(&BUFFER.cell_speed_8_ve[GRID.first_col + GRID.first_row * GRID.global_cols], GRID.global_rows, MPI_FLOAT, RANK.rank_before, RANK.tag,
			&BUFFER.cell_speed_8_ve[GRID.first_col + GRID.last_row_neighbour * GRID.global_cols], GRID.global_rows, MPI_FLOAT, RANK.rank_ahead, RANK.tag,
			MPI_COMM_WORLD, &RANK.status);
		RANK.tag = 2;
		MPI_Sendrecv(&BUFFER.cell_speed_2_ve[GRID.first_col + GRID.last_row * GRID.global_cols], GRID.global_rows, MPI_FLOAT, RANK.rank_ahead, RANK.tag,
			&BUFFER.cell_speed_2_ve[GRID.first_col + GRID.first_row_neighbour * GRID.global_cols], GRID.global_rows, MPI_FLOAT, RANK.rank_before, RANK.tag,
			MPI_COMM_WORLD, &RANK.status);
		RANK.tag = 5;
		MPI_Sendrecv(&BUFFER.cell_speed_5_ve[GRID.first_col + GRID.last_row * GRID.global_cols], GRID.global_rows, MPI_FLOAT, RANK.rank_ahead, RANK.tag,
			&BUFFER.cell_speed_5_ve[GRID.first_col + GRID.first_row_neighbour * GRID.global_cols], GRID.global_rows, MPI_FLOAT, RANK.rank_before, RANK.tag,
			MPI_COMM_WORLD, &RANK.status);
		RANK.tag = 6;
		MPI_Sendrecv(&BUFFER.cell_speed_6_ve[GRID.first_col + GRID.last_row * GRID.global_cols], GRID.global_rows, MPI_FLOAT, RANK.rank_ahead, RANK.tag,
			&BUFFER.cell_speed_6_ve[GRID.first_col + GRID.first_row_neighbour * GRID.global_cols], GRID.global_rows, MPI_FLOAT, RANK.rank_before, RANK.tag,
			MPI_COMM_WORLD, &RANK.status);
#ifdef DEBUG
		printf("==timestep: %d==\n", tt);
		printf("av velocity: %.12E\n", av_vels[tt]);
		printf("tot density: %.12E\n", total_density(params, cells));
#endif
	}

	gettimeofday(&timstr, NULL);
	toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
	getrusage(RUSAGE_SELF, &ru);
	timstr = ru.ru_utime;
	rank_usrtime = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
	timstr = ru.ru_stime;
	rank_systime = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

	/* write final values and free memory */
	reynolds_num = calc_reynolds(&GRID, &BUFFER, params, cells, obstacles);
	rank_time = toc - tic;
	MPI_Reduce(&rank_time, &cumulative_time, 1, MPI_DOUBLE, MPI_MAX, MASTER_RANK, MPI_COMM_WORLD);
	MPI_Reduce(&rank_usrtime, &cumulative_usrtime, 1, MPI_DOUBLE, MPI_MAX, MASTER_RANK, MPI_COMM_WORLD);
	MPI_Reduce(&rank_systime, &cumulative_systime, 1, MPI_DOUBLE, MPI_MAX, MASTER_RANK, MPI_COMM_WORLD);
	if (RANK.rank == MASTER_RANK)
	{
		printf("==done==\n");
		printf("Reynolds number:\t\t%.12E\n", reynolds_num);
		printf("Elapsed time:\t\t\t%.6lf (s)\n", cumulative_time);
		printf("Elapsed user CPU time:\t\t%.6lf (s)\n", cumulative_usrtime);
		printf("Elapsed system CPU time:\t%.6lf (s)\n", cumulative_systime);
	}
	//Copy rows from other ranks to Master rank
	message_length = GRID.local_rows * GRID.global_cols;
	first_row_for_global_gather = (RANK.rank == MASTER_RANK) ? GRID.extra_rows : GRID.first_row;

	for (int jj = first_row_for_global_gather, bufjj = 0; jj <= GRID.last_row && bufjj < GRID.local_rows; jj++, bufjj++)
		for (int ii = GRID.first_col; ii <= GRID.last_col; ii++)
			BUFFER.partialcellbuf[(ii + bufjj * GRID.global_cols)] = BUFFER.cell_speed_0_ve[ii + jj * GRID.global_cols];
	MPI_Gather(BUFFER.partialcellbuf, message_length, MPI_FLOAT,
		((RANK.rank == MASTER_RANK) ? &BUFFER.cell_speed_0_ve[GRID.extra_rows * GRID.global_cols] : BUFFER.cell_speed_0_ve),
		message_length, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

	for (int jj = first_row_for_global_gather, bufjj = 0; jj <= GRID.last_row && bufjj < GRID.local_rows; jj++, bufjj++)
		for (int ii = GRID.first_col; ii <= GRID.last_col; ii++)
			BUFFER.partialcellbuf[(ii + bufjj * GRID.global_cols)] = BUFFER.cell_speed_1_ve[ii + jj * GRID.global_cols];
	MPI_Gather(BUFFER.partialcellbuf, message_length, MPI_FLOAT,
		((RANK.rank == MASTER_RANK) ? &BUFFER.cell_speed_1_ve[GRID.extra_rows * GRID.global_cols] : BUFFER.cell_speed_1_ve),
		message_length, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

	for (int jj = first_row_for_global_gather, bufjj = 0; jj <= GRID.last_row && bufjj < GRID.local_rows; jj++, bufjj++)
		for (int ii = GRID.first_col; ii <= GRID.last_col; ii++)
			BUFFER.partialcellbuf[(ii + bufjj * GRID.global_cols)] = BUFFER.cell_speed_2_ve[ii + jj * GRID.global_cols];
	MPI_Gather(BUFFER.partialcellbuf, message_length, MPI_FLOAT,
		((RANK.rank == MASTER_RANK) ? &BUFFER.cell_speed_2_ve[GRID.extra_rows * GRID.global_cols] : BUFFER.cell_speed_2_ve),
		message_length, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

	for (int jj = first_row_for_global_gather, bufjj = 0; jj <= GRID.last_row && bufjj < GRID.local_rows; jj++, bufjj++)
		for (int ii = GRID.first_col; ii <= GRID.last_col; ii++)
			BUFFER.partialcellbuf[(ii + bufjj * GRID.global_cols)] = BUFFER.cell_speed_3_ve[ii + jj * GRID.global_cols];
	MPI_Gather(BUFFER.partialcellbuf, message_length, MPI_FLOAT,
		((RANK.rank == MASTER_RANK) ? &BUFFER.cell_speed_3_ve[GRID.extra_rows * GRID.global_cols] : BUFFER.cell_speed_3_ve),
		message_length, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

	for (int jj = first_row_for_global_gather, bufjj = 0; jj <= GRID.last_row && bufjj < GRID.local_rows; jj++, bufjj++)
		for (int ii = GRID.first_col; ii <= GRID.last_col; ii++)
			BUFFER.partialcellbuf[(ii + bufjj * GRID.global_cols)] = BUFFER.cell_speed_4_ve[ii + jj * GRID.global_cols];
	MPI_Gather(BUFFER.partialcellbuf, message_length, MPI_FLOAT,
		((RANK.rank == MASTER_RANK) ? &BUFFER.cell_speed_4_ve[GRID.extra_rows * GRID.global_cols] : BUFFER.cell_speed_4_ve),
		message_length, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

	for (int jj = first_row_for_global_gather, bufjj = 0; jj <= GRID.last_row && bufjj < GRID.local_rows; jj++, bufjj++)
		for (int ii = GRID.first_col; ii <= GRID.last_col; ii++)
			BUFFER.partialcellbuf[(ii + bufjj * GRID.global_cols)] = BUFFER.cell_speed_5_ve[ii + jj * GRID.global_cols];
	MPI_Gather(BUFFER.partialcellbuf, message_length, MPI_FLOAT,
		((RANK.rank == MASTER_RANK) ? &BUFFER.cell_speed_5_ve[GRID.extra_rows * GRID.global_cols] : BUFFER.cell_speed_5_ve),
		message_length, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

	for (int jj = first_row_for_global_gather, bufjj = 0; jj <= GRID.last_row && bufjj < GRID.local_rows; jj++, bufjj++)
		for (int ii = GRID.first_col; ii <= GRID.last_col; ii++)
			BUFFER.partialcellbuf[(ii + bufjj * GRID.global_cols)] = BUFFER.cell_speed_6_ve[ii + jj * GRID.global_cols];
	MPI_Gather(BUFFER.partialcellbuf, message_length, MPI_FLOAT,
		((RANK.rank == MASTER_RANK) ? &BUFFER.cell_speed_6_ve[GRID.extra_rows * GRID.global_cols] : BUFFER.cell_speed_6_ve),
		message_length, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

	for (int jj = first_row_for_global_gather, bufjj = 0; jj <= GRID.last_row && bufjj < GRID.local_rows; jj++, bufjj++)
		for (int ii = GRID.first_col; ii <= GRID.last_col; ii++)
			BUFFER.partialcellbuf[(ii + bufjj * GRID.global_cols)] = BUFFER.cell_speed_7_ve[ii + jj * GRID.global_cols];
	MPI_Gather(BUFFER.partialcellbuf, message_length, MPI_FLOAT,
		((RANK.rank == MASTER_RANK) ? &BUFFER.cell_speed_7_ve[GRID.extra_rows * GRID.global_cols] : BUFFER.cell_speed_7_ve),
		message_length, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

	for (int jj = first_row_for_global_gather, bufjj = 0; jj <= GRID.last_row && bufjj < GRID.local_rows; jj++, bufjj++)
		for (int ii = GRID.first_col; ii <= GRID.last_col; ii++)
			BUFFER.partialcellbuf[(ii + bufjj * GRID.global_cols)] = BUFFER.cell_speed_8_ve[ii + jj * GRID.global_cols];
	MPI_Gather(BUFFER.partialcellbuf, message_length, MPI_FLOAT,
		((RANK.rank == MASTER_RANK) ? &BUFFER.cell_speed_8_ve[GRID.extra_rows * GRID.global_cols] : BUFFER.cell_speed_8_ve),
		message_length, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

	if (RANK.rank == MASTER_RANK)
		write_values(params, cells, obstacles, av_vels, &GRID, &BUFFER);
	finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, &BUFFER);
	MPI_Barrier(MPI_COMM_WORLD);
	finalize_MPI();
	return EXIT_SUCCESS;
}

int timestep(t_rank_metadata* RANK, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
	accelerate_flow(RANK, GRID, BUFFER, params, cells, obstacles);
	propagate(RANK, GRID, BUFFER, params, cells, tmp_cells);
	rebound(RANK, GRID, BUFFER, params, cells, tmp_cells, obstacles);
	collision(RANK, GRID, BUFFER, params, cells, tmp_cells, obstacles);
	return EXIT_SUCCESS;
}

int accelerate_flow(t_rank_metadata* RANK, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, int* obstacles)
{
	float* cell_speed_1_ve = BUFFER->cell_speed_1_ve;
	float* cell_speed_3_ve = BUFFER->cell_speed_3_ve;
	float* cell_speed_5_ve = BUFFER->cell_speed_5_ve;
	float* cell_speed_6_ve = BUFFER->cell_speed_6_ve;
	float* cell_speed_7_ve = BUFFER->cell_speed_7_ve;
	float* cell_speed_8_ve = BUFFER->cell_speed_8_ve;

	/* compute weighting factors */
	float w1 = params.density * params.accel / 9.f;
	float w2 = params.density * params.accel / 36.f;

	/* modify the 2nd row of the grid */
	int jj = params.ny - 2;

	if (GRID->first_row <= jj && GRID->last_row >= jj)
	{
#pragma omp simd aligned(cell_speed_3_ve, cell_speed_6_ve, cell_speed_7_ve, cell_speed_1_ve, cell_speed_5_ve, cell_speed_8_ve: 32)
		for (int32_t ii = 0; ii < GRID->global_cols; ii++)
		{
			if (BUFFER->obstacles[ii + jj * GRID->global_cols] == 0)
			{
				//Start: body of the accelerate flow function
				//These values are being calculated for the next iteration of the timestep
				if ((BUFFER->cell_speed_3_ve[ii + jj * GRID->global_cols] - w1) > 0.f
					&& (BUFFER->cell_speed_6_ve[ii + jj * GRID->global_cols] - w2) > 0.f
					&& (BUFFER->cell_speed_7_ve[ii + jj * GRID->global_cols] - w2) > 0.f)
				{
					/* increase 'east-side' densities */
					BUFFER->cell_speed_1_ve[ii + jj * GRID->global_cols] += w1;
					BUFFER->cell_speed_5_ve[ii + jj * GRID->global_cols] += w2;
					BUFFER->cell_speed_8_ve[ii + jj * GRID->global_cols] += w2;
					/* decrease 'west-side' densities */
					BUFFER->cell_speed_3_ve[ii + jj * GRID->global_cols] -= w1;
					BUFFER->cell_speed_6_ve[ii + jj * GRID->global_cols] -= w2;
					BUFFER->cell_speed_7_ve[ii + jj * GRID->global_cols] -= w2;
				}
				//End: body of the accelerate flow function
			}
		}
	}

	return EXIT_SUCCESS;
}

int propagate(t_rank_metadata* RANK, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, t_speed* tmp_cells)
{
	float* cell_speed_0_ve = BUFFER->cell_speed_0_ve;
	float* cell_speed_1_ve = BUFFER->cell_speed_1_ve;
	float* cell_speed_2_ve = BUFFER->cell_speed_2_ve;
	float* cell_speed_3_ve = BUFFER->cell_speed_3_ve;
	float* cell_speed_4_ve = BUFFER->cell_speed_4_ve;
	float* cell_speed_5_ve = BUFFER->cell_speed_5_ve;
	float* cell_speed_6_ve = BUFFER->cell_speed_6_ve;
	float* cell_speed_7_ve = BUFFER->cell_speed_7_ve;
	float* cell_speed_8_ve = BUFFER->cell_speed_8_ve;

	float* tmp_cell_speed_0_ve = BUFFER->tmp_cell_speed_0_ve;
	float* tmp_cell_speed_1_ve = BUFFER->tmp_cell_speed_1_ve;
	float* tmp_cell_speed_2_ve = BUFFER->tmp_cell_speed_2_ve;
	float* tmp_cell_speed_3_ve = BUFFER->tmp_cell_speed_3_ve;
	float* tmp_cell_speed_4_ve = BUFFER->tmp_cell_speed_4_ve;
	float* tmp_cell_speed_5_ve = BUFFER->tmp_cell_speed_5_ve;
	float* tmp_cell_speed_6_ve = BUFFER->tmp_cell_speed_6_ve;
	float* tmp_cell_speed_7_ve = BUFFER->tmp_cell_speed_7_ve;
	float* tmp_cell_speed_8_ve = BUFFER->tmp_cell_speed_8_ve;

	/* loop over _all_ cells */
#pragma omp simd collapse(2) aligned(cell_speed_0_ve, cell_speed_1_ve, cell_speed_2_ve, cell_speed_3_ve, cell_speed_4_ve, cell_speed_5_ve, cell_speed_6_ve, cell_speed_7_ve, cell_speed_8_ve, tmp_cell_speed_0_ve, tmp_cell_speed_1_ve, tmp_cell_speed_2_ve, tmp_cell_speed_3_ve, tmp_cell_speed_4_ve, tmp_cell_speed_5_ve, tmp_cell_speed_6_ve, tmp_cell_speed_7_ve, tmp_cell_speed_8_ve: 32) 
	for (int jj = GRID->first_row; jj <= GRID->last_row; jj++)
		for (int ii = GRID->first_col; ii <= GRID->last_col; ii++)
		{
			//determine indices of axis-direction neighbours
			//respecting periodic boundary conditions (wrap around)
			int y_n = (jj + 1) % GRID->global_rows;
			int x_e = (ii + 1) % GRID->global_cols;
			int y_s = (jj == 0) ? (jj + GRID->global_rows - 1) : (jj - 1);
			int x_w = (ii == 0) ? (ii + GRID->global_cols - 1) : (ii - 1);

			// propagate densities from neighbouring cells, following
			// appropriate directions of travel and writing into
			// scratch space grid

			BUFFER->tmp_cell_speed_0_ve[ii + jj * GRID->global_cols] = BUFFER->cell_speed_0_ve[ii + jj * GRID->global_cols]; // central cell, no movement
			BUFFER->tmp_cell_speed_1_ve[ii + jj * GRID->global_cols] = BUFFER->cell_speed_1_ve[x_w + jj * GRID->global_cols]; // east			
			BUFFER->tmp_cell_speed_2_ve[ii + jj * GRID->global_cols] = BUFFER->cell_speed_2_ve[ii + y_s * GRID->global_cols]; // north
			BUFFER->tmp_cell_speed_3_ve[ii + jj * GRID->global_cols] = BUFFER->cell_speed_3_ve[x_e + jj * GRID->global_cols]; // west
			BUFFER->tmp_cell_speed_4_ve[ii + jj * GRID->global_cols] = BUFFER->cell_speed_4_ve[ii + y_n * GRID->global_cols]; // south
			BUFFER->tmp_cell_speed_5_ve[ii + jj * GRID->global_cols] = BUFFER->cell_speed_5_ve[x_w + y_s * GRID->global_cols]; // north-east
			BUFFER->tmp_cell_speed_6_ve[ii + jj * GRID->global_cols] = BUFFER->cell_speed_6_ve[x_e + y_s * GRID->global_cols]; // north-west
			BUFFER->tmp_cell_speed_7_ve[ii + jj * GRID->global_cols] = BUFFER->cell_speed_7_ve[x_e + y_n * GRID->global_cols]; // south-west
			BUFFER->tmp_cell_speed_8_ve[ii + jj * GRID->global_cols] = BUFFER->cell_speed_8_ve[x_w + y_n * GRID->global_cols]; // south-east
		}

	return EXIT_SUCCESS;
}

int rebound(t_rank_metadata* RANK, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
	float* cell_speed_1_ve = BUFFER->cell_speed_1_ve;
	float* cell_speed_2_ve = BUFFER->cell_speed_2_ve;
	float* cell_speed_3_ve = BUFFER->cell_speed_3_ve;
	float* cell_speed_4_ve = BUFFER->cell_speed_4_ve;
	float* cell_speed_5_ve = BUFFER->cell_speed_5_ve;
	float* cell_speed_6_ve = BUFFER->cell_speed_6_ve;
	float* cell_speed_7_ve = BUFFER->cell_speed_7_ve;
	float* cell_speed_8_ve = BUFFER->cell_speed_8_ve;

	float* tmp_cell_speed_1_ve = BUFFER->tmp_cell_speed_1_ve;
	float* tmp_cell_speed_2_ve = BUFFER->tmp_cell_speed_2_ve;
	float* tmp_cell_speed_3_ve = BUFFER->tmp_cell_speed_3_ve;
	float* tmp_cell_speed_4_ve = BUFFER->tmp_cell_speed_4_ve;
	float* tmp_cell_speed_5_ve = BUFFER->tmp_cell_speed_5_ve;
	float* tmp_cell_speed_6_ve = BUFFER->tmp_cell_speed_6_ve;
	float* tmp_cell_speed_7_ve = BUFFER->tmp_cell_speed_7_ve;
	float* tmp_cell_speed_8_ve = BUFFER->tmp_cell_speed_8_ve;

	/* loop over the cells in the grid */
#pragma omp simd collapse(2) aligned(cell_speed_1_ve, cell_speed_2_ve, cell_speed_3_ve, cell_speed_4_ve, cell_speed_5_ve, cell_speed_6_ve, cell_speed_7_ve, cell_speed_8_ve, tmp_cell_speed_1_ve, tmp_cell_speed_2_ve, tmp_cell_speed_3_ve, tmp_cell_speed_4_ve, tmp_cell_speed_5_ve, tmp_cell_speed_6_ve, tmp_cell_speed_7_ve, tmp_cell_speed_8_ve: 32)
	for (int jj = GRID->first_row; jj <= GRID->last_row; jj++)
		for (int ii = GRID->first_col; ii <= GRID->last_col; ii++)
		{
			/* if the cell contains an obstacle */
			if (BUFFER->obstacles[ii + jj * GRID->global_cols] == 1)
			{
				/* called after propagate, so taking values from scratch space
				** mirroring, and writing into main grid */
				BUFFER->cell_speed_1_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_3_ve[ii + jj * GRID->global_cols];
				BUFFER->cell_speed_2_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_4_ve[ii + jj * GRID->global_cols];
				BUFFER->cell_speed_3_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_1_ve[ii + jj * GRID->global_cols];
				BUFFER->cell_speed_4_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_2_ve[ii + jj * GRID->global_cols];
				BUFFER->cell_speed_5_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_7_ve[ii + jj * GRID->global_cols];
				BUFFER->cell_speed_6_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_8_ve[ii + jj * GRID->global_cols];
				BUFFER->cell_speed_7_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_5_ve[ii + jj * GRID->global_cols];
				BUFFER->cell_speed_8_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_6_ve[ii + jj * GRID->global_cols];
			}
		}

	return EXIT_SUCCESS;
}

int collision(t_rank_metadata* RANK, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
	float* cell_speed_0_ve = BUFFER->cell_speed_0_ve;
	float* cell_speed_1_ve = BUFFER->cell_speed_1_ve;
	float* cell_speed_2_ve = BUFFER->cell_speed_2_ve;
	float* cell_speed_3_ve = BUFFER->cell_speed_3_ve;
	float* cell_speed_4_ve = BUFFER->cell_speed_4_ve;
	float* cell_speed_5_ve = BUFFER->cell_speed_5_ve;
	float* cell_speed_6_ve = BUFFER->cell_speed_6_ve;
	float* cell_speed_7_ve = BUFFER->cell_speed_7_ve;
	float* cell_speed_8_ve = BUFFER->cell_speed_8_ve;

	float* tmp_cell_speed_0_ve = BUFFER->tmp_cell_speed_0_ve;
	float* tmp_cell_speed_1_ve = BUFFER->tmp_cell_speed_1_ve;
	float* tmp_cell_speed_2_ve = BUFFER->tmp_cell_speed_2_ve;
	float* tmp_cell_speed_3_ve = BUFFER->tmp_cell_speed_3_ve;
	float* tmp_cell_speed_4_ve = BUFFER->tmp_cell_speed_4_ve;
	float* tmp_cell_speed_5_ve = BUFFER->tmp_cell_speed_5_ve;
	float* tmp_cell_speed_6_ve = BUFFER->tmp_cell_speed_6_ve;
	float* tmp_cell_speed_7_ve = BUFFER->tmp_cell_speed_7_ve;
	float* tmp_cell_speed_8_ve = BUFFER->tmp_cell_speed_8_ve;

	const float c_sq = 1.f / 3.f; /* square of speed of sound */
	const float w0 = 4.f / 9.f;  /* weighting factor */
	const float w1 = 1.f / 9.f;  /* weighting factor */
	const float w2 = 1.f / 36.f; /* weighting factor */

#pragma omp simd collapse(2) aligned(cell_speed_0_ve, cell_speed_1_ve, cell_speed_2_ve, cell_speed_3_ve, cell_speed_4_ve, cell_speed_5_ve, cell_speed_6_ve, cell_speed_7_ve, cell_speed_8_ve, tmp_cell_speed_0_ve, tmp_cell_speed_1_ve, tmp_cell_speed_2_ve, tmp_cell_speed_3_ve, tmp_cell_speed_4_ve, tmp_cell_speed_5_ve, tmp_cell_speed_6_ve, tmp_cell_speed_7_ve, tmp_cell_speed_8_ve: 32)
	for (int jj = GRID->first_row; jj <= GRID->last_row; jj++)
		for (int ii = GRID->first_col; ii <= GRID->last_col; ii++)
		{
			/* don't consider occupied cells */
			if (BUFFER->obstacles[ii + jj * GRID->global_cols] == 0)
			{
				/* compute local density total */
				float local_density = 0.f;
				local_density = BUFFER->tmp_cell_speed_0_ve[ii + jj * GRID->global_cols] + BUFFER->tmp_cell_speed_1_ve[ii + jj * GRID->global_cols] +
					BUFFER->tmp_cell_speed_2_ve[ii + jj * GRID->global_cols] + BUFFER->tmp_cell_speed_3_ve[ii + jj * GRID->global_cols] +
					BUFFER->tmp_cell_speed_4_ve[ii + jj * GRID->global_cols];
				local_density += BUFFER->tmp_cell_speed_5_ve[ii + jj * GRID->global_cols] + BUFFER->tmp_cell_speed_6_ve[ii + jj * GRID->global_cols] +
					BUFFER->tmp_cell_speed_7_ve[ii + jj * GRID->global_cols] + BUFFER->tmp_cell_speed_8_ve[ii + jj * GRID->global_cols];

				/* compute x velocity component */
				float u_x = (BUFFER->tmp_cell_speed_1_ve[ii + jj * GRID->global_cols]
					+ BUFFER->tmp_cell_speed_5_ve[ii + jj * GRID->global_cols]
					+ BUFFER->tmp_cell_speed_8_ve[ii + jj * GRID->global_cols]
					- (BUFFER->tmp_cell_speed_3_ve[ii + jj * GRID->global_cols]
						+ BUFFER->tmp_cell_speed_6_ve[ii + jj * GRID->global_cols]
						+ BUFFER->tmp_cell_speed_7_ve[ii + jj * GRID->global_cols]))
					/ local_density;
				/* compute y velocity component */
				float u_y = (BUFFER->tmp_cell_speed_2_ve[ii + jj * GRID->global_cols]
					+ BUFFER->tmp_cell_speed_5_ve[ii + jj * GRID->global_cols]
					+ BUFFER->tmp_cell_speed_6_ve[ii + jj * GRID->global_cols]
					- (BUFFER->tmp_cell_speed_4_ve[ii + jj * GRID->global_cols]
						+ BUFFER->tmp_cell_speed_7_ve[ii + jj * GRID->global_cols]
						+ BUFFER->tmp_cell_speed_8_ve[ii + jj * GRID->global_cols]))
					/ local_density;

				/* velocity squared */
				float u_sq = u_x * u_x + u_y * u_y;

				/* directional velocity components */
				float u[NSPEEDS];
				u[1] = u_x;        /* east */
				u[2] = u_y;  /* north */
				u[3] = -u_x;        /* west */
				u[4] = -u_y;  /* south */
				u[5] = u_x + u_y;  /* north-east */
				u[6] = -u_x + u_y;  /* north-west */
				u[7] = -u_x - u_y;  /* south-west */
				u[8] = u_x - u_y;  /* south-east */

								   /* equilibrium densities */
				float d_equ[NSPEEDS];
				/* zero velocity density: weight w0 */
				d_equ[0] = w0 * local_density
					* (1.f - u_sq / (2.f * c_sq));

				/* axis speeds: weight w1 */
				d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
					+ (u[1] * u[1]) / (2.f * c_sq * c_sq)
					- u_sq / (2.f * c_sq));

				d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
					+ (u[2] * u[2]) / (2.f * c_sq * c_sq)
					- u_sq / (2.f * c_sq));

				d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
					+ (u[3] * u[3]) / (2.f * c_sq * c_sq)
					- u_sq / (2.f * c_sq));

				d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
					+ (u[4] * u[4]) / (2.f * c_sq * c_sq)
					- u_sq / (2.f * c_sq));

				/* diagonal speeds: weight w2 */
				d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
					+ (u[5] * u[5]) / (2.f * c_sq * c_sq)
					- u_sq / (2.f * c_sq));

				d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
					+ (u[6] * u[6]) / (2.f * c_sq * c_sq)
					- u_sq / (2.f * c_sq));

				d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
					+ (u[7] * u[7]) / (2.f * c_sq * c_sq)
					- u_sq / (2.f * c_sq));

				d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
					+ (u[8] * u[8]) / (2.f * c_sq * c_sq)
					- u_sq / (2.f * c_sq));

				/* relaxation step */
				BUFFER->cell_speed_0_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_0_ve[ii + jj * GRID->global_cols] + params.omega
					* (d_equ[0] - BUFFER->tmp_cell_speed_0_ve[ii + jj * GRID->global_cols]);

				BUFFER->cell_speed_1_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_1_ve[ii + jj * GRID->global_cols] + params.omega
					* (d_equ[1] - BUFFER->tmp_cell_speed_1_ve[ii + jj * GRID->global_cols]);

				BUFFER->cell_speed_2_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_2_ve[ii + jj * GRID->global_cols] + params.omega
					* (d_equ[2] - BUFFER->tmp_cell_speed_2_ve[ii + jj * GRID->global_cols]);

				BUFFER->cell_speed_3_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_3_ve[ii + jj * GRID->global_cols] + params.omega
					* (d_equ[3] - BUFFER->tmp_cell_speed_3_ve[ii + jj * GRID->global_cols]);

				BUFFER->cell_speed_4_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_4_ve[ii + jj * GRID->global_cols] + params.omega
					* (d_equ[4] - BUFFER->tmp_cell_speed_4_ve[ii + jj * GRID->global_cols]);

				BUFFER->cell_speed_5_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_5_ve[ii + jj * GRID->global_cols] + params.omega
					* (d_equ[5] - BUFFER->tmp_cell_speed_5_ve[ii + jj * GRID->global_cols]);

				BUFFER->cell_speed_6_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_6_ve[ii + jj * GRID->global_cols] + params.omega
					* (d_equ[6] - BUFFER->tmp_cell_speed_6_ve[ii + jj * GRID->global_cols]);

				BUFFER->cell_speed_7_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_7_ve[ii + jj * GRID->global_cols] + params.omega
					* (d_equ[7] - BUFFER->tmp_cell_speed_7_ve[ii + jj * GRID->global_cols]);

				BUFFER->cell_speed_8_ve[ii + jj * GRID->global_cols] = BUFFER->tmp_cell_speed_8_ve[ii + jj * GRID->global_cols] + params.omega
					* (d_equ[8] - BUFFER->tmp_cell_speed_8_ve[ii + jj * GRID->global_cols]);
				//End: body of the collision function
			}
		}

	return EXIT_SUCCESS;
}

float av_velocity(t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, int* obstacles)
{
	float* cell_speed_0_ve = BUFFER->cell_speed_0_ve;
	float* cell_speed_1_ve = BUFFER->cell_speed_1_ve;
	float* cell_speed_2_ve = BUFFER->cell_speed_2_ve;
	float* cell_speed_3_ve = BUFFER->cell_speed_3_ve;
	float* cell_speed_4_ve = BUFFER->cell_speed_4_ve;
	float* cell_speed_5_ve = BUFFER->cell_speed_5_ve;
	float* cell_speed_6_ve = BUFFER->cell_speed_6_ve;
	float* cell_speed_7_ve = BUFFER->cell_speed_7_ve;
	float* cell_speed_8_ve = BUFFER->cell_speed_8_ve;

	float tot_u;          /* accumulated magnitudes of velocity for each cell */

						  /* initialise */
	tot_u = 0.f;

	/* loop over all non-blocked cells */
#pragma omp simd collapse(2) reduction(+: tot_u) aligned(cell_speed_0_ve, cell_speed_1_ve, cell_speed_2_ve, cell_speed_3_ve, cell_speed_4_ve, cell_speed_5_ve, cell_speed_6_ve, cell_speed_7_ve, cell_speed_8_ve: 32)
	for (int jj = GRID->first_row; jj <= GRID->last_row; jj++)
		for (int ii = GRID->first_col; ii <= GRID->last_col; ii++)
		{
			if (BUFFER->obstacles[ii + jj * GRID->global_cols] == 0)
			{
				/* local density total */
				float local_density = 0.f;
				local_density = BUFFER->cell_speed_0_ve[ii + jj * GRID->global_cols] + BUFFER->cell_speed_1_ve[ii + jj * GRID->global_cols] +
					BUFFER->cell_speed_2_ve[ii + jj * GRID->global_cols] + BUFFER->cell_speed_3_ve[ii + jj * GRID->global_cols] +
					BUFFER->cell_speed_4_ve[ii + jj * GRID->global_cols];
				local_density += BUFFER->cell_speed_5_ve[ii + jj * GRID->global_cols] + BUFFER->cell_speed_6_ve[ii + jj * GRID->global_cols] +
					BUFFER->cell_speed_7_ve[ii + jj * GRID->global_cols] + BUFFER->cell_speed_8_ve[ii + jj * GRID->global_cols];

				/* x-component of velocity */
				float u_x = (BUFFER->cell_speed_1_ve[ii + jj * GRID->global_cols]
					+ BUFFER->cell_speed_5_ve[ii + jj * GRID->global_cols]
					+ BUFFER->cell_speed_8_ve[ii + jj * GRID->global_cols]
					- (BUFFER->cell_speed_3_ve[ii + jj * GRID->global_cols]
						+ BUFFER->cell_speed_6_ve[ii + jj * GRID->global_cols]
						+ BUFFER->cell_speed_7_ve[ii + jj * GRID->global_cols]))
					/ local_density;
				/* compute y velocity component */
				float u_y = (BUFFER->cell_speed_2_ve[ii + jj * GRID->global_cols]
					+ BUFFER->cell_speed_5_ve[ii + jj * GRID->global_cols]
					+ BUFFER->cell_speed_6_ve[ii + jj * GRID->global_cols]
					- (BUFFER->cell_speed_4_ve[ii + jj * GRID->global_cols]
						+ BUFFER->cell_speed_7_ve[ii + jj * GRID->global_cols]
						+ BUFFER->cell_speed_8_ve[ii + jj * GRID->global_cols]))
					/ local_density;
				/* accumulate the norm of x- and y- velocity components */
				tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
				//End: body of the average velocity function
			}
		}

	return tot_u / GRID->global_non_obstacle_cells;
}

int calc_rank_non_obstacle_cells(t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param* params, int** obstacles_ptr) {

	int rank_non_obstacle_cells = 0;  /* no. of cells used in calculation */

									  /* loop over all non-blocked cells */
	for (int jj = GRID->first_row; jj <= GRID->last_row; jj++)
	{
		for (int ii = GRID->first_col; ii <= GRID->last_col; ii++)
		{
			/* ignore occupied cells */
			if (BUFFER->obstacles[ii + jj * GRID->global_cols] == 0)
			{
				++rank_non_obstacle_cells;
			}
		}
	}

	return rank_non_obstacle_cells;
}

int initialise(int* argc, char*** argv, t_rank_metadata* RANK, t_grid_meta_data* GRID,
	const char* paramfile, const char* obstaclefile,
	t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
	int** obstacles_ptr, float** av_vels_ptr, t_buffer_ptrs* BUFFER)
{
	char   message[1024];  /* message buffer */
	FILE*   fp;            /* file pointer */
	int    xx, yy;         /* generic array indices */
	int    blocked;        /* indicates whether a cell is blocked by an obstacle */
	int    retval;         /* to hold return value for checking */

						   /* open the parameter file */
	fp = fopen(paramfile, "r");

	if (fp == NULL)
	{
		sprintf(message, "could not open input parameter file: %s", paramfile);
		die(message, __LINE__, __FILE__);
	}

	/* read in the parameter values */
	retval = fscanf(fp, "%d\n", &(params->nx));

	if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

	retval = fscanf(fp, "%d\n", &(params->ny));

	if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

	retval = fscanf(fp, "%d\n", &(params->maxIters));

	if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

	retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

	if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

	retval = fscanf(fp, "%f\n", &(params->density));

	if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

	retval = fscanf(fp, "%f\n", &(params->accel));

	if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

	retval = fscanf(fp, "%f\n", &(params->omega));

	if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

	/* and close up the file */
	fclose(fp);

	/*
	** Allocate memory.
	**
	** Remember C is pass-by-value, so we need to
	** pass pointers into the initialise function.
	**
	** NB we are allocating a 1D array, so that the
	** memory will be contiguous.  We still want to
	** index this memory as if it were a (row major
	** ordered) 2D array, however.  We will perform
	** some arithmetic using the row and column
	** coordinates, inside the square brackets, when
	** we want to access elements of this array.
	**
	** Note also that we are using a structure to
	** hold an array of 'speeds'.  We will allocate
	** a 1D array of these structs.
	*/

	/* main grid */
	*cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

	if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

	/* 'helper' grid, used as scratch space */
	*tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

	if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

	/* the map of obstacles */
	*obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

	if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

	/* the map of obstacles */
	BUFFER->obstacles = _mm_malloc(sizeof(int) * (params->ny * params->nx), 32);

	if (BUFFER->obstacles == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

	/* initialise densities */
	float w0 = params->density * 4.f / 9.f;
	float w1 = params->density / 9.f;
	float w2 = params->density / 36.f;

	for (int jj = 0; jj < params->ny; jj++)
	{
		for (int ii = 0; ii < params->nx; ii++)
		{
			/* centre */
			(*cells_ptr)[ii + jj * params->nx].speeds[0] = w0;
			/* axis directions */
			(*cells_ptr)[ii + jj * params->nx].speeds[1] = w1;
			(*cells_ptr)[ii + jj * params->nx].speeds[2] = w1;
			(*cells_ptr)[ii + jj * params->nx].speeds[3] = w1;
			(*cells_ptr)[ii + jj * params->nx].speeds[4] = w1;
			/* diagonals */
			(*cells_ptr)[ii + jj * params->nx].speeds[5] = w2;
			(*cells_ptr)[ii + jj * params->nx].speeds[6] = w2;
			(*cells_ptr)[ii + jj * params->nx].speeds[7] = w2;
			(*cells_ptr)[ii + jj * params->nx].speeds[8] = w2;
		}
	}

	/* first set all cells in obstacle array to zero */
	for (int jj = 0; jj < params->ny; jj++)
	{
		for (int ii = 0; ii < params->nx; ii++)
		{
			BUFFER->obstacles[ii + jj * params->nx] = 0;
			(*obstacles_ptr)[ii + jj * params->nx] = 0;
		}
	}

	/* open the obstacle data file */
	fp = fopen(obstaclefile, "r");

	if (fp == NULL)
	{
		sprintf(message, "could not open input obstacles file: %s", obstaclefile);
		die(message, __LINE__, __FILE__);
	}

	/* read-in the blocked cells list */
	while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
	{
		/* some checks */
		if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

		if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

		if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

		if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

		/* assign to array */
		BUFFER->obstacles[xx + yy * params->nx] = blocked;
		(*obstacles_ptr)[xx + yy * params->nx] = blocked;
	}

	/* and close the file */
	fclose(fp);

	/*
	** allocate space to hold a record of the avarage velocities computed
	** at each timestep
	*/
	*av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

	initialize_MPI(argc, argv);
	initialize_RANK(RANK);
	initialize_GRID(RANK, GRID, BUFFER, params, obstacles_ptr, w0, w1, w2);

	return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
	int** obstacles_ptr, float** av_vels_ptr, t_buffer_ptrs* BUFFER)
{
	/*
	** free up allocated memory
	*/
	free(*cells_ptr);
	*cells_ptr = NULL;

	free(*tmp_cells_ptr);
	*tmp_cells_ptr = NULL;

	free(*obstacles_ptr);
	*obstacles_ptr = NULL;

	free(*av_vels_ptr);
	*av_vels_ptr = NULL;

	_mm_free(BUFFER->cell_speed_0_ve);
	BUFFER->cell_speed_0_ve = NULL;
	_mm_free(BUFFER->cell_speed_1_ve);
	BUFFER->cell_speed_1_ve = NULL;
	_mm_free(BUFFER->cell_speed_2_ve);
	BUFFER->cell_speed_2_ve = NULL;
	_mm_free(BUFFER->cell_speed_3_ve);
	BUFFER->cell_speed_3_ve = NULL;
	_mm_free(BUFFER->cell_speed_4_ve);
	BUFFER->cell_speed_4_ve = NULL;
	_mm_free(BUFFER->cell_speed_5_ve);
	BUFFER->cell_speed_5_ve = NULL;
	_mm_free(BUFFER->cell_speed_6_ve);
	BUFFER->cell_speed_6_ve = NULL;
	_mm_free(BUFFER->cell_speed_7_ve);
	BUFFER->cell_speed_7_ve = NULL;
	_mm_free(BUFFER->cell_speed_8_ve);
	BUFFER->cell_speed_8_ve = NULL;

	_mm_free(BUFFER->tmp_cell_speed_0_ve);
	BUFFER->tmp_cell_speed_0_ve = NULL;
	_mm_free(BUFFER->tmp_cell_speed_1_ve);
	BUFFER->tmp_cell_speed_1_ve = NULL;
	_mm_free(BUFFER->tmp_cell_speed_2_ve);
	BUFFER->tmp_cell_speed_2_ve = NULL;
	_mm_free(BUFFER->tmp_cell_speed_3_ve);
	BUFFER->tmp_cell_speed_3_ve = NULL;
	_mm_free(BUFFER->tmp_cell_speed_4_ve);
	BUFFER->tmp_cell_speed_4_ve = NULL;
	_mm_free(BUFFER->tmp_cell_speed_5_ve);
	BUFFER->tmp_cell_speed_5_ve = NULL;
	_mm_free(BUFFER->tmp_cell_speed_6_ve);
	BUFFER->tmp_cell_speed_6_ve = NULL;
	_mm_free(BUFFER->tmp_cell_speed_7_ve);
	BUFFER->tmp_cell_speed_7_ve = NULL;
	_mm_free(BUFFER->tmp_cell_speed_8_ve);
	BUFFER->tmp_cell_speed_8_ve = NULL;

	_mm_free(BUFFER->obstacles);
	BUFFER->obstacles = NULL;

	_mm_free(BUFFER->partialcellbuf);

	return EXIT_SUCCESS;
}


float calc_reynolds(t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param params, t_speed* cells, int* obstacles)
{
	float rank_av_vel = 0;			//for reynolds number
	float iteration_av_vel = 0;			//for reynolds number
	const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
	rank_av_vel = av_velocity(GRID, BUFFER, params, cells, obstacles);
	MPI_Allreduce(&rank_av_vel, &iteration_av_vel, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	return iteration_av_vel * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
	float total = 0.f;  /* accumulator */

	for (int jj = 0; jj < params.ny; jj++)
	{
		for (int ii = 0; ii < params.nx; ii++)
		{
			for (int kk = 0; kk < NSPEEDS; kk++)
			{
				total += cells[ii + jj * params.nx].speeds[kk];
			}
		}
	}

	return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER)
{
	FILE* fp;                     /* file pointer */
	const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
	float local_density;         /* per grid cell sum of densities */
	float pressure;              /* fluid pressure in grid cell */
	float u_x;                   /* x-component of velocity in grid cell */
	float u_y;                   /* y-component of velocity in grid cell */
	float u;                     /* norm--root of summed squares--of u_x and u_y */

	fp = fopen(FINALSTATEFILE, "w");

	if (fp == NULL)
	{
		die("could not open file output file", __LINE__, __FILE__);
	}

	for (int32_t jj = 0; jj < GRID->global_rows; jj++)
		for (int32_t ii = 0; ii < GRID->global_cols; ii++)
		{
			/* an occupied cell */
			if (BUFFER->obstacles[ii + jj * GRID->global_cols])
			{
				u_x = u_y = u = 0.f;
				pressure = params.density * c_sq;
			}
			/* no obstacle */
			else
			{
				local_density = 0.f;
				local_density = BUFFER->cell_speed_0_ve[ii + jj * GRID->global_cols] + BUFFER->cell_speed_1_ve[ii + jj * GRID->global_cols] +
					BUFFER->cell_speed_2_ve[ii + jj * GRID->global_cols] + BUFFER->cell_speed_3_ve[ii + jj * GRID->global_cols] +
					BUFFER->cell_speed_4_ve[ii + jj * GRID->global_cols];
				local_density += BUFFER->cell_speed_5_ve[ii + jj * GRID->global_cols] + BUFFER->cell_speed_6_ve[ii + jj * GRID->global_cols] +
					BUFFER->cell_speed_7_ve[ii + jj * GRID->global_cols] + BUFFER->cell_speed_8_ve[ii + jj * GRID->global_cols];

				/* compute x velocity component */
				u_x = (BUFFER->cell_speed_1_ve[ii + jj * GRID->global_cols]
					+ BUFFER->cell_speed_5_ve[ii + jj * GRID->global_cols]
					+ BUFFER->cell_speed_8_ve[ii + jj * GRID->global_cols]
					- (BUFFER->cell_speed_3_ve[ii + jj * GRID->global_cols]
						+ BUFFER->cell_speed_6_ve[ii + jj * GRID->global_cols]
						+ BUFFER->cell_speed_7_ve[ii + jj * GRID->global_cols]))
					/ local_density;
				/* compute y velocity component */
				u_y = (BUFFER->cell_speed_2_ve[ii + jj * GRID->global_cols]
					+ BUFFER->cell_speed_5_ve[ii + jj * GRID->global_cols]
					+ BUFFER->cell_speed_6_ve[ii + jj * GRID->global_cols]
					- (BUFFER->cell_speed_4_ve[ii + jj * GRID->global_cols]
						+ BUFFER->cell_speed_7_ve[ii + jj * GRID->global_cols]
						+ BUFFER->cell_speed_8_ve[ii + jj * GRID->global_cols]))
					/ local_density;
				/* compute norm of velocity */
				u = sqrtf((u_x * u_x) + (u_y * u_y));
				/* compute pressure */
				pressure = local_density * c_sq;
			}

			/* write to file */
			fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, BUFFER->obstacles[ii * GRID->global_cols + jj]);
		}

	fclose(fp);

	fp = fopen(AVVELSFILE, "w");

	if (fp == NULL)
	{
		die("could not open file output file", __LINE__, __FILE__);
	}

	for (int ii = 0; ii < params.maxIters; ii++)
	{
		fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
	}

	fclose(fp);

	return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
	fprintf(stderr, "Error at line %d of file %s:\n", line, file);
	fprintf(stderr, "%s\n", message);
	fflush(stderr);
	exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
	fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
	exit(EXIT_FAILURE);
}

void initialize_MPI(int* argc, char*** argv)
{
	MPI_Init(argc, argv);
	//verify successful initialization of the MPI environment
	int mpi_init_flag;
	MPI_Initialized(&mpi_init_flag); /* check whether the initialisation was successful */
	if (mpi_init_flag != 1)
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	return;
}

void initialize_RANK(t_rank_metadata* RANK) {

	MPI_Comm_size(MPI_COMM_WORLD, &RANK->size);
	MPI_Comm_rank(MPI_COMM_WORLD, &RANK->rank);
	RANK->rank_ahead = ((RANK->rank + 1) == RANK->size) ? 0 : RANK->rank + 1;	//South = Ahead
	RANK->rank_before = ((RANK->rank - 1) < 0) ? RANK->size - 1 : RANK->rank - 1; //North = Before
	RANK->tag = 0;
	RANK->thread_limit = 32;
	omp_set_num_threads(RANK->thread_limit);
	//RANK->status = NULL; //Will it make a difference to uncomment this?	
	return;
}

void initialize_GRID(t_rank_metadata* RANK, t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, const t_param* params, int** obstacles_ptr, float w0, float w1, float w2) {

	int rank_non_obstacle_cells = 0;
	int global_non_obstacle_cells = 0;

	GRID->global_rows = params->ny;
	GRID->global_cols = params->nx;
	GRID->grid_extent = GRID->global_rows * GRID->global_cols;
	GRID->buffer_extent = GRID->global_cols;
	GRID->extra_rows = GRID->global_rows % RANK->size;
	GRID->local_rows = GRID->global_rows / RANK->size;
	GRID->first_row = GRID->local_rows * RANK->rank;
	GRID->last_row = (GRID->local_rows * (RANK->rank + 1)) - 1;
	GRID->first_row_neighbour = (GRID->first_row == 0) ? GRID->global_rows - 1 : GRID->first_row - 1;
	GRID->last_row_neighbour = (GRID->last_row == (GRID->global_rows - 1)) ? 0 : GRID->last_row + 1;
	GRID->first_col = 0;
	GRID->last_col = GRID->global_cols - 1;
	GRID->first_row_start_idx = (RANK->rank == 0) ? 0 : GRID->first_row * GRID->global_cols;
	GRID->last_row_start_idx = GRID->last_row * GRID->global_cols;
	if (RANK->rank != MASTER_RANK)
	{
		GRID->first_row += GRID->extra_rows;
		GRID->first_row_neighbour += GRID->extra_rows;
	}

	GRID->last_row += GRID->extra_rows;
	GRID->last_row_neighbour = (GRID->last_row == (GRID->global_rows - 1)) ? 0 : GRID->last_row + 1;

	initialize_BUFFER(GRID, BUFFER, w0, w1, w2);
	rank_non_obstacle_cells = calc_rank_non_obstacle_cells(GRID, BUFFER, params, obstacles_ptr);
	MPI_Allreduce(&rank_non_obstacle_cells, &global_non_obstacle_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	GRID->global_non_obstacle_cells = global_non_obstacle_cells;
	return;
}

void initialize_BUFFER(t_grid_meta_data* GRID, t_buffer_ptrs* BUFFER, float w0, float w1, float w2) {

	BUFFER->cell_speed_0_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->cell_speed_1_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->cell_speed_2_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->cell_speed_3_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->cell_speed_4_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->cell_speed_5_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->cell_speed_6_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->cell_speed_7_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->cell_speed_8_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);

	if (BUFFER->cell_speed_0_ve == NULL || BUFFER->cell_speed_1_ve == NULL || BUFFER->cell_speed_2_ve == NULL ||
		BUFFER->cell_speed_3_ve == NULL || BUFFER->cell_speed_4_ve == NULL || BUFFER->cell_speed_5_ve == NULL ||
		BUFFER->cell_speed_6_ve == NULL || BUFFER->cell_speed_7_ve == NULL || BUFFER->cell_speed_8_ve == NULL)
		die("cannot allocate memory for cells SoA", __LINE__, __FILE__);

	/*SoA equivalent of helper grid*/
	BUFFER->tmp_cell_speed_0_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->tmp_cell_speed_1_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->tmp_cell_speed_2_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->tmp_cell_speed_3_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->tmp_cell_speed_4_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->tmp_cell_speed_5_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->tmp_cell_speed_6_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->tmp_cell_speed_7_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);
	BUFFER->tmp_cell_speed_8_ve = (float*)_mm_malloc(sizeof(float) * (GRID->grid_extent), 32);

	if (BUFFER->tmp_cell_speed_0_ve == NULL || BUFFER->tmp_cell_speed_1_ve == NULL || BUFFER->tmp_cell_speed_2_ve == NULL ||
		BUFFER->tmp_cell_speed_3_ve == NULL || BUFFER->tmp_cell_speed_4_ve == NULL || BUFFER->tmp_cell_speed_5_ve == NULL ||
		BUFFER->tmp_cell_speed_6_ve == NULL || BUFFER->tmp_cell_speed_7_ve == NULL || BUFFER->tmp_cell_speed_8_ve == NULL)
		die("cannot allocate memory for tmp_cell SoA", __LINE__, __FILE__);

	BUFFER->partialcellbuf = (float*)_mm_malloc(sizeof(float) * (GRID->local_rows * GRID->global_cols), 32);


	for (int32_t jj = 0; jj < GRID->global_rows; jj++)
	{
		for (int32_t ii = 0; ii < GRID->global_cols; ii++)
		{
			/* centre */
			BUFFER->cell_speed_0_ve[ii + jj * GRID->global_cols] = w0;
			/* axis directions */
			BUFFER->cell_speed_1_ve[ii + jj * GRID->global_cols] = w1;
			BUFFER->cell_speed_2_ve[ii + jj * GRID->global_cols] = w1;
			BUFFER->cell_speed_3_ve[ii + jj * GRID->global_cols] = w1;
			BUFFER->cell_speed_4_ve[ii + jj * GRID->global_cols] = w1;

			/* diagonals */
			BUFFER->cell_speed_5_ve[ii + jj * GRID->global_cols] = w2;
			BUFFER->cell_speed_6_ve[ii + jj * GRID->global_cols] = w2;
			BUFFER->cell_speed_7_ve[ii + jj * GRID->global_cols] = w2;
			BUFFER->cell_speed_8_ve[ii + jj * GRID->global_cols] = w2;

			//all tmp_cells
			BUFFER->tmp_cell_speed_0_ve[ii + jj * GRID->global_cols] = 0;
			BUFFER->tmp_cell_speed_1_ve[ii + jj * GRID->global_cols] = 0;
			BUFFER->tmp_cell_speed_2_ve[ii + jj * GRID->global_cols] = 0;
			BUFFER->tmp_cell_speed_3_ve[ii + jj * GRID->global_cols] = 0;
			BUFFER->tmp_cell_speed_4_ve[ii + jj * GRID->global_cols] = 0;
			BUFFER->tmp_cell_speed_5_ve[ii + jj * GRID->global_cols] = 0;
			BUFFER->tmp_cell_speed_6_ve[ii + jj * GRID->global_cols] = 0;
			BUFFER->tmp_cell_speed_7_ve[ii + jj * GRID->global_cols] = 0;
			BUFFER->tmp_cell_speed_8_ve[ii + jj * GRID->global_cols] = 0;
		}
	}

	for (int jj = 0; jj < GRID->local_rows; jj++)
		for (int ii = 0; ii < GRID->global_cols; ii++)
			BUFFER->partialcellbuf[ii + jj * GRID->global_cols] = 0;

	return;
}

void finalize_MPI() {

	MPI_Finalize();
	return;
}
