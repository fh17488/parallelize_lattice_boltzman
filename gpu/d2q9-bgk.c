#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mm_malloc.h>

#define NSPEEDS 9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

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

	int* obstacles;    /* grid indicating which cells are blocked */

	float* av_vels;

} t_buffer_ptrs;

typedef struct
{
	int    global_cols;            /* no. of cells in x-direction */
	int    global_rows;            /* no. of cells in y-direction */
	int	   global_non_obstacle_cells;
	int    maxIters;      /* no. of iterations */
	int    reynolds_dim;  /* dimension for Reynolds number */
	float density;       /* density per link */
	float accel;         /* density redistribution */
	float omega;         /* relaxation parameter */
} t_param;

int initialise(const char* paramfile, const char* obstaclefile, t_param* params, t_buffer_ptrs* BUFFER);
void initialize_BUFFER(t_param* params, t_buffer_ptrs* BUFFER, const char* obstaclefile, float w0, float w1, float w2);
void finalize(t_param* params, t_buffer_ptrs* BUFFER);

float av_velocity(t_param* params, t_buffer_ptrs* BUFFER);

int write_values(t_param* params, t_buffer_ptrs* BUFFER);

float calc_reynolds(t_param* params, t_buffer_ptrs* BUFFER);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

int main(int argc, char* argv[])
{
	float cumulative_grid_val = 0;
	struct timeval timstr;        /* structure to hold elapsed time */
	struct rusage ru;             /* structure to hold CPU time--system and user */
	double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
	double usrtime = 0;		/* floating point number to record elapsed user CPU time */
	double systime = 0;
	t_buffer_ptrs BUFFER;
	t_param params;
	char*    paramfile = NULL;    /* name of the input parameter file */
	char*    obstaclefile = NULL; /* name of a the input obstacle file */

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

	initialise(paramfile, obstaclefile, &params, &BUFFER);

	float* cell_speed_0_ve = BUFFER.cell_speed_0_ve;
	float* cell_speed_1_ve = BUFFER.cell_speed_1_ve;
	float* cell_speed_2_ve = BUFFER.cell_speed_2_ve;
	float* cell_speed_3_ve = BUFFER.cell_speed_3_ve;
	float* cell_speed_4_ve = BUFFER.cell_speed_4_ve;
	float* cell_speed_5_ve = BUFFER.cell_speed_5_ve;
	float* cell_speed_6_ve = BUFFER.cell_speed_6_ve;
	float* cell_speed_7_ve = BUFFER.cell_speed_7_ve;
	float* cell_speed_8_ve = BUFFER.cell_speed_8_ve;

	float* tmp_cell_speed_0_ve = BUFFER.tmp_cell_speed_0_ve;
	float* tmp_cell_speed_1_ve = BUFFER.tmp_cell_speed_1_ve;
	float* tmp_cell_speed_2_ve = BUFFER.tmp_cell_speed_2_ve;
	float* tmp_cell_speed_3_ve = BUFFER.tmp_cell_speed_3_ve;
	float* tmp_cell_speed_4_ve = BUFFER.tmp_cell_speed_4_ve;
	float* tmp_cell_speed_5_ve = BUFFER.tmp_cell_speed_5_ve;
	float* tmp_cell_speed_6_ve = BUFFER.tmp_cell_speed_6_ve;
	float* tmp_cell_speed_7_ve = BUFFER.tmp_cell_speed_7_ve;
	float* tmp_cell_speed_8_ve = BUFFER.tmp_cell_speed_8_ve;

	int* obstacles = BUFFER.obstacles;

	int maximum_iterations = params.maxIters;
	int grid_extent = params.global_rows * params.global_cols;
	int global_rows = params.global_rows;
	int global_cols = params.global_cols;
	int global_non_obstacle_cells = params.global_non_obstacle_cells;

	float* av_vels = BUFFER.av_vels;

	/* compute weighting factors for accelerate flow*/
	float w1_af = params.density * params.accel / 9.f;
	float w2_af = params.density * params.accel / 36.f;

	/* modify the 2nd row of the grid for accelerate flow*/
	int jj_af = params.global_rows - 2;

	float omega = params.omega;
	float c_sq = 1.f / 3.f; /* square of speed of sound */
	float w0 = 4.f / 9.f;  /* weighting factor */
	float w1 = 1.f / 9.f;  /* weighting factor */
	float w2 = 1.f / 36.f; /* weighting factor */

	int A[1] = { -1 };

#pragma omp target enter data map(to: A, maximum_iterations, global_rows, global_cols, w1_af, w2_af, jj_af, omega, c_sq, w0, w1, w2, av_vels[0:maximum_iterations], obstacles[0:grid_extent], cell_speed_0_ve[0:grid_extent], cell_speed_1_ve[0:grid_extent], cell_speed_2_ve[0:grid_extent], cell_speed_3_ve[0:grid_extent], cell_speed_4_ve[0:grid_extent], cell_speed_5_ve[0:grid_extent], cell_speed_6_ve[0:grid_extent], cell_speed_7_ve[0:grid_extent], cell_speed_8_ve[0:grid_extent], tmp_cell_speed_0_ve[0:grid_extent], tmp_cell_speed_1_ve[0:grid_extent], tmp_cell_speed_2_ve[0:grid_extent], tmp_cell_speed_3_ve[0:grid_extent], tmp_cell_speed_4_ve[0:grid_extent], tmp_cell_speed_5_ve[0:grid_extent], tmp_cell_speed_6_ve[0:grid_extent], tmp_cell_speed_7_ve[0:grid_extent], tmp_cell_speed_8_ve[0:grid_extent])
	{}

	/* iterate for maxIters timesteps */
	gettimeofday(&timstr, NULL);
	tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

	//start iterations
	for (int tt = 0; tt < maximum_iterations; tt++)
	{
		//accelerate flow
#pragma omp target teams distribute parallel for 
		for (int ii = 0; ii < global_cols; ii++)
		{
			if (obstacles[ii + jj_af * global_cols] == 0)
			{
				//Start: body of the accelerate flow function
				//These values are being calculated for the next iteration of the timestep
				if ((cell_speed_3_ve[ii + jj_af * global_cols] - w1_af) > 0.f
					&& (cell_speed_6_ve[ii + jj_af * global_cols] - w2_af) > 0.f
					&& (cell_speed_7_ve[ii + jj_af * global_cols] - w2_af) > 0.f)
				{
					/* increase 'east-side' densities */
					cell_speed_1_ve[ii + jj_af * global_cols] += w1_af;
					cell_speed_5_ve[ii + jj_af * global_cols] += w2_af;
					cell_speed_8_ve[ii + jj_af * global_cols] += w2_af;
					/* decrease 'west-side' densities */
					cell_speed_3_ve[ii + jj_af * global_cols] -= w1_af;
					cell_speed_6_ve[ii + jj_af * global_cols] -= w2_af;
					cell_speed_7_ve[ii + jj_af * global_cols] -= w2_af;
				}
				//End: body of the accelerate flow function			
			}
		}

		//propagate
#pragma omp target teams distribute parallel for collapse(2) 
		for (int jj = 0; jj < global_rows; jj++)
			for (int ii = 0; ii < global_cols; ii++)
			{
				//determine indices of axis-direction neighbours
				//respecting periodic boundary conditions (wrap around)
				int y_n = (jj + 1) % global_rows;
				int x_e = (ii + 1) % global_cols;
				int y_s = (jj == 0) ? (jj + global_rows - 1) : (jj - 1);
				int x_w = (ii == 0) ? (ii + global_cols - 1) : (ii - 1);

				// propagate densities from neighbouring cells, following
				// appropriate directions of travel and writing into
				// scratch space grid

				tmp_cell_speed_0_ve[ii + jj * global_cols] = cell_speed_0_ve[ii + jj * global_cols]; // central cell, no movement
				tmp_cell_speed_1_ve[ii + jj * global_cols] = cell_speed_1_ve[x_w + jj * global_cols]; // east			
				tmp_cell_speed_2_ve[ii + jj * global_cols] = cell_speed_2_ve[ii + y_s * global_cols]; // north
				tmp_cell_speed_3_ve[ii + jj * global_cols] = cell_speed_3_ve[x_e + jj * global_cols]; // west
				tmp_cell_speed_4_ve[ii + jj * global_cols] = cell_speed_4_ve[ii + y_n * global_cols]; // south
				tmp_cell_speed_5_ve[ii + jj * global_cols] = cell_speed_5_ve[x_w + y_s * global_cols]; // north-east
				tmp_cell_speed_6_ve[ii + jj * global_cols] = cell_speed_6_ve[x_e + y_s * global_cols]; // north-west
				tmp_cell_speed_7_ve[ii + jj * global_cols] = cell_speed_7_ve[x_e + y_n * global_cols]; // south-west
				tmp_cell_speed_8_ve[ii + jj * global_cols] = cell_speed_8_ve[x_w + y_n * global_cols]; // south-east
			}

		//rebound and collision 
#pragma omp target teams distribute parallel for collapse(2) 
		for (int jj = 0; jj < global_rows; jj++)
			for (int ii = 0; ii < global_cols; ii++)
			{
				/* if the cell contains an obstacle */
				if (obstacles[ii + jj * global_cols] == 1)
				{
					/* called after propagate, so taking values from scratch space
					** mirroring, and writing into main grid */
					cell_speed_1_ve[ii + jj * global_cols] = tmp_cell_speed_3_ve[ii + jj * global_cols];
					cell_speed_2_ve[ii + jj * global_cols] = tmp_cell_speed_4_ve[ii + jj * global_cols];
					cell_speed_3_ve[ii + jj * global_cols] = tmp_cell_speed_1_ve[ii + jj * global_cols];
					cell_speed_4_ve[ii + jj * global_cols] = tmp_cell_speed_2_ve[ii + jj * global_cols];
					cell_speed_5_ve[ii + jj * global_cols] = tmp_cell_speed_7_ve[ii + jj * global_cols];
					cell_speed_6_ve[ii + jj * global_cols] = tmp_cell_speed_8_ve[ii + jj * global_cols];
					cell_speed_7_ve[ii + jj * global_cols] = tmp_cell_speed_5_ve[ii + jj * global_cols];
					cell_speed_8_ve[ii + jj * global_cols] = tmp_cell_speed_6_ve[ii + jj * global_cols];
				}
				else if (obstacles[ii + jj * global_cols] == 0)
				{
					/* compute local density total */
					float local_density = 0.f;
					local_density = tmp_cell_speed_0_ve[ii + jj * global_cols] + tmp_cell_speed_1_ve[ii + jj * global_cols] +
						tmp_cell_speed_2_ve[ii + jj * global_cols] + tmp_cell_speed_3_ve[ii + jj * global_cols] +
						tmp_cell_speed_4_ve[ii + jj * global_cols];
					local_density += tmp_cell_speed_5_ve[ii + jj * global_cols] + tmp_cell_speed_6_ve[ii + jj * global_cols] +
						tmp_cell_speed_7_ve[ii + jj * global_cols] + tmp_cell_speed_8_ve[ii + jj * global_cols];

					/* compute x velocity component */
					float u_x = (tmp_cell_speed_1_ve[ii + jj * global_cols]
						+ tmp_cell_speed_5_ve[ii + jj * global_cols]
						+ tmp_cell_speed_8_ve[ii + jj * global_cols]
						- (tmp_cell_speed_3_ve[ii + jj * global_cols]
							+ tmp_cell_speed_6_ve[ii + jj * global_cols]
							+ tmp_cell_speed_7_ve[ii + jj * global_cols]))
						/ local_density;
					/* compute y velocity component */
					float u_y = (tmp_cell_speed_2_ve[ii + jj * global_cols]
						+ tmp_cell_speed_5_ve[ii + jj * global_cols]
						+ tmp_cell_speed_6_ve[ii + jj * global_cols]
						- (tmp_cell_speed_4_ve[ii + jj * global_cols]
							+ tmp_cell_speed_7_ve[ii + jj * global_cols]
							+ tmp_cell_speed_8_ve[ii + jj * global_cols]))
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
					cell_speed_0_ve[ii + jj * global_cols] = tmp_cell_speed_0_ve[ii + jj * global_cols] + omega
						* (d_equ[0] - tmp_cell_speed_0_ve[ii + jj * global_cols]);

					cell_speed_1_ve[ii + jj * global_cols] = tmp_cell_speed_1_ve[ii + jj * global_cols] + omega
						* (d_equ[1] - tmp_cell_speed_1_ve[ii + jj * global_cols]);

					cell_speed_2_ve[ii + jj * global_cols] = tmp_cell_speed_2_ve[ii + jj * global_cols] + omega
						* (d_equ[2] - tmp_cell_speed_2_ve[ii + jj * global_cols]);

					cell_speed_3_ve[ii + jj * global_cols] = tmp_cell_speed_3_ve[ii + jj * global_cols] + omega
						* (d_equ[3] - tmp_cell_speed_3_ve[ii + jj * global_cols]);

					cell_speed_4_ve[ii + jj * global_cols] = tmp_cell_speed_4_ve[ii + jj * global_cols] + omega
						* (d_equ[4] - tmp_cell_speed_4_ve[ii + jj * global_cols]);

					cell_speed_5_ve[ii + jj * global_cols] = tmp_cell_speed_5_ve[ii + jj * global_cols] + omega
						* (d_equ[5] - tmp_cell_speed_5_ve[ii + jj * global_cols]);

					cell_speed_6_ve[ii + jj * global_cols] = tmp_cell_speed_6_ve[ii + jj * global_cols] + omega
						* (d_equ[6] - tmp_cell_speed_6_ve[ii + jj * global_cols]);

					cell_speed_7_ve[ii + jj * global_cols] = tmp_cell_speed_7_ve[ii + jj * global_cols] + omega
						* (d_equ[7] - tmp_cell_speed_7_ve[ii + jj * global_cols]);

					cell_speed_8_ve[ii + jj * global_cols] = tmp_cell_speed_8_ve[ii + jj * global_cols] + omega
						* (d_equ[8] - tmp_cell_speed_8_ve[ii + jj * global_cols]);
					//End: body of the collision function
				}
			}


		//average velocity 
		float tot_u_local = 0.f;
#pragma omp target teams distribute parallel for collapse(2) reduction(+:tot_u_local) map(tofrom: tot_u_local) 
		for (int jj = 0; jj < global_rows; jj++)
			for (int ii = 0; ii < global_cols; ii++)
			{
				/* if the cell contains an obstacle */
				if (obstacles[ii + jj * global_cols] == 0)
				{
					/* local density total */
					float local_density = 0.f;
					local_density = cell_speed_0_ve[ii + jj * global_cols] + cell_speed_1_ve[ii + jj * global_cols] +
						cell_speed_2_ve[ii + jj * global_cols] + cell_speed_3_ve[ii + jj * global_cols] +
						cell_speed_4_ve[ii + jj * global_cols];
					local_density += cell_speed_5_ve[ii + jj * global_cols] + cell_speed_6_ve[ii + jj * global_cols] +
						cell_speed_7_ve[ii + jj * global_cols] + cell_speed_8_ve[ii + jj * global_cols];

					/* x-component of velocity */
					float u_x = (cell_speed_1_ve[ii + jj * global_cols]
						+ cell_speed_5_ve[ii + jj * global_cols]
						+ cell_speed_8_ve[ii + jj * global_cols]
						- (cell_speed_3_ve[ii + jj * global_cols]
							+ cell_speed_6_ve[ii + jj * global_cols]
							+ cell_speed_7_ve[ii + jj * global_cols]))
						/ local_density;

					/* compute y velocity component */
					float u_y = (cell_speed_2_ve[ii + jj * global_cols]
						+ cell_speed_5_ve[ii + jj * global_cols]
						+ cell_speed_6_ve[ii + jj * global_cols]
						- (cell_speed_4_ve[ii + jj * global_cols]
							+ cell_speed_7_ve[ii + jj * global_cols]
							+ cell_speed_8_ve[ii + jj * global_cols]))
						/ local_density;

					/* accumulate the norm of x- and y- velocity components */
					tot_u_local += sqrtf((u_x * u_x) + (u_y * u_y));
					//End: body of the average velocity function
				}
			}

		BUFFER.av_vels[tt] = tot_u_local / global_non_obstacle_cells;
		//printf("\titeration:%d\tav_vel:%f\n", tt, av_vels[tt]);
	}

#pragma omp target update from(cell_speed_0_ve[0:grid_extent], cell_speed_1_ve[0:grid_extent], cell_speed_2_ve[0:grid_extent], cell_speed_3_ve[0:grid_extent], cell_speed_4_ve[0:grid_extent], cell_speed_5_ve[0:grid_extent], cell_speed_6_ve[0:grid_extent], cell_speed_7_ve[0:grid_extent], cell_speed_8_ve[0:grid_extent])
	{}
	//end iteration

	gettimeofday(&timstr, NULL);
	toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
	getrusage(RUSAGE_SELF, &ru);
	timstr = ru.ru_utime;
	usrtime = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
	timstr = ru.ru_stime;
	systime = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

	BUFFER.cell_speed_0_ve = cell_speed_0_ve;
	BUFFER.cell_speed_1_ve = cell_speed_1_ve;
	BUFFER.cell_speed_2_ve = cell_speed_2_ve;
	BUFFER.cell_speed_3_ve = cell_speed_3_ve;
	BUFFER.cell_speed_4_ve = cell_speed_4_ve;
	BUFFER.cell_speed_5_ve = cell_speed_5_ve;
	BUFFER.cell_speed_6_ve = cell_speed_6_ve;
	BUFFER.cell_speed_7_ve = cell_speed_7_ve;
	BUFFER.cell_speed_8_ve = cell_speed_8_ve;


	printf("Reynolds number:\t\t%.12E\n", calc_reynolds(&params, &BUFFER));
	printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
	printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtime);
	printf("Elapsed system CPU time:\t%.6lf (s)\n", systime);

	write_values(&params, &BUFFER);
	finalize(&params, &BUFFER);

	return EXIT_SUCCESS;
}

int initialise(const char* paramfile, const char* obstaclefile, t_param* params, t_buffer_ptrs* BUFFER)
{
	printf("entered initialize\n");
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
	retval = fscanf(fp, "%d\n", &(params->global_cols));

	if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

	retval = fscanf(fp, "%d\n", &(params->global_rows));

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

	/* initialise densities */
	float w0 = params->density * 4.f / 9.f;
	float w1 = params->density / 9.f;
	float w2 = params->density / 36.f;

	initialize_BUFFER(params, BUFFER, obstaclefile, w0, w1, w2);

	return EXIT_SUCCESS;
}
void initialize_BUFFER(t_param* params, t_buffer_ptrs* BUFFER, const char* obstaclefile, float w0, float w1, float w2)
{
	char   message[1024];  /* message buffer */
	FILE*   fp;
	int    xx, yy;         /* generic array indices */
	int    blocked;        /* indicates whether a cell is blocked by an obstacle */
	int    retval;         /* to hold return value for checking */
	int	   maxIterations = params->maxIters;
	int	   grid_extent = params->global_rows * params->global_cols;
	int	   global_obstacle_cells = 0;

	BUFFER->cell_speed_0_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->cell_speed_1_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->cell_speed_2_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->cell_speed_3_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->cell_speed_4_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->cell_speed_5_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->cell_speed_6_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->cell_speed_7_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->cell_speed_8_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);

	BUFFER->tmp_cell_speed_0_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->tmp_cell_speed_1_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->tmp_cell_speed_2_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->tmp_cell_speed_3_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->tmp_cell_speed_4_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->tmp_cell_speed_5_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->tmp_cell_speed_6_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->tmp_cell_speed_7_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);
	BUFFER->tmp_cell_speed_8_ve = (float*)_mm_malloc(sizeof(float) * (grid_extent), 32);

	BUFFER->obstacles = (int*)_mm_malloc(sizeof(int) * (grid_extent), 32);

	BUFFER->av_vels = (float*)malloc(sizeof(float) * maxIterations);

	// Have to place all pointers into local variables
	// for OpenMP to accept them in mapping clauses
	float *cell_speed_0_ve = BUFFER->cell_speed_0_ve;
	float *cell_speed_1_ve = BUFFER->cell_speed_1_ve;
	float *cell_speed_2_ve = BUFFER->cell_speed_2_ve;
	float *cell_speed_3_ve = BUFFER->cell_speed_3_ve;
	float *cell_speed_4_ve = BUFFER->cell_speed_4_ve;
	float *cell_speed_5_ve = BUFFER->cell_speed_5_ve;
	float *cell_speed_6_ve = BUFFER->cell_speed_6_ve;
	float *cell_speed_7_ve = BUFFER->cell_speed_7_ve;
	float *cell_speed_8_ve = BUFFER->cell_speed_8_ve;

	float *tmp_cell_speed_0_ve = BUFFER->tmp_cell_speed_0_ve;
	float *tmp_cell_speed_1_ve = BUFFER->tmp_cell_speed_1_ve;
	float *tmp_cell_speed_2_ve = BUFFER->tmp_cell_speed_2_ve;
	float *tmp_cell_speed_3_ve = BUFFER->tmp_cell_speed_3_ve;
	float *tmp_cell_speed_4_ve = BUFFER->tmp_cell_speed_4_ve;
	float *tmp_cell_speed_5_ve = BUFFER->tmp_cell_speed_5_ve;
	float *tmp_cell_speed_6_ve = BUFFER->tmp_cell_speed_6_ve;
	float *tmp_cell_speed_7_ve = BUFFER->tmp_cell_speed_7_ve;
	float *tmp_cell_speed_8_ve = BUFFER->tmp_cell_speed_8_ve;

	int *obstacles = BUFFER->obstacles;

	float *av_vels = BUFFER->av_vels;

	// Set up data region on device

	for (int jj = 0; jj < params->global_rows; jj++)
	{
		for (int ii = 0; ii < params->global_cols; ii++)
		{
			BUFFER->cell_speed_0_ve[ii + jj * params->global_cols] = w0;

			BUFFER->cell_speed_1_ve[ii + jj * params->global_cols] = w1;
			BUFFER->cell_speed_2_ve[ii + jj * params->global_cols] = w1;
			BUFFER->cell_speed_3_ve[ii + jj * params->global_cols] = w1;
			BUFFER->cell_speed_4_ve[ii + jj * params->global_cols] = w1;

			BUFFER->cell_speed_5_ve[ii + jj * params->global_cols] = w2;
			BUFFER->cell_speed_6_ve[ii + jj * params->global_cols] = w2;
			BUFFER->cell_speed_7_ve[ii + jj * params->global_cols] = w2;
			BUFFER->cell_speed_8_ve[ii + jj * params->global_cols] = w2;

			BUFFER->tmp_cell_speed_0_ve[ii + jj * params->global_cols] = 0;
			BUFFER->tmp_cell_speed_1_ve[ii + jj * params->global_cols] = 0;
			BUFFER->tmp_cell_speed_2_ve[ii + jj * params->global_cols] = 0;
			BUFFER->tmp_cell_speed_3_ve[ii + jj * params->global_cols] = 0;
			BUFFER->tmp_cell_speed_4_ve[ii + jj * params->global_cols] = 0;
			BUFFER->tmp_cell_speed_5_ve[ii + jj * params->global_cols] = 0;
			BUFFER->tmp_cell_speed_6_ve[ii + jj * params->global_cols] = 0;
			BUFFER->tmp_cell_speed_7_ve[ii + jj * params->global_cols] = 0;
			BUFFER->tmp_cell_speed_8_ve[ii + jj * params->global_cols] = 0;
		}
	}

	/* first set all cells in obstacle array to zero */
	for (int jj = 0; jj < params->global_rows; jj++)
	{
		for (int ii = 0; ii < params->global_cols; ii++)
		{
			BUFFER->obstacles[ii + jj * params->global_cols] = 0;
		}
	}

	/* open the obstacle data file */
	fp = fopen(obstaclefile, "r");

	if (fp == NULL)
	{
		sprintf(message, "could not open input obstacles file: %s", obstaclefile);
		die(message, __LINE__, __FILE__);
	}

	while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
	{
		/* some checks */
		if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

		if (xx < 0 || xx > params->global_cols - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

		if (yy < 0 || yy > params->global_rows - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

		if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

		/* assign to array */
		BUFFER->obstacles[xx + yy * params->global_cols] = blocked;
		++global_obstacle_cells;
	}
	fclose(fp);

	params->global_non_obstacle_cells = (params->global_rows * params->global_cols) - global_obstacle_cells;

	return;
}
void finalize(t_param* params, t_buffer_ptrs* BUFFER)
{
	int maxIterations = params->maxIters;
	int grid_extent = params->global_rows * params->global_cols;

	// Have to place all pointers into local variables
	// for OpenMP to accept them in mapping clauses
	float *cell_speed_0_ve = BUFFER->cell_speed_0_ve;
	float *cell_speed_1_ve = BUFFER->cell_speed_1_ve;
	float *cell_speed_2_ve = BUFFER->cell_speed_2_ve;
	float *cell_speed_3_ve = BUFFER->cell_speed_3_ve;
	float *cell_speed_4_ve = BUFFER->cell_speed_4_ve;
	float *cell_speed_5_ve = BUFFER->cell_speed_5_ve;
	float *cell_speed_6_ve = BUFFER->cell_speed_6_ve;
	float *cell_speed_7_ve = BUFFER->cell_speed_7_ve;
	float *cell_speed_8_ve = BUFFER->cell_speed_8_ve;

	float *tmp_cell_speed_0_ve = BUFFER->tmp_cell_speed_0_ve;
	float *tmp_cell_speed_1_ve = BUFFER->tmp_cell_speed_1_ve;
	float *tmp_cell_speed_2_ve = BUFFER->tmp_cell_speed_2_ve;
	float *tmp_cell_speed_3_ve = BUFFER->tmp_cell_speed_3_ve;
	float *tmp_cell_speed_4_ve = BUFFER->tmp_cell_speed_4_ve;
	float *tmp_cell_speed_5_ve = BUFFER->tmp_cell_speed_5_ve;
	float *tmp_cell_speed_6_ve = BUFFER->tmp_cell_speed_6_ve;
	float *tmp_cell_speed_7_ve = BUFFER->tmp_cell_speed_7_ve;
	float *tmp_cell_speed_8_ve = BUFFER->tmp_cell_speed_8_ve;

	int *obstacles = BUFFER->obstacles;

	float *av_vels = BUFFER->av_vels;

	// Set up data region on device
#pragma omp target exit data map(release: av_vels[0:maxIterations], obstacles[0:grid_extent], cell_speed_0_ve[0:grid_extent], cell_speed_1_ve[0:grid_extent], cell_speed_2_ve[0:grid_extent], cell_speed_3_ve[0:grid_extent], cell_speed_4_ve[0:grid_extent], cell_speed_5_ve[0:grid_extent], cell_speed_6_ve[0:grid_extent], cell_speed_7_ve[0:grid_extent], cell_speed_8_ve[0:grid_extent], tmp_cell_speed_0_ve[0:grid_extent], tmp_cell_speed_1_ve[0:grid_extent], tmp_cell_speed_2_ve[0:grid_extent], tmp_cell_speed_3_ve[0:grid_extent], tmp_cell_speed_4_ve[0:grid_extent], tmp_cell_speed_5_ve[0:grid_extent], tmp_cell_speed_6_ve[0:grid_extent], tmp_cell_speed_7_ve[0:grid_extent], tmp_cell_speed_8_ve[0:grid_extent])
	{}

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

	_mm_free(BUFFER->av_vels);
	BUFFER->av_vels = NULL;

	return;
}

float av_velocity(t_param* params, t_buffer_ptrs* BUFFER)
{
	int grid_extent = params->global_rows * params->global_cols;
	int global_rows = params->global_rows;
	int global_cols = params->global_cols;

	float tot_u = 0.f;

	for (int jj = 0; jj < global_rows; jj++)
	{
		for (int ii = 0; ii < global_cols; ii++)
		{
			if (BUFFER->obstacles[ii + jj * global_cols] == 0)
			{
				/* local density total */
				float local_density = 0.f;
				local_density = BUFFER->cell_speed_0_ve[ii + jj * global_cols] + BUFFER->cell_speed_1_ve[ii + jj * global_cols] +
					BUFFER->cell_speed_2_ve[ii + jj * global_cols] + BUFFER->cell_speed_3_ve[ii + jj * global_cols] +
					BUFFER->cell_speed_4_ve[ii + jj * global_cols];
				local_density += BUFFER->cell_speed_5_ve[ii + jj * global_cols] + BUFFER->cell_speed_6_ve[ii + jj * global_cols] +
					BUFFER->cell_speed_7_ve[ii + jj * global_cols] + BUFFER->cell_speed_8_ve[ii + jj * global_cols];

				/* x-component of velocity */
				float u_x = (BUFFER->cell_speed_1_ve[ii + jj * global_cols]
					+ BUFFER->cell_speed_5_ve[ii + jj * global_cols]
					+ BUFFER->cell_speed_8_ve[ii + jj * global_cols]
					- (BUFFER->cell_speed_3_ve[ii + jj * global_cols]
						+ BUFFER->cell_speed_6_ve[ii + jj * global_cols]
						+ BUFFER->cell_speed_7_ve[ii + jj * global_cols]))
					/ local_density;

				/* compute y velocity component */
				float u_y = (BUFFER->cell_speed_2_ve[ii + jj * global_cols]
					+ BUFFER->cell_speed_5_ve[ii + jj * global_cols]
					+ BUFFER->cell_speed_6_ve[ii + jj * global_cols]
					- (BUFFER->cell_speed_4_ve[ii + jj * global_cols]
						+ BUFFER->cell_speed_7_ve[ii + jj * global_cols]
						+ BUFFER->cell_speed_8_ve[ii + jj * global_cols]))
					/ local_density;

				/* accumulate the norm of x- and y- velocity components */
				tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
				//End: body of the average velocity function
			}
		}
	}

	return tot_u / params->global_non_obstacle_cells;
}

int write_values(t_param* params, t_buffer_ptrs* BUFFER)
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

	for (int32_t jj = 0; jj < params->global_rows; jj++)
		for (int32_t ii = 0; ii < params->global_cols; ii++)
		{
			/* an occupied cell */
			if (BUFFER->obstacles[ii + jj * params->global_cols])
			{
				u_x = u_y = u = 0.f;
				pressure = params->density * c_sq;
			}
			/* no obstacle */
			else
			{
				local_density = 0.f;
				local_density = BUFFER->cell_speed_0_ve[ii + jj * params->global_cols] + BUFFER->cell_speed_1_ve[ii + jj * params->global_cols] +
					BUFFER->cell_speed_2_ve[ii + jj * params->global_cols] + BUFFER->cell_speed_3_ve[ii + jj * params->global_cols] +
					BUFFER->cell_speed_4_ve[ii + jj * params->global_cols];
				local_density += BUFFER->cell_speed_5_ve[ii + jj * params->global_cols] + BUFFER->cell_speed_6_ve[ii + jj * params->global_cols] +
					BUFFER->cell_speed_7_ve[ii + jj * params->global_cols] + BUFFER->cell_speed_8_ve[ii + jj * params->global_cols];

				/* compute x velocity component */
				u_x = (BUFFER->cell_speed_1_ve[ii + jj * params->global_cols]
					+ BUFFER->cell_speed_5_ve[ii + jj * params->global_cols]
					+ BUFFER->cell_speed_8_ve[ii + jj * params->global_cols]
					- (BUFFER->cell_speed_3_ve[ii + jj * params->global_cols]
						+ BUFFER->cell_speed_6_ve[ii + jj * params->global_cols]
						+ BUFFER->cell_speed_7_ve[ii + jj * params->global_cols]))
					/ local_density;
				/* compute y velocity component */
				u_y = (BUFFER->cell_speed_2_ve[ii + jj * params->global_cols]
					+ BUFFER->cell_speed_5_ve[ii + jj * params->global_cols]
					+ BUFFER->cell_speed_6_ve[ii + jj * params->global_cols]
					- (BUFFER->cell_speed_4_ve[ii + jj * params->global_cols]
						+ BUFFER->cell_speed_7_ve[ii + jj * params->global_cols]
						+ BUFFER->cell_speed_8_ve[ii + jj * params->global_cols]))
					/ local_density;
				/* compute norm of velocity */
				u = sqrtf((u_x * u_x) + (u_y * u_y));
				/* compute pressure */
				pressure = local_density * c_sq;
			}

			/* write to file */
			fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, BUFFER->obstacles[ii * params->global_cols + jj]);
		}

	fclose(fp);

	fp = fopen(AVVELSFILE, "w");

	if (fp == NULL)
	{
		die("could not open file output file", __LINE__, __FILE__);
	}

	for (int ii = 0; ii < params->maxIters; ii++)
	{
		fprintf(fp, "%d:\t%.12E\n", ii, BUFFER->av_vels[ii]);
	}

	fclose(fp);

	return EXIT_SUCCESS;
}

float calc_reynolds(t_param* params, t_buffer_ptrs* BUFFER)
{
	const float viscosity = 1.f / 6.f * (2.f / params->omega - 1.f);
	return av_velocity(params, BUFFER) * params->reynolds_dim / viscosity;
}

void die(const char *message, const int line, const char *file) {
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

