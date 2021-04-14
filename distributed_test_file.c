#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>


//Example compilation
//mpicc -g distributed_test_file.c -lm -o distributed_test_file
//mpirun -np 2 -hostfile myhostfile.txt ./distributed_test_file 10 100 90 1 MSD_year_prediction_normalize_0_1_100k.txt 0

//Example execution
// Print nowhere
//mpirun -np 2 -hostfile myhostfile.txt ./distributed_yinyang 10 100 90 1 MSD_year_prediction_normalize_0_1_100k.txt 0

// Printing to console
//mpirun -np 2 -hostfile myhostfile.txt ./distributed_yinyang 10 100 90 1 MSD_year_prediction_normalize_0_1_100k.txt 1

// Printing to file
//mpirun -np 2 -hostfile myhostfile.txt ./distributed_yinyang 10 100 90 1 MSD_year_prediction_normalize_0_1_100k.txt 2

// Testing with valgrind
// mpirun -np 2 -hostfile myhostfile.txt valgrind --track-origins=yes ./distributed_yinyang 2 10 90 1 MSD_year_prediction_normalize_0_1_100k.txt 1

//function prototypes
int importDataset(char * fname, int N, int M, double ** dataset);
int importLocalDataset(char * fname, int start, int end, int M, double ** dataset);
double euclidianDistance( double * a, double * b, double dim );
void kmeans( double ** dataset, int K, int N, int M, int max_iter, int * clusters );

#define SEED 72

#define DONT_PRINT 0
#define PRINT_CONSOLE 1
#define PRINT_FILE 2

int main(int argc, char **argv) {

  // Initialize MPI
  int my_rank, nprocs;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Status status;
  MPI_Request request=MPI_REQUEST_NULL;

  // The number of clusters, the number of points, the dimensionality of the data,
  // and the number of cluster groups respectively
  int K, N, M, T, print_to_file;
  char fileName[500];
  double ** dataset;
  double ** local_data;
  double start, end;

  if( argc != 7 ){
    if( my_rank == 0 ) {
      printf("Expected arguments K, N, M, T, and filename\n");
    }
    exit(0);
  }

  // Parse command line arguments
  sscanf(argv[1],"%d",&K);
  sscanf(argv[2],"%d",&N);
  sscanf(argv[3],"%d",&M);
  sscanf(argv[4],"%d",&T);
  strcpy(fileName,argv[5]);
  sscanf(argv[6],"%d",&print_to_file);

  if( K<2 || N<1 || M <1) {
    if( my_rank == 0 ) {
      printf("K, N, or M is invalid\n");
    }
    exit(0);
  }

  if( T > K / 10 ) {
    T = ( K + (10-1) )/10;
    if( my_rank == 0 ) {
      printf("T should be no greater than K / 10; setting T to %d\n", T);
    }
  }

  if( K > N ) {
    N = K;
    if( my_rank == 0 ) {
      printf("N must be at least equal to K; setting N to K\n");
    }
  }

  if( print_to_file != DONT_PRINT && print_to_file != PRINT_CONSOLE && print_to_file != PRINT_FILE ) {
    print_to_file = DONT_PRINT;
  }

  if( my_rank == 0 ) {
    printf("K: %d, N: %d, M: %d, T: %d\n", K, N, M, T);
  }



  // Begin the distributed memory Yinyang algorithm
  int * row_starts;
  int * row_ends;
  int row_start;
  int row_end;

  // Each rank will be responsible for an equal number of points
  // The last rank will be responsible for any exess points
  // Have rank 0 calculate the ranges of points each rank is responsible for
  // calculating
  if( my_rank==0 ){

    row_starts = (int*)malloc(sizeof(int)*nprocs);
    row_ends = (int*)malloc(sizeof(int)*nprocs);

    // Divide the points as evenly as possible among the ranks
    int rows_per_rank = N / nprocs;

    for( int index=0; index<nprocs; index++ ) {
      if( index < nprocs-1){
        row_starts[ index ] = ( index * rows_per_rank );
        row_ends[ index ] = row_starts[ index ] + rows_per_rank;
      }
      // Assign any remaining points to the last rank. This shouldn't cause too
      // much of a load impbalance since at most the last rank will be computing
      // nprocs+1 extra points.
      else {
        row_starts[ index ] = ( index * rows_per_rank );
        row_ends[ index ] = N;
      }

    }
  }

  // Distribute the ranges of points to the ranks
  MPI_Scatter( row_starts, 1, MPI_INT, &row_start, 1, MPI_INT, 0, MPI_COMM_WORLD );
  MPI_Scatter( row_ends, 1, MPI_INT, &row_end, 1, MPI_INT, 0, MPI_COMM_WORLD );

  // Calculate how many points each rank is responisble for calculating
  int points_to_calc = row_end - row_start;
  printf( "I am rank %d, I calculate distance from row %d to row %d. In total I am calculating %d points\n", my_rank, row_start, row_end, points_to_calc );
  printf("\n");

  start = MPI_Wtime();

  // Allocate dataset and import file into it
  local_data=(double**)malloc(sizeof(double*)*points_to_calc);
  for( int index=0; index<N; index++ ) {
    local_data[index]=(double*)malloc(sizeof(double)*M);
  }
  int failure=importLocalDataset( fileName, row_start, row_end, M, local_data );
  if( failure==1 ) {
    printf("Problem reading file\n");
  }

  // Determine the longest time it took a rank to finish
  end = MPI_Wtime() - start;
  double global_end;
  MPI_Reduce( &end, &global_end, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
  if( my_rank == 0 ) {
    printf( "Data distribution by reading file took %f seconds\n", global_end );
  }


  MPI_Barrier( MPI_COMM_WORLD );
  for( int index=0; index<nprocs; index++ ) {
    if( my_rank==index ) {

      printf( "I am rank %d, this is my local dataset\n", my_rank );
      for( int point_index=0; point_index<points_to_calc; point_index++ ) {
        for( int dim_index=0; dim_index<M; dim_index++ ) {
          printf( "%f ", local_data[ point_index ][ dim_index ] );
        }
        printf("\n");
      }

    }
    MPI_Barrier( MPI_COMM_WORLD );
  }


  MPI_Finalize();
  return 0;
}



int importLocalDataset(char * fname, int start, int end, int M, double ** dataset)
{

    FILE *fp = fopen(fname, "r");

    if (!fp) {
        printf("Unable to open file\n");
        return(1);
    }

    char buf[4096];
    int rowCnt = 0;
    int colCnt = 0;
    while (fgets(buf, 4096, fp) && rowCnt<end) {
      if( rowCnt >= start ) {
        colCnt = 0;

        char *field = strtok(buf, ",");
        double tmp;
        sscanf(field,"%lf",&tmp);
        dataset[rowCnt-start][colCnt]=tmp;


        while (field) {
          colCnt++;
          field = strtok(NULL, ",");

          if (field!=NULL && colCnt<M)
          {
          double tmp;
          sscanf(field,"%lf",&tmp);
          dataset[rowCnt-start][colCnt]=tmp;
          }

        }
      }

      rowCnt++;
    }

    fclose(fp);

    return 0;
}



int importDataset(char * fname, int N, int M, double ** dataset)
{

    FILE *fp = fopen(fname, "r");

    if (!fp) {
        printf("Unable to open file\n");
        return(1);
    }

    char buf[4096];
    int rowCnt = 0;
    int colCnt = 0;
    while (fgets(buf, 4096, fp) && rowCnt<N) {
        colCnt = 0;

        char *field = strtok(buf, ",");
        double tmp;
        sscanf(field,"%lf",&tmp);
        dataset[rowCnt][colCnt]=tmp;


        while (field) {
          colCnt++;
          field = strtok(NULL, ",");

          if (field!=NULL && colCnt<M)
          {
          double tmp;
          sscanf(field,"%lf",&tmp);
          dataset[rowCnt][colCnt]=tmp;
          }

        }
        rowCnt++;
    }

    fclose(fp);

    return 0;
}



double euclidianDistance( double * a, double * b, double dim ) {
  double sum = 0;
  for( int index=0; index<dim; index++ ) {
    sum += pow( ( a[index] - b[index] ), 2 );
  }
  return sqrt( sum );
}



/*
  A naive, sequential implementation of the basic Kmeans algorithm

  dataset - the set of data to be clustered
  K - the number of clusters to sort the data into
  N - the number of points from the dataset to consider
  M - the number of dimensions of the dataset to consider
  max_iter - the maximum number of iterations to perform
  clusters - the array of size N which will store values from 0 to K

  returns - modfies clusters so that the data point at index i is assigned to
            cluster clusters[i]
*/
void kmeans( double ** dataset, int K, int N, int M, int max_iter, int * clusters ) {
  // Begin the sequential kmeans algorithm
  double ** cluster_centers;
  double ** cluster_sums;
  int * cluster_counts;

  // Initialize the clustering centers, using K random points
  // as the initial centers
  srand( SEED );
  int point_to_use;
  cluster_centers=(double**)malloc(sizeof(double*)*K);
  for( int point_index=0; point_index<K; point_index++ ) {
    point_to_use = rand()%N;
    cluster_centers[point_index]=(double*)malloc(sizeof(double)*M);
    for( int dim_index=0; dim_index<M; dim_index++ ) {
      cluster_centers[point_index][dim_index] = dataset[point_to_use][dim_index];
    }
  }

  // Initialize the cluster_sums matrix which stores the sum of all points in
  // each cluster
  cluster_sums=(double**)malloc(sizeof(double*)*K);
  for( int clust_index=0; clust_index<K; clust_index++ ){
    cluster_sums[ clust_index ]=(double*)malloc(sizeof(double)*M);
  }

  // Initialize the cluster_counts vector which stores how many points are in
  // each cluster
  cluster_counts=(int*)malloc(sizeof(int)*K);


  int iterations = 0;
  int moved_center = 1;

  while( moved_center && iterations < max_iter ) {

    iterations++;
    moved_center = 0;

    // Reset the data on the clusters
    for( int clust_index=0; clust_index<K; clust_index++ ) {
      cluster_counts[ clust_index ] = 0;
      for( int dim_index=0; dim_index<M; dim_index++ ) {
        cluster_sums[ clust_index ][ dim_index ] = 0;
      }
    }

    // For each data point in the dataset
    int nearest_center;
    double min_distance, distance;
    for( int row_index=0; row_index<N; row_index++ ) {
      // Find the closest clustering center to the point
      min_distance = -1;
      for( int cent_index=0; cent_index<K; cent_index++ ) {
        distance = euclidianDistance( dataset[ row_index ], cluster_centers[ cent_index ], M );
        if( distance < min_distance || min_distance == -1 ){
          nearest_center = cent_index;
          min_distance = distance;
        }
      }

      // Add the point to that cluster
      clusters[ row_index ] = nearest_center;
      cluster_counts[ nearest_center ] = cluster_counts[ nearest_center ] + 1;
      for( int dim_index=0; dim_index<M; dim_index++ ) {
        cluster_sums[ nearest_center ][ dim_index ] =
          cluster_sums[ nearest_center ][ dim_index ] + dataset[ row_index ][ dim_index ];
      }

    }

    // Calculate the mean of all points in each cluster
    double mean;

    // If the mean is the same as the coordinates of the cluster's center,
    // we're done. Otherwise, move the cluster's center to the mean and
    // perform another iteration
    for( int clust_index=0; clust_index<K; clust_index++ ) {

      for( int dim_index=0; dim_index<M; dim_index++ ) {
        mean = cluster_sums[ clust_index ][ dim_index ] / cluster_counts[ clust_index ];
        // If the mean is different from the current center, update the center
        // and mark that we moved it
        if( fabs( mean - cluster_centers[ clust_index ][ dim_index ] ) > 0.00001 ){
              moved_center = 1;
              cluster_centers[ clust_index ][ dim_index ] = mean;
            }
      }

    }

  }

  for( int index=0; index<K; index++ ) {
    free( cluster_centers[ index ] );
  }
  free( cluster_centers );

  for( int index=0; index<K; index++ ) {
    free( cluster_sums[ index ] );
  }
  free( cluster_sums );

  free( cluster_counts );
}
