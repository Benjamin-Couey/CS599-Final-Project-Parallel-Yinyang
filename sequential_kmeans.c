#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

//Example compilation
//gcc sequential_kmeans.c -lm -o sequential_kmeans

//Example execution
//./sequential_kmeans 10 100 90 MSD_year_prediction_normalize_0_1_100k.txt 0

//function prototypes
int importDataset(char * fname, int N, int M, double ** dataset);
double euclidianDistance( double * a, double * b, double dim );

#define SEED 72

int main(int argc, char **argv) {

  // The number of clusters, the number of points, and the dimensionality of the
  // data
  int K, N, M, print_to_file;
  char fileName[500];
  double ** dataset;
  clock_t start, end;

  // Parse command line arguments
  sscanf(argv[1],"%d",&K);
  sscanf(argv[2],"%d",&N);
  sscanf(argv[3],"%d",&M);
  strcpy(fileName,argv[4]);
  sscanf(argv[5],"%d",&print_to_file);

  if( K<2 || N<1 || M <1) {
    printf("K, N, or M is invalid\n");
    exit(0);
  }

  if( print_to_file != 0 && print_to_file != 1 ) {
    print_to_file = 1;
  }

  printf("K: %d, N: %d, M: %d\n", K, N, M );

  // Allocate dataset and import file into it
  dataset=(double**)malloc(sizeof(double*)*N);
  for( int index=0; index<N; index++ ) {
    dataset[index]=(double*)malloc(sizeof(double)*M);
  }
  int failure=importDataset( fileName, N, M, dataset );

  if( failure ){
    return 0;
  }



  // Begin the sequential kmeans algorithm
  start = clock();

  double ** cluster_centers;
  int * clusters;
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

  // Initialize the cluster vector which stores the index of the
  // clustering center each row in the matrix is associated with
  clusters=(int*)malloc(sizeof(int)*N);

  // Initialize the cluster_sums matrix which stores the sum of all points in
  // each cluster
  cluster_sums=(double**)malloc(sizeof(double*)*K);
  for( int clust_index=0; clust_index<K; clust_index++ ){
    cluster_sums[ clust_index ]=(double*)malloc(sizeof(double*)*M);
  }

  // Initialize the cluster_counts vector which stores how many points are in
  // each cluster
  cluster_counts=(int*)malloc(sizeof(int)*K);


  int iterations = 0;
  int moved_center = 1;

  while( moved_center ) {

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

  end = clock();
  printf( "Sequential Kmeans took %f seconds\n", ((double) (end - start)) / CLOCKS_PER_SEC );



  //Report the clustering
  if( print_to_file == 0 ) {
    for( int clust_index=0; clust_index<N; clust_index++ ) {
      printf("%d ", clusters[ clust_index ] );
    }
    printf("\n");
  } else {
    //Report the position on the centroids and the clutering
    FILE *file;

    file = fopen("cluster_centers.txt", "w");
    if (!file) {
        printf("Unable to open file\n");
        return(1);
    }
    for( int clust_index=0; clust_index<K; clust_index++ ) {
      for( int dim_index=0; dim_index<M; dim_index++ ) {
        if( dim_index<(M-1) ) {
          fprintf(file, "%f, ", cluster_centers[ clust_index ][ dim_index ] );
        } else {
          fprintf(file, "%f\n", cluster_centers[ clust_index ][ dim_index ] );
        }
      }
    }
    fclose(file);

    file = fopen("clustering.txt", "w");
    if (!file) {
        printf("Unable to open file\n");
        return(1);
    }
    for( int clust_index=0; clust_index<N; clust_index++ ) {
      if( clust_index<(N-1) ) {
        fprintf(file, "%d, ", clusters[ clust_index ] );
      } else {
        fprintf(file, "%d", clusters[ clust_index ] );
      }
    }
    fclose(file);
  }



  for( int index=0; index<N; index++ ) {
    free( dataset[ index ] );
  }
  free( dataset );

  for( int index=0; index<K; index++ ) {
    free( cluster_centers[ index ] );
  }
  free( cluster_centers );

  for( int index=0; index<K; index++ ) {
    free( cluster_sums[ index ] );
  }
  free( cluster_sums );

  free( clusters );
  free( cluster_counts );

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
