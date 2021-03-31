#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>


//Example compilation
//gcc -g sequential_yinyang.c -lm -o sequential_yinyang

//Example execution
//./sequential_yinyang 10 100 90 1 MSD_year_prediction_normalize_0_1_100k.txt

// valgrind --track-origins=yes ./sequential_yinyang 2 10 90 1 MSD_year_prediction_normalize_0_1_100k.txt

//function prototypes
int importDataset(char * fname, int N, int M, double ** dataset);
double euclidianDistance( double * a, double * b, double dim );
void kmeans( double ** dataset, int K, int N, int M, int max_iter, int * clusters );

#define SEED 72

int main(int argc, char **argv) {

  // The number of clusters, the number of points, the dimensionality of the data,
  // and the number of cluster groups respectively
  int K, N, M, T, print_to_file;
  char fileName[500];
  double ** dataset;
  clock_t start, end;

  if( argc != 7 ){
    printf("Expected arguments K, N, M, T, filename, and whether the results should be printed to file\n");
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
    printf("K, N, or M is invalid\n");
    exit(0);
  }

  if( T > K / 10 ) {
    T = ( K + (10-1) )/10;
    printf("T should be no greater than K / 10; setting T to %d\n", T);
  }

  if( K > N ) {
    N = K;
    printf("N must be at least equal to K; setting N to K\n");
  }

  if( print_to_file != 0 && print_to_file != 1 ) {
    print_to_file = 1;
  }

  printf("K: %d, N: %d, M: %d, T: %d\n", K, N, M, T);

  // Allocate dataset and import file into it
  dataset=(double**)malloc(sizeof(double*)*N);
  for( int index=0; index<N; index++ ) {
    dataset[index]=(double*)malloc(sizeof(double)*M);
  }
  int failure=importDataset( fileName, N, M, dataset );

  if( failure ){
    return 0;
  }



  // Begin the sequential Yinyang algorithm
  start = clock();

  int * clusters;

  double ** cluster_centers;
  int * cluster_groups;

  double * upper_bounds;
  double ** lower_bounds;

  double ** cluster_sums;
  int * cluster_counts;

  double * cluster_drift;
  double * group_drift;

  double * temp_point;

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

  // Using 5 iterations of the traditional Kmeans algorithm, sort those centers
  // into T groups
  cluster_groups=(int*)malloc(sizeof(int)*K);
  kmeans( cluster_centers, T, K, M, 5, cluster_groups );

  // Initialize the cluster vector which stores the index of the
  // clustering center each row in the matrix is associated with
  clusters=(int*)malloc(sizeof(int)*N);

  // Initialize the cluster_sums matrix which stores the sum of all points in
  // each cluster
  cluster_sums=(double**)malloc(sizeof(double*)*K);
  for( int clust_index=0; clust_index<K; clust_index++ ){
    cluster_sums[ clust_index ]=(double*)calloc( M, sizeof(double) );
  }

  // Initialize the cluster_counts vector which stores how many points are in
  // each cluster
  cluster_counts=(int*)calloc(K, sizeof(int));

  // Initialize the cluster_drift vector which stores how far each cluster moved
  // during the current iteration
  cluster_drift=(double*)malloc(sizeof(double)*K);

  // Initialize the vector which stores the highest drift for each group of clusters
  group_drift=(double*)malloc(sizeof(double)*T);

  // Initialize a vector to store points temporarily for various purposes
  temp_point=(double*)malloc(sizeof(double)*M);

  // Initialize the upper bounds vector which, for each point, stores the
  // distance to the clustering center that point is currently assigned to
  upper_bounds=(double*)malloc(sizeof(double)*N);

  // Initialize the group lower bounds matrix which, for each points, stores the
  // minimum distance to clustering center the point isn't assigned to for each
  // group
  lower_bounds=(double**)malloc(sizeof(double*)*N);
  for( int point_index=0; point_index<N; point_index++ ) {
    lower_bounds[ point_index ]=(double*)malloc(sizeof(double)*T);
    for( int group_index=0; group_index<T; group_index++ ) {
      lower_bounds[ point_index ][ group_index ] = -1;
    }
  }
  // Initial values for both of these bounds matrices will be assigned during the
  // the Kmeans algorithm below to save time

  // Assign points to cluster centers using the traditional Kmeans algorithm for
  // the first iteration
  // For each data point in the dataset
  int nearest_center;
  double min_distance, distance, lower_bound;
  for( int point_index=0; point_index<N; point_index++ ) {
    // Find the closest clustering center to the point
    min_distance = -1;
    for( int clust_index=0; clust_index<K; clust_index++ ) {
      distance = euclidianDistance( dataset[ point_index ], cluster_centers[ clust_index ], M );
      if( distance < min_distance || min_distance == -1 ){
        nearest_center = clust_index;
        min_distance = distance;
      }
      // If this distance is lower than the current lower bound for this cluster's
      // group, update this cluster's group's lower bound
      lower_bound = lower_bounds[ point_index ][ cluster_groups[ clust_index ] ];
      if( lower_bound > distance || lower_bound == -1 ) {
        lower_bounds[ point_index ][ cluster_groups[ clust_index ] ] = distance;
      }
    }

    // Add the point to that cluster
    clusters[ point_index ] = nearest_center;
    // Track how many points are assigned to each cluster
    cluster_counts[ nearest_center ] = cluster_counts[ nearest_center ] + 1;
    // Track the sum of all points assigned to each cluster
    for( int dim_index=0; dim_index<M; dim_index++ ) {
      cluster_sums[ nearest_center ][ dim_index ] =
        cluster_sums[ nearest_center ][ dim_index ] + dataset[ point_index ][ dim_index ];
    }
    // Update that point's upper bound
    // Technically, this is not part of the Kmeans algorithm but I'm doing this
    // here since we just calculated the distance
    upper_bounds[ point_index ] = min_distance;

  }



  // Start the Yinyang algorithm proper
  int iterations = 1;
  int moved_center = 1;

  double global_lower_bound, mean;
  while( moved_center ) {

    iterations++;
    moved_center = 0;

    // Calculate the mean of all points in each cluster
    // If the mean is the same as the coordinates of the cluster's center,
    // we're done. Otherwise, move the cluster's center to the mean and
    // perform another iteration
    for( int clust_index=0; clust_index<K; clust_index++ ) {

      for( int dim_index=0; dim_index<M; dim_index++ ) {
        // First, store the current (and potentially soon to be updated) value of
        // the cluster's center for later comparison
        temp_point[ dim_index ] = cluster_centers[ clust_index ][ dim_index ];

        // Calculate the mean
        mean = cluster_sums[ clust_index ][ dim_index ] / cluster_counts[ clust_index ];

        // If the mean is sufficiently different from the current center,
        // update the center, and mark that we moved a center
        if( fabs( mean - cluster_centers[ clust_index ][ dim_index ] ) > 0.00001 ) {
              moved_center = 1;
              cluster_centers[ clust_index ][ dim_index ] = mean;
            }
      }

      // Compare the new cluster center to the old cluster center to determine
      // the drift
      cluster_drift[ clust_index ] = euclidianDistance( temp_point, cluster_centers[ clust_index ], M );

    }

    for( int group_index=0; group_index<T; group_index++ ) {
      group_drift[ group_index ] = -1;
    }

    // Determine the highest drift for each cluster group
    for( int clust_index=0; clust_index<K; clust_index++ ) {
      if( cluster_drift[ clust_index ] > group_drift[ cluster_groups[ clust_index ] ] ) {
        group_drift[ cluster_groups[ clust_index ] ] = cluster_drift[ clust_index ];
      }
    }

    // Reset the data on the clusters
    for( int clust_index=0; clust_index<K; clust_index++ ) {
      cluster_counts[ clust_index ] = 0;
      for( int dim_index=0; dim_index<M; dim_index++ ) {
        cluster_sums[ clust_index ][ dim_index ] = 0;
      }
    }

    // Update the upper and lower bounds of all points
    global_lower_bound = -1;
    distance = -1;
    for( int point_index=0; point_index<N; point_index++ ) {
      // Add the drift of the center the point it currently assigned to to the
      // upper bound
      upper_bounds[ point_index ] = upper_bounds[ point_index ] + cluster_drift[ clusters[ point_index ] ];

      // Subtract the maximum drift of each group from that group's lower bound
      for( int group_index=0; group_index<T; group_index++ ) {
        lower_bounds[ point_index ][ group_index ] = lower_bounds[ point_index ][ group_index ] - group_drift[ group_index ];

        // Determine a global lower bound for this point
        if( global_lower_bound > lower_bounds[ point_index ][ group_index ] || global_lower_bound == -1 ) {
              global_lower_bound = lower_bounds[ point_index ][ group_index ];
        }
      }

      // Run all the points in the database through some filters to figure out
      // which ones need to be updated

      // Outer test
      // If the all of the group lower bounds of a point are greater than that
      // point's upper bound, then we don't need to move the point
      // Check this condition using the global_lower_bound computed above
      // Check the outer test again by tightening the upper bound
      if( global_lower_bound >= upper_bounds[ point_index ] || global_lower_bound >= ( upper_bounds[ point_index ] - cluster_drift[ clusters[ point_index ] ] ) ) {
        // Update the count and mean for the cluster this point is already
        // assigned to
        cluster_counts[ clusters[ point_index ] ] = cluster_counts[ clusters[ point_index ] ] + 1;
        for( int dim_index=0; dim_index<M; dim_index++ ) {
          cluster_sums[ clusters[ point_index ] ][ dim_index ] = cluster_sums[ clusters[ point_index ] ][ dim_index ] + dataset[ point_index ][ dim_index ];
        }

      }
      // Failed the outer test, check all the group lower bounds
      else {
        min_distance = -1;
        // Iterate through the clusters
        for( int clust_index=0; clust_index<K; clust_index++ ) {
          // If a cluster belongs to a group for which the group lower bound is
          // less than the upper bound, we have to perform the distance calculation
          if( lower_bounds[ point_index ][ cluster_groups[ clust_index ] ] < upper_bounds[ point_index ] ) {
            distance = euclidianDistance( dataset[ point_index ], cluster_centers[ clust_index ], M );

            if( distance < min_distance || min_distance == -1 ) {
              min_distance = distance;
              nearest_center = clust_index;
            }

          }

        }

        // Assign the point to the best cluster found above
        cluster_counts[ nearest_center ] = cluster_counts[ nearest_center ] + 1;
        for( int dim_index=0; dim_index<M; dim_index++ ) {
          cluster_sums[ nearest_center ][ dim_index ] = cluster_sums[ nearest_center ][ dim_index ] + dataset[ point_index ][ dim_index ];
        }

      }

    }
      // Ommiting the local test
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
  free( clusters );

  free( cluster_groups );

  free( upper_bounds );
  for( int index=0; index<N; index++ ) {
    free( lower_bounds[ index ] );
  }
  free( lower_bounds );

  free( cluster_counts );
  for( int index=0; index<K; index++ ) {
    free( cluster_sums[ index ] );
  }
  free( cluster_sums );

  free( cluster_drift );
  free( group_drift );

  free( temp_point );

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
