#include <stdlib.h>
int check_values(float *d, int n) {
  float *b = (float *) malloc(n*sizeof(float));
  CHECKCALL(cudaMemcpy(b, d, sizeof(float) * n, cudaMemcpyDeviceToHost));
  for (int i=0; i<n; i++) {
    if (isnan(b[i])) {
      free(b);
      return(i);
    }
  }
  free(b);
  return(-1);
}
