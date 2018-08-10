// from: ACM Transactions on Mathematical Software, Vol. 18, No. 4, December 1992

#include "R.h"
#include "Rmath.h"

#define s  0.449871
#define t -0.386595
#define a  0.19600
#define b  0.25472

#define abs(x) ((x)<0 ? -(x) : (x))

void rand_norm(double *z, int *n) {
  double v, u, Q, x, y;
  int i;

  //  GetRNGstate();

  for (i=0; i<*n; i++) { 
    while (1) {
      v = runif(0,1);
      u = runif(0,1);
      v = 1.7156*(v-0.5);
      x = u - s;
      y = abs(v) - t;
      Q = x*x + y*(a*y - b*x);
      if (Q  < 0.27597) break;
      if ((Q < 0.27846) & (v*v <= -4.0*u*u*log(u))) break;  
    }
    z[i] = v/u;
  }

  //PutRNGstate();
}
