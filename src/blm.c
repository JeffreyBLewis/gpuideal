#include "gpuideal.h"

void xpx_xpy(double *xpx, double *xpy, double *x, double *y, int *n) {
  int i;

  for (i=0;i<4;i++) xpx[i]=0.0;
  xpy[0]=xpy[1]=0.0;

  for (i=0;i<*n;i++) {
    xpx[2] += x[i];
    xpx[3] += x[i]*x[i];
    xpy[0] += y[i];
    xpy[1] += x[i]*y[i];
  }
  xpx[0] = *n;
  xpx[1] = xpx[2];
}

void  chol(double *chol, double *x) {
  double sqrtx0 = sqrt(x[0]);
  chol[0] = sqrtx0;
  chol[1] = 0.0;
  chol[2] = x[1]/sqrtx0;
  chol[3] = sqrt(x[0]*x[3]-x[1]*x[1])/sqrtx0;
}

void solve2d(double *b, double *iXpx, double *xpx, double *xpy) {
  const double denom = (xpx[0]*xpx[3] - xpx[1]*xpx[2]);
  b[0] = (xpy[0]*xpx[3]-xpy[1]*xpx[1])/denom;
  b[1] = (xpx[0]*xpy[1]-xpx[2]*xpy[0])/denom;
  iXpx[0] = xpx[3]/denom;
  iXpx[1] = -xpx[2]/denom;
  iXpx[2] = -xpx[1]/denom;
  iXpx[3] = xpx[0]/denom;
}


void blm(double *b, double *ixpx, double *y, double *x, double *iPrior, int *n) {
  double xpx[4], xpy[2];
  xpx_xpy(xpx,xpy,x,y,n);
  xpx[0] += iPrior[0];
  xpx[3] += iPrior[3];
  xpx[1]=xpx[2] += iPrior[2];
  solve2d(b,ixpx,xpx,xpy);
}


