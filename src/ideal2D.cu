#include <R.h>
#include <sys/time.h>

#include "Random123-1.09/include/Random123/philox.h"
#include "boxmuller.hpp"
#include "inv_mills.h"

#include "check_value.h"


#ifndef COMBINED_1_2__  // Only need this when compiling 2-D alone

#define abs(x) ((x)<0.0f ? -(x) : (x))

//Globals for util_cuda.h
#include "util_cuda.h"	// for cuda_init, CHECKCALL
#include "lookup.h"

int debug = 0;
const char *progname = "ideal2d.cu";

#endif 

/*****************************************************************
* Common functions for both the EM and Gibbs Sampling estimator  *
******************************************************************/

//
// Calculate cummulants for estimating bill parameters
//
__device__ static inline void xpx_xpy_2D(float *xpx, float *xpy, float *x1,
  float *x2, float *y, int *xidx, int n) {
  xpx[1]=xpx[2]=xpx[4]=xpx[5]=xpx[8]=0.0f;
  xpy[0]=xpy[1]=xpy[2]=0.0f;
  for (int i=0;i<n;i++) {
       xpx[1] += x1[xidx[i]];
       xpx[2] += x2[xidx[i]];
       xpx[4] += x1[xidx[i]]*x1[xidx[i]];
       xpx[5] += x2[xidx[i]]*x1[xidx[i]];
       xpx[8] += x2[xidx[i]]*x2[xidx[i]];	      
    
       xpy[0] += y[i];
       xpy[1] += x1[xidx[i]]*y[i];
       xpy[2] += x2[xidx[i]]*y[i];
  }
  xpx[0] = n;
  xpx[3] = xpx[1];
  xpx[6] = xpx[2];
  xpx[7] = xpx[5];
};


//
// Find the Cholesky decomposition of a 2x2 matrix
//
__device__ static inline void chol2d(float *chol, float *x) {
  const float sqrtx0 = sqrtf(x[0]);
  chol[0] = sqrtx0;
  chol[1] = 0.0f;
  chol[2] = x[1]/sqrtx0;
  chol[3] = sqrtf(x[0]*x[3]-x[1]*x[1])/sqrtx0;
};

//
// Find the Cholesky decomposition of a 3x3 matrix
//
__device__ static inline void chol3d(float *chol, float *x) {
  int i, j, k;
  for (i=0; i<3; i++)
      for (j=0; j<=i; j++) chol[3*i+j] = x[3*i+j];
  for (i=0; i<3; i++) { 
      chol[i*4] = sqrt(chol[i*4]);
      for (j=i+1; j<3; j++) 
      	  chol[j*3+i] = chol[j*3+i]/chol[i*4];
      for (k=i+1; k<3; k++) {
          for (j=k; j<3; j++) {
              chol[j*3+k] = chol[j*3+k] - chol[j*3+i]*chol[k*3+i];
          }
      }
  }
};

//
// Find inverse of a 3x3 matrix
//
__device__ static inline void inverse_3x3(float *invx, float *x) {
  float det = x[0]*x[4]*x[8] + 
              x[3]*x[7]*x[2] +
              x[6]*x[1]*x[5] -
              x[2]*x[4]*x[6] -
              x[5]*x[7]*x[0] -
              x[8]*x[1]*x[3];                      
  invx[0] = (x[4]*x[8] - x[5]*x[7])/det;
  invx[4] = (x[0]*x[8] - x[2]*x[6])/det;
  invx[8] = (x[0]*x[4] - x[1]*x[3])/det;
  invx[1] = (x[2]*x[7] - x[1]*x[8])/det; 
  invx[3] = invx[1];
  invx[2] = (x[1]*x[5] - x[2]*x[4])/det;
  invx[6] = invx[2];
  invx[5] = (x[2]*x[3] - x[0]*x[5])/det;
  invx[7] = invx[5];
}

//
// Solve system of linear equations with three unknowns (for estimating bill parameters)
//
__device__ static inline void solve3d(float *bbv, float *iXpx, float *xpx, float *xpy) {
  int i,j;
  inverse_3x3(iXpx, xpx);
  for (i=0;i<3;i++) {
      bbv[i]=0.0f;
      for (j=0;j<3;j++) {
      	 bbv[i]+=iXpx[3*i+j]*xpy[j];
      }
  }
};


#ifndef COMBINED_1_2__   // Already defined in 1-D version

//
// Solve system of linear equations with two unknowns (for estimating legislator parameters)
//

__device__ static inline void solve2d(float *bbv, float *iXpx, float *xpx, float *xpy) {
  const float denom = (xpx[0]*xpx[3] - xpx[1]*xpx[2]);
  bbv[0] = (xpy[0]*xpx[3]-xpy[1]*xpx[1])/denom;
  bbv[1] = (xpx[0]*xpy[1]-xpx[2]*xpy[0])/denom;
  iXpx[0] = xpx[3]/denom;
  iXpx[1] = -xpx[2]/denom;
  iXpx[2] = -xpx[1]/denom;
  iXpx[3] = xpx[0]/denom;
};

#endif;

//
// Bayesian linear regression with three parameters (for estimating bill parameters)
//
__device__ static inline void blm_2D(float *bv, float *ixpx, float *y, float *x1, float *x2,
	   	  	             float *iPrior, int *xidx, int n) {
  float xpx[9], xpy[3];
  int i;
  xpx_xpy_2D(xpx, xpy, x1, x2, y, xidx, n);
  for (i=0; i<9; i++) {
      xpx[i] += iPrior[i];
  }
  solve3d(bv, ixpx, xpx, xpy);
};


/************************************************************
* Functions needed only by the Gibbs sampler                *
*************************************************************/

//
// Draw truncated norm a la Christian P. Robert ``Simulation of Truncated normal variables''
// LSTA, Universite Pierre et Marie Carie, Paris (2009)
//
__device__ static inline float d_rtnorm_2D(float lb, 
	   philox4x32_key_t k, philox4x32_ctr_t c) {
  float z, zadiff2, u, astar, rejthresh; 
  r123::Philox4x32 rand;
  float2 f2;

  if (lb >= 0.0f) {
      astar = (lb + sqrtf( lb*lb + 4.0f )) / 2.0f;
      while (1) {
        typename r123::Philox4x32::ctr_type uu = rand(c, k);
        z = -__logf(  r123::u01<float>(uu[0]) )/astar + lb;
        u = r123::u01<float>(uu[1]);
        zadiff2 = (z - astar)*(z - astar);
        rejthresh = __expf( - zadiff2/2.0f );
        if (u <= rejthresh) {
          return(z);
	}
        c.incr();
      }
  }
  else {
    while (1) {
      c.incr();
      typename r123::Philox4x32::ctr_type uu = rand(c,k);
      f2  = r123::boxmuller(uu[0],uu[1]);
      if (f2.x > lb) {
        return(f2.x);  
      }  
      if (f2.y > lb) {
	return(f2.y);
      }
    }
  }
};


//
// Draw truncated norm using Metropolis sampling analogous to
// Christian P. Robert ``Simulation of Truncated normal variables''
// LSTA, Universite Pierre et Marie Carie, Paris (2009) rejection 
// sampler.  Might be faster for GPU because looping is avoided.
//
__device__ static inline float d_rtnorm_met_2D(float lb, float ystar0,
	   philox4x32_key_t k, philox4x32_ctr_t c) {
  float jlo, u, astar, r; 
  r123::Philox4x32 rand;
  float2 f2;
  //c.incr();
  typename r123::Philox4x32::ctr_type uu = rand(c, k);
  
  // low-prob truncation (lb > -0.4)
  if (lb > -0.4f) {
    astar = (lb + sqrtf( lb*lb + 4.0f )) / 2.0f;
    jlo = -__logf(  r123::u01<float>(uu[0]) )/astar + lb;
    r = __expf(-0.5f * (jlo*jlo - ystar0*ystar0) - astar*(ystar0-jlo));
    u = r123::u01<float>(uu[1]);
    return( u <= r ? jlo : ystar0 );   
  }
  // high-prob truncation (lb < -0.4)
  else {
    f2  = r123::boxmuller(uu[0],uu[1]); // since we get 2 norm draws we use both of them...
    if (f2.x>lb) {
      return( f2.x );
    }
    if (f2.y>lb) {
      return( f2.y );
    }
    return( ystar0 );
  }
};


//
// Device code to update y star
//
__global__ void gibbs_getYstar_2D(int *rw_rcidx,int *rw_memidx,float *a, float *b1,
	   	         float *b2, float *x1, float *x2,
                         int *rw_y, float *cw_ystar, float *rw_ystar, int *rw2cw, int seed,
			 int npt, int nn, int cntr) {

  int id = blockIdx.x * blockDim.x + threadIdx.x;
  philox4x32_key_t k = {{(unsigned int) id, (unsigned int) seed*3}};
  philox4x32_ctr_t c = {{(unsigned int) cntr}};
  
  for (int i=0; i<npt; i++) {
      int id = i*(blockDim.x*gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
      if (id < nn) {
       	 float xb = a[rw_rcidx[id]] + b1[rw_rcidx[id]]*x1[rw_memidx[id]] + b2[rw_rcidx[id]]*x2[rw_memidx[id]];
      	 float txb = - (float) rw_y[id] * xb;
         float ee = (float) d_rtnorm_2D(txb,k,c);
      	 cw_ystar[rw2cw[id]] = rw_ystar[id] = xb + (float) rw_y[id] * ee;
	 c.incr();
      }
  }
};


//
// Update a and b 
//
__device__ static inline void gibbs_getAB_2D(float *a, float *b1, float *b2, float *ystar,
	   	  	      		  float *x1, float *x2, float *iPrior, int *xidx,
					  int n, float2 veps1, float2 veps2) {
  float ixpx[9], cholixpx[9], beps[3];
  float bvec[3];
  
  blm_2D(bvec, ixpx, ystar, x1, x2, iPrior, xidx, n);
  chol3d(cholixpx,ixpx);
  beps[0] = veps1.x*cholixpx[0];
  beps[1] = veps1.x*cholixpx[0] + veps1.y*cholixpx[1];
  beps[2] = veps1.x*cholixpx[0] + veps1.y*cholixpx[1] + veps2.x*cholixpx[2];
  *a = bvec[0] + beps[0];
  *b1 = bvec[1] + beps[1];
  *b2 = bvec[2] + beps[2];
};

__global__ void gibbs_getABs_2D(float *a, float *b1, float *b2, float *cw_ystar,
	   	             float *x1, float *x2, float *iabprior, int *cw_memidx, 
		             int *cw_rclen, int *cw_rcstart, int nn_rc, int nn_mem,
			     int nn, int seed, int cntr) {
    const int np = nn_rc/(blockDim.x*gridDim.x) + 1;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float2 veps1, veps2;
    r123::Philox4x32 rand;
    philox4x32_key_t k = {{(unsigned int) id, (unsigned int) seed*2}};
    philox4x32_ctr_t c = {{(unsigned int) cntr}};

    for (int i=0;i<np;i++) {
    	int id = i*(blockDim.x*gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
        if (id < nn_rc) { 		  
	   typename r123::Philox4x32::ctr_type uu = rand(c,k);
	   veps1  = r123::boxmuller(uu[0],uu[1]);
	   uu = rand(c,k);
	   veps2 = r123::boxmuller(uu[0],uu[1]);   
	   int j = cw_rcstart[id];
      	   gibbs_getAB_2D(&a[id], &b1[id], &b2[id], &cw_ystar[j], x1, x2, iabprior,
	   	       &cw_memidx[j], cw_rclen[id], veps1, veps2 ); 
	   c.incr();
	}
    }
}

//
// Update X
//
__device__ static void gibbs_getX_2D(float *x1, float *x2, float *ystar, float *a, float *b1,
	   	       	          float *b2, float *iPrior, int *abidx, int n, float2 e) {
  float meanx[2], iBpB[4], BpB[4], BpY[2], choliBpB[4], xeps[2];
  int i;
  
  BpB[0] = 0.0f;
  BpB[1] = 0.0f;
  BpB[2] = 0.0f;
  BpB[3] = 0.0f;
  BpY[0] = 0.0f;
  BpY[1] = 0.0f;
  
  for (i=0; i<n; i++) {
     BpB[0] += b1[abidx[i]]*b1[abidx[i]];
     BpB[1] += b1[abidx[i]]*b2[abidx[i]];
     BpB[3] += b2[abidx[i]]*b2[abidx[i]];
     BpY[0] += (ystar[i]-a[abidx[i]])*b1[abidx[i]];
     BpY[1] += (ystar[i]-a[abidx[i]])*b2[abidx[i]];   
  }
  BpB[2] = BpB[1];
  for (i=0; i<4; i++) BpB[i] += iPrior[i];
  solve2d(meanx, iBpB, BpB, BpY);
  chol2d(choliBpB,iBpB);
  xeps[0] = e.x*choliBpB[0];
  xeps[1] = e.x*choliBpB[2] + e.y*choliBpB[3];
  *x1 = meanx[0] + xeps[0];
  *x2 = meanx[1] + xeps[1];
};


__global__ void gibbs_getXs_2D(float *x1, float *x2, float *rw_ystar, float *a,
	   		       float *b1, float *b2, float *ixprior, int *rw_rcidx, 
		               int *rw_memlen, int *rw_memstart, int nn_mem,
			       int nn_rc, int seed, int cntr) {
    const int np = nn_mem/(blockDim.x*gridDim.x) + 1;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float2 f2;
    r123::Philox4x32 rand;
    philox4x32_key_t k = {{(unsigned int) id, (unsigned int) seed*3}};
    philox4x32_ctr_t c = {{(unsigned int) cntr}};

    for (int i=0;i<np;i++) {
    	int id = i*(blockDim.x*gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
	if (id < nn_mem) {
	    int j = rw_memstart[id];
	    typename r123::Philox4x32::ctr_type uu = rand(c,k);
	    f2  = r123::boxmuller(uu[0],uu[1]);
	    gibbs_getX_2D(&x1[id], &x2[id], &rw_ystar[j], a, b1, b2, ixprior, &rw_rcidx[j], rw_memlen[id], f2);
	}
	c.incr();
    }
}

//
// Main call
//
extern "C" void gibbs_ideal_2D(float *x1, float *x2, float *a, float *b1, float *b2, float *iabprior,
       	   float *ixprior, 
	   int *rw_y, float *rw_ystar, int *rw_rcidx, int *rw_memidx, int *rw_memstart, int *rw_memlen, 
	   int *cw_y, float * cw_ystar, int *cw_memidx, int *cw_rcstart, int *cw_rclen, int *rw2cw, 
	   int *burnin, int *samples, int *n, int *n_rc, int *n_mem, int *thin, int *blocks, int *threads) {

  int i;
  int ssamples = *samples;
  const int nn = *n;
  const int bburnin = *burnin;
  const int nn_rc = *n_rc;
  const int nn_mem = *n_mem;
  const int tthin = *thin;
  
  ssamples += bburnin; // ssamples is total draws including burnin draws.
  
  CUDAInfo *infop = cuda_init(getenv("R123_CUDA_DEVICE")); 
  const int nv_tpb = (*threads) > 0 ? *threads : infop->threads_per_block;
  const int nv_bpg_ystar = (*blocks) > 0 ? *blocks : infop->blocks_per_grid; 
  const int nv_bpg_ab = nv_bpg_ystar > nn_rc/nv_tpb ? nn_rc/nv_tpb +1 : nv_bpg_ystar ;
  const int nv_bpg_x  = nv_bpg_ystar > nn_mem/nv_tpb ? nn_mem/nv_tpb +1 : nv_bpg_ystar ;
  const int nthreads = nv_bpg_ystar*nv_tpb;

  Rprintf("\nGPU Info...\n");
  Rprintf("Threads per block:        %i\n", nv_tpb);
  Rprintf("Max. Blocks per grid:     %i\n", nv_bpg_ystar);  
  Rprintf("RC blocks:                %i\n", nv_bpg_ab);
  Rprintf("Member blocks:            %i\n\n", nv_bpg_x);

  Rprintf("Drawing %i samples (%i per dot)...\n\n", *samples, *samples/50);
  Rprintf("|-------------------------------------------------|\n|");

  // Allocate device memory
  float* d_ixprior;
  CHECKCALL(cudaMalloc(&d_ixprior, sizeof(float) * 1));
  CHECKCALL(cudaMemcpy(d_ixprior, ixprior, sizeof(float)*1, cudaMemcpyHostToDevice));

  float* d_iabprior;
  CHECKCALL(cudaMalloc(&d_iabprior, sizeof(float) * 4));
  CHECKCALL(cudaMemcpy(d_iabprior, iabprior, sizeof(float)*4, cudaMemcpyHostToDevice));

  int* d_cw_rclen;
  CHECKCALL(cudaMalloc(&d_cw_rclen, sizeof(int) * nn_rc));
  CHECKCALL(cudaMemcpy(d_cw_rclen, cw_rclen, sizeof(int)*nn_rc, cudaMemcpyHostToDevice));

  int* d_cw_rcstart;
  CHECKCALL(cudaMalloc(&d_cw_rcstart, sizeof(int) * nn_rc));
  CHECKCALL(cudaMemcpy(d_cw_rcstart, cw_rcstart, sizeof(int)*nn_rc, cudaMemcpyHostToDevice));

  int* d_cw_memidx;
  CHECKCALL(cudaMalloc(&d_cw_memidx, sizeof(int) * nn));
  CHECKCALL(cudaMemcpy(d_cw_memidx, cw_memidx, sizeof(int)* (nn), cudaMemcpyHostToDevice));

  int* d_rw_memidx;
  CHECKCALL(cudaMalloc(&d_rw_memidx, sizeof(int) * nn));
  CHECKCALL(cudaMemcpy(d_rw_memidx, rw_memidx, sizeof(int)* (nn), cudaMemcpyHostToDevice));

  int* d_rw_memstart;
  CHECKCALL(cudaMalloc(&d_rw_memstart, sizeof(int) * nn_mem));
  CHECKCALL(cudaMemcpy(d_rw_memstart, rw_memstart, sizeof(int)* (nn_mem), cudaMemcpyHostToDevice));

  int* d_rw_memlen;
  CHECKCALL(cudaMalloc(&d_rw_memlen, sizeof(int) * nn_mem));
  CHECKCALL(cudaMemcpy(d_rw_memlen, rw_memlen, sizeof(int)* (nn_mem), cudaMemcpyHostToDevice));

  int* d_rw_rcidx;
  CHECKCALL(cudaMalloc(&d_rw_rcidx, sizeof(int) * nn));
  CHECKCALL(cudaMemcpy(d_rw_rcidx, rw_rcidx, sizeof(int)* (nn), cudaMemcpyHostToDevice));

  float* d_a;
  CHECKCALL(cudaMalloc(&d_a, sizeof(float) * nn_rc));
  CHECKCALL(cudaMemcpy(d_a, a, sizeof(float) * nn_rc, cudaMemcpyHostToDevice));

  float* d_b1;
  CHECKCALL(cudaMalloc(&d_b1, sizeof(float)* nn_rc));
  CHECKCALL(cudaMemcpy(d_b1, b1, sizeof(float) * nn_rc, cudaMemcpyHostToDevice));

  float* d_b2;
  CHECKCALL(cudaMalloc(&d_b2, sizeof(float)* nn_rc));
  CHECKCALL(cudaMemcpy(d_b2, b2, sizeof(float) * nn_rc, cudaMemcpyHostToDevice));

  float* d_x1;
  CHECKCALL(cudaMalloc(&d_x1, sizeof(float) * nn_mem));
  CHECKCALL(cudaMemcpy(d_x1, x1, sizeof(float) * nn_mem, cudaMemcpyHostToDevice));

  float* d_x2;
  CHECKCALL(cudaMalloc(&d_x2, sizeof(float) * nn_mem));
  CHECKCALL(cudaMemcpy(d_x2, x2, sizeof(float) * nn_mem, cudaMemcpyHostToDevice));

  int* d_rw_y;
  CHECKCALL(cudaMalloc(&d_rw_y, sizeof(int) * nn));
  CHECKCALL(cudaMemcpy(d_rw_y, rw_y, sizeof(int)* (nn), cudaMemcpyHostToDevice));

  int* d_rw2cw;
  CHECKCALL(cudaMalloc(&d_rw2cw, sizeof(int) * nn));
  CHECKCALL(cudaMemcpy(d_rw2cw, rw2cw, sizeof(int)* (nn), cudaMemcpyHostToDevice));

  float* d_rw_ystar;
  CHECKCALL(cudaMalloc(&d_rw_ystar, sizeof(float) * nn));
  CHECKCALL(cudaMemcpy(d_rw_ystar, rw_ystar, sizeof(float)* nn, cudaMemcpyHostToDevice));

  float* d_cw_ystar;
  CHECKCALL(cudaMalloc(&d_cw_ystar, sizeof(float) * nn));
  CHECKCALL(cudaMemcpy(d_cw_ystar, cw_ystar, sizeof(float)* nn, cudaMemcpyHostToDevice));

  // grab the current time for use as a seed in device random numbers
  struct timeval tv;
  gettimeofday(&tv,NULL);
  int seed = tv.tv_usec;

  int cntr = 0;

  for (i=1;i<ssamples;i++) {
    if (i % (ssamples/50) == 0) {
      if (i<bburnin) Rprintf("b"); //Show heartbeat...
      else Rprintf("s");
    }
    
    // Update a's and b's
    gibbs_getABs_2D<<<nv_bpg_ab,nv_tpb>>>(d_a, d_b1, d_b2, d_cw_ystar,
                                          d_x1, d_x2, d_iabprior, d_cw_memidx, d_cw_rclen, 
			                  d_cw_rcstart, nn_rc, nn_mem, nn, seed, cntr);
    CHECKCALL(cudaDeviceSynchronize());
    cntr+=20;

    // Update x's
    //gibbs_getXs_2D<<<nv_bpg_x,nv_tpb>>>(d_x1, d_x2, d_rw_ystar, d_a, d_b1, d_b2,
    //					d_ixprior, d_rw_rcidx, d_rw_memlen, 
    //	         	      		d_rw_memstart, nn_mem, nn_rc, seed, cntr);
    //CHECKCALL(cudaDeviceSynchronize());
    cntr+=20;

    // Update y stars
    gibbs_getYstar_2D<<<nv_bpg_ystar,nv_tpb>>>(d_rw_rcidx, d_rw_memidx, d_a, d_b1, d_b2,
    					       d_x1, d_x2, d_rw_y, d_cw_ystar,
			  	               d_rw_ystar, d_rw2cw, seed,
					       nn/nthreads+1, nn, cntr); 
    CHECKCALL(cudaDeviceSynchronize());
    cntr+=20;

    if (i > bburnin & (i - bburnin) % tthin == 0) {
       a += nn_rc; b1 += nn_rc; b2 += nn_rc; x1 += nn_mem; x2 += nn_mem;
       CHECKCALL(cudaMemcpy(a, d_a, sizeof(float) * nn_rc, cudaMemcpyDeviceToHost));
       CHECKCALL(cudaMemcpy(b1, d_b1, sizeof(float) * nn_rc, cudaMemcpyDeviceToHost));
       CHECKCALL(cudaMemcpy(b2, d_b1, sizeof(float) * nn_rc, cudaMemcpyDeviceToHost));
       CHECKCALL(cudaMemcpy(x1, d_x1, sizeof(float) * nn_mem, cudaMemcpyDeviceToHost));     
       CHECKCALL(cudaMemcpy(x2, d_x2, sizeof(float) * nn_mem, cudaMemcpyDeviceToHost));     
    }
  }

  // Free malloc'ed memory
  CHECKCALL(cudaFree(d_ixprior));
  CHECKCALL(cudaFree(d_iabprior));
  CHECKCALL(cudaFree(d_a));
  CHECKCALL(cudaFree(d_b1));
  CHECKCALL(cudaFree(d_b2));
  CHECKCALL(cudaFree(d_x1));
  CHECKCALL(cudaFree(d_x2));
  CHECKCALL(cudaFree(d_cw_rclen));
  CHECKCALL(cudaFree(d_cw_rcstart));
  CHECKCALL(cudaFree(d_cw_memidx));
  CHECKCALL(cudaFree(d_rw_rcidx));
  CHECKCALL(cudaFree(d_rw_memidx));
  CHECKCALL(cudaFree(d_rw_memstart));
  CHECKCALL(cudaFree(d_rw_y));
  CHECKCALL(cudaFree(d_rw2cw));
  CHECKCALL(cudaFree(d_cw_ystar));
  CHECKCALL(cudaFree(d_rw_ystar));
}


/****************************************************
* Functions used only in the EM estimation          *
*****************************************************/

//
// Device code to update a single y star
//
__global__ void em_getYstar_lookup_2D(int *rw_rcidx, int *rw_memidx, float *a, float *b1,
	   		 float *b2, float *x1, float *x2, int *rw_y, float *cw_ystar,
			 float *rw_ystar, int *rw2cw, int npt, int nn, float *im_tbl,
			 float im_tbl_min, float im_tbl_range, int im_tbl_length) {

  // Move im table to shared memory
  __shared__ float im_tbl_share[10000];
  for (int j=0; j<im_tbl_length/blockDim.x+1; j++) {
      int id = j*blockDim.x + threadIdx.x;
      if (id < im_tbl_length) {
        im_tbl_share[id] = im_tbl[id];
      }
  }
  __syncthreads();
   
  for (int i=0; i<npt; i++) {
      int id = i*(blockDim.x*gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
      if (id < nn) {
       	 float xb = a[rw_rcidx[id]] + b1[rw_rcidx[id]]*x1[rw_memidx[id]] + b2[rw_rcidx[id]]*x2[rw_memidx[id]];
      	 float txb = - (float) rw_y[id] * xb;
	 float Ee = -txb + lookup(txb, im_tbl_share, im_tbl_min, im_tbl_range, im_tbl_length); 
      	 cw_ystar[rw2cw[id]] = rw_ystar[id] = (float) rw_y[id] * Ee;
      }
  }
};

//
// Device code to update a single y star
//
__global__ void em_getYstar_2D(int *rw_rcidx, int *rw_memidx, float *a, float *b1, float *b2,
	   	         float *x1, float *x2, int *rw_y, float *cw_ystar, float *rw_ystar,
			 int *rw2cw, int npt, int nn) {

  for (int i=0; i<npt; i++) {
      int id = i*(blockDim.x*gridDim.x) + blockIdx.x*blockDim.x + threadIdx.x;
      if (id < nn) {
       	 float xb = a[rw_rcidx[id]] + b1[rw_rcidx[id]]*x1[rw_memidx[id]] + b2[rw_rcidx[id]]*x2[rw_memidx[id]];
      	 float txb = - (float) rw_y[id] * xb;
	 float Ee = -txb + inv_mills(txb);
      	 cw_ystar[rw2cw[id]] = rw_ystar[id] = (float) rw_y[id] * Ee;
      }
  }
};


//
// Update a and b 
//
__device__ static inline void em_getAB_2D(float *a, float *b1, float *b2, float *ystar,
	   	  	      	          float *x1, float *x2, float *iPrior, int *xidx, int n) {
  float ixpx[9];
  float bvec[3];
  blm_2D(bvec, ixpx, ystar, x1, x2, iPrior, xidx, n);
  *a = bvec[0];
  *b1 = bvec[1];
  *b2 = bvec[2];
};

__global__ void em_getABs_2D(float *a, float *b1, float *b2, float *cw_ystar, float *x1, float *x2,
	   	             float *iabprior, int *cw_memidx, int *cw_rclen, int *cw_rcstart, int nn_rc,
			     int nn_mem, int nn) {
    const int np = nn_rc/(blockDim.x*gridDim.x) + 1;
    for (int i=0;i<np;i++) {
    	int id = i*(blockDim.x*gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
        if (id < nn_rc) { 		  
	   int j = cw_rcstart[id];
      	   em_getAB_2D(&a[id], &b1[id], &b2[id], &cw_ystar[j], x1, x2, iabprior, &cw_memidx[j], cw_rclen[id]);
	}
    }
}

//
// Update X
//
__device__ static void em_getX_2D(float *x1, float *x2, float *ystar, float *a, float *b1,
	   	       		  float *b2, float *iPrior, int *abidx, int n) {
  float meanx[2], iBpB[4], BpB[4], BpY[2];
  int i;
  BpB[0] = BpB[1] = BpB[2] = BpB[3] = BpY[0] = BpY[1] = 0.0f;
  for (i=0;i<n;i++) {
     BpB[0] += b1[abidx[i]]*b1[abidx[i]];
     BpB[1] += b1[abidx[i]]*b2[abidx[i]];
     BpB[3] += b2[abidx[i]]*b2[abidx[i]];
     BpY[0] += (ystar[i]-a[abidx[i]])*b1[abidx[i]];
     BpY[1] += (ystar[i]-a[abidx[i]])*b2[abidx[i]];   
  }
  BpB[2] = BpB[1];
  for (i=0; i<4; i++) BpB[i] += iPrior[i];
  solve2d(meanx, iBpB, BpB, BpY);
  *x1 = meanx[0];
  *x2 = meanx[1];
};

__global__ void em_getXs_2D(float *x1, float *x2, float *rw_ystar, float *a, float *b1, float *b2,
	   	            float *ixprior, int *rw_rcidx, int *rw_memlen, int *rw_memstart,
			    int nn_mem, int nn_rc) {
    const int np = nn_mem/(blockDim.x*gridDim.x) + 1;
    for (int i=0;i<np;i++) {
    	int id = i*(blockDim.x*gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
	if (id < nn_mem) {
	    int j = rw_memstart[id];
	    em_getX_2D(&x1[id], &x2[id], &rw_ystar[j], a, b1, b2, ixprior,
	            &rw_rcidx[j], rw_memlen[id]); 
	}
    }
}

//
// Main call
//
extern "C" void em_ideal_2D(float *x1, float *x2, float *a, float *b1, float *b2, float *iabprior, float *ixprior, 
	   int *rw_y, float *rw_ystar, int *rw_rcidx, int *rw_memidx, int *rw_memstart, int *rw_memlen, 
	   int *cw_y, float * cw_ystar, int *cw_memidx, int *cw_rcstart, int *cw_rclen, int *rw2cw, 
           int *burnin, int *samples, int *n, int *n_rc, int *n_mem, int *thin, int *blocks, int *threads,
           float *im_tbl, float *im_tbl_min, float *im_tbl_range, int *im_tbl_length) {
	   
  int i;
  int ssamples = *samples;
  const int nn = *n;
  const int bburnin = *burnin;
  const int nn_rc = *n_rc;
  const int nn_mem = *n_mem;
  const int tthin = *thin;

  Rprintf("\nNumber of IM table pts %i.\n", *im_tbl_length);
  Rprintf("Number of IM table min %5.2f.\n", *im_tbl_min);
  Rprintf("Number of IM table range %5.2f.\n\n", *im_tbl_range);
     
  CUDAInfo *infop = cuda_init(getenv("R123_CUDA_DEVICE")); 
  const int nv_tpb = (*threads) > 0 ? *threads : infop->threads_per_block;
  const int nv_bpg_ystar = (*blocks) > 0 ? *blocks : infop->blocks_per_grid; 
  const int nv_bpg_ab = nv_bpg_ystar > nn_rc/nv_tpb ? nn_rc/nv_tpb +1 : nv_bpg_ystar ;
  const int nv_bpg_x  = nv_bpg_ystar > nn_mem/nv_tpb ? nn_mem/nv_tpb +1 : nv_bpg_ystar ;
  const int nthreads = nv_bpg_ystar*nv_tpb;

  Rprintf("\nGPU Info...\n");
  Rprintf("Threads per block:        %i\n", nv_tpb);
  Rprintf("Max. Blocks per grid:     %i\n", nv_bpg_ystar);  
  Rprintf("RC blocks:                %i\n", nv_bpg_ab);
  Rprintf("Member blocks:            %i\n\n", nv_bpg_x);

  Rprintf("Taking %i EM steps (%i per dot)...\n\n", *samples, *samples/50);
  Rprintf("|-------------------------------------------------|\n|");

  // Allocate device memory
  float* d_ixprior;
  CHECKCALL(cudaMalloc(&d_ixprior, sizeof(float) * 4));
  CHECKCALL(cudaMemcpy(d_ixprior, ixprior, sizeof(float)*4, cudaMemcpyHostToDevice));

  float* d_iabprior;
  CHECKCALL(cudaMalloc(&d_iabprior, sizeof(float) * 9));
  CHECKCALL(cudaMemcpy(d_iabprior, iabprior, sizeof(float)*9, cudaMemcpyHostToDevice));

  int* d_cw_rclen;
  CHECKCALL(cudaMalloc(&d_cw_rclen, sizeof(int) * nn_rc));
  CHECKCALL(cudaMemcpy(d_cw_rclen, cw_rclen, sizeof(int)*nn_rc, cudaMemcpyHostToDevice));

  int* d_cw_rcstart;
  CHECKCALL(cudaMalloc(&d_cw_rcstart, sizeof(int) * nn_rc));
  CHECKCALL(cudaMemcpy(d_cw_rcstart, cw_rcstart, sizeof(int)*nn_rc, cudaMemcpyHostToDevice));

  int* d_cw_memidx;
  CHECKCALL(cudaMalloc(&d_cw_memidx, sizeof(int) * nn));
  CHECKCALL(cudaMemcpy(d_cw_memidx, cw_memidx, sizeof(int)* (nn), cudaMemcpyHostToDevice));

  int* d_rw_memidx;
  CHECKCALL(cudaMalloc(&d_rw_memidx, sizeof(int) * nn));
  CHECKCALL(cudaMemcpy(d_rw_memidx, rw_memidx, sizeof(int)* (nn), cudaMemcpyHostToDevice));

  int* d_rw_memstart;
  CHECKCALL(cudaMalloc(&d_rw_memstart, sizeof(int) * nn_mem));
  CHECKCALL(cudaMemcpy(d_rw_memstart, rw_memstart, sizeof(int)* (nn_mem), cudaMemcpyHostToDevice));

  int* d_rw_memlen;
  CHECKCALL(cudaMalloc(&d_rw_memlen, sizeof(int) * nn_mem));
  CHECKCALL(cudaMemcpy(d_rw_memlen, rw_memlen, sizeof(int)* (nn_mem), cudaMemcpyHostToDevice));

  int* d_rw_rcidx;
  CHECKCALL(cudaMalloc(&d_rw_rcidx, sizeof(int) * nn));
  CHECKCALL(cudaMemcpy(d_rw_rcidx, rw_rcidx, sizeof(int)* (nn), cudaMemcpyHostToDevice));

  float* d_a;
  CHECKCALL(cudaMalloc(&d_a, sizeof(float) * nn_rc));
  CHECKCALL(cudaMemcpy(d_a, a, sizeof(float) * nn_rc, cudaMemcpyHostToDevice));

  float* d_b1;
  CHECKCALL(cudaMalloc(&d_b1, sizeof(float)* nn_rc));
  CHECKCALL(cudaMemcpy(d_b1, b1, sizeof(float) * nn_rc, cudaMemcpyHostToDevice));

  float* d_b2;
  CHECKCALL(cudaMalloc(&d_b2, sizeof(float)* nn_rc));
  CHECKCALL(cudaMemcpy(d_b2, b2, sizeof(float) * nn_rc, cudaMemcpyHostToDevice));

  float* d_x1;
  CHECKCALL(cudaMalloc(&d_x1, sizeof(float) * nn_mem));
  CHECKCALL(cudaMemcpy(d_x1, x1, sizeof(float) * nn_mem, cudaMemcpyHostToDevice));

  float* d_x2;
  CHECKCALL(cudaMalloc(&d_x2, sizeof(float) * nn_mem));
  CHECKCALL(cudaMemcpy(d_x2, x2, sizeof(float) * nn_mem, cudaMemcpyHostToDevice));

  int* d_rw_y;
  CHECKCALL(cudaMalloc(&d_rw_y, sizeof(int) * nn));
  CHECKCALL(cudaMemcpy(d_rw_y, rw_y, sizeof(int)* (nn), cudaMemcpyHostToDevice));

  int* d_rw2cw;
  CHECKCALL(cudaMalloc(&d_rw2cw, sizeof(int) * nn));
  CHECKCALL(cudaMemcpy(d_rw2cw, rw2cw, sizeof(int)* (nn), cudaMemcpyHostToDevice));

  float* d_rw_ystar;
  CHECKCALL(cudaMalloc(&d_rw_ystar, sizeof(float) * nn));
  CHECKCALL(cudaMemcpy(d_rw_ystar, rw_ystar, sizeof(float)* nn, cudaMemcpyHostToDevice));

  float* d_cw_ystar;
  CHECKCALL(cudaMalloc(&d_cw_ystar, sizeof(float) * nn));
  CHECKCALL(cudaMemcpy(d_cw_ystar, cw_ystar, sizeof(float)* nn, cudaMemcpyHostToDevice));

  // value of inverse mills table for lookup
  float* d_im_tbl;
  CHECKCALL(cudaMalloc(&d_im_tbl, sizeof(float) * (*im_tbl_length)));
  CHECKCALL(cudaMemcpy(d_im_tbl, im_tbl, sizeof(float)* (*im_tbl_length), cudaMemcpyHostToDevice));

  for (i=1;i<ssamples;i++) {
    if (i % (ssamples/50) == 0) Rprintf("s");

    // Update y stars
    // Rprintf("Update Y stars...\n");
    em_getYstar_2D<<<nv_bpg_ystar,nv_tpb>>>(d_rw_rcidx, d_rw_memidx, d_a, d_b1, d_b2, d_x1, d_x2,
     					    d_rw_y, d_cw_ystar, d_rw_ystar, d_rw2cw, nn/nthreads+1, nn); 
//    em_getYstar_lookup_2D<<<nv_bpg_ystar,nv_tpb>>>(d_rw_rcidx, d_rw_memidx, d_a, d_b1, d_b2,
//    					           d_x1, d_x2, d_rw_y, d_cw_ystar,
//			  	                   d_rw_ystar, d_rw2cw, nn/nthreads+1, nn,
//				                   d_im_tbl, *im_tbl_min, *im_tbl_range,
//						   *im_tbl_length);
//    if (check_values(d_rw_ystar, nn) > -1) {
//       int pos = check_values(d_rw_ystar,nn);
//       Rprintf("Found in NaN in d_rw_ystar at %i.\n", pos);
//    }
    CHECKCALL(cudaDeviceSynchronize());

    for (int j=0; j<1; j++) {
    	// Update a's and b's
	//Rprintf("Update ABs...\n");
    	em_getABs_2D<<<nv_bpg_ab,nv_tpb>>>(d_a, d_b1, d_b2, d_cw_ystar, d_x1, d_x2,
				 d_iabprior, d_cw_memidx, d_cw_rclen, 
			         d_cw_rcstart, nn_rc, nn_mem, nn);
    	CHECKCALL(cudaDeviceSynchronize());

	//if (check_values(d_b1, nn_rc) > -1) Rprintf("Found in NaN in d_b1 at step %i\n",i);
	//if (check_values(d_b2, nn_rc) > -1) Rprintf("Found in NaN in d_b2 at step %i\n",i);
	//if (check_values(d_a, nn_rc) > -1) Rprintf("Found in NaN in d_a at step %i\n", i);

    	// Update x's
	// Rprintf("Update Xs...\n");
    	em_getXs_2D<<<nv_bpg_x,nv_tpb>>>(d_x1, d_x2, d_rw_ystar, d_a, d_b1, d_b2, d_ixprior,
			       d_rw_rcidx, d_rw_memlen, 
	         	       d_rw_memstart, nn_mem, nn_rc);

	//if (check_values(d_x1, nn_mem) > -1) Rprintf("Found in NaN in d_x1\n");
	//if (check_values(d_x2, nn_mem) > -1) Rprintf("Found in NaN in d_x2\n");

        CHECKCALL(cudaDeviceSynchronize());
    }


    if (i > bburnin & (i - bburnin) % tthin == 0) {
       a += nn_rc; b1 += nn_rc; b2 += nn_rc; x1 += nn_mem; x2 += nn_mem;
       CHECKCALL(cudaMemcpy(a, d_a, sizeof(float) * nn_rc, cudaMemcpyDeviceToHost));
       CHECKCALL(cudaMemcpy(b1, d_b1, sizeof(float) * nn_rc, cudaMemcpyDeviceToHost));
       CHECKCALL(cudaMemcpy(b2, d_b2, sizeof(float) * nn_rc, cudaMemcpyDeviceToHost));
       CHECKCALL(cudaMemcpy(x1, d_x1, sizeof(float) * nn_mem, cudaMemcpyDeviceToHost));     
       CHECKCALL(cudaMemcpy(x2, d_x2, sizeof(float) * nn_mem, cudaMemcpyDeviceToHost));     
    }
  }

  // Free malloc'ed memory
  CHECKCALL(cudaFree(d_ixprior));
  CHECKCALL(cudaFree(d_iabprior));
  CHECKCALL(cudaFree(d_a));
  CHECKCALL(cudaFree(d_b1));
  CHECKCALL(cudaFree(d_b2));
  CHECKCALL(cudaFree(d_x1));
  CHECKCALL(cudaFree(d_x2));
  CHECKCALL(cudaFree(d_cw_rclen));
  CHECKCALL(cudaFree(d_cw_rcstart));
  CHECKCALL(cudaFree(d_cw_memidx));
  CHECKCALL(cudaFree(d_rw_rcidx));
  CHECKCALL(cudaFree(d_rw_memidx));
  CHECKCALL(cudaFree(d_rw_memstart));
  CHECKCALL(cudaFree(d_rw_y));
  CHECKCALL(cudaFree(d_rw2cw));
  CHECKCALL(cudaFree(d_cw_ystar));
  CHECKCALL(cudaFree(d_rw_ystar));
  CHECKCALL(cudaFree(d_im_tbl));
}
