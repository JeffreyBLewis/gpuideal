#include <R.h>
#include <sys/time.h>

#include "Random123-1.09/include/Random123/philox.h"
#include "boxmuller.hpp"
#include "inv_mills.h"
#include "util_cuda.h"	// for cuda_init, CHECKCALL
#include "lookup.h"

#define abs(x) ((x)<0.0f ? -(x) : (x))

//Globals for util_cuda.h
int debug = 0;
const char *progname = "ideal.cu";


/*****************************************************************
* Common functions for both the EM and Gibbs Sampling estimator  *
******************************************************************/

__device__ static inline void xpx_xpy(float *xpx, float *xpy, float *x, float *y, int *xidx, int n) {
  xpx[2]=xpx[3]=0.0f;
  xpy[0]=xpy[1]=0.0f;
  for (int i=0;i<n;i++) {
    xpx[2] += x[xidx[i]];
    xpx[3] += x[xidx[i]]*x[xidx[i]];
    xpy[0] += y[i];
    xpy[1] += x[xidx[i]]*y[i];
  }
  xpx[0] = (float) n;
  xpx[1] = xpx[2];
};

__device__ static inline void  chol(float *chol, float *x) {
  const float sqrtx0 = sqrtf(x[0]);
  chol[0] = sqrtx0;
  chol[1] = 0.0f;
  chol[2] = x[1]/sqrtx0;
  chol[3] = sqrtf(x[0]*x[3]-x[1]*x[1])/sqrtx0;
};

__device__ static inline void solve2d(float *bbv, float *iXpx, float *xpx, float *xpy) {
  const float denom = (xpx[0]*xpx[3] - xpx[1]*xpx[2]);
  bbv[0] = (xpy[0]*xpx[3]-xpy[1]*xpx[1])/denom;
  bbv[1] = (xpx[0]*xpy[1]-xpx[2]*xpy[0])/denom;
  iXpx[0] = xpx[3]/denom;
  iXpx[1] = -xpx[2]/denom;
  iXpx[2] = -xpx[1]/denom;
  iXpx[3] = xpx[0]/denom;
};

__device__ static inline void blm(float *bv, float *ixpx, float *y, float *x, float *iPrior, int *xidx, int n) {
  float xpx[4], xpy[2];
  xpx_xpy(xpx,xpy,x,y,xidx,n);
  xpx[0] += iPrior[0];
  xpx[3] += iPrior[3];
  xpx[1] += iPrior[2];
  xpx[2] = xpx[1];
  solve2d(bv,ixpx,xpx,xpy);
};


/************************************************************
* Functions needed only by the Gibbs sampler                *
*************************************************************/

//
// Draw truncated norm a la Christian P. Robert ``Simulation of Truncated normal variables''
// LSTA, Universite Pierre et Marie Carie, Paris (2009)
//
__device__ static inline float d_rtnorm(float lb, 
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
__device__ static inline float d_rtnorm_met(float lb, float ystar0,
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
__global__ void gibbs_getYstar(int *rw_rcidx,int *rw_memidx,float *a,float *b,float *x,
                         int *rw_y, float *cw_ystar, float *rw_ystar, int *rw2cw, int seed,
			 int npt, int nn, int cntr) {

  int id = blockIdx.x * blockDim.x + threadIdx.x;
  philox4x32_key_t k = {{(unsigned int) id, (unsigned int) seed*3}};
  philox4x32_ctr_t c = {{(unsigned int) cntr}};
  
  for (int i=0; i<npt; i++) {
      int id = i*(blockDim.x*gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
      if (id < nn) {
       	 float xb = a[rw_rcidx[id]] + b[rw_rcidx[id]]*x[rw_memidx[id]];
      	 float txb = - (float) rw_y[id] * xb;
         float ee = (float) d_rtnorm(txb,k,c);
      	 cw_ystar[rw2cw[id]] = rw_ystar[id] = xb + (float) rw_y[id] * ee;
	 c.incr();
      }
  }
};


//
// Update a and b 
//
__device__ static inline void gibbs_getAB(float *a, float *b, float *ystar, float *x, float *iPrior, int *xidx, int n, float2 veps) {
  float ixpx[4], cholixpx[4], beps[2];
  float bvec[2];

  blm(bvec,ixpx,ystar,x,iPrior,xidx,n);
  beps[0] = veps.x;
  beps[1] = veps.y;
  chol(cholixpx,ixpx);
  beps[0] = veps.x*cholixpx[0];
  beps[1] = veps.x*cholixpx[2] + veps.y*cholixpx[3];
  *a = bvec[0] + beps[0];
  *b = bvec[1] + beps[1];
};

__global__ void gibbs_getABs(float *a, float *b, float *cw_ystar, float *x, float *iabprior, int *cw_memidx, 
		       int *cw_rclen, int *cw_rcstart, int nn_rc, int nn_mem, int nn, int seed, int cntr) {
    const int np = nn_rc/(blockDim.x*gridDim.x) + 1;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float2 veps;
    r123::Philox4x32 rand;
    philox4x32_key_t k = {{(unsigned int) id, (unsigned int) seed*2}};
    philox4x32_ctr_t c = {{(unsigned int) cntr}};

    /*
    // Move x's to shared memory 
    __shared__ float x_share[10000];    
    for (int j=0; j<nn_mem/blockDim.x+1; j++) {
      int id = j*blockDim.x + threadIdx.x;
      if (id < nn_mem) {
        x_share[id] = x[id];
      }
    }
    __syncthreads();
    */

    for (int i=0;i<np;i++) {
    	int id = i*(blockDim.x*gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
        if (id < nn_rc) { 		  
	   typename r123::Philox4x32::ctr_type uu = rand(c,k);
	   veps  = r123::boxmuller(uu[0],uu[1]);
	   int j = cw_rcstart[id];
      	   gibbs_getAB(&a[id], &b[id], &cw_ystar[j], x, iabprior, &cw_memidx[j], cw_rclen[id], veps ); // Could be x_share if using shared memory
	   c.incr();
	}
    }
}

//
// Update X
//
__device__ static void gibbs_getX(float *x, float *ystar, float *a, float *b, float *iPrior, int *abidx, int n, double e) {
  float sumbb, sumyb, meanx, sdx;
  int i;
  
  sumbb = sumyb = 0.0f;
  for (i=0;i<n;i++) {
     sumbb += b[abidx[i]]*b[abidx[i]];
     sumyb += (ystar[i]-a[abidx[i]])*b[abidx[i]];
  }
  sumbb += (*iPrior);
  meanx = sumyb/sumbb;
  sdx = sqrtf(1.0f/sumbb);
  *x = meanx + sdx*e; 
};

__global__ void gibbs_getXs(float *x, float *rw_ystar, float *a, float *b, float *ixprior, int *rw_rcidx, 
		      int *rw_memlen, int *rw_memstart, int nn_mem, int nn_rc, int seed, int cntr) {
    const int np = nn_mem/(blockDim.x*gridDim.x) + 1;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float2 f2;
    r123::Philox4x32 rand;
    philox4x32_key_t k = {{(unsigned int) id, (unsigned int) seed*3}};
    philox4x32_ctr_t c = {{(unsigned int) cntr}};

    // Move a's and b's to shared memory
   /*
    __shared__ float a_share[4000];
    __shared__ float b_share[4000];
    for (int j=0; j<nn_rc/blockDim.x+1; j++) {
      int id = j*blockDim.x + threadIdx.x;
      if (id < nn_rc) {
        a_share[id] = a[id];
        b_share[id] = b[id];
      }
    }
    __syncthreads();
    */

    for (int i=0;i<np;i++) {
    	int id = i*(blockDim.x*gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
	if (id < nn_mem) {
	    int j = rw_memstart[id];
	    typename r123::Philox4x32::ctr_type uu = rand(c,k);
	    f2  = r123::boxmuller(uu[0],uu[1]);
	    gibbs_getX(&x[id], &rw_ystar[j], a, b, ixprior, &rw_rcidx[j], rw_memlen[id], f2.x); // Could be a_share, b_share
	}
	c.incr();
    }
}

//
// Main call
//
extern "C" void gibbs_ideal(float *x, float *a, float *b, float *iabprior, float *ixprior, 
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

  float* d_b;
  CHECKCALL(cudaMalloc(&d_b, sizeof(float)* nn_rc));
  CHECKCALL(cudaMemcpy(d_b, b, sizeof(float) * nn_rc, cudaMemcpyHostToDevice));

  float* d_x;
  CHECKCALL(cudaMalloc(&d_x, sizeof(float) * nn_mem));
  CHECKCALL(cudaMemcpy(d_x, x, sizeof(float) * nn_mem, cudaMemcpyHostToDevice));

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
    gibbs_getABs<<<nv_bpg_ab,nv_tpb>>>(d_a, d_b, d_cw_ystar, d_x, d_iabprior, d_cw_memidx, d_cw_rclen, 
			         d_cw_rcstart, nn_rc, nn_mem, nn, seed, cntr);
    CHECKCALL(cudaDeviceSynchronize());
    cntr+=20;

    // Update x's
    gibbs_getXs<<<nv_bpg_x,nv_tpb>>>(d_x, d_rw_ystar, d_a, d_b, d_ixprior, d_rw_rcidx, d_rw_memlen, 
	         	       d_rw_memstart, nn_mem, nn_rc, seed, cntr);
    CHECKCALL(cudaDeviceSynchronize());
    cntr+=20;

    // Update y stars
    gibbs_getYstar<<<nv_bpg_ystar,nv_tpb>>>(d_rw_rcidx, d_rw_memidx, d_a, d_b, d_x, d_rw_y, d_cw_ystar,
			  	      d_rw_ystar, d_rw2cw, seed, nn/nthreads+1,nn, cntr); 
    CHECKCALL(cudaDeviceSynchronize());
    cntr+=20;

    if (i > bburnin & (i - bburnin) % tthin == 0) {
       a += nn_rc; b += nn_rc; x += nn_mem;
       CHECKCALL(cudaMemcpy(a, d_a, sizeof(float) * nn_rc, cudaMemcpyDeviceToHost));
       CHECKCALL(cudaMemcpy(b, d_b, sizeof(float) * nn_rc, cudaMemcpyDeviceToHost));
       CHECKCALL(cudaMemcpy(x, d_x, sizeof(float) * nn_mem, cudaMemcpyDeviceToHost));     
    }
  }

  // Free malloc'ed memory
  CHECKCALL(cudaFree(d_ixprior));
  CHECKCALL(cudaFree(d_iabprior));
  CHECKCALL(cudaFree(d_a));
  CHECKCALL(cudaFree(d_b));
  CHECKCALL(cudaFree(d_x));
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
__global__ void em_getYstar_lookup(int *rw_rcidx,int *rw_memidx,float *a,float *b,float *x,
                         int *rw_y, float *cw_ystar, float *rw_ystar, int *rw2cw, int seed,
			 int npt, int nn, float *im_tbl, float im_tbl_min, float im_tbl_range,
			 int im_tbl_length) {


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
       	 float xb = a[rw_rcidx[id]] + b[rw_rcidx[id]]*x[rw_memidx[id]];
      	 float txb = - (float) rw_y[id] * xb;
	 float Ee = -txb + lookup(txb, im_tbl_share, im_tbl_min, im_tbl_range, im_tbl_length); 
      	 cw_ystar[rw2cw[id]] = rw_ystar[id] = (float) rw_y[id] * Ee;
      }
  }
};

//
// Device code to update a single y star
//
__global__ void em_getYstar(int *rw_rcidx,int *rw_memidx,float *a,float *b,float *x,
                         int *rw_y, float *cw_ystar, float *rw_ystar, int *rw2cw, int seed,
			 int npt, int nn) {

  for (int i=0; i<npt; i++) {
      int id = i*(blockDim.x*gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
      if (id < nn) {
       	 float xb = a[rw_rcidx[id]] + b[rw_rcidx[id]]*x[rw_memidx[id]];
      	 float txb = - (float) rw_y[id] * xb;
	 float Ee = -txb + inv_mills(txb); 
      	 cw_ystar[rw2cw[id]] = rw_ystar[id] = (float) rw_y[id] * Ee;
      }
  }
};



//
// Update a and b 
//
__device__ static inline void em_getAB(float *a, float *b, float *ystar, float *x, float *iPrior, int *xidx, int n) {
  float ixpx[4];
  float bvec[2];

  blm(bvec,ixpx,ystar,x,iPrior,xidx,n);
  *a = bvec[0];
  *b = bvec[1];
};

__global__ void em_getABs(float *a, float *b, float *cw_ystar, float *x, float *iabprior, int *cw_memidx, 
		       int *cw_rclen, int *cw_rcstart, int nn_rc, int nn_mem, int nn) {
    const int np = nn_rc/(blockDim.x*gridDim.x) + 1;

    /*
    // Move x's to shared memory 
    __shared__ float x_share[10000];    
    for (int j=0; j<nn_mem/blockDim.x+1; j++) {
      int id = j*blockDim.x + threadIdx.x;
      if (id < nn_mem) {
        x_share[id] = x[id];
      }
    }
    __syncthreads();
    */

    for (int i=0;i<np;i++) {
    	int id = i*(blockDim.x*gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
        if (id < nn_rc) { 		  
	   int j = cw_rcstart[id];
      	   em_getAB(&a[id], &b[id], &cw_ystar[j], x, iabprior, &cw_memidx[j], cw_rclen[id]); // Could be x_share if using shared memory
	}
    }
}

//
// Update X
//
__device__ static void em_getX(float *x, float *ystar, float *a, float *b, float *iPrior, int *abidx, int n) {
  float sumbb, sumyb, meanx;
  int i;
  
  sumbb = sumyb = 0.0f;
  for (i=0;i<n;i++) {
     sumbb += b[abidx[i]]*b[abidx[i]];
     sumyb += (ystar[i]-a[abidx[i]])*b[abidx[i]];
  }
  sumbb += (*iPrior);
  meanx = sumyb/sumbb;
  *x = meanx; 
};

__global__ void em_getXs(float *x, float *rw_ystar, float *a, float *b, float *ixprior, int *rw_rcidx, 
		      int *rw_memlen, int *rw_memstart, int nn_mem, int nn_rc) {
    const int np = nn_mem/(blockDim.x*gridDim.x) + 1;
 
    // Move a's and b's to shared memory
   /*
    __shared__ float a_share[4000];
    __shared__ float b_share[4000];
    for (int j=0; j<nn_rc/blockDim.x+1; j++) {
      int id = j*blockDim.x + threadIdx.x;
      if (id < nn_rc) {
        a_share[id] = a[id];
        b_share[id] = b[id];
      }
    }
    __syncthreads();
    */

    for (int i=0;i<np;i++) {
    	int id = i*(blockDim.x*gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
	if (id < nn_mem) {
	    int j = rw_memstart[id];
	    em_getX(&x[id], &rw_ystar[j], a, b, ixprior, &rw_rcidx[j], rw_memlen[id]); // Could be a_share, b_share
	}
    }
}

//
// Main call
//
extern "C" void em_ideal(float *x, float *a, float *b, float *iabprior, float *ixprior, 
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

  float* d_b;
  CHECKCALL(cudaMalloc(&d_b, sizeof(float)* nn_rc));
  CHECKCALL(cudaMemcpy(d_b, b, sizeof(float) * nn_rc, cudaMemcpyHostToDevice));

  float* d_x;
  CHECKCALL(cudaMalloc(&d_x, sizeof(float) * nn_mem));
  CHECKCALL(cudaMemcpy(d_x, x, sizeof(float) * nn_mem, cudaMemcpyHostToDevice));

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


  // grab the current time for use as a seed in device random numbers
  struct timeval tv;
  gettimeofday(&tv,NULL);
  int seed = tv.tv_usec;

  for (i=1;i<ssamples;i++) {
    if (i % (ssamples/50) == 0) Rprintf("s");

    // Update y stars
    em_getYstar<<<nv_bpg_ystar,nv_tpb>>>(d_rw_rcidx, d_rw_memidx, d_a, d_b, d_x, d_rw_y, d_cw_ystar,
  			  	      d_rw_ystar, d_rw2cw, seed, nn/nthreads+1,nn); 
//  em_getYstar_lookup<<<nv_bpg_ystar,nv_tpb>>>(d_rw_rcidx, d_rw_memidx, d_a, d_b, d_x, d_rw_y, d_cw_ystar,
//			  	      d_rw_ystar, d_rw2cw, seed, nn/nthreads+1, nn,
//  				      d_im_tbl, *im_tbl_min, *im_tbl_range, *im_tbl_length); 
    CHECKCALL(cudaDeviceSynchronize());

    for (int j=0; j<1; j++) {
    	// Update a's and b's
    	em_getABs<<<nv_bpg_ab,nv_tpb>>>(d_a, d_b, d_cw_ystar, d_x, d_iabprior, d_cw_memidx, d_cw_rclen, 
 			         d_cw_rcstart, nn_rc, nn_mem, nn);
    	CHECKCALL(cudaDeviceSynchronize());


    	// Update x's
     	em_getXs<<<nv_bpg_x,nv_tpb>>>(d_x, d_rw_ystar, d_a, d_b, d_ixprior, d_rw_rcidx, d_rw_memlen, 
	         	       d_rw_memstart, nn_mem, nn_rc);
        CHECKCALL(cudaDeviceSynchronize());
    }


    if (i > bburnin & (i - bburnin) % tthin == 0) {
       a += nn_rc; b += nn_rc; x += nn_mem;
       CHECKCALL(cudaMemcpy(a, d_a, sizeof(float) * nn_rc, cudaMemcpyDeviceToHost));
       CHECKCALL(cudaMemcpy(b, d_b, sizeof(float) * nn_rc, cudaMemcpyDeviceToHost));
       CHECKCALL(cudaMemcpy(x, d_x, sizeof(float) * nn_mem, cudaMemcpyDeviceToHost));     
    }
  }

  // Free malloc'ed memory
  CHECKCALL(cudaFree(d_ixprior));
  CHECKCALL(cudaFree(d_iabprior));
  CHECKCALL(cudaFree(d_a));
  CHECKCALL(cudaFree(d_b));
  CHECKCALL(cudaFree(d_x));
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
