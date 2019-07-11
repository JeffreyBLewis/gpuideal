/*  lookup table function */

__device__ inline float lookup(float z, float *tbl, float tbl_min,
			       float tbl_range, int tbl_length) {
  register float x1_frac, y1, y2;
  register int x1;
  x1_frac = (z-tbl_min)/tbl_range*((float) tbl_length);
  x1 = floorf(x1_frac);
  y1 = tbl[(int) x1];
  y2 = tbl[(int) x1+1];
  return( y1 + (y2-y1)*(x1_frac-x1) );
}
