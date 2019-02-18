#define fabsf(x) ((x)<0.0 ? -(x) : (x))
#define sign(x) ((x>0.0) - (x<0.0))

/* 
inv_mills

Implements A. V. Swan. "Algorithm 17: The Reciprocal of 
    the Mill Ratio." Journal of the Royal Statistical Society. Series C 
     (Applied Statistics), Vol. 18, No. 1 (1969), pp. 115-116       
*/
float inv_mills(float xx) {
  int d = sign(xx);
  float x = fabsf(xx);
  float a, b, r, s, t, A0, A1, A2, B0, B1, B2;
  float fpi = 1.253314137;
  float least = -22.9;
  if (x == 0.0) return(1.0/fpi);
  if (x < least) return(0.0);
  if (x <= 2.0f) {
    s = 0.0; a = 1.0;
    t = x; r = t; b = x*x;
    while(s != t) {
      a += 2;
      s = t; r = r*b/a;
      t = t + r;
    }
    return(1.0/(fpi*__expf(0.5*x*x) - d*t));
  }
  a = 2.0; r = x; s = x; B1 = x;
  A1 = x*x + 1.0; A2 = x*(A1 + 2.0);
  B2 = A1 + 1; t = A2/B2;
  while ((r != t) & (s != t)) {
    a = a + 1.0;
    A0 = A1; A1 = A2;
    A2 = x*A1 + a*A0; B0=B1;
    B1 = B2; B2 = x*B1 + a*B0;
    r = s; s = t;
    t = A2/B2;
  }
  if (d==1) {
    return(t);
  }
  else{
    return( t/(2.0*fpi*__expf(0.5*x*x)*t - 1.0) );
  }
}

