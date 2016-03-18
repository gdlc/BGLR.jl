#include <math.h>
#include <stdlib.h>
 
double randn(double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }
 
  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);
}


void sample_beta(int n,  int p, double *X,  double *x2, double *b, double *e, double varBj, 
            double varE, double minAbsBeta)
{
        double *xj;
	    double rhs, c;
	
        int j,i;	

        xj=malloc(n*sizeof(double));

        for(j=0; j<p;j++)
        {
	  		rhs=0;
	  		for(i=0; i<n; i++)
	  		{
	    		xj[i]=X[i+j*n];
	    		e[i] = e[i] + b[j]*xj[i];
	    		rhs+=xj[i]*e[i];
	  		}
	  		rhs=rhs/varE;
  	  		c=x2[j]/varE + 1.0/varBj;
	  		b[j]=rhs/c + sqrt(1.0/c)*randn(0.0,1.0);
	  
	  		for(i=0; i<n; i++)
	  		{
	    		e[i] = e[i] - b[j]*xj[i];
	  		}
	  		
          	if(fabs(b[j])<minAbsBeta)
          	{
             	b[j]=minAbsBeta;
          	}
        }
}