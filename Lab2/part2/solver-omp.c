#include "heat.h"

#define NB 8

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )

/*
 * Blocked Jacobi solver: one iteration step
 */

// Jacobi algorithm does not have any dependecies as it uses one matrix as input
// and another one as output. This way we can organize that any thread does does
// the computation of a whole area and then we have to synchornize the final sum
// between areas to not overwrite or misacces the variable.
// Now its only preference (should be based on performace) on how we organize this
// "areas", doing it by rows, columns or blocks are the most popular options.

// As the code is already given to do it in blocks we will do it in blocks.
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
    int nbx, bx, nby, by;
  
    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    #pragma omp parallel for collapse(2) reduction(+:sum) private(diff)
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            utmp[i*sizey+j]= 0.25 * (u[ i*sizey     + (j-1) ]+
					    u[ i*sizey     + (j+1) ]+
				        u[ (i-1)*sizey + j     ]+
				        u[ (i+1)*sizey + j     ]);
	            diff = utmp[i*sizey+j] - u[i*sizey + j]; // we could declare it here (not sure if there is performance difference)
	            sum += diff * diff; 
	        }

    return sum;
}

/*
 * Blocked Red-Black solver: one iteration step
 */
double relax_redblack (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
    int lsw;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    // Computing "Red" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = ii%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }

    // Computing "Black" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = (ii+1)%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }

    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */

// With Gauss-Seidel we are facing a different type of parallelization as
// each step requires values from previous ones (except th first). This
// means each task depends on previous values that have to be computed,
// before starting our task. This is just another alogorithm that can
// also be parallelized but it won't be embarrassingly parallel, it will
// follow a hiererchical patter similar to a tree.

// (1) We are told to use explicit tasks with dependencies (task + depend (in, out, inout))
// (2) Parallelize using a do-across (ordered + depend (sink, source))

// (1)
/*
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
	

    #pragma omp parallel
    #pragma omp single
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) {
            #pragma omp task depend(in: u[(ii-1)*sizey+jj], u[ii*sizey+(jj-1)]) \
                             depend(out: u[ii*sizey+jj]) \
                             private(diff,unew)
                            {
                double local_sum = 0.0;
                for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) {
                    for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                    unew= 0.25 * (u[i*sizey + (j-1)]+
                        u[ i*sizey	+ (j+1) ]+
                        u[ (i-1)*sizey	+ j     ]+
                        u[ (i+1)*sizey	+ j     ]);
                    diff = unew - u[i*sizey+ j];
                    local_sum += diff * diff; 
                    u[i*sizey+j]=unew;
                    }
                }
                #pragma omp atomic
                sum += local_sum;
            }
        }
            

    return sum;
}
*/
// (2)

double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    
    #pragma omp parallel reduction(+:sum) private(unew, diff) 
    {
        #pragma omp for ordered(2)
        for (int ii=0; ii<nbx; ii++){
            for (int jj=0; jj<nby; jj++) {
                #pragma omp ordered depend(sink: ii-1, jj) depend(sink: ii,jj-1) 
                    for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) {
                        for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                            unew= 0.25 * (u[i*sizey + (j-1)]+
                                u[ i*sizey	+ (j+1) ]+
                                u[ (i-1)*sizey	+ j     ]+
                                u[ (i+1)*sizey	+ j     ]);
                            diff = unew - u[i*sizey+ j];
                            sum += diff * diff; 
                        	u[i*sizey+j]=unew;
                        }
                    }
                #pragma omp ordered depend(source)
    	    }
        }
    }
    return sum;
}

