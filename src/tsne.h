/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

// From src/tsne.cpp and src/tsne.h of Rtsne 0.17. The class is changed. Changed lines in the functions are marked "ART:".

#ifndef ART_TSNE_H
#define ART_TSNE_H

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include <vector>
#include <Rcpp.h>
using namespace std;

static inline double sign_tsne(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

template <int NDims>
class TSNE
{
public:
    TSNE(double perplexity, double theta, bool verbose, int max_iter, bool init, int stop_lying_iter,
       int mom_switch_iter, double momentum, double final_momentum, double eta, double exaggeration_factor,int num_threads);
    void trainIterations(unsigned int N, double* Y, double* cost, double* itercost);
    std::vector<double> P;                      // Set in art_layout

private:
    void computeExactGradient(double* P, double* Y, unsigned int N, int D, double* dC);
    double evaluateError(double* P, double* Y, unsigned int N, int D);
    void getCost(double* P, double* Y, unsigned int N, int D, double* costs);
    void computeSquaredEuclideanDistanceDirect(double* X, unsigned int N, int D, double* DD);
    void zeroMean(double* X, unsigned int N, int D);
    double randn();

    // Not used
    void computeGradient(double*, unsigned int*, unsigned int*, double*, double*, unsigned int, int, double*, double) {}
    double evaluateError(unsigned int*, unsigned int*, double*, double*, unsigned int, int, double) { return 0; }
    void getCost(unsigned int*, unsigned int*, double*, double*, unsigned int, int, double, double*) {}
    std::vector<double> val_P;
    std::vector<unsigned int> row_P, col_P;

    double perplexity, theta, momentum, final_momentum, eta, exaggeration_factor;
    int max_iter, stop_lying_iter, mom_switch_iter, num_threads;
    bool verbose, init, exact;
};

template <int NDims>
TSNE<NDims>::TSNE(double Perplexity, double Theta, bool Verbose, int Max_iter, bool Init, int Stop_lying_iter,
        int Mom_switch_iter, double Momentum, double Final_momentum, double Eta, double Exaggeration_factor, int Num_threads) :
    perplexity(Perplexity), theta(Theta), momentum(Momentum), final_momentum(Final_momentum), eta(Eta), exaggeration_factor(Exaggeration_factor),
    max_iter(Max_iter), stop_lying_iter(Stop_lying_iter), mom_switch_iter(Mom_switch_iter), num_threads(Num_threads),
    verbose(Verbose), init(Init), exact(theta==.0) {

    #ifdef _OPENMP
      int threads = num_threads;
      if (num_threads==0) {
        threads = omp_get_max_threads();
      }
      
      // Print notice whether OpenMP is used
      if (verbose) Rprintf("OpenMP is working. %d threads.\n", threads);
    #endif

    return;
}

template<int NDims>
void TSNE<NDims>::trainIterations(unsigned int N, double* Y, double* cost, double* itercost) {
    // Allocate some memory
    double* dY    = (double*) malloc(N * NDims * sizeof(double));
    double* uY    = (double*) malloc(N * NDims * sizeof(double));
    double* gains = (double*) malloc(N * NDims * sizeof(double));
    if(dY == NULL || uY == NULL || gains == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    for(unsigned int i = 0; i < N * NDims; i++)    uY[i] =  .0;
    for(unsigned int i = 0; i < N * NDims; i++) gains[i] = 1.0;

    // Lie about the P-values
    if(exact) { for(unsigned long i = 0; i < (unsigned long)N * N; i++) P[i] *= exaggeration_factor; }
    else {      for(unsigned long i = 0; i < row_P[N]; i++)    val_P[i] *= exaggeration_factor; }

	// Initialize solution (randomly), if not already done
	if (!init) { for(unsigned int i = 0; i < N * NDims; i++) Y[i] = randn() * .0001; }

  clock_t start = clock(), end;
  float total_time=0;
  int costi = 0; //iterator for saving the total costs for the iterations

	for(int iter = 0; iter < max_iter; iter++) {

        // Stop lying about the P-values after a while, and switch momentum
        if(iter == stop_lying_iter) {
          if(exact) { for(unsigned long i = 0; i < (unsigned long)N * N; i++)        P[i] /= exaggeration_factor; }
          else      { for(unsigned int i = 0; i < row_P[N]; i++) val_P[i] /= exaggeration_factor; }
        }
        if(iter == mom_switch_iter) momentum = final_momentum;

        // Compute (approximate) gradient
        if(exact) computeExactGradient(P.data(), Y, N, NDims, dY);
        else computeGradient(P.data(), row_P.data(), col_P.data(), val_P.data(), Y, N, NDims, dY, theta);

        // Update gains
        for(unsigned int i = 0; i < N * NDims; i++) gains[i] = (sign_tsne(dY[i]) != sign_tsne(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
        for(unsigned int i = 0; i < N * NDims; i++) if(gains[i] < .01) gains[i] = .01;

        // Perform gradient update (with momentum and gains)
        for(unsigned int i = 0; i < N * NDims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
        for(unsigned int i = 0; i < N * NDims; i++)  Y[i] = Y[i] + uY[i];

        // Make solution zero-mean
        zeroMean(Y, N, NDims);

        // Print out progress
        if((iter > 0 && (iter+1) % 50 == 0) || iter == max_iter - 1) {
            end = clock();
            double C = .0;
            if(exact) C = evaluateError(P.data(), Y, N, NDims);
            else      C = evaluateError(row_P.data(), col_P.data(), val_P.data(), Y, N, NDims, theta);  // doing approximate computation here!
            if(iter == 0) {
                if (verbose) Rprintf("Iteration %d: error is %f\n", iter + 1, C);
            }
            else {
                total_time += (float) (end - start) / CLOCKS_PER_SEC;
                if (verbose) Rprintf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter+1, C, (float) (end - start) / CLOCKS_PER_SEC);
            }
            itercost[costi] = C;
            costi++;
			  start = clock();
        }
    }
    end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;

    if(exact) getCost(P.data(), Y, N, NDims, cost);
    else      getCost(row_P.data(), col_P.data(), val_P.data(), Y, N, NDims, theta, cost);  // doing approximate computation here!
    
    // Clean up memory
    free(dY);
    free(uY);
    free(gains);
    if (verbose) Rprintf("Fitting performed in %4.2f seconds.\n", total_time);
    return;
}

template <int NDims>
void TSNE<NDims>::computeExactGradient(double* P, double* Y, unsigned int N, int D, double* dC) {
	
	// Make sure the current gradient contains zeros
	for(unsigned int i = 0; i < N * D; i++) dC[i] = 0.0;

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc((unsigned long)N * N * sizeof(double));
    if(DD == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    computeSquaredEuclideanDistanceDirect(Y, N, D, DD);   // ART: was computeSquaredEuclideanDistance

    // Compute Q-matrix and normalization sum
    double* Q    = (double*) malloc((unsigned long)N * N * sizeof(double));
    if(Q == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    double sum_Q = .0;
    for(unsigned long n = 0; n < N; n++) {
    	for(unsigned long m = 0; m < N; m++) {
            if(n != m) {
                Q[n * N + m] = 1 / (1 + DD[n * N + m]);
                sum_Q += Q[n * N + m];
            }
        }
    }

	// Perform the computation of the gradient
	for(unsigned long n = 0; n < N; n++) {
    	for(unsigned long m = 0; m < N; m++) {
            if(n != m) {
                double mult = (P[n * N + m] - (Q[n * N + m] / sum_Q)) * Q[n * N + m];
                for(int d = 0; d < D; d++) {
                    dC[n * D + d] += (Y[n * D + d] - Y[m * D + d]) * mult;
                }
            }
		}
	}

    // Free memory
    free(DD); DD = NULL;
    free(Q);  Q  = NULL;
}

template <int NDims>
double TSNE<NDims>::evaluateError(double* P, double* Y, unsigned int N, int D) {

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc((unsigned long)N * N * sizeof(double));
    double* Q = (double*) malloc((unsigned long)N * N * sizeof(double));
    if(DD == NULL || Q == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    computeSquaredEuclideanDistanceDirect(Y, N, D, DD);   // ART: was computeSquaredEuclideanDistance

    // Compute Q-matrix and normalization sum
    double sum_Q = DBL_MIN;
    for(unsigned long n = 0; n < N; n++) {
    	for(unsigned long m = 0; m < N; m++) {
            if(n != m) {
                Q[n * N + m] = 1 / (1 + DD[n * N + m]);
                sum_Q += Q[n * N + m];
            }
            else Q[n * N + m] = DBL_MIN;
        }
    }
    for(unsigned long i = 0; i < (unsigned long)N * N; i++) Q[i] /= sum_Q;

    // Sum t-SNE error
    double C = .0;
  	for(unsigned long n = 0; n < N; n++) {
  		for(unsigned long m = 0; m < N; m++) {
  			C += P[n * N + m] * log((P[n * N + m] + 1e-9) / (Q[n * N + m] + 1e-9));
  		}
  	}

    // Clean up memory
    free(DD);
    free(Q);
	return C;
}

template <int NDims>
void TSNE<NDims>::getCost(double* P, double* Y, unsigned int N, int D, double* costs) {

  // Compute the squared Euclidean distance matrix
  double* DD = (double*) malloc((unsigned long)N * N * sizeof(double));
  double* Q = (double*) malloc((unsigned long)N * N * sizeof(double));
  if(DD == NULL || Q == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
  computeSquaredEuclideanDistanceDirect(Y, N, D, DD);   // ART: was computeSquaredEuclideanDistance

  // Compute Q-matrix and normalization sum
  double sum_Q = DBL_MIN;
  for(unsigned long n = 0; n < N; n++) {
    for(unsigned long m = 0; m < N; m++) {
      if(n != m) {
        Q[n * N + m] = 1 / (1 + DD[n * N + m]);
        sum_Q += Q[n * N + m];
      }
      else Q[n * N + m] = DBL_MIN;
    }
  }
  for(unsigned long i = 0; i < (unsigned long)N * N; i++) Q[i] /= sum_Q;

  // Sum t-SNE error
  for(unsigned long n = 0; n < N; n++) {
    costs[n] = 0.0;
    for(unsigned long m = 0; m < N; m++) {
      costs[n] += P[n * N + m] * log((P[n * N + m] + 1e-9) / (Q[n * N + m] + 1e-9));
    }
  }

  // Clean up memory
  free(DD);
  free(Q);
}

template <int NDims>
void TSNE<NDims>::computeSquaredEuclideanDistanceDirect(double* X, unsigned int N, int D, double* DD) {
  const double* XnD = X;
  for(unsigned int n = 0; n < N; ++n, XnD += D) {
    const double* XmD = XnD + D;
    double* curr_elem = &DD[n*N + n];
    *curr_elem = 0.0;
    double* curr_elem_sym = curr_elem + N;
    for(unsigned int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
      *(++curr_elem) = 0.0;
      for(int d = 0; d < D; ++d) {
        *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
      }
      *curr_elem_sym = *curr_elem;
    }
  }
}

template <int NDims>
void TSNE<NDims>::zeroMean(double* X, unsigned int N, int D) {
	
	// Compute data mean
	double* mean = (double*) calloc(D, sizeof(double));
  if(mean == NULL) {  Rcpp::stop("Memory allocation failed!\n"); }
  int nD = 0;
  for(unsigned int n = 0; n < N; n++) {
    for(int d = 0; d < D; d++) {
      mean[d] += X[nD + d];
    }
    nD += D;
  }
  for(int d = 0; d < D; d++) {
    mean[d] /= (double) N;
  }

  // Subtract data mean
  nD = 0;
  for(unsigned int n = 0; n < N; n++) {
    for(int d = 0; d < D; d++) {
      X[nD + d] -= mean[d];
    }
    nD += D;
  }
  free(mean); mean = NULL;
}

template <int NDims>
double TSNE<NDims>::randn() {
  Rcpp::RNGScope scope;
	double x, y, radius;
	do {
		x = 2 * (double)R::runif(0,1) - 1;
		y = 2 * (double)R::runif(0,1) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

#endif
