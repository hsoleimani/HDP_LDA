
#ifndef HDPLDA_H_
#define HDPLDA_H_

#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>


#define NUM_INIT 20
#define SEED_INIT_SMOOTH 1.0
#define EPS 1e-30
#define PI 3.14159265359


typedef struct
{
    int* words;
    int* counts;
    int length;
    int total;
} document;


typedef struct
{
    document* docs;
    int nterms;
    int ndocs;
} hdplda_corpus;


typedef struct hdplda_model
{
    int k; // max # topics
    int t; // max # topics in each doc
    int D; // # docs
    int n; // # terms
    double alpha;
    double eta;
    double omega;
    double** Elogbeta; // Psi(lambda[k][n]) - Psi(\sum_{n'}lambda[k][n'])
} hdplda_model;


typedef struct hdplda_var
{
	double** lambda; // wrd prob
	double* sumlambda; // sum_{n=1}{N}lambda[k][n]
	double** phi; // phi[nmax][T] for each doc
	double** sumphi; //sumphi[T][2] doc-level; sumphi[t][0]=\sum_{n}phi[n][t], sumphi[i][1]=\sum_n\sum_{l=t+1}^{T}phi[n][l]
	double* oldphi; //oldphi[T]

	double** gamma; // gamma[T][2] for each doc
	double** Psigamma; //Psi(gamma[t][2])
	double* sumgamma; // \sum_{l=1}^{2}gamma[t][l]
	double* Psisumgamma; //Psi(sumgamma[t]);

	double** xi; // xi[T][K] for each doc

	double** sumxi; //sumxi[K][2]doc-level; sumxi[k][0]=\sum_{t}xi[t][k], sumxi[k][1]=\sum_{t\sum_{l=k+1}xi[t][l]
	double* a; // a[K] corpus-level
	double* b; // b[K] corpus-level
	double* psia;
	double* psib;
	double* psiab;

	double* phiss1; //phiss[T]
	double* xiss1; //xiss1[k]
	double** xiss2; //xiss[T][K]

} hdplda_var;

typedef struct hdplda_ss
{
    double* a; //a[K]; a[k] = \sum_{d}\sum_{t}xi[d][t][k]
    double* b; //b[K]; b[k] = \sum_{d}\sum_{t}\sum_{l=k+1}^{K}xi[d][t][l]
    double** lambda; //lambdass[k][n]
    double* sumlambda; //sumlambda[k]
} hdplda_ss;


#endif /* HDPLDA_H_ */
