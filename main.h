
#ifndef MAIN_H_
#define MAIN_H_

#include "hdplda.h"
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>


const gsl_rng_type * T;
gsl_rng * r;
double CONVERGED;
int MAXITER;
int NUMINIT;
int BatchSize;
double Kappa;

hdplda_ss * new_hdplda_ss(hdplda_model* model);
hdplda_model* new_hdplda_model(int kmax, int tmax, int nterms, int ndocs,
		double alpha, double eta, double omega);
hdplda_var * new_hdplda_var(hdplda_model* model, int nmax);

void BatchVB(char* dataset, int kmax, int tmax, char* start, char* dir,
		double alpha, double eta, double omega, char* model_name);
void StochVB(char* dataset, char* test_data, int kmax, int tmax, char* start, char* dir,
		double alpha, double eta, double omega, char* model_name);
void doc_init_vars(hdplda_corpus* corpus, hdplda_model* model,
		hdplda_var* var, hdplda_ss* ss, int d);
void hdp_lda_est(hdplda_corpus* corpus, hdplda_model* model,
		hdplda_var* var, double** theta, int nmax);

int main(int argc, char* argv[]);
void write_hdplda_model(hdplda_model * model, hdplda_var* var, char * root,hdplda_corpus * corpus, double** theta);
void corpus_initialize_model(hdplda_model* model, hdplda_corpus* corpus, hdplda_ss* ss, hdplda_var* var);

hdplda_corpus* read_data(const char* data_filename);

double doc_inference(hdplda_corpus* corpus, hdplda_model* model,
		hdplda_ss* ss, hdplda_var* var, int d, int test);

hdplda_model* load_model(char* model_root, int ndocs);
void write_pred_time(hdplda_corpus* corpus, char * filename);
void read_time(hdplda_corpus* corpus, char * filename);

//void corpus_initialize_model(hdplda_var* alpha, hdplda_model* model, hdplda_corpus* c);//, gsl_rng * r);
void test(char* dataset, char* model_name, char* dir);
void random_initialize_model(hdplda_model * model, hdplda_corpus* corpus, hdplda_ss* ss, hdplda_var* var);
void write_word_assignment(hdplda_corpus* c,char * filename, hdplda_model* model);
double log_sum(double log_a, double log_b);

#endif /* MAIN_H_ */
