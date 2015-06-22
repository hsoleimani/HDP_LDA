#include "main.h"


int main(int argc, char* argv[])
{

	char task[40];
	char dir[400];
	char corpus_file[400];
	char test_corpus_file[400];
	char model_name[400];
	char init[400];
	int tmax, kmax;
	double alpha, eta, omega;
	long int seed;


	seed = atoi(argv[1]);

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	gsl_rng_set (r, seed);

	printf("SEED = %ld\n", seed);

	MAXITER = 1000;
	CONVERGED = 5e-5;
	NUMINIT = 10;
	Kappa = 0.9; //Forgetting factor
	BatchSize = 50; //Batch Size

	strcpy(task,argv[2]);
	strcpy(corpus_file,argv[3]);

    if (argc > 1)
    {
        if (strcmp(task, "BatchVB")==0)
        {
        	kmax = atoi(argv[4]);
        	tmax = atoi(argv[5]);
        	alpha = atof(argv[6]);
        	eta = atof(argv[7]);
        	omega = atof(argv[8]);
			strcpy(init,argv[9]);
			strcpy(dir,argv[10]);
			BatchVB(corpus_file, kmax, tmax, init, dir, alpha, eta, omega, model_name);

			gsl_rng_free (r);
            return(0);
        }
        if (strcmp(task, "StochVB")==0)
        {
        	strcpy(test_corpus_file,argv[4]);
        	kmax = atoi(argv[5]);
        	tmax = atoi(argv[6]);
        	alpha = atof(argv[7]);
        	eta = atof(argv[8]);
        	omega = atof(argv[9]);
			strcpy(init,argv[10]);
			strcpy(dir,argv[11]);
			StochVB(corpus_file, test_corpus_file, kmax, tmax, init, dir, alpha, eta, omega, model_name);
			gsl_rng_free (r);
            return(0);
        }
        if (strcmp(task, "test")==0)
        {
			strcpy(model_name,argv[4]);
			strcpy(dir,argv[5]);
			test(corpus_file, model_name, dir);

			gsl_rng_free (r);
            return(0);
        }
    }
    return(0);
}


void StochVB(char* dataset, char* test_dataset, int kmax, int tmax, char* start, char* dir,
		double alpha, double eta, double omega, char* model_name)
{
    FILE* lhood_fptr;
    FILE* fp;
    char string[100];
    char filename[100];
    int iteration, nmax;
	double lhood, prev_lhood, conv, doclkh;
	double rho;
	int d, n;
    hdplda_corpus* corpus;
    hdplda_corpus* test_corpus;
    hdplda_model *model = NULL;
    hdplda_ss* ss = NULL;
    hdplda_var* var = NULL;
    time_t t1,t2;
    int k, s;
    double temp;
    double** theta; //Expected posterior topic proportions
    double** test_theta; //Expected posterior topic proportions
    //double* Epi; //expected posterior sticks for each doc

    corpus = read_data(dataset);
    test_corpus = read_data(test_dataset);

    // nmax
    nmax = 0;
    theta = malloc(sizeof(double*)*corpus->ndocs);
    for (d = 0; d < corpus->ndocs; d++){
    	if (corpus->docs[d].length > nmax)
    		nmax = corpus->docs[d].length;
    	theta[d] = malloc(sizeof(double)*kmax);
    	for (k = 0; k < kmax; k++){
    		theta[d][k] = 0.0;
    	}
    }
    test_theta = malloc(sizeof(double*)*test_corpus->ndocs);
    for (d = 0; d < test_corpus->ndocs; d++){
    	if (test_corpus->docs[d].length > nmax)
    		nmax = test_corpus->docs[d].length;
    	test_theta[d] = malloc(sizeof(double)*kmax);
    	for (k = 0; k < kmax; k++){
    		test_theta[d][k] = 0.0;
    	}
    }
    /*Epi = malloc(sizeof(double)*tmax);
    for (t = 0; t < tmax; t++){
    	Epi[t] = 0.0;
    }*/


    mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

    // set up the log likelihood log file

    sprintf(string, "%s/likelihood.dat", dir);
    lhood_fptr = fopen(string, "w");

    if (strcmp(start, "random")==0){
    	printf("random\n");
    	model = new_hdplda_model(kmax, tmax, corpus->nterms, corpus->ndocs, alpha, eta, omega);
		ss = new_hdplda_ss(model);
		var =  new_hdplda_var(model, nmax);
		random_initialize_model(model, corpus, ss, var);
    }

	for (k = 0; k < model->k; k++){
		var->a[k] = 1.0;
		var->b[k] = model->omega;
		var->psia[k] = gsl_sf_psi(var->a[k]);
		var->psib[k] = gsl_sf_psi(var->b[k]);
		var->psiab[k] = gsl_sf_psi(var->a[k] + var->b[k]);
	}

	//variational params for fitting an approximate LDA to estimate topic proportions
	double* ldaalpha;
	double** ldaphi;
	double* ldaoldphi;
	ldaphi = malloc(sizeof(double*)*nmax);
	for (n = 0; n < nmax; n++){
		ldaphi[n] = malloc(sizeof(double)*model->k);
		for (k = 0; k < model->k; k++){
			ldaphi[n][k] = 0.0;
		}
	}
	ldaalpha = malloc(sizeof(double)*model->k);
	ldaoldphi = malloc(sizeof(double)*model->k);

    iteration = 0;
    sprintf(filename, "%s/%03d", dir,iteration);
    printf("%s\n",filename);
	write_hdplda_model(model, var, filename, corpus, theta);

    time(&t1);
	prev_lhood = -1e100;
	conv = 1e10;

	do{

		lhood = 0.0;

		var->xiss1[0] = 0.0;
		for (k = 1; k < model->k; k++){
			var->xiss1[k] = var->xiss1[k-1]+var->psib[k-1]-var->psiab[k-1];
		}

		for (s = 0; s < BatchSize; s++){

			//choose a doc
			d = floor(gsl_rng_uniform(r) * corpus->ndocs);

			doc_init_vars(corpus, model, var, ss, d);

			doclkh = doc_inference(corpus, model, ss, var, d, 0);

			lhood += doclkh;

			//compute expected topic proportions
			/*prod = 1.0;
			normsum = 0.0;
			for (t = 0; t < model->t; t++){
				Epi[t] = prod*var->gamma[t][0]/var->sumgamma[t];
				normsum += Epi[t];
				prod *= (1.0 - var->gamma[t][0]/var->sumgamma[t]);
			}
			for (t = 0; t < model->t; t++){
				Epi[t] /= normsum;
			}
			for (k = 0; k < model->k; k++){
				theta[d][k] = 0.0;
				for (t = 0; t < model->t; t++){
					theta[d][k] += Epi[t]*var->xi[t][k];
				}
			}*/

		}

		rho = pow(1.0/(iteration + 1.0), Kappa); //TAU = 1.0;

		//update lambda, a, and b
		for (k = 0; k < model->k; k++){

			//update lambda
			var->sumlambda[k] = (1-rho)*var->sumlambda[k] + rho*((double)model->n*model->eta)
									+ rho*((double)corpus->ndocs*ss->sumlambda[k])/((double)BatchSize);

			for (n = 0; n < model->n; n++){

				var->lambda[k][n] = (1-rho)*var->lambda[k][n] + rho*(model->eta)
						+ rho*((double)corpus->ndocs*ss->lambda[k][n])/((double)BatchSize);

				ss->lambda[k][n] = 0.0;

				model->Elogbeta[k][n] = (gsl_sf_psi(var->lambda[k][n])-gsl_sf_psi(var->sumlambda[k]));
				//model->expElogbeta[j][n] = exp(model->Elogbeta[j][n]);
			}
			ss->sumlambda[k] = 0.0;

			//update a and b
			var->a[k] = var->a[k]*(1-rho) + rho*(1.0 + ((double)corpus->ndocs)*ss->a[k]/((double)BatchSize));
			var->b[k] = var->b[k]*(1-rho)
					+ rho*(model->omega + ((double)corpus->ndocs)*ss->b[k]/((double)BatchSize));
			var->psia[k] = gsl_sf_psi(var->a[k]);
			var->psib[k] = gsl_sf_psi(var->b[k]);
			var->psiab[k] = gsl_sf_psi(var->a[k]+var->b[k]);

			ss->a[k] = 0.0;
			ss->b[k] = 0.0;
		}


		if ((iteration%5) == 0){
			printf("***** VB ITERATION %d *****\n", iteration);
			// compute lkh on test set

			//alternative method for estimating tpc prop for each doc
			hdp_lda_est(test_corpus, model, var, test_theta, ldaphi, ldaoldphi, ldaalpha);


			lhood = 0.0;
			for (d = 0; d < test_corpus->ndocs; d++){
				temp = 0.0;
				for (k = 0; k < model->k; k++){
					temp += test_theta[d][k];
				}
				for (k = 0; k < model->k; k++){
					test_theta[d][k] /= temp;
				}
				doclkh = 0.0;
				for (n = 0; n < test_corpus->docs[d].length; n++){
					temp = 0.0;
					for (k = 0; k < model->k; k++){
						temp += test_theta[d][k]*exp(model->Elogbeta[k][test_corpus->docs[d].words[n]]);
					}
					doclkh += (double) test_corpus->docs[d].counts[n]*log(temp);
				}
				lhood += doclkh;
			}

			conv = fabs(prev_lhood - lhood)/fabs(prev_lhood);
			prev_lhood = lhood;

			time(&t2);

		    // write theta
			sprintf(filename, "%s/%03d.theta", dir,1);
			fp = fopen(filename, "w");
			for (d = 0; d < test_corpus->ndocs; d++){
				for (k = 0; k < model->k; k++){
					fprintf(fp, "%5.10lf ", test_theta[d][k]);
				}
				fprintf(fp, "\n");
			}
			fclose(fp);
			sprintf(filename, "%s/%03d", dir,1);
			write_hdplda_model(model, var, filename, test_corpus, test_theta);

			fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld \n",iteration, lhood, conv, (t2-t1));
			fflush(lhood_fptr);
		}

		iteration ++;

	}while((iteration < 1e5) || ((iteration < MAXITER) && (conv > CONVERGED)));
	fclose(lhood_fptr);

	// final estimate on the test set
	hdp_lda_est(test_corpus, model, var, test_theta, ldaphi, ldaoldphi, ldaalpha);
	sprintf(filename, "%s/testfinal.theta", dir);
	fp = fopen(filename, "w");
	for (d = 0; d < test_corpus->ndocs; d++){
		for (k = 0; k < model->k; k++){
			fprintf(fp, "%5.10lf ", test_theta[d][k]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	//final estimate on training set
	hdp_lda_est(corpus, model, var, theta, ldaphi, ldaoldphi, ldaalpha);

    sprintf(filename, "%s/final", dir);
    write_hdplda_model(model, var, filename, corpus, theta);

}

void BatchVB(char* dataset, int kmax, int tmax, char* start, char* dir,
		double alpha, double eta, double omega, char* model_name)
{
    FILE* lhood_fptr;
    //FILE* fp;
    char string[100];
    char filename[100];
    int iteration, nmax;
	double lhood, prev_lhood, conv, doclkh;
	int d, n;
    hdplda_corpus* corpus;
    hdplda_model *model = NULL;
    hdplda_ss* ss = NULL;
    hdplda_var* var = NULL;
    time_t t1,t2;
    int k;
    double temp, temp2;
    double** theta; //Expected posterior topic proportions
    //double* Epi; //expected posterior sticks for each doc
    //FILE* fileptr;
    //float x;
    //double y;

    corpus = read_data(dataset);

    // nmax
    nmax = 0;
    theta = malloc(sizeof(double*)*corpus->ndocs);
    for (d = 0; d < corpus->ndocs; d++){
    	if (corpus->docs[d].length > nmax)
    		nmax = corpus->docs[d].length;
    	theta[d] = malloc(sizeof(double)*kmax);
    	for (k = 0; k < kmax; k++){
    		theta[d][k] = 0.0;
    	}
    }
    /*Epi = malloc(sizeof(double)*tmax);
    for (t = 0; t < tmax; t++){
    	Epi[t] = 0.0;
    }*/

    mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

    // set up the log likelihood log file

    sprintf(string, "%s/likelihood.dat", dir);
    lhood_fptr = fopen(string, "w");

    if (strcmp(start, "random")==0){
    	printf("random\n");
    	model = new_hdplda_model(kmax, tmax, corpus->nterms, corpus->ndocs, alpha, eta, omega);
		ss = new_hdplda_ss(model);
		var =  new_hdplda_var(model, nmax);
		random_initialize_model(model, corpus, ss, var);
    }

	for (k = 0; k < model->k; k++){
		var->a[k] = 1.0;
		var->b[k] = model->omega;
		var->psia[k] = gsl_sf_psi(var->a[k]);
		var->psib[k] = gsl_sf_psi(var->b[k]);
		var->psiab[k] = gsl_sf_psi(var->a[k] + var->b[k]);
	}


    iteration = 0;
    sprintf(filename, "%s/%03d", dir,iteration);
    printf("%s\n",filename);
	write_hdplda_model(model, var, filename, corpus, theta);

    time(&t1);
	prev_lhood = -1e100;

	do{

		printf("***** VB ITERATION %d *****\n", iteration);
		lhood = 0.0;

		var->xiss1[0] = 0.0;
		for (k = 1; k < model->k; k++){
			var->xiss1[k] = var->xiss1[k-1]+var->psib[k-1]-var->psiab[k-1];
		}

		for (d = 0; d < corpus->ndocs; d++){

			doc_init_vars(corpus, model, var, ss, d);

			doclkh = doc_inference(corpus, model, ss, var, d, 0);

			lhood += doclkh;

			//compute expected topic proportions
			/*prod = 1.0;
			normsum = 0.0;
			for (t = 0; t < model->t; t++){
				Epi[t] = prod*var->gamma[t][0]/var->sumgamma[t];
				normsum += Epi[t];
				prod *= (1.0 - var->gamma[t][0]/var->sumgamma[t]);
			}
			for (t = 0; t < model->t; t++){
				Epi[t] /= normsum;
			}
			for (k = 0; k < model->k; k++){
				theta[d][k] = 0.0;
				for (t = 0; t < model->t; t++){
					theta[d][k] += Epi[t]*var->xi[t][k];
				}
			}*/

		}

		//update mu and lhood
		for (k = 0; k < model->k; k++){

			//update lambda
			var->sumlambda[k] = ((double)model->n)*model->eta + ss->sumlambda[k];
			temp = gsl_sf_psi(var->sumlambda[k]);
			for (n = 0; n < model->n; n++){
				var->lambda[k][n] = model->eta + ss->lambda[k][n];
				temp2 = gsl_sf_psi(var->lambda[k][n]);
				model->Elogbeta[k][n] = temp2 - temp;

				ss->lambda[k][n] = 0.0;
				lhood += lgamma(var->lambda[k][n]);
			}
			lhood -= lgamma(var->sumlambda[k]);
			ss->sumlambda[k] = 0.0;

			//update a and b
			var->a[k] = 1.0 + ss->a[k];
			var->b[k] = model->omega + ss->b[k];
			var->psia[k] = gsl_sf_psi(var->a[k]);
			var->psib[k] = gsl_sf_psi(var->b[k]);
			var->psiab[k] = gsl_sf_psi(var->a[k]+var->b[k]);

			//lhood += (model->omega+ss->b[k]-var->b[k])*(var->psib[k]-var->psiab[k])
			//		+ (1.0 + ss->a[k] - var->a[k])*(var->psia[k]-var->psiab[k]); //This term is zero

			lhood += lgamma(var->a[k])+lgamma(var->b[k])-lgamma(var->a[k]+var->b[k]);

			ss->a[k] = 0.0;
			ss->b[k] = 0.0;
		}


		conv = fabs(prev_lhood - lhood)/fabs(prev_lhood);

		if (prev_lhood > lhood){
			printf("Oops, likelihood is decreasing! \n");
		}
		time(&t2);
		prev_lhood = lhood;

		//sprintf(filename, "%s/%03d", dir,1);
		//write_hdplda_model(model, var, filename, corpus, theta);

		printf("likelihood = %5.5e, Conv = %5.5e, Time = %5ld\n", lhood, conv, (int)t2-t1);

		fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld \n",iteration, lhood, conv, (int)t2-t1);
		fflush(lhood_fptr);
		iteration ++;

	}while((iteration < MAXITER) && (conv > CONVERGED));
	fclose(lhood_fptr);

	//variational params for fitting an approximate LDA to estimate topic proportions
	double* ldaalpha;
	double** ldaphi;
	double* ldaoldphi;
	ldaphi = malloc(sizeof(double*)*nmax);
	for (n = 0; n < nmax; n++){
		ldaphi[n] = malloc(sizeof(double)*model->k);
		for (k = 0; k < model->k; k++){
			ldaphi[n][k] = 0.0;
		}
	}
	ldaalpha = malloc(sizeof(double)*model->k);
	ldaoldphi = malloc(sizeof(double)*model->k);

	//alternative method for estimating tpc prop for each doc
	hdp_lda_est(corpus, model, var, theta, ldaphi, ldaoldphi, ldaalpha);

    sprintf(filename, "%s/final", dir);

    write_hdplda_model(model, var, filename, corpus, theta);

}


void doc_init_vars(hdplda_corpus* corpus, hdplda_model* model,
		hdplda_var* var, hdplda_ss* ss, int d){

	int t, k, n, w;
	double maxval, c, normsum, sumphitotal;


	// init vars for this doc
		for (t = 0; t < model->t; t++){
			var->sumphi[t][0] = 0.0;
			var->sumphi[t][1] = 0.0;
			if (t > 0){
				for (k = 0; k < model->k; k++){
					var->xi[t][k] = var->xi[0][k];
					var->xiss2[t][k] = 0.0;
				}
			}else{
				maxval = -1e50;
				for (k = 0; k < model->k; k++){
					var->xi[t][k] = var->psia[k] - var->psiab[k] + var->xiss1[k];
					for (n = 0; n < corpus->docs[d].length; n++){
						w = corpus->docs[d].words[n];
						c = (double) corpus->docs[d].counts[n];
						var->xi[t][k] += c*model->Elogbeta[k][w];
					}
					if (var->xi[t][k] > maxval)
						maxval = var->xi[t][k];
				}
				normsum = 0.0;
				for (k = 0; k < model->k; k++){
					var->xi[t][k] = exp(var->xi[t][k] - maxval);
					normsum += var->xi[t][k];
				}
				for (k = 0; k < model->k; k++){
					var->xi[t][k] /= normsum;
					var->xiss2[t][k] = 0.0;
				}
			}
		}
		sumphitotal = 0.0;
		for (n = 0; n < corpus->docs[d].length; n++){
			w = corpus->docs[d].words[n];
			c = (double) corpus->docs[d].counts[n];
			maxval = -1e50;
			for (t = 0; t < model->t; t++){
				var->phi[n][t] = 0.0;
				for (k = 0; k < model->k; k++){
					var->phi[n][t] += c*var->xi[t][k]*model->Elogbeta[k][w];
				}
				if (var->phi[n][t] > maxval)
					maxval = var->phi[n][t];
			}
			normsum = 0.0;
			for (t = 0; t < model->t; t++){
				var->phi[n][t] = exp(var->phi[n][t] - maxval);
				normsum += var->phi[n][t];
			}
			for (t = 0; t < model->t; t++){
				var->phi[n][t] /= normsum;
				var->sumphi[t][0] += c*var->phi[n][t];
				sumphitotal += c*var->phi[n][t];
				for (k = 0; k < model->k; k++){
					if (n == 0) var->xiss2[t][k] = 0.0;
					var->xiss2[t][k] += c*var->phi[n][t]*model->Elogbeta[k][w];
				}
			}
		}
		for (t = model->t-1; t >= 0; t--){
			if (t < model->t-1) var->sumphi[t][1] = var->sumphi[t+1][1] + var->sumphi[t+1][0];
			else var->sumphi[t][1] = 0.0;
			var->gamma[t][0] = 1 + var->sumphi[t][0];
			var->gamma[t][1] = model->alpha + var->sumphi[t][1];
			var->Psigamma[t][0] = gsl_sf_psi(var->gamma[t][0]);
			var->Psigamma[t][1] = gsl_sf_psi(var->gamma[t][1]);
			var->sumgamma[t] = var->gamma[t][0] + var->gamma[t][1];
			var->Psisumgamma[t] = gsl_sf_psi(var->sumgamma[t]);
		}
		// update phiss
		var->phiss1[0] = 0.0;
		for (t = 1; t < model->t; t++){
			var->phiss1[t] = var->phiss1[t-1] + var->Psigamma[t-1][1] - var->Psisumgamma[t-1];
		}

}

void hdp_lda_est(hdplda_corpus* corpus, hdplda_model* model,
		hdplda_var* var, double** theta, double** phi, double* oldphi, double* alpha){

	int k, d, n, iter, w;
	double lkh, conv, prev_lkh, left, temp, normsum, c, sumgamma;


	k = 0;
	temp = var->a[k]/(var->a[k]+var->b[k]);
	alpha[k] = temp;
	left = 1.0 - temp;
	for (k = 1; k < model->k; k++){
		temp = var->a[k]/(var->a[k]+var->b[k]);
		alpha[k] = temp*left;
		left *= 1.0 - temp;
		//left -= temp;

		//alpha[k] *= model->alpha;
	}

	for (d = 0; d < corpus->ndocs; d++){

		//init
		sumgamma = 0.0;
		for (k = 0; k < model->k; k++){
			theta[d][k] = alpha[k] + ((double)corpus->docs[d].total)/((double)model->k);
			sumgamma += theta[d][k];
		}
		for (n = 0; n < corpus->docs[d].length; n++){
			for (k = 0; k < model->k; k++){
				phi[n][k] = 1.0/((double)model->k);
			}
			/*w = corpus->docs[d].words[n];
			c = (double) corpus->docs[d].counts[n];
			normsum = 0.0;
			for (k = 0; k < model->k; k++){
				phi[n][k] = model->Elogbeta[k][w];
				if (k > 0)
					normsum = log_sum(normsum, phi[n][k]);
				else
					normsum = phi[n][k];
			}
			for (k = 0; k < model->k; k++){
				phi[n][k] = exp(phi[n][k] - normsum);
				theta[d][k] += c*phi[n][k];
				sumgamma += c*phi[n][k];
			}*/
		}

		iter = 0;
		prev_lkh = -1e50;

		do{

			lkh = 0.0;
			for (n = 0; n < corpus->docs[d].length; n++){
				w = corpus->docs[d].words[n];
				c = (double) corpus->docs[d].counts[n];

				normsum = 0.0;
				for (k = 0; k < model->k; k++){
					oldphi[k] = phi[n][k];

					phi[n][k] = gsl_sf_psi(theta[d][k]) + model->Elogbeta[k][w];
					if (k > 0)
						normsum = log_sum(normsum, phi[n][k]);
					else
						normsum = phi[n][k];
				}
				for (k = 0; k < model->k; k++){

					phi[n][k] = exp(phi[n][k] - normsum);

					temp = c*(phi[n][k] - oldphi[k]);
					theta[d][k] += temp;
					sumgamma += temp;

					if (phi[n][k] > 0){
						lkh += c*phi[n][k]*(model->Elogbeta[k][w]-log(phi[n][k]));
					}
				}
			}
			lkh -= lgamma(sumgamma);
			for (k = 0; k < model->k; k++){
				lkh += lgamma(theta[d][k]);
			}

			conv = fabs(prev_lkh - lkh)/fabs(prev_lkh);
			prev_lkh = lkh;
			iter ++;

		}while((iter < MAXITER) && (conv > CONVERGED));


	}

}


double doc_inference(hdplda_corpus* corpus, hdplda_model* model,
		hdplda_ss* ss, hdplda_var* var, int d, int test){

	int n, variter, w, t, k;
	double c, normsum, maxval;
	double varlkh, prev_varlkh, conv;

	prev_varlkh = -1e100;
	conv = 0.0;
	variter = 0;
	do{
		varlkh = 0.0;

		// update xi
		for (t = 0; t < model->t; t++){

			normsum = 0.0;
			maxval = -1e100;
			for (k = 0; k < model->k; k++){

				var->xi[t][k] = var->psia[k] - var->psiab[k] + var->xiss1[k] + var->xiss2[t][k];

				if (var->xi[t][k] > maxval) maxval = var->xi[t][k];
				//if (k > 0)	normsum = log_sum(normsum, var->xi[t][k]);
				//else	normsum = var->xi[t][k];

			}
			for (k = 0; k < model->k; k++){
				var->xi[t][k] = exp(var->xi[t][k] - maxval);
				normsum += var->xi[t][k];
			}
			for (k = 0; k < model->k; k++){
				var->xi[t][k] /= normsum;
				//var->xi[t][k] = exp(var->xi[t][k] - normsum);
				// update sums
				if (t == 0) var->sumxi[k][0] = 0.0;
				var->sumxi[k][0] += var->xi[t][k];
			}
		}
		var->sumxi[model->k-1][1] = 0.0;
		for (k = model->k-2; k >= 0; k--){
			var->sumxi[k][1] = var->sumxi[k+1][1] + var->sumxi[k+1][0];
		}

		// update phi
		for (n = 0; n < corpus->docs[d].length; n++){
			w = corpus->docs[d].words[n];
			c = (double) corpus->docs[d].counts[n];

			maxval = -1e100;
			normsum = 0.0;
			for (t = 0; t < model->t; t++){

				var->phi[n][t] = var->Psigamma[t][0]-var->Psisumgamma[t]+ var->phiss1[t];
				for (k = 0; k < model->k; k++){
					var->phi[n][t] += var->xi[t][k]*model->Elogbeta[k][w];
				}
				if (var->phi[n][t] > maxval)
					maxval = var->phi[n][t];
				//if (t > 0)	normsum = log_sum(normsum, var->phi[n][t]);
				//else normsum = var->phi[n][t];
			}
			for (t = 0; t < model->t; t++){
				var->phi[n][t] = exp(var->phi[n][t] - maxval);
				normsum += var->phi[n][t];
			}
			for (t = 0; t < model->t; t++){
				//var->phi[n][t] = exp(var->phi[n][t] - normsum);
				var->phi[n][t] /= normsum;
				if (n == 0) var->sumphi[t][0] = 0.0;
				var->sumphi[t][0] += c*var->phi[n][t];
				for (k = 0; k < model->k; k++){
					if (n == 0) var->xiss2[t][k] = 0.0;
					var->xiss2[t][k] += c*var->phi[n][t]*model->Elogbeta[k][w];
				}
				if (var->phi[n][t] > 0) varlkh -= c*var->phi[n][t]*log(var->phi[n][t]);
			}
		}
		var->sumphi[model->t-1][1] = 0.0;
		for (t = model->t-2; t >= 0; t--){
			var->sumphi[t][1] = var->sumphi[t+1][1] + var->sumphi[t+1][0];
		}
		var->phiss1[0] = 0.0;
		for (t = 0; t < model->t; t++){
			var->gamma[t][0] = 1 + var->sumphi[t][0];
			var->gamma[t][1] = model->alpha + var->sumphi[t][1];
			var->Psigamma[t][0] = gsl_sf_psi(var->gamma[t][0]);
			var->Psigamma[t][1] = gsl_sf_psi(var->gamma[t][1]);
			var->sumgamma[t] = var->gamma[t][0] + var->gamma[t][1];
			var->Psisumgamma[t] = gsl_sf_psi(var->sumgamma[t]);

			if (t > 0)
				var->phiss1[t] = var->phiss1[t-1] + var->Psigamma[t-1][1] - var->Psisumgamma[t-1];
		}

		// compute likelihood
		for (k = 0; k < model->k; k++){
			varlkh += (var->psia[k]-var->psiab[k])*var->sumxi[k][0] +
					(var->psib[k]-var->psiab[k])*var->sumxi[k][1];
			for (t = 0; t < model->t; t++){
				if (var->xi[t][k] > 0)
					varlkh += var->xi[t][k]*(var->xiss2[t][k] - log(var->xi[t][k]));
			}
			if (isnan(varlkh)){
				printf("oops\n");
				assert(!isnan(varlkh));
			}
		}
		for (t = 0; t < model->t; t++){
			//varlkh += (var->Psigamma[t][0] - var->Psisumgamma[t])*(var->sumphi[t][0]-var->gamma[t][0]+1)
			//		+ (var->Psigamma[t][1] - var->Psisumgamma[t])*(var->sumphi[t][1]-var->gamma[t][1]+model->alpha);
			varlkh += lgamma(var->gamma[t][0])+lgamma(var->gamma[t][1])-lgamma(var->sumgamma[t]);
			if (isnan(varlkh)){
				printf("oops\n");
				assert(!isnan(varlkh));
			}
		}

		conv = fabs(prev_varlkh - varlkh)/fabs(prev_varlkh);
		prev_varlkh = varlkh;
		variter ++;

	}while((variter < MAXITER) && (conv > CONVERGED));

	if (test == 0){
		for (n = 0; n < corpus->docs[d].length; n++){
			w = corpus->docs[d].words[n];
			c = (double) corpus->docs[d].counts[n];
			for (k = 0; k < model->k; k++){

				for (t = 0; t < model->t; t++){
					ss->lambda[k][w] += c*var->phi[n][t]*var->xi[t][k];
					ss->sumlambda[k] += c*var->phi[n][t]*var->xi[t][k];

					if (n == 0){
						varlkh -= var->xi[t][k]*var->xiss2[t][k];
						if (isnan(varlkh)){
							printf("oops\n");
							assert(!isnan(varlkh));
						}
					}
				}

			}
		}
		for (k = 0; k < model->k; k++){
			ss->a[k] += var->sumxi[k][0];
			ss->b[k] += var->sumxi[k][1];
			varlkh -= (var->psia[k]-var->psiab[k])*var->sumxi[k][0] +
					(var->psib[k]-var->psiab[k])*var->sumxi[k][1];
		}
	}
	return(varlkh);

}


void test(char* dataset, char* model_name, char* dir)
{

	FILE* lhood_fptr;
	FILE* fp;
	char string[100];
	char filename[100];
	int iteration;
	int d, n, k, doclkh, nmax;
	double lhood, temp;
	double x, y;

	hdplda_corpus* corpus;
	hdplda_model *model = NULL;
	hdplda_ss* ss = NULL;
	hdplda_var* var = NULL;
	time_t t1,t2;

    double** theta; //Expected posterior topic proportions
    //double* Epi; //expected posterior sticks for each doc

    corpus = read_data(dataset);

	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

	// set up the log likelihood log file
	sprintf(string, "%s/test-lhood.dat", dir);
	lhood_fptr = fopen(string, "w");

	model = load_model(model_name, corpus->ndocs);
    // nmax, theta
    nmax = 0;
    theta = malloc(sizeof(double*)*corpus->ndocs);
    for (d = 0; d < corpus->ndocs; d++){
    	if (corpus->docs[d].length > nmax)
    		nmax = corpus->docs[d].length;
    	theta[d] = malloc(sizeof(double)*model->k);
    	for (k = 0; k < model->k; k++){
    		theta[d][k] = 0.0;
    	}
    }
    /*Epi = malloc(sizeof(double)*model->t);
    for (t = 0; t < model->t; t++){
    	Epi[t] = 0.0;
    }*/

	ss = new_hdplda_ss(model);
	var =  new_hdplda_var(model, nmax);

	sprintf(filename, "%s.ab", model_name);
    printf("loading %s\n", filename);
    fp = fopen(filename, "r");
	for (k = 0; k < model->k; k++){
		fscanf(fp, "%lf %lf", &x, &y);
		var->a[k] = x;
		var->b[k] = y;
		var->psia[k] = gsl_sf_psi(var->a[k]);
		var->psib[k] = gsl_sf_psi(var->b[k]);
		var->psiab[k] = gsl_sf_psi(var->a[k]+var->b[k]);
	}
    fclose(fp);

	//*************************************
	for (k = 0; k < model->k; k++){
		var->sumlambda[k] = 0.0;
		for (n = 0; n < model->n; n++){
			var->lambda[k][n] = model->Elogbeta[k][n];
			var->sumlambda[k] += var->lambda[k][n];
		}
		temp = gsl_sf_psi(var->sumlambda[k]);
		for (n = 0; n < model->n; n++){
			model->Elogbeta[k][n] = gsl_sf_psi(var->lambda[k][n]) - temp;
		}
	}


    iteration = 0;
    /*sprintf(filename, "%s/test%03d", dir,iteration);
    printf("%s\n",filename);
	write_hdplda_model(model, var, filename, corpus, theta);*/

    time(&t1);

	lhood = 0.0;

	var->xiss1[0] = 0.0;
	for (k = 1; k < model->k; k++){
		var->xiss1[k] = var->xiss1[k-1]+var->psib[k-1]-var->psiab[k-1];
	}

	for (d = 0; d < corpus->ndocs; d++){

		doc_init_vars(corpus, model, var, ss, d);

		doclkh = doc_inference(corpus, model, ss, var, d, 1);

		lhood += doclkh;

		//compute expected topic proportions
		/*prod = 1.0;
		normsum = 0.0;
		for (t = 0; t < model->t; t++){
			Epi[t] = prod*var->gamma[t][0]/var->sumgamma[t];
			normsum += Epi[t];
			prod *= (1.0 - var->gamma[t][0]/var->sumgamma[t]);
		}
		for (t = 0; t < model->t; t++){
			Epi[t] /= normsum;
		}
		for (k = 0; k < model->k; k++){
			theta[d][k] = 0.0;
			for (t = 0; t < model->t; t++){
				theta[d][k] += Epi[t]*var->xi[t][k];
			}
		}*/

	}


	time(&t2);

	fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld \n",iteration, lhood, 0.0, (int)t2-t1);
	fflush(lhood_fptr);
	fclose(lhood_fptr);
	//*************************************

	//variational params for fitting an approximate LDA to estimate topic proportions
	double* ldaalpha;
	double** ldaphi;
	double* ldaoldphi;
	ldaphi = malloc(sizeof(double*)*nmax);
	for (n = 0; n < nmax; n++){
		ldaphi[n] = malloc(sizeof(double)*model->k);
		for (k = 0; k < model->k; k++){
			ldaphi[n][k] = 0.0;
		}
	}
	ldaalpha = malloc(sizeof(double)*model->k);
	ldaoldphi = malloc(sizeof(double)*model->k);

	//alternative method for estimating tpc prop for each doc
	hdp_lda_est(corpus, model, var, theta, ldaphi, ldaoldphi, ldaalpha);

	sprintf(filename, "%s/testfinal", dir);
	write_hdplda_model(model, var, filename, corpus, theta);

}

hdplda_model* new_hdplda_model(int kmax, int tmax, int nterms, int ndocs,
		double alpha, double eta, double omega)
{
	int n, k;

	hdplda_model* model = malloc(sizeof(hdplda_model));
	model->k = kmax;
	model->D = ndocs;
	model->t = tmax;
    model->n = nterms;
    model->alpha = alpha;
    model->eta = eta;
    model->omega = omega;

    model->Elogbeta = malloc(sizeof(double*)*kmax);
    for (k = 0; k < kmax; k++){
		model->Elogbeta[k] = malloc(sizeof(double)*nterms);
		for (n = 0; n < nterms; n++){
			model->Elogbeta[k][n] = 0.0;
		}
	}

    return(model);
}

hdplda_var * new_hdplda_var(hdplda_model* model, int nmax){

	int n, k, t;

	hdplda_var * var;
    var = malloc(sizeof(hdplda_var));

    var->lambda = malloc(sizeof(double*)*model->k);
    var->sumlambda = malloc(sizeof(double)*model->k);
	for (k = 0; k < model->k; k++){
		var->sumlambda[k] = 0.0;
		var->lambda[k] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			var->lambda[k][n] = 0.0;
		}
	}

	var->phi = malloc(sizeof(double*)*nmax);
	for (n = 0; n < nmax; n++){
		var->phi[n] = malloc(sizeof(double)*model->t);
		for (t = 0; t < model->t; t++){
			var->phi[n][t] = 0.0;
		}
	}

	var->sumphi = malloc(sizeof(double*)*model->t);
	var->oldphi = malloc(sizeof(double)*model->t);
	var->gamma = malloc(sizeof(double*)*model->t);
	var->Psigamma = malloc(sizeof(double*)*model->t);
	var->sumgamma = malloc(sizeof(double)*model->t);
	var->Psisumgamma = malloc(sizeof(double)*model->t);
	var->xi = malloc(sizeof(double*)*model->t);
	var->xiss2 = malloc(sizeof(double*)*model->t);
	var->phiss1 = malloc(sizeof(double)*model->t);
	for (t = 0; t < model->t; t++){
		var->sumphi[t] = malloc(sizeof(double)*2);
		var->sumphi[t][0] = 0.0;
		var->sumphi[t][1] = 0.0;
		var->oldphi[t] = 0.0;
		var->gamma[t] = malloc(sizeof(double)*2);
		var->gamma[t][0] = 0.0;
		var->gamma[t][1] = 0.0;
		var->Psigamma[t] = malloc(sizeof(double)*2);
		var->Psigamma[t][0] = 0.0;
		var->Psigamma[t][1] = 0.0;
		var->sumgamma[t] = 0.0;
		var->Psisumgamma[t] = 0.0;

		var->xi[t] = malloc(sizeof(double)*model->k);
		var->xiss2[t] = malloc(sizeof(double)*model->k);
		for (k = 0; k < model->k; k++){
			var->xi[t][k] = 0.0;
			var->xiss2[t][k] = 0.0;
		}
		var->phiss1[t] = 0.0;
	}

	var->sumxi = malloc(sizeof(double*)*model->k);
	var->a = malloc(sizeof(double)*model->k);
	var->b = malloc(sizeof(double)*model->k);
	var->psia = malloc(sizeof(double)*model->k);
	var->psib = malloc(sizeof(double)*model->k);
	var->psiab = malloc(sizeof(double)*model->k);
	var->xiss1 = malloc(sizeof(double)*model->k);
	for (k = 0; k < model->k; k++){
		var->sumxi[k] = malloc(sizeof(double)*2);
		var->sumxi[k][0] = 0.0;
		var->sumxi[k][1] = 0.0;
		var->a[k] = 0.0;
		var->b[k] = 0.0;
		var->psia[k] = 0.0;
		var->psib[k] = 0.0;
		var->psiab[k] = 0.0;
		var->xiss1[k] = 0.0;
	}

	return(var);
}


hdplda_ss * new_hdplda_ss(hdplda_model* model)
{
	int k, n;
	hdplda_ss * ss;
    ss = malloc(sizeof(hdplda_ss));

    ss->a = malloc(sizeof(double)*model->k);
    ss->b = malloc(sizeof(double)*model->k);
    ss->sumlambda = malloc(sizeof(double)*model->k);
    ss->lambda = malloc(sizeof(double*)*model->k);
    for (k = 0; k < model->k; k++){
    	ss->a[k] = 0.0;
    	ss->b[k] = 0.0;
    	ss->sumlambda[k] = 0.0;
    	ss->lambda[k] = malloc(sizeof(double)*model->n);
    	for (n = 0; n < model->n; n++){
    		ss->lambda[k][n] = 0.0;
    	}
    }

    return(ss);
}



hdplda_corpus* read_data(const char* data_filename)
{
	FILE *fileptr;
	int length, count, word, n, nd, nw;
	hdplda_corpus* c;

	printf("reading data from %s\n", data_filename);
	c = malloc(sizeof(hdplda_corpus));
	c->docs = 0;
	c->nterms = 0;
	c->ndocs = 0;
	fileptr = fopen(data_filename, "r");
	nd = 0; nw = 0;
	while ((fscanf(fileptr, "%10d", &length) != EOF)){
		c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
		c->docs[nd].length = length;
		c->docs[nd].total = 0;
		c->docs[nd].words = malloc(sizeof(int)*length);
		c->docs[nd].counts = malloc(sizeof(int)*length);
		for (n = 0; n < length; n++){
			fscanf(fileptr, "%10d:%10d", &word, &count);
			c->docs[nd].words[n] = word;
			c->docs[nd].counts[n] = count;
			c->docs[nd].total += count;
			if (word >= nw) { nw = word + 1; }
		}
		nd++;
	}
	fclose(fileptr);
	c->ndocs = nd;
	c->nterms = nw;
	printf("number of docs    : %d\n", nd);
	printf("number of terms   : %d\n", nw);
	return(c);
}

void write_hdplda_model(hdplda_model * model, hdplda_var* var, char * root,hdplda_corpus * corpus, double** theta)
{
    char filename[200];
    FILE* fileptr;
    int n, k, d;

    //beta
    sprintf(filename, "%s.beta", root);
    fileptr = fopen(filename, "w");
    for (n = 0; n < model->n; n++){
    	for (k = 0; k < model->k; k++){
    		fprintf(fileptr, "%.10lf ",var->lambda[k][n]);
    	}
    	fprintf(fileptr, "\n");
    }
    fclose(fileptr);

    //theta
	sprintf(filename, "%s.theta", root);
	fileptr = fopen(filename, "w");
	for (d = 0; d < corpus->ndocs; d++){
		for (k = 0; k < model->k; k++){
			fprintf(fileptr, "%5.10lf ", theta[d][k]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

    //a,b
    sprintf(filename, "%s.ab", root);
    fileptr = fopen(filename, "w");
   	for (k = 0; k < model->k; k++){
   		fprintf(fileptr, "%.10lf %.10lf\n",var->a[k], var->b[k]);
   	}
    fclose(fileptr);

	sprintf(filename, "%s.other", root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr,"K %d \n",model->k);
	fprintf(fileptr,"T %d \n",model->t);
	fprintf(fileptr,"num_terms %d \n",model->n);
	fprintf(fileptr,"num_docs %d \n",model->D);
	fprintf(fileptr,"alpha %lf \n",model->alpha);
	fprintf(fileptr,"omega %lf \n",model->omega);
	fprintf(fileptr,"eta %lf \n",model->eta);
	fclose(fileptr);

}

hdplda_model* load_model(char* model_root, int ndocs){

	char filename[100];
	FILE* fileptr;
	int k, n, num_terms, num_docs, kmax, tmax;
	//float x;
	double y, alpha, eta, omega;

	hdplda_model* model;
	sprintf(filename, "%s.other", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "K %d\n", &kmax);
	fscanf(fileptr, "T %d\n", &tmax);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "num_docs %d\n", &num_docs);
	fscanf(fileptr, "alpha %lf\n", &alpha);
	fscanf(fileptr, "omega %lf\n", &omega);
	fscanf(fileptr, "eta %lf\n", &eta);
	fclose(fileptr);

	model  = new_hdplda_model(kmax, tmax, num_terms, ndocs, alpha, eta, omega);
	model->n = num_terms;
	model->D = ndocs;
	model->alpha = alpha;

	sprintf(filename, "%s.beta", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
    for (n = 0; n < num_terms; n++){
		for (k = 0; k < kmax; k++){
			fscanf(fileptr, " %lf", &y);
			model->Elogbeta[k][n] = y; //this is not really correct. Need to copy this value to var->lambda later.
		}
	}
    fclose(fileptr);


    return(model);
}



void random_initialize_model(hdplda_model * model, hdplda_corpus* corpus, hdplda_ss* ss, hdplda_var* var){

	int n, j;
	double temp, mu;
	//double* beta = malloc(sizeof(double)*model->n);
	//double* nu = malloc(sizeof(double)*model->n);

	/*for (n = 0; n < model->n; n++){
		beta[n] = 0.0;
		nu[n] = 0.0;
	}*/


	for (j = 0; j < model->k; j++){
		/*for (n = 0; n < model->n; n++){
			nu[n] = model->eta;
		}*/
		var->sumlambda[j] = 0.0;

		//gsl_ran_dirichlet (r, model->n, nu, beta);
		mu = 100.0*corpus->ndocs/((double)model->k*model->n);
		for (n = 0; n < model->n; n++){
			//model->mu[j][n] = beta[n];
			var->lambda[j][n] = model->eta + gsl_ran_exponential(r, mu);
			//var->lambda[j][n] = model->eta + 1.0 + gsl_rng_uniform(r);
			var->sumlambda[j] += var->lambda[j][n];
		}
		temp = gsl_sf_psi(var->sumlambda[j]);
		for (n = 0; n < model->n; n++){
			model->Elogbeta[j][n] = gsl_sf_psi(var->lambda[j][n]) - temp;
		}
	}

  	//free(beta);
  	//free(nu);
}



/*
 * given log(a) and log(b), return log(a + b)
 *
 */

double log_sum(double log_a, double log_b)
{
  double v;

  if (log_a < log_b)
  {
      v = log_b+log(1 + exp(log_a-log_b));
  }
  else
  {
      v = log_a+log(1 + exp(log_b-log_a));
  }
  return(v);
}

