# Python code to run HDP_LDA
import numpy as np
import os

trainingfile = '' #training data set in lda-c format
testfile = '' #test data set
K =  # Corpus-Level max topics
T =  # Max topics in each doc

alpha = 1.0
eta = 0.01
omega = 1.0

seed0 = 1000000001
np.random.seed(seed0)

path = 'dir'

## run HDP_LDA on training data
seed = np.random.randint(seed0)
if 0: # Batch VI
	cmdtxt = './hdplda ' + str(seed) + ' BatchVB ' + trainingfile + ' ' + str(K) + ' ' + str(T) + ' ' + str(alpha)
else: # stochastic VI
	cmdtxt = './hdplda ' + str(seed) + ' StochVB ' + trainingfile + ' ' +testfile + ' '+ str(K) + ' ' + str(T) + ' ' + str(alpha)

cmdtxt +=  ' ' + str(eta) + ' ' + str(omega) + ' random ' + path
os.system(cmdtxt)

### read training topic proportions and word probabilities
'''theta = np.loadtxt(path+'/final.theta')
beta = np.loadtxt(path+'/final.beta')
beta = beta/np.sum(beta,0)'''


# inference on test set
seed = np.random.randint(seed0)
cmdtxt = './hdplda ' + str(seed) + ' test ' + testfile + ' ' + path + '/final ' + path
os.system(cmdtxt)
  
### read test topic proportions
'''
theta_test = np.loadtxt(path+'/testfinal.theta')
'''

