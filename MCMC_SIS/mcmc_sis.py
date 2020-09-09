#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np #for data handling and numerical process
import pylab as plb #Graphics and data
from scipy.integrate import quad # for numerical integrate with the quadratures methods 
from scipy.integrate import nquad # for numerical integrate with the quadratures methods
from scipy.misc import derivative # for numerical derivatives
import emcee #for the MCMC
import corner #for the obtaining and graphics of uncertainties
import matplotlib.pyplot as plt #for graphics
import pandas as pd #for data handling
from astropy import table as Table # For fast and easy reading / writing with tables using numpy library



# In[ ]:


#Main parameters

D_ds = 1 #distance lens-source
D_d = 1 #distance observer-lens
D_s = D_ds + D_d #distance oberver-source
c = 1 #light velocity  in natural unities
o = 1 #Dispersion velocity
G = 1 #Universal Gravitation Constant 

# Circular source
r = 1 #radius
h = 0.8 #center in X
k = 0.8 #center in y
y0 = np.sqrt(h**2 + k**2)
N = 1000
alpha = np.linspace(0, 2*np.pi, N) #Number of points for the circunference
Beta = r
Beta1 = Beta*np.cos(alpha)+h
Beta2 = Beta*np.sin(alpha)+k


# In[ ]:


#Analytical solution to the lens equation for the SIS profile

Thetap = (Beta1**2+Beta2**2)**0.5 + (4*np.pi*o**2*D_ds/(c**2*D_s)) 
Thetan = -(Beta1**2+Beta2**2)**0.5 + (4*np.pi*o**2*D_ds/(c**2*D_s)) 

Theta1p = Beta1/(1-(4*np.pi*o**2*D_ds/(c**2*D_s*Thetap)))
Theta2p = Beta2/(1-(4*np.pi*o**2*D_ds/(c**2*D_s*Thetap)))
thetap = np.sqrt(Theta1p**2 +Theta2p**2) 

Theta1n = Beta1/(1-(4*np.pi*o**2*D_ds/(c**2*D_s*Thetan)))
Theta2n = Beta2/(1-(4*np.pi*o**2*D_ds/(c**2*D_s*Thetan)))


# In[ ]:


#Formation of images while the SIS profile for the circular source
fig = plt.figure()
plb.rcParams['figure.figsize'] =(5,5)
plb.plot(Theta1p, Theta2p, color='black')
plb.plot(Theta1n, Theta2n, color='black')
plb.plot(Beta1, Beta2, color='blue')
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
plb.savefig('images_sis.pdf')



# In[ ]:


#Bayesian statistics for the parameters exploration

#parameter space for quartils 
N2 = 4
O = np.linspace(0.1, 2, N2) #radio de escala del NFW
R = np.linspace(0.1, 2, N2) #Masa central del NFW
H = np.linspace(-2, 2, N2)
K = np.linspace(-2, 2, N2)


# In[ ]:


#Compute of the initial guess with the multiparameter minimization function for quartils

BETA = R
BETA1 = np.zeros((len(R), len(H),len(alpha)), float)
BETA2 = np.zeros((len(R), len(H),len(alpha)), float)
THETAP = np.zeros((len(R), len(H), len(K), len(O), len(alpha)), float)
THETA1P = np.zeros((len(R), len(H), len(H), len(O), len(alpha)), float)
THETA2P = np.zeros((len(R), len(H), len(H), len(O), len(alpha)), float)
THETAPOS = np.zeros((len(R), len(H), len(H), len(O), len(alpha)), float)
X = np.zeros((len(R), len(H), len(H), len(O), len(alpha)), float)
L = np.zeros((len(R), len(H), len(H), len(O)), float)

for i in range(len(R)):
    for j in range(len(H)):
        for k in range(len(K)):
            for l in range(len(O)):
                for m in range(len(alpha)):
                    BETA1[i,j,m] = BETA[i]*np.cos(alpha[m])+H[j]
                    BETA2[i,k,m] = BETA[i]*np.sin(alpha[m])+K[k]
                    THETAP[i,j,k,l,m] = (BETA1[i,j,m]**2+BETA2[i,k,m]**2)**0.5 + (4*np.pi*O[l]**2*D_ds/(c**2*D_s)) 
                    THETA1P[i,j,k,l,m] = BETA1[i,j,m]/(1-(4*np.pi*O[l]**2*D_ds/(c**2*D_s*THETAP[i,j,k,l,m])))
                    THETA2P[i,j,k,l,m] = BETA2[i,k,m]/(1-(4*np.pi*O[l]**2*D_ds/(c**2*D_s*THETAP[i,j,k,l,m])))
                    THETAPOS[i,j,k,l,m] = np.sqrt(THETA1P[i,j,k,l,m]**2+THETA2P[i,j,k,l,m]**2)
                    X[i,j,k,l,m] = ((thetap[m]-THETAPOS[i,j,k,l,m])**2) #minimization function
                L[i,j,k,l] = np.sum(X[i,j,k,l]) #minimization function for the set of data           


# In[ ]:


#Minimization for each parameter

LIKE = np.zeros((len(R), len(H), len(K)), float)
lik = np.zeros((len(R), len(H)), float)
like = np.zeros(len(R),float)
Minim = []
Sigma = []
Radio = []
H_x = []
K_y = []

for i in range(len(R)):
    for j in range(len(H)):
        for k in range(len(K)):
            for l in range(len(O)):
                LIKE[i,j,k]=min(L[i,j,k])
                lik[i,j]=min(LIKE[i,j])
                like[i]=min(lik[i])
                
for i in range(len(R)):
    for j in range(len(H)):
        for k in range(len(K)):
            for l in range(len(O)):
                for m in range(len(like)):
                    if L[i,j,k,l]==like[m]:
                        Minim.append(L[i,j,k,l])
                        Sigma.append(O[l])
                        Radio.append(BETA[i])
                        H_x.append(H[j])
                        K_y.append(K[k])

                        
#Parameters get for the initial guess                        
for i in range(len(Minim)):
    if Minim[i]==min(Minim):
        Likehood = Minim[i]
        SIGMA = Sigma[i]
        RADIO = Radio[i]
        H = H_x[i]
        K = K_y[i]


# In[ ]:


#Source of the initial guess
Beta = RADIO
Beta1 = Beta*np.cos(alpha)+H
Beta2 = Beta*np.sin(alpha)+K
o = SIGMA

#Images of the initial guess
THETAP = (Beta1**2+Beta2**2)**0.5 + (4*np.pi*o**2*D_ds/(c**2*D_s)) 
THETA1P = Beta1/(1-(4*np.pi*o**2*D_ds/(c**2*D_s*THETAP)))
THETA2P = Beta2/(1-(4*np.pi*o**2*D_ds/(c**2*D_s*THETAP)))
THETP = np.sqrt(THETA1P**2 +THETA2P**2) 


# In[ ]:


#Comparison between the images of the analytical solution and the images of the initial guess
fig = plt.figure()
plb.rcParams['figure.figsize'] =(5,5)
plb.plot(Theta1p, Theta2p, color='black')
plb.plot(THETA1P, THETA2P, color='red')
plb.plot(Beta1, Beta2, color='blue')
plb.legend(['Observed images', 'Images of the model', 'Source'], loc='upper right', fontsize=12)
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
plb.xlim(-3,12)
plb.ylim(-3,12)
plb.savefig('initial_guess.pdf')


# In[ ]:


#Run of the MCMC from the initial guess
sigma = 0.0001

#Model of the MCMC
def model(parameters, Theta1p, Theta2p, sigma):
    Rad, c_x, c_y, SIGMA_0 = parameters
    beta = Rad
    beta1 = beta*np.cos(alpha)+c_x
    beta2 = beta*np.sin(alpha)+c_y
    o = SIGMA_0
    THETAP = (beta1**2+beta2**2)**0.5 + (4*np.pi*o**2*D_ds/(c**2*D_s)) 
    THETA1P = beta1/(1-(4*np.pi*o**2*D_ds/(c**2*D_s*THETAP)))
    THETA2P = beta2/(1-(4*np.pi*o**2*D_ds/(c**2*D_s*THETAP)))
    THETA_teor = np.sqrt(THETA1P**2 +THETA2P**2) 
    return THETA_teor

#Likelihood
def lnlike(parameters, Theta1p, Theta2p, sigma):
    Rad, c_x, c_y, SIGMA_0 = parameters
    THETA_teor = model(parameters, Theta1p, Theta2p, sigma)
    X = np.zeros((len(Theta1p)),float)
    for l in range(len(Theta1p)):
        X[l]=((thetap[l]-THETA_teor[l])**2)/(sigma**2)
    return -0.5*np.sum(X)


# In[ ]:


#Initial guess
start=np.zeros(4,float)
start[0] = RADIO
start[1] = H
start[2] = K
start[3] = SIGMA


# In[ ]:


#parametric space
def lnprior(parameters):
    Rad, c_x, c_y, SIGMA_0 = parameters
    if Rad>0 and -2<c_x<2 and -2<c_y<2 and SIGMA_0>0:
        return 0.0
    return -np.inf


# In[ ]:


#Priori function
def lnprob(parameters, Theta1p, Theta2p, sigma):
    lp = lnprior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(parameters, Theta1p, Theta2p, sigma)


# In[ ]:


#Dimention and walkers
ndim, nwalkers = 4, 100
#lentgh of the step
pos_step = 1e-8
pos_in = [abs(start + pos_step*start*np.random.randn(ndim)+1e-9*np.random.randn(ndim)) for i in range(nwalkers)]


# In[ ]:


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(Theta1p, Theta2p, sigma))


# In[ ]:


#Number of steps for the chain
sampler.run_mcmc(pos_in, 1000)


# In[ ]:


#path of the chain for the radius parameter
fig = plt.figure()
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))
chain_steps = [i for i in range(len(sampler.chain[:,:,0].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,0].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[0]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
plb.xlabel(r"$steps$", fontsize=20)
plb.ylabel(r"$r$", fontsize=20)
plb.savefig('radius.pdf')


# In[ ]:


#path of the chain for the h parameter
fig = plt.figure()
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))
chain_steps = [i for i in range(len(sampler.chain[:,:,1].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,1].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[1]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
plb.ylabel(r"$h$", fontsize=20)
plb.xlabel(r"$steps$", fontsize=20)
plb.savefig('h_paramtere.pdf')


# In[ ]:


#path of the chain for the k parameter
fig = plt.figure()
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))
chain_steps = [i for i in range(len(sampler.chain[:,:,2].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,2].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[2]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
plb.ylabel(r"$k$", fontsize=20)
plb.xlabel(r"$steps$", fontsize=20)
plb.savefig('k_parameter.pdf')


# In[ ]:


#path of the chain for the sigma parameter
fig = plt.figure()
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))
chain_steps = [i for i in range(len(sampler.chain[:,:,3].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,3].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[3]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
plb.ylabel(r"$\sigma$", fontsize=20)
plb.xlabel(r"$steps$", fontsize=20)
plb.savefig('sigma.pdf')


# In[ ]:


samples = sampler.chain[:, 500:, :].reshape((-1, ndim))


# In[ ]:


#Contours of obtained values 
fig = corner.corner(np.log10(samples), labels=["R", "h", "k", r"$\sigma$"],
                    truths=[start[0],start[1],start[2],start[3]])
fig.savefig("triangle.pdf")


# In[ ]:


#estimation of uncertainties
para = []
parap68=[]; paran68=[]
parap95=[]; paran95=[]
fit_para = []

for i in range(ndim):	
	mcmc = np.percentile(samples[:, i], [50.-0.5*95, 50.-0.5*68, 50., 50.+0.5*68, 50.+0.5*95])
	para.append(mcmc[2])
	fit_para.append(mcmc[2]) 
	parap68.append(mcmc[3]-mcmc[2])
	paran68.append(mcmc[2]-mcmc[1])
	parap95.append(mcmc[4]-mcmc[2])
	paran95.append(mcmc[2]-mcmc[0])


# In[ ]:


#Obtained source
Beta = para[0]
Beta1 = Beta*np.cos(alpha)+para[1]
Beta2 = Beta*np.sin(alpha)+para[2]
O = para[3]

#Obtained images
THETAP = (Beta1**2+Beta2**2)**0.5 + (4*np.pi*O**2*D_ds/(c**2*D_s)) 
THETA1P = Beta1/(1-(4*np.pi*O**2*D_ds/(c**2*D_s*THETAP)))
THETA2P = Beta2/(1-(4*np.pi*O**2*D_ds/(c**2*D_s*THETAP)))
THETP = np.sqrt(THETA1P**2 +THETA2P**2) 


# In[ ]:


#Comparison between images of model and images of analytical solution
fig = plt.figure()
plb.rcParams['figure.figsize'] =(5,5)
plb.plot(Theta1p, Theta2p, color='black')
plb.plot(THETA1P, THETA2P, '--r')
plb.plot(Beta1, Beta2, color='blue')
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
plb.legend(['Analytical Images', 'Model Imáges', 'Source'], loc='upper right', fontsize=12)
plb.xlim(-3,12)
plb.ylim(-3,12)
plb.savefig('comparison.pdf')


# In[ ]:


#Values and its uncertainties
r = para[0]; r_95pos = parap95[0]; r_95neg = paran95[0]; r_68pos = parap68[0]; r_68neg = paran68[0]
h = para[1]; h_95pos = parap95[1]; h_95neg = paran95[1]; h_68pos = parap68[1]; h_68neg = paran68[1]
k = para[2]; k_95pos = parap95[2]; k_95neg = paran95[2]; k_68pos = parap68[2]; k_68neg = paran68[2]
O = para[3]; O_95pos = parap95[3]; O_95neg = paran95[3]; O_68pos = parap68[3]; O_68neg = paran68[3]

# In[ ]:

table_data = []

table_para = [r"r",r"h",r"k", r"sigma"]
table_units = [r"arcseg",r"arcseg",r"arcseg", r"Kpc/s"]
para = [r,h,k,O]
parap68=[r_68pos, h_68pos, k_68pos, O_68pos]
paran68=[r_68neg, h_68neg, k_68neg, O_68neg]
parap95=[r_95pos, h_95pos, k_95pos, O_95pos]
paran95=[r_95neg, h_95neg, k_95neg, O_95neg]
index=[r"radius",r"center_x",r"center_y",r"sigma"]

for i in range(len(para)):
	table_data.append([table_para[i], table_units[i], para[i], parap95[i], paran95[i],  parap68[i], paran68[i]])

column_name = [r"PARAMETER", r"UNITS", r"FIT", r"95%(+)", r"95%(-)",  r"68%(+)", r"68%(-)"]	
table_p = pd.DataFrame(table_data, index=index, columns=column_name)
table_p.to_csv("parameters_MCMC.txt", sep='\t', encoding='utf-8')
print ("\n#####################################################################")
print(table_p)
print ("\nDone")
print ("\n#####################################################################\n")




