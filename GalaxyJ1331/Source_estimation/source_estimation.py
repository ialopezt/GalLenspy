from scipy.misc import *
import numpy as np
import pylab as plb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import quad
from scipy.integrate import nquad
from scipy.misc import derivative
import pandas as pd
import emcee
import corner
from astropy import table as Table # For fast and easy reading / writing with tables using numpy library


# In[2]:

#coordinates of the images
theta_ra = [12.1, -8.5, 21.7, -3.3]
theta_dec = [16.6, -10.4, -0.5, 19.2]


# In[3]:

#Position of images respect to the galactic center

theta_1 = np.zeros(len(theta_ra), float)
theta_2 = np.zeros(len(theta_dec), float)

for i in range(len(theta_1)):
    theta_1[i] = theta_ra[i]-0.5
    theta_2[i] = theta_dec[i]-0.5


# In[4]:

#Position of images in radians 
theta1 = 0.05*theta_1*np.pi/(180*3600)
theta2 = 0.05*theta_2*np.pi/(180*3600)

# In[ ]:

theta = np.zeros(len(theta1),float)
for i in range(len(theta1)):
    theta[i]=np.sqrt(theta1[i]**2+theta2[i]**2)


# In[ ]:

#Error in the position of images
sigma = 0.009*np.pi/(180*3600)


# In[ ]:
#Conversion factor between arcs and radians
FC = np.pi/(180*3600)

#initial values of the source

beta1 = 0
beta2 = 0

# In[ ]:

#Cosmological distances
D_ds = 442.7e3
D_d = 422e3
D_s = 817.9e3
d_ds = D_ds*1e3*3.086e16
d_d = D_d*1e3*3.086e16
d_s = D_s*1e3*3.086e16

#Light velocity
c = 3e8
#Universal gravitation constant
G = 6.67e-11

SIGMA_CRIT = (c**2)*d_s/(4*np.pi*G*d_d*d_ds) #Critical Sigma in kg/m^2",
SIGMA_CRIT = SIGMA_CRIT*5.027e-31*1e6/((3.241e-17)**2)

#Functions of deflector potential
def POTDEF1(x):
    def integ(TheTa, theta):
        return 2*TheTa*np.log(THETA/TheTa)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

def POTDEF2(x):
    def integ(TheTa, theta):
        return 2*TheTa*np.log(THETA/TheTa)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x


#Gradient of deflector potential
GRADPOT1 = np.zeros((len(theta1)), float)
GRADPOT2 = np.zeros((len(theta1)), float)
THETA1 = np.zeros((len(theta1)), float)
THETA2 = np.zeros((len(theta1)), float)

for l in range(len(theta1)):
    GRADPOT1[l]= derivative(POTDEF1, theta1[l], dx=1e-9, order=7)
    GRADPOT2[l]= derivative(POTDEF2, theta2[l], dx=1e-9, order=7)

# In[ ]:


#Model of the MCMC
def model(parameters, theta1, theta2, sigma):
    BETA1,BETA2 = parameters
    Beta1 = BETA1*FC
    Beta2 = BETA2*FC
    for l in range(len(theta1)):
        THETA1[l] = BETA1+GRADPOT1[l]
        THETA2[l] = BETA2+GRADPOT2[l]
    THETA_teor = np.zeros((len(theta1)), float)
    for l in range(len(theta1)):
        THETA_teor[l] = np.sqrt(THETA1[l]**2+THETA2[l]**2)
    return THETA_teor

#Likelihood
def lnlike(parameters, theta1, theta2, sigma):
    BETA1,BETA2 = parameters
    THETA_teor = model(parameters, theta1, theta2, sigma)
    X = np.zeros((len(theta1)),float)
    for l in range(len(theta1)):
        X[l]=((theta[l]-THETA_teor[l])**2)/(sigma**2)
    return -0.5*np.sum(X)


# In[ ]:

#Initial guess of the MCMC
start=np.zeros(2,float)
start[0] = beta1
start[1] = beta2

# In[ ]:

#Defining of the prior

def lnprior(parameters):
    BETA1,BETA2 = parameters
    if -1<BETA1<1 and -1<BETA2<1:
        return 0.0
    return -np.inf

#Probability function
def lnprob(parameters, theta1, theta2, sigma):
    lp = lnprior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(parameters, theta1, theta2, sigma)


# In[ ]:


#Dimension and walkers
ndim, nwalkers = 2, 100
#Initial position and step length
pos_step = 1e-8
pos_in = [abs(start + pos_step*start*np.random.randn(ndim)+1e-9*np.random.randn(ndim)) for i in range(nwalkers)]


# In[ ]:


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(theta1, theta2,sigma))


# In[ ]:

#Number of steps
sampler.run_mcmc(pos_in, 1000)


# In[ ]:
#MCMC beta1
fig = plt.figure()
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))

chain_steps = [i for i in range(len(sampler.chain[:,:,0].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,0].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[0]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
plb.savefig('BETA1.pdf')


# In[ ]:
#MCMC beta2
fig = plt.figure()
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))
chain_steps = [i for i in range(len(sampler.chain[:,:,1].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,1].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[1]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
plb.savefig('BETA2.pdf')

#Cut in the chain
samples = sampler.chain[:, 500:, :].reshape((-1, ndim))


# In[ ]:

#Obtaining contours
percentage = 0.68
fig = corner.corner(samples, labels=[r"$\beta_1$", r"$\beta_2$"],
                    truths=[start[0],start[1]], quantiles = [0.5-0.5*percentage, 0.5, 0.5+0.5*percentage],fill_contours=True, plot_datapoints=True)
fig.savefig("contours_source.pdf")


# In[ ]:

#Parameters of the lens and its uncertainties
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

#Source in radians 
Beta1 = para[0]*FC
Beta2 = para[1]*FC

#Images of the source
THETA1 = np.zeros(len(theta1),float)
THETA2 = np.zeros(len(theta1),float)

for l in range(len(theta1)):
    THETA1[l] = Beta1+GRADPOT1[l]
    THETA2[l] = Beta2+GRADPOT2[l]


# In[ ]:


#Comparison observational and model images
fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(Beta1/FC, Beta2/FC, 'or')
plb.plot(theta1/FC, theta2/FC, 'ob')
plb.plot(THETA1/FC, THETA2/FC, 'og')
plb.xlim(-2.5,2.5)
plb.ylim(-2.5,2.5)
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
plb.legend(['Source', 'Observational data', 'Model values'], loc='upper right', fontsize=15)
#plb.show()
plb.savefig('images_source.pdf')

fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(theta1/FC, theta2/FC, 'xr')
plb.plot(THETA1/FC, THETA2/FC, 'og')
plb.xlim(-2.5,2.5)
plb.ylim(-2.5,2.5)
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
plb.legend(['Observational data', 'Model values'], loc='upper right', fontsize=15)
#plb.show()
plb.savefig('images.pdf')

fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(Beta1/FC, Beta2/FC, 'or')
plb.xlim(-2.5,2.5)
plb.ylim(-2.5,2.5)
plb.xlabel(r"$\beta_1$", fontsize=20)
plb.ylabel(r"$\beta_2$", fontsize=20)
plb.legend(['Source'], loc='upper right', fontsize=15)
#plb.show()
plb.savefig('source.pdf')


#Parameters and its errors
beta1 = para[0]; beta1_95pos = parap95[0]; beta1_95neg = paran95[0]; beta1_68pos = parap68[0]; beta1_68neg = paran68[0]
beta2 = para[1]; beta2_95pos = parap95[1]; beta2_95neg = paran95[1]; beta2_68pos = parap68[1]; beta2_68neg = paran68[1]

ang1=np.zeros(len(theta),float)
ang2=np.zeros(len(theta),float)
ang=np.zeros(len(theta),float)
rad_eins=np.zeros(len(theta),float)

#Compute of Einstein radius
for i in range(len(theta)):
    ang1[i]=(theta1[i]/FC)-beta1
    ang2[i]=(theta2[i]/FC)-beta2
    ang[i]=np.sqrt(ang1[i]**2+ang2[i]**2)
    rad_eins[i]=np.sqrt(theta[i]/FC)*ang[i]

theta_eins=np.sum(rad_eins)/len(theta)


# In[ ]:

table_data = []
table_para = [r"$\beta1",r"$\beta1"]
table_units = [r"arcseg",r"arcseg"]
para = [beta1,beta2]
parap68=[beta1_68pos, beta2_68pos]
paran68=[beta1_68neg, beta2_68neg]
parap95=[beta1_95pos, beta2_95pos]
paran95=[beta1_95neg, beta2_95neg]
index=[r"$\beta1",r"$\beta2"]


for i in range(len(para)):
	table_data.append([table_para[i], table_units[i], para[i], parap95[i], paran95[i],  parap68[i], paran68[i]])

column_name = [r"PARAMETER", r"UNITS", r"FIT", r"95%(+)", r"95%(-)",  r"68%(+)", r"68%(-)"]	
table_p = pd.DataFrame(table_data, index=index, columns=column_name)
table_p.to_csv("source_parameters.txt", sep='\t', encoding='utf-8')
print (table_p)
print("R_Eins=",theta_eins)
print ("\n#####################################################################")
print ("\nDone")
print ("\n#####################################################################\n")



