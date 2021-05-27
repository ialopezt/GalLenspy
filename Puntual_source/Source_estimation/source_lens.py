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
from galpy.potential import MiyamotoNagaiPotential, NFWPotential, RazorThinExponentialDiskPotential, BurkertPotential # GALPY potentials


# In[ ]:

tt=Table.Table.read('coordinates.txt', format='ascii.tab') 

#Import coordinates of images
theta1=tt['theta1'] 
theta2=tt['theta2']
sigma=tt['sigma']

theta=np.zeros(len(theta1),float)
for i in range(len(theta1)):
    theta[i]=np.sqrt(theta1[i]**2+theta2[i]**2)


tt=Table.Table.read('Cosmological_distances.txt', format='ascii.tab') # import cosmological distances
D_ds=tt['D_ds'][0] 
D_d=tt['D_d'][0]
D_s=tt['D_s'][0]
SIGMA_CRIT=tt['SIGMA_CRIT'][0]

tt=Table.Table.read('init_guess_params.txt', format='ascii.tab') # import initial guess of the source and lens
CX = tt['value'][0]
h = CX*np.pi/(180*3600)
CY = tt['value'][1]
k = CY*np.pi/(180*3600)
escala_r = tt['value'][2]
den_0 = tt['value'][3]

#R = 0.03
#r = R*np.pi/(180*3600)
#CX = -0.09
#h = CX*np.pi/(180*3600)
#CY = -0.01
#k = CY*np.pi/(180*3600)
    
FC = np.pi/(180*3600) #conversion factor between arcs and radians
Beta1 = h
Beta2 = k

#escala_r = 18
#den_0 = 44.5e8

#Illustration of obtained images with initial Guess

def POTDEFdisk_exp1(x):
    def integ(TheTa, theta):
        Sigma = den_0*np.exp(-D_d*TheTa/escala_r) #Volumetric density
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, theta):
        Sigma = den_0*np.exp(-D_d*TheTa/escala_r) #Volumetric density
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

GRADPOT1disk_exp = np.zeros((len(theta1)), float)
GRADPOT2disk_exp = np.zeros((len(theta1)), float)
GRADPOT1 = np.zeros((len(theta1)), float)
GRADPOT2 = np.zeros((len(theta1)), float)
THETA1 = np.zeros((len(theta1)), float)
THETA2 = np.zeros((len(theta1)), float)

for l in range(len(theta1)):
    GRADPOT1disk_exp[l]= derivative(POTDEFdisk_exp1, theta1[l], dx=1e-9, order=7)
    GRADPOT2disk_exp[l]= derivative(POTDEFdisk_exp2, theta2[l], dx=1e-9, order=7)
    GRADPOT1[l]=(SIGMA_CRIT**2)*(GRADPOT1disk_exp[l])
    GRADPOT2[l]=(SIGMA_CRIT**2)*(GRADPOT2disk_exp[l])

#Images obtained with initial guess
for l in range(len(theta1)):
    THETA1[l] = Beta1+GRADPOT1[l]
    THETA2[l] = Beta2+GRADPOT2[l]


# In[ ]:

#Graphics of source and images
fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(Beta1*1e6, Beta2*1e6, 'or')
plb.plot(theta1*1e6, theta2*1e6, 'ob')
plb.plot(THETA1*1e6, THETA2*1e6, 'og')
plb.xlim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.ylim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.savefig('Guess_initial_source_lens.pdf')

# In[ ]:


print ("\n#####################################################################")
print("MCMC------GALLENSPY")

#Model of the lens
def model(parameters, theta1, theta2, sigma):
    H, K, SIGMA_0, H_R  = parameters
    h = H*np.pi/(180*3600)
    k = K*np.pi/(180*3600)    
    Beta1 = h
    Beta2 = k
    def POTDEFdisk_exp1(TheTa1,theta):
        TheTa = np.sqrt(TheTa1**2+theta2[l]**2)
        R = D_d*TheTa
        Sigma = SIGMA_0*np.exp(-D_d*TheTa/H_R) #Volumetric density
        kappa = Sigma/SIGMA_CRIT
        return (2/theta1[l])*TheTa1*kappa/SIGMA_CRIT**2
    def POTDEFdisk_exp2(TheTa2,theta):
        TheTa = np.sqrt(TheTa2**2+theta1[l]**2)
        R = D_d*TheTa
        Sigma = SIGMA_0*np.exp(-D_d*TheTa/H_R) #Volumetric density
        kappa = Sigma/SIGMA_CRIT
        return (2/theta2[l])*TheTa2*kappa/SIGMA_CRIT**2
    GRADPOT1disk_exp = np.zeros((len(theta1)), float)
    GRADPOT2disk_exp = np.zeros((len(theta1)), float)
    GRADPOT1 = np.zeros((len(theta1)), float)
    GRADPOT2 = np.zeros((len(theta1)), float)
    THETA1 = np.zeros((len(theta1)), float)
    THETA2 = np.zeros((len(theta1)), float)
    for l in range(len(theta1)):
        GRADPOT1disk_exp[l]= quad(POTDEFdisk_exp1, 0, theta1[l], limit=100, args=(theta[l]))[0]
        GRADPOT2disk_exp[l]= quad(POTDEFdisk_exp2, 0, theta2[l], limit=100, args=(theta[l]))[0]
    for l in range(len(theta1)):
        GRADPOT1[l]=(SIGMA_CRIT**2)*(GRADPOT1disk_exp[l])
        GRADPOT2[l]=(SIGMA_CRIT**2)*(GRADPOT2disk_exp[l])
    for l in range(len(theta1)):
        THETA1[l] = Beta1+GRADPOT1[l]
        THETA2[l] = Beta2+GRADPOT2[l]
    THETA_teor = np.zeros((len(theta1)), float)
    for l in range(len(theta1)):
        THETA_teor[l] = np.sqrt(THETA1[l]**2+THETA2[l]**2)
    return THETA_teor

# In[ ]:

#Likelihood function
def lnlike(parameters, theta1, theta2, sigma):
    H, K, SIGMA_0, H_R  = parameters
    THETA_teor = model(parameters, theta1, theta2, sigma)
    X = np.zeros((len(theta1)),float)
    for l in range(len(theta1)):
        X[l]=((theta[l]-THETA_teor[l])**2)/(sigma[l]**2)
    return -0.5*np.sum(X)


# In[ ]:


#initial guess in the MCMC
start=np.zeros(4,float)
start[0] = CX
start[1] = CY
start[2] = den_0
start[3] = escala_r

# In[ ]:


#Parametric space in the MCMC
def lnprior(parameters):
    H, K, SIGMA_0, H_R  = parameters  
#    if 0.05<R_S<32 and 0.05e11<m_0<12e11 and 0.8e8<SIGMA_0<17e8 and 1<H_R<7 and 0.05<B<17 and 0.09e10<MASS<1.1e10:
    if -0.2<H<0.2 and -0.2<K<0.2 and 1e8<SIGMA_0<1e10 and 0<H_R<30:
        return 0.0
    return -np.inf

# In[ ]:


#Probability function
def lnprob(parameters, theta1, theta2, sigma):
    lp = lnprior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(parameters, theta1, theta2, sigma)

# In[ ]:


#Dimension and walkers
ndim, nwalkers = 4, 100
#initial posicion and step length
pos_step = 1e-8
pos_in = [abs(start + pos_step*start*np.random.randn(ndim)+1e-9*np.random.randn(ndim)) for i in range(nwalkers)]


# In[ ]:


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(theta1, theta2,sigma))


# In[ ]:


#Number of Steps
sampler.run_mcmc(pos_in, 1000, progress=True)


# In[ ]: Graphics of the chains for each parameter

fig = plt.figure()
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))

chain_steps = [i for i in range(len(sampler.chain[:,:,0].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,0].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[0]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
plb.savefig('h.pdf')


# In[ ]:

fig = plt.figure()
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))

chain_steps = [i for i in range(len(sampler.chain[:,:,1].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,1].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[1]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
plb.savefig('k.pdf')


# In[ ]:

fig = plt.figure()
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))

chain_steps = [i for i in range(len(sampler.chain[:,:,2].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,2].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[2]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
plb.savefig('Sigma_0.pdf')


# In[ ]:

fig = plt.figure()
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))

chain_steps = [i for i in range(len(sampler.chain[:,:,3].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,3].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[3]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
plb.savefig('h_r.pdf')


# In[ ]:

#Step of cut in the MCMC
samples = sampler.chain[:, 600:, :].reshape((-1, ndim))

# In[ ]:


percentage=0.68
#Contours of values
fig = corner.corner(samples, labels=["$h$", r"$k$", r"$\Sigma_0$", "$h_r$"],
                    quantiles = [0.5-0.5*percentage, 0.5, 0.5+0.5*percentage],fill_contours=True, plot_datapoints=True)
fig.savefig("contours_source_lens.pdf")

# In[ ]:


#Parameters and errors
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


#Visualization of generated images for the parameter set obtained 
H = para[0]
K = para[1]
Sigma_0 = para[2] 
h_r = para[3]

h = H*np.pi/(180*3600)
k = K*np.pi/(180*3600)    
Beta1 = h
Beta2 = k

def POTDEFdisk_exp1(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r) #Volumetric density
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r) #Volumetric density
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

GRADPOT1disk_exp = np.zeros((len(theta1)), float)
GRADPOT2disk_exp = np.zeros((len(theta1)), float)
GRADPOT1 = np.zeros((len(theta1)), float)
GRADPOT2 = np.zeros((len(theta1)), float)
THETA1 = np.zeros((len(theta1)), float)
THETA2 = np.zeros((len(theta1)), float)

for l in range(len(theta1)):
    GRADPOT1disk_exp[l]= derivative(POTDEFdisk_exp1, theta1[l], dx=1e-9, order=7)
    GRADPOT2disk_exp[l]= derivative(POTDEFdisk_exp2, theta2[l], dx=1e-9, order=7)
    GRADPOT1[l]=(SIGMA_CRIT**2)*(GRADPOT1disk_exp[l])
    GRADPOT2[l]=(SIGMA_CRIT**2)*(GRADPOT2disk_exp[l])

#Images obtained with initial guess
for l in range(len(theta1)):
    THETA1[l] = Beta1+GRADPOT1[l]
    THETA2[l] = Beta2+GRADPOT2[l]


# In[ ]:


#Graphics of source and images
fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(Beta1/FC, Beta2/FC, 'or')
plb.plot(theta1/FC, theta2/FC, 'ob')
plb.plot(THETA1/FC, THETA2/FC, 'og')
plb.xlim(-2.5,2.5)
plb.ylim(-2.5,2.5)
plb.xlabel(r"$\theta_1$", fontsize = 15)
plb.ylabel(r"$\theta_2$", fontsize = 15)
plb.legend(['Source', 'Observational data', 'Model values'], loc='upper right', fontsize=15)
#plb.show()
plt.savefig('fitting.pdf')

# In[ ]:


#Parameters and errors
h= para[0]; h_95pos = parap95[0]; h_95neg = paran95[0]; h_68pos = parap68[0]; h_68neg = paran68[0]
k = para[1]; k_95pos = parap95[1]; k_95neg = paran95[1]; k_68pos = parap68[1]; k_68neg = paran68[1]
Sigma_0 = para[2]; Sigma_0_95pos = parap95[2]; Sigma_0_95neg = paran95[2]; Sigma_0_68pos = parap68[2]; Sigma_0_68neg = paran68[2]
h_r = para[3]; h_r_95pos = parap95[3]; h_r_95neg = paran95[3]; h_r_68pos = parap68[3]; h_r_68neg = paran68[3]

# In[ ]:

table_data = []

table_para = [r"h",r"k",r"Sigma_0", r"h_r"]
table_units = [r"arcs",r"arcs", r"Solar_Mass/Kpc^2", r"Kpc"]
para = [h, k, Sigma_0, h_r]
parap68=[h_68pos, k_68pos, Sigma_0_68pos, h_r_68pos]
paran68=[h_68neg, k_68neg, Sigma_0_68neg, h_r_68neg]
parap95=[h_95pos, k_95pos, Sigma_0_95pos, h_r_95pos]
paran95=[h_95neg, k_95neg, Sigma_0_95neg, h_r_95neg]
index=[r"h",r"k",r"Sigma_0", r"h_r"]

for i in range(len(para)):
	table_data.append([table_para[i], table_units[i], para[i], parap95[i], paran95[i],  parap68[i], paran68[i]])

column_name = [r"PARAMETER", r"UNITS", r"FIT", r"95%(+)", r"95%(-)",  r"68%(+)", r"68%(-)"]	
table_p = pd.DataFrame(table_data, index=index, columns=column_name)
table_p.to_csv("parameters_lens_source.txt", sep='\t', encoding='utf-8')
print ("\n#####################################################################")
print(table_p)
print ("\nDone")
print ("\n#####################################################################\n")


