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

tt=Table.Table.read('coordinates.txt', format='ascii.tab') # importando los datos de las imágenes

#Import coordinates of images
theta1=tt['theta1'] 
theta2=tt['theta2']
sigma=tt['sigma']

theta=np.zeros(len(theta1),float)
for i in range(len(theta1)):
    theta[i]=np.sqrt(theta1[i]**2+theta2[i]**2)

tt=Table.Table.read('alpha.txt', format='ascii.tab') # Import the values of the angles belonging to the circle for the arc
#Import the values alpha
alpha=tt['alpha']

tt=Table.Table.read('Cosmological_distances.txt', format='ascii.tab') # importando los datos de distancias cosmológicas
#Importando distancias cosmológicas y Sigma Crítico
D_ds=tt['D_ds'][0] 
D_d=tt['D_d'][0]
D_s=tt['D_s'][0]
SIGMA_CRIT=tt['SIGMA_CRIT'][0]

tt=Table.Table.read('parameters_lens_source.txt', format='ascii.tab') # importando los datos de distancias cosmológicas

#Importando parámetros de la fuente
H=tt['FIT'][0] 
K=tt['FIT'][1]


R = float(input("\nEnter the radius of the circular source:\n"))

#Recommended value R=0.03
r = R*np.pi/(180*3600)
h = H*np.pi/(180*3600)
K = K*np.pi/(180*3600)
    
Beta1 = r*np.cos(alpha)+h
Beta2 = r*np.sin(alpha)+K
FC = np.pi/(180*3600) #conversion factor between arcs and radians


radio_s = float(input("\nEnter initial guess for r_s in NFW:\n"))
M_0 = float(input("\nEnter initial guess for m_0 in NFW:\n"))
escala_r = float(input("\nEnter initial guess for h_r in Exponential disk:\n"))
den_0 = float(input("\nEnter initial guess for Sigma_0 in Exponential disk:\n"))
height = float(input("\nEnter initial guess for b in Miyamoto Nagai:\n"))
Mass = float(input("\nEnter initial guess for M in Miyamoto Nagai:\n"))


#radio_s = 5.1
#M_0 = 2.3e11
#escala_r = 15.6
#den_0 = 4.3e9
#height = 1.48
#Mass = 0.25e10

# In[ ]:


#Illustration of obtained images with initial Guess

def POTDEFnfw1(x):
    def integ(TheTa, theta):
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=M_0, a=radio_s, normalize=False)
        Sigma = NFW_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta[l]))[0]
    return x

def POTDEFnfw2(x):
    def integ(TheTa, theta):
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=M_0, a=radio_s, normalize=False)
        Sigma = NFW_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta[l]))[0]
    return x

def MN1(x):
    def integ(TheTa, theta):
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=Mass,a=0,b=height,normalize=False)
        Sigma = MN_Bulge_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta[l]))[0]
    return x

def MN2(x):
    def integ(TheTa, theta):
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=Mass,a=0,b=height,normalize=False)
        Sigma = MN_Bulge_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta[l]))[0]
    return x

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

#Obteniendo gradiente del potencial deflector
GRADPOT1nfw = np.zeros((len(theta1)), float)
GRADPOT2nfw = np.zeros((len(theta1)), float)
GRADPOT1disk_exp = np.zeros((len(theta1)), float)
GRADPOT2disk_exp = np.zeros((len(theta1)), float)
GRADPOT1MN = np.zeros((len(theta1)), float)
GRADPOT2MN = np.zeros((len(theta1)), float)
GRADPOT1 = np.zeros((len(theta1)), float)
GRADPOT2 = np.zeros((len(theta1)), float)
THETA1 = np.zeros((len(theta1)), float)
THETA2 = np.zeros((len(theta1)), float)

for l in range(len(theta1)):
    GRADPOT1nfw[l]= derivative(POTDEFnfw1, theta1[l], dx=1e-9, order=7)
    GRADPOT2nfw[l]= derivative(POTDEFnfw2, theta2[l], dx=1e-9, order=7)
    GRADPOT1disk_exp[l]= derivative(POTDEFdisk_exp1, theta1[l], dx=1e-9, order=7)
    GRADPOT2disk_exp[l]= derivative(POTDEFdisk_exp2, theta2[l], dx=1e-9, order=7)
    GRADPOT1MN[l]= derivative(MN1, theta1[l], dx=1e-9, order=7)
    GRADPOT2MN[l]= derivative(MN2, theta2[l], dx=1e-9, order=7)
    GRADPOT1[l]=(SIGMA_CRIT**2)*(GRADPOT1nfw[l]+GRADPOT1disk_exp[l]+GRADPOT1MN[l])
    GRADPOT2[l]=(SIGMA_CRIT**2)*(GRADPOT2nfw[l]+GRADPOT2disk_exp[l]+GRADPOT2MN[l])

#Images obtained with initial guess
for l in range(len(theta1)):
    THETA1[l] = Beta1[l]+GRADPOT1[l]
    THETA2[l] = Beta2[l]+GRADPOT2[l]


# In[ ]:

#Graphics of source and images
fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(Beta1*1e6, Beta2*1e6, '--r')
plb.plot(theta1*1e6, theta2*1e6, 'ob')
plb.plot(THETA1*1e6, THETA2*1e6, 'og')
plb.xlim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.ylim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.savefig('Guess_initial.pdf')

# In[ ]:


print ("\n#####################################################################")
print("MCMC------GALLENSPY")

#Model of the lens
def model(parameters, theta1, theta2, sigma):
    R_S, m_0, SIGMA_0, H_R, B, MASS  = parameters
    def POTDEFnfw1(x):
        def integ(TheTa, theta):
            R = D_d*TheTa
            NFW_p = NFWPotential(amp=m_0, a=R_S, normalize=False)
            Sigma = NFW_p.dens(R,0.)
            kappa = Sigma/SIGMA_CRIT
            return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
        THETA = np.sqrt(x**2 + theta2[l]**2)
        x = quad(integ, 0, theta[l], limit=100, args=(theta[l]))[0]
        return x
    def POTDEFnfw2(x):
        def integ(TheTa, theta):
            R = D_d*TheTa
            NFW_p = NFWPotential(amp=m_0, a=R_S, normalize=False)
            Sigma = NFW_p.dens(R,0.)
            kappa = Sigma/SIGMA_CRIT
            return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
        THETA = np.sqrt(x**2 + theta1[l]**2)
        x = quad(integ, 0, theta[l], limit=100, args=(theta[l]))[0]
        return x
    def MN1(x):
        def integ(TheTa, theta):
            R = D_d*TheTa
            MN_Bulge_p= MiyamotoNagaiPotential(amp=MASS,a=0,b=B,normalize=False)
            Sigma = MN_Bulge_p.dens(R,0.)
            kappa = Sigma/SIGMA_CRIT
            return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
        THETA = np.sqrt(x**2 + theta2[l]**2)
        x = quad(integ, 0, theta[l], limit=100, args=(theta[l]))[0]
        return x
    def MN2(x):
        def integ(TheTa, theta):
            R = D_d*TheTa
            MN_Bulge_p= MiyamotoNagaiPotential(amp=MASS,a=0,b=B,normalize=False)
            Sigma = MN_Bulge_p.dens(R,0.)
            kappa = Sigma/SIGMA_CRIT
            return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
        THETA = np.sqrt(x**2 + theta1[l]**2)
        x = quad(integ, 0, theta[l], limit=100, args=(theta[l]))[0]
        return x
    def POTDEFdisk_exp1(x):
        def integ(TheTa, theta):
            Sigma = SIGMA_0*np.exp(-D_d*TheTa/H_R) #Volumetric density
            return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
        THETA = np.sqrt(x**2 + theta2[l]**2)
        x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
        return x
    def POTDEFdisk_exp2(x):
        def integ(TheTa, theta):
            Sigma = SIGMA_0*np.exp(-D_d*TheTa/H_R) #Volumetric density
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
    GRADPOT1nfw = np.zeros((len(theta1)), float)
    GRADPOT2nfw = np.zeros((len(theta1)), float)
    GRADPOT1MN = np.zeros((len(theta1)), float)
    GRADPOT2MN = np.zeros((len(theta1)), float)
    for l in range(len(theta1)):
        GRADPOT1nfw[l]= derivative(POTDEFnfw1, theta1[l], dx=1e-9, order=7)
        GRADPOT2nfw[l]= derivative(POTDEFnfw2, theta2[l], dx=1e-9, order=7)
    for l in range(len(theta1)):
        GRADPOT1disk_exp[l]= derivative(POTDEFdisk_exp1, theta1[l], dx=1e-9, order=7)
        GRADPOT2disk_exp[l]= derivative(POTDEFdisk_exp2, theta2[l], dx=1e-9, order=7)
    for l in range(len(theta1)):
        GRADPOT1MN[l]= derivative(MN1, theta1[l], dx=1e-9, order=7)
        GRADPOT2MN[l]= derivative(MN2, theta2[l], dx=1e-9, order=7)    
    for l in range(len(theta1)):
        GRADPOT1[l]=(SIGMA_CRIT**2)*(GRADPOT1nfw[l]+GRADPOT1disk_exp[l]+GRADPOT1MN[l])
        GRADPOT2[l]=(SIGMA_CRIT**2)*(GRADPOT2nfw[l]+GRADPOT2disk_exp[l]+GRADPOT2MN[l])
    for l in range(len(theta1)):
        THETA1[l] = Beta1[l]+GRADPOT1[l]
        THETA2[l] = Beta2[l]+GRADPOT2[l]
    THETA_teor = np.zeros((len(theta1)), float)
    for l in range(len(theta1)):
        THETA_teor[l] = np.sqrt(THETA1[l]**2+THETA2[l]**2)
    return THETA_teor

# In[ ]:


#Likelihood function
def lnlike(parameters, theta1, theta2, sigma):
    R_S, m_0, SIGMA_0, H_R, B, MASS = parameters
    THETA_teor = model(parameters, theta1, theta2, sigma)
    X = np.zeros((len(theta1)),float)
    for l in range(len(theta1)):
        X[l]=((theta[l]-THETA_teor[l])**2)/(sigma**2)
    return -0.5*np.sum(X)

# In[ ]:


#initial guess in the MCMC
start=np.zeros(6,float)
start[0] = radio_s
start[1] = M_0
start[2] = den_0
start[3] = escala_r
start[4] = height
start[5] = Mass 

# In[ ]:


#Parametric space in the MCMC
def lnprior(parameters):
    R_S, m_0, SIGMA_0, H_R, B, MASS = parameters  
    if 0.1<R_S<60 and 0.1e11<m_0<1e12 and 1e8<SIGMA_0<1e10 and 2<H_R<30 and 0.1<B<2 and 0.1e10<MASS<2e10:
#    if R_S>0 and m_0>0 and SIGMA_0>0 and H_R>0 and B>0 and MASS>0:
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
ndim, nwalkers = 6, int(input("\nEnter the number of walkers:\n"))
#initial posicion and step length
pos_step = 1e-8
pos_in = [abs(start + pos_step*start*np.random.randn(ndim)+1e-9*np.random.randn(ndim)) for i in range(nwalkers)]


# In[ ]:


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(theta1, theta2,sigma))


# In[ ]:


#Number of Steps
sampler.run_mcmc(pos_in, int(input("\nEnter the number of steps:\n")), progress=True)


# In[ ]:

fig = plt.figure()
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))

chain_steps = [i for i in range(len(sampler.chain[:,:,0].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,0].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[0]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
plb.savefig('para1.pdf')


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
plb.savefig('para2.pdf')


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
plb.savefig('para3.pdf')


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
plb.savefig('para4.pdf')

# In[ ]:

fig = plt.figure()
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))

chain_steps = [i for i in range(len(sampler.chain[:,:,4].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,4].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[4]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
plb.savefig('para5.pdf')

# In[ ]:

fig = plt.figure()
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))

chain_steps = [i for i in range(len(sampler.chain[:,:,5].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,5].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[5]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
plb.savefig('para6.pdf')


# In[ ]:


#Step of cut in the MCMC
samples = sampler.chain[:, 600:, :].reshape((-1, ndim))

# In[ ]:


percentage=0.68
#Contours of values
fig = corner.corner(samples, labels=["$r_s$", r"$mo_0$", r"$\Sigma_0$", "$h_R$", "$b$", "$M$"],
                    quantiles = [0.5-0.5*percentage, 0.5, 0.5+0.5*percentage],fill_contours=True, plot_datapoints=True)
fig.savefig("contours_J2141.pdf")

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
r_s = para[0]
m_0 = para[1]
Sigma_0 = para[2] 
h_r = para[3]
b = para[4]
M = para[5]

def POTDEFnfw1(x):
    def integ(TheTa, theta):
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=m_0, a=r_s, normalize=False)
        Sigma = NFW_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta[l]))[0]
    return x

def POTDEFnfw2(x):
    def integ(TheTa, theta):
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=m_0, a=r_s, normalize=False)
        Sigma = NFW_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta[l]))[0]
    return x

def MN1(x):
    def integ(TheTa, theta):
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=M,b=b,normalize=False)
        Sigma = MN_Bulge_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta[l]))[0]
    return x

def MN2(x):
    def integ(TheTa, theta):
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=M,b=b,normalize=False)
        Sigma = MN_Bulge_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta[l]))[0]
    return x

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

#Obteniendo gradiente del potencial deflector
GRADPOT1nfw = np.zeros((len(theta1)), float)
GRADPOT2nfw = np.zeros((len(theta1)), float)
GRADPOT1disk_exp = np.zeros((len(theta1)), float)
GRADPOT2disk_exp = np.zeros((len(theta1)), float)
GRADPOT1MN = np.zeros((len(theta1)), float)
GRADPOT2MN = np.zeros((len(theta1)), float)
GRADPOT1 = np.zeros((len(theta1)), float)
GRADPOT2 = np.zeros((len(theta1)), float)
THETA1 = np.zeros((len(theta1)), float)
THETA2 = np.zeros((len(theta1)), float)

for l in range(len(theta1)):
    GRADPOT1nfw[l]= derivative(POTDEFnfw1, theta1[l], dx=1e-9, order=7)
    GRADPOT2nfw[l]= derivative(POTDEFnfw2, theta2[l], dx=1e-9, order=7)
    GRADPOT1disk_exp[l]= derivative(POTDEFdisk_exp1, theta1[l], dx=1e-9, order=7)
    GRADPOT2disk_exp[l]= derivative(POTDEFdisk_exp2, theta2[l], dx=1e-9, order=7)
    GRADPOT1MN[l]= derivative(MN1, theta1[l], dx=1e-9, order=7)
    GRADPOT2MN[l]= derivative(MN2, theta2[l], dx=1e-9, order=7)
    GRADPOT1[l]=(SIGMA_CRIT**2)*(GRADPOT1nfw[l]+GRADPOT1disk_exp[l]+GRADPOT1MN[l])
    GRADPOT2[l]=(SIGMA_CRIT**2)*(GRADPOT2nfw[l]+GRADPOT2disk_exp[l]+GRADPOT2MN[l])

#Images obtained with initial guess
for l in range(len(theta1)):
    THETA1[l] = Beta1[l]+GRADPOT1[l]
    THETA2[l] = Beta2[l]+GRADPOT2[l]


# In[ ]:


#Graphics of source and images
fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(Beta1/FC, Beta2/FC, 'or')
plb.plot(theta1/FC, theta2/FC, 'ob')
plb.plot(THETA1/FC, THETA2/FC, 'og')
plb.xlim(-2.5,2.5)
plb.ylim(-2.5,2.5)
plb.legend(['Source', 'Observational data', 'Model values'], loc='upper right', fontsize=15)
#plb.show()
plt.savefig('fitting.pdf')

# In[ ]:


#Parameters and errors
r_s = para[0]; r_s_95pos = parap95[0]; r_s_95neg = paran95[0]; r_s_68pos = parap68[0]; r_s_68neg = paran68[0]
m_0 = para[1]; m_0_95pos = parap95[1]; m_0_95neg = paran95[1]; m_0_68pos = parap68[1]; m_0_68neg = paran68[1]
Sigma_0 = para[2]; Sigma_0_95pos = parap95[2]; Sigma_0_95neg = paran95[2]; Sigma_0_68pos = parap68[2]; Sigma_0_68neg = paran68[2]
h_r = para[3]; h_r_95pos = parap95[3]; h_r_95neg = paran95[3]; h_r_68pos = parap68[3]; h_r_68neg = paran68[3]
b = para[4]; b_95pos = parap95[4]; b_95neg = paran95[4]; b_68pos = parap68[4]; b_68neg = paran68[4]
M = para[5]; M_95pos = parap95[5]; M_95neg = paran95[5]; M_68pos = parap68[5]; M_68neg = paran68[5]

# In[ ]:

table_data = []

table_para = [r"r_s",r"m_0",r"Sigma_0", r"h_r", r"b", r"M"]
table_units = [r"Kpc",r"Solar_Mass", r"Solar_Mass/Kpc^2", r"Kpc", r"Kpc", r"Solar_Mass"]
para = [r_s, m_0, Sigma_0, h_r, b, M]
parap68=[r_s_68pos, m_0_68pos, Sigma_0_68pos, h_r_68pos, b_68pos, M_68pos]
paran68=[r_s_68neg, m_0_68neg, Sigma_0_68neg, h_r_68neg, b_68neg, M_68neg]
parap95=[r_s_95pos, m_0_95pos, Sigma_0_95pos, h_r_95pos, b_95pos, M_95pos]
paran95=[r_s_95neg, m_0_95neg, Sigma_0_95neg, h_r_95neg, b_95neg, M_95neg]
index=[r"r_s",r"m_0",r"Sigma_0", r"h_r", r"b", r"M"]

for i in range(len(para)):
	table_data.append([table_para[i], table_units[i], para[i], parap95[i], paran95[i],  parap68[i], paran68[i]])

column_name = [r"PARAMETER", r"UNITS", r"FIT", r"95%(+)", r"95%(-)",  r"68%(+)", r"68%(-)"]	
table_p = pd.DataFrame(table_data, index=index, columns=column_name)
table_p.to_csv("parameters_MCMC.txt", sep='\t', encoding='utf-8')
print ("\n#####################################################################")
print(table_p)
print ("\nDone")
print ("\n#####################################################################\n")




