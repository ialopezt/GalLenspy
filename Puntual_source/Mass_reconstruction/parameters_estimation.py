#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, TextBox # Matplotlib widgets
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


# In[2]:


print ("\n#####################################################################")
print("MCMC------GALLENSPY")

# In[ ]:

tt=Table.Table.read('coordinates.txt', format='ascii.tab') # import image data

#Import coordinates of images
theta1=tt['theta1'] 
theta2=tt['theta2']
sigma=tt['sigma']

theta=np.zeros(len(theta1),float)
for i in range(len(theta1)):
    theta[i]=np.sqrt(theta1[i]**2+theta2[i]**2)


tt=Table.Table.read('Cosmological_distances.txt', format='ascii.tab') # import cosmological distances
#Importando cosmological distances and Critical density
D_ds=tt['D_ds'][0] 
D_d=tt['D_d'][0]
D_s=tt['D_s'][0]
SIGMA_CRIT=tt['SIGMA_CRIT'][0]

tt=Table.Table.read('parameters_lens_source.txt', format='ascii.tab') # import source data

#import source center
H=tt['FIT'][0] 
K=tt['FIT'][1]

h = H*np.pi/(180*3600)
K = K*np.pi/(180*3600)
    
Beta1 = h
Beta2 = K
FC = np.pi/(180*3600) #conversion factor between arcs and radians

#Import parameters of initial guess
tt=Table.Table.read('init_guess_params.txt', format='ascii.tab') # import cosmological distances

aa=tt['a (kpc)'] 
bb=tt['b (kpc)']
masses=tt['mass']
chk=tt['checked']


# In[3]:


"""
Here the parameters associated to the selected models are defined, and also
the initial guesses are given.
"""
para_labels = []
labels = []
labels_log = []
para_in = []

if chk[0]=='True':
	para_labels.append("b1");    para_in.append(float(bb[0])); labels.append(r"$b_B$");		 labels_log.append(r"$\log(b_B)$")
	para_labels.append("amp1");  para_in.append(float(masses[0]));  labels.append(r"$M_B$");		 labels_log.append(r"$\log(M_B)$")

if chk[1]=='True':
	para_labels.append("a1");    para_in.append(float(aa[1]));		 labels.append(r"$a_B$");		 labels_log.append(r"$\log(a_B)$")
	para_labels.append("b1");    para_in.append(float(bb[1]));		 labels.append(r"$b_B$");		 labels_log.append(r"$\log(b_B)$")
	para_labels.append("amp1");  para_in.append(float(masses[1]));  labels.append(r"$M_B$");		 labels_log.append(r"$\log(M_B)$")

if chk[2]=='True':
	para_labels.append("h_r");   para_in.append(float(aa[2]));		 labels.append(r"$h_{r}$");		 labels_log.append(r"$\log(h_{r})$")	
	para_labels.append("amp4");  para_in.append(float(masses[2]));	 labels.append(r"$\Sigma_{0}$"); labels_log.append(r"$\log(\Sigma_{0})$")	

if chk[3]=='True':
	para_labels.append("a2");    para_in.append(float(aa[3])); 	 labels.append(r"$a_{TD}$");	 labels_log.append(r"$\log(a_{TD})$")
	para_labels.append("b2");    para_in.append(float(bb[3]));   	 labels.append(r"$b_{TD}$");	 labels_log.append(r"$\log(b_{TD})$")
	para_labels.append("amp2");  para_in.append(float(masses[3]));	 labels.append(r"$M_{TD}$");	 labels_log.append(r"$\log(M_{TD})$")

if chk[4]=='True':
	para_labels.append("a3");    para_in.append(float(aa[4]));		 labels.append(r"$a_{TkD}$");	 labels_log.append(r"$\log(a_{TkD})$")		
	para_labels.append("b3");    para_in.append(float(bb[4]));		 labels.append(r"$b_{TkD}$");	 labels_log.append(r"$\log(b_{TkD})$")	
	para_labels.append("amp3");  para_in.append(float(masses[4]));  labels.append(r"$M_{TkD}$");	 labels_log.append(r"$\log(M_{TkD})$")	
	
if chk[5]=='True':
	para_labels.append("a5");    para_in.append(float(aa[5]));		 labels.append(r"$a_{NFW}$");	 labels_log.append(r"$\log(a_{NFW})$")
	para_labels.append("amp5");  para_in.append(float(masses[5]));	 labels.append(r"$M_{0}$");	 labels_log.append(r"$\log(M_{0})$")
	
if chk[6]=='True':
	para_labels.append("a6");    para_in.append(float(aa[6]));		 labels.append(r"$a_{Bk}$");	 labels_log.append(r"$\log(a_{Bk})$")
	para_labels.append("amp6");  para_in.append(float(masses[6]));	 labels.append(r"$\rho_{0}$");	 labels_log.append(r"$\log(\rho_{0})$")


# In[4]:


#Functions of deflector potential

#Define deflector angle for each gravitational potential
# Burket potential
def alpha1_burket(M_0,r_s):
    GRADPOT1 = np.zeros((len(theta1)), float)
    def POTDEF1(z,TheTa1):
        TheTa = np.sqrt(TheTa1**2+theta2[l]**2)
        R = D_d*TheTa
        Burket_p = BurkertPotential(amp=M_0, a=r_s, normalize=False)
        Sigma = Burket_p.dens(R,z)
        kappa = Sigma/SIGMA_CRIT
        return (4/theta1[l])*TheTa1*kappa/SIGMA_CRIT**2
    for l in range(len(theta1)):
        GRADPOT1[l]= nquad(POTDEF1, [[0, np.inf],[0, theta1[l]]])[0]
    return GRADPOT1*SIGMA_CRIT**2

def alpha2_burket(M_0,r_s):
    GRADPOT1 = np.zeros((len(theta1)), float)
    def POTDEF1(z,TheTa2):
        TheTa = np.sqrt(TheTa2**2+theta1[l]**2)
        R = D_d*TheTa
        Burket_p = BurkertPotential(amp=M_0, a=r_s, normalize=False)
        Sigma = Burket_p.dens(R,z)
        kappa = Sigma/SIGMA_CRIT
        return (4/theta2[l])*TheTa2*kappa/SIGMA_CRIT**2
    for l in range(len(theta1)):
        GRADPOT1[l]= nquad(POTDEF1, [[0, np.inf],[0, theta2[l]]])[0]
    return GRADPOT1*(SIGMA_CRIT**2)

#Miyamoto Nagai potential
def alpha1_MN1(M1,b1):
    GRADPOT1 = np.zeros((len(theta1)), float)
    def POTDEF1(z,TheTa1):
        TheTa = np.sqrt(TheTa1**2+theta2[l]**2)
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=M1,a=0,b=b1,normalize=False)
        Sigma = MN_Bulge_p.dens(R,z)
        kappa = Sigma/SIGMA_CRIT
        return (4/theta1[l])*TheTa1*kappa/SIGMA_CRIT**2
    for l in range(len(theta1)):
        GRADPOT1[l]= nquad(POTDEF1, [[0, np.inf],[0, theta1[l]]])[0]
    return GRADPOT1*(SIGMA_CRIT**2)

def alpha2_MN1(M1,b1):
    GRADPOT1 = np.zeros((len(theta1)), float)
    def POTDEF1(z,TheTa2):
        TheTa = np.sqrt(TheTa2**2+theta1[l]**2)
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=M1,a=0,b=b1,normalize=False)
        Sigma = MN_Bulge_p.dens(R,z)
        kappa = Sigma/SIGMA_CRIT
        return (4/theta2[l])*TheTa2*kappa/SIGMA_CRIT**2
    for l in range(len(theta1)):
        GRADPOT1[l]= nquad(POTDEF1, [[0, np.inf],[0, theta2[l]]])[0]
    return GRADPOT1*(SIGMA_CRIT**2)


def alpha1_MN(M2,a2,b2):
    GRADPOT1 = np.zeros((len(theta1)), float)
    def POTDEF1(z,TheTa1):
        TheTa = np.sqrt(TheTa1**2+theta2[l]**2)
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=M2,a=a2,b=b2,normalize=False)
        Sigma = MN_Bulge_p.dens(R,z)
        kappa = Sigma/SIGMA_CRIT
        return (4/theta1[l])*TheTa1*kappa/SIGMA_CRIT**2
    for l in range(len(theta1)):
        GRADPOT1[l]= nquad(POTDEF1, [[0, np.inf],[0, theta1[l]]])[0]
    return GRADPOT1*(SIGMA_CRIT**2)

def alpha2_MN(M2,a2,b2):
    GRADPOT1 = np.zeros((len(theta1)), float)
    def POTDEF1(z,TheTa2):
        TheTa = np.sqrt(TheTa2**2+theta1[l]**2)
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=M2,a=a2,b=b2,normalize=False)
        Sigma = MN_Bulge_p.dens(R,z)
        kappa = Sigma/SIGMA_CRIT
        return (4/theta2[l])*TheTa2*kappa/SIGMA_CRIT**2
    for l in range(len(theta1)):
        GRADPOT1[l]= nquad(POTDEF1, [[0, np.inf],[0, theta2[l]]])[0]
    return GRADPOT1*(SIGMA_CRIT**2)

#NFW potential
def alpha1_NFW(M_0,r_s):
    GRADPOT1 = np.zeros((len(theta1)), float)
    def POTDEF1(z,TheTa1):
        TheTa = np.sqrt(TheTa1**2+theta2[l]**2)
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=M_0, a=r_s, normalize=False)
        Sigma = NFW_p.dens(R,z)
        kappa = Sigma/SIGMA_CRIT
        return (4/theta1[l])*TheTa1*kappa/SIGMA_CRIT**2
    for l in range(len(theta1)):
        GRADPOT1[l]= nquad(POTDEF1, [[0, np.inf],[0, theta1[l]]])[0]
    return GRADPOT1*(SIGMA_CRIT**2)

def alpha2_NFW(M_0,r_s):
    GRADPOT1 = np.zeros((len(theta1)), float)
    def POTDEF1(z,TheTa2):
        TheTa = np.sqrt(TheTa2**2+theta1[l]**2)
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=M_0, a=r_s, normalize=False)
        Sigma = NFW_p.dens(R,z)
        kappa = Sigma/SIGMA_CRIT
        return (4/theta2[l])*TheTa2*kappa/SIGMA_CRIT**2
    for l in range(len(theta1)):
        GRADPOT1[l]= nquad(POTDEF1, [[0, np.inf],[0, theta2[l]]])[0]
    return GRADPOT1*SIGMA_CRIT**2

#Exponential Disk potential
def alpha1_ExpDisk(Sigma_0,h_r):
    GRADPOT1 = np.zeros((len(theta1)), float)
    def POTDEF1(TheTa1,theta):
        TheTa = np.sqrt(TheTa1**2+theta2[l]**2)
        R = D_d*TheTa
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)
        kappa = Sigma/SIGMA_CRIT
        return (2/theta1[l])*TheTa1*kappa/SIGMA_CRIT**2
    for l in range(len(theta1)):
        GRADPOT1[l]= quad(POTDEF1, 0, theta1[l], limit=100, args=(theta[l]))[0]
    return GRADPOT1*SIGMA_CRIT**2

def alpha2_ExpDisk(Sigma_0,h_r):
    GRADPOT1 = np.zeros((len(theta1)), float)
    def POTDEF1(TheTa2,theta):
        TheTa = np.sqrt(TheTa2**2+theta1[l]**2)
        R = D_d*TheTa
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)
        kappa = Sigma/SIGMA_CRIT
        return (2/theta2[l])*TheTa2*kappa/SIGMA_CRIT**2
    for l in range(len(theta1)):
        GRADPOT1[l]= quad(POTDEF1, 0, theta2[l], limit=100, args=(theta[l]))[0]
    return GRADPOT1*SIGMA_CRIT**2


# In[5]:


def model(parameters, theta1, theta2, sigma):
    global  chk, para_labels
 
    para = {}
    GRADPOT1 = np.zeros((len(theta1)), float)
    GRADPOT2 = np.zeros((len(theta1)), float)
    THETA1 = np.zeros((len(theta1)), float)
    THETA2 = np.zeros((len(theta1)), float)
    
    for i in range(len(para_labels)):
        para[para_labels[i]] = parameters[i]

    if chk[0]=='True':
        amp1=para["amp1"]; b1=para["b1"]
        GRADPOT1_BULGE=alpha1_MN1(amp1,b1)
        GRADPOT2_BULGE=alpha2_MN1(amp1,b1)

    if chk[1]=='True':
        amp1=para["amp1"]; b1=para["b1"]; a1=para["a1"]
        GRADPOT1_BULGE=alpha1_MN(amp1,a1,b1)
        GRADPOT2_BULGE=alpha2_MN(amp1,a1,b1)
    
    if chk[2]=='True':
        amp4=para["amp4"]; h_r=para["h_r"]
        GRADPOT1_DISK = alpha1_ExpDisk(amp4,h_r)
        GRADPOT2_DISK = alpha2_ExpDisk(amp4,h_r)

    if chk[3]=='True':
        amp2=para["amp2"]; b2=para["b2"]; a2=para["a2"]
        GRADPOT1_DISK=alpha1_MN(amp2,a2,b2)
        GRADPOT2_DISK=alpha2_MN(amp2,a2,b2)

    if chk[4]=='True':
        amp3=para["amp3"]; b2=para["b3"]; a2=para["a3"]
        GRADPOT1_DISK=alpha1_MN(amp3,a2,b2)
        GRADPOT2_DISK=alpha2_MN(amp3,a2,b2)

    if chk[5]=='True':
        amp5=para["amp5"]; a5=para["a5"]
        GRADPOT1_HALO=alpha1_NFW(amp5,a5)
        GRADPOT2_HALO=alpha2_NFW(amp5,a5)

    if chk[6]=='True':
        amp6=para["amp6"]; a6=para["a6"]
        GRADPOT1_HALO=alpha1_burket(amp6,a6)
        GRADPOT2_HALO=alpha2_burket(amp6,a6)

    for l in range(len(theta1)):
        GRADPOT1[l]= GRADPOT1_BULGE[l]+GRADPOT1_DISK[l]+GRADPOT1_HALO[l]
        GRADPOT2[l]= GRADPOT2_BULGE[l]+GRADPOT2_DISK[l]+GRADPOT2_HALO[l]
        
    for l in range(len(theta1)):
        THETA1[l] = Beta1+GRADPOT1[l]
        THETA2[l] = Beta2+GRADPOT2[l]
        
    THETA_teor = np.zeros((len(theta1)), float)
    for l in range(len(theta1)):
        THETA_teor[l] = np.sqrt(THETA1[l]**2+THETA2[l]**2)
    return THETA_teor


# In[6]:


# In[ ]:

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
#Probability distributions

#ln Prior
def lnprior(parameters):

    para = {}
    
    for i in range(len(para_labels)):
        para[para_labels[i]] = parameters[i]

    booL = []

    if chk[0]=='True':
        amp1=para["amp1"]; b1=para["b1"]
#        if 0.1e10<amp1<1e10 and 0.0<b1<0.5:
        if amp1>0 and 0.0<b1<1.0:

            booL.append(True)
        else:
            booL.append(False)

    if chk[1]=='True':
        amp1=para["amp1"]; b1=para["b1"]; a1=para["a1"]
#        if 1e10<amp1<5e10 and 0.5<b1<1.5 and 0.01<a1<0.05:
        if amp1>0 and 0.0<b1<3.0 and 0.0<a1<0.1:
            booL.append(True)
        else:
            booL.append(False)
    
    if chk[2]=='True':
        amp4=para["amp4"]; h_r=para["h_r"]
#        if 1e8<amp4<15e8 and 2<h_r<6:
        if amp4>0 and 0<h_r<12:
            booL.append(True)
        else:
            booL.append(False)

    if chk[3]=='True':
        amp2=para["amp2"]; b2=para["b2"]; a2=para["a2"]
#        if 0.5e11<amp2<1.5e11 and 0.1<b2<1.0 and 1<a2<10:
        if amp2>0 and 0.0<b2<2.0 and 0<a2<20:
            booL.append(True)
        else:
            booL.append(False)

    if chk[4]=='True':
        amp3=para["amp3"]; b2=para["b3"]; a2=para["a3"]
#        if 0.5e11<amp3<1.5e11 and 0.1<b2<15 and 1<a2<10:
        if amp3>0 and 0.0<b2<30 and 0<a2<20:
            booL.append(True)
        else:
            booL.append(False)

    if chk[5]=='True':
        amp5=para["amp5"]; a5=para["a5"]
#        if 0.1e11<amp5<10e11 and 0.1<a5<30:
        if amp5>0 and 0.0<a5<60:
            booL.append(True)
        else:
            booL.append(False)

    if chk[6]=='True':
        amp6=para["amp6"]; a6=para["a6"]
#        if 0.1e6<amp6<10e6 and 2<a6<38:
        if amp6>0 and 0<a6<76:
            booL.append(True)
        else:
            booL.append(False)

    if False in booL:
        return -np.inf
    else:
        return 0.0

#Likelihood function

def lnlike(parameters, theta1, theta2, sigma):
    THETA_teor = model(parameters, theta1, theta2, sigma)
    X = np.zeros((len(theta1)),float)
    for l in range(len(theta1)):
        X[l]=((theta[l]-THETA_teor[l])**2)/(sigma[l]**2)
    return -0.5*np.sum(X)

#Probability function
def lnprob(parameters, theta1, theta2, sigma):
    lp = lnprior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(parameters, theta1, theta2, sigma)


# In[7]:


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
# Dimension

start = np.array(para_in)
ndim = len(start)
print ("Dimension: ", ndim, "\n")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Nwalkers and Steps

nwalkers = int(input("\nEnter the number of walkers you want to use:\n"))
steps = int(input("\nEnter the number of steps you want the walkers to take:\n"))


# In[8]:


#initial posicion and step length
pos_step = 1e-8
pos_in = [abs(start + pos_step*start*np.random.randn(ndim)+1e-9*np.random.randn(ndim)) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(theta1, theta2,sigma))


# In[9]:


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(theta1, theta2,sigma))
#Number of Steps
sampler.run_mcmc(pos_in, steps, progress=True)


# In[10]:


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here we plot the chains for each parameter

fig = plt.figure(2)
ax = fig.add_axes((0.15, 0.3, 0.75, 0.6))

chain_steps = [i for i in range(len(sampler.chain[:,:,0].T))]
chain_W = []
for i in range(nwalkers):
	chain_value = sampler.chain[:,:,0].T[:][:,i]
	ax.plot(chain_steps, chain_value, '-', color='k', alpha=0.3)
ax.plot(chain_steps, len(chain_steps)*[start[0]], '-', color='r', lw=1)
ax.set_xlim(0, len(chain_steps)-1)
ax.set_xlabel(r"$Steps$", fontsize = 10)
ax.set_ylabel(labels[0], fontsize = 15)


class Index(object):

	ind = 0
 
	def next(self, event):
		global ndim, start, chain_W, nwalkers, chain_steps
		

		self.ind += 1
		if self.ind >= ndim:
			self.ind = 0	
		ax.clear()
		#plt.subplots_adjust(bottom=0.2)	
		for i in range(nwalkers):
			data_a = np.array(sampler.chain[:,:,self.ind].T)[:,i]	
			ax.plot(chain_steps, data_a, '-', color='k', alpha=0.3)
			ax.plot(chain_steps, len(chain_steps)*[start[self.ind]], '-', color='r', lw=1)
		ax.set_xlim(0, len(chain_steps)-1)
		ax.set_xlabel(r"$Steps$", fontsize = 10)
		ax.set_ylabel(labels[self.ind], fontsize = 15)
		plt.tight_layout()
		plt.draw()

	def prev(self, event):
		global ndim, start, chain_W, nwalkers, chain_steps
		

		self.ind -= 1
		if self.ind == -1:
			self.ind = ndim-1
			
		ax.clear()
		#plt.subplots_adjust(bottom=0.2)	
		for i in range(nwalkers):
			data_a = np.array(sampler.chain[:,:,self.ind].T)[:,i]	
			ax.plot(chain_steps, data_a, '-', color='k', alpha=0.3)
			ax.plot(chain_steps, len(chain_steps)*[start[self.ind]], '-', color='r', lw=1)
		ax.set_xlim(0, len(chain_steps)-1)
		ax.set_xlabel(r"$Steps$", fontsize = 10)
		ax.set_ylabel(labels[self.ind], fontsize = 15)
		plt.tight_layout()
		plt.draw()
		
axcolor="lavender"
callback = Index()
axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next', color=axcolor)
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous', color=axcolor)
bprev.on_clicked(callback.prev)

def burn(event):
    plt.close()


resetax = fig.add_axes((0.45, 0.05, 0.1, 0.075))
button_reset = Button(resetax, 'Burn-in', color=axcolor)
button_reset.on_clicked(burn)

plt.show()


# In[11]:


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here we plot the region of confidence

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Nwalkers and Steps

burn_in = int(input("Enter the cut step:\n"))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


print ("\n#####################################################################\n")
print ("Plotting...")

if burn_in == 0.:
	samples = sampler.chain[:, :, :].reshape((-1, ndim))
else:
	samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
	
samples.shape

percentage = 0.68

fig = corner.corner(np.log10(samples), labels=labels_log, label_kwargs = {"fontsize": 21.5},
					  bins=50, use_math_text =True, color = "gray", max_n_ticks=3,#truth_color = "red",   truths= np.log10(start),             
					  smooth=1., levels=[1-np.exp(-0.5), 1-np.exp(-2.) ], quantiles = [0.5-0.5*percentage, 0.5, 0.5+0.5*percentage], 
                      fill_contours=True, plot_datapoints=True)

axes = np.array(fig.axes).reshape((ndim, ndim))

for yi in range(ndim):
	for xi in range(yi+1):
		ax = axes[yi, xi]
		ax.tick_params(axis='both', which='major', labelsize=14.5, pad=3, direction = "in")
		
fig.savefig("Conf_Regions.pdf",bbox_inches='tight',pad_inches=0.15)


# In[14]:


table_data = []
index = []
para = []
parap68=[]; paran68=[]
parap95=[]; paran95=[]
table_para = []
table_units = []
final_para_labels = []
fit_para = []

def M_MN1(MN_b,MN_M):
    def integ(z, TheTa1, TheTa2):
        TheTa = np.sqrt(TheTa1**2+TheTa2**2)
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=MN_M,a=0,b=MN_b,normalize=False)
        Densidad = MN_Bulge_p.dens(R,z)
        Kappa = 2*Densidad
        return Kappa/(SIGMA_CRIT**2)
    A=nquad(integ, [[0, np.inf], [0, 2*lim],[0, 2*lim]])[0]
    return A*(D_d**2)*(SIGMA_CRIT**2)  

def M_MN2(MN_a,MN_b,MN_M):
    def integ(z, TheTa1, TheTa2):
        TheTa = np.sqrt(TheTa1**2+TheTa2**2)
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=MN_M,a=MN_a,b=MN_b,normalize=False)
        Densidad = MN_Bulge_p.dens(R,z)
        Kappa = 2*Densidad
        return Kappa/(SIGMA_CRIT**2)
    A=nquad(integ, [[0, np.inf], [0, 2*lim],[0, 2*lim]])[0]
    return A*(D_d**2)*(SIGMA_CRIT**2)  

def M_NFW(nfw_a,nfw_M0):
    def integ(z, TheTa1, TheTa2):
        TheTa = np.sqrt(TheTa1**2+TheTa2**2)
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=nfw_M0, a=nfw_a, normalize=False)
        Densidad = NFW_p.dens(R,z)
        Kappa = 2*Densidad
        return Kappa/(SIGMA_CRIT**2)
    A=nquad(integ, [[0, np.inf], [0, 2*lim],[0, 2*lim]])[0]
    return A*(D_d**2)*(SIGMA_CRIT**2)  

def M_burk(b_a,b_M0):
    def integ(z, TheTa1, TheTa2):
        TheTa = np.sqrt(TheTa1**2+TheTa2**2)
        R = D_d*TheTa
        Burket_p = BurkertPotential(amp=b_M0, a=b_a, normalize=False)
        Densidad = Burket_p.dens(R,z)
        Kappa = 2*Densidad
        return Kappa/(SIGMA_CRIT**2)
    A=nquad(integ, [[0, np.inf], [0, 2*lim],[0, 2*lim]])[0]
    return A*(D_d**2)*(SIGMA_CRIT**2)  

def M_disk(Sigma_0,h_r):
    def integ(TheTa1, TheTa2):
        TheTa = np.sqrt(TheTa1**2+TheTa2**2)
        Densidad = Sigma_0*np.exp(-D_d*TheTa/h_r)
        Kappa = Densidad
        return Kappa/(SIGMA_CRIT**2)
    A=nquad(integ, [[0, 2*lim],[0, 2*lim]])[0]
    return A*(D_d**2)*(SIGMA_CRIT**2)  

for i in range(ndim):
    mcmc = np.percentile(samples[:, i], [50.-0.5*95, 50.-0.5*68, 50., 50.+0.5*68, 50.+0.5*95])
    para.append(mcmc[2])
    fit_para.append(mcmc[2]) 
    parap68.append(mcmc[3]-mcmc[2])
    paran68.append(mcmc[2]-mcmc[1])
    parap95.append(mcmc[4]-mcmc[2])
    paran95.append(mcmc[2]-mcmc[0])
    final_para_labels.append(para_labels[i])

	#MN1 y MN2 BULGE
    if para_labels[i]=="b1":
        MN_b = np.array(samples[:, i])
    if para_labels[i]=="amp1":
        MN_M = np.array(samples[:, i])
        if para_labels[i]=="a1":
            MN_a = np.array(samples[:, i])
            radius=float(input("\nEnter the radius of the enclosed mass in arcs for the bulge:\n"))
            lim = radius*FC
            M_bulge=np.zeros(len(MN_b),float)
            for j in range(len(MN_b)):		
                M_bulge[j]=M_MN2(MN_a[j],MN_b[j],MN_M[j])
        else:
            radius=float(input("\nEnter the radius of the enclosed mass in arcs for the bulge:\n"))
            lim = radius*FC
            M_bulge=np.zeros(len(MN_b),float)
            for j in range(len(MN_b)):		
                M_bulge[j]=M_MN1(MN_b[j],MN_M[j])
        mcmc = np.percentile(M_bulge, [50.-0.5*95, 50.-0.5*68, 50., 50.+0.5*68, 50.+0.5*95])
        para.append(mcmc[2])
        parap68.append(mcmc[3]-mcmc[2])
        paran68.append(mcmc[2]-mcmc[1])
        parap95.append(mcmc[4]-mcmc[2])
        paran95.append(mcmc[2]-mcmc[0])
        final_para_labels.append("M_bulge")  


	#MN1 Disk
    if para_labels[i]=="a2":
        MN_a = np.array(samples[:, i])
    if para_labels[i]=="amp2":
        MN_M = np.array(samples[:, i])
    if para_labels[i]=="b2":
        MN_b = np.array(samples[:, i])
        radius=float(input("\nEnter the radius of the enclosed mass in arcs for the bulge:\n"))
        lim = radius*FC
        M_Disk=np.zeros(len(MN_a),float)
        for j in range(len(MN_a)):		
            M_Disk[j]=M_MN2(MN_a[j],MN_b[j],MN_M[j])
        mcmc = np.percentile(M_Disk, [50.-0.5*95, 50.-0.5*68, 50., 50.+0.5*68, 50.+0.5*95])
        para.append(mcmc[2])
        parap68.append(mcmc[3]-mcmc[2])
        paran68.append(mcmc[2]-mcmc[1])
        parap95.append(mcmc[4]-mcmc[2])
        paran95.append(mcmc[2]-mcmc[0])
        final_para_labels.append("M_disk")  

	#MN2 Disk
    if para_labels[i]=="a3":
        MN_a = np.array(samples[:, i])
    if para_labels[i]=="amp3":
        MN_M = np.array(samples[:, i])
    if para_labels[i]=="b3":
        MN_b = np.array(samples[:, i])
        radius=float(input("\nEnter the radius of the enclosed mass in arcs for the bulge:\n"))
        lim = radius*FC
        M_Disk=np.zeros(len(MN_a),float)
        for j in range(len(MN_a)):		
            M_Disk[j]=M_MN2(MN_a[j],MN_b[j],MN_M[j])
        mcmc = np.percentile(M_Disk, [50.-0.5*95, 50.-0.5*68, 50., 50.+0.5*68, 50.+0.5*95])
        para.append(mcmc[2])
        parap68.append(mcmc[3]-mcmc[2])
        paran68.append(mcmc[2]-mcmc[1])
        parap95.append(mcmc[4]-mcmc[2])
        paran95.append(mcmc[2]-mcmc[0])
        final_para_labels.append("M_disk")  

	#Exponential Disc
    if para_labels[i]=="h_r":
        ed_h_r = np.array(samples[:, i])
    if para_labels[i]=="amp4":
        ed_sigma0 = np.array(samples[:, i])
        radius=float(input("\nEnter the radius of the enclosed mass in arcs for the disk:\n"))
        lim = radius*FC
        M_disc=np.zeros(len(ed_h_r),float)
        for j in range(len(ed_h_r)):		
            M_disc[j]=M_disk(ed_sigma0[j],ed_h_r[j])
        mcmc = np.percentile(M_disc, [50.-0.5*95, 50.-0.5*68, 50., 50.+0.5*68, 50.+0.5*95])
        para.append(mcmc[2])
        parap68.append(mcmc[3]-mcmc[2])
        paran68.append(mcmc[2]-mcmc[1])
        parap95.append(mcmc[4]-mcmc[2])
        paran95.append(mcmc[2]-mcmc[0])
        final_para_labels.append("M_disc")
	
	#NFW
    if para_labels[i]=="a5":
        nfw_a = np.array(samples[:, i])
    if para_labels[i]=="amp5":
        nfw_M0 = np.array(samples[:, i])
        radius=float(input("\nEnter the radius of the enclosed mass in arcs for the halo:\n"))
        lim = radius*FC
        M_halo=np.zeros(len(nfw_a),float)
        for j in range(len(nfw_a)):		
            M_halo[j]=M_NFW(nfw_a[j],nfw_M0[j])
        mcmc = np.percentile(M_halo, [50.-0.5*95, 50.-0.5*68, 50., 50.+0.5*68, 50.+0.5*95])
        para.append(mcmc[2])
        parap68.append(mcmc[3]-mcmc[2])
        paran68.append(mcmc[2]-mcmc[1])
        parap95.append(mcmc[4]-mcmc[2])
        paran95.append(mcmc[2]-mcmc[0])
        final_para_labels.append("M_halo")  
	
	#Burkert
    if para_labels[i]=="a6":
        b_a = np.array(samples[:, i])
    if para_labels[i]=="amp6":
        b_M0 = np.array(samples[:, i])
        radius=float(input("\nEnter the radius of the enclosed mass in arcs for the halo:\n"))
        lim = radius*FC		
        M_halo=np.zeros(len(b_a),float)
        for j in range(len(b_a)):		
            M_halo[j]=M_burk(b_a[j],b_M0[j])
        mcmc = np.percentile(M_halo, [50.-0.5*95, 50.-0.5*68, 50., 50.+0.5*68, 50.+0.5*95])
        para.append(mcmc[2])
        parap68.append(mcmc[3]-mcmc[2])
        paran68.append(mcmc[2]-mcmc[1])
        parap95.append(mcmc[4]-mcmc[2])
        paran95.append(mcmc[2]-mcmc[0])
        final_para_labels.append("M_halo")

best_para = {}
    
for i in range(len(final_para_labels)):
    best_para[final_para_labels[i]] = para[i]

if chk[0]=='True':
    amp1=best_para["amp1"]; b1=best_para["b1"]
    GRADPOT1_BULGE=alpha1_MN1(amp1,b1)
    GRADPOT2_BULGE=alpha2_MN1(amp1,b1)

if chk[1]=='True':
    amp1=best_para["amp1"]; b1=best_para["b1"]; a1=best_para["a1"]
    GRADPOT1_BULGE=alpha1_MN(amp1,a1,b1)
    GRADPOT2_BULGE=alpha2_MN(amp1,a1,b1)

if chk[2]=='True':
    amp4=best_para["amp4"]; h_r=best_para["h_r"]
    GRADPOT1_DISK = alpha1_ExpDisk(amp4,h_r)
    GRADPOT2_DISK = alpha2_ExpDisk(amp4,h_r)

if chk[3]=='True':
    amp2=best_para["amp2"]; b2=best_para["b2"]; a2=best_para["a2"]
    GRADPOT1_DISK=alpha1_MN(amp2,a2,b2)
    GRADPOT2_DISK=alpha2_MN(amp2,a2,b2)

if chk[4]=='True':
    amp3=best_para["amp3"]; b2=best_para["b3"]; a2=best_para["a3"]
    GRADPOT1_DISK=alpha1_MN(amp3,a2,b2)
    GRADPOT2_DISK=alpha2_MN(amp3,a2,b2)

if chk[5]=='True':
    amp5=best_para["amp5"]; a5=best_para["a5"]
    GRADPOT1_HALO=alpha1_NFW(amp5,a5)
    GRADPOT2_HALO=alpha2_NFW(amp5,a5)

if chk[6]=='True':
    amp6=best_para["amp6"]; a6=best_para["a6"]
    GRADPOT1_HALO=alpha1_burket(amp6,a6)
    GRADPOT2_HALO=alpha2_burket(amp6,a6)

GRADPOT1 = np.zeros((len(theta1)), float)
GRADPOT2 = np.zeros((len(theta1)), float)
THETA1 = np.zeros((len(theta1)), float)
THETA2 = np.zeros((len(theta1)), float)

for l in range(len(theta1)):
    GRADPOT1[l]= GRADPOT1_BULGE[l]+GRADPOT1_DISK[l]+GRADPOT1_HALO[l]
    GRADPOT2[l]= GRADPOT2_BULGE[l]+GRADPOT2_DISK[l]+GRADPOT2_HALO[l]

for l in range(len(theta1)):
    THETA1[l] = Beta1+GRADPOT1[l]
    THETA2[l] = Beta2+GRADPOT2[l]


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
plb.xlabel(r"$\theta_1$", fontsize = 15)
plb.ylabel(r"$\theta_2$", fontsize = 15)
plt.savefig('fitting.pdf')
    


# In[19]:


print ("\n#####################################################################\n")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here we construct the table with the final results

if chk[0]=='True':
    index.append(r"BULGE");    index.append(r"---"); index.append(r"---")
    table_para.append(r"b");   table_units.append(r"kpc")
    table_para.append(r"M");   table_units.append(r"M_Sun")
    table_para.append(r"Mass_bulge");    table_units.append(r"M_Sun")

if chk[1]=='True':
    index.append(r"BULGE");    index.append(r"---"); index.append(r"---"); index.append(r"---")
    table_para.append(r"a");   table_units.append(r"kpc")
    table_para.append(r"b");   table_units.append(r"kpc")
    table_para.append(r"M");   table_units.append(r"M_Sun")
    table_para.append(r"Mass_bulge");    table_units.append(r"M_Sun")

if chk[2]=='True':
    index.append(r"EXPONENTIAL DISK"); index.append(r"---"); index.append(r"---")
    table_para.append(r"h_r");      table_units.append(r"kpc")
    table_para.append(r"Sigma_0");  table_units.append(r"M_Sun/pc^2")
    table_para.append(r"Mass_disk");  table_units.append(r"M_Sun")
    
if chk[3]=='True':
    index.append(r"THIN DISK"); index.append(r"---"); index.append(r"---"); index.append(r"---")
    table_para.append(r"a");    table_units.append(r"kpc")
    table_para.append(r"b");    table_units.append(r"kpc")
    table_para.append(r"M");    table_units.append(r"M_Sun")
    table_para.append(r"Mass_disk");    table_units.append(r"M_Sun")

if chk[4]=='True':
    index.append(r"THICK DISK"); index.append(r"---"); index.append(r"---"); index.append(r"---")
    table_para.append(r"a");    table_units.append(r"kpc")
    table_para.append(r"b");    table_units.append(r"kpc")
    table_para.append(r"M");    table_units.append(r"M_Sun")
    table_para.append(r"Mass_disk");    table_units.append(r"M_Sun")

if chk[5]=='True':
    index.append(r"NFW HALO"); index.append(r"---"); index.append(r"---")
    table_para.append(r"a");    table_units.append(r"kpc")
    table_para.append(r"M_0");  table_units.append(r"M_Sun")
    table_para.append(r"Mass_halo");  table_units.append(r"M_Sun")

if chk[6]=='True':
    index.append(r"BURKERT HALO"); index.append(r"---"); index.append(r"---")
    table_para.append(r"a");        table_units.append(r"kpc")
    table_para.append(r"rho_0");    table_units.append(r"M_Sun/kpc^3")
    table_para.append(r"Mass_halo");  table_units.append(r"M_Sun")

for i in range(len(para)):
    table_data.append([table_para[i], table_units[i], paran95[i], paran68[i], para[i], parap68[i], parap95[i]])

column_name = [r"PARAMETER", r"UNITS", r"95%(-)", r"68%(-)", r"FIT", r"68%(+)", r"95%(+)"]	
table_p = pd.DataFrame(table_data, index=index, columns=column_name)
table_p.to_csv("final_params.txt", sep='\t', encoding='utf-8')
print (table_p)
print ("\n#####################################################################")
print ("\nDone")
print ("\n#####################################################################\n")






