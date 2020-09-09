#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.misc import *
import numpy as np
import pylab as plb
from scipy.misc import *
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import nquad
from scipy.misc import derivative
import emcee
import corner
import pandas as pd
from astropy import table as Table # For fast and easy reading / writing with tables using numpy library


# In[2]:


theta_ra = [0, 1.4743, 2.4664, 0.9378]
theta_dec = [0, 0.5518, -0.6022, -1.6160]

Lente = [1.1706, -0.5665]

# In[3]:


theta_1 = np.zeros(len(theta_ra), float)
theta_2 = np.zeros(len(theta_dec), float)

for i in range(len(theta_1)):
    theta_1[i] = theta_ra[i]-Lente[0]
    theta_2[i] = theta_dec[i]-Lente[1]


# In[4]:

FC=np.pi/(180*3600)

theta1 = theta_1*np.pi/(180*3600)
theta2 = theta_2*np.pi/(180*3600)
theta = np.zeros(len(theta1),float)
for i in range(len(theta1)):
    theta[i]=np.sqrt(theta1[i]**2+theta2[i]**2)

# In[5]:

fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(theta1/FC, theta2/FC, 'ob')
plb.xlim(-5,5)
plb.ylim(-5,5)
#plb.show()
plb.savefig('Datos.pdf')


# In[6]:


D_ds = 1070.324e3
D_d = 1163.342e3
D_s = 1700.487e3
d_ds = D_ds*1e3*3.086e16
d_d = D_d*1e3*3.086e16
d_s = D_s*1e3*3.086e16
c = 3e8
G = 6.67e-11
SIGMA_CRIT = (c**2)*d_s/(4*np.pi*G*d_d*d_ds) #Sigma crítico para la convergencia en kg/m^2\n",
SIGMA_CRIT = SIGMA_CRIT*5.027e-31*1e6/((3.241e-17)**2)

# In[13]:

#Espacio paramétrico
N2 = 4
r_s = np.linspace(0.1, 30, N2) #radio de escala del NFW\n",
M_0 = np.linspace(0.1e11, 10e11, N2) #Masa central del NFW\n",
rho_0 = np.zeros((N2,N2), float) #Establendiendo valores de la densidad central del NFW\n",
for i in range(N2):
    for j in range(N2):
        rho_0[i,j]=(M_0[i]/(4*np.pi*(r_s[j]**3)))
#Delimitando valores para el disco exponencial\n",
h_r = np.linspace(2, 6, N2)
Sigma_0 = np.linspace(1e8, 15e8, N2)

#Delimitando la posición de la fuente
BETA1 = 0
BETA2 = 0

Sigma1 = [0.0001, 0.0004, 0.0003, 0.0005]
Sigma2 = [0.0001, 0.0006, 0.0013, 0.0006]

sigma1 = np.zeros(len(Sigma1), float)
sigma2 = np.zeros(len(Sigma1), float)

for i in range(len(Sigma1)):
    sigma1[i] = Sigma1[i]*FC
    sigma2[i] = Sigma2[i]*FC

sigma = np.zeros(len(Sigma1), float)
for i in range(len(Sigma1)):
    sigma[i]=np.sqrt(sigma1[i]**2+sigma2[i]**2)

# In[14]:


#función del potencial deflector

def POTDEFnfw1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = rho_0[i,j]/((r/r_s[j])*((1+(r/r_s[j]))**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def POTDEFnfw2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = rho_0[i,j]/((r/r_s[j])*((1+(r/r_s[j]))**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def POTDEFdisk_exp1(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0[m]*np.exp(-D_d*TheTa/h_r[n])/2 #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0[m]*np.exp(-D_d*TheTa/h_r[n])/2 #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x


# In[9]:

#valores iniciales del gradiente del potencial deflector
X = np.zeros((N2,N2,N2,N2,N2,N2,len(theta1)), float)
GRADPOT1nfw = np.zeros((N2,N2,len(theta1)), float)
GRADPOT2nfw = np.zeros((N2,N2,len(theta1)), float)
GRADPOT1disk_exp = np.zeros((N2,N2,len(theta1)), float)
GRADPOT2disk_exp = np.zeros((N2,N2,len(theta1)), float)
GRADPOT1 = np.zeros((N2,N2,N2,N2,len(theta1)), float)
GRADPOT2 = np.zeros((N2,N2,N2,N2,len(theta1)), float)
THETA1 = np.zeros((N2,N2,N2,N2,len(theta1)), float)
THETA2 = np.zeros((N2,N2,N2,N2,len(theta1)), float)
THETA = np.zeros((N2,N2,N2,N2,len(theta1)), float)
L = np.zeros((N2,N2,N2,N2), float)

#Obteniendo el gradiente del potencial deflector y la función de minimización junto con cada likelihood
for i in range(N2):
    for j in range(N2):
        for l in range(len(theta1)):
            GRADPOT1nfw[i,j,l]= derivative(POTDEFnfw1, theta1[l], dx=1e-9, order=7)
            GRADPOT2nfw[i,j,l]= derivative(POTDEFnfw2, theta2[l], dx=1e-9, order=7)
            print(i,j,l)


# In[10]:


for m in range(N2):
    for n in range(N2):
        for l in range(len(theta1)):
            GRADPOT1disk_exp[m,n,l]= derivative(POTDEFdisk_exp1, theta1[l], dx=1e-9, order=7)
            GRADPOT2disk_exp[m,n,l]= derivative(POTDEFdisk_exp2, theta2[l], dx=1e-9, order=7)
            print(m,n,l)


# In[16]:


for i in range(N2):
    for j in range(N2):
        for m in range(N2):
            for n in range(N2):
                for l in range(len(theta1)):
                    GRADPOT1[i,j,m,n,l]=2*(SIGMA_CRIT**2)*(GRADPOT1nfw[i,j,l]+GRADPOT1disk_exp[m,n,l])
                    GRADPOT2[i,j,m,n,l]=2*(SIGMA_CRIT**2)*(GRADPOT2nfw[i,j,l]+GRADPOT2disk_exp[m,n,l])
                    print(i,j,m,n,l)


# In[17]:


for i in range(N2):
    for j in range(N2):
        for m in range(N2):
            for n in range(N2):
                for t in range(len(theta1)):
                    THETA1[i,j,m,n,t] = BETA1+GRADPOT1[i,j,m,n,t]
                    THETA2[i,j,m,n,t] = BETA2+GRADPOT2[i,j,m,n,t]
                    THETA[i,j,m,n,t] = np.sqrt(THETA1[i,j,m,n,t]**2+THETA2[i,j,m,n,t]**2)
                    X[i,j,m,n,t] = ((theta[t]-THETA[i,j,m,n,t])**2)/(sigma[t]**2)
                    print(i,j,m,n,t)
                L[i,j,m,n] = np.sum(X[i,j,m,n])

# In[20]:


LIKE = np.zeros((N2,N2,N2), float)
like = np.zeros((N2,N2), float)
lik = np.zeros(N2,float)

Minim = []
DEN_0 = []
RADIO_ESCALA = []
Den_0 = []
Radio_Escala = []

for i in range(N2):
    for j in range(N2):
        for m in range(N2):
            for n in range(N2):
                LIKE[i,j,m]=min(L[i,j,m])
                like[i,j]=min(LIKE[i,j])
                lik[i]=min(like[i])

for i in range(N2):
    for j in range(N2):
        for m in range(N2):
            for n in range(N2):
                for t in range(len(lik)):
                    if L[i,j,m,n]==lik[t]:
                        Minim.append(L[i,j,m,n])
                        DEN_0.append(M_0[i])
                        RADIO_ESCALA.append(r_s[j])
                        Den_0.append(Sigma_0[m])
                        Radio_Escala.append(h_r[n])

for i in range(len(Minim)):
    if Minim[i]==min(Minim):
        Likehood = Minim[i]
        radio_s = RADIO_ESCALA[i]
        M_0 = DEN_0[i]
        escala_r = Radio_Escala[i] 
        den_0 = Den_0[i] 

print(Likehood, radio_s, M_0, escala_r, den_0)

# In[22]:


def POTDEFnfw1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad_0=(M_0/(4*np.pi*(radio_s**3)))
        densidad = densidad_0/((r/radio_s)*((1+(r/radio_s))**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def POTDEFnfw2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad_0=(M_0/(4*np.pi*(radio_s**3)))
        densidad = densidad_0/((r/radio_s)*((1+(r/radio_s))**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def POTDEFdisk_exp1(x):
    def integ(TheTa, theta):
        Sigma = den_0*np.exp(-D_d*TheTa/escala_r)/2 #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, theta):
        Sigma = den_0*np.exp(-D_d*TheTa/escala_r)/2 #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x


#Obteniendo gradiente del potencial deflector
GRADPOT1nfw = np.zeros((len(theta1)), float)
GRADPOT2nfw = np.zeros((len(theta1)), float)
GRADPOT1disk_exp = np.zeros((len(theta1)), float)
GRADPOT2disk_exp = np.zeros((len(theta1)), float)
GRADPOT1 = np.zeros((len(theta1)), float)
GRADPOT2 = np.zeros((len(theta1)), float)
THETA1 = np.zeros((len(theta1)), float)
THETA2 = np.zeros((len(theta1)), float)

for l in range(len(theta1)):
    GRADPOT1nfw[l]= derivative(POTDEFnfw1, theta1[l], dx=1e-9, order=7)
    GRADPOT2nfw[l]= derivative(POTDEFnfw2, theta2[l], dx=1e-9, order=7)
    GRADPOT1disk_exp[l]= derivative(POTDEFdisk_exp1, theta1[l], dx=1e-9, order=7)
    GRADPOT2disk_exp[l]= derivative(POTDEFdisk_exp2, theta2[l], dx=1e-9, order=7)
    GRADPOT1[l]=2*(SIGMA_CRIT**2)*(GRADPOT1nfw[l]+GRADPOT1disk_exp[l])
    GRADPOT2[l]=2*(SIGMA_CRIT**2)*(GRADPOT2nfw[l]+GRADPOT2disk_exp[l])

#Obteniendo las coordenadas de las imágenes para los parámetros obtenidos
for l in range(len(theta1)):
    THETA1[l] = BETA1+GRADPOT1[l]
    THETA2[l] = BETA2+GRADPOT2[l]

# In[23]:


#Graficando la fuente, las imágenes observadas y las de los parámetros
fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(BETA1/FC, BETA2/FC, 'or')
plb.plot(theta1/FC, theta2/FC, 'ob')
plb.plot(THETA1/FC, THETA2/FC, 'og')
plb.xlim(-5,5)
plb.ylim(-5,5)
#plb.show()
plb.savefig('Guess_inicial.pdf')

# In[24]:


#Definir modelo
def model(parameters, theta1, theta2, sigma):
    R_S, m_0, SIGMA_0, H_R = parameters
    def POTDEFnfw1(x):
        def integ(z,TheTa):
            R = D_d*TheTa
            r = np.sqrt(R**2+z**2)
            RHO_0 = m_0/(4*np.pi*(R_S**3))
            densidad = RHO_0/((r/R_S)*((1+(r/R_S))**2)) #Densidad volumétrica de masa
            return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
        THETA = np.sqrt(x**2 + theta2[l]**2)
        x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
        return x
    def POTDEFnfw2(x):
        def integ(z,TheTa):
            R = D_d*TheTa
            r = np.sqrt(R**2+z**2)
            RHO_0 = m_0/(4*np.pi*(R_S**3))
            densidad = RHO_0/((r/R_S)*((1+(r/R_S))**2)) #Densidad volumétrica de masa
            return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
        THETA = np.sqrt(x**2 + theta1[l]**2)
        x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
        return x
    def POTDEFdisk_exp1(x):
        def integ(TheTa, theta):   
            Sigma = SIGMA_0*np.exp(-D_d*TheTa/H_R)/2 #Densidad superficial de masa
            return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
        THETA = np.sqrt(x**2 + theta2[l]**2)
        x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
        return x
    def POTDEFdisk_exp2(x):
        def integ(TheTa, theta):
            Sigma = SIGMA_0*np.exp(-D_d*TheTa/H_R)/2 #Densidad superficial de masa
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
    for l in range(len(theta1)):
        GRADPOT1nfw[l]= derivative(POTDEFnfw1, theta1[l], dx=1e-9, order=7)
        GRADPOT2nfw[l]= derivative(POTDEFnfw2, theta2[l], dx=1e-9, order=7)
    for l in range(len(theta1)):
        GRADPOT1disk_exp[l]= derivative(POTDEFdisk_exp1, theta1[l], dx=1e-9, order=7)
        GRADPOT2disk_exp[l]= derivative(POTDEFdisk_exp2, theta2[l], dx=1e-9, order=7)
    for l in range(len(theta1)):
        GRADPOT1[l]=2*(SIGMA_CRIT**2)*(GRADPOT1nfw[l]+GRADPOT1disk_exp[l])
        GRADPOT2[l]=2*(SIGMA_CRIT**2)*(GRADPOT2nfw[l]+GRADPOT2disk_exp[l])
    for l in range(len(theta1)):
        THETA1[l] = BETA1+GRADPOT1[l]
        THETA2[l] = BETA2+GRADPOT2[l]
    THETA_teor = np.zeros((len(theta1)), float)
    for l in range(len(theta1)):
        THETA_teor[l] = np.sqrt(THETA1[l]**2+THETA2[l]**2)
    return THETA_teor

# In[25]:


#Función de likelihood
def lnlike(parameters, theta1, theta2, sigma):
    R_S, m_0, SIGMA_0, H_R = parameters
    THETA_teor = model(parameters, theta1, theta2, sigma)
    X = np.zeros((len(theta1)),float)
    for l in range(len(theta1)):
        X[l]=((theta[l]-THETA_teor[l])**2)/(sigma[l]**2)
    return -0.5*np.sum(X)

# In[26]:


start=np.zeros(4,float)
start[0] = radio_s
start[1] = M_0
start[2] = den_0
start[3] = escala_r

def lnprior(parameters):
    R_S, m_0, SIGMA_0, H_R = parameters  
    if 0.1<R_S<60 and 0.1e11<m_0<20e11 and 0.5e8<SIGMA_0<30e8 and 1<H_R<12:
        return 0.0
    return -np.inf

# In[37]:


#Función de probabilidad
def lnprob(parameters, theta1, theta2, sigma):
    lp = lnprior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(parameters, theta1, theta2, sigma)

# In[38]:


#Dimensión y caminantes
ndim, nwalkers = 4, 100
#Estableciendo posición incial y longitud del paso
pos_step = 1e-8
pos_in = [abs(start + pos_step*start*np.random.randn(ndim)+1e-9*np.random.randn(ndim)) for i in range(nwalkers)]


# In[39]:


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(theta1, theta2,sigma))


# In[40]:

sampler.run_mcmc(pos_in, 1000, progress=True)
    

# In[47]:

samples = sampler.chain[:, 800:, :].reshape((-1, ndim))

# In[49]:

percentage=0.68

fig = corner.corner(samples, labels=["$r_s$", r"$m_0$", r"$\Sigma_0$", r"$h_r$"],
                    quantiles = [0.5-0.5*percentage, 0.5, 0.5+0.5*percentage],fill_contours=True, plot_datapoints=True)
fig.savefig("triangle.pdf")


# In[50]:


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


# In[52]:


#Visualización de los parámetros en la imágen
r_s = para[0]
m_0 = para[1]
Sigma_0 = para[2] 
h_r = para[3]

#Visualización de los parámetros en la imágen

def POTDEFnfw1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad_0 = m_0/(4*np.pi*(r_s**3))
        densidad = densidad_0/((r/r_s)*((1+(r/r_s))**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def POTDEFnfw2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad_0 = m_0/(4*np.pi*(r_s**3))
        densidad = densidad_0/((r/r_s)*((1+(r/r_s))**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def POTDEFdisk_exp1(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)/2 #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)/2 #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x


#Obteniendo gradiente del potencial deflector
GRADPOT1nfw = np.zeros((len(theta1)), float)
GRADPOT2nfw = np.zeros((len(theta1)), float)
GRADPOT1disk_exp = np.zeros((len(theta1)), float)
GRADPOT2disk_exp = np.zeros((len(theta1)), float)
GRADPOT1 = np.zeros((len(theta1)), float)
GRADPOT2 = np.zeros((len(theta1)), float)
THETA1 = np.zeros((len(theta1)), float)
THETA2 = np.zeros((len(theta1)), float)

for l in range(len(theta1)):
    GRADPOT1nfw[l]= derivative(POTDEFnfw1, theta1[l], dx=1e-9, order=7)
    GRADPOT2nfw[l]= derivative(POTDEFnfw2, theta2[l], dx=1e-9, order=7)
    GRADPOT1disk_exp[l]= derivative(POTDEFdisk_exp1, theta1[l], dx=1e-9, order=7)
    GRADPOT2disk_exp[l]= derivative(POTDEFdisk_exp2, theta2[l], dx=1e-9, order=7)
    GRADPOT1[l]=2*(GRADPOT1nfw[l]+GRADPOT1disk_exp[l])
    GRADPOT2[l]=2*(GRADPOT2nfw[l]+GRADPOT2disk_exp[l])

#Obteniendo las coordenadas de las imágenes para los parámetros obtenidos

for l in range(len(theta1)):
    THETA1[l] = BETA1+GRADPOT1[l]
    THETA2[l] = BETA2+GRADPOT2[l]


# In[53]:


#Graficando la fuente, las imágenes observadas y las de los parámetros
fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(BETA1/FC, BETA2/FC, 'or')
plb.plot(theta1/FC, theta2/FC, 'ob')
plb.plot(THETA1/FC, THETA2/FC, 'og')
plb.xlim(-5,5)
plb.ylim(-5,5)
plb.legend(['Fuente', 'Datos observacionales', 'Valores del modelo'], loc='upper right', fontsize=15)
#plb.show()
plt.savefig('Ajuste.pdf')

#Definición de parámetros con sus errores
r_s = para[0]; r_s_95pos = parap95[0]; r_s_95neg = paran95[0]; r_s_68pos = parap68[0]; r_s_68neg = paran68[0]
m_0 = para[1]; rho_0_95pos = parap95[1]; rho_0_95neg = paran95[1]; rho_0_68pos = parap68[1]; rho_0_68neg = paran68[1]
Sigma_0 = para[2]; Sigma_0_95pos = parap95[2]; Sigma_0_95neg = paran95[2]; Sigma_0_68pos = parap68[2]; Sigma_0_68neg = paran68[2]
h_r = para[3]; h_r_95pos = parap95[3]; h_r_95neg = paran95[3]; h_r_68pos = parap68[3]; h_r_68neg = paran68[3]


table_data = []

table_para = [r"r_s",r"m_0",r"Sigma_0", r"h_r"]
table_units = [r"Kpc",r"Solar_Mass", r"Solar_Mass/Kpc^2", r"Kpc"]
para = [r_s, m_0, Sigma_0, h_r]
parap68=[r_s_68pos, m_0_68pos, Sigma_0_68pos, h_r_68pos]
paran68=[r_s_68neg, m_0_68neg, Sigma_0_68neg, h_r_68neg]
parap95=[r_s_95pos, m_0_95pos, Sigma_0_95pos, h_r_95pos]
paran95=[r_s_95neg, m_0_95neg, Sigma_0_95neg, h_r_95neg]
index=[r"r_s",r"m_0",r"Sigma_0", r"h_r"]

for i in range(len(para)):
	table_data.append([table_para[i], table_units[i], para[i], parap95[i], paran95[i],  parap68[i], paran68[i]])

column_name = [r"PARAMETER", r"UNITS", r"FIT", r"95%(+)", r"95%(-)",  r"68%(+)", r"68%(-)"]	
table_p = pd.DataFrame(table_data, index=index, columns=column_name)
table_p.to_csv("parameters_MCMC.txt", sep='\t', encoding='utf-8')
print ("\n#####################################################################")
print(table_p)
print ("\nDone")
print ("\n#####################################################################\n")


