#!/usr/bin/env python
# coding: utf-8

# In[1]:

from scipy.misc import *
import numpy as np #for data handling and numerical process
import pylab as plb  #Graphics and data
import matplotlib.pyplot as plt #for graphics
from scipy.integrate import quad # for numerical integrate with the quadratures methods 
from scipy.integrate import nquad  # for numerical integrate with the quadratures methods
from scipy.misc import derivative  # for numerical derivatives
import emcee #for the MCMC
import corner #for the obtaining and graphics of uncertainties
import pandas as pd #for data handling


# In[2]:

#Position of images
theta_ra = [12.1, -8.5, 21.7, -3.3]
theta_dec = [16.6, -10.4, -0.5, 19.2]


# In[3]:

#Position of images respect to center of galaxy J1331
theta_1 = np.zeros(len(theta_ra), float)
theta_2 = np.zeros(len(theta_dec), float)

for i in range(len(theta_1)):
    theta_1[i] = theta_ra[i]-0.5
    theta_2[i] = theta_dec[i]-0.5


# In[4]:

#Position of images in radians
theta1 = 0.05*theta_1*np.pi/(180*3600)
theta2 = 0.05*theta_2*np.pi/(180*3600)
theta = np.zeros(len(theta1),float)
for i in range(len(theta1)):
    theta[i]=np.sqrt(theta1[i]**2+theta2[i]**2)

# In[5]:

fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(theta1*1e6, theta2*1e6, 'ob')
plb.xlim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.ylim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.savefig('Datos.pdf')


# In[6]:

#Cosmological distance
D_ds = 442.7e3
D_d = 422e3
D_s = 817.9e3
d_ds = D_ds*1e3*3.086e16
d_d = D_d*1e3*3.086e16
d_s = D_s*1e3*3.086e16
#light velocity and universal gravitation constant 
c = 3e8
G = 6.67e-11
#Obtention of Critical Sigma
SIGMA_CRIT = (c**2)*d_s/(4*np.pi*G*d_d*d_ds) #Sigma crítico para la convergencia en kg/m^2\n",
SIGMA_CRIT = SIGMA_CRIT*5.027e-31*1e6/((3.241e-17)**2)

# In[13]:

#Parameter space
N2 = 4
r_s = np.linspace(0.1, 30, N2) #scale radius NFW",
M_0 = np.linspace(0.1e11, 10e11, N2) #Central mass NFW",
rho_0 = np.zeros((N2,N2), float) #Central density NFW",
for i in range(N2):
    for j in range(N2):
        rho_0[i,j]=(M_0[i]/(4*np.pi*(r_s[j]**3)))
#Values of exponential disc",
h_r = np.linspace(2, 6, N2) #scale radius
Sigma_0 = np.linspace(1e8, 15e8, N2) #central superficial density
#Values of bulge with Miyamoto-Nagai
a = np.linspace(1,10, N2)
b = np.linspace(0.1, 15, N2)
M = np.linspace(0.5e11, 1.5e11, N2)
#Source Position
FC = np.pi/(180*3600) #conversion factor between arcs and radians
BETA1 = 5.711025521733834e-10*FC
BETA2 = -8.474849110173339e-10*FC
#error en images position
sigma = 0.05*np.pi/(180*3600)

# In[14]:


#Function of deflector potential

def POTDEFnfw1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = rho_0[i,j]/((r/r_s[j])*((1+(r/r_s[j]))**2)) #Volumetric density
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def POTDEFnfw2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = rho_0[i,j]/((r/r_s[j])*((1+(r/r_s[j]))**2)) #Volumetric density
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def POTDEFdisk_exp1(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0[m]*np.exp(-D_d*TheTa/h_r[n])/2 #Volumetric density
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0[m]*np.exp(-D_d*TheTa/h_r[n])/2 #Volumetric density
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

def MN1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = ((b[p]**2)*M[o]/4*np.pi)*(a[q]*R**2+(a[q]+3*np.sqrt(z**2+b[p]**2))*(a[q]+ np.sqrt(z**2+b[p]**2))**2)/((R**2+(a[q]+np.sqrt(z**2+b[p]**2))**2)**2.5*(z**2+b[p]**2)**1.5)#Volumetric density
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def MN2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = ((b[p]**2)*M[o]/4*np.pi)*(a[q]*R**2+(a[q]+3*np.sqrt(z**2+b[p]**2))*(a[q]+ np.sqrt(z**2+b[p]**2))**2)/((R**2+(a[q]+np.sqrt(z**2+b[p]**2))**2)**2.5*(z**2+b[p]**2)**1.5)#Volumetric density
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

# In[9]:

#Compute of initial guess
X = np.zeros((N2,N2,N2,N2,N2,N2,N2,N2,N2,len(theta1)), float)
GRADPOT1nfw = np.zeros((N2,N2,len(theta1)), float)
GRADPOT2nfw = np.zeros((N2,N2,len(theta1)), float)
GRADPOT1disk_exp = np.zeros((N2,N2,len(theta1)), float)
GRADPOT2disk_exp = np.zeros((N2,N2,len(theta1)), float)
GRADPOT1MN = np.zeros((N2,N2,N2,len(theta1)), float)
GRADPOT2MN = np.zeros((N2,N2,N2,len(theta1)), float)
GRADPOT1 = np.zeros((N2,N2,N2,N2,N2,N2,N2,len(theta1)), float)
GRADPOT2 = np.zeros((N2,N2,N2,N2,N2,N2,N2,len(theta1)), float)
THETA1 = np.zeros((N2,N2,N2,N2,N2,N2,N2,len(theta1)), float)
THETA2 = np.zeros((N2,N2,N2,N2,N2,N2,N2,len(theta1)), float)
THETA = np.zeros((N2,N2,N2,N2,N2,N2,N2,len(theta1)), float)
L = np.zeros((N2,N2,N2,N2,N2,N2,N2), float)

#Exploration of parameters in NFW profile
for i in range(N2):
    for j in range(N2):
        for l in range(len(theta1)):
            GRADPOT1nfw[i,j,l]= derivative(POTDEFnfw1, theta1[l], dx=1e-9, order=7)
            GRADPOT2nfw[i,j,l]= derivative(POTDEFnfw2, theta2[l], dx=1e-9, order=7)
            print(i,j,l)


# In[10]:

#Exploration of parameters in Exponential Disc profile
for m in range(N2):
    for n in range(N2):
        for l in range(len(theta1)):
            GRADPOT1disk_exp[m,n,l]= derivative(POTDEFdisk_exp1, theta1[l], dx=1e-9, order=7)
            GRADPOT2disk_exp[m,n,l]= derivative(POTDEFdisk_exp2, theta2[l], dx=1e-9, order=7)
            print(m,n,l)


# In[15]:

#Exploration of parameters in MN profile
for o in range(N2):
    for p in range(N2):
        for q in range(N2):
            for l in range(len(theta1)):
                GRADPOT1MN[o,p,q,l] = derivative(MN1, theta1[l], dx=1e-9, order=7)
                GRADPOT2MN[o,p,q,l] = derivative(MN2, theta2[l], dx=1e-9, order=7)
                print(o,p,q,l)


# In[16]:

for i in range(N2):
    for j in range(N2):
        for m in range(N2):
            for n in range(N2):
                for o in range(N2):
                    for p in range(N2):
                        for q in range(N2):
                            for l in range(len(theta1)):
                                GRADPOT1[i,j,m,n,o,p,q,l]=2*(SIGMA_CRIT**2)*(GRADPOT1nfw[i,j,l]+GRADPOT1disk_exp[m,n,l]+GRADPOT1MN[o,p,q,l])
                                GRADPOT2[i,j,m,n,o,p,q,l]=2*(SIGMA_CRIT**2)*(GRADPOT2nfw[i,j,l]+GRADPOT2disk_exp[m,n,l]+GRADPOT2MN[o,p,q,l])
                                print(i,j,m,n,o,p,q,l)


# In[17]:

#Compute of comparison with observational images
for i in range(N2):
    for j in range(N2):
        for m in range(N2):
            for n in range(N2):
                for o in range(N2):
                    for p in range(N2):
                        for q in range(N2):
                            for t in range(len(theta1)):
                                THETA1[i,j,m,n,o,p,q,t] = BETA1+GRADPOT1[i,j,m,n,o,p,q,t]
                                THETA2[i,j,m,n,o,p,q,t] = BETA2+GRADPOT2[i,j,m,n,o,p,q,t]
                                THETA[i,j,m,n,o,p,q,t] = np.sqrt(THETA1[i,j,m,n,o,p,q,t]**2+THETA2[i,j,m,n,o,p,q,t]**2)
                                X[i,j,m,n,o,p,q,t] = ((theta[t]-THETA[i,j,m,n,o,p,q,t])**2)/(sigma**2)
                                print(i,j,m,n,o,p,q,t)
                            L[i,j,m,n,o,p,q] = np.sum(X[i,j,m,n,o,p,q])

# In[20]:

#Multiparametric minimization function
LIKE = np.zeros((N2,N2,N2,N2,N2,N2), float)
like = np.zeros((N2,N2,N2,N2,N2), float)
lik = np.zeros((N2,N2,N2,N2),float)
li = np.zeros((N2,N2,N2),float)
l = np.zeros((N2,N2),float)
LI = np.zeros((N2),float)

Minim = []
DEN_0 = []
RADIO_ESCALA = []
Den_0 = []
Radio_Escala = []
Masa = []
Altura = []
Longitud = []
beta1 = []
beta2 = []

for i in range(N2):
    for j in range(N2):
        for m in range(N2):
            for n in range(N2):
                for o in range(N2):
                    for p in range(N2):
                        for q in range(N2):
                            LIKE[i,j,m,n,o,p]=min(L[i,j,m,n,o,p])
                            like[i,j,m,n,o]=min(LIKE[i,j,m,n,o])
                            lik[i,j,m,n]=min(like[i,j,m,n])
                            li[i,j,m]=min(lik[i,j,m])
                            l[i,j]=min(li[i,j])
                            LI[i]=min(l[i])

for i in range(N2):
    for j in range(N2):
        for m in range(N2):
            for n in range(N2):
                for o in range(N2):
                    for p in range(N2):
                        for q in range(N2):
                            for t in range(len(LI)):
                                if L[i,j,m,n,o,p,q]==LI[t]:
                                    Minim.append(L[i,j,m,n,o,p,q])
                                    DEN_0.append(M_0[i])
                                    RADIO_ESCALA.append(r_s[j])
                                    Den_0.append(Sigma_0[m])
                                    Radio_Escala.append(h_r[n])
                                    Masa.append(M[o])
                                    Altura.append(b[p])
                                    Longitud.append(a[q])

for i in range(len(Minim)):
    if Minim[i]==min(Minim):
        Likehood = Minim[i]
        radio_s = RADIO_ESCALA[i]
        M_0 = DEN_0[i]
        escala_r = Radio_Escala[i] 
        den_0 = Den_0[i] 
        Mass = Masa[i] 
        height = Altura[i]
        length = Longitud[i]

#illustration of initial guess
print ("\n#####################################################################")
print("Initial Guess")
print("Likelihood=",Likehood, "radio_s=", radio_s, "M_0", M_0, "escala_r=", escala_r, "den_0=", den_0, "Mass=", Mass, "height=", height, "length=", length)
print ("\n#####################################################################")

# In[22]:

#Illustration of obtained images with initial Guess

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

def MN1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = ((height**2)*Mass/4*np.pi)*(length*R**2+(length+3*np.sqrt(z**2+height**2))*(length+np.sqrt(z**2+height**2))**2)/((R**2+(length+np.sqrt(z**2+height**2))**2)**2.5*(z**2+height**2)**1.5)#Densidad volumétrica de masa 
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def MN2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = ((height**2)*Mass/4*np.pi)*(length*R**2+(length+3*np.sqrt(z**2+height**2))*(length+np.sqrt(z**2+height**2))**2)/((R**2+(length+np.sqrt(z**2+height**2))**2)**2.5*(z**2+height**2)**1.5)#Densidad volumétrica de masa 
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
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
    GRADPOT1[l]=2*(SIGMA_CRIT**2)*(GRADPOT1nfw[l]+GRADPOT1disk_exp[l]+GRADPOT1MN[l])
    GRADPOT2[l]=2*(SIGMA_CRIT**2)*(GRADPOT2nfw[l]+GRADPOT2disk_exp[l]+GRADPOT2MN[l])

#Images obtained with initial guess
for l in range(len(theta1)):
    THETA1[l] = BETA1+GRADPOT1[l]
    THETA2[l] = BETA2+GRADPOT2[l]

# In[23]:


#Graphics of source and images
fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(BETA1*1e6, BETA2*1e6, 'or')
plb.plot(theta1*1e6, theta2*1e6, 'ob')
plb.plot(THETA1*1e6, THETA2*1e6, 'og')
plb.xlim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.ylim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.savefig('Guess_inicial.pdf')

# In[24]:

print ("\n#####################################################################")
print("MCMC------GALLENSPY")

#Model of the lens
def model(parameters, theta1, theta2, sigma):
    R_S, m_0, SIGMA_0, H_R, B, A, MASS  = parameters
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
    def MN1(x):
        def integ(z,TheTa):
            R = D_d*TheTa
            r = np.sqrt(R**2+z**2)
            densidad = ((B**2)*MASS/4*np.pi)*(A*R**2+(A+3*np.sqrt(z**2+B**2))*(A+np.sqrt(z**2+B**2))**2)/((R**2+(A+np.sqrt(z**2+B**2))**2)**2.5*(z**2+B**2)**1.5)#Densidad volumétrica de masa 
            return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
        THETA = np.sqrt(x**2 + theta2[l]**2)
        x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
        return x
    def MN2(x):
        def integ(z,TheTa):
            R = D_d*TheTa
            r = np.sqrt(R**2+z**2)
            densidad = ((B**2)*MASS/4*np.pi)*(A*R**2+(A+3*np.sqrt(z**2+B**2))*(A+np.sqrt(z**2+B**2))**2)/((R**2+(A+np.sqrt(z**2+B**2))**2)**2.5*(z**2+B**2)**1.5)#Densidad volumétrica de masa
            return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
        THETA = np.sqrt(x**2 + theta1[l]**2)
        x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
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
        GRADPOT1[l]=2*(SIGMA_CRIT**2)*(GRADPOT1nfw[l]+GRADPOT1disk_exp[l]+GRADPOT1MN[l])
        GRADPOT2[l]=2*(SIGMA_CRIT**2)*(GRADPOT2nfw[l]+GRADPOT2disk_exp[l]+GRADPOT2MN[l])
    for l in range(len(theta1)):
        THETA1[l] = BETA1+GRADPOT1[l]
        THETA2[l] = BETA2+GRADPOT2[l]
    THETA_teor = np.zeros((len(theta1)), float)
    for l in range(len(theta1)):
        THETA_teor[l] = np.sqrt(THETA1[l]**2+THETA2[l]**2)
    return THETA_teor

# In[25]:


#Likelihood function
def lnlike(parameters, theta1, theta2, sigma):
    R_S, m_0, SIGMA_0, H_R, B, A, MASS = parameters
    THETA_teor = model(parameters, theta1, theta2, sigma)
    X = np.zeros((len(theta1)),float)
    for l in range(len(theta1)):
        X[l]=((theta[l]-THETA_teor[l])**2)/(sigma**2)
    return -0.5*np.sum(X)

# In[26]:

#initial guess in the MCMC
start=np.zeros(7,float)
start[0] = radio_s
start[1] = M_0
start[2] = den_0
start[3] = escala_r
start[4] = height
start[5] = length
start[6] = Mass 

#Parametric space in the MCMC
def lnprior(parameters):
    R_S, m_0, SIGMA_0, H_R, B, A, MASS = parameters  
    if 0.05<R_S<32 and 0.05e11<m_0<12e11 and 0.8e8<SIGMA_0<17e8 and 1<H_R<7 and 0.05<B<17 and 0.5<A<12 and 0.3e11<MASS<1.7e11:
        return 0.0
    return -np.inf

# In[37]:


#Probability function
def lnprob(parameters, theta1, theta2, sigma):
    lp = lnprior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(parameters, theta1, theta2, sigma)

# In[38]:


#Dimension and walkers
ndim, nwalkers = 7, 100
#initial posicion and step length
pos_step = 1e-8
pos_in = [abs(start + pos_step*start*np.random.randn(ndim)+1e-9*np.random.randn(ndim)) for i in range(nwalkers)]


# In[39]:


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(theta1, theta2,sigma))


# In[40]:

#Number of Steps
sampler.run_mcmc(pos_in, 100, progress=True)
    

# In[41]:

#Step of cut in the MCMC
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# In[42]:

percentage=0.68
#Contours of values
fig = corner.corner(samples, labels=["$r_s$", r"$mo_0$", r"$\Sigma_0$", "$h_R$", "$b$", "a", "$M$"],
                    quantiles = [0.5-0.5*percentage, 0.5, 0.5+0.5*percentage],fill_contours=True, plot_datapoints=True)
fig.savefig("contours_J1331.pdf")

# In[43]:

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


# In[44]:


#Visualization of generated images for the parameter set obtained 
r_s = para[0]
m_0 = para[1]
Sigma_0 = para[2] 
h_r = para[3]
b = para[4]
a = para[5]
M = para[6]

def POTDEFnfw1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad_0 = m_0/(4*np.pi*(r_s**3))
        densidad = densidad_0/((r/r_s)*((1+(r/r_s))**2)) #Volumetric Density
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def POTDEFnfw2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad_0 = m_0/(4*np.pi*(r_s**3))
        densidad = densidad_0/((r/r_s)*((1+(r/r_s))**2)) #Volumetric Density
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def POTDEFdisk_exp1(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)/2 #Volumetric Density
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)/2 #Volumetric Density
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

def MN1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = ((b**2)*M/4*np.pi)*(a*R**2+(a+3*np.sqrt(z**2+b**2))*(a+np.sqrt(z**2+b**2))**2)/((R**2+(a+np.sqrt(z**2+b**2))**2)**2.5*(z**2+b**2)**1.5)#Volumetric Density
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def MN2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = ((b**2)*M/4*np.pi)*(a*R**2+(a+3*np.sqrt(z**2+b**2))*(a+np.sqrt(z**2+b**2))**2)/((R**2+(a+np.sqrt(z**2+b**2))**2)**2.5*(z**2+b**2)**1.5)#Volumetric Density
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
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
    GRADPOT1[l]=2*(GRADPOT1nfw[l]+GRADPOT1disk_exp[l]+GRADPOT1MN[l])
    GRADPOT2[l]=2*(GRADPOT2nfw[l]+GRADPOT2disk_exp[l]+GRADPOT2MN[l])

for l in range(len(theta1)):
    THETA1[l] = BETA1+GRADPOT1[l]
    THETA2[l] = BETA2+GRADPOT2[l]


# In[45]:

#Graphics of source and images
fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(BETA1/FC, BETA2/FC, 'or')
plb.plot(theta1/FC, theta2/FC, 'ob')
plb.plot(THETA1/FC, THETA2/FC, 'og')
plb.xlim(-2.5,2.5)
plb.ylim(-2.5,2.5)
plb.legend(['Fuente', 'Datos observacionales', 'Valores del modelo'], loc='upper right', fontsize=15)
#plb.show()
plt.savefig('Ajuste.pdf')

#Parameters and errors
r_s = para[0]; r_s_95pos = parap95[0]; r_s_95neg = paran95[0]; r_s_68pos = parap68[0]; r_s_68neg = paran68[0]
m_0 = para[1]; m_0_95pos = parap95[1]; m_0_95neg = paran95[1]; m_0_68pos = parap68[1]; m_0_68neg = paran68[1]
Sigma_0 = para[2]; Sigma_0_95pos = parap95[2]; Sigma_0_95neg = paran95[2]; Sigma_0_68pos = parap68[2]; Sigma_0_68neg = paran68[2]
h_r = para[3]; h_r_95pos = parap95[3]; h_r_95neg = paran95[3]; h_r_68pos = parap68[3]; h_r_68neg = paran68[3]
b = para[4]; b_95pos = parap95[4]; b_95neg = paran95[4]; b_68pos = parap68[4]; b_68neg = paran68[4]
a = para[5]; a_95pos = parap95[5]; a_95neg = paran95[5]; a_68pos = parap68[5]; a_68neg = paran68[5]
M = para[6]; M_95pos = parap95[6]; M_95neg = paran95[6]; M_68pos = parap68[6]; M_68neg = paran68[6]


# In[46]:

table_data = []

table_para = [r"r_s",r"m_0",r"Sigma_0", r"h_r", r"b", r"a", r"M"]
table_units = [r"Kpc",r"Solar_Mass", r"Solar_Mass/Kpc^2", r"Kpc", r"Kpc", r"Kpc", r"Solar_Mass"]
para = [r_s, m_0, Sigma_0, h_r, b, a, M]
parap68=[r_s_68pos, m_0_68pos, Sigma_0_68pos, h_r_68pos, b_68pos, a_68pos, M_68pos]
paran68=[r_s_68neg, m_0_68neg, Sigma_0_68neg, h_r_68neg, b_68neg, a_68neg, M_68neg]
parap95=[r_s_95pos, m_0_95pos, Sigma_0_95pos, h_r_95pos, b_95pos, a_95pos, M_95pos]
paran95=[r_s_95neg, m_0_95neg, Sigma_0_95neg, h_r_95neg, b_95neg, a_95neg, M_95neg]
index=[r"r_s",r"m_0",r"Sigma_0", r"h_r", r"b", r"a", r"M"]

for i in range(len(para)):
	table_data.append([table_para[i], table_units[i], para[i], parap95[i], paran95[i],  parap68[i], paran68[i]])

column_name = [r"PARAMETER", r"UNITS", r"FIT", r"95%(+)", r"95%(-)",  r"68%(+)", r"68%(-)"]	
table_p = pd.DataFrame(table_data, index=index, columns=column_name)
table_p.to_csv("parameters_MCMC.txt", sep='\t', encoding='utf-8')
print ("\n#####################################################################")
print(table_p)
print ("\nDone")
print ("\n#####################################################################\n")


