#!/usr/bin/env python
# coding: utf-8

# In[1]:

from scipy.misc import *
import numpy as np
import pylab as plb
from scipy.misc import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.integrate import quad
from scipy.integrate import nquad
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.misc import derivative
from scipy.optimize import fsolve
from PIL import Image
import pandas as pd
from astropy import table as Table # For fast and easy reading / writing with tables using numpy library


# In[2]:

#Grid of the lens plane

N = 100
Theta1 = np.linspace(-4*np.pi/(180*3600),4*np.pi/(180*3600),N)
Theta2 = np.linspace(-4*np.pi/(180*3600),4*np.pi/(180*3600),N)
Theta = np.zeros((N,N), float)
for i in range(N):
    for j in range(N):
        Theta[i,j]=np.sqrt(Theta1[i]**2+Theta2[j]**2)


# In[3]:


#Parameters of the lens 
     
r_s = 10.0377
m_0 = 9827961671.7142
Sigma_0 = 1498800353.764 
h_r = 5.961867851492107
b = 5.171380120172431
a = 1.0310246430876124
M = 86818768473.91742


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

#Critical Sigma
SIGMA_CRIT = (c**2)*d_s/(4*np.pi*G*d_d*d_ds) #Sigma crítico para la convergencia en kg/m^2\n",
SIGMA_CRIT = SIGMA_CRIT*5.027e-31*1e6/((3.241e-17)**2)


# In[4]:

#Deflector potential in the all grid
def POTDEFnfw1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        RHO_0 = m_0/(4*np.pi*(r_s**3))
        densidad = RHO_0/((r/r_s)*((1+(r/r_s))**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta2[j]**2)
    x = nquad(integ, [[0, np.inf],[0, Theta[i,j]]])[0]
    return x

def POTDEFnfw2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad_0 = m_0/(4*np.pi*(r_s**3))
        densidad = densidad_0/((r/r_s)*((1+(r/r_s))**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta1[i]**2)
    x = nquad(integ, [[0, np.inf],[0, Theta[i,j]]])[0]
    return x

def POTDEFdisk_exp1(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)/2 #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta2[j]**2)
    x = quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)/2 #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta1[i]**2)
    x = quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]
    return x

def MN1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = ((b**2)*M/4*np.pi)*(a*R**2+(a+3*np.sqrt(z**2+b**2))*(a+np.sqrt(z**2+b**2))**2)/((R**2+(a+np.sqrt(z**2+b**2))**2)**2.5*(z**2+b**2)**1.5)#Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta2[j]**2)
    x = nquad(integ, [[0, np.inf],[0, Theta[i,j]]])[0]
    return x

def MN2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = ((b**2)*M/4*np.pi)*(a*R**2+(a+3*np.sqrt(z**2+b**2))*(a+np.sqrt(z**2+b**2))**2)/((R**2+(a+np.sqrt(z**2+b**2))**2)**2.5*(z**2+b**2)**1.5)#Volumetric density
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta1[i]**2)
    x = nquad(integ, [[0, np.inf],[0, Theta[i,j]]])[0]
    return x

# In[5]:

#Second order derivatives for the compute of the Shear
alpha_1NFW = np.zeros((len(Theta1),len(Theta2)),float)
alpha_2NFW = np.zeros((len(Theta1),len(Theta2)),float)
alpha_1disc_exp = np.zeros((len(Theta1),len(Theta2)),float)
alpha_2disc_exp = np.zeros((len(Theta1),len(Theta2)),float)
alpha_1MN = np.zeros((len(Theta1),len(Theta2)),float)
alpha_2MN = np.zeros((len(Theta1),len(Theta2)),float)

for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        alpha_1NFW[i,j]=derivative(POTDEFnfw1, Theta1[i], dx=1e-9, n=2, order=7)
        print('NFW1',i,j)
        
for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        alpha_2NFW[i,j]=derivative(POTDEFnfw2, Theta2[j], dx=1e-9, n=2, order=7)
        print('NFW2',i,j)
        
for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        alpha_1disc_exp[i,j]=derivative(POTDEFdisk_exp1, Theta1[i], dx=1e-9, n=2, order=7)
        print('disc_exp1',i,j)
        
for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        alpha_2disc_exp[i,j]=derivative(POTDEFdisk_exp2, Theta2[j], dx=1e-9, n=2, order=7)
        print('disc_exp2',i,j)

for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        alpha_1MN[i,j]=derivative(MN1, Theta1[i], dx=1e-9, n=2, order=7)
        print('MN1',i,j)
        
for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        alpha_2MN[i,j]=derivative(MN2, Theta2[j], dx=1e-9, n=2, order=7)
        print('MN2',i,j)

# In[6]:

alpha_1 = 2*(SIGMA_CRIT**2)*(alpha_1NFW+alpha_1disc_exp+alpha_1MN)
alpha_2 = 2*(SIGMA_CRIT**2)*(alpha_2NFW+alpha_2disc_exp+alpha_2MN)


# In[7]:

def POTDEFnfw(x,y):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        RHO_0 = m_0/(4*np.pi*(r_s**3))
        densidad = RHO_0/((r/r_s)*((1+(r/r_s))**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + y**2)
    return nquad(integ, [[0, np.inf],[0, Theta[i,j]]])[0]

def POTDEFdisk_exp(x,y):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)/2 #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + y**2)
    return quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]

def MN(x,y):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = ((b**2)*M/4*np.pi)*(a*R**2+(a+3*np.sqrt(z**2+b**2))*(a+np.sqrt(z**2+b**2))**2)/((R**2+(a+np.sqrt(z**2+b**2))**2)**2.5*(z**2+b**2)**1.5)#Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + y**2)
    return nquad(integ, [[0, np.inf],[0, Theta[i,j]]])[0]


# In[8]:


def DERIVADA_1(x,y):
    return (-POTDEFnfw(x+2*h,y)+8*POTDEFnfw(x+h,y)-8*POTDEFnfw(x-h,y)+POTDEFnfw(x-2*h,y))/(12*h)

def DERIVADA_2(x,y):
    return (-POTDEFdisk_exp(x+2*h,y)+8*POTDEFdisk_exp(x+h,y)-8*POTDEFdisk_exp(x-h,y)+POTDEFdisk_exp(x-2*h,y))/(12*h)

def DERIVADA_3(x,y):
    return (-MN(x+2*h,y)+8*MN(x+h,y)-8*MN(x-h,y)+MN(x-2*h,y))/(12*h)

def SEGDERIV_1(x,y):
    return (-DERIVADA_1(x,y+2*h)+8*DERIVADA_1(x,y+h)-8*DERIVADA_1(x,y-h)+DERIVADA_1(x,y-2*h))/(12*h)

def SEGDERIV_2(x,y):
    return (-DERIVADA_2(x,y+2*h)+8*DERIVADA_2(x,y+h)-8*DERIVADA_2(x,y-h)+DERIVADA_2(x,y-2*h))/(12*h)

def SEGDERIV_3(x,y):
    return (-DERIVADA_3(x,y+2*h)+8*DERIVADA_3(x,y+h)-8*DERIVADA_3(x,y-h)+DERIVADA_3(x,y-2*h))/(12*h)


# In[9]:


h = 1e-9

alpha_12nfw = np.zeros((len(Theta1),len(Theta2)),float)
alpha_12disc = np.zeros((len(Theta1),len(Theta2)),float)
alpha_12MN = np.zeros((len(Theta1),len(Theta2)),float)

for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        x = Theta1[i]
        y = Theta2[j]
        alpha_12nfw[i,j]=SEGDERIV_1(x,y)
        print('NFW1',i,j)
        
for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        x = Theta1[i]
        y = Theta2[j]
        alpha_12disc[i,j]=SEGDERIV_2(x,y)
        print('disc',i,j)

for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        x = Theta1[i]
        y = Theta2[j]
        alpha_12MN[i,j]=SEGDERIV_3(x,y)
        print('MN',i,j)


# In[10]:


alpha_12 = 2*(SIGMA_CRIT**2)*(alpha_12nfw+alpha_12disc+alpha_12MN)

# In[11]:


#Obtention of the convergence

#Galactocentric radius
R = D_d*Theta
#Definición de la función a integrar, se multiplica por 2 para integrar de -inf a inf

def densNFW(R, z):
    r = np.sqrt(R**2+z**2)
    RHO_0 = m_0/(4*np.pi*(r_s**3))
    densidad = RHO_0/((r/r_s)*((1+(r/r_s))**2)) #Densidad volumétrica de masa
    return 2*densidad

def densMN(R, z):
    r = np.sqrt(R**2+z**2)
    densidad = ((b**2)*M/4*np.pi)*(a*R**2+(a+3*np.sqrt(z**2+b**2))*(a+np.sqrt(z**2+b**2))**2)/((R**2+(a+np.sqrt(z**2+b**2))**2)**2.5*(z**2+b**2)**1.5)#Densidad volumétrica de masa
    return 2*densidad


#Obtention of Sigma
def SigmaNFW(R):
    def I(R):
        return quad(densNFW, 0, np.inf, limit=N, args=(R))[0]
    vec_I = np.vectorize(I) 
    P = vec_I(R) #Generando lista de datos
    return P

SigmaDISC = Sigma_0*np.exp(-D_d*Theta/h_r)

def SigmaMN(R):
    def I(R):
        return quad(densMN, 0, np.inf, limit=N, args=(R))[0]
    vec_I = np.vectorize(I) 
    P = vec_I(R) #Generando lista de datos
    return P

Sigma = SigmaNFW(R)+SigmaDISC+SigmaMN(R) 

#Compute of kappa
k = Sigma/SIGMA_CRIT


# In[12]:


# Get the shear

gamma1 = 0.5*(alpha_1-alpha_2)
gamma2 = alpha_12

gamma = np.sqrt(gamma1**2+gamma2**2) 


# In[13]:


#Determinant of Jacobian Matriz
detA = (1-k)**2-(gamma**2)


# In[14]:


# Evaluation of the matriz for the critical curve

S = np.zeros((len(Theta1), len(Theta2)), float)

for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        if detA[i,j]>0:
            S[i,j]=1
        if detA[i,j]<0:
            S[i,j]=-1

# In[15]:


crit_1 = []
crit_2 = []

for i in range(len(Theta1)-1):
    for j in range(len(Theta2)-1):
        if np.abs(S[i+1,j]+S[i-1,j]+S[i,j+1]+S[i,j-1])<4:
            print(i,j)
            crit_1.append(Theta1[i])
            crit_2.append(Theta2[j])
            
crit_1 = np.array(crit_1)
crit_2 = np.array(crit_2)


# In[16]:

plt.rcParams['figure.figsize'] =(10,10)
plb.plot(crit_1*1e6, crit_2*1e6, 'or')
plb.xlim(-4*np.pi*1e6/(180*3600),4*np.pi*1e6/(180*3600))
plb.ylim(-4*np.pi*1e6/(180*3600),4*np.pi*1e6/(180*3600))
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
#plb.show()
plb.savefig('critical_curve.pdf')


# In[17]:

#Obtention of critical radius
rad_crit = np.sqrt(crit_1**2+crit_2**2)

# In[18]:

FC = np.pi/(180*3600) #conversion factor between arcs and radians
RAD_CRIT = np.sum(rad_crit)/len(rad_crit)
THETA_EINS = 0.901*FC

# In[19]:


theta = np.linspace(0, 2*np.pi, 1000)
CRIT_1 = RAD_CRIT*np.cos(theta) 
CRIT_2 = RAD_CRIT*np.sin(theta) 
CRIT = np.sqrt(CRIT_1**2+CRIT_2**2)

THETA_EINS_1 = THETA_EINS*np.cos(theta) 
THETA_EINS_2 = THETA_EINS*np.sin(theta) 

# In[20]:

fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(crit_1*1e6, crit_2*1e6, 'or')
plb.plot(CRIT_1*1e6, CRIT_2*1e6, '-b')
plb.xlim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
plb.ylim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
#plb.show()
plb.savefig('adjust_critical.pdf')

# In[21]:

#Deflector potential in the critical points
def POTDEFnfw1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        RHO_0 = m_0/(4*np.pi*(r_s**3))
        densidad = RHO_0/((r/r_s)*((1+(r/r_s))**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + CRIT_2[i]**2)
    x = nquad(integ, [[0, np.inf],[0, CRIT[i]]])[0]
    return x

def POTDEFnfw2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad_0 = m_0/(4*np.pi*(r_s**3))
        densidad = densidad_0/((r/r_s)*((1+(r/r_s))**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + CRIT_1[i]**2)
    x = nquad(integ, [[0, np.inf],[0, CRIT[i]]])[0]
    return x

def POTDEFdisk_exp1(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)/2 #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + CRIT_2[i]**2)
    x = quad(integ, 0, CRIT[i], limit=100, args=(Theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)/2 #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + CRIT_1[i]**2)
    x = quad(integ, 0, CRIT[i], limit=100, args=(Theta))[0]
    return x

def MN1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = ((b**2)*M/4*np.pi)*(a*R**2+(a+3*np.sqrt(z**2+b**2))*(a+np.sqrt(z**2+b**2))**2)/((R**2+(a+np.sqrt(z**2+b**2))**2)**2.5*(z**2+b**2)**1.5)#Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + CRIT_2[i]**2)
    x = nquad(integ, [[0, np.inf],[0, CRIT[i]]])[0]
    return x

def MN2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = ((b**2)*M/4*np.pi)*(a*R**2+(a+3*np.sqrt(z**2+b**2))*(a+np.sqrt(z**2+b**2))**2)/((R**2+(a+np.sqrt(z**2+b**2))**2)**2.5*(z**2+b**2)**1.5)#Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + CRIT_1[i]**2)
    x = nquad(integ, [[0, np.inf],[0, CRIT[i]]])[0]
    return x


# In[22]:


#Lens equation for the obtention of the caustic curve
GRADPOT_1NFW = np.zeros(len(CRIT_1), float)
GRADPOT_2NFW = np.zeros(len(CRIT_1), float)
GRADPOT_1disc_exp = np.zeros(len(CRIT_1), float)
GRADPOT_2disc_exp = np.zeros(len(CRIT_1), float)
GRADPOT_1MN = np.zeros(len(CRIT_1), float)
GRADPOT_2MN = np.zeros(len(CRIT_1), float)

for i in range(len(CRIT_1)):
    GRADPOT_1NFW[i]=derivative(POTDEFnfw1, CRIT_1[i], dx=1e-9, order=7)
    print('NFW1',i)


for i in range(len(CRIT_1)):
    GRADPOT_2NFW[i]=derivative(POTDEFnfw2, CRIT_2[i], dx=1e-9, order=7)
    print('NFW2',i)
        
for i in range(len(CRIT_1)):
    GRADPOT_1disc_exp[i]=derivative(POTDEFdisk_exp1, CRIT_1[i], dx=1e-9, order=7)
    print('disc_exp1',i)
        
for i in range(len(CRIT_1)):
    GRADPOT_2disc_exp[i]=derivative(POTDEFdisk_exp2, CRIT_2[i], dx=1e-9, order=7)
    print('disc_exp2',i)
        
for i in range(len(CRIT_1)):
    GRADPOT_1MN[i]=derivative(MN1, CRIT_1[i], dx=1e-9, order=7)
    print('MN1',i)


# In[23]:


for i in range(len(CRIT_1)):
    GRADPOT_2MN[i]=derivative(MN2, CRIT_2[i], dx=1e-9, order=7)
    print('MN2',i)

GRADPOT_1 = 2*(SIGMA_CRIT**2)*(GRADPOT_1NFW +GRADPOT_1disc_exp+GRADPOT_1MN)
GRADPOT_2 = 2*(SIGMA_CRIT**2)*(GRADPOT_2NFW +GRADPOT_2disc_exp+GRADPOT_2MN)


# In[24]:

CAUST_1 = CRIT_1-GRADPOT_1
CAUST_2 = CRIT_2-GRADPOT_2


# In[25]:


#Visualization of the caustic curve
fig = plt.figure()
plb.plot(CAUST_1*1e6, CAUST_2*1e6, '-r')
plb.xlim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
plb.ylim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
plb.xlabel(r"$\beta_1$", fontsize=20)
plb.ylabel(r"$\beta_2$", fontsize=20)
plb.savefig('caustic.pdf')
#plb.show()


# In[26]:

#Coordinates of observational images

theta_ra = [12.1, -8.5, 21.7, -3.3]
theta_dec = [16.6, -10.4, -0.5, 19.2]

theta_1 = np.zeros(len(theta_ra), float)
theta_2 = np.zeros(len(theta_dec), float)

for i in range(len(theta_1)):
    theta_1[i] = theta_ra[i]-0.5
    theta_2[i] = theta_dec[i]-0.5
    
theta1 = 0.05*theta_1*np.pi/(180*3600)
theta2 = 0.05*theta_2*np.pi/(180*3600)


# In[27]:

#Source Position
BETA1 = 5.711025521733834e-10
BETA2 = -8.474849110173339e-10

# In[28]:

#Deflector potential in the points of the images
    
def POTDEFnfw1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad_0 = m_0/(4*np.pi*(r_s**3))
        densidad = densidad_0/((r/r_s)*((1+(r/r_s))**2)) #Volumetric density
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta2[j]**2)
    x = nquad(integ, [[0, np.inf],[0, Theta[i,j]]])[0]
    return x

def POTDEFnfw2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad_0 = m_0/(4*np.pi*(r_s**3))
        densidad = densidad_0/((r/r_s)*((1+(r/r_s))**2)) #Volumetric Density
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta1[i]**2)
    x = nquad(integ, [[0, np.inf],[0, Theta[i,j]]])[0]
    return x

def POTDEFdisk_exp1(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)/2 #Superficial density
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta2[j]**2)
    x = quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)/2 #Superficial density
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta1[i]**2)
    x = quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]
    return x

def MN1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = ((b**2)*M/4*np.pi)*(a*R**2+(a+3*np.sqrt(z**2+b**2))*(a+np.sqrt(z**2+b**2))**2)/((R**2+(a+np.sqrt(z**2+b**2))**2)**2.5*(z**2+b**2)**1.5)#Volumetric Density
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta2[j]**2)
    x = nquad(integ, [[0, np.inf],[0, Theta[i,j]]])[0]
    return x

def MN2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = ((b**2)*M/4*np.pi)*(a*R**2+(a+3*np.sqrt(z**2+b**2))*(a+np.sqrt(z**2+b**2))**2)/((R**2+(a+np.sqrt(z**2+b**2))**2)**2.5*(z**2+b**2)**1.5)#Volumetric density
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta1[i]**2)
    x = nquad(integ, [[0, np.inf],[0, Theta[i,j]]])[0]
    return x


# In[29]:


GRADPOT1nfw = np.zeros((len(Theta1),len(Theta1)), float)
GRADPOT2nfw = np.zeros((len(Theta1),len(Theta1)), float)
GRADPOT1disk_exp = np.zeros((len(Theta1),len(Theta1)), float)
GRADPOT2disk_exp = np.zeros((len(Theta1),len(Theta1)), float)
GRADPOT1MN = np.zeros((len(Theta1),len(Theta1)), float)
GRADPOT2MN = np.zeros((len(Theta1),len(Theta1)), float)
GRADPOT1 = np.zeros((len(Theta1),len(Theta1)), float)
GRADPOT2 = np.zeros((len(Theta1),len(Theta1)), float)

#Gradient of deflecto potential
for i in range(len(Theta1)):
    for j in range(len(Theta1)):
        GRADPOT1nfw[i,j]= derivative(POTDEFnfw1, Theta1[i], dx=1e-9, order=7)
        GRADPOT2nfw[i,j]= derivative(POTDEFnfw2, Theta2[j], dx=1e-9, order=7)
        print(i,j)

for i in range(len(Theta1)):
    for j in range(len(Theta1)):
        GRADPOT1disk_exp[i,j]= derivative(POTDEFdisk_exp1, Theta1[i], dx=1e-9, order=7)
        GRADPOT2disk_exp[i,j]= derivative(POTDEFdisk_exp2, Theta2[j], dx=1e-9, order=7)
        print('disc',i,j)
        
for i in range(len(Theta1)):
    for j in range(len(Theta1)):
        GRADPOT1MN[i,j] = derivative(MN1, Theta1[i], dx=1e-9, order=7)
        GRADPOT2MN[i,j] = derivative(MN2, Theta2[j], dx=1e-9, order=7)
        print('MN',i,j)
        
for i in range(len(Theta1)):
    for j in range(len(Theta1)):
        GRADPOT1[i,j]=2*(SIGMA_CRIT**2)*(GRADPOT1nfw[i,j]+GRADPOT1disk_exp[i,j]+GRADPOT1MN[i,j])
        GRADPOT2[i,j]=2*(SIGMA_CRIT**2)*(GRADPOT2nfw[i,j]+GRADPOT2disk_exp[i,j]+GRADPOT2MN[i,j])


# In[30]:


GRADPOT = np.sqrt(GRADPOT1**2+GRADPOT2**2)


# In[31]:


thet1, thet2= np.meshgrid(Theta1, Theta2)
fig = plb.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(thet1*1e6,thet2*1e6,GRADPOT1*1e6, rstride=2, cstride=2, color='g')
ax.set_xlabel(r"$\theta_1$",fontsize=20)
ax.set_ylabel(r"$\theta_2$",fontsize=20)
ax.set_zlabel(r"$\alpha_1$",fontsize=20)
#plb.show()
plb.savefig('Grad_pot1.pdf')

# In[32]:


thet1, thet2= np.meshgrid(Theta1, Theta2)
fig = plb.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(thet1*1e6,thet2*1e6,GRADPOT2*1e6, rstride=2, cstride=2, color='r')
ax.set_xlabel(r"$\theta_1$",fontsize=20)
ax.set_ylabel(r"$\theta_2$",fontsize=20)
ax.set_zlabel(r"$\alpha_2$",fontsize=20)
#plb.show()
plb.savefig('gradpot2.pdf')

# In[33]:


thet1, thet2= np.meshgrid(Theta1, Theta2)
fig = plb.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(thet1*1e6,thet2*1e6,GRADPOT*1e6, rstride=2, cstride=2, color='g')
ax.set_xlabel(r"$\theta_1$",fontsize=20)
ax.set_ylabel(r"$\theta_2$",fontsize=20)
ax.set_zlabel(r"$\alpha$",fontsize=20)
#plb.show()
plb.savefig('GRADPOT.pdf')

# In[34]:

fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(CRIT_1/FC, CRIT_2/FC, '-m')
plb.plot(THETA_EINS_1/FC, THETA_EINS_2/FC, '-r')
plb.plot(theta1/FC, theta2/FC, 'ob')
plb.xlim(-4,4)
plb.ylim(-4,4)
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
plb.legend(['Critical curve','Einstein ring','Observational images'], loc='upper right', fontsize=15)
#plb.show()
plb.savefig('Images_CriticaCurve.pdf')

fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(CAUST_1/FC, CAUST_2/FC, '-m')
plb.plot(BETA1/FC, BETA1/FC, 'or')
plb.xlim(-4,4)
plb.ylim(-4,4)
plb.xlabel(r"$\beta_1$", fontsize=20)
plb.ylabel(r"$\beta_2$", fontsize=20)
plb.legend(['Caustic', 'Source'], loc='upper right', fontsize=15)
#plb.show()
plb.savefig('Source_Caustic.pdf')


# In[35]: Mass estimation

#errors 95%

r_s95pos = 10.037798696013676+7.870479264017108
m_095pos = 39827961671.7142+3578324100.5798607
r_s95neg = 10.037798696013676-7.535556311172149
m_095neg = 39827961671.7142-4507619893.300674
Sigma_095neg = 1498800353.7647953-656502611.9169159 
Sigma_095pos = 1498800353.7647953+194110201.77950072 
h_r95pos = 5.961867851492107+0.9939159553005048
h_r95neg = 5.961867851492107-3.0912856813363625
b95pos = 5.171380120172431+1.717471262188468
b95neg = 5.171380120172431-1.7427154623099308
a95pos = 1.0310246430876124+0.9506178125818747
a95neg = 1.0310246430876124-0.48769029318804225
M95pos = 86818768473.91742+68023486691.86847
M95neg = 86818768473.91742-37007542298.44429

#Superficial densities central
densidad_0 = m_0/(4*np.pi*(r_s**3))
def integ(z,TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    r = np.sqrt(R**2+z**2)
    densidad = densidad_0/((r/r_s)*((1+(r/r_s))**2)) #Volumetric density
    return 2*densidad
def POTDEFdisk_exp(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r) #Superficial density
    return Sigma
def MN(z,TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    r = np.sqrt(R**2+z**2)
    densidad = ((b**2)*M/4*np.pi)*(a*R**2+(a+3*np.sqrt(z**2+b**2))*(a+np.sqrt(z**2+b**2))**2)/((R**2+(a+np.sqrt(z**2+b**2))**2)**2.5*(z**2+b**2)**1.5)#Volumetric density
    return 2*densidad

#Superficial densities 95pos
densidad_095pos = m_095pos/(4*np.pi*(r_s95neg**3))
def integ95pos(z,TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    r = np.sqrt(R**2+z**2)
    densidad = densidad_095pos/((r/r_s95neg)*((1+(r/r_s95neg))**2)) #Volumetric density
    return 2*densidad
def POTDEFdisk_exp95pos(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    Sigma = Sigma_095pos*np.exp(-D_d*TheTa/h_r95neg) #Superficial density
    return Sigma
def MN95pos(z,TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    r = np.sqrt(R**2+z**2)
    densidad = ((b95pos**2)*M95pos/4*np.pi)*(a95pos*R**2+(a95pos+3*np.sqrt(z**2+b95pos**2))*(a95pos+np.sqrt(z**2+b95pos**2))**2)/((R**2+(a95pos+np.sqrt(z**2+b95pos**2))**2)**2.5*(z**2+b95pos**2)**1.5)#Volumetric density
    return 2*densidad

#Superficial densities 95neg
densidad_095neg = m_095neg/(4*np.pi*(r_s95pos**3))
def integ95neg(z,TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    r = np.sqrt(R**2+z**2)
    densidad = densidad_095neg/((r/r_s95pos)*((1+(r/r_s95pos))**2)) #Volumetric density
    return 2*densidad
def POTDEFdisk_exp95neg(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    Sigma = Sigma_095neg*np.exp(-D_d*TheTa/h_r95pos) #Superficial density
    return Sigma
def MN95neg(z,TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    r = np.sqrt(R**2+z**2)
    densidad = ((b95neg**2)*M95neg/4*np.pi)*(a95neg*R**2+(a95neg+3*np.sqrt(z**2+b95neg**2))*(a95neg+np.sqrt(z**2+b95neg**2))**2)/((R**2+(a95neg+np.sqrt(z**2+b95neg**2))**2)**2.5*(z**2+b95neg**2)**1.5)#Volumetric density
    return 2*densidad

#Mass within critical radius
lim = RAD_CRIT
M_dark_crit = nquad(integ, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_bulge_crit = nquad(MN, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_disc_crit = nquad(POTDEFdisk_exp, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_crit = M_dark_crit+M_bulge_crit+M_disc_crit

M_dark_crit95pos = nquad(integ95pos, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_bulge_crit95pos = nquad(MN95pos, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_disc_crit95pos = nquad(POTDEFdisk_exp95pos, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_crit95pos = M_dark_crit95pos+M_bulge_crit95pos+M_disc_crit95pos

M_dark_crit95neg = nquad(integ95neg, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_bulge_crit95neg = nquad(MN95neg, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_disc_crit95neg = nquad(POTDEFdisk_exp95neg, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_crit95neg = M_dark_crit95neg+M_bulge_crit95neg+M_disc_crit95neg

M_dark_pos_crit= np.abs(M_dark_crit-M_dark_crit95pos)
M_dark_neg_crit= np.abs(M_dark_crit-M_dark_crit95neg)
M_bulge_pos_crit= np.abs(M_bulge_crit-M_bulge_crit95pos)
M_bulge_neg_crit= np.abs(M_bulge_crit-M_bulge_crit95neg)
M_disc_pos_crit= np.abs(M_disc_crit-M_disc_crit95pos)
M_disc_neg_crit= np.abs(M_disc_crit-M_disc_crit95neg)
M_crit_pos= np.abs(M_crit-M_crit95pos)
M_crit_neg= np.abs(M_crit-M_crit95neg)

#Mass within Einstein radius

lim = THETA_EINS

M_dark_eins = nquad(integ, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_bulge_eins = nquad(MN, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_disc_eins = nquad(POTDEFdisk_exp, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_eins = M_dark_eins+M_bulge_eins+M_disc_eins

M_dark_eins95pos = nquad(integ95pos, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_bulge_eins95pos = nquad(MN95pos, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_disc_eins95pos = nquad(POTDEFdisk_exp95pos, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_eins95pos = M_dark_eins95pos+M_bulge_eins95pos+M_disc_eins95pos

M_dark_eins95neg = nquad(integ95neg, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_bulge_eins95neg = nquad(MN95neg, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_disc_eins95neg = nquad(POTDEFdisk_exp95neg, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_eins95neg = M_dark_eins95neg+M_bulge_eins95neg+M_disc_eins95neg

M_dark_pos_eins= np.abs(M_dark_eins-M_dark_eins95pos)
M_dark_neg_eins= np.abs(M_dark_eins-M_dark_eins95neg)
M_bulge_pos_eins= np.abs(M_bulge_eins-M_bulge_eins95pos)
M_bulge_neg_eins= np.abs(M_bulge_eins-M_bulge_eins95neg)
M_disc_pos_eins= np.abs(M_disc_eins-M_disc_eins95pos)
M_disc_neg_eins= np.abs(M_disc_eins-M_disc_eins95neg)
M_eins_pos= np.abs(M_eins-M_eins95pos)
M_eins_neg= np.abs(M_eins-M_eins95neg)


# In[ ]:

table_data = []
table_para = [r"$M_crit",r"$M_dark_crit",r"$M_bulge_crit",r"$M_disc_crit",r"$M_eins",r"$M_dark_eins",r"$M_bulge_eins",r"$M_disc_eins"]
table_units = [r"solar masses",r"solar masses",r"solar masses",r"solar masses",r"solar masses",r"solar masses",r"solar masses",r"solar masses"]
para = [M_crit, M_dark_crit, M_bulge_crit, M_disc_crit, M_eins, M_dark_eins, M_bulge_eins, M_disc_eins]
parap95=[M_crit_pos, M_dark_pos_crit, M_bulge_pos_crit, M_disc_pos_crit, M_eins_pos, M_dark_pos_eins, M_bulge_pos_eins, M_disc_pos_eins]
paran95=[M_crit_neg, M_dark_neg_crit, M_bulge_neg_crit, M_disc_neg_crit, M_eins_neg, M_dark_neg_eins, M_bulge_neg_eins, M_disc_neg_eins]
index=[r"$M_crit",r"$M_dark_crit",r"$M_bulge_crit",r"$M_disc_crit",r"$M_eins",r"$M_dark_eins",r"$M_bulge_eins",r"$M_disc_eins"]


for i in range(len(para)):
	table_data.append([table_para[i], table_units[i], para[i], parap95[i], paran95[i]])

column_name = [r"PARAMETER", r"UNITS", r"FIT", r"95%(+)", r"95%(-)"]	
table_p = pd.DataFrame(table_data, index=index, columns=column_name)
table_p.to_csv("mass_values.txt", sep='\t', encoding='utf-8')
print (table_p)
print("R_crit=", RAD_CRIT/FC)
print ("\n#####################################################################")
print ("\nDone")
print ("\n#####################################################################\n")




