#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from galpy.potential import MiyamotoNagaiPotential, NFWPotential, RazorThinExponentialDiskPotential, BurkertPotential # GALPY potentials


# In[ ]:


#Grid of the lens plane

N = 100
Theta1 = np.linspace(-4*np.pi/(180*3600),4*np.pi/(180*3600),N)
Theta2 = np.linspace(-4*np.pi/(180*3600),4*np.pi/(180*3600),N)
Theta = np.zeros((N,N), float)
for i in range(N):
    for j in range(N):
        Theta[i,j]=np.sqrt(Theta1[i]**2+Theta2[j]**2)


# In[ ]:


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


# In[ ]:


#Deflector potential in the all grid
def POTDEFnfw1(x):
    def integ(TheTa, Theta):
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=m_0, a=r_s, normalize=False)
        Sigma = NFW_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + Theta2[j]**2)
    x = quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]
    return x

def POTDEFnfw2(x):
    def integ(TheTa, Theta):
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=m_0, a=r_s, normalize=False)
        Sigma = NFW_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + Theta1[i]**2)
    x = quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]
    return x

def POTDEFdisk_exp1(x):
    def integ(TheTa, Theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r) #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta2[j]**2)
    x = quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, Theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r) #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + Theta1[i]**2)
    x = quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]
    return x

def MN1(x):
    def integ(TheTa, Theta):
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=M,a=a,b=b,normalize=False)
        Sigma = MN_Bulge_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + Theta2[j]**2)
    x = quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]
    return x

def MN2(x):
    def integ(TheTa, Theta):
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=M,a=a,b=b,normalize=False)
        Sigma = MN_Bulge_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + Theta1[i]**2)
    x = quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]
    return x


# In[ ]:


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
#        print('NFW1',i,j)
        
for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        alpha_2NFW[i,j]=derivative(POTDEFnfw2, Theta2[j], dx=1e-9, n=2, order=7)
#        print('NFW2',i,j)
        
for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        alpha_1disc_exp[i,j]=derivative(POTDEFdisk_exp1, Theta1[i], dx=1e-9, n=2, order=7)
#        print('disc_exp1',i,j)
        
for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        alpha_2disc_exp[i,j]=derivative(POTDEFdisk_exp2, Theta2[j], dx=1e-9, n=2, order=7)
 #       print('disc_exp2',i,j)

for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        alpha_1MN[i,j]=derivative(MN1, Theta1[i], dx=1e-9, n=2, order=7)
#        print('MN1',i,j)
        
for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        alpha_2MN[i,j]=derivative(MN2, Theta2[j], dx=1e-9, n=2, order=7)
#        print('MN2',i,j)


# In[ ]:


alpha_1 = (SIGMA_CRIT**2)*(alpha_1NFW+alpha_1disc_exp+alpha_1MN)
alpha_2 = (SIGMA_CRIT**2)*(alpha_2NFW+alpha_2disc_exp+alpha_2MN)


# In[ ]:


def POTDEFnfw(x,y):
    def integ(TheTa, Theta):
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=m_0, a=r_s, normalize=False)
        Sigma = NFW_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + y**2)
    return quad(integ, 0, Theta[i,j], limit=100, args=(Theta[i,j]))[0]

def POTDEFdisk_exp(x,y):
    def integ(TheTa, Theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r) #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + y**2)
    return quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]

def MN(x,y):
    def integ(TheTa, Theta):
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=M,a=a,b=b,normalize=False)
        Sigma = MN_Bulge_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + y**2)
    return quad(integ, 0, Theta[i,j], limit=100, args=(Theta[i,j]))[0]


# In[ ]:


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


# In[ ]:


h = 1e-9

alpha_12nfw = np.zeros((len(Theta1),len(Theta2)),float)
alpha_12disc = np.zeros((len(Theta1),len(Theta2)),float)
alpha_12MN = np.zeros((len(Theta1),len(Theta2)),float)

for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        x = Theta1[i]
        y = Theta2[j]
        alpha_12nfw[i,j]=SEGDERIV_1(x,y)
#        print('NFW1',i,j)
        
for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        x = Theta1[i]
        y = Theta2[j]
        alpha_12disc[i,j]=SEGDERIV_2(x,y)
#        print('disc',i,j)

for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        x = Theta1[i]
        y = Theta2[j]
        alpha_12MN[i,j]=SEGDERIV_3(x,y)
#        print('MN',i,j)


# In[ ]:


alpha_12 = (SIGMA_CRIT**2)*(alpha_12nfw+alpha_12disc+alpha_12MN)


# In[ ]:


#Obtention of the convergence

#Galactocentric radius
R = D_d*Theta

def kappa_NFW(R):
    NFW_p = NFWPotential(amp=m_0, a=r_s, normalize=False)
    Sigma = NFW_p.dens(R,0.)
    kappa = Sigma/SIGMA_CRIT
    return kappa

def kappa_MN(R):
    MN_Bulge_p = MiyamotoNagaiPotential(amp=M,a=a,b=b,normalize=False)
    Sigma = MN_Bulge_p.dens(R,0.)
    kappa = Sigma/SIGMA_CRIT
    return kappa

def kappa_Disc_Exp(R):
    Sigma = Sigma_0*np.exp(-R/h_r)
    kappa = Sigma/SIGMA_CRIT
    return kappa

#Kappa compute

k = kappa_NFW(R)+kappa_MN(R)+kappa_Disc_Exp(R)


# In[ ]:


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


# In[ ]:


thet1, thet2= np.meshgrid(Theta1, Theta2)
fig = plb.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(thet1*1e6,thet2*1e6,k, rstride=2, cstride=2, color='g')
ax.set_xlabel(r"$\theta_1$",fontsize=20)
ax.set_ylabel(r"$\theta_2$",fontsize=20)
ax.set_zlabel(r"$\kappa$",fontsize=20)
#plb.show()
plb.savefig('kappa.pdf')


# In[ ]:


# Get the shear

gamma1 = 0.5*(alpha_1-alpha_2)
gamma2 = alpha_12

gamma = np.sqrt(gamma1**2+gamma2**2) 


# In[ ]:


#Determinant of Jacobian Matriz
detA = (1-k)**2-(gamma**2)
# Evaluation of the matriz for the critical curve

S = np.zeros((len(Theta1), len(Theta2)), float)

for i in range(len(Theta1)):
    for j in range(len(Theta2)):
        if detA[i,j]>0:
            S[i,j]=1
        if detA[i,j]<0:
            S[i,j]=-1


# In[ ]:


crit_1 = []
crit_2 = []

for i in range(len(Theta1)-1):
    for j in range(len(Theta2)-1):
        if np.abs(S[i+1,j]+S[i-1,j]+S[i,j+1]+S[i,j-1])<4:
#            print(i,j)
            crit_1.append(Theta1[i])
            crit_2.append(Theta2[j])
            
crit_1 = np.array(crit_1)
crit_2 = np.array(crit_2)
crit = np.sqrt(crit_1**2+crit_2**2)


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
plt.rcParams['figure.figsize'] =(5,5)
plb.plot(crit_1*1e6, crit_2*1e6, 'or')
plb.xlim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
plb.ylim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
ax.set_aspect('equal', adjustable='box')
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
#plb.show()
plb.savefig('critical_curve.pdf')


# In[ ]:


critx_1 = []; crity_1 = []; critx_2 = []; crity_2 = []

for i in range(len(crit)):
    if crit[i]<=crit[0]+(16*np.pi/(N*(180*3600))) and crit[i]>=crit[0]-(16*np.pi/(N*(180*3600))):
        critx_1.append(crit_1[i])
        crity_1.append(crit_2[i])
    else:
        critx_2.append(crit_1[i])
        crity_2.append(crit_2[i])

critx_1=np.array(critx_1)
crity_1=np.array(crity_1)
critx_2=np.array(critx_2)
crity_2=np.array(crity_2)

crit1 = np.sqrt(critx_1**2+crity_1**2)
crit2 = np.sqrt(critx_2**2+crity_2**2)


# In[ ]:


#Obtention of critical radius
FC = np.pi/(180*3600) #conversion factor between arcs and radians
RAD_CRIT1 = np.sum(crit1)/len(crit1)

theta = np.linspace(0, 2*np.pi, 1000)
CRITx_1 = RAD_CRIT1*np.cos(theta) 
CRITy_1 = RAD_CRIT1*np.sin(theta) 
CRIT1 = np.sqrt(CRITx_1**2+CRITy_1**2)


# In[ ]:


fig = plt.figure()
plt.rcParams['figure.figsize'] =(5,5)
plb.plot(critx_1*1e6, crity_1*1e6, 'or')
plb.plot(CRITx_1*1e6, CRITy_1*1e6, '-b')
plb.xlim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
plb.ylim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
ax.set_aspect('equal', adjustable='box')
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
#plb.show()
plb.savefig('adjust_critical1.pdf')


# In[ ]:


RAD_CRIT2 = np.sum(crit2)/len(crit2)
CRITx_2 = RAD_CRIT2*np.cos(theta) 
CRITy_2 = RAD_CRIT2*np.sin(theta) 
CRIT2 = np.sqrt(CRITx_2**2+CRITy_2**2)


# In[ ]:


fig = plt.figure()
plt.rcParams['figure.figsize'] =(5,5)
plb.plot(critx_2/FC, crity_2/FC, 'or')
plb.plot(CRITx_2/FC, CRITy_2/FC, '-b')
plb.xlim(-3,3)
plb.ylim(-3,3)
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
#plb.show()
plb.savefig('adjust_critical2.pdf')


# In[ ]:


fig = plt.figure()
plt.rcParams['figure.figsize'] =(5,5)
#plb.plot(critx_2/FC, crity_2/FC, 'or')
plb.plot(CRITx_2/FC, CRITy_2/FC, '-b')
plb.plot(CRITx_1/FC, CRITy_1/FC, '-b')
plb.xlim(-3,3)
plb.ylim(-3,3)
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
ax.set_aspect('equal', adjustable='box')
#plb.show()
plb.savefig('adjust_critical2.pdf')


# In[ ]:


#Einstein radius
THETA_EINS = 0.901*FC
THETA_EINS_1 = THETA_EINS*np.cos(theta) 
THETA_EINS_2 = THETA_EINS*np.sin(theta) 


# In[ ]:


#Deflector potential in the critical points
def POTDEFnfw1(x):
    def integ(TheTa, CRIT1):
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=m_0, a=r_s, normalize=False)
        Sigma = NFW_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + CRITy_1[i]**2)
    x = quad(integ, 0, CRIT1[i], limit=100, args=(CRIT1[i]))[0]
    return x

def POTDEFnfw2(x):
    def integ(TheTa, CRIT1):
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=m_0, a=r_s, normalize=False)
        Sigma = NFW_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + CRITx_1[i]**2)
    x = quad(integ, 0, CRIT1[i], limit=100, args=(CRIT1[i]))[0]
    return x

def POTDEFdisk_exp1(x):
    def integ(TheTa, CRIT1):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r) #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + CRITy_1[i]**2)
    x = quad(integ, 0, CRIT1[i], limit=100, args=(Theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, CRIT1):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r) #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + CRITx_1[i]**2)
    x = quad(integ, 0, CRIT1[i], limit=100, args=(Theta))[0]
    return x

def MN1(x):
    def integ(TheTa, CRIT1):
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=M,a=a,b=b,normalize=False)
        Sigma = MN_Bulge_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + CRITy_1[i]**2)
    x = quad(integ, 0, CRIT1[i], limit=100, args=(CRIT1[i]))[0]
    return x

def MN2(x):
    def integ(TheTa, CRIT1):
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=M,a=a,b=b,normalize=False)
        Sigma = MN_Bulge_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + CRITx_1[i]**2)
    x = quad(integ, 0, CRIT1[i], limit=100, args=(CRIT1[i]))[0]
    return x


# In[ ]:


#Lens equation for the obtention of the caustic curve
GRADPOT_1NFW = np.zeros(len(CRIT1), float)
GRADPOT_2NFW = np.zeros(len(CRIT1), float)
GRADPOT_1disc_exp = np.zeros(len(CRIT1), float)
GRADPOT_2disc_exp = np.zeros(len(CRIT1), float)
GRADPOT_1MN = np.zeros(len(CRIT1), float)
GRADPOT_2MN = np.zeros(len(CRIT1), float)

for i in range(len(CRIT1)):
    GRADPOT_1NFW[i]=derivative(POTDEFnfw1, CRITx_1[i], dx=1e-9, order=7)
#    print('NFW1',i)


for i in range(len(CRIT1)):
    GRADPOT_2NFW[i]=derivative(POTDEFnfw2, CRITy_1[i], dx=1e-9, order=7)
#    print('NFW2',i)
        
for i in range(len(CRIT1)):
    GRADPOT_1disc_exp[i]=derivative(POTDEFdisk_exp1, CRITx_1[i], dx=1e-9, order=7)
#    print('disc_exp1',i)
        
for i in range(len(CRIT1)):
    GRADPOT_2disc_exp[i]=derivative(POTDEFdisk_exp2, CRITy_1[i], dx=1e-9, order=7)
#    print('disc_exp2',i)
        
for i in range(len(CRIT1)):
    GRADPOT_1MN[i]=derivative(MN1, CRITx_1[i], dx=1e-9, order=7)
#    print('MN1',i)

for i in range(len(CRIT1)):
    GRADPOT_2MN[i]=derivative(MN2, CRITy_1[i], dx=1e-9, order=7)


# In[ ]:


GRADPOT_1 = (SIGMA_CRIT**2)*(GRADPOT_1NFW +GRADPOT_1disc_exp+GRADPOT_1MN)
GRADPOT_2 = (SIGMA_CRIT**2)*(GRADPOT_2NFW +GRADPOT_2disc_exp+GRADPOT_2MN)


# In[ ]:


CAUSTx_1 = CRITx_1-GRADPOT_1
CAUSTy_1 = CRITy_1-GRADPOT_2


# In[ ]:


#Visualization of the caustic curve
fig = plt.figure()
plb.plot(CAUSTx_1*1e6, CAUSTy_1*1e6, '-r')
plb.xlim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
plb.ylim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
plb.xlabel(r"$\beta_1$", fontsize=20)
plb.ylabel(r"$\beta_2$", fontsize=20)
plb.savefig('caustic1.pdf')


# In[ ]:


#Deflector potential in the critical points
def POTDEFnfw1(x):
    def integ(TheTa, CRIT2):
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=m_0, a=r_s, normalize=False)
        Sigma = NFW_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + CRITy_2[i]**2)
    x = quad(integ, 0, CRIT2[i], limit=100, args=(CRIT2[i]))[0]
    return x

def POTDEFnfw2(x):
    def integ(TheTa, CRIT2):
        R = D_d*TheTa
        NFW_p = NFWPotential(amp=m_0, a=r_s, normalize=False)
        Sigma = NFW_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + CRITx_2[i]**2)
    x = quad(integ, 0, CRIT2[i], limit=100, args=(CRIT2[i]))[0]
    return x

def POTDEFdisk_exp1(x):
    def integ(TheTa, CRIT2):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r) #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + CRITy_2[i]**2)
    x = quad(integ, 0, CRIT2[i], limit=100, args=(Theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, CRIT2):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r) #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + CRITx_2[i]**2)
    x = quad(integ, 0, CRIT2[i], limit=100, args=(Theta))[0]
    return x

def MN1(x):
    def integ(TheTa, CRIT2):
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=M,a=a,b=b,normalize=False)
        Sigma = MN_Bulge_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + CRITy_2[i]**2)
    x = quad(integ, 0, CRIT2[i], limit=100, args=(CRIT2[i]))[0]
    return x

def MN2(x):
    def integ(TheTa, CRIT2):
        R = D_d*TheTa
        MN_Bulge_p= MiyamotoNagaiPotential(amp=M,a=a,b=b,normalize=False)
        Sigma = MN_Bulge_p.dens(R,0.)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(SIGMA_CRIT**2)
    THETA = np.sqrt(x**2 + CRITx_2[i]**2)
    x = quad(integ, 0, CRIT2[i], limit=100, args=(CRIT2[i]))[0]
    return x


# In[ ]:


#Lens equation for the obtention of the caustic curve
GRADPOT_1NFW = np.zeros(len(CRIT2), float)
GRADPOT_2NFW = np.zeros(len(CRIT2), float)
GRADPOT_1disc_exp = np.zeros(len(CRIT2), float)
GRADPOT_2disc_exp = np.zeros(len(CRIT2), float)
GRADPOT_1MN = np.zeros(len(CRIT2), float)
GRADPOT_2MN = np.zeros(len(CRIT2), float)

for i in range(len(CRIT2)):
    GRADPOT_1NFW[i]=derivative(POTDEFnfw1, CRITx_2[i], dx=1e-9, order=7)
#    print('NFW1',i)


for i in range(len(CRIT2)):
    GRADPOT_2NFW[i]=derivative(POTDEFnfw2, CRITy_2[i], dx=1e-9, order=7)
#    print('NFW2',i)
        
for i in range(len(CRIT2)):
    GRADPOT_1disc_exp[i]=derivative(POTDEFdisk_exp1, CRITx_2[i], dx=1e-9, order=7)
#    print('disc_exp1',i)
        
for i in range(len(CRIT2)):
    GRADPOT_2disc_exp[i]=derivative(POTDEFdisk_exp2, CRITy_2[i], dx=1e-9, order=7)
#    print('disc_exp2',i)
        
for i in range(len(CRIT2)):
    GRADPOT_1MN[i]=derivative(MN1, CRITx_2[i], dx=1e-9, order=7)
#    print('MN1',i)

for i in range(len(CRIT2)):
    GRADPOT_2MN[i]=derivative(MN2, CRITy_2[i], dx=1e-9, order=7)


# In[ ]:


GRADPOT_1 = (SIGMA_CRIT**2)*(GRADPOT_1NFW +GRADPOT_1disc_exp+GRADPOT_1MN)
GRADPOT_2 = (SIGMA_CRIT**2)*(GRADPOT_2NFW +GRADPOT_2disc_exp+GRADPOT_2MN)


# In[ ]:


CAUSTx_2 = CRITx_2-GRADPOT_1
CAUSTy_2 = CRITy_2-GRADPOT_2


# In[ ]:


#Visualization of the caustic curve 2
fig = plt.figure()
plt.rcParams['figure.figsize'] =(5,5)
plb.plot(CAUSTx_2*1e6, CAUSTy_2*1e6, '-r')
plb.xlim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
plb.ylim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
plb.xlabel(r"$\beta_1$", fontsize=20)
plb.ylabel(r"$\beta_2$", fontsize=20)
ax.set_aspect('equal', adjustable='box')
plb.savefig('caustic2.pdf')


# In[ ]:


fig = plt.figure()
plt.rcParams['figure.figsize'] =(5,5)
plb.plot(CAUSTx_2*1e6, CAUSTy_2*1e6, '-r')
plb.plot(CAUSTx_1*1e6, CAUSTy_1*1e6, '-r')
plb.xlim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
plb.ylim(-3*np.pi*1e6/(180*3600),3*np.pi*1e6/(180*3600))
plb.xlabel(r"$\beta_1$", fontsize=20)
plb.ylabel(r"$\beta_2$", fontsize=20)
ax.set_aspect('equal', adjustable='box')
plb.savefig('caustic.pdf')


print(RAD_CRIT1, RAD_CRIT2)


