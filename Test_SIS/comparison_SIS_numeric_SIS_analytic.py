#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Main libraries
import numpy as np 
import pylab as plb
import matplotlib.pyplot as plt #for graphics
from scipy.integrate import quad
from scipy.integrate import nquad
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.misc import derivative


# In[2]:


#parameters main
D_ds = 1 #Distancie lens-source 
D_d = 1 #Distancie observer-lens
D_s = D_ds + D_d #Distance observer-source
c = 1 #light velocity in natural units
o = 1 #dispersion velocity
G = 1 #Gravitation constant in natural units
#Lens-plane
N = 100
valmax = 10
Theta1 = np.linspace(-valmax, valmax, N)
Theta2 = np.linspace(-valmax, valmax, N)


# In[3]:


# Values of circular source 
r = 1 #radius
h = 0.8 #center in x
k = h #center in y
y0 = np.sqrt(h**2 + k**2)


# In[4]:


# SIS ANALYTICS


# In[5]:


Theta = np.zeros((N,N), float)
for i in range(N):
    for j in range(N):
        Theta[i,j] = np.sqrt(Theta1[i]**2 + Theta2[j]**2)

#Deflector potential with analytical solution to the lens equation        
potdef = np.zeros((N, N), float)
for i in range(N):
    for j in range(N):
        potdef[i,j] = 4*np.pi*(o**2)*D_ds*np.sqrt(Theta1[i]**2 + Theta2[j]**2)/((c**2)*D_s)

#Deflection angle with analytical solution to the lens equation        
gradpot1 = np.zeros((N, N), float)
gradpot2 = np.zeros((N, N), float)

for i in range(N):
    for j in range(N):
        gradpot1[i,j] = 4*np.pi*(o**2)*D_ds*Theta1[i]/((c**2)*D_s*np.sqrt(Theta1[i]**2 + Theta2[j]**2))
        gradpot2[i,j] = 4*np.pi*(o**2)*D_ds*Theta2[i]/((c**2)*D_s*np.sqrt(Theta1[j]**2 + Theta2[i]**2))


# In[6]:


#Values of deflector potential for SIS profile
theta1, theta2= np.meshgrid(Theta1, Theta2)
fig = plb.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(theta1,theta2,potdef, rstride=2, cstride=2, color='r')
ax.set_xlabel('Theta1')
ax.set_ylabel('Theta2')
ax.set_zlabel('podef')
plb.savefig('Deflector_Potential_AnalyticsSolution.pdf')


# In[7]:


#Values of deflector angle for SIS profile
theta1, theta2= np.meshgrid(Theta1, Theta2)
fig = plb.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(theta1,theta2,gradpot1, rstride=2, cstride=2, color='r')
ax.set_xlabel('Theta1')
ax.set_ylabel('Theta2')
ax.set_zlabel('podef')
plb.savefig('Deflector_Angle1_AnalyticsSolution.pdf')



# In[8]:

theta1, theta2= np.meshgrid(Theta1, Theta2)
fig = plb.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(theta1,theta2,gradpot2, rstride=2, cstride=2, color='b')
ax.set_xlabel('Theta1')
ax.set_ylabel('Theta2')
ax.set_zlabel('podef')
plb.savefig('Deflector_Angle2_AnalyticsSolution.pdf')


# In[9]:


#Isophots method for obtention of images
X1 = []; X2 = []
r = 1
h = 0.8
k = h

#Función para obtener los valores de theta ante determinada posición de la fuente
def theta(X1, X2):
    for i in range(N):
        for j in range(N):
            #Distancia entre cada punto de la grilla del plano de la fuente y el centro de una fuente de radio r
            def X(Theta1, Theta2):
                return np.sqrt(((Theta1[i]-Gradpot1)-h)**2 + ((Theta2[j]-Gradpot2)-k)**2)
            Gradpot1 = gradpot1[i,j]
            Gradpot2 = gradpot2[j,i]
            if  X(Theta1, Theta2)<=r+0.09 and X(Theta1, Theta2)>=r-0.09:
                X1.append(Theta1[i])
                X2.append(Theta2[j])
            else:
                None
    return np.array(X1), np.array(X2)


# In[10]:


theta1, theta2 = theta(X1, X2)


# In[11]:


alpha = np.linspace(0, 2*np.pi, N)
Beta = r
Beta1 = Beta*np.cos(alpha)+h
Beta2 = Beta*np.sin(alpha)+k


# In[12]:
fig = plt.figure()
plt.rcParams['figure.figsize'] =(5,5)
plb.plot(theta1, theta2, 'xg')
plb.plot(Beta1, Beta2, '--r')
plb.xlabel('Theta1')
plb.ylabel('Theta2 ')
plb.xlim(Theta1[0],Theta1[N-1])
plb.ylim(Theta2[0],Theta2[N-1])
plb.savefig('AnalyticsSolution_Images.pdf')


# In[13]:


# SIS NUMÉRICS
print ("\n#####################################################################")
print ("\nGallenspy")
print ("\n#####################################################################\n")


# In[14]:


Theta = np.zeros((N,N), float)
for i in range(N):
    for j in range(N):
        Theta[i,j] = np.sqrt(Theta1[i]**2 + Theta2[j]**2)
        
#Galactoncentric radius
R = D_d*Theta
#Volumetric density
def dens(R, z):
    densidad = o**2/(2*np.pi*G*(R**2+z**2))
    return 2*densidad


# In[15]:


#Mass superficial density
def Sigma(R):
    def I(R):
        return quad(dens, 0, np.inf, limit=N, args=(R))[0]
    vec_I = np.vectorize(I) 
    P = vec_I(R) 
    return P


# In[16]:


#Obtention of numerical Sigma
Sigma = Sigma(R)/D_d


# In[17]:


#Analytical Sigma
sigma = o**2/(2*G*R)


# In[18]:


#Comparison between analytical Sigma and numerical Sigma
fig = plt.figure()
plb.plot(R, Sigma, 'xr')
plb.plot(R, sigma, '-m')
plb.savefig('AnalyticsSigma_NumericalSigma.pdf')


# In[19]:


#Obtention of kappa
Sigma_crit = (c**2)*D_s/(4*np.pi*D_d*D_ds)
kappa = Sigma/Sigma_crit


# In[20]:


#Obtention of deflector potential
SIGMA_CRIT = (c**2)*D_s/(4*np.pi*D_d*D_ds) #Sigma critics for the convergence

def integ(z,TheTa):
    densidad = o**2/(2*np.pi*G*((D_d*TheTa)**2+z**2)) #Volumetric density
    return 2*TheTa*np.log(Theta[i,j]/TheTa)*densidad/(SIGMA_CRIT*1e12)

I = np.zeros((N,N), float)

for i in range(N):
    for j in range(N):
        I[i,j] = nquad(integ, [[0, np.inf],[0, Theta[i,j]]])[0]
        print(i,j)


# In[21]:


def POTDEF(Theta1, Theta2):
    return 2*1e12*I


# In[22]:


#Values of deflector potential
theta1, theta2= np.meshgrid(Theta1, Theta2)
fig = plb.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(theta1,theta2,POTDEF(Theta1, Theta2), rstride=2, cstride=2, color='r')
ax.set_xlabel('Theta1')
ax.set_ylabel('Theta2')
ax.set_zlabel('potdef')
plb.savefig('Numerical_DeflectorPotential.pdf')


# In[23]:


def POTDEF(x):
    def integ(z,TheTa):
        densidad = o**2/(2*np.pi*G*((D_d*TheTa)**2+z**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT*1e12)
    THETA = np.sqrt(x**2 + Theta2[j]**2)
    x = nquad(integ, [[0, np.inf],[0, Theta[i,j]]])[0]
    return x


# In[24]:


GRADPOT1 = np.zeros((N,N), float)
for i in range(N):
    for j in range(N):
        GRADPOT1[i,j] = derivative(POTDEF, Theta1[i], dx=0.1, order=7)
        print(i,j)


# In[25]:


GRADPOT1 = 2*1e12*GRADPOT1


# In[26]:


#Values of deflection angle
theta1, theta2= np.meshgrid(Theta1, Theta2)
fig = plb.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(theta1,theta2,GRADPOT1, rstride=2, cstride=2, color='r')
ax.set_xlabel('Theta1')
ax.set_ylabel('Theta2')
ax.set_zlabel('gradpot1')
plb.savefig('Numerical_DeflectionAngle1.pdf')


# In[27]:


def POTDEF(x):
    def integ(z,TheTa):
        densidad = o**2/(2*np.pi*G*((D_d*TheTa)**2+z**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT*1e12)
    THETA = np.sqrt(x**2 + Theta1[j]**2)
    x = nquad(integ, [[0, np.inf],[0, Theta[i,j]]])[0]
    return x


# In[28]:


GRADPOT2 = np.zeros((N,N), float)
for i in range(N):
    for j in range(N):
        GRADPOT2[i,j] = derivative(POTDEF, Theta2[i], dx=0.1, order=7)
        print(i,j)


# In[29]:


GRADPOT2 = 2*1e12*GRADPOT2


# In[30]:

theta1, theta2= np.meshgrid(Theta1, Theta2)
fig = plb.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(theta1,theta2,GRADPOT2, rstride=2, cstride=2, color='g')
ax.set_xlabel('Theta1')
ax.set_ylabel('Theta2')
ax.set_zlabel('gradpot1')
plb.savefig('Numerical_DeflectionAngle2.pdf')


# In[31]:


#Isophots method for obtention of images
X1 = []; X2 = []
h = 0.8
k = h
r = 1

def theta(X1, X2):
    for i in range(N):
        for j in range(N):
            #Distancia entre cada punto de la grilla del plano de la fuente y el centro de una fuente de radio r
            def X(Theta1, Theta2):
                return np.sqrt(((Theta1[i]-Gradpot1)-h)**2 + ((Theta2[j]-Gradpot2)-k)**2)
            Gradpot1 = GRADPOT1[i,j]
            Gradpot2 = GRADPOT2[j,i]
            if  X(Theta1, Theta2)<=r+0.09 and X(Theta1, Theta2)>=r-0.09:
                X1.append(Theta1[i])
                X2.append(Theta2[j])
            else:
                None
    return np.array(X1), np.array(X2)


# In[32]:


THETA1, THETA2 = theta(X1, X2)


# In[33]:


alpha = np.linspace(0, 2*np.pi, N)
Beta = r
Beta1 = Beta*np.cos(alpha)+h
Beta2 = Beta*np.sin(alpha)+k


# In[34]:
fig = plt.figure()
plt.rcParams['figure.figsize'] =(5,5)
plb.plot(THETA1, THETA2, 'xg')
plb.plot(Beta1, Beta2, '-r')
plb.xlabel('Theta1')
plb.ylabel('Theta2 ')
plb.xlim(Theta1[0],Theta1[N-1])
plb.ylim(Theta2[0],Theta2[N-1])
plb.savefig('Numerical_Images.pdf')

