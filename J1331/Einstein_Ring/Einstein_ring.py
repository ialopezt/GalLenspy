import numpy as np 
import pylab as plb
import matplotlib.pyplot as plt #for graphics
from scipy.integrate import quad
from scipy.integrate import nquad
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.misc import derivative
import pandas as pd
from astropy import table as Table # For fast and easy reading / writing with tables using numpy library

alpha = np.linspace(0,2*np.pi,10000)

tt=Table.Table.read('Cosmological_distances.txt', format='ascii.tab') # importando los datos de distancias cosmológicas

#Importando distancias cosmológicas y Sigma Crítico
D_ds=tt['D_ds'][0] 
D_d=tt['D_d'][0]
D_s=tt['D_s'][0]
SIGMA_CRIT=tt['SIGMA_CRIT'][0]

tt=Table.Table.read('parameters_lens_source.txt', format='ascii.tab') # importando los datos de distancias cosmológicas

#Importando distancias cosmológicas y Sigma Crítico
Sigma_0=tt['FIT'][2] 
h_r=26

#Lens-plane
N = 100
FC = np.pi/(180*3600) #conversion factor between arcs and radians
Theta1 = np.linspace(-5*FC,5*FC,N)
Theta2 = np.linspace(-5*FC,5*FC,N)
Theta = np.zeros((N,N), float)
for i in range(N):
    for j in range(N):
        Theta[i,j]=np.sqrt(Theta1[i]**2+Theta2[j]**2)

# Values of circular source 
FC = np.pi/(180*3600) #conversion factor between arcs and radians
H = 0.0
h = H*FC
K = 0.00
K = K*FC
    
Beta1 = h
Beta2 = K

#Galactoncentric radius
R = D_d*Theta

def POTDEF(x):
    def integ(TheTa, Theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)
        kappa = Sigma/SIGMA_CRIT
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(1e12)
    THETA = np.sqrt(x**2 + Theta2[j]**2)
    x = quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]
    return x

GRADPOT1 = np.zeros((N,N), float)
for i in range(N):
    for j in range(N):
        GRADPOT1[i,j] = derivative(POTDEF, Theta1[i], dx=1e-9, order=7)
        

GRADPOT1=GRADPOT1*1e12

def POTDEF(x):
    def integ(TheTa, Theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)
        kappa = Sigma/SIGMA_CRIT 
        return 2*TheTa*np.log(THETA/TheTa)*kappa/(1e12)
    THETA = np.sqrt(x**2 + Theta1[j]**2)
    x = quad(integ, 0, Theta[i,j], limit=100, args=(Theta))[0]
    return x

GRADPOT2 = np.zeros((N,N), float)
for i in range(N):
    for j in range(N):
        GRADPOT2[i,j] = derivative(POTDEF, Theta2[i], dx=1e-9, order=7)

GRADPOT2=GRADPOT2*1e12

theta1, theta2= np.meshgrid(Theta1, Theta2)
fig = plb.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(theta1/FC,theta2/FC,GRADPOT2/FC, rstride=2, cstride=2, color='g')
ax.set_xlabel('Theta1')
ax.set_ylabel('Theta2')
ax.set_zlabel('gradpot2')
plb.savefig('Gradpot2.pdf')

theta1, theta2= np.meshgrid(Theta1, Theta2)
fig = plb.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(theta1/FC,theta2/FC,GRADPOT1/FC, rstride=2, cstride=2, color='g')
ax.set_xlabel('Theta1')
ax.set_ylabel('Theta2')
ax.set_zlabel('gradpot1')
plb.savefig('Gradpot1.pdf')


#Isophots method for obtention of images
X1 = []; X2 = []
h = 0.0
k = h


def theta(X1, X2):
    for i in range(N):
        for j in range(N):
            #Distancia entre cada punto de la grilla del plano de la fuente y el centro de una fuente de radio r
            def X(Theta1, Theta2):
                return np.sqrt(((Theta1[i]-Gradpot1))**2 + ((Theta2[j]-Gradpot2))**2)
            Gradpot1 = GRADPOT1[i,j]
            Gradpot2 = GRADPOT2[j,i]
            if  X(Theta1, Theta2)<=(0.0009*FC):
                X1.append(Theta1[i])
                X2.append(Theta2[j])
            else:
                None
    return np.array(X1), np.array(X2)

THETA1, THETA2 = theta(X1, X2)

fig = plt.figure()
plt.rcParams['figure.figsize'] =(5,5)
ax = fig.add_subplot(111)
plb.plot(THETA1/FC, THETA2/FC, 'og')
plb.plot(Beta1/FC, Beta2/FC, 'or')
ax.set_aspect('equal', adjustable='box')
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
plb.xlim(-2.5,2.5)
plb.ylim(-2.5,2.5)
plb.savefig('Numerical_Images.pdf')

#Obtention of Critical-radius

THETA=np.sqrt(THETA1**2+THETA2**2)
RAD_EINS = np.sum(THETA)/len(THETA)
RAD_EINS_p=np.abs(RAD_EINS-max(THETA))
RAD_EINS_n=np.abs(RAD_EINS-min(THETA))

EINS_1 = RAD_EINS*np.cos(alpha) 
EINS_2 = RAD_EINS*np.sin(alpha) 


fig = plt.figure()
plt.rcParams['figure.figsize'] =(5,5)
ax = fig.add_subplot(111)
plb.plot(EINS_1/FC, EINS_2/FC, '--g')
plb.plot(Beta1/FC, Beta2/FC, '-r')
ax.set_aspect('equal', adjustable='box')
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
plb.xlim(-2.5,2.5)
plb.ylim(-2.5,2.5)
plb.savefig('Einstein_radius_AND_Source.pdf')


tt=Table.Table.read('coordinates.txt', format='ascii.tab') # importando los datos de las imágenes

#Import coordinates of images
theta1=tt['theta1'] 
theta2=tt['theta2']

fig = plt.figure()
plt.rcParams['figure.figsize'] =(5,5)
ax = fig.add_subplot(111)
plb.plot(EINS_1/FC, EINS_2/FC, '--r')
plb.plot(theta1/FC, theta2/FC, 'og')
ax.set_aspect('equal', adjustable='box')
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
plb.legend(['Einstein Ring', 'Lensed images'], loc='upper right', fontsize=8)
plb.xlim(-2.5,2.5)
plb.ylim(-2.5,2.5)
plb.savefig('Einstein_radius_AND_images.pdf')

# In[ ]:

table_data = []

table_para = [r"Einstein Radius"]
table_units = [r"arcs"]
para = [RAD_EINS/FC]
parap68=[RAD_EINS_p/FC]
paran68=[RAD_EINS_n/FC]

for i in range(len(para)):
	table_data.append([table_para[i], table_units[i], para[i], parap68[i], paran68[i]])

column_name = [r"PARAMETER", r"UNITS", r"Eins_Radius", r"Eins_Radius(+)", r"Eins_Radius(-)"]	
table_p = pd.DataFrame(table_data, columns=column_name)
table_p.to_csv("Einstein_radius.txt", sep='\t', encoding='utf-8')
print ("\n#####################################################################")
print(table_p)
print ("\nDone")
print ("\n#####################################################################\n")

