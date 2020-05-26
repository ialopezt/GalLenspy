from scipy.misc import *
from PIL import Image
import numpy as np
import pylab as plb
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import nquad
from scipy.misc import derivative
import emcee
import corner
from scipy.optimize import fsolve
from skimage.io import imread


# In[ ]:

I = Image.open("./imagen.png") #para abrir la imagen de la galaxia lensada
I1 = I.convert('L') #pasando la imagen a escala de grises
I1.save('lensed.png') #guardando la imagen a escala de grises en el computador
I2 = imread('lensed.png') #dando la lectura a la imagen de la intensidad de luz de los pixeles que posee.
#plt.imshow(I2,cmap=cm.gray) #evidenciar las coordenadas de la imagen en escala de pixeles


# In[ ]:


#Para conocer las coordenadas de los pixeles, para las imágenes theta#
theta_ra = []
theta_dec = [] 
Theta_ra = []
Theta_dec = [] 

i, j = I2.shape

for i in range(i):
    for j in range(j):
        if I2[i][j]>=200:
            None
        else:
            if i>100 and i<450 and j>100 and j<450:
                theta_dec = i
                theta_ra = j
                Theta_dec.append(theta_dec)
                Theta_ra.append(theta_ra)
                
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(Theta_ra, Theta_dec, 'ob')
plb.xlim(0,500)
plb.ylim(0,500)
#plb.show()


# In[ ]:


#Para conocer las coordenadas de los pixeles, para las imágenes theta#
theta_ra = []
theta_dec = [] 
Theta_ra = []
Theta_dec = [] 

i, j = I2.shape
filtro = 200
for i in range(i):
    for j in range(j):
        if i>100 and i<450 and j>100 and j<300:
            if (I2[i][j]<filtro and I2[i][j-1]>filtro) or (I2[i][j]<filtro and I2[i][j+1]>filtro):
                if (I2[i][j]<filtro and I2[i-1][j]>filtro) or (I2[i][j]<filtro and I2[i+1][j]>filtro):
                    theta_dec = i
                    theta_ra = j
                    Theta_dec.append(theta_dec)
                    Theta_ra.append(theta_ra)
        else:
            None


# In[ ]:


plt.rcParams['figure.figsize'] =(10,10)
plb.plot(Theta_ra, Theta_dec, 'ob')
plb.xlim(0,500)
plb.ylim(0,500)
#plb.show()


# In[ ]:


#Expresando las coordenadas con el orígen en el centro del plano
THETA_RA = []
THETA_DEC = []
for i in range(len(Theta_ra)):
    THETA_RA.append(Theta_ra[i]-546/2)
    THETA_DEC.append(Theta_dec[i]-538/2)


# In[ ]:


plt.rcParams['figure.figsize'] =(10,10)
plb.plot(THETA_RA, THETA_DEC, 'ob')
plb.xlim(-250,250)
plb.ylim(-250,250)
#plb.show()


# In[ ]:


Theta_ra = np.array(THETA_RA)
Theta_dec = np.array(THETA_DEC)
#Obteniendo las coordenadas en segundos de arco
#Equivalencia de 1 pixel por cada segundo de arco es de 0.009#
Theta1 = 0.009*Theta_ra*np.pi/(180*3600)
Theta2 = 0.009*Theta_dec*np.pi/(180*3600)

plt.rcParams['figure.figsize'] =(10,10)
plb.plot(Theta1*1e6, Theta2*1e6, 'ob')
plb.xlim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.ylim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
#plb.show()


# In[ ]:


#Modelando la fuente
r = 0.2*np.pi/(180*3600)
alpha = np.linspace(0,2*np.pi,88*4)
h = -0.4*np.pi/(180*3600)
k = 0*np.pi/(180*3600)
Beta1 = r*np.cos(alpha)+h
Beta2 = r*np.sin(alpha)+k

#Graficando la fuente y las imágenes
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(Beta1*1e6, Beta2*1e6, '-r')
plb.plot(Theta1*1e6, Theta2*1e6, 'ob')
plb.xlim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.ylim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
#plb.show()


# In[ ]:


#Obteniendo algunos bordes de la banana para definir cuadrantes de la circunferencia
for i in range(len(Theta1)):
    if Theta1[i]==min(Theta1):
        print('menorx',i,Theta1[i]*1e6,Theta2[i]*1e6)

for i in range(len(Theta2)):
    if Theta2[i]==min(Theta2):
        print('menory',i,Theta1[i]*1e6,Theta2[i]*1e6)


# In[ ]:


#Valor en y del borde izquierdo-central
y1 = Theta2[34]
#Valor en y del borde inferior-derecho
y3 = Theta2[1]
#Obtener parámetro para altura media de la banana
h = np.abs(y1-y3)
#Valor en y del borde superior-derecho
y2 = h+y1


# In[ ]:


#Trazar 4 rectas para discriminar puntos de la banana por cuadrante

#r1 desde y3 hasta y1
m1 = (y3-y1)/(Theta1[1]-Theta1[34])
b1 = y1 - (m1*Theta1[34])
X1 = Theta1
Y1 = (m1*X1)+b1

#r2 desde y1 hasta y2
m2 = (y2-y1)/(Theta1[1]-Theta1[34])
b2 = y1 - (m2*Theta1[34])
X2 = Theta1
Y2 = (m2*X2)+b2

#r3 desde y1 hasta y2
m3 = 0
b3 = y1
X3 = Theta1
Y3 = (m3*X3)+b3


# In[ ]:


#Obteniendo puntos de la banana para el primer cuadrante
theta1_I=[];theta2_I=[];theta1_II=[];theta2_II=[];theta1_III=[];theta2_III=[];theta1_IV=[];theta2_IV=[]

for i in range(len(Theta1)):
    if Theta2[i] < Y2[i]:
        if Theta2[i] > Y3[i]:
            theta1_I.append(Theta1[i])
            theta2_I.append(Theta2[i])
    if Theta2[i] > Y2[i]:
        theta1_II.append(Theta1[i])
        theta2_II.append(Theta2[i])
    if Theta2[i] < Y1[i]:
        theta1_III.append(Theta1[i])
        theta2_III.append(Theta2[i])
    if Theta2[i] < Y3[i]:
        if Theta2[i] > Y1[i]:
            theta1_IV.append(Theta1[i])
            theta2_IV.append(Theta2[i])


# In[ ]:


#Arrays de los puntos de cada cuadrante de la banana
theta1_I = np.array(theta1_I)
theta2_I = np.array(theta2_I)
theta1_II = np.array(theta1_II)
theta2_II = np.array(theta2_II)
theta1_III = np.array(theta1_III)
theta2_III = np.array(theta2_III)
theta1_IV = np.array(theta1_IV)
theta2_IV = np.array(theta2_IV)


# In[ ]:


# Obtener la medida en grados de los puntos de la banana

#Para escala en grados se tiene que
dy = np.abs(y2-y1)/90

#Asignar los valores angulares V.A. a cada punto de la banana
VA_I = np.zeros(len(theta2_I), float)
VA_II = np.zeros(len(theta2_II), float)
VA_III = np.zeros(len(theta2_III), float)
VA_IV = np.zeros(len(theta2_IV), float)

for i in range(len(theta2_I)):
    VA_I[i] = np.abs(theta2_I[i]-y1)/dy
    
for i in range(len(theta2_II)):
    VA_II[i] = (np.abs(y2-theta2_II[i])/dy)+90

for i in range(len(theta2_III)):
    VA_III[i] = (np.abs(y1-theta2_III[i])/dy)+180

for i in range(len(theta2_IV)):
    VA_IV[i] = (np.abs(y3-theta2_IV[i])/dy)+270    


# In[ ]:


#Introduciendo los valores del ángulo de 270 en la lista de ángulos y en la banana del tercer cuadrante
va_III = np.zeros(len(VA_III)+1,float)
va_III[0] = 270
for i in range(len(VA_III)):
    va_III[i+1] = VA_III[i]

Theta2_III = np.zeros(len(theta2_III)+1,float)
Theta2_III[0] = y3
for i in range(len(theta2_III)):
    Theta2_III[i+1] = theta2_III[i]

Theta1_III = np.zeros(len(theta1_III)+1,float)
Theta1_III[0] = Theta1[1]
for i in range(len(theta1_III)):
    Theta1_III[i+1] = theta1_III[i]


# In[ ]:


#Organizar los puntos de la banana en un orden secuencial con el recorrido de la circunferencia
theta1_I = theta1_I
theta2_I = theta2_I
theta1_II = np.array(list(reversed(theta1_II)))
theta2_II = np.array(list(reversed(theta2_II)))
theta1_III = np.array(list(reversed(Theta1_III)))
theta2_III = np.array(list(reversed(Theta2_III)))
theta1_IV = theta1_IV
theta2_IV = theta2_IV
#Organizar los valores de los ángulos en orden secuencia con el recorrido de la circunferencia
alpha_I = VA_I
alpha_II = np.array(list(reversed(VA_II)))
alpha_III = np.array(list(reversed(va_III)))
alpha_IV = VA_IV


# In[ ]:


#Unificar las coordenadas de la banana de los 4 cuadrantes
theta1 = np.zeros((len(theta1_I)+len(theta1_II)+len(theta1_III)+len(theta1_IV)),float)
theta2 = np.zeros((len(theta1_I)+len(theta1_II)+len(theta1_III)+len(theta1_IV)),float)

for i in range(len(theta1_I)):
    theta1[i]=theta1_I[i]
    theta2[i]=theta2_I[i]
for i in range(len(theta1_II)):
    theta1[i+len(theta1_I)]=theta1_II[i]
    theta2[i+len(theta1_I)]=theta2_II[i]
for i in range(len(theta1_III)):
    theta1[i+len(theta1_I)+len(theta1_II)]=theta1_III[i]
    theta2[i+len(theta1_I)+len(theta1_II)]=theta2_III[i]
for i in range(len(theta1_IV)):
    theta1[i+len(theta1_I)+len(theta1_II)+len(theta1_III)]=theta1_IV[i]
    theta2[i+len(theta1_I)+len(theta1_II)+len(theta1_III)]=theta2_IV[i]

#Unificar los ángulos de los 4 cuadrantes
alpha = np.zeros((len(alpha_I)+len(alpha_II)+len(alpha_III)+len(alpha_IV)),float)

for i in range(len(alpha_I)):
    alpha[i]=alpha_I[i]
for i in range(len(alpha_II)):
    alpha[i+len(alpha_I)]=alpha_II[i]
for i in range(len(alpha_III)):
    alpha[i+len(alpha_I)+len(alpha_II)]=alpha_III[i]
for i in range(len(alpha_IV)):
    alpha[i+len(alpha_I)+len(alpha_II)+len(alpha_III)]=alpha_IV[i]


# In[ ]:


#Expresar los ángulos de la fuente circular en radianes
alpha = alpha*np.pi/180
#Error de cada punto de la banana
sigma = 0.009*np.pi/(180*3600)


# In[ ]:


#Obtención de la probabilidad priori
                          #Definiendo la fuente y sus parámetros principales
R = 0.0014288888196957957
r = R*np.pi/(180*3600)
H = 0.0008851496273404472
h = H*np.pi/(180*3600)
K = 0.0002294107395879724
K = K*np.pi/(180*3600)
    
Beta1 = r*np.cos(alpha)+h
Beta2 = r*np.sin(alpha)+K


# In[ ]:


#Definiendo la lente y sus parámetros principales
D_ds = 1179.6e3
D_d = 497.6e3
D_s = 1510.2e3
SIGMA_CRIT = 4285.3e6 #Sigma crítico para la convergencia
#Espacio paramétrico
N2 = 4
r_s = np.linspace(0.1, 30, N2) #radio de escala del NFW
M_0 = np.linspace(0.1e11, 10e11, N2) #Masa central del NFW

rho_0 = np.zeros((N2,N2), float) #Establendiendo valores de la densidad central del NFW
for i in range(N2):
    for j in range(N2):
        rho_0[i,j]=(M_0[i]/(4*np.pi*(r_s[j]**3)))
#Delimitando valores de las densidades centrales a usar del NFW
Rho_0 = np.zeros((N2,N2), float)
for i in range(N2):
    for j in range(N2):
        Rho_0[i,j] = rho_0[i,j]
#Delimitando valores para el disco exponencial
h_r = np.linspace(2, 6, N2)
Sigma_0 = np.linspace(1e8, 15e8, N2)
#Delimitando valores para el bulbo con perfil Miyamoto-Nagai
b = np.linspace(0.1, 0.5, N2)
M = np.linspace(0.1e10, 1e10, N2)


# In[ ]:


#función del potencial deflector
theta = np.zeros(len(theta1),float)
for i in range(len(theta1)):
    theta[i]=np.sqrt(theta1[i]**2+theta2[i]**2)
    
def POTDEFnfw1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = Rho_0[i,j]/((r/r_s[j])*((1+(r/r_s[j]))**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def POTDEFnfw2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = Rho_0[i,j]/((r/r_s[j])*((1+(r/r_s[j]))**2)) #Densidad volumétrica de masa
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

def MN1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = (3*M[o]/(4*np.pi*(b[p]**3)))*((1+(r**2/b[p]**2))**(-5/2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def MN2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = (3*M[o]/(4*np.pi*(b[p]**3)))*((1+(r**2/b[p]**2))**(-5/2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x


# In[ ]:


#valores iniciales del gradiente del potencial deflector
X = np.zeros((N2,N2,N2,N2,N2,N2,len(theta1)), float)
GRADPOT1nfw = np.zeros((N2,N2,len(theta1)), float)
GRADPOT2nfw = np.zeros((N2,N2,len(theta1)), float)
GRADPOT1disk_exp = np.zeros((N2,N2,len(theta1)), float)
GRADPOT2disk_exp = np.zeros((N2,N2,len(theta1)), float)
GRADPOT1MN = np.zeros((N2,N2,len(theta1)), float)
GRADPOT2MN = np.zeros((N2,N2,len(theta1)), float)
GRADPOT1 = np.zeros((N2,N2,N2,N2,N2,N2,len(theta1)), float)
GRADPOT2 = np.zeros((N2,N2,N2,N2,N2,N2,len(theta1)), float)
THETA1 = np.zeros((N2,N2,N2,N2,N2,N2,len(theta1)), float)
THETA2 = np.zeros((N2,N2,N2,N2,N2,N2,len(theta1)), float)
THETA = np.zeros((N2,N2,N2,N2,N2,N2,len(theta1)), float)
L = np.zeros((N2,N2,N2,N2,N2,N2), float)

#Obteniendo el gradiente del potencial deflector y la función de minimización junto con cada likelihood
for i in range(N2):
    for j in range(N2):
        for l in range(len(theta1)):
            GRADPOT1nfw[i,j,l]= derivative(POTDEFnfw1, theta1[l], dx=1e-9, order=7)
            GRADPOT2nfw[i,j,l]= derivative(POTDEFnfw2, theta2[l], dx=1e-9, order=7)
            print(i,j,l)


# In[ ]:


for m in range(N2):
    for n in range(N2):
        for l in range(len(theta1)):
            GRADPOT1disk_exp[m,n,l]= derivative(POTDEFdisk_exp1, theta1[l], dx=1e-9, order=7)
            GRADPOT2disk_exp[m,n,l]= derivative(POTDEFdisk_exp2, theta2[l], dx=1e-9, order=7)
            print(m,n,l)


# In[ ]:


for o in range(N2):
    for p in range(N2):
        for l in range(len(theta1)):
            GRADPOT1MN[o,p,l] = derivative(MN1, theta1[l], dx=1e-9, order=7)
            GRADPOT2MN[o,p,l] = derivative(MN2, theta2[l], dx=1e-9, order=7)
            print(o,p,l)


# In[ ]:


for i in range(N2):
    for j in range(N2):
        for m in range(N2):
            for n in range(N2):
                for o in range(N2):
                    for p in range(N2):
                        for l in range(len(theta1)):
                            GRADPOT1[i,j,m,n,o,p,l]=2*(SIGMA_CRIT**2)*(GRADPOT1nfw[i,j,l]+GRADPOT1disk_exp[m,n,l]+GRADPOT1MN[o,p,l])
                            GRADPOT2[i,j,m,n,o,p,l]=2*(SIGMA_CRIT**2)*(GRADPOT2nfw[i,j,l]+GRADPOT2disk_exp[m,n,l]+GRADPOT2MN[o,p,l])
                            print(i,j,m,n,o,p,l)


# In[ ]:


for i in range(N2):
    for j in range(N2):
        for m in range(N2):
            for n in range(N2):
                for o in range(N2):
                    for p in range(N2):
                        for l in range(len(theta1)):
                            THETA1[i,j,m,n,o,p,l] = Beta1[l]+GRADPOT1[i,j,m,n,o,p,l]
                            THETA2[i,j,m,n,o,p,l] = Beta2[l]+GRADPOT2[i,j,m,n,o,p,l]
                            THETA[i,j,m,n,o,p,l] = np.sqrt(THETA1[i,j,m,n,o,p,l]**2+THETA2[i,j,m,n,o,p,l]**2)
                            X[i,j,m,n,o,p,l] = ((theta[l]-THETA[i,j,m,n,o,p,l])**2)/(sigma**2)
                            print('GRADPOT1',i,j,m,n,o,p,l)
                        L[i,j,m,n,o,p] = np.sum(X[i,j,m,n,o,p])


# In[ ]:


l = np.zeros((N2,N2,N2,N2,N2), float)
lik = np.zeros((N2,N2,N2,N2), float)
like = np.zeros((N2,N2,N2),float)
LIK = np.zeros((N2,N2),float)
LIKE = np.zeros((N2),float)
Minim = []
R_s = []
Dens_0 = []
H_r = []
SIGMA_0 = []
MASS =[]
B = []

for i in range(N2):
    for j in range(N2):
        for m in range(N2):
            for n in range(N2):
                for o in range(N2):
                    for p in range(N2):
                        l[i,j,m,n,o]=min(L[i,j,m,n,o])
                        lik[i,j,m,n]=min(l[i,j,m,n])
                        like[i,j,m]=min(lik[i,j,m])
                        LIK[i,j]=min(like[i,j])
                        LIKE[i]=min(LIK[i])
                
for i in range(N2):
    for j in range(N2):
        for m in range(N2):
            for n in range(N2):
                for o in range(N2):
                    for p in range(N2):
                        for k in range(len(LIKE)):
                            if L[i,j,m,n,o,p]==LIKE[k]:
                                Minim.append(L[i,j,m,n,o,p])
                                R_s.append(r_s[j])
                                Dens_0.append(Rho_0[i,j])
                                H_r.append(h_r[n])
                                SIGMA_0.append(Sigma_0[m])
                                MASS.append(M[o])
                                B.append(b[p])

for i in range(len(Minim)):
    if Minim[i]==min(Minim):
        Likehood = Minim[i]
        radio_s = R_s[i]
        densidad_0 = Dens_0[i]
        escala_r = H_r[i]
        den_0 = SIGMA_0[i]
        Mass = MASS[i]
        height = B[i]


# In[ ]:


print(Likehood, radio_s, densidad_0, escala_r, den_0, Mass, height)


# In[ ]:

#Visualización de los parámetros en la imágen
theta = np.zeros(len(theta1),float)
for i in range(len(theta1)):
    theta[i]=np.sqrt(theta1[i]**2+theta2[i]**2)

def POTDEFnfw1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = densidad_0/((r/radio_s)*((1+(r/radio_s))**2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def POTDEFnfw2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
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
        densidad = (3*Mass/(4*np.pi*(height**3)))*((1+(r**2/height**2))**(-5/2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def MN2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = (3*Mass/(4*np.pi*(height**3)))*((1+(r**2/height**2))**(-5/2)) #Densidad volumétrica de masa
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

#Obteniendo las coordenadas de las imágenes para los parámetros obtenidos
for l in range(len(theta1)):
    THETA1[l] = Beta1[l]+GRADPOT1[l]
    THETA2[l] = Beta2[l]+GRADPOT2[l]


# In[ ]:


#Graficando la fuente, las imágenes observadas y las de los parámetros
fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(Beta1*1e6, Beta2*1e6, '-r')
plb.plot(theta1*1e6, theta2*1e6, 'ob')
plb.plot(THETA1*1e6, THETA2*1e6, 'og')
plb.xlim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.ylim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
#plb.show()
plb.savefig('Guess_inicial.pdf')


# In[ ]:


#Definir modelo
def model(parameters, theta1, theta2, sigma):
    R_S, m_0, SIGMA_0, H_R, B, MASS  = parameters
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
    def POTDEFdisk_exp1(x):
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
            densidad = (3*MASS/(4*np.pi*(B**3)))*((1+(r**2/B**2))**(-5/2)) #Densidad volumétrica de masa
            return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT**3)
        THETA = np.sqrt(x**2 + theta2[l]**2)
        x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
        return x
    def MN2(x):
        def integ(z,TheTa):
            R = D_d*TheTa
            r = np.sqrt(R**2+z**2)
            densidad = (3*MASS/(4*np.pi*(B**3)))*((1+(r**2/B**2))**(-5/2)) #Densidad volumétrica de masa
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
        GRADPOT1disk_exp[l]= derivative(POTDEFdisk_exp1, theta1[l], dx=1e-9, order=7)
        GRADPOT2disk_exp[l]= derivative(POTDEFdisk_exp2, theta2[l], dx=1e-9, order=7)
        GRADPOT1MN[l]= derivative(MN1, theta1[l], dx=1e-9, order=7)
        GRADPOT2MN[l]= derivative(MN2, theta2[l], dx=1e-9, order=7)    
        GRADPOT1[l]=2*(SIGMA_CRIT**2)*(GRADPOT1nfw[l]+GRADPOT1disk_exp[l]+GRADPOT1MN[l])
        GRADPOT2[l]=2*(SIGMA_CRIT**2)*(GRADPOT2nfw[l]+GRADPOT2disk_exp[l]+GRADPOT2MN[l])
    BETA1 = r*np.cos(alpha)+h
    BETA2 = r*np.sin(alpha)+K
    for l in range(len(theta1)):
        THETA1[l] = BETA1[l]+GRADPOT1[l]
        THETA2[l] = BETA2[l]+GRADPOT2[l]
    THETA_teor = np.zeros((len(theta1)), float)
    for l in range(len(theta1)):
        THETA_teor[l] = np.sqrt(THETA1[l]**2+THETA2[l]**2)
    return THETA_teor

#Función de likelihood
def lnlike(parameters, theta1, theta2, sigma):
    R_S, m_0, SIGMA_0, H_R, B, MASS = parameters
    THETA_teor = model(parameters, theta1, theta2, sigma)
    X = np.zeros((len(theta1)),float)
    for l in range(len(theta1)):
        X[l]=((theta[l]-THETA_teor[l])**2)/(sigma**2)
    return -0.5*np.sum(X)


# In[ ]:


start=np.zeros(6,float)
start[0] = 21
start[1] = 5.5e11
start[2] = 9.67e8
start[3] = 5.3
start[4] = 0.49
start[5] = 1.1e8 


# In[ ]:
#Delimitando valores para el bulbo con perfil Miyamoto-Nagai

def lnprior(parameters):
    R_S, m_0, SIGMA_0, H_R, B, MASS = parameters
    if 20<R_S<60 and 0.1e11<m_0<1.4e12 and 7.5e8<SIGMA_0<30.0e8 and 4.8<H_R<12 and 0.1<B<1 and 0.1e9<MASS<2e10:
        return 0.0
    return -np.inf


# In[ ]:


#Función de probabilidad
def lnprob(parameters, theta1, theta2, sigma):
    lp = lnprior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(parameters, theta1, theta2, sigma)


# In[ ]:


#Dimensión y caminantes
ndim, nwalkers = 6, 100
#Estableciendo posición incial y longitud del paso
pos_step = 1e-8
pos_in = [abs(start + pos_step*start*np.random.randn(ndim)+1e-9*np.random.randn(ndim)) for i in range(nwalkers)]


# In[ ]:


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(theta1, theta2,sigma))


# In[ ]:


sampler.run_mcmc(pos_in, 1000)


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
plb.ylabel(r"$r_s$", fontsize=20)
plb.xlabel(r"$steps$", fontsize=20)
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
plb.ylabel(r"$m_0$", fontsize=20)
plb.xlabel(r"$steps$", fontsize=20)
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
plb.ylabel(r"$Sigma_0$", fontsize=20)
plb.xlabel(r"$steps$", fontsize=20)
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
plb.ylabel(r"$h_r$", fontsize=20)
plb.xlabel(r"$steps$", fontsize=20)
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
plb.ylabel(r"$b$", fontsize=20)
plb.xlabel(r"$steps$", fontsize=20)
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
plb.ylabel(r"$M$", fontsize=20)
plb.xlabel(r"$steps$", fontsize=20)
plb.savefig('para6.pdf')


# In[ ]:


samples = sampler.chain[:, 600:, :].reshape((-1, ndim))


# In[ ]:

percentage = 0.68
fig = corner.corner(samples, labels=["$r_s$", r"$\rho_0$", r"$\Sigma_0$", "$h_r$", "$b$", "$M$"],
                    quantiles = [0.5-0.5*percentage, 0.5, 0.5+0.5*percentage],fill_contours=True, plot_datapoints=True)
fig.savefig("triangle.pdf")


# In[ ]:


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


print(para[0],para[1],para[2],para[3],para[4],para[5])


# In[ ]:


#Visualización de los parámetros en la imágen 
r_s = para[0]
m_0 = para[1]
Sigma_0 = para[2] 
h_r = para[3]
b = para[4]
M = para[5]

#Visualización de los parámetros en la imágen
theta = np.zeros(len(theta1),float)
for i in range(len(theta1)):
    theta[i]=np.sqrt(theta1[i]**2+theta2[i]**2)

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
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r) #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

def POTDEFdisk_exp2(x):
    def integ(TheTa, theta):
        Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r) #Densidad superficial de masa
        return 2*TheTa*np.log(THETA/TheTa)*Sigma/(SIGMA_CRIT)
    THETA = np.sqrt(x**2 + theta1[l]**2)
    x = quad(integ, 0, theta[l], limit=100, args=(theta))[0]
    return x

def MN1(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = (3*M/(4*np.pi*(b**3)))*((1+(r**2/b**2))**(-5/2)) #Densidad volumétrica de masa
        return 2*TheTa*np.log(THETA/TheTa)*densidad/(SIGMA_CRIT)
    THETA = np.sqrt(x**2 + theta2[l]**2)
    x = nquad(integ, [[0, np.inf],[0, theta[l]]])[0]
    return x

def MN2(x):
    def integ(z,TheTa):
        R = D_d*TheTa
        r = np.sqrt(R**2+z**2)
        densidad = (3*M/(4*np.pi*(b**3)))*((1+(r**2/b**2))**(-5/2)) #Densidad volumétrica de masa
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

#Obteniendo las coordenadas de las imágenes para los parámetros obtenidos
for l in range(len(theta1)):
    THETA1[l] = Beta1[l]+GRADPOT1[l]
    THETA2[l] = Beta2[l]+GRADPOT2[l]


# In[ ]:


#Graficando la fuente, las imágenes observadas y las de los parámetros
fig = plt.figure()
plt.rcParams['figure.figsize'] =(10,10)
plb.plot(Beta1*1e6, Beta2*1e6, '-r')
plb.plot(theta1*1e6, theta2*1e6, 'ob')
plb.plot(THETA1*1e6, THETA2*1e6, 'og')
plb.xlim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.ylim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
plb.legend(['Fuente', 'Datos observacionales', 'Valores del modelo'], loc='upper right', fontsize=15)
#plb.show()
plb.savefig('Ajuste_final.pdf')

# In[ ]:


#Definición de parámetros con sus errores
r_s = para[0]; r_s_95pos = parap95[0]; r_s_95neg = paran95[0]; r_s_68pos = parap68[0]; r_s_68neg = paran68[0]
m_0 = para[1]; m_0_95pos = parap95[1]; m_0_95neg = paran95[1]; m_0_68pos = parap68[1]; m_0_68neg = paran68[1]
Sigma_0 = para[2]; Sigma_0_95pos = parap95[2]; Sigma_0_95neg = paran95[2]; Sigma_0_68pos = parap68[2]; Sigma_0_68neg = paran68[2]
h_r = para[3]; h_r_95pos = parap95[3]; h_r_95neg = paran95[3]; h_r_68pos = parap68[3]; h_r_68neg = paran68[3]
b = para[4]; b_95pos = parap95[4]; b_95neg = paran95[4]; b_68pos = parap68[4]; b_68neg = paran68[4]
M = para[5]; M_95pos = parap95[5]; M_95neg = paran95[5]; M_68pos = parap68[5]; M_68neg = paran68[5]


# In[ ]:


print('r_s =', r_s, r_s_95pos, r_s_95neg, r_s_68pos, r_s_68neg)
print('m_0 =', m_0, m_0_95pos, m_0_95neg, m_0_68pos, m_0_68neg)
print('Sigma_0 =', Sigma_0, Sigma_0_95pos, Sigma_0_95neg, Sigma_0_68pos, Sigma_0_68neg)
print('h_r =', h_r, h_r_95pos, h_r_95neg, h_r_68pos, h_r_68neg)
print('b =', b, b_95pos, b_95neg, b_68pos, b_68neg)
print('M =', M, M_95pos, M_95neg, M_68pos, M_68neg)

