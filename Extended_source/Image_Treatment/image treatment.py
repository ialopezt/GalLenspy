from scipy.misc import *
from PIL import Image #For image treatment 
import numpy as np #for data handling and numerical process
import pylab as plb #Graphics and data
import matplotlib.pyplot as plt #for graphics
from skimage.io import imread #For image treatment
import pandas as pd #for data handling

# In[ ]:

I = Image.open("./imagen.png") #open image of the galaxy
I1 = I.convert('L') #grayscale image
I1.save('lensed.png') #save image in grayscale
I2 = imread('lensed.png') #read the image based on the light intensity for each pixel


# In[ ]:


#To know the coordinates of the pixels for theta images#
theta_ra = []
theta_dec = [] 
Theta_ra = []
Theta_dec = [] 

i, j = I2.shape
filtro = 200 #for the intensity of the pixel
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


#Expressing the coordinates with the origin in the plane center
THETA_RA = []
THETA_DEC = []
for i in range(len(Theta_ra)):
    THETA_RA.append(Theta_ra[i]-546/2)
    THETA_DEC.append(Theta_dec[i]-538/2)

# In[ ]:


Theta_ra = np.array(THETA_RA)
Theta_dec = np.array(THETA_DEC)
#Getting the coordinates in arcsec
#Equivalence between 1 pixel and each arcsec is 0.009#
Theta1 = 0.009*Theta_ra*np.pi/(180*3600)
Theta2 = 0.009*Theta_dec*np.pi/(180*3600)


# In[ ]:


#Modelling the source
r = 0.2*np.pi/(180*3600)
alpha = np.linspace(0,2*np.pi,88*4)
h = -0.4*np.pi/(180*3600)
k = 0*np.pi/(180*3600)
Beta1 = r*np.cos(alpha)+h
Beta2 = r*np.sin(alpha)+k


# In[ ]:


#Getting contours of the arc for determine the circumference quadrants
for i in range(len(Theta1)):
    if Theta1[i]==min(Theta1):
        y1 = Theta2[i]

for i in range(len(Theta2)):
    if Theta2[i]==min(Theta2):
        y3 = Theta2[i]

# In[ ]:

#Getting parameter for the half of the height in the arc
h = np.abs(y1-y3)
#Value in y of upper-right edge 
y2 = h+y1

# In[ ]:


#Trazar 4 rectas para discriminar puntos de la banana por cuadrante

#r1 from y3 to y1
m1 = (y3-y1)/(Theta1[1]-Theta1[34])
b1 = y1 - (m1*Theta1[34])
X1 = Theta1
Y1 = (m1*X1)+b1

#r2 from y1 to y2
m2 = (y2-y1)/(Theta1[1]-Theta1[34])
b2 = y1 - (m2*Theta1[34])
X2 = Theta1
Y2 = (m2*X2)+b2

#r3 from y1 to y2
m3 = 0
b3 = y1
X3 = Theta1
Y3 = (m3*X3)+b3


# In[ ]:

#Getting points of the arc belonging first circumference quadrant 
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

#Arrays of the points of the arc belonging to each circumference quadrant
theta1_I = np.array(theta1_I)
theta2_I = np.array(theta2_I)
theta1_II = np.array(theta1_II)
theta2_II = np.array(theta2_II)
theta1_III = np.array(theta1_III)
theta2_III = np.array(theta2_III)
theta1_IV = np.array(theta1_IV)
theta2_IV = np.array(theta2_IV)


# In[ ]:


#Getting the values in grades for each arc points

#Scale in grades
dy = np.abs(y2-y1)/90

#Assign angular values V.A. to each arc point
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


#Introducing values of 270 angle in array of angles and in the quadrant third of the arc
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


#Organize the points of the arc in a sequential order with the road of the circumference
theta1_I = theta1_I
theta2_I = theta2_I
theta1_II = np.array(list(reversed(theta1_II)))
theta2_II = np.array(list(reversed(theta2_II)))
theta1_III = np.array(list(reversed(Theta1_III)))
theta2_III = np.array(list(reversed(Theta2_III)))
theta1_IV = theta1_IV
theta2_IV = theta2_IV
#Organize the values of angles of the arc in a sequential order with the road of the circumference
alpha_I = VA_I
alpha_II = np.array(list(reversed(VA_II)))
alpha_III = np.array(list(reversed(va_III)))
alpha_IV = VA_IV


# In[ ]:


#Unify the coordinates of the arc in the 4 quadrants
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

plt.rcParams['figure.figsize'] =(10,10)
plb.plot(theta1*1e6, theta2*1e6, 'ob')
plb.xlim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.ylim(-2.5*np.pi*1e6/(180*3600),2.5*np.pi*1e6/(180*3600))
plb.xlabel(r"$\theta_1$", fontsize=20)
plb.ylabel(r"$\theta_2$", fontsize=20)
#plb.show()
plb.savefig('contours.pdf')


#Unify the angles of all quadrants
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


#Express the angles in radians
alpha = alpha*np.pi/180

#Export data of theta1, theta2 and alpha to txt archive
table_data = []

sigma=np.zeros(len(theta1), float)
for i in range(len(theta1)):
    sigma[i]=0.009*np.pi/(180*3600)

for i in range(len(theta1)):
    table_data.append([theta1[i], theta2[i], sigma[i]])

column_name = [r"theta1", r"theta2", r"sigma"]	
table_p = pd.DataFrame(table_data, columns=column_name)
table_p.to_csv("coordinates.txt", sep='\t', encoding='utf-8')
    
table_data = []

for i in range(len(theta1)):
    table_data.append([alpha[i]])

column_name = [r"alpha"]	
table_p = pd.DataFrame(table_data, columns=column_name)
table_p.to_csv("alpha.txt", sep='\t', encoding='utf-8')










