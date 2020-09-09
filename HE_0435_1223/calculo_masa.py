from scipy.misc import *
import numpy as np
import pylab as plb
from scipy.misc import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.integrate import quad
from scipy.integrate import nquad
from scipy.misc import derivative
import pandas as pd
from astropy import table as Table # For fast and easy reading / writing with tables using numpy library

r_s = 52.79263675428489 
m_0 = 1996136344618.109 
Sigma_0 = 2999581178.7873983 
h_r = 11.999473075500532 

r_s95p = 52.79263675428489+0.13369760280055942 
m_095p = 1996136344618.109+3751045579.1853027 
Sigma_095p = 2999581178.7873983+402499.67374420166 
h_r95p = 11.999473075500532+0.00050 

r_s95n = 52.79263675428489-27.01408697990078 
m_095n = 1996136344618.109-1315657836236.3062 
Sigma_095n = 2999581178.7873983-2501373.118262291 
h_r95n = 11.999473075500532-0.00772223 


FC = np.pi/(180*3600)

#Obtención de las densidades superficiales dentro del anillo de Einstein obtenido
densidad_0 = m_0/(4*np.pi*(r_s**3))
def integ(z,TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    r = np.sqrt(R**2+z**2)
    densidad = densidad_0/((r/r_s)*((1+(r/r_s))**2)) #Densidad volumétrica de masa
    return 2*densidad
def POTDEFdisk_exp(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r)/2 #Densidad superficial de masa
    return Sigma

densidad_095p = m_095p/(4*np.pi*(r_s95p**3))
def integ95p(z,TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    r = np.sqrt(R**2+z**2)
    densidad95p = densidad_095p/((r/r_s95p)*((1+(r/r_s95p))**2)) #Densidad volumétrica de masa
    return 2*densidad95p
def POTDEFdisk_exp95p(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    Sigma95p = Sigma_095p*np.exp(-D_d*TheTa/h_r95p)/2 #Densidad superficial de masa
    return Sigma95p

densidad_095n = m_095n/(4*np.pi*(r_s95n**3))
def integ95n(z,TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    r = np.sqrt(R**2+z**2)
    densidad95n = densidad_095n/((r/r_s95n)*((1+(r/r_s95n))**2)) #Densidad volumétrica de masa
    return 2*densidad95n
def POTDEFdisk_exp95n(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    Sigma95n = Sigma_095n*np.exp(-D_d*TheTa/h_r95n)/2 #Densidad superficial de masa
    return Sigma95n



RE = 6.66
D_d = 1163.342e3

re = RE/D_d

lim = re
SIGMA1 = nquad(integ, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]
SIGMA2 = nquad(POTDEFdisk_exp, [[0, 2*lim],[0, 2*lim]])[0]
SIGMA_BAR = 2*SIGMA2*(D_d**2)
SIGMA = 2*(SIGMA1+SIGMA2)*(D_d**2)

SIGMA195p = nquad(integ95p, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]
SIGMA295p = nquad(POTDEFdisk_exp95p, [[0, 2*lim],[0, 2*lim]])[0]
SIGMA_BAR95p = 2*(SIGMA295p-SIGMA2)*(D_d**2)
SIGMA95p = 2*(SIGMA195p-SIGMA1+SIGMA295p-SIGMA2)*(D_d**2)

SIGMA195n = nquad(integ95n, [[0, np.inf],[0, 2*lim],[0, 2*lim]])[0]
SIGMA295n = nquad(POTDEFdisk_exp95n, [[0, 2*lim],[0, 2*lim]])[0]
SIGMA_BAR95n = 2*SIGMA295n*(D_d**2)
SIGMA95n = 2*(SIGMA195n-SIGMA1+SIGMA295n-SIGMA2)*(D_d**2)


# In[48]:

table_data = []
table_para = [r"$M_eins",r"$M_barionic"]
table_units = [r"solar masses",r"solar masses"]
para = [SIGMA, SIGMA_BAR]
parap95=[SIGMA95p, SIGMA_BAR95p]
paran95=[SIGMA95n, SIGMA_BAR95n]
index=[r"$M_eins",r"$M_barionic"]


for i in range(len(para)):
	table_data.append([table_para[i], table_units[i], para[i], parap95[i], paran95[i]])

column_name = [r"PARAMETER", r"UNITS", r"FIT", r"95%(+)", r"95%(-)"]	
table_p = pd.DataFrame(table_data, index=index, columns=column_name)
table_p.to_csv("mass_values.txt", sep='\t', encoding='utf-8')
print (table_p)
print("Fracc_barion=", SIGMA_BAR/SIGMA)
print ("\n#####################################################################")
print ("\nDone")
print ("\n#####################################################################\n")

