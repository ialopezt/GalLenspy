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

tt=Table.Table.read('Cosmological_distances.txt', format='ascii.tab') # importando los datos de distancias cosmológicas
#Importando distancias cosmológicas y Sigma Crítico
D_ds=tt['D_ds'][0] 
D_d=tt['D_d'][0]
D_s=tt['D_s'][0]
SIGMA_CRIT=tt['SIGMA_CRIT'][0]


tt=Table.Table.read('parameters_MCMC.txt', format='ascii.tab') # importando los datos de las imágenes

#Import coordinates of images

FIT=tt['FIT'] 
r_s = FIT[0]
m_0 = FIT[1]
Sigma_0 = FIT[2]
h_r = FIT[3]
b = FIT[4]
M = FIT[5]
a = 0


#Uncertainties of each parameter

p=tt['95%(+)']
r_s95pos = r_s+p[0]
m_095pos = m_0+p[1]
Sigma_095pos = Sigma_0+p[2] 
h_r95pos = h_r+p[3]
b95pos = b+p[4]
a95pos = a
M95pos = M+p[5]

n=tt['95%(-)']
r_s95neg = r_s-n[0]
m_095neg = m_0-n[1]
Sigma_095neg = Sigma_0-n[2] 
h_r95neg = h_r-n[3]
b95neg = b-n[4]
a95neg = a
M95neg = M-n[5]

#Superficial densities central

def integ(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    NFW_p = NFWPotential(amp=m_0, a=r_s, normalize=False)
    Sigma = NFW_p.dens(R,0.)
    return Sigma
def POTDEFdisk_exp(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    Sigma = Sigma_0*np.exp(-D_d*TheTa/h_r) #Superficial density
    return Sigma
def MN(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    MN_Bulge_p= MiyamotoNagaiPotential(amp=M,a=a,b=b,normalize=False)
    Sigma = MN_Bulge_p.dens(R,0.)
    return Sigma

#Superficial densities 95pos

def integ95pos(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    NFW_p = NFWPotential(amp=m_095pos, a=r_s95pos, normalize=False)
    Sigma = NFW_p.dens(R,0.)
    return Sigma

def POTDEFdisk_exp95pos(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    Sigma = Sigma_095pos*np.exp(-D_d*TheTa/h_r95neg) #Superficial density
    return Sigma

def MN95pos(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    MN_Bulge_p= MiyamotoNagaiPotential(amp=M95pos,a=a95pos,b=b95pos,normalize=False)
    Sigma = MN_Bulge_p.dens(R,0.)
    return Sigma

#Superficial densities 95neg
def integ95neg(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    NFW_p = NFWPotential(amp=m_095neg, a=r_s95neg, normalize=False)
    Sigma = NFW_p.dens(R,0.)
    return Sigma

def POTDEFdisk_exp95neg(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    Sigma = Sigma_095neg*np.exp(-D_d*TheTa/h_r95pos) #Superficial density
    return Sigma

def MN95neg(TheTa1, TheTa2):
    TheTa = np.sqrt(TheTa1**2+TheTa2**2)
    R = D_d*TheTa
    MN_Bulge_p= MiyamotoNagaiPotential(amp=M95neg,a=a95neg,b=b95neg,normalize=False)
    Sigma = MN_Bulge_p.dens(R,0.)
    return Sigma

#Enclosed Mass
 
FC = np.pi/(180*3600) #conversion factor between arcs and radians
radius=float(input("\nEnter the radius of the enclosed mass in arcs:\n"))
lim = radius*FC
M_dark = nquad(integ, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_bulge = nquad(MN, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_disc = nquad(POTDEFdisk_exp, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_bar = M_bulge+M_disc
M = M_dark+M_bulge+M_disc

M_dark_95pos = nquad(integ95pos, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_bulge_95pos = nquad(MN95pos, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_disc_95pos = nquad(POTDEFdisk_exp95pos, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_bar_95pos = M_bulge_95pos+M_disc_95pos
M_95pos = M_dark_95pos+M_bulge_95pos+M_disc_95pos

M_dark_95neg = nquad(integ95neg, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_bulge_95neg = nquad(MN95neg, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_disc_95neg = nquad(POTDEFdisk_exp95neg, [[0, 2*lim],[0, 2*lim]])[0]*(D_d**2)
M_bar_95neg = M_bulge_95neg+M_disc_95neg
M_95neg = M_dark_95neg+M_bulge_95neg+M_disc_95neg

M_dark_pos= np.abs(M_dark-M_dark_95pos)
M_dark_neg= np.abs(M_dark-M_dark_95neg)
M_bulge_pos= np.abs(M_bulge-M_bulge_95pos)
M_bulge_neg= np.abs(M_bulge-M_bulge_95neg)
M_disc_pos= np.abs(M_disc-M_disc_95pos)
M_disc_neg= np.abs(M_disc-M_disc_95neg)
M_bar_pos=np.abs(M_bar-M_bar_95pos)
M_bar_neg=np.abs(M_bar-M_bar_95neg)
M_pos= np.abs(M-M_95pos)
M_neg= np.abs(M-M_95neg)

logM_dark = np.log10(M_dark)
logM_bar = np.log10(M_bar)
logM_bulge = np.log10(M_bulge)
logM_disc = np.log10(M_disc)
logM = np.log10(M)

logM_dark_pos = np.log10(M_dark_pos)
logM_bar_pos = np.log10(M_bar_pos)
logM_bulge_pos = np.log10(M_bulge_pos)
logM_disc_pos = np.log10(M_disc_pos)
logM_pos = np.log10(M_pos)

logM_dark_neg = np.log10(M_dark_neg)
logM_bar_neg = np.log10(M_bar_neg)
logM_bulge_neg = np.log10(M_bulge_neg)
logM_disc_neg = np.log10(M_disc_neg)
logM_neg = np.log10(M_neg)


# In[ ]:


table_data = []
table_para = [r"$M",r"$M_dark",r"$M_bulge",r"$M_disc",r"$M_bar",r"$logM",r"$logM_dark",r"$logM_bulge",r"$logM_disc",r"$logM_bar"]
table_units = [r"solar masses",r"solar masses",r"solar masses",r"solar masses",r"solar masses","-","-","-","-","-"]
para = [M, M_dark, M_bulge, M_disc, M_bar,logM, logM_dark, logM_bulge, logM_disc, logM_bar]
parap95=[M_pos, M_dark_pos, M_bulge_pos, M_disc_pos, M_bar_pos, "-", "-", "-", "-", "-"]
paran95=[M_neg, M_dark_neg, M_bulge_neg, M_disc_neg, M_bar_neg, "-", "-", "-", "-","-"]
index=[r"$M",r"$M_dark",r"$M_bulge",r"$M_disc",r"$M_bar",r"$logM",r"$logM_dark",r"$logM_bulge",r"$logM_disc",r"$logM_bar"]


for i in range(len(para)):
	table_data.append([table_para[i], table_units[i], para[i], parap95[i], paran95[i]])

column_name = [r"PARAMETER", r"UNITS", r"FIT", r"95%(+)", r"95%(-)"]	
table_p = pd.DataFrame(table_data, index=index, columns=column_name)
table_p.to_csv("mass_values.txt", sep='\t', encoding='utf-8')
print (table_p)
print ("\n#####################################################################")
print ("\nDone")
print ("\n#####################################################################\n")








