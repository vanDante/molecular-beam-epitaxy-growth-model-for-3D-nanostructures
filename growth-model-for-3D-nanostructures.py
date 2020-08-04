import numpy as np
import matplotlib.pyplot as plt
import os
clear = lambda: os.system('clear')
clear()

#--------------------------------------------------------------------------
# INTRODUCTION
#--------------------------------------------------------------------------

# This code allows to model the molecular beam epitaxy (MBE) of III-N materials on 
# 3D nano- or microstructures.
#
# The model simulates the shell growth on the nano-/microstructure sidefacets 
# as well as the growth on its top facet and the substrate. It is based on 
# a system of coupled one-dimensional(1D) diffusion equations which is solved 
# by a numerical finite difference method (FDM) using the NumPy package for 
# scientific computing with Python. The model is presentedin detail in the 
# paper and complemented in the supplementalinformation (see links below). 
#
# Physical Review Materials:
# https://doi.org/10.1103/PhysRevMaterials.4.013404
#
# ArXiv (open source):
# https://arxiv.org/abs/1907.10358



#--------------------------------------------------------------------------
# INPUT PARAMETERS
#--------------------------------------------------------------------------

# Please insert all relevent input parameters (in the declared units) in 
# this section. There are two types of parameters:
# The 'growth conditions' are given by the experimentally used growth 
# parameters, the chamber geometry of the MBE system, and the spacing of 
# the base nano-/microstructure that was overgrown.
# The 'modeling parameters' determine the modeled materiel growth and can be 
# adjusted until the model describes well the experimental data.


# GROWTH CONDITIONS
#spacing of NW
L=920
d=40
S=800

#Growth conditions
J_Ga =6.5           # impinging Ga flux (in nm/min) - parallel to substrate
J_N =7.3            # impinging N flux (in nm/min) - parallel to substrate
r=7                 # rotation speed (in rounds per min)  
t_total=35          # total growth time (in min)
alpha = 37.5              # angle of cell with respect to substrate normal (in degree)
gamma=144             # angle between Ga and N cell (in degree) - variation not yet implemented


#adatom diffusion coefficient  (in sqcm/s )
diffS =3*10**(-12)     
diffF =4*10**(-10)
diffT =3*10**(-12)       

#mean residence time of adatoms due to incorporation (in seconds)
tauS =0.5       #LS**2/diffS
tauF =0.55     #LF**2/diffF
tauT =0.45  #LT**2/diffT

# Load the experimental data for visual fitting of the shell thickness 
data = np.loadtxt('./G4.csv') 



#--------------------------------------------------------------------------
# CREATE TARGET DIRECTORY FOR MODELLING RESULTS
#--------------------------------------------------------------------------

# Here we use the input parameters as directory name.

#Create lists of name and value of all output parameters
parameterList=['L','d','S','J_Ga','J_N','r','t_total','alpha','gamma','diffS','diffF','diffT','tauS','tauF','tauT']
parameterValues=[L,d,S,J_Ga,J_N,r,t_total,alpha,gamma,diffS,diffF,diffT,tauS,tauF,tauT] #dictionary?

# create folder in which the final results will be stored
directory='./results/'
for element in parameterList:
   directory=directory+'%('+element+')s--'
directory=directory % locals()

if not os.path.exists(directory):
    os.makedirs(directory)
# Print a note if the respective set of parameters has alredy been calulated
else:
    print('!!! This set of parameters has already been calculated !!!')



#--------------------------------------------------------------------------
# PREPARE PARAMETERS FOR MODELLING
#--------------------------------------------------------------------------

#prefactor K=diff*dt/dx^2 has to be below certain value for
#to allow continuous modelling with FDMthe Finite Difference Method.
K=0.4

#angle between Ga and N in rad
alpha = alpha/360*2*np.pi
gamma = gamma/360*2*np.pi    

#fraction of J which impinges on side facet
chiF = np.sin(alpha) /np.cos(alpha)

#Scaling of parameters for modelling
#scale factors
f_length=10    # in nm
f_time=60     # in seconds

#scaling of J, diff, and tau
J_Ga =J_Ga/f_length     
J_N =J_N/f_length   

diffS = diffS/(f_length**2/f_time*10**(-14))   
diffF =diffF/(f_length**2/f_time*10**(-14))
diffT =diffT/(f_length**2/f_time*10**(-14))

tauS = tauS/ f_time         
tauF = tauF/f_time        
tauT = tauT/f_time  
    


#Set spaitial parameters for different regimes
#substrate (regim 1)
L1=L/f_length           #length of regime
nx1=31                  #number of steps in x direction
dx1=L1/(nx1-1)          #step distance dx

#side facet (regime 2)
L2=L/f_length           #length of regime
nx2=31                  #number of steps in x direction
dx2=L2/(nx2-1)          #step distance dx

#top facet (regime 3)
L3=d/f_length           #length of regime
nx3=5                   #number of steps in x direction
dx3=L3/(nx3-1)          #step distance dx

#side facet (regime 4)
L4=L/f_length           #length of regime
nx4=31                  #number of steps in x direction
dx4=L4/(nx4-1)          #step distance dx

#substrate (regim 5)
L5=L/f_length           #length of regime
nx5=31                  #number of steps in x direction
dx5=L5/(nx5-1)          #step distance dx

# ratio between different regimes
c_1=dx1/dx2
c_2=dx2/dx3
c_3=dx3/dx4
c_4=dx4/dx5


#Set number of time steps nt to maximum value needed for continuous calculation 
# -> diff*dt/dx^2 < K   with   dt=t_total/(nt-1)
nt_regimes=np.zeros(3)
nt_regimes[0]=diffS*t_total/K/dx1**2
nt_regimes[1]=diffF*t_total/K/dx2**2
nt_regimes[2]=diffT*t_total/K/dx3**2

nt=np.int(np.amax(nt_regimes))  # maximum of 10000 time steps
if nt< 10000:
    nt=10000
    
#calculate time steps dt 
dt=t_total/(nt-1)

#calculate prefactor for different regimes needed for modelling
m1=dt/dx1**2
m2=dt/dx2**2
m3=dt/dx3**2
m4=dt/dx2**2
m5=dt/dx1**2



#--------------------------------------------------------------------------
# INITIALIZATION OF TARGET PARAMETERS
#--------------------------------------------------------------------------

#adatom concentration U
U1=np.zeros((nx1,nt))
U2=np.zeros((nx2,nt))
U3=np.zeros((nx3,nt))
U4=np.zeros((nx4,nt))
U5=np.zeros((nx5,nt))

#growth rate GR
GR1=np.zeros((nx1,nt))
GR2=np.zeros((nx2,nt))
GR3=np.zeros((nx3,nt))
GR4=np.zeros((nx4,nt))
GR5=np.zeros((nx5,nt))

#potential growth rate GR withou limitation by N flux 
GR1pot=np.zeros((nx1,nt))
GR2pot=np.zeros((nx2,nt))
GR3pot=np.zeros((nx3,nt))
GR4pot=np.zeros((nx4,nt))
GR5pot=np.zeros((nx5,nt))

#grown material thickness
G1=np.zeros(nx1)
G2=np.zeros(nx2)
G3=np.zeros(nx3)
G4=np.zeros(nx4)
G5=np.zeros(nx5)

G2_dt=np.zeros((nx2,nt))

#other parameters
J_N_r2=np.zeros(nt)
b2=np.zeros(nt)
c2=np.zeros(nt)
d2=np.zeros(nt)

J_N_r4=np.zeros(nt)
b4=np.zeros(nt)
c4=np.zeros(nt)
d4=np.zeros(nt)


# start indices of arrays (takes into account that the array size as well as
# the start index for the modelling of a certain regime/facet has to be 
# adjusted with increasing amount of grown material)
k_S=1    #start index considering growth on substrate (impacts regime 2 and 4) 
k_T=1    #start index considering growth on top facet (impacts regime 2 and 4)
k_F=1    #start index considering growth on side facets (impacts regime 3)

# helper index counting increase on right side of regime 3 array over time
k_F_r=np.zeros(nt) 
# helper index counting increase on left side of regime 3 array over time
k_F_l=np.zeros(nt)
# helper index counting increase of regime 2/4 arrays over time
k_T_t=np.zeros(nt)
# helper parameters
add=np.zeros((1,nt))
even=0



#--------------------------------------------------------------------------
# MODELLING - CALCULATE ADATOM CONCENTRATION AND GROWN MATERIAL 
#--------------------------------------------------------------------------

# Here we calculate the adatom concetration U and the grown material G for
# the different regimes 1 (substrate), 2 (side facet), 3 (top facet), 
# 4 (side facet), and 5 (substrate). 


# set initial conditions for time t=0
U1[:,0]=0
U2[:,0]=0
U3[:,0]=0
U4[:,0]=0
U5[:,0]=0

# quotients for flux conservation 
C_1=diffF/diffS*dx1/dx2
C_2=diffT/diffF*dx2/dx3
C_3=diffF/diffT*dx3/dx4
C_4=diffS/diffF*dx4/dx5

# check conditions
if (C_2!=1/C_3) & (C_1!=1/C_4):
    print('Check boundary conditions!!')

for j in range(nt-1):
    
    #CALCULATE j-DEPENDENT PARAMETERS FOR SIDE FACETs (REGIME 2 + 4)
    #parameter c takes rotation posion for growth on side facet into account
    c2[j]=np.sin(2*np.pi*r*j*dt+1/2*np.pi - gamma)
    if c2[j]<0:
        c2[j]=0
        b2[j]=diffF
    else:
        b2[j]=diffF
    c4[j]=np.sin(2*np.pi*r*j*dt+1/2*np.pi)
    if c4[j]<0:
        c4[j]=0
        b4[j]=diffF
    else:
        b4[j]=diffF
    #parameter d takes rotation posion for Ga deposition on side facet into account
    d2[j]=np.sin(2*np.pi*r*j*dt+1/2*np.pi)
    if d2[j]<0:
        d2[j]=0
    d4[j]=np.sin(2*np.pi*r*j*dt+1/2*np.pi + gamma)
    if d4[j]<0:
        d4[j]=0
    #N flux in dependence on rotation position for regime 2 abd 4   
    J_N_r2[j]=c2[j]*chiF*J_N
    J_N_r4[j]=c4[j]*chiF*J_N
    #set counter for lateral growth of regime 3
    if j>0:
        k_F_r[j]=k_F_r[j-1]
        k_F_l[j]=k_F_l[j-1]   
        k_T_t[j]=k_T_t[j-1]    
    
           
    #CALCULATE U FOR SUBSTRATE (REGIME 1)

    #A: calculate boundary conditions at right side
    #growth rate
    GR1[0,j]=U1[0,j]/tauS
    if GR1[0,j]>J_N:
        GR1[0,j]=J_N
    #potential GR without limitation by N flux        
    GR1pot[0,j]=U1[0,j]/tauS
    #calculate grown material for time tmax (growth rate limited by N flux J_N) 
    G1[0]=G1[0]+GR1[0,j]*dt
    # calculate new U
    U1[0,j+1]=0     
    #If U negative: set value to zero
    if U1[0,j+1]<0:
        U1[0,j+1]=0
  
    #B: calculate U for whole i range 
    for i in range(1, nx1-1):
        #growth rate
        GR1[i,j]=U1[i,j]/tauS
        if GR1[i,j]>J_N:
            GR1[i,j]=J_N
        #potential GR without limitation by N flux        
        GR1pot[i,j]=U1[i,j]/tauS
        #calculate grown material for time tmax (growth rate limited by N flux J_N) 
        G1[i]=G1[i]+GR1[i,j]*dt
        # calculate new U
        U1[i,j+1]=U1[i,j]+m1*diffS*(U1[i+1,j]-2*U1[i,j]+U1[i-1,j]) - dt*GR1[i,j] + dt*J_Ga
        #If U negative: set value to zero
        if U1[i,j+1]<0:
            U1[i,j+1]=0
            
    #C: calculate boundary conditions at left side
    #growth rate
    GR1[nx1-1,j]=U1[nx1-1,j]/tauS
    if GR1[nx1-1,j]>J_N:
        GR1[nx1-1,j]=J_N
    #potential GR without limitation by N flux        
    GR1pot[nx1-1,j]=U1[nx1-1,j]/tauS
    #calculate grown material for time tmax (growth rate limited by N flux J_N) 
    G1[nx1-1]=G1[nx1-1]+GR1[nx1-1,j]*dt
    # calculate new U
    U2[1,j+1]=U2[1,j]+m2*b2[j]*(U2[1+1,j]-2*U2[1,j]+U2[0,j]) - dt*GR2[1,j] + dt*d2[j]*chiF*J_Ga
    U1[nx1-1,j+1] = (c_1* U2[1,j+1] + U1[nx1-2,j+1])/(1+c_1)   
    #If U negative: set value to zero
    if U1[nx1-1,j+1]<0:
        U1[nx1-1,j+1]=0
    #calculate grown material for time tmax  (growth rate limited by N flux J_N)
    U1[nx1-1,j] 
        
    
    #CALCULATE U FOR SIDE FACET (REGIME 2)    

    #A: calculate boundary conditions at right side
    #growth rate
    GR2[0,j]=c2[j]/tauF*U2[0,j]
    if GR2[0,j]>J_N_r2[j]:
        GR2[0,j]=J_N_r2[j]
    #potential GR without limitation by N flux
    GR2pot[0,j]=c2[j]/tauF*U2[0,j]
    #calculate grown material for time tmax
    G2[0+k_S-1]=G2[0+k_S-1]+GR2[0,j]*dt
    G2_dt[0+k_S-1,j]=GR2[0,j]*dt  
    # calculate new U
    U2[0,j+1]=U1[nx1-1,j+1]
    #If U negative: set value to zero
    if U2[0,j+1]<0:
        U2[0,j+1]=0
       
    #B:calculate U for whole i range
    for i in range(1, nx2-1):
        #growth rate
        GR2[i,j]=c2[j]/tauF*U2[i,j]
        if GR2[i,j]>J_N_r2[j]:
            GR2[i,j]=J_N_r2[j]
        #potential GR without limitation by N flux
        GR2pot[i,j]=c2[j]/tauF*U2[i,j]
        #calculate grown material for time tmax
        G2[i+k_S-1]=G2[i+k_S-1]+GR2[i,j]*dt
        G2_dt[i+k_S-1,j]=GR2[i,j]*dt
        # calculate new U
        U2[i,j+1]=U2[i,j]+m2*b2[j]*(U2[i+1,j]-2*U2[i,j]+U2[i-1,j]) - dt*GR2[i,j] + dt*d2[j]*chiF*J_Ga
        #If U negative: set value to zero
        if U2[i,j+1]<0:
            U2[i,j+1]=0
          
    #C: calculate boundary conditions at left side
    #growth rate
    GR2[nx2-1,j]=c2[j]/tauF*U2[nx2-1,j]
    if GR2[nx2-1,j]>J_N_r2[j]:
        GR2[nx2-1,j]=J_N_r2[j]
    #potential GR without limitation by N flux
    GR2pot[nx2-1,j]=c2[j]/tauF*U2[nx2-1,j]
    #calculate grown material for time tmax
    G2[nx2-1+k_S-1]=G2[nx2-1+k_S-1]+GR2[nx2-1,j]*dt
    G2_dt[nx2-1+k_S-1,j]=GR2[nx2-1,j]*dt
    # calculate new U
    U3[1,j+1]=U3[1,j]+m3*diffT*(U3[2,j]-2*U3[1,j]+U3[0,j]) - dt*GR3[1,j] + dt*J_Ga
    U2[nx2-1,j+1]=(U2[nx2-2,j+1] + c_2*U3[1,j+1])/(1+c_2)
    #If U negative: set value to zero
    if U2[nx2-1,j+1]<0:
        U2[nx2-1,j+1]=0



    #CALCULATE U FOR TOP FACET (REGIME 3)
    
    #A: calculate boundary conditions at right side
    #growth rate
    GR3[0,j]=U3[0,j]/tauT
    if GR3[0,j]>J_N:
        GR3[0,j]=J_N
    #potential GR without limitation by N flux        
    GR3pot[0,j]=U3[0,j]/tauT
    #calculate grown material for time tmax (growth rate limited by N flux J_N) 
    G3[0]=G3[0]+GR3[0,j]*dt
    # calculate new U
    U3[0,j+1]=U2[nx2-1,j+1]
    #If U negative: set value to zero
    if U3[0,j+1]<0:
        U3[0,j+1]=0

    #B: calculate U for whole i range
    for i in range(1, nx3-1):
        GR3[i,j]=U3[i,j]/tauT
        if GR3[i,j]>J_N:
            GR3[i,j]=J_N
        #potential GR without limitation by N flux        
        GR3pot[i,j]=U3[i,j]/tauT
        #calculate grown material for time tmax (growth rate limited by N flux J_N) 
        G3[i]=G3[i]+GR3[i,j]*dt
        # calculate new U
        U3[i,j+1]=U3[i,j]+m3*diffT*(U3[i+1,j]-2*U3[i,j]+U3[i-1,j]) - dt*GR3[i,j] + dt*J_Ga
        #If U negative: set value to zero
        if U3[i,j+1]<0:
            U3[i,j+1]=0

    #C: calculate boundary conditions at left side
    #growth rate
    GR3[nx3-1,j]=U3[nx3-1,j]/tauT
    if GR3[nx3-1,j]>J_N:
        GR3[nx3-1,j]=J_N
    #potential GR without limitation by N flux        
    GR3pot[nx3-1,j]=U3[nx3-1,j]/tauT
    #calculate grown material for time tmax (growth rate limited by N flux J_N) 
    G3[nx3-1]=G3[nx3-1]+GR3[nx3-1,j]*dt
    # calculate new U
    U4[nx4-2,j+1]=U4[nx4-2,j]+m4*b4[j]*(U4[nx4-3,j]-2*U4[nx4-2,j]+U4[nx4-1,j]) - dt*GR4[nx4-2,j] + dt*d4[j]*chiF*J_Ga
    U3[nx3-1,j+1]=(U3[nx3-2,j+1] + c_3*U4[nx4-2,j+1])/(1+c_3)
    
    #If U negative: set value to zero
    if U3[nx3-1,j+1]<0:
        U3[nx3-1,j+1]=0


    #CALCULATE U FOR SIDE FACET (REGIME 4)    

    #A: calculate boundary conditions at right side
    #growth rate
    GR4[nx4-1,j]=c4[j]/tauF*U4[nx4-1,j]
    if GR4[nx4-1,j]>J_N_r4[j]:
        GR4[nx4-1,j]=J_N_r4[j]
    #potential GR without limitation by N flux
    GR4pot[nx4-1,j]=c4[j]/tauF*U4[nx4-1,j]
    #calculate grown material for time tmax
    G4[nx4-1+(k_S-1)]=G4[nx4-1+(k_S-1)]+GR4[nx4-1,j]*dt
    # calculate new U
    U4[nx4-1,j+1]=U3[nx3-1,j+1]
    #If U negative: set value to zero
    if U4[nx4-1,j+1]<0:
        U4[nx4-1,j+1]=0
       
    #B:calculate U for whole i range
    for i in range(1, nx4-1):
        i=nx4-1-i
        #growth rate
        GR4[i,j]=c4[j]/tauF*U4[i,j]
        if GR4[i,j]>J_N_r4[j]:
            GR4[i,j]=J_N_r4[j]
        #potential GR without limitation by N flux
        GR4pot[i,j]=c4[j]/tauF*U4[i,j]
        #calculate grown material for time tmax
        G4[i+(k_S-1)]=G4[i+(k_S-1)]+GR4[i,j]*dt
        # calculate new Unx4
        U4[i,j+1]=U4[i,j]+m4*b4[j]*(U4[i-1,j]-2*U4[i,j]+U4[i+1,j]) - dt*GR4[i,j] + dt*d4[j]*chiF*J_Ga
        #If U negative: set value to zero
        if U4[i,j+1]<0:
            U4[i,j+1]=0
          
    #C: calculate boundary conditions at left side
    #growth rate
    GR4[0,j]=c4[j]/tauF*U4[0,j]
    if GR4[0,j]>J_N_r4[j]:
        GR4[0,j]=J_N_r4[j]
    #potential GR without limitation by N flux
    GR4pot[0,j]=c4[j]/tauF*U4[0,j]
    #calculate grown material for time tmax
    G4[0+(k_S-1)]=G4[0+(k_S-1)]+GR4[0,j]*dt
    # calculate new U
    U5[1,j+1]=U5[1,j]+m5*diffS*(U5[2,j]-2*U5[1,j]+U5[0,j]) - dt*GR5[1,j] + dt*J_Ga
    U4[0,j+1]=(U4[1,j+1] + c_4*U5[1,j+1])/(1+c_4)
       
  
    #If U negative: set value to zero
    if U4[0,j+1]<0:
        U4[0,j+1]=0


    #CALCULATE U FOR SUBSTRATE (REGIME 5)

    #A: calculate boundary conditions at right side
    #growth rate
    GR5[0,j]=U5[0,j]/tauS
    if GR5[0,j]>J_N:
        GR5[0,j]=J_N
    #potential GR without limitation by N flux        
    GR5pot[0,j]=U5[0,j]/tauS
    #calculate grown material for time tmax (growth rate limited by N flux J_N) 
    G5[0]=G5[0]+GR5[0,j]*dt
    # calculate new U
    U5[0,j+1]=U4[0,j+1]
    #If U negative: set value to zero
    if U5[0,j+1]<0:
        U5[0,j+1]=0
  
    #B: calculate U for whole i range 
    for i in range(1, nx5-1):
        #growth rate
        GR5[i,j]=U5[i,j]/tauS
        if GR5[i,j]>J_N:
            GR5[i,j]=J_N
        #potential GR without limitation by N flux        
        GR5pot[i,j]=U5[i,j]/tauS
        #calculate grown material for time tmax (growth rate limited by N flux J_N) 
        G5[i]=G5[i]+GR5[i,j]*dt
        # calculate new U
        U5[i,j+1]=U5[i,j]+m5*diffS*(U5[i+1,j]-2*U5[i,j]+U5[i-1,j]) - dt*GR5[i,j] + dt*J_Ga
        #If U negative: set value to zero
        if U5[i,j+1]<0:
            U5[i,j+1]=0
            
    #C: calculate boundary conditions at left side
    #growth rate 
    GR5[nx5-1,j]=U5[nx5-1,j]/tauS
    if GR5[nx5-1,j]>J_N:
        GR5[nx5-1,j]=J_N
    #potential GR without limitation by N flux        
    GR5pot[nx5-1,j]=U5[nx5-1,j]/tauS
    #calculate grown material for time tmax (growth rate limited by N flux J_N) 
    G5[nx5-1]=G5[nx5-1]+GR5[nx5-1,j]*dt
    # calculate new U
    U5[nx5-1,j+1]=0    
    #If U negative: set value to zero
    if U5[nx5-1,j+1]<0:
        U5[nx5-1,j+1]=0



    # ADJUSTMENT OF OF ARRAY SHAPES WITH INCREASING MATERIAL GROWTH 

    # adjust array shape and start index k_S for regime 2 and 4 due to 
    # vertical growth on substrate    
    if G1[nx1-1]>=(k_S*dx2):
        G2=np.append(G2, G2[nx2-1+k_S-1])
        G4=np.append(G4, G4[nx4-1+k_S-1])
        G2_dt=np.concatenate((G2_dt, add))
        G2_dt[nx2-1+k_S,j]=G2_dt[nx2-1+k_S-1,j]
        k_S=k_S+1  
            
    # adjust array shape and start index k_F for regime 3 due to lateral growth 
    if (G2[nx2-1]+G4[nx4-1])>=(k_F*dx3):                 
        if even==0:
            #increase size of arrays 
            U3=np.concatenate((U3, add))
            GR3=np.concatenate((GR3, add))
            GR3pot=np.concatenate((GR3pot, add))
            #Set values of new x-point
            G3=np.append(G3, G3[nx3-1])
            L3=L3+dx3
            nx3=nx3+1
            U3[nx3-1,j+1]=U3[nx3-2,j+1]
            GR3[nx3-1,j+1]=GR3[nx3-2,j+1]
            GR3pot[nx3-1,j+1]=GR3pot[nx3-2,j+1]
            even=even+1
            k_F_r[j]=k_F_r[j]+1
        else: 
            #increase size of arrays 
            U3=np.concatenate((add, U3))
            GR3=np.concatenate((add, GR3))
            GR3pot=np.concatenate((add, GR3pot))
            #Set values of new x-point
            G3=np.append(G3[0], G3)
            L3=L3+dx3
            nx3=nx3+1
            U3[0,j+1]=U3[1,j+1]
            GR3[0,j+1]=GR3[1,j+1]
            GR3pot[0,j+1]=GR3pot[1,j+1]
            even=even-1   
            k_F_l[j]=k_F_l[j]+1
        k_F=k_F+1   
             
    #Show progress of calculation in console window
    for i in range(t_total):
        if j==round(i*nt/t_total):
            clear()
            plt.figure(figsize=(16,3))
            plt.subplot(1, 1, 1)
            x = [i*920/30 for i in range(31)]
            plt.scatter(x, f_length*data[7:], c='orange')
            plt.legend(['Experimental data'])
            #plt.plot(x, nt/(j+1)*f_length*G4[k_S-1:nx4-1+k_S+k_T])
            plt.plot(x, f_length*G4[k_S-1:nx4-1+k_S+k_T])
            plt.xlabel('nanowire length (nm)')
            plt.ylabel('grown material (nm)')
            plt.ylim([0,60])
            plt.title(f'Grown material on side facet - after {int(dt*j)} min of growth')
            plt.show()
            




#--------------------------------------------------------------------------
# POSTPROCESSING - SHRINKING OF ARRAY SIZES FOR COMPACT DATA STORAGE
#--------------------------------------------------------------------------

# factor which defines time resolution of data 
# (here the data is reduced to nt/factor time steps)
factor=int(round(nt/10000))

#adatom concentration
U1=U1[:,0::factor]
U2=U2[:,0::factor]
U3=U3[:,0::factor]
U4=U4[:,0::factor]
U5=U5[:,0::factor]

#growth rate GR
GR1=GR1[:,0::factor]
GR2=GR2[:,0::factor]
GR3=GR3[:,0::factor]
GR4=GR4[:,0::factor]
GR5=GR5[:,0::factor]
#
#potential growth rate GR withou limitation by N flux 
GR1pot=GR1pot[:,0::factor]
GR2pot=GR2pot[:,0::factor]
GR3pot=GR3pot[:,0::factor]
GR4pot=GR4pot[:,0::factor]
GR5pot=GR5pot[:,0::factor]

#grown material
G2_dt=G2_dt[:,0::factor]

#other parameters
J_N_r2=J_N_r2[0::factor]
b2=b2[0::factor]
c2=c2[0::factor]
d2=d2[0::factor]

J_N_r4=J_N_r4[0::factor]
b4=b4[0::factor]
c4=c4[0::factor]
d4=d4[0::factor]



#--------------------------------------------------------------------------
# SAVE DATA
#--------------------------------------------------------------------------

#Create Lists for Read & Wrrite
outputList=['L1','nx1','dx1','L2','nx2','dx2','L3','nx3','dx3','L4','nx4','dx4','L5','nx5','dx5','nt_regimes','nt','dt','m1','m2','m3','m4','m5','U1','U2','U3','U4','U5','GR1','GR2','GR3','GR4','GR5','GR1pot','GR2pot','GR3pot','GR4pot','GR5pot','G1','G2','G3','G4','G5','G2_dt','J_N_r2','b2','c2','d2','J_N_r4','b4','c4','d4','k_S','k_T','k_F','k_F_r','k_F_l','k_T_t']
outputValues=[L1,nx1,dx1,L2,nx2,dx2,L3,nx3,dx3,L4,nx4,dx4,L5,nx5,dx5,nt_regimes,nt,dt,m1,m2,m3,m4,m5,U1,U2,U3,U4,U5,GR1,GR2,GR3,GR4,GR5,GR1pot,GR2pot,GR3pot,GR4pot,GR5pot,G1,G2,G3,G4,G5,G2_dt,J_N_r2,b2,c2,d2,J_N_r4,b4,c4,d4,k_S,k_T,k_F,k_F_r,k_F_l,k_T_t]

#write
for i in range(len(outputList)):
    name=outputList[i]
    if type(outputValues[i]).__name__ != 'ndarray':
        value=[outputValues[i]]
    else:
        value=outputValues[i]
    fname=directory+'/'+name+'.csv'
    np.savetxt(fname, value, delimiter=",")
    


        
#--------------------------------------------------------------------------
# PLOT GROWN MATERIAL ON THE SUBSTRATE, SIDE FACET, AND TOP FACET
#--------------------------------------------------------------------------

# Here we plot the modeled grown material and compare the modeled  
# shell shape to the experimental one. This is only a quick check whether
# our model describes well the experimental data. To evaluate the final 
# quality of the model, also other output parameters (see long list above) 
# have to be taken into account and have to  be checked for physical 
# reasonablity. A detailed analysis of this model being applied to GaN
# growth on GaN nanostructures can befound in the sientific articles mentioned
# in the introduction.

plt.rcParams.update({'font.size': 18})

##plot - grown material on side facet (regim 4)
clear()
plt.figure(figsize=(16,3))
plt.subplot(1, 1, 1)
x = [i*920/30 for i in range(31)]
plt.scatter(x, f_length*data[k_S-1:nx4-1+k_S+k_T],c='orange')
plt.legend(['Experimental data'])
plt.plot(x, f_length*G4[k_S-1:nx4-1+k_S+k_T])
plt.xlabel('nanowire length (nm)')
plt.ylabel('grown material (nm)')
plt.title(f'Grown material on side facet - after full {t_total} min of growth')
plt.show()








