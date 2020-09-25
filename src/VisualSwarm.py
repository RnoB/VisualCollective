#MIT License

#Copyright (c) 2019 Renaud Bastien

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
###########




import math
import numpy as np
import numpy.matlib
import time
import uuid
import os
import time
import traceback
import numba
pi = math.pi
pi2 = math.pi/2.



#cartesian to spherical
# input : 
# x0 = (x,y,z)
# output : 
# r = (rho,theta,phi)
@numba.njit(fastmath=True)
def cart2sph(x0,dim = 3):
    if dim == 3:
        r = np.zeros(x0.shape)
        r[:, 0] = np.sqrt(np.power(x0[:, 0], 2) +
                          np.power(x0[:, 1], 2) +
                          np.power(x0[:, 2], 2))
        r2=np.copy(r[:,0])
        r2[r2==0]=1
        r[:, 1] = (pi/2.)-np.arccos(x0[:, 2] / r2)
        r[:, 2] = np.arctan2(x0[:, 1], x0[:, 0])
    elif dim == 2:
        r = np.zeros(x0.shape)
        r[:, 0] = np.sqrt(np.power(x0[:, 0], 2) +
                          np.power(x0[:, 1], 2))
        r2=np.copy(r[:,0])
        r2[r2==0]=1
        
        r[:, 1] = np.arctan2(x0[:, 1], x0[:, 0])
    return(r)



#Computing the visual Field with speherical individuals
# with size R = 1
# input : 
# X position of all individuals in 3d cartesian (x,y,z)
# U velocity of all individuals in 3d cylindrical  (r,phi,z)
# k the focal individual
# nPhi the resolution of the visual field 
# output :
# V the visual field as a Matrix with coordinates (phi,theta)
@numba.njit(fastmath=True)
def visualField3d(X,U,k,nPhi):
    # define the variables in memory
    phi = np.linspace(-pi,pi,2*nPhi+1)
    theta = np.linspace(-pi2,pi2,nPhi+1)
    dPhi     = phi[1]-phi[0]
    s = (len(theta),len(phi))
    V = np.zeros(s)
    Xv = np.zeros(X.shape)
    XvRot = np.zeros(X.shape)
    Xv = X-X[k,:]
    
    #rotate and translate each position around the focal individual

    XvRot[:,0]=Xv[:,0]*np.cos(-U[k,1])-Xv[:,1]*np.sin(-U[k,1])
    XvRot[:,1]=Xv[:,0]*np.sin(-U[k,1])+Xv[:,1]*np.cos(-U[k,1])
    XvRot[:,2]=Xv[:,2]
    Rv = cart2sph(XvRot)
    

    # dT apparent angle
    dT = np.arctan2(1,Rv[:,0])
    N = Xv.shape[0]

    
    #loop through all individuals
    for j in range(0,N):
        if k is not j:
            L = len(np.arange(-dT[j],dT[j],dPhi))
            
            #loop through the apparent angle and calculate the deformation due to the projection on a surface
            for l in np.arange(-dT[j],dT[j]+dT[j]/L,dT[j]/L):
                idxT = round((pi2+Rv[j,1])/dPhi )
                idxPhi = int(round((pi + Rv[j,2])/dPhi  ))
                r = -Rv[j,0]*math.tan(l)

                x = r*math.sin(Rv[j,1])+Rv[j,0]*math.cos(Rv[j,1])

                if 1-r*r<0:
                    y=0
                else:
                    y = np.sqrt(1-r*r)
                
                dP = int(round(np.arctan2(abs(y),abs(x))/dPhi   ))
                

                if abs(l + Rv[j,1])>pi2:
                    idxPhi = int(round(Rv[j,2]/dPhi ))
                    dP =  int(round(np.arctan2(y,-x)/dPhi ))


                l1=round(idxT+(l/dPhi   ))
                if l1>s[0]-1:
                    l1= 2 * s[0]-l1-1
                elif l1<0:
                    l1 = -l1#+1
                for m2 in range(idxPhi-dP,idxPhi+dP+1):
                    
                    
                    if m2 < 0:
                        m2 = m2 + s[1]
                    while m2> s[1]-1:
                        m2 = m2-s[1]
                    
                    V[int(l1),int(m2)] = 1  
    return V



#Computing the visual Field with disk individuals in 2D
# with size R = 1
# input : 
# X position of all individuals in 2d cartesian (x,y)
# U velocity of all individuals in 2d polar  (r,phi)
# k the focal individual
# nPhi the resolution of the visual field 
# output :
# V the visual field as a line with coordinates (phi)

@numba.njit(fastmath=True)
def visualField2d(X,U,k,nPhi,d=1):
    # define the variables in memory
    phi = np.linspace(-pi,pi,2*nPhi+1)
    dPhi     = phi[1]-phi[0]
    s = len(phi)
    V = np.zeros(s)
    Xv = np.zeros(X.shape)
    XvRot = np.zeros(X.shape)
    Xv = X-X[k,:]
    
    #rotate and translate each position around the focal individual

    XvRot[:,0]=Xv[:,0]*np.cos(-U[k,1])-Xv[:,1]*np.sin(-U[k,1])
    XvRot[:,1]=Xv[:,0]*np.sin(-U[k,1])+Xv[:,1]*np.cos(-U[k,1])
    Rv = cart2sph(XvRot,dim = 2)
    

    # dT apparent angle
    #print(np.shape(Xv))
    N = np.shape(Xv)[0]
    
    #loop through all individuals
    for j in range(0,N):
        if k is not j:
            
            idxPhi = int(round((math.pi+Rv[j,1])/dPhi  ))
            


            
            dP = int(round(np.arctan2(d,Rv[j,0])/dPhi   ))
                


            m2 = np.arange(idxPhi-dP,idxPhi+dP+1)
            m2[m2<0]=(2*nPhi+1)+m2[m2<0]
            m2=m2%(2*nPhi+1)
            V[m2] = 1  
    return V


#generate all the simple sin/cos function necessary to compute the variation of direction and speed
# input : 
#  nPhi the resolution of the visual field 
# output :
#  Vcc = cos(theta)* cos(theta) * cos(phi)
#  Vcs = cos(theta)* cos(theta) * sin(phi)
#  Vs  = cos(theta)* sin(theta)



# the first cos(theta) accounts for he deformation due to the projection on a plane
# Vcc is used for planar speed variation
# Vcs for direction variation
# Vs for the vertical component of the speed

def generateVisualFunction(nPhi):
    
    s = (nPhi+1,2*nPhi+1)
    phi = np.zeros((1,s[1]))
    theta = np.zeros((s[0],1))
    phi[0,:] = np.linspace(-pi,pi,s[1])
    theta[:,0] = np.linspace(-pi2,pi2,s[0])
    dPhi     = phi[0,1]-phi[0,0]
    Phi = np.matlib.repmat(phi,s[0],1)
    Theta = np.matlib.repmat(theta,1,s[1])

    Vcc = np.cos(Theta)*np.cos(Theta)*np.cos(Phi)
    Vcs = np.cos(Theta)*np.cos(Theta)*np.sin(Phi)
    Vs = np.cos(Theta)*np.sin(Theta)

    return Vcc,Vcs,Vs,dPhi  

@numba.njit(fastmath=True)
def generateVisualFunction2d(nPhi):
    
    s = 2*nPhi+1
    phi = np.zeros((s))
    phi[:] = np.linspace(-pi,pi,s)
    dPhi     = phi[1]-phi[0]

    Vc = np.cos(phi)
    Vs = np.sin(phi)

    return Vc,Vs,dPhi  

#where to save the data
def pather(expId):

    path = 'd:/VisualModel/3d/data/'
    if not os.path.exists(path):
        os.makedirs(path)
    path = path  + expId+ '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    return path



#initialise randomly the position
# input:
#  N    number of individuals
#  xMax maximal position (-xMax,xMax)
#  ref  coordinates system
# output:
#  X intial positions of the individuals
@numba.njit(fastmath=True)
def initPosition(N,xMax,ref = 'cart',dim=3):
    if dim == 3:
        if ref == 'spherical':
            X=(.5-np.random.rand(N,3))
            X[:,0] = X[:,0] * xMax
            X[:,1] = X[:,1] * pi2
            X[:,2] = X[:,2] * pi
        if ref == 'cylindrical':
            X=(np.random.rand(N,3))
            X[:,0] = X[:,0] * xMax
            X[:,1] = (X[:,1]-.5) * 4*pi
            X[:,2] = X[:,2] 
        else:
            X=xMax*(.5-np.random.rand(N,3))
    elif dim == 2:
        if ref == 'spherical':
            X=(.5-np.random.rand(N,2))
            X[:,0] = X[:,0] * xMax
            X[:,1] = X[:,1] * pi2
            
        if ref == 'cylindrical':
            X=(np.random.rand(N,2))
            X[:,0] = X[:,0] * xMax
            X[:,1] = (X[:,1]-.5) * 4*pi
            
        else:
            X=xMax*(.5-np.random.rand(N,2))
    return X


# write a csv file with all the position
#  input:
#   X matrix to write

# Sometimes open() cannot find the file. No reason why so we just loop it back

def writeCsv(X,N,U,name,path,writeMode = 0):
    filepath = os.path.join(path, name)
    if writeMode == 0:
        fd = open(filepath,'wb')
    elif writeMode == 1:
        checkFile = True
        while checkFile:
            try:
                fd = open(filepath,'ab')
                checkFile = False
            except Exception as e:
                traceback.print_exc()
                print()
    nId=np.zeros((10,1))
    nId = np.arange(0,N)
    wri = np.c_[nId,X]
    wri = np.c_[wri,U]
    
    np.savetxt(fd,np.c_[nId,X,U],delimiter=',',fmt= '%5.5f')

    fd.close()



#integrate the visual field
# inut :
#  V     visual Field
#  dV    spatail derivative of the Visual Field 
#  Vcc = cos(theta)* cos(theta) * cos(phi)
#  Vcs = cos(theta)* cos(theta) * sin(phi)
#  Vs  = cos(theta)* sin(theta)
# dPhi   differential of the anngle of the visual field

@numba.njit(fastmath=True)
def parameterVision(V,dV,Vcc,Vcs,Vs,dPhi):
    Vup=np.zeros((2,3))
    Vup[0,0]=np.sum(V*Vcc*dPhi    *dPhi   )
    Vup[0,1]=np.sum(V*Vcs*dPhi    *dPhi   )
    Vup[0,2]=np.sum(V*Vs  *dPhi   *dPhi   )
    Vup[1,0]=np.sum(dV*Vcc*dPhi   )
    Vup[1,1]=np.sum(dV*Vcs*dPhi   )
    Vup[1,2]=np.sum(dV*Vs *dPhi   )
    return Vup


@numba.njit(fastmath=True)
def parameterVision2d(V,dV,Vc,Vs,dPhi):
    Vup=np.zeros((2,2))

    Vup[0,0]=np.sum(V*Vc*dPhi   )
    Vup[0,1]=np.sum(V*Vs*dPhi   )
    Vup[1,0]=np.sum(dV*Vc   )
    Vup[1,1]=np.sum(dV*Vs   )

    return Vup


@numba.njit(fastmath=True)
def derVis(V):
    dV = np.hstack((V[:,-2:-1],V[:,0:-1]))-np.hstack((V[:,1:],V[:,0:1]))
    dV = np.abs(dV/2.0)

    return dV

# visual Field Model
@numba.njit(fastmath=True)
def updatePosSpeed(X,U,N,Vu,Vp,Vz,Vuu,Vpp,Vzz,dVu,dVp,dVz,dU,v0,drag,Vcc,Vcs,Vs,dPhi,nPhi,dt):
    for k in range (0,N):
        V = visualField(X,U,k,nPhi)
        #dV = abs(np.roll(V,1,axis=1)-np.roll(V,-1,axis=1))/2.0
        dV = derVis(V)

        Vup = parameterVision(V,dV,Vcc,Vcs,Vs,dPhi)
        dU[k][0,0] = (drag * (v0 - U[k,0] ) + Vuu * ( Vu * Vup[0,0] + dVu * Vup[1,0] ) )
        dU[k][0,1] =  Vpp * (( Vp * Vup[0,1] + dVp * Vup[1,1] ));
        dU[k][0,2] =  ( -drag * U[k,2] + Vzz * ( Vz * Vup[0,2] + dVz * Vup[1,2] ));  


    for k in range (0,N):
        U[k,0] = U[k,0] + dU[k][0,0] *dt;
        U[k,1] = U[k,1] + dU[k][0,1] *dt;
        U[k,2] = U[k,2] + dU[k][0,2] *dt;
    
        X[k,0]=X[k,0]+U[k,0]*np.cos(U[k,1])*dt;
        X[k,1]=X[k,1]+U[k,0]*np.sin(U[k,1])*dt;
        X[k,2] = X[k,2]+ U[k,2]*dt;
    return X,U

def visModel(N,xMax=10,nPhi=512,v0=1,Vu=-1,Vp=-1,Vz=-1,dVu=.1,dVp=.1,dVz=.1,Vuu=1,Vpp=2,Vzz=1,drag=.1,dt=.1,tMax = 1e3,expId='test',dims=3,tMax=1e5,dataPath = "./"):

    
    numpy.random.seed(int((time.time()-1520000000)*50))


    path = pather(expId,dataPath)
    running = True

    X = initPosition(N,xMax,dim=dims)
    U = initPosition(N,v0,'cylindrical',dim=dims)

    writeCsv(X,N,U,'position.csv',path)
    if dim == 3:
        Vcc,Vcs,Vs,dPhi  = generateVisualFunction(nPhi)



        Vup = []
        dU = []
        for k in range (0,N):
            Vup.append(np.zeros((2,3)))

            dU.append(np.zeros((1,3)))

        t = 0

        t0=time.time()


        while running:
            X,U = updatePosSpeed(X,U,N,Vu,Vp,Vz,Vuu,Vpp,Vzz,dVu,dVp,dVz,dU,v0,drag,Vcc,Vcs,Vs,dPhi,nPhi,dt)
            t=t+1

            writeCsv(X,N,U,'position.csv',path,writeMode=  1)
            #if t % 1 == 0:
            
            #    t1=time.time()
            #    print('fps : '+str((t1-t0)))
            #    t0=t1

            if t>tMax:
                running = False
    if dim == 2:

        Vc,Vs,dPhi  = generateVisualFunction2d(nPhi)

    Vup = []
    dU = []
    for k in range (0,N):
        Vup.append(np.zeros((2,2)))

        dU.append(np.zeros((1,2)))

    t = 0

    t0=time.time()

    while running:
        for k in range (0,N):
            V = visualField2d(X,U,k,nPhi)
            dV = abs(np.roll(V,1)-np.roll(V,-1))

            Vup[k] = parameterVision2d(V,dV,Vc,Vs,dPhi)
            dU[k][0,0] = (drag * (v0 - U[k,0] )) + Vuu * ( Vu * Vup[k][0,0] + dVu * Vup[k][1,0] ) 
            dU[k][0,1] =  Vpp * (( Vp * Vup[k][0,1] + dVp * Vup[k][1,1] ));
            

        for k in range (0,N):

            U[k,0] = U[k,0] + dU[k][0,0] *dt;
            U[k,1] = U[k,1] + dU[k][0,1] *dt;
            #print('ids : '+str(k)+' vis :'+str(U[k]))
            X[k,0]=X[k,0]+U[k,0]*np.cos(U[k,1])*dt;
            X[k,1]=X[k,1]+U[k,0]*np.sin(U[k,1])*dt;
            
        t=t+1
        #print(N)
        writeCsv(X,N,U,'position.csv',path,writeMode=  1)
        #if t % 1 == 0:
        
        #    t1=time.time()
        #    print('fps : '+str((t1-t0)))
        #    t0=t1

        if t%(tMax/10)==0:
            print('--- '+str(t/tMax)+' done')
        if t>tMax:
            running = False

