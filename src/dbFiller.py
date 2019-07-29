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
import sqlite3
import VisualSwarm3d as vs
import datetime
import threading
import multiprocessing 
import time
import random
import traceback


lockDB = False  
parametersName = 'VisualParameters.db'
dbName = 'VisualSimulation.db'
dbVideo = 'VisualVideo.db'


def FirstGen():
    """generate the empty database files

    """
    conn = sqlite3.connect(parametersName)
    c = conn.cursor()
    c.execute('''CREATE TABLE parameters (id text,  
                                          N integer, nPhi integer,
                                          dt real,
                                          v0 real,drag real, 
                                          Vuu real, Vpp real, Vzz real,
                                          Vu real, Vp real, Vz real,
                                          dVu real, dVp real, dVz real)''')
    conn.commit()
    conn.close()
    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    c.execute('''CREATE TABLE simulation (id text, repId text,date text,
                                          N integer, nPhi integer,
                                          dt real,
                                          v0 real,drag real, 
                                          Vuu real, Vpp real, Vzz real,
                                          Vu real, Vp real, Vz real,
                                          dVu real, dVp real, dVz real)''')
    conn.commit()
    conn.close()



def dbFiller3d(NRange = [2],nPhiRange = [256],
                    VuRange = [-1.0],VpRange = [-1.0],VzRange = [-1.0],
                    dVuRange = [1.0],dVpRange = [1.0],dVzRange = [1.0],
                    VuuRange = [1.0],VppRange = [1.0],VzzRange = [1.0],
                    dt = 0.1,drag = 0.1,v0 = 1,dims = 3):
    """Fill the parameters that accounts for all the simulation that should be done.
        
    Args:
      NRange: number of individuals
      nPhiRange : resolution of the visual field (2 * nPhi + 1) x (nPhi + 1)
      VuRange : sensitivity to subtended angle on acceleration in the plane (x,y)
      VpRange : sensitivity to subtended angle on turning rate in the plane (x,y)
      VzRange : sensitivity to subtended angle on acceleration in the z direction
      dVuRange : sensitivity to edges on acceleration in the plane (x,y)
      dVpRange : sensitivity to edges on turning rate in the plane (x,y)
      dVzRange : sensitivity to edges on acceleration in the z direction
      VuuRange : sensitivity to acceleration in the plane (x,y)
      VppRange : sensitivity to turning rate in the plane (x,y)
      VzzRange : sensitivity to acceleration in the z direction
      dt : time step
      drag : drag value
      v0 : velocity of individual
      dims : dimension of the simulation 2 or 3


    """


   
    if dims == 2:

        VzRange= [0.0]
        dVzRange= [0]

        VzzRange = [0.0]
    conn = sqlite3.connect(parametersName)
    c = conn.cursor()
    for N in Nrange:
        for nPhi in nPhiRange:
            for Vu in VuRange:
                for Vp in VpRange:
                    for Vz in VzRange:
                        for dVu in dVuRange:
                            for dVz in dVzRange: 
                                for Vuu in VuuRange:
                                    for Vpp in VppRange:
                                        for Vzz in VzzRange:
                                            expId = str(uuid.uuid4())
                                            values = [expId,N,nPhi,dt,v0,drag,Vuu,Vpp,Vzz,Vu,Vp,Vz,dVu,dVu,dVz]
                    
                                            c.execute("INSERT INTO parameters VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",values)
    conn.commit()
    conn.close()






def checkExpParam(expId):
        """
       Check the parameters associated with a specific simulation Id
        
    Args:
        expId : id of the simulation

    Returns:
      N: number of individuals
      nPhi : resolution of the visual field (if 3D : (2 * nPhi + 1) x (nPhi + 1))
      Vu : sensitivity to subtended angle on acceleration in the plane (x,y)
      Vp : sensitivity to subtended angle on turning rate in the plane (x,y)
      Vz : sensitivity to subtended angle on acceleration in the z direction
      dVu : sensitivity to edges on acceleration in the plane (x,y)
      dVp : sensitivity to edges on turning rate in the plane (x,y)
      dVz : sensitivity to edges on acceleration in the z direction
      Vuu : sensitivity to acceleration in the plane (x,y)
      Vpp : sensitivity to turning rate in the plane (x,y)
      Vzz : sensitivity to acceleration in the z direction
      dt : time step
      drag : drag value
      v0 : velocity of individual
      
    """
    connParam = sqlite3.connect(parametersName, check_same_thread=False)
    cursorParam = connParam.cursor()


    cursorParam.execute("Select N from parameters where id = ?",(expId,))
    N  = (cursorParam.fetchall())[0][0]
    cursorParam.execute("Select nPhi from parameters where id = ?",(expId,))
    nPhi  = (cursorParam.fetchall())[0][0]
    cursorParam.execute("Select dt from parameters where id = ?",(expId,))
    dt  = (cursorParam.fetchall())[0][0]
    cursorParam.execute("Select v0 from parameters where id = ?",(expId,))
    v0  = (cursorParam.fetchall())[0][0]
    cursorParam.execute("Select drag from parameters where id = ?",(expId,))
    drag  = (cursorParam.fetchall())[0][0]
    
    cursorParam.execute("Select Vu from parameters where id = ?",(expId,))
    Vu  = (cursorParam.fetchall())[0][0]
    cursorParam.execute("Select Vp from parameters where id = ?",(expId,))
    Vp  = (cursorParam.fetchall())[0][0]
    cursorParam.execute("Select Vz from parameters where id = ?",(expId,))
    Vz  = (cursorParam.fetchall())[0][0]
    cursorParam.execute("Select dVu from parameters where id = ?",(expId,))
    dVu  = (cursorParam.fetchall())[0][0]
    cursorParam.execute("Select dVp from parameters where id = ?",(expId,))
    dVp  = (cursorParam.fetchall())[0][0]
    cursorParam.execute("Select dVz from parameters where id = ?",(expId,))
    dVz  = (cursorParam.fetchall())[0][0]
    cursorParam.execute("Select Vuu from parameters where id = ?",(expId,))
    Vuu  = (cursorParam.fetchall())[0][0]
    cursorParam.execute("Select Vpp from parameters where id = ?",(expId,))
    Vpp  = (cursorParam.fetchall())[0][0]
    cursorParam.execute("Select Vzz from parameters where id = ?",(expId,))
    Vzz  = (cursorParam.fetchall())[0][0]





    
    connParam.close()

    return N,nPhi,v0,Vu,Vp,Vz,dVu,dVp,dVz,Vuu,Vpp,Vzz,drag,dt


def startSimulation(expId):
        """Start a simulation
        
    Args:
      expId[0] : the id of the simulation to recover the parameters
      expId[1] : the id of the the replicate
      expId[2] : the dimension of the simulation (2 or 3)
      expId[3] : the maximal time of the simulation
      expId[3] : the path where the data are saved

    """


    global lockDB
    try:
        print("The following experiment is analyzed : "+str(expId[0]))
        print("The following replicate is analyzed  : "+str(expId[1]))
       
        time.sleep(random.random())
        N,nPhi,v0,Vu,Vp,Vz,dVu,dVp,dVz,Vuu,Vpp,Vzz,drag,dt = checkExpParam(expId[0])
        if N<100:
            print([checkExpParam(expId[0])])
            
            vs.visModel(N,xMax,nPhi,v0,Vu,Vp,Vz,dVu,dVp,dVz,Vuu,Vpp,Vzz,drag,dt,expId=expId[1],dims = expId[2],tMax = expId[3],dataPath=expId[4])
            
            while lockDB:
                time.sleep(random.random())
            lockDB = True
            conn = sqlite3.connect(dbName, check_same_thread=False)
            c = conn.cursor()
            
            values = [expId[0],expId[1],datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),N,nPhi,dt,v0,drag,Vuu,Vpp,Vzz,Vu,Vp,Vz,dVu,dVp,dVz]
            print('----- writing in database')                    
            c.execute("INSERT INTO simulation VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",values)
            conn.commit()
            conn.close()
            print('----- wrote in database')    
            lockDB = False
    except Exception as e:

        traceback.print_exc()
        print()


def main(args):
    """Compute the simulations in the database
        
    Args:
      args[0] : dims the dimension of the simulation
      args[1] : tMax the maximal time of the simulation
      args[2] : replicate the number of replicates
      args[3] : nThreads the number of thread
      args[4] : path the path where simulation results are saved

    """
    dims = args[0]
    tMax = args[1]
    replicate = args[2]
    nThreads = args[3]
    path = args[4]

    if nThreads > 1:
        threaded = True
    else:
        threaded = False
    print('Starting')

    connParam = sqlite3.connect(parametersName, check_same_thread=False)
    cursorParam = connParam.cursor()
    cursorParam.execute("Select id from parameters")
    expIds=cursorParam.fetchall()
    
    connSim = sqlite3.connect(dbName, check_same_thread=False)
    cursorSim = connSim.cursor()
    parametersList = []




    print('checking the ids')
    running= True
    exp=0
    print('making ' +str(replicate)+' replicates')
    for expId in expIds:
        #print('is the experiment '+expId[0]+' already analyzed ?')
        
        cursorSim.execute("Select * from simulation where id = ?",(str(expId[0]),))
        n=len(cursorSim.fetchall())
        
        k=0
        while n+k<replicate:
            
            k=k+1
            #print('No')
            repId = str(uuid.uuid4())
            parametersList.append([expId[0],repId,dims,tMax,path])
            exp =exp+1

    print('experiments type : '+str(len(expIds)))
    print('experiments todo : '+str(len(expIds)*replicate))
    print('experiments left : '+str(exp))
                

    connParam.close()
    connSim.close()  
    if threaded:
    

        pool = multiprocessing.Pool(processes=nThreads)
        pool.map_async(startSimulation, parametersList)
        pool.close()
        pool.join()
    else:
        for parmater in parametersList:
            startSimulation(parmater)
    

if __name__ == "__main__":
    main()



