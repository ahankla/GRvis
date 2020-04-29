#!/usr/bin/env python
# coding: utf-8

import argparse
import warnings
import numpy as np
import sys
sys.path.append('../modules/')
from raw_data_utils import *
# import new_athena_read as read
from kerrmetric import kerr,fourvector
           
def main(**kwargs):

    # --------
    raw_data_path = "/mnt/c/Users/liaha/scratch/"
    config = "1.1.1-torus2_b-gz2"
    specs = "a0beta500torBeta_br32x32x64rl2x2"
    time = 26
    filename = raw_data_path + config + "_" + specs + ".prim.{:05}.athdf".format(time)
    kwargs['radius'] = 10.0
    # ---


    # fi=kwargs['file']
    fi=filename
    quantities=['rho','vel1','vel2','vel3','Bcc1','Bcc2','Bcc3']
    # p=read.athdf(fi,quantities=quantities,x1_min=kwargs['radius'], x1_max=kwargs['radius'])
    p=read_athdf(fi,quantities=quantities,x1_min=kwargs['radius'], x1_max=kwargs['radius'])
    r=p['x1v']
    theta=p['x2v']
    phi=p['x3v']
    R=p['x1f']
    nx1=len(r)
    nx2=len(theta)
    nx3=len(phi)
    dp=(2*np.pi/nx3)*np.ones(nx3)
    dt=(np.pi/nx2)*np.ones(nx2)
    Dp=dp.reshape(nx3,1,1)
    Dt=dt.reshape(1,nx2,1)
    dr=np.ones(nx1)
    for i in range(nx1):
        dr[i]=R[i+1]-R[i]
    Dr=dr.reshape(1,1,nx1)
    rho=p['rho']
    ks=kerr(r,theta)
    f=fourvector(r,theta,p['vel1'],p['vel2'],p['vel3'],p['Bcc1'],p['Bcc2'],p['Bcc3'])
    print("gamma")
    print(np.max(f.gamma))
    k=Dt*Dp*ks.g
    Mflux=f.u1*k*rho
    x=open("massflux.txt","a")
    t=open("Timemassflux.txt","a")
    print(np.sum(np.sum(Mflux, axis=0), axis=0))
    # np.savetxt(x,np.sum(np.sum(Mflux,axis=0),axis=0))
    # np.savetxt(t,np.array(p['Time']).reshape(1,))
    x.close()
    t.close()
# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('file',
                        # help='name of input file, possibly including path')
    # parser.add_argument('radius',
                        # type=float,
                        # default=None,
                        # help='specific radius to measure at')    
    # args = parser.parse_args()
    # main(**vars(args))
main()
