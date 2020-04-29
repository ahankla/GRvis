#!/usr/bin/env python
# coding: utf-8



import numpy as np
class kerr():
    def __init__(self,r,t):
        self.rlen=len(r)
        self.tlen=len(t)
        self.R=r.reshape(1,self.rlen)
        self.th=t.reshape(self.tlen,1)
        cos2=np.power(np.cos(self.th),2)
        sin2=np.power(np.sin(self.th),2)
        sin=np.sin(self.th)
        r=self.R
        r2=np.power(self.R,2)
        M=1
        a=0.85
        a2=a**2
        p=2*M*r
        sigma=r2+(a2*cos2)
        delta=r2+a2-p
        d=np.divide((a2*p),sigma)

        self.sin2 = sin2
        self.sigma = sigma
        self.p = p

        self.g00=-(1.0-np.divide(p,sigma))
        self.g10=np.divide(p,sigma)
        self.g03=sin2*np.divide(-a*p,sigma)
        self.g11=1.+np.divide(p,sigma)
        self.g13=-a*sin2*(1.+np.divide(p,sigma))
        self.g22=sigma
        self.g33=(a2+r2+(d*sin2))*sin2
        self.G00=-(1.0+np.divide(p,sigma))
        self.G10=np.divide(p,sigma)
        self.G11=np.divide(delta,sigma)
        self.G22=np.divide(1.,sigma)
        self.G31=np.divide(a, sigma)
        self.G33=np.divide(1.,(sigma*sin2))
        self.alpha=np.sqrt(-1.0/self.G00)
        self.g=(r2+a2*cos2)*sin
        self.g30=self.g03
        self.g01=self.g10
        self.g31=self.g13

class fourvector():
    def __init__(self,r,theta,vel1,vel2,vel3,Bcc1,Bcc2,Bcc3):
        self.v1=vel1
        self.u2=vel2
        self.u3=vel3
        ks=kerr(r,theta)
        print("shapes: ")
        print(self.v1.shape)
        self.test = self.v1*self.v1*ks.g11
        self.comp2 = self.v1*self.u3*ks.g13
        print((self.v1*self.v1).shape)
        print(ks.g11.shape)
        print((self.v1*self.v1*ks.g11).shape)
        self.usq=(self.v1*self.v1*ks.g11)+2*(self.v1*self.u3*ks.g13)+(ks.g22*self.u2*self.u2)+(self.u3*self.u3*ks.g33)
        self.gamma=np.sqrt(1+((self.v1*self.v1*ks.g11)+2*(self.v1*self.u3*ks.g13)+(ks.g22*self.u2*self.u2)+(self.u3*self.u3*ks.g33)))
        self.u0=self.gamma/ks.alpha
        self.u1=self.v1-(self.gamma*ks.alpha*ks.G10)
        self.B0= self.u1*Bcc1+self.u2*Bcc2+self.u3*Bcc3 
        self.B1=(Bcc1+(self.B0*self.u1))/self.u0
        self.B2=(Bcc2+(self.B0*self.u2))/self.u0
        self.B3=(Bcc3+(self.B0*self.u3))/self.u0
        self.b0=ks.g00*self.B0+ks.g10*self.B1+ks.g03*self.B3 
        self.b2=ks.g22*self.B2
        self.b1=ks.g10*self.B0+ks.g11*self.B1+ks.g13*self.B3
        self.b3=ks.g03*self.B0+ks.g13*self.B1+ks.g33*self.B3
        self.bsq=self.B0*self.b0+self.B1*self.b1+self.B2*self.b2+self.B3*self.b3
        self.pmag=0.5*self.bsq
        

class tp2c():
    def __init__(self,rho,pgas,bsq,u0,u1,u2,u3,B0,B1,B2,B3,r,theta,phi):
        ks=kerr(r,theta)
        self.r=r.reshape(1,1,len(r))
        self.theta=theta
        self.phi=phi

        self.st=np.sin(self.theta).reshape(1,len(theta),1)
        self.ct=np.cos(self.theta).reshape(1,len(theta),1)
        self.sp=np.sin(self.phi).reshape(len(phi),1,1)
        self.cp=np.cos(self.phi).reshape(len(phi),1,1)

        self.z=self.r*self.ct
        self.y=self.r*self.st*self.sp
        self.x=self.r*self.st*self.cp
        Gamma=13./9 

        self.Tf1=(rho+pgas*(Gamma/(Gamma-1)))*u0*u1+pgas*ks.G10
        self.Tm1=bsq*u0*u1+0.5*bsq*ks.G10-B0*B1
        self.Tr=self.Tm1+self.Tf1
        self.Tf2=(rho+pgas*(Gamma/(Gamma-1)))*u0*u2
        self.Tm2=(bsq*u0*u2)-B0*B2
        self.Tt=self.Tm2+self.Tf2
        self.Tf3=(rho+pgas*(Gamma/(Gamma-1)))*u0*u3
        self.Tm3=(bsq*u0*u3)-B0*B3
        self.Tp=self.Tm3+self.Tf3

        self.Az=self.Tr*self.ct-self.Tt*self.st
        self.Ay=self.Tr*self.st*self.sp+self.Tt*self.ct*self.sp+self.Tp*self.cp
        self.Ax=self.Tr*self.st*self.cp+self.Tt*self.ct*self.cp-self.Tp*self.sp

        self.Lx=self.y*self.Az-self.z*self.Ay
        self.Ly=self.z*self.Ax-self.x*self.Az
        self.Lz=self.x*self.Ay-self.y*self.Ax


class rotate():
    def __init__(self,var,tilt,p,nx3):
        self.t=np.round(np.append(np.linspace(-tilt,tilt,nx3/2),np.linspace(tilt,-tilt,nx3/2))).astype(int)
        self.v=np.roll(var,-int(p),axis=0)
        for i in range(nx3):
            self.v[i]=np.roll(self.v[i],self.t[i],axis=0)

class rotateN():
    def __init__(self,vN,tilt,p,nx1):
        self.nx=np.len(vN)
        self.t=np.round(np.append(np.linspace(-tilt,tilt,nx1/2),np.linspace(tilt,-tilt,nx1/2))).astype(int)
        self.v=np.zeroes_like(vN)
        for j in range(nx):
            self.v[j]=np.roll(vN[j],-int(p),axis=0)
            for i in range(nx1):
                self.v[j,i]=np.roll(self.v[j,i],self.t[i],axis=0)  
