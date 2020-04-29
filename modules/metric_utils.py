#!/usr/bin/env python
# coding: utf-8

"""
Adapted from Michelle Athay's kerrmetric.py (sent Jan. 29 2020).
Calculates the value of the metric at every point (init) and uses
the metric components to calculate:
    - four-velocity (used for e.g. mass flux),
    - projected magnetic field (used for magnetic pressure)

TO DO:
    - calculate Maxwell tensor components

Last updated:
2020-04-27 LH (discovered discrepancy with MA version)
2020-04-27 Lia Hankla (initial tweak/correction (?) from MA's version)
"""

import numpy as np

class kerrschild():
    def __init__(self, x1v, x2v, x3v, M=1, a=0.85):
        """
        x1v contains the r coordinates in Kerr-Schild.
        x2v contains the theta coordinates in Kerr-Schild.
        x3v is not necessary because the metric is axisymmetric, but
            having it clarifies multiplication issues later on.
        """
        nx1 = len(x1v)
        nx2 = len(x2v)
        nx3 = len(x3v)

        r_vals = np.repeat(x1v.reshape(1, nx1)[np.newaxis, :, :], nx3, axis=0)
        theta_vals = np.repeat(x2v.reshape(nx2, 1)[np.newaxis, :, :], nx3, axis=0)
        # r_vals = x1v.reshape(1, self.nx1)
        # theta_vals = x2v.reshape(self.nx2, 1)

        # Calculate some quantities that will be useful
        cos2theta = np.power(np.cos(theta_vals),2)
        sin2theta = np.power(np.sin(theta_vals),2)
        sintheta = np.sin(theta_vals)
        r2 = np.power(r_vals, 2)
        a2 = a**2
        Sigma = r2 + (a2*cos2theta)
        Delta = r2 - 2.0*M*r_vals + a2

        # the Jacobian is sqrt{-g}, which is given by the below formula
        # according to White, Stone, Quataert 2019 (right under 2. Simulations)
        # wiki claims g = -1.0 everywhere...???
        self.jacobian = (r2 + a2*cos2theta) * sintheta

        # Metric and inverse metric components
        # from Athena++ src/coordinates/kerr-schild.cpp
        # (LowerVectorCell, lines 1804 - 1819) for the lower g_{\mu\nu}
        # and from (RaiseVectorCell, lines 1758 - 1773) for the raised g^{\mu\nu}
        self.g00 = -(1.0-np.divide(2.0*M*r_vals, Sigma))
        self.g10 = np.divide(2.0*M*r_vals, Sigma)
        self.g20 = 0.0
        self.g30 = -np.divide(2.0*M*a*r_vals, Sigma)*sin2theta
        self.g11 = 1.0 + np.divide(2.0*M*r_vals, Sigma)
        self.g12 = 0.0
        # self.g13 = -a*sin2theta*(1.0 + np.divide(2.0*M*r_vals, Sigma))
        self.g13 = -(1.0 + np.divide(2.0*M*r_vals, Sigma))*a*sin2theta
        self.g22 = Sigma
        self.g23 = 0.0
        self.g33 = (r2 + a2 + np.divide(2.0*M*r_vals, Sigma)*a2* sin2theta) * sin2theta
        # self.g33 = (r2 + a2 + np.divide(2.0*M*r_vals, Sigma)*a2* sin2theta) * sin2theta

        self.g01 = self.g10
        self.g02 = self.g20
        self.g03 = self.g30
        self.g21 = self.g12
        self.g31 = self.g13
        self.g32 = self.g23

        self.G00 = -(1.0 + np.divide(2.0*M*r_vals, Sigma))
        self.G01 = np.divide(2.0*M*r_vals, Sigma)
        self.G02 = 0.0
        self.G03 = 0.0
        self.G11 = np.divide(Delta, Sigma)
        self.G12 = 0.0
        self.G13 = a/Sigma
        self.G22 = 1.0/Sigma
        self.G23 = 0.0
        self.G33 = 1.0/(Sigma * sin2theta)

        self.G10 = self.G01
        self.G20 = self.G02
        self.G30 = self.G03
        self.G21 = self.G12
        self.G31 = self.G13
        self.G32 = self.G23


    def get_normal_frame_gamma(self, output_velocity):
        """
        Note this gamma is different from MA's version...
        all the equations are the same but the output gamma is not.

        The difference (I believe as of 2020-04-28 without emailing MA)
        is because this version logs the metric coefficients as 3D arrays,
        whereas MA's script keeps them as 2D, which results in NOT element-wise
        multiplication when she does v1*v1*g11, etc..as a result, the outcome
        depends on the order of multiplication.

        The present version DOES element-wise (all arrays are 3D) and as a result
        the order of multiplication is not important.
        """
        (uu1, uu2, uu3) = output_velocity
        # usq is \tilde u^i\tilde u_i
        # Note in the kerr-schild metric, g12 = g23 = 0,
        # in which case gamma matches MA's kerrmetric.py
        usq = self.g11*uu1*uu1 + 2.0*self.g12*uu1*uu2 + 2.0*self.g13*uu1*uu3
        + self.g22*uu2*uu2 + 2.0*self.g23*uu2*uu3
        + self.g33*uu3*uu3

        # Gamma is \sqrt{1.0 + u^2}
        gamma = np.sqrt(1.0 + usq)
        return gamma


    def get_four_velocity_from_output(self, output_velocity):
        """
        Athena++ outputs the projected velocity.
        But for pretty much everything, we need the actual 4-velocity.
        Fortunately we see how to back-calculate this in gr_torus's method
        UserWorkInLoop() (lines 1282 - 1300).
        """
        (uu1, uu2, uu3) = output_velocity
        gamma = self.get_normal_frame_gamma(output_velocity)

        # alpha is the lapse function, 1/sqrt(-G00)
        alpha = 1.0/np.sqrt(-1.0*self.G00)

        # Now transform back.
        # Note that in the Kerr-schild metric, G02 = G03 = 0,
        # which matches MA's kerrmetric.py
        u0 = gamma/alpha
        u1 = uu1 - alpha * gamma * self.G01
        u2 = uu2 - alpha * gamma * self.G02
        u3 = uu3 - alpha * gamma * self.G03

        return (u0, u1, u2, u3)

    def get_proj_bfield_from_outputB_fourV(self, four_velocity, output_mag_field):
        """
        Athena++ outputs the 0th components of the maxwell tensor.
        But we want the projected magnetic field for calculating magnetic pressure, etc. 
        Fortunately we see how to back-calculate this in gr_torus's method
        UserWorkInLoop() (lines 1304 - 1319).

        TO DO:
        - reconcile with MA's version
        """

        (u0, u1, u2, u3) = four_velocity
        (bb1, bb2, bb3) = output_mag_field
        # Note in the Kerr-schild metric, g12 = g23 = 0
        # XX matches MA?
        b0 = self.g01*u0*bb1 + self.g02*u0*bb2 + self.g03*u0*bb3
        + self.g11*u1*bb1 + self.g12*u1*bb2 + self.g13*u1*bb3
        + self.g12*u2*bb1 + self.g22*u2*bb2 + self.g23*u2*bb3
        + self.g13*u3*bb1 + self.g23*u3*bb2 + self.g33*u3*bb3
        b1 = (bb1 + b0 * u1)/u0
        b2 = (bb2 + b0 * u2)/u0
        b3 = (bb3 + b0 * u3)/u0

        return (b0, b1, b2, b3)

    def get_benergy_from_projB(self, projected_b):
        (B0, B1, B2, B3) = projected_b
        (b0, b1, b2, b3) = self.lower_fourvec_index(projected_b)
        b_sq = B0*b0 + B1*b1 * B2*b2 + B3*b3
        return b_sq/2.0


    def lower_fourvec_index(four_vector):
        (A0, A1, A2, A3) = four_vector
        a0 = self.g00*A0 + self.g01*A1 + self.g02*A2 + self.g03*A3
        a1 = self.g10*A0 + self.g11*A1 + self.g12*A2 + self.g13*A3
        a2 = self.g20*A0 + self.g21*A1 + self.g22*A2 + self.g23*A3
        a3 = self.g30*A0 + self.g31*A1 + self.g32*A2 + self.g33*A3

        return (a0, a1, a2, a3)


# From here down is directly copied/pasted from MA's kerrmetric.py.
# TO DO:
#     - port into present form (and check it! last I heard, MA was
#          still trying to get it correct)

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


# I believe these functions are useful for tilted tori
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
