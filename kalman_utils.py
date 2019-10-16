#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def kalman_filter(f, P, doys, qa, rho, kern, R, A, Q, backwards=False):
    """A ruff'n'ready Kalman filter implementation for BRDF
    linear model fitting. Takes the initial estimate of the
    state (mean `f`, covariance matrix `P`), information on 
    timing (`doys`), quality assurance (`qa`), reflectance and
    the observation opertor (the MODIS kernels!), as well as 
    uncertainty information (`R`), a (linear) dynamic model `A`
    and associated covariance matrix (`Q`).
    """
    kf_m = []
    kf_P = []
    innovations = []
    # Loop over days
    if backwards:
        time_axs = doys[::-1]
    else:
        time_axs = doys
    for k in time_axs:
        # Propagate state with dynamic model
        # and inflate uncertainty!
        f = A@f  
        P = A@P@A.T + Q  
        passer = (doys == k)
        # If we have an observation...
        if qa[passer] == 1 and rho[passer] > 0:
            # Retrieve observation operator
            H = kern[passer, :]
            # Kalman gain Kalculations...
            S = H@P@H.T + R
            K = (P@H.T)@np.linalg.inv(S)
            # Update state with new observations
            y = rho[passer] - H@f
            f = f + K@(y)
            P = P-K@S@K.T
            innovations.append(y)
        else:
            innovations.append(np.nan)
        kf_m.append(f)
        kf_P.append(P)
    kf_m = np.array(kf_m)
    kf_P = np.array(kf_P)
    innovations = np.array(innovations)
    return (kf_m, kf_P, innovations)


def kalman_smoother(f, P, kf_m, kf_P, A, Q):
    ms = f
    Ps = P
    rts_m = [f]
    rts_P = [P]
    n_obs = kf_m.shape[0]

    for k in range(n_obs-1, 0, -1):
        mf = A@kf_m[k, :] 
        Pp = A@kf_P[k, :, :]@A.T + Q 
        Ck = (kf_P[k, :, :]@A.T)@np.linalg.inv(Pp)
        ms = kf_m[k, :] + Ck@(ms-mf)
        Ps = kf_P[k, :, :] + Ck@(Ps-Pp)@Ck.T 
        rts_m.append(ms)
        rts_P.append(Ps)
    rts_m = np.array(rts_m)
    rts_P = np.array(rts_P)
    rts_m = rts_m[::-1, :]
    rts_P = rts_P[::-1, :, :]
    return rts_m, rts_P