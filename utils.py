#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from kernels import Kernels

def prepare_data(fname="data.r2023.c87.dat"):
    """Reads in the data"""
    # Read data in, but skip first line
    data = np.loadtxt(fname, skiprows=1)
    doys = data[:, 0]
    qa = data[:, 1]
    vza = data[:, 2]
    sza = data[:, 4]
    raa = data[:, 3] - data[:, 5]
    rho = data[:, 7]
    # Generate the kernels,
    K_obs = Kernels(vza, sza, raa,
                    LiType='Sparse', doIntegrals=False,
                    normalise=1, RecipFlag=True, RossHS=False, MODISSPARSE=True,
                    RossType='Thick')
    n_obs = vza.shape[0]
    kern = np.ones((n_obs, 3))  # Store the kernels in an array
    kern[:, 1] = K_obs.Ross
    kern[:, 2] = K_obs.Li
    return doys, qa, vza, sza, raa, rho, kern, n_obs

def fit_period_prior(doys, qa, rho, start_doy, end_doy, band_unc, kernels,
                     prior_mean=None, prior_std=None):
    """Fit a period of observations

    doys: iter
        An array with the DoYs
    rho: iter
        A matrix (n_bands * n_obs) with observations
    start_doy: int
        A starting DoY
    end_doy: int
        An end DoY
    band_no: int
        The band number to use (indices the first dimension of rho)
    band_unc: float
        The per band_uncertainty
    kernels: iter
        The kernels matrix, size (n_obs * 3)
    """
    passer = np.logical_and(qa==1, np.logical_and(doys >= start_doy, doys <= end_doy))
    obs = rho[passer]

    n_obs = passer.sum()
    if n_obs < 7:
        raise np.linalg.LinAlgError
    K = kernels[passer, :]
    C = np.eye(n_obs)/(band_unc*band_unc)
    A = K.T @(C@K)
    b = (K.T.dot(C).dot(obs))
    
    if prior_mean is not None and prior_std is not None:
        C_prior = np.eye(3)/prior_std**2
        A = A + C_prior
        b = b + C_prior@(prior_mean)
    f = np.linalg.solve(A, b)
    fwd = K.dot(f)
    rmse = (obs - fwd).var()
    r2 = np.corrcoef(obs, fwd)[0, 1]
    the_unc = np.linalg.inv(A)
    sigma_f = np.sqrt(the_unc.diagonal())
    f_upper = K@(f - 1.96*sigma_f)
    f_lower = K@(f + 1.96*sigma_f)
    return f, fwd, obs, rmse, r2, the_unc, f_upper, f_lower, sigma_f

