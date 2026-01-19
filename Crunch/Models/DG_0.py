import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.polynomial.legendre import leggauss
from tqdm import tqdm
from Crunch.Models.polynomials import  *
import pandas as pd


legendre_polynomials = [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10] # Extend this list if using k > 10
legendre_derivatives = [dL0, dL1, dL2, dL3, dL4, dL5, dL6, dL7, dL8, dL9, dL10]
chebyshev_polynomials = [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]
chebyshev_derivatives = [dT0, dT1, dT2, dT3, dT4, dT5, dT6, dT7, dT8, dT9, dT10]


def phi_j_l(x, x_j, h_j,l):
    xi = (x - x_j) / h_j  # Normalized coordinate in [-1, 1]
    return xi ** l
def dphi_j_l(x, x_j, h_j,l):
    xi = (x - x_j) / h_j
    if l == 0:
        return np.zeros_like(x)
    else:
        return  l*xi ** (l - 1) / h_j

# Basis functions and their derivatives
def phi_j_l_legendre(x, x_j, h_j,l):
    xi = 2*(x - x_j) / h_j  # Normalized coordinate in [-1, 1]
    return legendre_polynomials[l](xi)
def dphi_j_l_legendre(x, x_j, h_j, l):
    xi = 2 * (x - x_j) / h_j
    dPdxi = legendre_derivatives[l](xi)
    return dPdxi * (2.0 / h_j)

def phi_j_l_chebyshev(x, x_j, h_j, l):
    # Chebyshev polynomials are defined on [-1, 1].
    xi = 2.0 * (x - x_j) / h_j
    print(xi)
    return chebyshev_polynomials[l](xi)

def dphi_j_l_chebyshev(x, x_j, h_j, l):
    xi = 2.0 * (x - x_j) / h_j
    print(xi)
    dPdxi = chebyshev_derivatives[l](xi)
    return dPdxi * (2.0 / h_j)


def Int(func, a, b,gauss_points, gauss_weights):
    def map_to_interval(xi, a, b):
        return 0.5 * (b - a) * xi + 0.5 * (a + b)
    jacobian = 0.5 * (b - a)
    integral_value = 0.0
    for i in range(len(gauss_points)):
        x_i = map_to_interval(gauss_points[i], a, b)
        integral_value += gauss_weights[i] * func(x_i)
    return integral_value * jacobian


def M_ml(x_mh, x_ph, x_j, h, num_basis,gauss_points, gauss_weights,phi_j_l=phi_j_l_legendre):
    M = np.zeros((num_basis, num_basis))
    for m in range(num_basis):
        for l in range(num_basis):
            M[m, l] = Int(lambda x: phi_j_l(x, x_j, h,l) *  phi_j_l(x, x_j, h,m), x_mh, x_ph,gauss_points, gauss_weights)
    return M

def b_m(u0,x_mh, x_ph, x_j, h, num_basis,gauss_points, gauss_weights,phi_j_l=phi_j_l_legendre):
    b = np.zeros(num_basis)
    for m in range(num_basis):
        b[m] = Int(lambda x: u0(x) *  phi_j_l(x, x_j, h, m), x_mh, x_ph,gauss_points, gauss_weights)
    return b

def S_ml(x_mh, x_ph, x_j, h, num_basis,gauss_points, gauss_weights,phi_j_l=phi_j_l_legendre,dphi_j_l=dphi_j_l_legendre):
    S = np.zeros((num_basis, num_basis))
    for m in range(num_basis):
        for l in range(num_basis):
            S[m, l] = Int(lambda x: phi_j_l(x, x_j, h,l) *  dphi_j_l(x, x_j, h,m), x_mh, x_ph,gauss_points, gauss_weights)
    return S

def L(U_j,delta_t,M_inv,S_j,Phi_ph,Phi_mh):
    u_jph=np.sum(U_j*Phi_ph,axis=1,keepdims=True)
    F_j=-(u_jph*Phi_ph-np.roll(u_jph,1)*Phi_mh)
    return U_j+delta_t*np.matmul(M_inv, (np.matmul(S_j, U_j)+F_j))

def RK3(U_j, dt, M_j_inv, S_j, Phi_ph, Phi_mh):
    U_1 = L(U_j, dt, M_j_inv, S_j, Phi_ph, Phi_mh)
    U_2 = (3/4) * U_j + (1/4) * L(U_1, dt, M_j_inv, S_j, Phi_ph, Phi_mh)
    U_3 = (1/3) * U_j + (2/3) * L(U_2, dt, M_j_inv, S_j, Phi_ph, Phi_mh)
    return U_3

def get_u_at(T, dt, U_j, M_j_inv, S_j, Phi_ph, Phi_mh):
    t = 0.0
    homogenous_steps = int(T // dt)
    last_step = T - homogenous_steps * dt
    for i in range(homogenous_steps + 1):
        U_j = RK3(U_j, dt, M_j_inv, S_j, Phi_ph, Phi_mh)
        if homogenous_steps-1 == i:
            dt = last_step
        t += dt  
    return U_j

def update_order(X_0,X_1,error='L1',order='r1'):
  X_1[order]=np.log(X_0[error]/X_1[error])/np.log(2)
  return X_1

def get_orders(Results_all):
    i=0
    for key in Results_all.keys():
        if i>0:
            error,order='L1','r1'
            Results_all[key]=update_order(Results_all[initial_key],Results_all[key],error,order)
            error,order='L2','r2'
            Results_all[key]=update_order(Results_all[initial_key],Results_all[key],error,order)
            error,order='L_inf','r_inf'
            Results_all[key]=update_order(Results_all[initial_key],Results_all[key],error,order)
        initial_key=key
        i=i+1
    return Results_all