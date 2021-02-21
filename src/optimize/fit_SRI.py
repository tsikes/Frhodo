# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
import cantera as ct
import nlopt
import warnings
from copy import deepcopy
from scipy.optimize import curve_fit, OptimizeWarning, approx_fprime
from timeit import default_timer as timer

from optimize.misc_fcns import generalized_loss_fcn

Ru = ct.gas_constant
# Ru = 1.98720425864083

min_neg_system_value = np.finfo(float).min*(1E-20) # Don't push the limits too hard
min_pos_system_value = np.finfo(float).eps*(1.1)
max_pos_system_value = np.finfo(float).max*(1E-20)

default_SRI_coefNames = ['Ea_0', 'A_0', 'n_0', 'Ea_inf', 'A_inf', 'n_inf', 'a', 'b', 'c', 'd', 'e']

def fit_SRI(rates, T, M, x0=[], coefNames=default_SRI_coefNames, bnds=[], scipy_curvefit=True):
    def fit_fcn_decorator(x0, alter_idx, s=[], jac=False):               
        def set_coeffs(*args):
            coeffs = x0
            for n, idx in enumerate(alter_idx):
                coeffs[idx] = args[n]
            return coeffs
        
        def ln_SRI(T, *args):
            [Ea_0, ln_A_0, n_0, Ea_inf, ln_A_inf, n_inf, a, b, c, d, e] = set_coeffs(*args)
            A_0, A_inf = np.exp(ln_A_0), np.exp(ln_A_inf)
            k_0 = A_0*T**n_0*np.exp(-Ea_0/(Ru*T))
            k_inf = A_inf*T**n_inf*np.exp(-Ea_inf/(Ru*T))
            P_r = k_0*M/k_inf

            #n = 1/(1+np.log10(P_r)**2)
            #F = (a*np.exp(-b/T) + np.exp(-T/c))**n*d*np.exp(-e/T)
            #k = k_inf*P_r/(1 + P_r)*F
            #ln_k = np.log(k)

            ln_k = np.log(d*k_inf*P_r/(1 + P_r)) + 1/(1+np.log10(P_r)**2)*np.log(a*np.exp(-b/T) + np.exp(-T/c)) - e*np.log(T) # TODO: ineq constraint that a*np.exp(-b/T) + np.exp(-T/c) > 0
            
            return ln_k

        def ln_SRI_jac(T, *args):
            [Ea_0, ln_A_0, n_0, Ea_inf, ln_A_inf, n_inf, a, b, c, d, e] = set_coeffs(*args)
            A_0, A_inf = np.exp(ln_A_0), np.exp(ln_A_inf)
            k_0 = A_0*T**n_0*np.exp(-Ea_0/(Ru*T))
            k_inf = A_inf*T**n_inf*np.exp(-Ea_inf/(Ru*T))
            P_r = k_0/k_inf*M

            abc = 1/(1 + np.log10(P_r)**2)/(a*np.exp(T/c) + np.exp(b/T))
     
            if (set([0, 1, 2, 3, 4, 5]) & set(alter_idx)):  # if any arrhenius variable is being altered
                P_r_frac = 1/(1 + P_r)
                u = np.log(a*np.exp(-b/T)+np.exp(-T/c))
                ln_P_r_term = P_r_frac - 2*u*np.log10(P_r)/(1 + np.log10(P_r)**2)**2 # might not need log conversion?

            jac = []
            for n in alter_idx:
                if n == 0:   # dlnk_dEa_0
                    jac.append(-1/(Ru*T)*ln_P_r_term)
                elif n == 1: # dlnk_dA_0
                    jac.append(1/A_0*ln_P_r_term)
                elif n == 2: # dlnk_dn_0
                    jac.append(np.log(T)*ln_P_r_term)
                elif n == 3: # dlnk_dEa_inf
                    jac.append(1/(Ru*T)*ln_P_r_term)
                elif n == 4: # dlnk_dA_inf
                    jac.append(-1/A_inf*ln_P_r_term)
                elif n == 5: # dlnk_dn_inf
                    jac.append(-np.log(T)*ln_P_r_term)
                elif n == 6: # dlnk_da
                    jac.append(np.exp(T/c)*abc)
                elif n == 7: # dlnk_db
                    jac.append(-a/T*np.exp(T/c)*abc)
                elif n == 8: # dlnk_dc
                    jac.append(T/c**2*np.exp(b/T)*abc)
                elif n == 9: # dlnk_d_d
                    jac.append(np.ones_like(T)/d)
                elif n == 10:# dlnk_de
                    jac.append(np.log(T))

            jac = np.vstack(jac).T
            return jac

        if not jac:
            return ln_SRI
        else:
            return ln_SRI_jac
    
    def nlopt_fit_fcn_decorator(fit_fcn, grad_fcn, x0, alter_idx, T, ln_k_original):
        def nlopt_fit_fcn(x, grad):
            x = x/s + x0[alter_idx]

            resid = fit_func(T, *x) - ln_k_original
            loss = generalized_loss_fcn(resid).sum()

            s[:] = np.abs(np.sum(loss*grad_fcn(T, *x).T, axis=1))

            if len(grad) > 0:
                grad[:] = np.sum(loss*grad_fcn(T, *x).T, axis=1)

            return loss
        return nlopt_fit_fcn
        

    ln_k = np.log(rates)
    
    alter_idx = []
    for n, coefName in enumerate(default_SRI_coefNames): # ['Ea_0', 'A_0', 'n_0', 'Ea_inf', 'A_inf', 'n_inf', 'a', 'b', 'c', 'd', 'e']
        if coefName in coefNames:
            alter_idx.append(n)
    
    if (set([0, 1, 2]) & set(alter_idx)) and len(x0) == 0:
        a0 = np.polyfit(np.reciprocal(T[0:3]), ln_k[0:3], 1)
        x0[0:3] = np.array([-a0[0]*Ru, np.exp(a0[1]), 0])

    if (set([3, 4, 5]) & set(alter_idx)) and len(x0) < 4:
        a0 = np.polyfit(np.reciprocal(T[3:6]), ln_k[3:6], 1)
        x0[3:6] = np.array([-a0[0]*Ru, np.exp(a0[1]), 0])

    if len(x0) < 7:
        #x0[6:10] = [1.0, 10.0, 1000, 1.0, 1.0] # initial guesses for fitting SRI if none exist
        #x0[6:10] = [1.0, -100.0, 1000, 1.0, 0.0] # initial guesses for fitting SRI if none exist
        x0[6:10] = [1.0, -1.0, 100.0, 1.0, 0.01] # initial guesses for fitting SRI if none exist

    x0[1] = np.log(x0[1])
    x0[4] = np.log(x0[4])

    x0 = np.array(x0)

    A_idx = None
    if set(['A_0', 'A_inf']) & set(coefNames):
        A_idx = [i for i, coef in enumerate(coefNames) if coef in ['A_0', 'A_inf']]
    
    # set default bounds
    if len(bnds) == 0:
        bnds = [[], []]
        for SRI_coef in coefNames:
            if SRI_coef == 'a': # this restriction isn't stricly necessary but can run into issues with log(-val) without
                bnds[0].append(0)  
            elif SRI_coef == 'c': # c must be 0 or greater
                bnds[0].append(10000/np.log(max_pos_system_value))   # needs to be large enough for exp(T/c) to not blow up
            elif SRI_coef == 'd': # d must be positive value
                bnds[0].append(min_pos_system_value)  
            else:
                bnds[0].append(min_neg_system_value)
                    
            if SRI_coef == 'b':
                bnds[1].append(np.log(max_pos_system_value))
            else:
                bnds[1].append(max_pos_system_value)
    else:
        bnds = deepcopy(bnds)

    if A_idx is not None:
        bnds[0][A_idx] = np.log(bnds[0][A_idx])
        bnds[1][A_idx] = np.log(bnds[1][A_idx])

    # only valid initial guesses
    p0 = x0[alter_idx]
    for n, val in enumerate(p0):
        if val < bnds[0][n]:
            p0[n] = bnds[0][n]
        elif val > bnds[1][n]:
            p0[n] = bnds[1][n]
    
    if scipy_curvefit:
        fit_func = fit_fcn_decorator(x0, alter_idx)
        fit_func_jac = fit_fcn_decorator(x0, alter_idx, jac=True)

        if len(bnds) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', OptimizeWarning)
                try:
                    x, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='dogbox', bounds=bnds,
                                     jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)
                                        #jac='2-point', x_scale='jac', max_nfev=len(p0)*1000)
                except:
                    return

        else:           
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', OptimizeWarning)
                try:
                    x, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='dogbox',
                                     jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)
                except:
                    return

    else:
        #s = fit_fcn_decorator(x0, alter_idx, jac=True)(T, *x0)
        #s = 1/np.linalg.norm(s, axis=1)
        s = np.ones_like(p0)
        fit_func = fit_fcn_decorator(x0, alter_idx, s=s)
        fit_func_jac = fit_fcn_decorator(x0, alter_idx, s=s, jac=True)
        nlopt_fit_fcn = nlopt_fit_fcn_decorator(fit_func, fit_func_jac, x0, alter_idx, T, ln_k)
        
        opt = nlopt.opt(nlopt.LN_SBPLX, len(alter_idx)) # nlopt.LN_SBPLX nlopt.LN_COBYLA nlopt.LD_MMA nlopt.LD_LBFGS

        opt.set_min_objective(nlopt_fit_fcn)
        #opt.set_maxeval(int(options['stop_criteria_val'])-1)
        #opt.set_maxtime(options['stop_criteria_val']*60)

        opt.set_xtol_rel(1E-2)
        opt.set_ftol_rel(1E-2)
        opt.set_lower_bounds((bnds[0]-x0[alter_idx])*s)
        opt.set_upper_bounds((bnds[1]-x0[alter_idx])*s)

        opt.set_initial_step(1E-3)
        x = opt.optimize(np.zeros_like(p0)) # optimize!

    print(f'ln_k_resid [{np.sum((ln_k - fit_func(T, *x))**2)**0.5}]')

    if A_idx is not None:
        x[A_idx] = np.exp(x[A_idx])

    return x   

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    start = timer()
    
    rates = np.array([9.13674578])/200
    T = [1513.8026716]
    x0 = [1439225332.24, 5.8499038e+276, -71.113552]
    coefNames = ['pre_exponential_factor']
    bnds = [[2.4424906541753446e-16], [1.7976931348623155e+288]]

    #bnds = [[0, 2.4424906541753446e-16, -1.7976931348623155e+288], 
    #        [1.7976931348623155e+288, 1.7976931348623155e+288, 1.7976931348623155e+288]]
    
    # rates = np.array([9.74253640e-01, 8.74004054e+02, 1.41896847e+05])
    # rates = np.array([1.54283654e-02, 3.89226810e+02, 1.65380781e+04])
    # rates = np.array([4.73813308e+00, 1.39405144e+03, 1.14981010e+05])
    #rates = np.array([6.17844122e-02, 9.74149806e+01, 2.01630443e+04])
    # rates = np.array([2.43094099e-02, 4.02305872e+01, 3.95740585e+03])

    # rates = rates*np.array([1, 1.1, 0.9])
    # rates = [1529339.05689338, 1548270.86688399, 1567437.0352583]*np.array([1, 1.00002, 1])
    #T = [1359.55345014, 1725.11257135, 2359.55345014]

    #print(fit_coeffs(rates, T, P, X, coefNames, rxnIdx, mech))
    [A] = fit_arrhenius(rates, T, x0=x0, coefNames=coefNames, bnds=bnds)
    Ea, n = x0[0], x0[2]
    print(timer() - start)
    print(x0)
    print([Ea, A, n])
    print(A/x0[1])
    
    T_fit = np.linspace(T[0], T[-1], 100)
    rate_fit = A*T_fit**n*np.exp(-Ea/(Ru*T_fit))
    
    plt.plot(10000*np.reciprocal(T), np.log10(rates), 'o')
    plt.plot(10000/T_fit, np.log10(rate_fit))
    plt.show()