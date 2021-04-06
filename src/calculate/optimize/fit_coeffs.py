# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
import cantera as ct
import nlopt
import warnings
from copy import deepcopy
from scipy.optimize import curve_fit, OptimizeWarning, least_squares, approx_fprime
from timeit import default_timer as timer
import itertools

from calculate.convert_units import OoM
from calculate.optimize.misc_fcns import generalized_loss_fcn

Ru = ct.gas_constant
# Ru = 1.98720425864083

min_pos_system_value = np.finfo(float).eps*(1E2)
max_pos_system_value = (np.finfo(float).max*(1E-20))**(1/3)
min_neg_system_value = -max_pos_system_value
min_ln_val = np.log(min_pos_system_value)
max_ln_val = np.log(max_pos_system_value)

default_arrhenius_coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']
default_Troe_coefNames = ['Ea_0', 'A_0', 'n_0', 'Ea_inf', 'A_inf', 'n_inf', 'A', 'T3', 'T1', 'T2']


def fit_arrhenius(rates, T, x0=[], coefNames=default_arrhenius_coefNames, bnds=[]):
    def fit_fcn_decorator(x0, alter_idx, jac=False):               
        def set_coeffs(*args):
            coeffs = x0
            for n, idx in enumerate(alter_idx):
                coeffs[idx] = args[n]
            return coeffs
        
        def ln_arrhenius(T, *args):
            [Ea, ln_A, n] = set_coeffs(*args)
            return ln_A + n*np.log(T) - Ea/(Ru*T)

        def ln_arrhenius_jac(T, *args):
            [Ea, ln_A, n] = set_coeffs(*args)
            jac = np.array([-1/(Ru*T), np.ones_like(T), np.log(T)]).T
            return jac[:, alter_idx]

        if not jac:
            return ln_arrhenius
        else:
            return ln_arrhenius_jac
    
    ln_k = np.log(rates)
    if len(x0) == 0:
        x0 = np.polyfit(np.reciprocal(T), ln_k, 1)
        x0 = np.array([-x0[0]*Ru, x0[1], 0]) # Ea, ln(A), n
    else:
        x0 = np.array(x0)
        x0[1] = np.log(x0[1])
    
    idx = []
    for n, coefName in enumerate(default_arrhenius_coefNames):
        if coefName in coefNames:
            idx.append(n)
    
    A_idx = None
    if 'pre_exponential_factor' in coefNames:
        if isinstance(coefNames, np.ndarray):
            A_idx = np.argwhere(coefNames == 'pre_exponential_factor')[0]
        else:
            A_idx = coefNames.index('pre_exponential_factor')
    
    fit_func = fit_fcn_decorator(x0, idx)
    fit_func_jac = fit_fcn_decorator(x0, idx, jac=True)
    p0 = x0[idx]

    if len(bnds) > 0:
        if A_idx is not None:
            bnds[0][A_idx] = np.log(bnds[0][A_idx])
            bnds[1][A_idx] = np.log(bnds[1][A_idx])

        # only valid initial guesses
        for n, val in enumerate(p0):
            if val < bnds[0][n]:
                p0[n] = bnds[0][n]
            elif val > bnds[1][n]:
                p0[n] = bnds[1][n]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            try:
                popt, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='trf', bounds=bnds,
                                    jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)
            except:
                return
    else:           
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            try:
                popt, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='trf',
                                    jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)
            except:
                return
    
    if A_idx is not None:
        popt[A_idx] = np.exp(popt[A_idx])

    return popt


def fit_Troe(rates, T, M, x0=[], coefNames=default_Troe_coefNames, bnds=[], scipy_curvefit=True, HPL_LPL_defined=True):    
    def fit_fcn_decorator(ln_k, x0, M, alter_idx, s=[], sfcn=[], jac=False):
        def set_coeffs(*args):
            coeffs = x0
            for n, idx in enumerate(alter_idx):
                coeffs[idx] = args[n]
            return coeffs
        
        def ln_Troe(T, *x, grad_calc=True):
            #if grad_calc:
            #    s[:] = ln_Troe_grad(T, *x)

            #x = x*s + x0[idx]

            [Ea_0, ln_A_0, n_0, Ea_inf, ln_A_inf, n_inf, A, T3, T1, T2] = set_coeffs(*x)
            A_0, A_inf = np.exp(ln_A_0), np.exp(ln_A_inf)
            k_0 = A_0*T**n_0*np.exp(-Ea_0/(Ru*T))
            k_inf = A_inf*T**n_inf*np.exp(-Ea_inf/(Ru*T))
            if ([k_0, k_inf] <= min_pos_system_value).any():
                return np.ones_like(T)*max_pos_system_value
            P_r = k_0/k_inf*M
            log_P_r = np.log10(P_r)

            if T3 == 0.0 or (-T/T3 < -max_ln_val).any():
                exp_T3 = 0
            elif (-T/T3 > max_ln_val).any():
                exp_T3 = max_pos_system_value
            else:
                exp_T3 = np.exp(-T/T3)

            if T1 == 0.0 or (-T/T1 < -max_ln_val).any():
                exp_T1 = 0
            elif (-T/T1 > max_ln_val).any():
                exp_T1 = max_pos_system_value
            else:
                exp_T1 = np.exp(-T/T1)

            if (-T2/T > max_ln_val).any():
                exp_T2 = max_pos_system_value
            else:
                exp_T2 = np.exp(-T2/T)

            Fcent = (1-A)*exp_T3 + A*exp_T1 + exp_T2
            if (Fcent <= 0.0).any():
                return np.ones_like(T)*max_pos_system_value

            log_Fcent = np.log10(Fcent)
            C = -0.4 - 0.67*log_Fcent
            N = 0.75 - 1.27*log_Fcent
            f1 = (log_P_r + C)/(N - 0.14*(log_P_r + C))

            e = np.exp(1)

            ln_F = log_Fcent/np.log10(e)/(1 + f1**2)

            ln_k_calc = np.log(k_inf*P_r/(1 + P_r)) + ln_F
            
            return ln_k_calc

        def ln_Troe_jac(T, *args):
            [Ea_0, ln_A_0, n_0, Ea_inf, ln_A_inf, n_inf, A, T3, T1, T2] = set_coeffs(*args)
            A_0, A_inf = np.exp(ln_A_0), np.exp(ln_A_inf)
            k_0 = A_0*T**n_0*np.exp(-Ea_0/(Ru*T))
            k_inf = A_inf*T**n_inf*np.exp(-Ea_inf/(Ru*T))
            P_r = k_0/k_inf*M
            
            if T3 == 0.0 or (-T/T3 < -max_ln_val).any():
                exp_T3 = 0
            elif (-T/T3 > max_ln_val).any():
                exp_T3 = max_pos_system_value
            else:
                exp_T3 = np.exp(-T/T3)

            if T1 == 0.0 or (-T/T1 < -max_ln_val).any():
                exp_T1 = 0
            elif (-T/T1 > max_ln_val).any():
                exp_T1 = max_pos_system_value
            else:
                exp_T1 = np.exp(-T/T1)
            
            if (-T2/T > max_ln_val).any():
                exp_T2 = max_pos_system_value
            else:
                exp_T2 = np.exp(-T2/T)

            Fcent = (1-A)*exp_T3 + A*exp_T1 + exp_T2
            if (Fcent <= 0.0).any():
                return np.ones_like(T)*np.inf

            log_Fcent = np.log10(Fcent)
            C = -0.4 - 0.67*log_Fcent
            N = 0.75 - 1.27*log_Fcent

            v = np.log10(P_r) + C
            u = N - 0.14*v

            e = np.exp(1)
            log_e = np.log10(e)

            upv = (u**2 + v**2)
            upvs = upv**2

            if (set([0, 1, 2, 3, 4, 5]) & set(alter_idx)):  # if any arrhenius variable is being altered
                M_k0 = M*k_0
                Pr_denom = M_k0 + k_inf
                interior_term_2 = 2*log_Fcent*N*log_e**3*v/(u*upv)

            falloff_term = (u*log_e/(Fcent*upvs))*(u**3/np.log(10) - 2*log_e*log_Fcent*v*(1.27*v - 0.67*N))

            jac = []
            for n in alter_idx:
                if n == 0:   # dlnk_dEa_0
                    jac.append(1/(Ru*T)*(k_inf/Pr_denom + interior_term_2))
                elif n == 1: # dlnk_dln_A_0
                    #jac.append(1/A_0*(1-M_k0/Pr_denom - interior_term_2))   # dlnk_dA_0
                    jac.append(1-M_k0/Pr_denom - interior_term_2)
                elif n == 2: # dlnk_dn_0
                    jac.append(np.log(T)*(k_inf/Pr_denom - interior_term_2))
                elif n == 3: # dlnk_dEa_inf
                    jac.append(1/(Ru*T)*(M_k0/Pr_denom - interior_term_2))
                elif n == 4: # dlnk_dln_A_inf
                    #jac.append(1/A_inf*(1-k_inf/Pr_denom + interior_term_2)) # dlnk_dA_inf
                    jac.append(1-k_inf/Pr_denom + interior_term_2)
                elif n == 5: # dlnk_dn_inf
                    jac.append(np.log(T)*(M_k0/Pr_denom + interior_term_2))
                elif n == 6: # dlnk_dA
                    jac.append((exp_T1 - exp_T3)*falloff_term)
                elif n == 7: # dlnk_dT3
                    if T3 == 0.0 or (-T/T3 < -max_ln_val).any():
                        jac.append(np.zeros_like(T))
                    elif (-T/T3 > max_ln_val).any():
                        jac.append(np.ones_like(T)*np.sign(-A)*max_pos_system_value)
                    else:
                        jac.append(((1-A)*(T/T3**2)*exp_T3)*falloff_term)
                elif n == 8: # dlnk_dT1
                    if T1 == 0.0 or (-T/T1 < -max_ln_val).any():
                        jac.append(np.zeros_like(T))
                    elif (-T/T1 > max_ln_val).any():
                        jac.append(np.ones_like(T)*np.sign(A)*max_pos_system_value)
                    else:
                        jac.append(A*T/T1**2*exp_T1*falloff_term)
                elif n == 9: # dlnk_dT2
                    jac.append(-1/T*exp_T2*falloff_term)

            jac = np.vstack(jac).T
            return jac

        if not jac:
            return ln_Troe
        elif jac:
            return ln_Troe_jac
    
    def calc_s(jac):
        x = np.linalg.norm(jac, axis=0)

        if (x == 0.0).all():
            x = np.ones_like(x)*1E-14
        else:
            x[x==0.0] = 10**(OoM(np.min(x[x!=0])) - 1)  # TODO: MAKE THIS BETTER running into problem when s is zero, this is a janky workaround

        return 1/x

    def obj_fcn_decorator(fit_fcn, fit_func_jac, x0, idx, T, ln_k, bnds, return_sum=True):
        def obj_fcn(x, grad=[]):
            #x = x/s + x0[idx]

            #warnings.simplefilter('ignore', OptimizeWarning)
            #x, _ = curve_fit(fit_func, T, ln_k, p0=x, method='trf', bounds=bnds, # dogbox
            #                 jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)

            resid = fit_func(T, *x) - ln_k
            if return_sum:              
                obj_val = generalized_loss_fcn(resid).sum()
            else:
                obj_val = resid

            #s[:] = np.abs(np.sum(loss*fit_func_jac(T, *x).T, axis=1))

            if grad.size > 0:
                grad[:] = np.sum(fit_func_jac(T, *x).T*resid, axis=1)

            print(obj_val)

            return obj_val
        return obj_fcn
    
    ln_k = np.log(rates)
    
    alter_idx = {'low_rate': [], 'high_rate': [], 'falloff_parameters': [], 'all': []}
    for n, coefName in enumerate(default_Troe_coefNames):
        if coefName in coefNames:
            alter_idx['all'].append(n)
            if coefName in ['Ea_0', 'A_0', 'n_0']:
                alter_idx['low_rate'].append(n)
            elif coefName in ['Ea_inf', 'A_inf', 'n_inf']:
                alter_idx['high_rate'].append(n)
            else:
                alter_idx['falloff_parameters'].append(n)

    if (set([0, 1, 2]) & set(alter_idx)) and len(x0) == 0:
        idx = alter_idx['low_rate']
        a0 = np.polyfit(np.reciprocal(T[idx]), ln_k[idx], 1)
        x0[idx] = np.array([-a0[0]*Ru, np.exp(a0[1]), 0])

    if (set([3, 4, 5]) & set(alter_idx)) and len(x0) < 4:
        idx = alter_idx['high_rate']
        a0 = np.polyfit(np.reciprocal(T[idx]), ln_k[idx], 1)
        x0[idx] = np.array([-a0[0]*Ru, np.exp(a0[1]), 0])
    
    # initial guesses for falloff
    #if len(x0) != 10:
    x0 = [*x0[:6], 0.7, 200, 300, 400] # initial guesses for fitting Troe if none exist

    #x0_falloff = list(itertools.product([0, 1], repeat=4))
    #print(x0_falloff)

    x0 = np.array(x0)

    A_idx = [1, 4]
    #A_idx = None
    #if set(['A_0', 'A_inf']) & set(coefNames):
    #    A_idx = [i for i, coef in enumerate(coefNames) if coef in ['A_0', 'A_inf']]

    # only valid initial guesses
    bnds = deepcopy(bnds)
    for n, val in enumerate(x0):
        if val < bnds[0][n]:
            x0[n] = bnds[0][n]
        elif val > bnds[1][n]:
            x0[n] = bnds[1][n]
    
    # Fit HPL and LPL (for falloff this is exact, otherwise a guess)
    for arrhenius_type in ['low_rate', 'high_rate']:
        idx = alter_idx[arrhenius_type]
        if len(idx) > 0:
            x0[idx] = fit_arrhenius(rates[idx], T[idx], x0=x0[idx], bnds=[bnds[0][idx], bnds[1][idx]])

    if A_idx is not None:
        x0[A_idx] = np.log(x0[A_idx])
        bnds[0][A_idx] = np.log(bnds[0][A_idx])
        bnds[1][A_idx] = np.log(bnds[1][A_idx])

    if HPL_LPL_defined: # Fit falloff
        idx = alter_idx['falloff_parameters']    
    else:
        idx = alter_idx['all'] 

    T, M, ln_k = T[idx], M[idx], ln_k[idx]
    p0 = x0[idx]
    #p0 = np.zeros_like(x0[idx])
    s = np.ones_like(x0[idx])

    if len(bnds) == 0:
        bnds = [-np.ones_like(p0), np.ones_like(p0)]*np.inf
    else:
        bnds = [bnds[0][idx], bnds[1][idx]]

    fit_func = fit_fcn_decorator(ln_k, x0, M, idx, s=s)
    fit_func_jac = fit_fcn_decorator(ln_k, x0, M, idx, s=s, jac=True)

    if scipy_curvefit:
        # for testing scipy least_squares, curve_fit is a wrapper for this fcn
        #obj_fcn = obj_fcn_decorator(fit_func, fit_func_jac, x0, idx, T, ln_k, return_sum=False)
        #obj_fcn_jac = lambda x: fit_func_jac(T, *x)

        #res = least_squares(obj_fcn, p0, method='trf', bounds=bnds, 
        #                    jac=obj_fcn_jac, x_scale='jac', max_nfev=len(p0)*1000)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            try:
                x_fit, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='trf', bounds=bnds, # dogbox
                                        jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)
                                        #jac='2-point', x_scale='jac', max_nfev=len(p0)*1000)
            except:
                return

    else:
        #s[:] = calc_s(fit_func_jac(T, *p0))
        nlopt_fit_fcn = obj_fcn_decorator(fit_func, fit_func_jac, x0, idx, T, ln_k, bnds)

        opt = nlopt.opt(nlopt.GN_CRS2_LM, len(idx))
        #opt = nlopt.opt(nlopt.GN_DIRECT, len(idx))
        #opt = nlopt.opt(nlopt.LN_SBPLX, len(idx)) # nlopt.LN_SBPLX nlopt.LN_COBYLA nlopt.LD_MMA nlopt.LD_LBFGS

        opt.set_min_objective(nlopt_fit_fcn)
        #opt.set_maxeval(int(options['stop_criteria_val'])-1)
        #opt.set_maxtime(options['stop_criteria_val']*60)

        opt.set_xtol_rel(1E-2)
        opt.set_ftol_rel(1E-2)
        #opt.set_lower_bounds((bnds[0][idx]-p0)*s)
        #opt.set_upper_bounds((bnds[1][idx]-p0)*s)
        opt.set_lower_bounds(bnds[0][idx])
        opt.set_upper_bounds(bnds[1][idx])

        opt.set_initial_step(calc_s(fit_func_jac(T, *p0))*1E-2)
        x_fit = opt.optimize(p0) # optimize!

    if HPL_LPL_defined: # Fit falloff
        #x = np.array([*x0[:6], *(x_fit*s + x0[idx])])
        x = np.array([*x0[:6], *x_fit])
    else:
        x = np.array(x_fit)

    #if (x[-4:] == x0[-4:]).all():
    #    print('no change')
    #else:
    #    print('fit values found')

    #print(f'x  {x[-4:]}')
    #print(ln_k)
    fit_k = fit_func(T, *x_fit, grad_calc=False)
    #print(fit_k)
    rss = np.sum((ln_k - fit_k)**2)
    print(f'ln_k_resid {rss}')

    if A_idx is not None:
        x[A_idx] = np.exp(x[A_idx])
    
    return x


def fit_generic(rates, T, P, X, rxnIdx, coefKeys, coefNames, mech, x0, bnds):    
    rxn = mech.gas.reaction(rxnIdx)
    rates = np.array(rates)
    T = np.array(T)
    P = np.array(P)
    x0 = np.array(x0).copy()
    coefNames = np.array(coefNames)
    bnds = np.array(bnds).copy()

    # Faster and works for extreme values like n = -70
    if type(rxn) is ct.ElementaryReaction or type(rxn) is ct.ThreeBodyReaction:
        x0 = [mech.coeffs_bnds[rxnIdx]['rate'][coefName]['resetVal'] for coefName in mech.coeffs_bnds[rxnIdx]['rate']]
        coeffs = fit_arrhenius(rates, T, x0=x0, coefNames=coefNames, bnds=bnds)

        if type(rxn) is ct.ThreeBodyReaction and 'pre_exponential_factor' in coefNames:
            A_idx = np.argwhere(coefNames == 'pre_exponential_factor')[0]
            coeffs[A_idx] = coeffs[A_idx]/mech.M(rxn)

    elif type(rxn) is ct.FalloffReaction:
        M = mech.M(rxn, [T, P, X])

        falloff_coefNames = []
        for key, coefName in zip(coefKeys, coefNames):
            if coefName == 'activation_energy':
                falloff_coefNames.append('Ea')
            elif coefName == 'pre_exponential_factor':
                falloff_coefNames.append('A')
            elif coefName == 'temperature_exponent':
                falloff_coefNames.append('n')

            if key['coeffs'] == 'low_rate':
                falloff_coefNames[-1] = f'{falloff_coefNames[-1]}_0'
            elif key['coeffs'] == 'high_rate':
                falloff_coefNames[-1] = f'{falloff_coefNames[-1]}_inf'

        if rxn.falloff.type == 'Troe':
            falloff_coefNames.extend(['A', 'T3', 'T1', 'T2'])
            coeffs = fit_Troe(rates, T, M, x0=x0, coefNames=falloff_coefNames, bnds=bnds, scipy_curvefit=True)

        elif rxn.falloff.type == 'SRI':
            falloff_coefNames.extend(['a', 'b', 'c', 'd', 'e'])
            coeffs = fit_SRI(rates, T, M, x0, coefNames=SRI_coefNames, bnds=bnds, scipy_curvefit=True)

    return coeffs


def fit_coeffs(rates, T, P, X, rxnIdx, coefKeys, coefNames, x0, bnds, mech): 
    if len(coefNames) == 0: return # if not coefs being optimized in rxn, return 

    return fit_generic(rates, T, P, X, rxnIdx, coefKeys, coefNames, mech, x0, bnds)
    

def debug(mech):
    import matplotlib.pyplot as plt
    from timeit import default_timer as timer
    start = timer()
    # rates = [1529339.05689338, 1548270.86688399, 1567437.0352583]
    rates = [1529339.05689338, 1548270.86688399, 1567437.0352583]*np.array([1.000002, 1.00002, 1])
    T = [2387.10188629, 2389.48898818, 2391.88086905]
    P = [16136.20900077, 16136.20900077, 16136.20900077]
    X = {'Kr': 0.99, 'C8H8': 0.01}
        
    coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']
    coefBndsKeys = {'coeffs': [0, 0, 0], 'coeffs_bnds': ['rate', 'rate', 'rate']}
    rxnIdx = 0
    coeffs = fit_coeffs(rates, T, P, X, rxnIdx, coefKeys, coefNames, mech)
    print(timer() - start)
    # print(coeffs)
    # print(np.array([2.4442928e+08, 3.4120000e+11, 0.0000000e+00]))
    
    rate_fit = []
    for n, T_val in enumerate(T):
        mech.set_TPX(T_val, P[0], X)
        rate_fit.append(mech.gas.forward_rate_constants[rxnIdx])
    
    print(np.sqrt(np.mean((rates - rate_fit)**2)))

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