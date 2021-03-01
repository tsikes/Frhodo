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

from convert_units import OoM
from optimize.misc_fcns import generalized_loss_fcn

Ru = ct.gas_constant
# Ru = 1.98720425864083

min_pos_system_value = np.finfo(float).eps*(1E2)
min_ln_val = np.log(min_pos_system_value)
max_pos_system_value = np.finfo(float).max*(1E-20)
max_ln_val = np.log(max_pos_system_value)

default_arrhenius_coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']
default_SRI_coefNames = ['Ea_0', 'A_0', 'n_0', 'Ea_inf', 'A_inf', 'n_inf', 'a', 'b', 'c', 'd', 'e']
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
                popt, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='dogbox', bounds=bnds,
                                    jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)
            except:
                return
    else:           
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            try:
                popt, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='dogbox',
                                    jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)
            except:
                return
    
    if A_idx is not None:
        popt[A_idx] = np.exp(popt[A_idx])

    return popt


def fit_SRI(rates, T, M, x0=[], coefNames=default_SRI_coefNames, bnds=[], scipy_curvefit=True, Fit_LPL_HPL=False):
    def fit_fcn_decorator(x0, M, alter_idx, s=[], jac=False):               
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
            P_r = k_0/k_inf*M

            n = 1/(1+np.log10(P_r)**2)
            if c == 0.0:
                exp_neg_T_c = 0
            else:
                exp_neg_T_c = np.exp(-T/c)

            F = ((a*np.exp(-b/T) + exp_neg_T_c)**n)*d*T**e
            k = k_inf*P_r/(1 + P_r)*F
            ln_k = np.log(k)

            #ln_k = np.log(d*k_inf*P_r/(1 + P_r)) + 1/(1+np.log10(P_r)**2)*np.log(a*np.exp(-b/T) + np.exp(-T/c)) - e*np.log(T) # TODO: ineq constraint that a*np.exp(-b/T) + np.exp(-T/c) > 0
            
            return ln_k

        def ln_SRI_jac(T, *args):
            [Ea_0, ln_A_0, n_0, Ea_inf, ln_A_inf, n_inf, a, b, c, d, e] = set_coeffs(*args)
            A_0, A_inf = np.exp(ln_A_0), np.exp(ln_A_inf)
            k_0 = A_0*T**n_0*np.exp(-Ea_0/(Ru*T))
            k_inf = A_inf*T**n_inf*np.exp(-Ea_inf/(Ru*T))
            P_r = k_0/k_inf*M

            if c == 0.0:
                exp_neg_T_c = 0
            else:
                exp_neg_T_c = np.exp(-T/c)

            abc_interior = a*np.exp(-b/T) + exp_neg_T_c
            abc = 1/((1 + np.log10(P_r)**2)*abc_interior)
     
            if (set([0, 1, 2, 3, 4, 5]) & set(alter_idx)):  # if any arrhenius variable is being altered
                ln_P_r_term = 1/(1 + P_r) - 2*np.log(abc_interior)*np.log10(P_r)/(1 + np.log10(P_r)**2)**2

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
                    jac.append(np.exp(-b/T)*abc)
                elif n == 7: # dlnk_db
                    jac.append(-a/T*np.exp(-b/T)*abc)
                elif n == 8: # dlnk_dc
                    if c == 0.0:
                        jac.append(np.zeros_like(T))
                    else:
                        jac.append(T/c**2*exp_neg_T_c*abc)
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
    
    alter_idx = {'low_rate': [], 'high_rate': [], 'falloff_parameters': [], 'all': []}
    for n, coefName in enumerate(default_SRI_coefNames):
        if coefName in coefNames:
            alter_idx['all'].append(n)
            if coefName in ['Ea_0', 'A_0', 'n_0']:
                alter_idx['low_rate'].append(n)
            elif coefName in ['Ea_inf', 'A_inf', 'n_inf']:
                alter_idx['low_rate'].append(n)
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

    # initial guesses for SRI a, b
    #Ea_0, ln_A_0, n_0, Ea_inf, ln_A_inf, n_inf
    k_0 = x0[1]*T[6:]**x0[2]*np.exp(-x0[0]/(Ru*T[6:]))
    k_inf = x0[4]*T[6:]**x0[5]*np.exp(-x0[3]/(Ru*T[6:]))
    P_r = k_0/k_inf*M[6:]
    left_side = (1 + np.log10(P_r)**2)*(ln_k[6:] - np.log(k_0*M[6:]/(1+P_r)))

    a0 = np.polynomial.polynomial.Polynomial.fit(np.reciprocal(T[6:]), left_side, 1)
    a0 = a0.convert().coef
    x0[6:8] = [np.exp(a0[0]), -a0[1]]   # TODO: This could result in invalid numbers if a*exp(-b/T) > 1

    if len(x0) < 11:
    #    #x0[6:11] = [1.0, 10.0, 1000, 1.0, 1.0] # initial guesses for fitting SRI if none exist
        x0[8:11] = [0.001, 10.0, 0.001] # initial guesses for fitting SRI if none exist
    #    #x0[6:11] = [1.0, -1.0, 100.0, 1.0, 0.01] # initial guesses for fitting SRI if none exist
    
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

    for arrhenius_type in ['low_rate', 'high_rate']:
        idx = alter_idx[arrhenius_type]
        if len(idx) > 0:
            x0[idx] = fit_arrhenius(rates[idx], T[idx], x0=x0[idx], bnds=[bnds[0][idx], bnds[1][idx]])

    if A_idx is not None:
        x0[A_idx] = np.log(x0[A_idx])
        bnds[0][A_idx] = np.log(bnds[0][A_idx])
        bnds[1][A_idx] = np.log(bnds[1][A_idx])

    if not Fit_LPL_HPL:
        idx = alter_idx['falloff_parameters']
        p0 = x0[idx]

        if scipy_curvefit:
            fit_func = fit_fcn_decorator(x0, M[idx], idx)
            fit_func_jac = fit_fcn_decorator(x0, M[idx], idx, jac=True)

            if len(bnds) == 0:
                bnds = [np.ones_like(p0[idx]), np.ones_like(p0[idx])]*np.inf
            else:
                bnds = [bnds[0][idx], bnds[1][idx]]

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', OptimizeWarning)
                #try:
                x, _ = curve_fit(fit_func, T[idx], ln_k[idx], p0=p0, method='dogbox', bounds=bnds,
                                        jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)
                                        #jac='2-point', x_scale='jac', max_nfev=len(p0)*1000)
                #except:
                #    return
            
            arrhenius_idx = [*alter_idx['low_rate'], *alter_idx['high_rate']]
            x = np.array([*x0[arrhenius_idx], *x])

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
    
    print(f'x {x}')
    print(f'ln_k_resid [{np.sum((ln_k[alter_idx["falloff_parameters"]] - fit_fcn_decorator(x0, M[alter_idx["falloff_parameters"]], alter_idx["all"])(T[alter_idx["falloff_parameters"]], *x))**2)**0.5}]')

    if A_idx is not None:
        x[A_idx] = np.exp(x[A_idx])

    return x   

def fit_Troe_use_ct(rates, T, P, X, rxnIdx, coefKeys, coefNames, mech, x0, bnds):
    def fit_rate_eqn(ln_k, P, X, mech, key, coefNames, rxnIdx):
        rxn = mech.gas.reaction(rxnIdx)
        def inner(temperatures, coeffs, scale_calc):
            mech.coeffs[rxnIdx][key] = coeffs
            mech.modify_reactions(mech.coeffs, rxnNums=rxnIdx)

            rate = []
            for n, T in enumerate(temperatures):
                mech.set_TPX(T, P[n], X[n])
                rate.append(mech.gas.forward_rate_constants[rxnIdx])

            if not scale_calc:
                loss = generalized_loss_fcn(np.log(rate)-ln_k)
                return loss.sum()    # defaults to L2 aka SSE
            else:
                return np.log10(rate)
        return inner

    old_coeffs = deepcopy(mech.coeffs[rxnIdx])
    key_dict = {}
    old_key = ''
    for n, key in enumerate(coefKeys):  # break apart falloff reaction arrhenius/falloff
        if key['coeffs'] != old_key:
            key_dict[key['coeffs']] = [n]
            old_key = key['coeffs']
        else:
            key_dict[key['coeffs']].append(n)
        
    coeffs = []
    for key, idxs in key_dict.items():
        idxs = np.array(idxs)
        if 'rate' in key:
            arrhenius_coeffs = fit_arrhenius(rates[idxs], T[idxs], x0=x0[idxs], 
                                                coefNames=coefNames[idxs], bnds=bnds[:,idxs])
            for n, coefName in enumerate(['activation_energy', 'pre_exponential_factor', 'temperature_exponent']):
                mech.coeffs[rxnIdx][key][coefName] = arrhenius_coeffs[n]

            coeffs.extend(arrhenius_coeffs)

        else:   # fit falloff
            T = T[idxs]
            ln_k = np.log(rates[idxs])
            x0 = x0[idxs]
            x0s = 10**OoM(x0)
            x0 = x0/x0s
    
            if not isinstance(X, (list, np.ndarray)):   # if only a single composition is given, duplicate
                X = [X]*len(T)
                
            eqn = lambda T, x, s_calc: fit_rate_eqn(ln_k, P[idxs], X, mech, 'falloff_parameters', coefNames[idxs], rxnIdx)(T, (x*x0s), s_calc)
            s = np.abs(approx_fprime(x0, lambda x: eqn([np.mean(T)], x, True), 1E-2))
            s[s==0] = 10**(OoM(np.min(s[s!=0])) - 1)  # TODO: MAKE THIS BETTER running into problem when s is zero, this is a janky workaround
            scaled_eqn = lambda x, grad: eqn(T, (x/s + x0), False)
            p0 = np.zeros_like(x0)

            opt = nlopt.opt(nlopt.LN_SBPLX, 4) # either nlopt.LN_SBPLX or nlopt.LN_COBYLA

            opt.set_min_objective(scaled_eqn)
            #opt.set_maxeval(int(options['stop_criteria_val'])-1)
            #opt.set_maxtime(options['stop_criteria_val']*60)

            opt.set_xtol_rel(1E-2)
            opt.set_ftol_rel(1E-2)
            #opt.set_lower_bounds(self.bnds['lower'])
            #opt.set_upper_bounds(self.bnds['upper'])

            opt.set_initial_step(1E-1)
            x = opt.optimize(p0) # optimize!

            print((x/s + x0)*x0s, opt.get_numevals())
            coeffs.extend((x/s + x0)*x0s)

    mech.coeffs[rxnIdx] = old_coeffs    # reset coeffs

    return coeffs

def fit_Troe(rates, T, M, x0=[], coefNames=default_Troe_coefNames, bnds=[], scipy_curvefit=True, Fit_LPL_HPL=False):    
    def fit_fcn_decorator(x0, M, alter_idx, s=[], jac=False):               
        def set_coeffs(*args):
            coeffs = x0
            for n, idx in enumerate(alter_idx):
                coeffs[idx] = args[n]
            return coeffs
        
        def ln_Troe(T, *args):
            [Ea_0, ln_A_0, n_0, Ea_inf, ln_A_inf, n_inf, A, T3, T1, T2] = set_coeffs(*args)
            A_0, A_inf = np.exp(ln_A_0), np.exp(ln_A_inf)
            k_0 = A_0*T**n_0*np.exp(-Ea_0/(Ru*T))
            k_inf = A_inf*T**n_inf*np.exp(-Ea_inf/(Ru*T))
            P_r = k_0/k_inf*M
            log_P_r = np.log10(P_r)

            if T3 == 0.0 or (T/T3 > max_ln_val).any():
                exp_T_3 = 0
            else:
                exp_T_3 = np.exp(-T/T3)

            if T1 == 0.0 or (T/T1 > max_ln_val).any():
                exp_T_1 = 0
            else:
                exp_T_1 = np.exp(-T/T1)

            Fcent = (1-A)*exp_T_3 + A*exp_T_1 + np.exp(-T2/T)
            if (Fcent <= 0.0).any():
                return np.ones_like(T)*np.inf

            log_Fcent = np.log10(Fcent)
            C = -0.4 - 0.67*log_Fcent
            N = 0.75 - 1.27*log_Fcent
            f1 = (log_P_r + C)/(N - 0.14*(log_P_r + C))

            e = np.exp(1)

            ln_F = log_Fcent/np.log10(e)/(1+f1**2)

            ln_k = np.log(k_inf*P_r/(1 + P_r)) + ln_F
            
            return ln_k

        def ln_Troe_jac(T, *args):  # TODO: currently not working, maybe use autodiff?
            [Ea_0, ln_A_0, n_0, Ea_inf, ln_A_inf, n_inf, a, b, c, d, e] = set_coeffs(*args)
            A_0, A_inf = np.exp(ln_A_0), np.exp(ln_A_inf)
            k_0 = A_0*T**n_0*np.exp(-Ea_0/(Ru*T))
            k_inf = A_inf*T**n_inf*np.exp(-Ea_inf/(Ru*T))
            P_r = k_0/k_inf*M

            if c == 0.0 or (-T/c > max_ln_val).any():
                exp_neg_T_c = 0
            else:
                exp_neg_T_c = np.exp(-T/c)

            abc_interior = a*np.exp(-b/T) + exp_neg_T_c
            abc = 1/((1 + np.log10(P_r)**2)*abc_interior)
     
            if (set([0, 1, 2, 3, 4, 5]) & set(alter_idx)):  # if any arrhenius variable is being altered
                ln_P_r_term = 1/(1 + P_r) - 2*np.log(abc_interior)*np.log10(P_r)/(1 + np.log10(P_r)**2)**2

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
                    jac.append(np.exp(-b/T)*abc)
                elif n == 7: # dlnk_db
                    jac.append(-a/T*np.exp(-b/T)*abc)
                elif n == 8: # dlnk_dc
                    if c == 0.0:
                        jac.append(np.zeros_like(T))
                    else:
                        jac.append(T/c**2*exp_neg_T_c*abc)
                elif n == 9: # dlnk_d_d
                    jac.append(np.ones_like(T)/d)
                elif n == 10:# dlnk_de
                    jac.append(np.log(T))

            jac = np.vstack(jac).T
            return jac

        if not jac:
            return ln_Troe
        else:
            return ln_Troe_jac
    
    def nlopt_fit_fcn_decorator(fit_fcn, grad_fcn, x0, idx, T, ln_k_original):
        def nlopt_fit_fcn(x, grad=[]):
            x = x/s + x0[idx]

            resid = fit_func(T, *x) - ln_k_original
            loss = generalized_loss_fcn(resid).sum()

            #s[:] = np.abs(np.sum(loss*grad_fcn(T, *x).T, axis=1))

            #if len(grad) > 0:
            #    grad[:] = np.sum(loss*grad_fcn(T, *x).T, axis=1)

            return loss
        return nlopt_fit_fcn
    
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
    if len(x0) != 10:
        x0 = [*x0[:6], 0.1, 100, 1000, 10000] # initial guesses for fitting Troe if none exist
    
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

    for arrhenius_type in ['low_rate', 'high_rate']:
        idx = alter_idx[arrhenius_type]
        if len(idx) > 0:
            x0[idx] = fit_arrhenius(rates[idx], T[idx], x0=x0[idx], bnds=[bnds[0][idx], bnds[1][idx]])

    if A_idx is not None:
        x0[A_idx] = np.log(x0[A_idx])
        bnds[0][A_idx] = np.log(bnds[0][A_idx])
        bnds[1][A_idx] = np.log(bnds[1][A_idx])

    if not Fit_LPL_HPL:
        idx = alter_idx['falloff_parameters']
        T, M, ln_k = T[idx], M[idx], ln_k[idx]
        p0 = x0[idx]

        if scipy_curvefit:
            fit_func = fit_fcn_decorator(x0, M, idx)
            fit_func_jac = fit_fcn_decorator(x0, M, idx, jac=True)

            if len(bnds) == 0:
                bnds = [np.ones_like(p0[idx]), np.ones_like(p0[idx])]*np.inf
            else:
                bnds = [bnds[0][idx], bnds[1][idx]]

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', OptimizeWarning)
                #try:
                x, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='dogbox', bounds=bnds,
                                        jac='3-point', x_scale='jac', max_nfev=len(p0)*1000)
                #except:
                #    return

        else:
            s = np.ones_like(idx)
            fit_func = fit_fcn_decorator(x0, M, idx, s=s)
            fit_func_jac = lambda x: approx_fprime(x, lambda x: fit_func(T, x/s + p0), 1E-5)
            nlopt_fit_fcn = nlopt_fit_fcn_decorator(fit_func, fit_func_jac, x0, idx, T, ln_k)


            #s[:] = fit_func_jac(np.zeros_like(p0))
            #print('s', s)
            #s[s == 0.0] = 1E-9
            #s[s==0] = 10**(OoM(np.min(s[s!=0])) - 1)  # TODO: MAKE THIS BETTER running into problem when s is zero, this is a janky workaround
            s[:] = np.median(np.abs(s), axis=0)

            opt = nlopt.opt(nlopt.LN_SBPLX, len(idx)) # nlopt.LN_SBPLX nlopt.LN_COBYLA nlopt.LD_MMA nlopt.LD_LBFGS

            opt.set_min_objective(nlopt_fit_fcn)
            #opt.set_maxeval(int(options['stop_criteria_val'])-1)
            #opt.set_maxtime(options['stop_criteria_val']*60)

            opt.set_xtol_rel(1E-2)
            opt.set_ftol_rel(1E-2)
            #opt.set_lower_bounds((bnds[0][idx]-p0)*s)
            #opt.set_upper_bounds((bnds[1][idx]-p0)*s)
            opt.set_lower_bounds(bnds[0][idx])
            opt.set_upper_bounds(bnds[1][idx])


            print('p0 ', p0)
            #opt.set_initial_step(np.min(s[s != 1E-9]))
            x = opt.optimize(p0) # optimize!

        x = np.array([*x0[:6], *x])
    
    print(f'x {x[-4:]}')
    print(ln_k)
    fit_k = fit_fcn_decorator(x0, M, idx)(T, *x[6:])
    print(fit_k)
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
    x0 = np.array(x0)
    coefNames = np.array(coefNames)
    bnds = np.array(bnds)

    # Faster and works for extreme values like n = -70
    if type(rxn) is ct.ElementaryReaction or type(rxn) is ct.ThreeBodyReaction:
        #x0 = [mech.coeffs_bnds[rxnIdx]['rate'][coefName]['resetVal'] for coefName in mech.coeffs_bnds[rxnIdx]['rate']]
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
    
    x0 = deepcopy(x0)
    bnds = deepcopy(bnds)

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