# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
import cantera as ct
import warnings
from copy import deepcopy
import nlopt
from scipy.optimize import curve_fit, OptimizeWarning, approx_fprime
from scipy.special import logsumexp
from timeit import default_timer as timer

from convert_units import OoM
from optimize.misc_fcns import generalized_loss_fcn
from optimize.fit_SRI import fit_SRI

Ru = ct.gas_constant
# Ru = 1.98720425864083

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
        A_idx = np.argwhere(coefNames == 'pre_exponential_factor')[0]
    
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

def fit_Troe(rates, T, P, X, rxnIdx, coefKeys, coefNames, mech, x0, bnds):
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

def fit_Troe_no_ct(rates, T, M, x0=[], coefNames=default_Troe_coefNames, bnds=[]):
    def fit_fcn_decorator(x0, alter_idx):               
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
            Fcent = (1-A)*np.exp(-T/T3)+A*np.exp(-T/T1)+np.exp(-T2/T)
            log_Fcent = np.log10(Fcent)
            C = -0.4 - 0.67*log_Fcent
            N = 0.75 - 1.27*log_Fcent
            f1 = (log_P_r + C)/(N - 0.14*(log_P_r + C))

            e = np.exp(1)

            ln_F = log_Fcent/np.log10(e)/(1+f1**2)

            ln_k = np.log(k_inf*P_r/(1 + P_r)) + ln_F
            
            return ln_k

        return ln_Troe

    ln_k = np.log(rates)
    
    alter_idx = []
    for n, coefName in enumerate(default_Troe_coefNames): # ['Ea_0', 'A_0', 'n_0', 'Ea_inf', 'A_inf', 'n_inf', 'A', 'T3', 'T1', 'T2']
        if coefName in coefNames:
            alter_idx.append(n)
    
    if (set([0, 1, 2]) & set(alter_idx)) and len(x0) == 0:
        a0 = np.polyfit(np.reciprocal(T[0:3]), ln_k[0:3], 1)
        x0[0:3] = np.array([-a0[0]*Ru, np.exp(a0[1]), 0])

    if (set([3, 4, 5]) & set(alter_idx)) and len(x0) < 4:
        a0 = np.polyfit(np.reciprocal(T[3:6]), ln_k[3:6], 1)
        x0[3:6] = np.array([-a0[0]*Ru, np.exp(a0[1]), 0])

    if len(x0) < 7:
        x0[6:9] = [0.1, 100, 1000, 10000] # initial guesses for fitting Troe if none exist

    x0[1] = np.log(x0[1])
    x0[4] = np.log(x0[4])

    x0 = np.array(x0)

    A_idx = None
    if set(['A_0', 'A_inf']) & set(coefNames):
        A_idx = np.argwhere(coefNames in ['A_0', 'A_inf'])

    fit_func = fit_fcn_decorator(x0, alter_idx)
    p0 = x0[alter_idx]

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
            popt, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='dogbox', bounds=bnds,
                                jac='2-point', x_scale='jac', max_nfev=len(p0)*1000)
    else:           
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            popt, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='dogbox',
                                jac='2-point', x_scale='jac', max_nfev=len(p0)*1000)
    
    if A_idx is not None:
        popt[A_idx] = np.exp(popt[A_idx])

    return popt

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
        if rxn.falloff.type == 'Troe':
            coeffs = fit_Troe(rates, T, P, X, rxnIdx, coefKeys, coefNames, mech, x0, bnds)
        elif rxn.falloff.type == 'SRI':
            SRI_coefNames = []
            for key, coefName in zip(coefKeys, coefNames):
                if coefName == 'activation_energy':
                    SRI_coefNames.append('Ea')
                elif coefName == 'pre_exponential_factor':
                    SRI_coefNames.append('A')
                elif coefName == 'temperature_exponent':
                    SRI_coefNames.append('n')

                if key['coeffs'] == 'low_rate':
                    SRI_coefNames[-1] = f'{SRI_coefNames[-1]}_0'
                elif key['coeffs'] == 'high_rate':
                    SRI_coefNames[-1] = f'{SRI_coefNames[-1]}_inf'
            
            SRI_coefNames.extend(['a', 'b', 'c', 'd', 'e'])
            coeffs = fit_SRI(rates, T, mech.M(rxn), x0, coefNames=SRI_coefNames, bnds=bnds)

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