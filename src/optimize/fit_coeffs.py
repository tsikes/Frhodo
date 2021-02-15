# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
import cantera as ct
import warnings
from copy import deepcopy
import nlopt
from scipy.optimize import curve_fit, OptimizeWarning, approx_fprime
from scipy.misc import logsumexp
from timeit import default_timer as timer

from convert_units import OoM
from optimize.optimize_misc_fcns import generalized_loss_fcn

Ru = ct.gas_constant
# Ru = 1.98720425864083

default_arrhenius_coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']

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

def fit_SRI(rates, T, M, x0=[], coefNames=default_arrhenius_coefNames, bnds=[]):
    def fit_fcn_decorator(x0, alter_idx, jac=False):               
        def set_coeffs(*args):
            coeffs = x0
            for n, idx in enumerate(alter_idx):
                coeffs[idx] = args[n]
            return coeffs
        
        def ln_SRI(T, *args):
            [Ea_0, A_0, n_0, Ea_inf, A_inf, n_inf, a, b, c, d, e] = set_coeffs(*args)
            k_0 = A_0*T**n_0*np.exp(-Ea_0/(Ru*T))
            k_inf = A_inf*T**n_inf*np.exp(-Ea_inf/(Ru*T))
            P_r = k_0/k_inf*M
            ln_k = np.log(d*k_inf*P_r/(1 + P_r)) + 1/(1+np.log10(P_r)**2)*logsumexp([-b/T, -T/c], b=[a, 1]) + e*np.log(T)
            
            return ln_k

        def ln_SRI_jac(T, *args):
            [Ea_0, A_0, n_0, Ea_inf, A_inf, n_inf, a, b, c, d, e] = set_coeffs(*args)
            k_0 = A_0*T**n_0*np.exp(-Ea_0/(Ru*T))
            k_inf = A_inf*T**n_inf*np.exp(-Ea_inf/(Ru*T))
            P_r = k_0/k_inf*M

            u = np.log(a*np.exp(-b/T)+np.exp(-T/c))
            dlnk_Pr_0 = 1/(P_r*(P_r + 1))
            dlnk_Pr_1 = -2*u*np.log10(P_r)/(P_r*(log10(P_r)**2 + 1)**2)
            dlnk_Pr = dlnk_Pr_0 + dlnk_Pr_1

            Arrhen_temp = M*T**(n_0-n_inf)/A_inf*np.exp((Ea_inf-Ea_0)/(Ru*T))
            temp_0_inf = M*A_0*T**n_0*np.exp(Ea_inf/(Ru*T))
            inf_temp = temp_0_inf/(A_inf*T**n_inf*np.exp(Ea_0/(Ru*T)) + temp_0_inf)

            abc = 1/(1 + np.log10(P_r)**2)/(a*np.exp(T/c) + np.exp(b/T))

            dlnk_d = {'Ea_0': -dlnk_Pr/(Ru*T)*A_0*Arrhen_temp, 
                      'A_0': dlnk_Pr*Arrhen_temp, 
                      'n_0': dlnk_Pr*A_0*np.log(T)*Arrhen_temp, 
                      'Ea_inf': -inf_temp/(Ru*T) - A_0/(R*T)*dlnk_Pr_1*Arrhen_temp, 
                      'A_inf': inf_temp/A_inf + A_0/A_inf*dlnk_Pr_1*Arrhen_temp, 
                      'n_inf': np.log(T)*inf_temp + A_0*np.log(T)*dlnk_Pr_1*Arrhen_temp, 
                      'a': abc*np.exp(T/c), 'b': -a/T*abc*np.exp(T/c), 'c': T/c**2*abc*np.exp(b/T),
                      'd': np.ones_like(T)/d, 'e': np.log(T)} 

            jac = np.array(dlink_d.values()).T
            return jac[:, alter_idx]

        if not jac:
            return ln_SRI
        else:
            return ln_SRI_jac

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
    
    fit_func = fit_fcn_decorator(x0, idx)
    fit_func_jac = fit_fcn_decorator(x0, idx, jac=True)
    p0 = x0[idx]

    if len(bnds) > 0:
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

def fit_generic(rates, T, P, X, rxnIdx, coefKeys, coefNames, mech, x0, bnds):
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
    
    rxn = mech.gas.reaction(rxnIdx)
    rates = np.array(rates)
    T = np.array(T)
    P = np.array(P)
    x0 = np.array(x0)
    coefNames = np.array(coefNames)
    bnds = np.array(bnds)

    # Faster and works for extreme values like n = -70
    if type(rxn) is ct.ElementaryReaction or type(rxn) is ct.ThreeBodyReaction:
        x0 = [mech.coeffs_bnds[rxnIdx]['rate'][coefName]['resetVal'] for coefName in mech.coeffs_bnds[rxnIdx]['rate']]
        coeffs = fit_arrhenius(rates, T, x0=x0, coefNames=coefNames, bnds=bnds)

        if type(rxn) is ct.ThreeBodyReaction and 'pre_exponential_factor' in coefNames:
            A_idx = np.argwhere(coefNames == 'pre_exponential_factor')[0]

            if not rxn.efficiencies:
                M = 1/mech.gas.density_mole
            else:
                M = 0
                for (s, conc) in zip(mech.gas.species_names, mech.gas.concentrations):
                    if s in rxn.efficiencies:
                        M += conc*rxn.efficiencies[s]
                    else:
                        M += conc
            
            coeffs[A_idx] = coeffs[A_idx]/M

    elif type(rxn) is ct.FalloffReaction:
        x0 = np.array(x0)
        coefNames = np.array(coefNames)
        bnds = np.array(bnds)
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
                lnk = np.log(rates[idxs])
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