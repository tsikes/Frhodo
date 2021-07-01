# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
import cantera as ct
import nlopt
import warnings
from copy import deepcopy
from scipy.optimize import curve_fit, minimize, root_scalar, OptimizeWarning, least_squares, approx_fprime
from timeit import default_timer as timer
import itertools

from calculate.convert_units import OoM
from calculate.optimize.misc_fcns import generalized_loss_fcn

Ru = ct.gas_constant
# Ru = 1.98720425864083

min_pos_system_value = (np.finfo(float).tiny*(1E20))**(1/2)
max_pos_system_value = (np.finfo(float).max*(1E-20))**(1/2)
min_ln_val = np.log(min_pos_system_value)
max_ln_val = np.log(max_pos_system_value)

default_arrhenius_coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']
default_Troe_coefNames = ['Ea_0', 'A_0', 'n_0', 'Ea_inf', 'A_inf', 'n_inf', 'A', 'T3', 'T1', 'T2']

troe_falloff_0 = [[0.6, 200, 600, 1200],            # (0, 0, 0)
                                                    # (0, 0, 1)
                [0.05,   1000,  -2000,   3000],     # (0, 1, 0)
                [-0.3,   200,   -20000,   -50],     # (0, 1, 1)
                [0.9,   -2000,   500,    10000],    # (1, 0, 0)
                                                    # (1, 0, 1)
                                                    # (1, 1, 0)
                                                    # (1, 1, 1)
                ]

#troe_min_pos_system_value = (np.finfo(float).tiny*(1E20))**(1/3)
#troe_max_pos_system_value = (np.finfo(float).max*(1E-20))**(1/3)
#troe_min_neg_system_value = -max_pos_system_value
#troe_min_ln_val = np.log(troe_min_pos_system_value)
#troe_max_ln_val = np.log(troe_max_pos_system_value)
#T_max = 6000
#troe_all_bnds = {'A':  {'-': [troe_min_neg_system_value, troe_max_pos_system_value], 
#                        '+': [troe_min_neg_system_value, troe_max_pos_system_value]},
#                 'T3': {'-': [troe_min_neg_system_value, -T_max/troe_max_ln_val], 
#                        '+': [-T_max/troe_min_ln_val, troe_max_pos_system_value]},
#                 'T1': {'-': [troe_min_neg_system_value, -T_max/troe_max_ln_val], 
#                        '+': [-T_max/troe_min_ln_val, troe_max_pos_system_value]},
#                 'T2': {'-': [-T_max*troe_max_ln_val, -T_max*troe_min_ln_val], 
#                        '+': [-T_max*troe_max_ln_val, -T_max*troe_min_ln_val]}}
troe_all_bnds = {'A':  {'-': [-1E2, 1.0],    '+': [-1E2, 1.0]},
                 'T3': {'-': [-1E30, -30],   '+': [30.0, 1E30]},
                 'T1': {'-': [-1E30, -30],   '+': [30.0, 1E30]},
                 'T2': {'-': [-1E4, 1E30],  '+':  [-1E4, 1E30]}}

def fit_arrhenius(rates, T, x0=[], coefNames=default_arrhenius_coefNames, bnds=[], loss='linear'):
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
                                    jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000,
                                    loss=loss)
            except:
                return
    else:           
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            try:
                popt, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='trf',
                                    jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000,
                                    loss=loss)
            except:
                return
    
    if A_idx is not None:
        popt[A_idx] = np.exp(popt[A_idx])

    return popt


bisymlog_C = 1/(np.exp(1)-1)
class falloff_parameters:   # based on ln_Fcent
    def __init__(self, T, Fcent_target, x0=[0.6, 200, 600, 1200], use_scipy=False, 
                 nlopt_algo=nlopt.LN_SBPLX, loss_fcn_par=[2, 1]):
        self.T = T
        self.Fcent = Fcent_target
        self.x0 = x0
        self.s = np.ones_like(x0)

        self.Fcent_min = 1.1E-8
        self.Tmin = 100
        self.Tmax = 20000

        self.use_scipy = use_scipy

        self.opt_algo = nlopt_algo # nlopt.GN_DIRECT_L, nlopt.GN_DIRECT, nlopt.GN_CRS2_LM, LN_COBYLA, LN_SBPLX, LD_MMA
        self.loss_alpha = loss_fcn_par[0]   # warning unknown how well this functions outside of alpha=2, C=1
        self.loss_scale = loss_fcn_par[1]

    def x_bnds(self, x0):
        bnds = []
        for n, coef in enumerate(['A', 'T3', 'T1', 'T2']):
            if x0[n] < 0:
                bnds.append(troe_all_bnds[coef]['-'])
            else:
                bnds.append(troe_all_bnds[coef]['+'])
        return np.array(bnds).T

    def fit(self):
        Fcent = self.Fcent
        T = self.T
        x0 = self.x0
        p0 = self.convert(x0, 'base2opt')
        self.p_bnds = self.convert(self.x_bnds(x0))

        if self.use_scipy:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', OptimizeWarning)
                x_fit, _ = curve_fit(self.function, T, Fcent, p0=p0, method='trf', bounds=self.p_bnds, # dogbox
                                        #jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)
                                        jac='2-point', x_scale='jac', max_nfev=len(p0)*1000, loss='huber')

        #print('scipy:', x_fit)
        #cmp = np.array([T, Fcent, np.exp(fit_func(T, *x_fit))]).T
        #for entry in cmp:
        #    print(*entry)
        #print('')
        #scipy_fit = np.exp(self.function(T, *x_fit))

        else:
            xtol_rel = 1E-8
            ftol_rel = 1E-4
            initial_step = 1E-4

            self.opt = opt = nlopt.opt(nlopt.AUGLAG, 4)

            opt.set_min_objective(self.objective)
            opt.add_inequality_constraint(self.constraint, 1E-8)
            opt.set_maxeval(10000)
            opt.set_maxtime(10)

            opt.set_xtol_rel(xtol_rel)
            opt.set_ftol_rel(ftol_rel)

            self.p0 = p0
            p0_opt = np.zeros_like(p0)
            self.s = self.calc_s(p0_opt)

            opt.set_lower_bounds((self.p_bnds[0]-self.p0)/self.s)
            opt.set_upper_bounds((self.p_bnds[1]-self.p0)/self.s)

            opt.set_initial_step(initial_step)
            #opt.set_population(int(np.rint(10*(len(idx)+1)*10)))
            
            sub_opt = nlopt.opt(self.opt_algo, 4)
            sub_opt.set_initial_step(initial_step)
            sub_opt.set_xtol_rel(xtol_rel)
            sub_opt.set_ftol_rel(ftol_rel)
            opt.set_local_optimizer(sub_opt)

            x_fit = opt.optimize(p0_opt) # optimize!
            x_fit = x_fit*self.s + self.p0

        #print('nlopt:', x_fit)
        #cmp = np.array([T, Fcent, self.function(T, *x_fit)]).T
        ##cmp = np.array([T, Fcent, scipy_fit, np.exp(self.function(T, *x_fit))]).T
        #for entry in cmp:
        #    print(*entry)
        #print('')

        x = self.convert(x_fit, 'opt2base')
        res = {'x': x, 'fval': opt.last_optimum_value(), 'nfev': opt.get_numevals()}

        return res

    def convert(self, x, conv_type='base2opt'):
        #x = x*self.s + self.p0
        y = np.array(x)
        C = bisymlog_C

        flatten = False
        if y.ndim == 1:
            y = y[np.newaxis, :]
            flatten = True

        if conv_type == 'base2opt': # y = [A, T3, T1, T2]
            y[:,1:] = np.sign(y[:,1:])*np.log(np.abs(y[:,1:]/C) + 1)

        else:
            y[:,1:] = np.sign(y[:,1:])*C*(np.exp(np.abs(y[:,1:])) - 1)

            #A = np.log(y[0])
            #T3, T1 = 1000/y[1], 1000/y[2]
            #T2 = y[3]*100

        if flatten: # if it's 1d it's the guess
            y = y.flatten()
        else:       # if it 2d it's the bnds, they need to be sorted
            y = np.sort(y, axis=0)

        return y

    def function(self, T, *x):
        [A, T3, T1, T2] = self.convert(x, 'opt2base')

        Fcent_fit = self.Fcent_calc(T, A, T3, T1, T2)
        Fcent_fit[Fcent_fit <= 0.0] = 10000

        return Fcent_fit

    def Fcent_calc(self, T, A, T3, T1, T2):
        exp_T3 = np.zeros_like(T)
        exp_T3[:] = np.exp(-T/T3)

        exp_T1 = np.zeros_like(T)
        exp_T1[:] = np.exp(-T/T1)

        T2_T = T2/T
        exp_T2 = np.where(T2_T <= max_ln_val, np.exp(-T2_T), 0.0)

        Fcent = (1-A)*exp_T3 + A*exp_T1 + exp_T2

        return Fcent

    def jacobian(self, T, *x):
        [A, B, C, D] = x
        [A, T3, T1, T2] = self.convert(x, 'opt2base') # A, B, C, D = x
        bC = bisymlog_C

        jac = []
        jac.append((np.exp(-T/T1) - np.exp(-T/T3)))                   # dFcent/dA
        jac.append(bC*np.exp(np.abs(B))*(1-A)*T/T3**2*np.exp(-T/T3))  # dFcent/dB
        jac.append(bC*np.exp(np.abs(C))*A*T/T1**2*np.exp(-T/T1))      # dFcent/dC
        jac.append(-bC*np.exp(np.abs(D))/T*np.exp(-T2/T))             # dFcent/dD

        jac = np.vstack(jac).T

        return jac

    def objective(self, x_fit, grad=np.array([]), obj_type='obj_sum'):
        x = x_fit*self.s + self.p0
        T = self.T

        resid = self.function(T, *x) - self.Fcent
        if obj_type == 'obj_sum':              
            obj_val = generalized_loss_fcn(resid, a=self.loss_alpha, c=self.loss_scale).sum()
        elif obj_type == 'obj':
            obj_val = generalized_loss_fcn(resid, a=self.loss_alpha, c=self.loss_scale)
        elif obj_type == 'resid':
            obj_val = resid

        #s[:] = np.abs(np.sum(loss*fit_func_jac(T, *x).T, axis=1))
        if grad.size > 0:
            grad[:] = self.objective_gradient(x, resid)
        #else:
        #    grad = self.objective_gradient(x, resid)

        #self.s = self.calc_s(x_fit, grad)

        #self.opt.set_lower_bounds((self.p_bnds[0] - self.p0)/self.s)
        #self.opt.set_upper_bounds((self.p_bnds[1] - self.p0)/self.s)

        return obj_val
    
    def objective_gradient(self, x, resid=[], numerical_gradient=False):
        if numerical_gradient:
        #x = (x - self.p0)/self.s
            grad = approx_fprime(x, self.objective, 1E-10)
            
        else:
            if len(resid) == 0:
                resid = self.objective(x, obj_type='resid')

            x = x*self.s + self.p0
            T = self.T
            jac = self.jacobian(T, *x)
            if np.isfinite(jac).all():
                with np.errstate(all='ignore'):
                    grad = np.sum(jac.T*resid, axis=1)*self.s
                    grad[grad == np.inf] = max_pos_system_value
            else:
                grad = np.ones_like(self.p0)*max_pos_system_value

        return grad

    def calc_s(self, x, grad=[]):
        if len(grad) == 0:
            grad = self.objective_gradient(x)

        y = np.abs(grad)
        if (y < min_pos_system_value).all():
            y = np.ones_like(y)*1E-14
        else:
            y[y < min_pos_system_value] = 10**(OoM(np.min(y[y>=min_pos_system_value])) - 1)  # TODO: MAKE THIS BETTER running into problem when s is zero, this is a janky workaround
        
        s = 1/y
        #s = s/np.min(s)
        s = s/np.max(s)

        return s

    def constraint(self, x, grad=np.array([])):
        def f_fp(T, A, T3, T1, T2, fprime=False, fprime2=False): # dFcent_dT 
            f = T2/T**2*np.exp(-T2/T) - (1-A)/T3*np.exp(-T/T3) - A/T1*np.exp(-T/T1)

            if not fprime and not fprime2:
                return f
            elif fprime and not fprime2:
                fp = T2*(T2 - 2*T)/T**4*np.exp(-T2/T) + (1-A)/T3**2*np.exp(-T/T3) +A/T1**2*np.exp(-T/T1)
                return f, fp

        x = x*self.s + self.p0
        [A, T3, T1, T2] = self.convert(x, 'opt2base')
        Fcent_min = self.Fcent_min
        Tmin = self.Tmin
        Tmax = self.Tmax

        try:
            T_deriv_eq_0 = root_scalar(lambda T: f_fp(A, T3, T1, T2), 
                                        x0=(Tmax+Tmin)/4, x1=3*(Tmax+Tmin)/4, method='secant')
            T = np.array([Tmin, T_deriv_eq_0, Tmax])
        except:
            T = np.array([Tmin, Tmax])

        if len(T) == 3:
            print(T)

        Fcent = self.Fcent_calc(T, A, T3, T1, T2)   #TODO: OVERFLOW WARNING HERE
        con = np.min(Fcent_min - Fcent)

        if grad.size > 0:
            grad[:] = self.constraint_gradient(x, numerical_gradient=True)

        return con

    def constraint_gradient(self, x, const_eval=[], numerical_gradient=False):
        if numerical_gradient:
            grad = approx_fprime(x, self.constraint, 1E-10)
            
        else:   # I've not calculated the derivatives wrt coefficients for analytical
            if len(resid) == 0:
                const_eval = self.objective(x)

            T = self.T
            jac = self.jacobian(T, *x)
            if np.isfinite(jac).all():
                with np.errstate(all='ignore'):
                    grad = np.sum(jac.T*const_eval, axis=1)*self.s
                    grad[grad == np.inf] = max_pos_system_value
            else:
                grad = np.ones_like(self.x0)*max_pos_system_value

        return grad


def falloff_parameters_decorator(args_list):
    T_falloff, Fcent, x0, use_scipy, nlopt_algo = args_list
    falloff = falloff_parameters(T_falloff, Fcent, x0=x0, use_scipy=use_scipy, nlopt_algo=nlopt_algo) #GN_CRS2_LM LN_SBPLX
    return falloff.fit()


class Troe:
    def __init__(self, rates, T, M, x0=[], coefNames=default_Troe_coefNames, bnds=[], HPL_LPL_defined=True,  
                 scipy_curvefit=False, mpPool=None, nlopt_algo=nlopt.LN_SBPLX):

        self.debug = False

        self.k = rates
        self.ln_k = np.log(rates)
        self.T = T
        self.M = M

        self.x0 = x0
        if len(self.x0) != 10:
            self.x0[6:] = troe_falloff_0[0]
        self.x0 = np.array(x0)

        # only valid initial guesses
        self.bnds = np.array(bnds)
        if len(self.bnds) > 0:
            self.x0 = np.clip(self.x0, self.bnds[0, :], self.bnds[1, :])

        self.x = np.zeros((10, 1)).flatten()

        self.scipy_curvefit = scipy_curvefit
        self.HPL_LPL_defined = HPL_LPL_defined
        self.nlopt_algo = nlopt_algo
        self.pool = mpPool

        if self.HPL_LPL_defined:
            self.loss_fcn_par = [2, 1]
        else:
            self.loss_fcn_par = [1, 1]  # huber-like

        self.alter_idx = {'low_rate': [], 'high_rate': [], 'falloff_parameters': [], 'all': []}
        for n, coefName in enumerate(default_Troe_coefNames):
            if coefName in coefNames:
                self.alter_idx['all'].append(n)
                if coefName in ['Ea_0', 'A_0', 'n_0']:
                    self.alter_idx['low_rate'].append(n)
                elif coefName in ['Ea_inf', 'A_inf', 'n_inf']:
                    self.alter_idx['high_rate'].append(n)
                else:
                    self.alter_idx['falloff_parameters'].append(n)
    
    def fit(self):
        x = self.x
        start = timer()

        # fit LPL, HPL, Fcent
        res = self.LPL_HPL_Fcent(self.T, self.M, self.ln_k, self.x0, self.bnds)

        # fit Troe falloff parameters
        idx = self.alter_idx['falloff_parameters']
        #if len(self.x0) > 0 and list(self.x0[idx]) not in troe_falloff_0:
        #    x0 = np.array(troe_falloff_0)
        #    x0 = np.vstack([x0, self.x0[idx]])
        #else:
        #    x0 = troe_falloff_0
        x0 = troe_falloff_0

        T_falloff, Fcent = res[:,0], res[:,3]
        
        if self.pool is not None:
            args_list = ((T_falloff, Fcent, Troe_0, False, self.nlopt_algo) for Troe_0 in x0)
            falloff_output = self.pool.map(falloff_parameters_decorator, args_list)
        else:
            falloff_output = []
            for i, Troe_0 in enumerate(x0):
                falloff = falloff_parameters(T_falloff, Fcent, x0=Troe_0, use_scipy=False, 
                                             nlopt_algo=nlopt.LN_SBPLX, loss_fcn_par=self.loss_fcn_par) #GN_CRS2_LM LN_SBPLX
                falloff_output.append(falloff.fit())

        HoF = {'obj_fcn': np.inf, 'coeffs': []}
        for i, res in enumerate(falloff_output):
            if res['fval'] < HoF['obj_fcn']:
                HoF['obj_fcn'] = res['fval']
                HoF['coeffs'] = res['x']
                HoF['i'] = i

        x[idx] = HoF['coeffs']

        if self.debug:
            T = self.T
            M = self.M
            ln_k = self.ln_k

            ln_k_0 = np.log(x[1]) + x[2]*np.log(T) - x[0]/(Ru*T)
            ln_k_inf = np.log(x[4]) + x[5]*np.log(T) - x[3]/(Ru*T)
            ln_Fcent = np.log((1-x[6])*np.exp(-T/x[7]) + x[6]*np.exp(-T/x[8]) + np.exp(-x[9]/T))

            cmp = np.array([T, M, ln_k, self.ln_Troe(M, ln_k_0, ln_k_inf, ln_Fcent)]).T
            for entry in cmp:
                print(*entry)
            print('')

        return x

    def LPL_HPL_Fcent(self, T, M, ln_k, x0, bnds):
        x = self.x
        alter_idx = self.alter_idx

        if self.HPL_LPL_defined:    # if Troe or SRI set so that LPL and HPL are explicitly defined
            rates = self.k

            # Fit HPL and LPL
            for arrhenius_type in ['low_rate', 'high_rate']:
                idx = alter_idx[arrhenius_type]
                if len(idx) > 0:
                    x[idx] = fit_arrhenius(rates[idx], T[idx], x0=x0[idx], bnds=[bnds[0][idx], bnds[1][idx]]) # [Ea, A, n]

            # Fit Falloff
            ln_k_0 = np.log(x[1]) + x[2]*np.log(T) - x[0]/(Ru*T)
            ln_k_inf = np.log(x[4]) + x[5]*np.log(T) - x[3]/(Ru*T)

            res = []
            for idx in alter_idx['falloff_parameters']:  # keep T constant and fit ln_k_0, ln_k_inf, ln_Fcent
                #fit_func = lambda x: (fit_const_T_decorator(ln_k[idx], T[idx])(M[idx], [ln_k_0[idx], ln_k_inf[idx], x[0]]) - ln_k[idx])**2 # only fitting Fcent
                fit_func = lambda M, x: self.ln_Troe(M, [ln_k_0[idx], ln_k_inf[idx], x])
                if len(res) == 0:
                    p0 = [-0.5]                      # ln(k_0), ln(k_inf), ln(Fcent)
                else:
                    p0 = np.log(res[-1][3:])

                p_bnds = np.log([[0.1], [1]])  # ln(Fcent)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', OptimizeWarning)
                    x_fit, _ = curve_fit(fit_func, M[idx], ln_k[idx], p0=p0, method='trf', bounds=p_bnds, # dogbox
                                                        #jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)
                                                        jac='2-point', x_scale='jac', max_nfev=len(p0)*1000)

                #temp_res = minimize(fit_func, x0=p0, method='L-BFGS-B', bounds=p_bnds, jac='2-point')

                res.append([T[idx], *np.exp([ln_k_0[idx], ln_k_inf[idx], *x_fit])])

                #cmp = np.array([T[idx], M[idx], ln_k[idx], fit_func(M[idx], *x_fit)])
                #print(*cmp)

            res = np.array(res)

        else:   # if fitting just from rates subject to falloff
            T_idx_unique = np.unique(self.T, return_index=True)[1]
            P_len = T_idx_unique[1] - T_idx_unique[0]
            p_bnds = np.log([[1E-12, 1E-12, 0.1], [1E30, 1E30, 1]])  # ln(k_0), ln(k_inf), ln(Fcent)
            res = []
            for idx_start in T_idx_unique:  # keep T constant and fit log_k_0, log_k_inf, log_Fcent
                idx = np.arange(idx_start, idx_start+P_len)

                fit_func = self.ln_Troe
                if len(res) == 0:
                    p0 = [27, 14, -0.5]       # ln(k_0), ln(k_inf), ln(Fcent)
                else:
                    p0 = np.log(res[-1][1:])

                x_fit, _ = curve_fit(fit_func, M[idx], ln_k[idx], p0=p0, method='trf', bounds=p_bnds, # dogbox
                                                        #jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)
                                                        jac='2-point', x_scale='jac', max_nfev=len(p0)*1000, loss='huber')

                res.append([T[idx_start], *np.exp(x_fit)])

                #cmp = np.array([T[idx], M[idx], ln_k[idx], fit_func(M[idx], *x_fit)]).T
                #for entry in cmp:
                #    print(*entry)
                #print('')

            res = np.array(res)

            # Fit HPL and LPL Arrhenius parameters
            for res_idx, arrhenius_type in enumerate(['low_rate', 'high_rate']):
                idx = alter_idx[arrhenius_type]
                if len(idx) > 0:
                    x[idx] = fit_arrhenius(res[:, res_idx+1], res[:,0], bnds=[bnds[0][idx], bnds[1][idx]], loss='huber') # 'soft_l1', 'huber', 'cauchy', 'arctan'

        return res


    def ln_Troe(self, M, *x):
        if len(x) == 1:
            x = x[0]

        [k_0, k_inf, Fcent] = np.exp(x)
        with np.errstate(all='raise'):
            try:
                P_r = k_0/k_inf*M
                log_P_r = np.log10(P_r)
            except:
                return np.ones_like(M)*max_pos_system_value           

        log10_Fcent = np.log10(Fcent)
        C = -0.4 - 0.67*log10_Fcent
        N = 0.75 - 1.27*log10_Fcent
        f1 = (log_P_r + C)/(N - 0.14*(log_P_r + C))

        ln_F = np.log(Fcent)/(1 + f1**2)

        ln_k_calc = np.log(k_inf*P_r/(1 + P_r)) + ln_F

        return ln_k_calc


def fit_generic(rates, T, P, X, rxnIdx, coefKeys, coefNames, mech, x0, bnds, mpPool):    
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

        falloff_coefNames.extend(['A', 'T3', 'T1', 'T2'])
        Troe_parameters = Troe(rates, T, M, x0=x0, coefNames=falloff_coefNames, bnds=bnds, 
                               scipy_curvefit=False, HPL_LPL_defined=True, mpPool=mpPool)
        coeffs = Troe_parameters.fit()

    return coeffs


def fit_coeffs(rates, T, P, X, rxnIdx, coefKeys, coefNames, x0, bnds, mech, mpPool=None): 
    if len(coefNames) == 0: return # if not coefs being optimized in rxn, return 

    return fit_generic(rates, T, P, X, rxnIdx, coefKeys, coefNames, mech, x0, bnds, mpPool)
    

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