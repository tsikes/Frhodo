# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
import cantera as ct


Ru = ct.gas_constant
min_pos_system_value = np.finfo(float).eps*(1E2)
max_pos_system_value = (np.finfo(float).max*(1E-20))**(1/3)
min_neg_system_value = -max_pos_system_value
T_max = 6000

default_arrhenius_coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']

def rates(rxn_coef_opt, mech):
    output = []
    for rxn_coef in rxn_coef_opt:
        rxnIdx = rxn_coef['rxnIdx']
        for n, (T, P) in enumerate(zip(rxn_coef['T'], rxn_coef['P'])):
            if n < len(rxn_coef['key']):
                coeffs_key = rxn_coef['key'][n]['coeffs']
                coeffs_bnds_key = rxn_coef['key'][n]['coeffs_bnds']
            else:
                coeffs_bnds_key = coeffs_key = None           

            if type(coeffs_bnds_key) is str and 'rate' in coeffs_bnds_key:
                A = mech.coeffs[rxnIdx][coeffs_key]['pre_exponential_factor']
                b = mech.coeffs[rxnIdx][coeffs_key]['temperature_exponent']
                Ea = mech.coeffs[rxnIdx][coeffs_key]['activation_energy']

                k = A*T**b*np.exp(-Ea/Ru/T)

                output.append(k)

            else:
                mech.set_TPX(T, P)
                output.append(mech.gas.forward_rate_constants[rxnIdx])
            
    return np.log(output)

def weighted_quantile(values, quantiles, weights=None, values_sorted=False, old_style=False):
    """ https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    nonNan_idx = np.where(values!=np.nan)
    values = np.array(values[nonNan_idx])
    quantiles = np.array(quantiles)
    if weights is None or len(weights) == 0:
        weights = np.ones(len(values))
    weights = np.array(weights[nonNan_idx])
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]

    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    if old_style: # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(weights)
    return np.interp(quantiles, weighted_quantiles, values)

def outlier(x, a=2, c=1, weights=[], max_iter=25, percentile=0.25):
    def diff(x_outlier):
        if len(x_outlier) < 2: 
            return 1
        else:
            return np.diff(x_outlier)[0]

    x = np.abs(x.copy())
    percentiles = [percentile, 1-percentile]
    x_outlier = []
    if a != 2: # define outlier with 1.5 IQR rule
        for n in range(max_iter):
            if diff(x_outlier) == 0:   # iterate until res_outlier is the same as prior iteration
                break
                
            if len(x_outlier) > 0:
                x = x[x < x_outlier[-1]] 
            
            [q1, q3] = weighted_quantile(x, percentiles, weights=weights)
            iqr = q3 - q1       # interquartile range      
            
            if len(x_outlier) == 2:
                del x_outlier[0]
            
            x_outlier.append(q3 + iqr*1.5)
        
        x_outlier = x_outlier[-1]
    else:
        x_outlier = 1

    return c*x_outlier
    
def generalized_loss_fcn(x, mu=0, a=2, c=1):    # defaults to L2 loss
    x_c_2 = ((x-mu)/c)**2
    
    if a == 1:          # generalized function reproduces
        loss = (x_c_2 + 1)**(0.5) - 1
    if a == 2:
        loss = 0.5*x_c_2
    elif a == 0:
        loss = np.log(0.5*x_c_2+1)
    elif a == -2:       # generalized function reproduces
        loss = 2*x_c_2/(x_c_2 + 4)
    elif a <= -1000:    # supposed to be negative infinity
        loss = 1 - np.exp(-0.5*x_c_2)
    else:
        loss = np.abs(a-2)/a*((x_c_2/np.abs(a-2) + 1)**(a/2) - 1)

    return loss*c**a + mu  # multiplying by c^2 is not necessary, but makes order appropriate

def set_bnds(mech, rxnIdx, keys, coefNames):
    rxn = mech.gas.reaction(rxnIdx)
    coef_bnds = {'lower': [], 'upper': [], 'exist': []}
            
    for coefNum, (key, coefName) in enumerate(zip(keys, coefNames)):
        if coefName not in default_arrhenius_coefNames: continue    # skip anything not Arrhenius. Falloff follows this

        coef_x0 = mech.coeffs_bnds[rxnIdx][key['coeffs_bnds']][coefName]['resetVal']
        coef_limits = mech.coeffs_bnds[rxnIdx][key['coeffs_bnds']][coefName]['limits']()

        if np.isnan(coef_limits).any():
            coef_bnds['exist'].append([False, False])
            # set lower bnds
            if coefName == 'activation_energy':
                if coef_x0 > 0:
                    coef_bnds['lower'].append(0)                                # Ea shouldn't change sign
                else:
                    coef_bnds['lower'].append(-Ru*T_max*np.log(max_pos_system_value))
            elif coefName == 'pre_exponential_factor':
                coef_bnds['lower'].append(min_pos_system_value)             # A should be positive
            elif not isinstance(coefName, int):     # ints will be falloff, they will be taken care of below
                coef_bnds['lower'].append(min_neg_system_value)
                    
            # set upper bnds
            if coefName == 'activation_energy' and coef_x0 < 0:   # Ea shouldn't change sign
                coef_bnds['upper'].append(0)
            elif coefName == 'temperature_exponent':
                coef_bnds['upper'].append(np.log(max_pos_system_value)/np.log(T_max))
            elif not isinstance(coefName, int):
                coef_bnds['upper'].append(max_pos_system_value)
        else:
            coef_bnds['lower'].append(coef_limits[0])
            coef_bnds['upper'].append(coef_limits[1])
            coef_bnds['exist'].append([True, True])
            
    if type(rxn) in [ct.FalloffReaction, ct.PlogReaction]:
        for coef in ['A', 'T3', 'T1', 'T2']:
            coef_bnds['exist'].append([False, False])
            coef_bnds['lower'].append(min_neg_system_value)      
            coef_bnds['upper'].append(max_pos_system_value)

    coef_bnds['exist'] = np.array(coef_bnds['exist'])
    coef_bnds['lower'] = np.array(coef_bnds['lower'])
    coef_bnds['upper'] = np.array(coef_bnds['upper'])

    return coef_bnds