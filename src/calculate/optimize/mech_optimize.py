# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

from qtpy.QtCore import QThreadPool
import numpy as np
import cantera as ct
import multiprocessing as mp
from copy import deepcopy
from timeit import default_timer as timer

from scipy import stats

from calculate.optimize.optimize_worker import Worker
from calculate.optimize.fit_fcn import update_mech_coef_opt
from calculate.optimize.misc_fcns import rates, set_bnds
from calculate.optimize.fit_coeffs import fit_Troe


default_arrhenius_coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']

class Multithread_Optimize:
    def __init__(self, parent):
        self.parent = parent

        # Initialize Threads
        parent.optimize_running = False
        # parent.threadpool = QThreadPool()
        # parent.threadpool.setMaxThreadCount(2) # Sets thread count to 1 (1 for gui - this is implicit, 1 for calc)
        # log_txt = 'Multithreading with maximum {:d} threads\n'.format(parent.threadpool.maxThreadCount())
        # parent.log.append(log_txt, alert=False)

        # Set Distribution
        self.dist = stats.gennorm

        # Connect Toolbar Functions
        parent.action_Run.triggered.connect(self.start_threads)
        parent.action_Abort.triggered.connect(self.abort_workers)
     
    def start_threads(self):
        parent = self.parent
        parent.path_set.optimized_mech()

        self.last_plot_timer = 0.0
        self.time_between_plots = 0.0  # maximum update rate, updated based on time to plot

        ## Check fit_coeffs
        #from optimize.fit_coeffs import debug
        #debug(parent.mech)

        if parent.directory.invalid: 
            parent.log.append('Invalid directory found\n')
            return
        elif parent.optimize_running:
            parent.log.append('Optimize running flag already set to True\n')
            return
        elif not self._has_opt_rxns(): # if nothing to optimize, don't!
            parent.log.append('No reactions or coefficients set to be optimized\n')
            return         
                
        # Set shocks to be run
        self.shocks2run = []
        for series in parent.series.shock:
            for shock in series:
                # skip not included or exp_data not loaded from experiment
                if not shock['include'] or 'exp_data' in shock['err']: 
                    shock['SIM'] = None
                    continue

                self.shocks2run.append(shock)
        
        if len(self.shocks2run) == 0: # optimize current shock if nothing selected
            self.shocks2run = [parent.display_shock]
        else:
            if not parent.load_full_series_box.isChecked(): # TODO: May want to remove this limitation in future
                parent.log.append('"Load Full Series Into Memory" must be checked for optimization of multiple experiments\n')
                return
            elif len(parent.series_viewer.data_table) == 0:
                parent.log.append('Set Series in Series Viewer and select experiments\n')
                return
        
        if len(self.shocks2run) == 0: return    # if no shocks to run return, not sure if necessary anymore

        # Initialize variables in shocks if need be
        opt_options = self.parent.optimization_settings.settings
        weight_fcn = parent.series.weights
        unc_fcn = parent.series.uncertainties
        for shock in self.shocks2run:      # TODO NEED TO UPDATE THIS FOR UNCERTAINTIES AND WEIGHTS
            # if weight variables aren't set, update
            if opt_options['obj_fcn']['type'] == 'Residual':
                weight_var = [shock[key] for key in ['weight_max', 'weight_min', 'weight_shift', 'weight_k']]
                if np.isnan(np.hstack(weight_var)).any():
                    parent.weight.update(shock=shock)
                    shock['weights'] = weight_fcn(shock['exp_data'][:,0], shock)
            else: # otherwise bayesian
                unc_var = [shock[key] for key in ['unc_max', 'unc_min', 'unc_shift', 'unc_k', 'unc_cutoff']]
                if np.isnan(np.hstack(unc_var)).any():
                    parent.exp_unc.update(shock=shock)
                    shock['uncertainties'] = unc_fcn(shock['exp_data'][:,0], shock, calcWeights=True)

            # if reactor temperature and pressure aren't set, update
            if np.isnan([shock['T_reactor'], shock['P_reactor']]).any():
                parent.series.set('zone', shock['zone'])
                    
            parent.series.rate_bnds(shock)
        
        # Set conditions of rates to be fit for each opt coefficient and rates/bnds
        self._initialize_opt()
        self._update_gas()
        
        parent.multiprocessing = parent.multiprocessing_box.isChecked()
        
        parent.update_user_settings()
        # parent.set_weights()
        
        parent.abort = False
        parent.optimize_running = True
        
        # Create mechs and duplicate mech variables
        if parent.multiprocessing == True:
            cpu_count = mp.cpu_count()
            # if cpu_count > 1: # leave open processor
                # cpu_count -= 1
            parent.max_processors = np.min([len(self.shocks2run), cpu_count])
            if parent.max_processors == 1:      # if only 1 shock, turn multiprocessing off
                parent.multiprocessing = False
            
            log_str = 'Number of processors: {:d}'.format(parent.max_processors)
            parent.log.append(log_str, alert=False)
        else:
            parent.max_processors = 1
        
        # Pass the function to execute
        self.worker = Worker(parent, self.shocks2run, parent.mech, self.coef_opt, self.rxn_coef_opt, self.rxn_rate_opt)
        self.worker.signals.result.connect(self.on_worker_done)
        self.worker.signals.finished.connect(self.thread_complete)
        self.worker.signals.update.connect(self.update)
        self.worker.signals.progress.connect(self.on_worker_progress)
        self.worker.signals.log.connect(parent.log.append)
        self.worker.signals.abort.connect(self.worker.abort)
        
        # Optimization plot
        parent.plot.opt.clear_plot()

        # Reset Hall of Fame
        self.HoF = []   # hall of fame for tracking the best result so far

        # Create Progress Bar
        # parent.create_progress_bar()
                
        if not parent.abort:
            s = 'Optimization starting\n\n   Iteration\t\t Objective Func\tBest Objetive Func'
            parent.log.append(s, alert=False)
            parent.threadpool.start(self.worker)
    
    def _has_opt_rxns(self):              
        mech = self.parent.mech
        for rxnIdx in range(mech.gas.n_reactions):      # searches all rxns
            if mech.rate_bnds[rxnIdx]['opt']: 
                return True
        return False    

    def _initialize_opt(self):  # initialize various dictionaries for optimization
        self.coef_opt = self._set_coef_opt()
        self.rxn_coef_opt = self._set_rxn_coef_opt()
        self.rxn_rate_opt = self._set_rxn_rate_opt()

    def _set_coef_opt(self):                   
        mech = self.parent.mech
        coef_opt = []
        #for rxnIdx in range(mech.gas.n_reactions):      # searches all rxns
        for rxnIdx, rxn in enumerate(mech.gas.reactions()):      # searches all rxns
            if not mech.rate_bnds[rxnIdx]['opt']: continue        # ignore fixed reactions

            # check all coefficients
            for bndsKey, subRxn in mech.coeffs_bnds[rxnIdx].items():
                for coefIdx, (coefName, coefDict) in enumerate(subRxn.items()):
                    if coefDict['opt']:
                        coefKey, bndsKey = mech.get_coeffs_keys(rxn, bndsKey, rxnIdx=rxnIdx)
                        coef_opt.append({'rxnIdx': rxnIdx, 'key': {'coeffs': coefKey, 'coeffs_bnds': bndsKey}, 
                                         'coefIdx': coefIdx, 'coefName': coefName})

        return coef_opt                    
    
    def _set_rxn_coef_opt(self, min_T_range=1000, min_P_range=1E4):
        coef_opt = deepcopy(self.coef_opt)
        mech = self.parent.mech
        rxn_coef_opt = []
        for coef in coef_opt:
            if len(rxn_coef_opt) == 0 or coef['rxnIdx'] != rxn_coef_opt[-1]['rxnIdx']:
                rxn_coef_opt.append(coef)
                rxn_coef_opt[-1]['key'] = [rxn_coef_opt[-1]['key']]
                rxn_coef_opt[-1]['coefIdx'] = [rxn_coef_opt[-1]['coefIdx']]
                rxn_coef_opt[-1]['coefName'] = [rxn_coef_opt[-1]['coefName']]
            else:
                rxn_coef_opt[-1]['key'].append(coef['key'])
                rxn_coef_opt[-1]['coefIdx'].append(coef['coefIdx'])
                rxn_coef_opt[-1]['coefName'].append(coef['coefName'])

        # Generate shock conditions to be optimized
        shock_conditions = {'T_reactor': [], 'P_reactor': [], 'thermo_mix': []}
        for shock in self.shocks2run:
            for shock_condition in shock_conditions:
                shock_conditions[shock_condition].append(shock[shock_condition])

        for rxn_coef in rxn_coef_opt:
            # Set coefficient initial values and bounds
            rxnIdx = rxn_coef['rxnIdx']
            rxn = mech.gas.reaction(rxnIdx)
            rxn_coef['coef_x0'] = []
            
            for coefNum, (key, coefName) in enumerate(zip(rxn_coef['key'], rxn_coef['coefName'])):
                coef_x0 = mech.coeffs_bnds[rxnIdx][key['coeffs_bnds']][coefName]['resetVal']
                rxn_coef['coef_x0'].append(coef_x0)

            rxn_coef['coef_bnds'] = set_bnds(mech, rxnIdx, rxn_coef['key'], rxn_coef['coefName'])

            # Set evaluation rate conditions
            T_bnds = np.array([np.min(shock_conditions['T_reactor']), np.max(shock_conditions['T_reactor'])])
            if T_bnds[1] - T_bnds[0] < min_T_range:  # if T_range isn't large enough increase it
                T_median = np.median(T_bnds)
                T_bnds = np.array([T_median-min_T_range/2, T_median+min_T_range/2])
                if T_bnds[0] < 298.15:
                    T_bnds[1] += 298.15 - T_bnds[0]
                    T_bnds[0] = 298.15

            invT_bnds = np.divide(10000, T_bnds)
            
            if type(rxn) is ct.PlogReaction:
                P = []
                for PlogRxn in mech.coeffs[rxnIdx]:
                    P.append(PlogRxn['Pressure'])
                
                if len(P) < 4:
                    P = np.geomspace(np.min(P), np.max(P), 4)

                P = np.roll(P, 1)       # put low pressure first followed by high pressure and the rest
                P[[0, 1]] = P[[1, 0]]
            else:
                P = np.median(shock_conditions['P_reactor'])

            if type(rxn) in [ct.ElementaryReaction, ct.ThreeBodyReaction]:
                n_coef = len(rxn_coef['coefIdx'])
                rxn_coef['invT'] = np.linspace(*invT_bnds, n_coef)
                rxn_coef['T'] = np.divide(10000, rxn_coef['invT'])
                rxn_coef['P'] = np.ones_like(rxn_coef['T'])*P

            elif type(rxn) is ct.PlogReaction:
                invT = np.linspace(*invT_bnds, 3) # fit 3 temperatures per arrhenius expression

                rxn_coef['invT'], rxn_coef['P'] = np.meshgrid(invT, P)
                rxn_coef['invT'] = rxn_coef['invT'].flatten()
                rxn_coef['P'] = rxn_coef['P'].flatten()
                rxn_coef['T'] = np.divide(10000, rxn_coef['invT'])

            elif type(rxn) is ct.FalloffReaction:
                rxn_coef['T'] = []
                rxn_coef['invT'] = []
                rxn_coef['P'] = []
                for coef_type in ['low_rate', 'high_rate']:
                    n_coef = 0
                    for coef in rxn_coef['key']:
                        if coef_type in coef['coeffs_bnds']:
                            n_coef += 1
                   
                    rxn_coef['invT'].append(np.linspace(*invT_bnds, n_coef))
                    rxn_coef['T'].append(np.divide(10000, rxn_coef['invT'][-1]))
                    if coef_type == 'low_rate':
                        rxn_coef['P'].append(np.ones(n_coef)*1E-1) # will evaluate LPL
                    elif coef_type == 'high_rate':
                        rxn_coef['P'].append(np.ones(n_coef)*1E8) # will evaluate HPL
                    
                rxn_coef['T'].append(np.linspace(*T_bnds, 6)[1:-1])
                rxn_coef['invT'].append(np.divide(10000, rxn_coef['T'][-1]))
                rxn_coef['P'].append(np.ones(4)*P) # Doesn't matter, will evaluate median P for falloff

                for key in ['T', 'invT', 'P']:
                    rxn_coef[key] = np.concatenate(rxn_coef[key], axis=0)

            rxn_coef['X'] = shock_conditions['thermo_mix'][0]   # TODO: IF MIXTURE COMPOSITION FOR DUMMY RATES MATTER CHANGE HERE

        return rxn_coef_opt

    def _set_rxn_rate_opt(self):
        mech = self.parent.mech
        rxn_rate_opt = {}

        # Calculate x0 (initial rates)
        prior_coeffs = mech.reset()  # reset mechanism and get mech that it was
        rxn_rate_opt['x0'] = rates(self.rxn_coef_opt, mech)
        
        # Determine rate bounds
        lb = []
        ub = []
        i = 0
        for rxn_coef in self.rxn_coef_opt:      # LB and UB are off for troe arrhenius parts
            rxnIdx = rxn_coef['rxnIdx']
            rate_bnds_val = mech.rate_bnds[rxnIdx]['value']
            rate_bnds_type = mech.rate_bnds[rxnIdx]['type']
            for T, P in zip(rxn_coef['T'], rxn_coef['P']):
                bnds = mech.rate_bnds[rxnIdx]['limits'](np.exp(rxn_rate_opt['x0'][i]))
                bnds = np.sort(np.log(bnds))  # operate on ln and scale
                scaled_bnds = np.sort(bnds/rxn_rate_opt['x0'][i])
                lb.append(scaled_bnds[0])
                ub.append(scaled_bnds[1])
                
                i += 1
        
        rxn_rate_opt['bnds'] = {'lower': np.array(lb), 'upper': np.array(ub)}

        # set mech to prior mech
        mech.coeffs = prior_coeffs
        mech.modify_reactions(mech.coeffs)

        return rxn_rate_opt

    def _update_gas(self): # TODO: What happens if a second optimization is run?
        mech = self.parent.mech
        reset_mech = mech.reset_mech
        coef_opt = self.coef_opt

        # delete any falloff coefficients with 5 indices
        delete_idx = []
        for i, idxDict in enumerate(coef_opt):
            rxnIdx, coefName = idxDict['rxnIdx'], idxDict['coefName']
            if coefName == 4:
                delete_idx.append(i)

        for i in delete_idx[::-1]:
            del coef_opt[i]

        generate_new_mech = False
        rxns_changed = []
        i = 0
        for rxn_coef_idx, rxn_coef in enumerate(self.rxn_coef_opt):      # TODO: RXN_COEF_OPT INCORRECT FOR CHANGING RXN TYPES
            rxnIdx = rxn_coef['rxnIdx']
            rxn = mech.gas.reaction(rxnIdx)
            if type(rxn) in [ct.ElementaryReaction, ct.ThreeBodyReaction]:
                continue    # arrhenius type equations don't need to be converted

            T, P, X = rxn_coef['T'], rxn_coef['P'], rxn_coef['X']
            M = mech.M(rxn, [T, P, X])
            rates = np.exp(self.rxn_rate_opt['x0'][i:i+len(T)])

            if type(rxn) is ct.FalloffReaction:
                lb = rxn_coef['coef_bnds']['lower']
                ub = rxn_coef['coef_bnds']['upper']
                if rxn.falloff.type == 'SRI':   
                    rxns_changed.append(rxn_coef['rxnIdx'])
                    rxn_coef['coef_x0'] = fit_Troe(rates, T, M, x0=rxn_coef['coef_x0'], coefNames=['A', 'T3', 'T1', 'T2'], 
                                                  bnds=[lb, ub], scipy_curvefit=False)

                # comment below, only for testing fit
                print(rxn_coef['coef_x0'])
                rxn_coef['coef_x0'] = fit_Troe(rates, T, M, x0=rxn_coef['coef_x0'], coefNames=['A', 'T3', 'T1', 'T2'], 
                                                  bnds=[lb, ub], HPL_LPL_defined=True, scipy_curvefit=True)
                
                mech.coeffs[rxnIdx]['falloff_type'] = 'Troe'
                mech.coeffs[rxnIdx]['falloff_parameters'] = rxn_coef['coef_x0'][6:]

            else:
                rxns_changed.append(rxn_coef['rxnIdx'])

                lb = rxn_coef['coef_bnds']['lower']
                ub = rxn_coef['coef_bnds']['upper']
                rxn_coef['coef_x0'] = fit_Troe(rates, T, M, x0=rxn_coef['coef_x0'], bnds=[lb, ub], HPL_LPL_defined=False, scipy_curvefit=False)

                rxn_coef['coefIdx'].extend(range(0, 4)) # extend to include falloff coefficients
                rxn_coef['coefName'].extend(range(0, 4)) # extend to include falloff coefficients
                rxn_coef['key'].extend([{'coeffs': 'falloff_parameters', 'coeffs_bnds': 'falloff_parameters'} for _ in range(0, 4)])

                # modify mech.coeffs from plog to falloff
                mech.coeffs[rxnIdx] = {'falloff_type': 'Troe', 'high_rate': {}, 'low_rate': {}, 'falloff_parameters': rxn_coef['coef_x0'][6:], 
                        'default_efficiency': 1.0, 'efficiencies': {}}

                n = 0
                for key in ['low_rate', 'high_rate']:
                    for coefName in default_arrhenius_coefNames:
                        rxn_coef['key'][n]['coeffs'] = key                          # change key value to match new reaction type
                        mech.coeffs[rxnIdx][key][coefName] = rxn_coef['coef_x0'][n] # updates new arrhenius values

                        n += 1
                
                rxn_coef['coef_bnds'] = set_bnds(mech, rxnIdx, rxn_coef['key'], rxn_coef['coefName'])

                # set reset_mech for new mechanism
                generate_new_mech = True
                reset_mech[rxnIdx]['rxnType'] = 'FalloffReaction'
                reset_mech[rxnIdx]['rxnCoeffs'] = mech.coeffs[rxnIdx]

            i += len(T)
        
        if generate_new_mech:
            # copy bounds to use in updated mechanism
            bnds = {'rate_bnds': [], 'coeffs_bnds': []}
            for rxnIdx in range(mech.gas.n_reactions):
                coeffs_bnds = {}
                for bndsKey, subRxn in mech.coeffs_bnds[rxnIdx].items():
                    coeffs_bnds[bndsKey] = {}
                    for coefIdx, (coefName, coefDict) in enumerate(subRxn.items()):
                        coeffs_bnds[bndsKey][coefName] = {'type': 'F', 'value': np.nan, 'opt': False}
                        if coefDict['opt'] and bndsKey != 'falloff_parameters':
                            coeffs_bnds[bndsKey][coefName] = {'type': coefDict['type'], 'value': coefDict['value'], 
                                                              'opt': coefDict['opt']}
                
                bnds['rate_bnds'].append({'value': mech.rate_bnds[rxnIdx]['value'], 'type': mech.rate_bnds[rxnIdx]['type'],
                                          'opt': mech.rate_bnds[rxnIdx]['opt']})
                bnds['coeffs_bnds'].append(coeffs_bnds)

            mech.set_mechanism(reset_mech, bnds=bnds)

        if len(rxns_changed) > 0:
            self._initialize_opt()


    def update(self, result, writeLog=True):
        # Update Hall of Fame
        if not self.HoF:
            self.HoF = result
        elif result['obj_fcn'] < self.HoF['obj_fcn']:
            self.HoF = result

        # Update log
        obj_fcn_str = f"{result['obj_fcn']:.3e}"
        replace_strs = [['e+', 'e'], ['e0', 'e'], ['e-0', 'e-']]
        for pair in replace_strs:
            obj_fcn_str = obj_fcn_str.replace(pair[0], pair[1])
        result['obj_fcn_str'] = obj_fcn_str

        if writeLog:
            if result['i'] > 999:
                obj_fcn_space = '\t\t'
            else:
                obj_fcn_space = '\t\t\t'
            
            if 'inf' in obj_fcn_str:
                obj_fcn_str_space = '\t\t\t\t\t'
            elif len(obj_fcn_str) < 6:
                obj_fcn_str_space = '\t\t\t\t'
            else:
                obj_fcn_str_space = '\t\t\t'

            self.parent.log.append('\t{:s} {:^5d}{:s}{:^s}{:s}{:^s}'.format(
                result['type'][0].upper(), result['i'], obj_fcn_space, obj_fcn_str, 
                obj_fcn_str_space, self.HoF['obj_fcn_str']), alert=False)

        self.parent.tree.update_coef_rate_from_opt(result['coef_opt'], result['x'])

        if timer() - self.last_plot_timer > self.time_between_plots:
            plot_start_time = timer()
            # if displayed shock isn't in shocks being optimized, calculate the new plot
            if result['ind_var'] is None and result['observable'] is None:
                self.parent.run_single()
            else:       # if displayed shock in list being optimized, show result
                self.parent.plot.signal.update_sim(result['ind_var'][:,0], result['observable'][:,0])
            self.parent.plot.opt.update(result['stat_plot'])

            # this keeps gui responsive, minimum time between plots = max time to plot*0.1
            current_time_to_plot = timer() - plot_start_time
            if current_time_to_plot*0.1 > self.time_between_plots:
                self.time_between_plots = current_time_to_plot*0.1

            self.last_plot_timer = timer()
    
    def on_worker_progress(self, perc_completed, time_left):
        self.parent.update_progress(perc_completed, time_left)
    
    def thread_complete(self): pass
    
    def on_worker_done(self, result):
        parent = self.parent
        parent.optimize_running = False
        if result is None or len(result) == 0: return
        
        # update mech to optimized one
        if 'local' in result:
            update_mech_coef_opt(parent.mech, self.coef_opt, result['local']['x'])
        else:
            update_mech_coef_opt(parent.mech, self.coef_opt, result['global']['x'])
        
        for opt_type, res in result.items():
            total_shock_eval = (res['nfev']+1)*len(self.shocks2run)
            message = res['message'][:1].lower() + res['message'][1:]
            
            parent.log.append('\n{:s} {:s}'.format(opt_type.capitalize(), message))
            parent.log.append('\telapsed time:\t{:.2f}'.format(res['time']), alert=False)
            parent.log.append('\tAvg Std Residual:\t{:.3e}'.format(res['fval']), alert=False)
            parent.log.append('\topt iters:\t\t{:.0f}'.format(res['nfev']+1), alert=False)
            parent.log.append('\tshock evals:\t{:.0f}'.format(total_shock_eval), alert=False)
            parent.log.append('\tsuccess:\t\t{:}'.format(res['success']), alert=False)
        
        parent.log.append('\n', alert=False)
        parent.save.chemkin_format(parent.mech.gas, parent.path_set.optimized_mech())
        parent.path_set.mech()  # update mech pulldown choices
        parent.tree._copy_expanded_tab_rates() # trigger copy rates

        parent.app.alert(parent, 5*1000) # make application alert for x millisec on completion

    def abort_workers(self):
        if hasattr(self, 'worker'):
            self.worker.signals.abort.emit()
            self.parent.abort = True
            if self.HoF:
                self.update(self.HoF, writeLog=False)
            # self.parent.update_progress(100, '00:00:00') # This turns off the progress bar
