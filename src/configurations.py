#!/usr/bin/env python3
import os, logging
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from bins_and_cuts import obs_cent_list, obs_range_list


# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'
#fix the random seed for cross validation, that sets are deleted consistently
np.random.seed(1)
# Work, Design, and Exp directories
workdir = Path(os.getenv('WORKDIR', '.'))
design_dir =  str(workdir/'production_designs/500pts')
dir_obs_exp = "HIC_experimental_data"

####################################
### USEFUL LABELS / DICTIONARIES ###
####################################
#only using data from these experimental collabs
expt_for_system = { 'Au-Au-200' : 'STAR',
                    'Pb-Pb-2760' : 'ALICE',
                    'Pb-Pb-5020' : 'ALICE',
                    'Xe-Xe-5440' : 'ALICE',
                    }

#for STAR we have measurements of pi+ dN/dy, k+ dN/dy etc... so we need to scale them by 2 after reading in
STAR_id_yields = {
            'dN_dy_pion' : 'dN_dy_pion_+',
            'dN_dy_kaon' : 'dN_dy_kaon_+',
            'dN_dy_proton' : 'dN_dy_proton_+',
}

idf_label = {
            0 : 'Grad',
            1 : 'Chapman-Enskog R.T.A',
            2 : 'Pratt-McNelis',
            3 : 'Pratt-Bernhard'
            }
idf_label_short = {
            0 : 'Grad',
            1 : 'C.E.',
            2 : 'P.M.',
            3 : 'P.B.'
            }

####################################
### SWITCHES AND OPTIONS !!!!!!! ###
####################################

#how many versions of the model are run, for instance
# 4 versions of delta-f with SMASH and a fifth model with UrQMD totals 5
number_of_models_per_run = 4

# the choice of viscous correction. 0 : 14 Moment, 1 : C.E. RTA, 2 : McNelis, 3 : Bernhard
idf = 0
print("Using idf = " + str(idf) + " : " + idf_label[idf])

#the Collision systems
systems = [
        ('Pb', 'Pb', 2760),
        ('Au', 'Au', 200),
        #('Pb', 'Pb', 5020),
        #('Xe', 'Xe', 5440)
        ]
system_strs = ['{:s}-{:s}-{:d}'.format(*s) for s in systems]
num_systems = len(system_strs)

#these are problematic points for Pb Pb 2760 run with 500 design points
nan_sets_by_deltaf = {
                        0 : set([334, 341, 377, 429, 447, 483]),
                        1 : set([285, 334, 341, 447, 483, 495]),
                        2 : set([209, 280, 322, 334, 341, 412, 421, 424, 429, 432, 446, 447, 453, 468, 483, 495]),
                        3 : set([60, 232, 280, 285, 322, 324, 341, 377, 432, 447, 464, 468, 482, 483, 485, 495])
                    }
nan_design_pts_set = nan_sets_by_deltaf[idf]

#nan_design_pts_set = set([60, 285, 322, 324, 341, 377, 432, 447, 464, 468, 482, 483, 495])
unfinished_events_design_pts_set = set([289, 324, 326, 459, 462, 242, 406, 440, 123])
strange_features_design_pts_set = set([289, 324, 440, 459, 462])

delete_design_pts_set = nan_design_pts_set.union(
                            unfinished_events_design_pts_set.union(
                                        strange_features_design_pts_set
                                        )
                                    )

delete_design_pts_validation_set = [10, 68, 93] # idf 0

class systems_setting(dict):
    def __init__(self, A, B, sqrts):
        super().__setitem__("proj", A)
        super().__setitem__("targ", B)
        super().__setitem__("sqrts", sqrts)
        sysdir = "/design_pts_{:s}_{:s}_{:d}_production".format(A, B, sqrts)
        super().__setitem__("main_design_file",
            design_dir+sysdir+'/design_points_main_{:s}{:s}-{:d}.dat'.format(A, B, sqrts)
            )
        super().__setitem__("main_range_file",
            design_dir+sysdir+'/design_ranges_main_{:s}{:s}-{:d}.dat'.format(A, B, sqrts)
            )
        super().__setitem__("validation_design_file",
            design_dir+sysdir+'/design_points_validation_{:s}{:s}-{:d}.dat'.format(A, B, sqrts)
            )
        super().__setitem__("validation_range_file",
            design_dir+sysdir+'//design_ranges_validation_{:s}{:s}-{:d}.dat'.format(A, B, sqrts)
            )
        with open(design_dir+sysdir+'/design_labels_{:s}{:s}-{:d}.dat'.format(A, B, sqrts), 'r') as f:
            labels = [r""+line[:-1] for line in f]
        super().__setitem__("labels", labels)
    def __setitem__(self, key, value):
        if key == 'run_id':
            super().__setitem__("main_events_dir",
                str(workdir/'model_calculations/{:s}/Events/main/'.format(value))
                )
            super().__setitem__("validation_events_dir",
                str(workdir/'model_calculations/{:s}/Events/validation/'.format(value))
                )
            super().__setitem__("main_obs_file",
                str(workdir/'model_calculations/{:s}/Obs/main.dat'.format(value))
                )
            super().__setitem__("validation_obs_file",
                str(workdir/'model_calculations/{:s}/Obs/validation.dat'.format(value))
                )
        else:
            super().__setitem__(key, value)

SystemsInfo = {"{:s}-{:s}-{:d}".format(*s): systems_setting(*s) \
                for s in systems
               }

if 'Pb-Pb-2760' in system_strs:
    SystemsInfo["Pb-Pb-2760"]["run_id"] = "production_500pts_Pb_Pb_2760"
    SystemsInfo["Pb-Pb-2760"]["n_design"] = 500
    SystemsInfo["Pb-Pb-2760"]["n_validation"] = 100
    SystemsInfo["Pb-Pb-2760"]["design_remove_idx"]=list(delete_design_pts_set)
    SystemsInfo["Pb-Pb-2760"]["npc"]=10
    SystemsInfo["Pb-Pb-2760"]["MAP_obs_file"]=str(workdir/'model_calculations/MAP') + '/' + idf_label_short[idf] + '/Obs/obs_Pb-Pb-2760.dat'


if 'Au-Au-200' in system_strs:
    SystemsInfo["Au-Au-200"]["run_id"] = "production_500pts_Au_Au_200"
    SystemsInfo["Au-Au-200"]["n_design"] = 500
    SystemsInfo["Au-Au-200"]["n_validation"] = 100
    SystemsInfo["Au-Au-200"]["design_remove_idx"]=list(delete_design_pts_set)
    SystemsInfo["Au-Au-200"]["npc"] = 6
    SystemsInfo["Au-Au-200"]["MAP_obs_file"]=str(workdir/'model_calculations/MAP') + '/' + idf_label_short[idf] + '/Obs/obs_Au-Au-200.dat'


###############################################################################
############### BAYES #########################################################

# if True : perform emulator validation
# if False : use experimental data for parameter estimation
validation = False
#if true, we will validate emulator against points in the training set
pseudovalidation = False
#if true, we will omit 20% of the training design when training emulator
crossvalidation = False

fixed_validation_pt=0

if validation:
    print("Performing emulator validation type ...")
    if pseudovalidation:
        print("... pseudo-validation")
        pass
    elif crossvalidation:
        print("... cross-validation")
        cross_validation_pts = np.random.choice(n_design_pts_main,
                                                n_design_pts_main // 5,
                                                replace = False)
        delete_design_pts_set = cross_validation_pts #omit these points from training
    else:
        validation_pt = fixed_validation_pt
        print("... independent-validation, using validation_pt = " + str(validation_pt))

#if this switch is True, all experimental errors will be set to zero
set_exp_error_to_zero = False

# if this switch is True, then when performing MCMC each experimental error
# will be multiplied by the corresponding factor
change_exp_error = True
change_exp_error_vals = {
                        'Au-Au-200': {},

                        'Pb-Pb-2760' : {
                                        'dN_dy_proton' : 1.e-1,
                                        'mean_pT_proton' : 1.e-1
                                        }

}

#if this switch is turned on, some parameters will be fixed
#to certain values in the bayesian analysis. see bayes_mcmc.py
hold_parameters = False
# hold are pairs of parameter (index, value)
# count the index correctly when have multiple systems!!!
# e.g [(1, 10.5), (5, 0.3)] will hold parameter[1] at 10.5, and parameter[5] at 0.3
#hold_parameters_set = [(7, 0.0), (8, 0.154), (9, 0.0), (15, 0.0), (16, 5.0)] #these should hold the parameters to Jonah's prior for LHC+RHIC
#hold_parameters_set = [(6, 0.0), (7, 0.154), (8, 0.0), (14, 0.0), (15, 5.0)] #these should hold the parameters to Jonah's prior for LHC only
#hold_parameters_set = [(16, 8.0)] #this will fix the shear relaxation time factor for LHC+RHIC
hold_parameters_set = [(17, 0.155)] #this will fix the T_sw for LHC+RHIC
if hold_parameters:
    print("Warning : holding parameters to fixed values : ")
    print(hold_parameters_set)

#if this switch is turned on, the emulator will be trained on the values of
# eta/s (T_i) and zeta/s (T_i), where T_i are a grid of temperatures, rather
# than the parameters such as slope, width, etc...
do_transform_design = True

#if this switch is turned on, the emulator will be trained on log(1 + dY_dx)
#where dY_dx includes dET_deta, dNch_deta, dN_dy_pion, etc...
transform_multiplicities = False

#this switches on/off parameterized experimental covariance btw. centrality bins and groups
assume_corr_exp_error = False
cent_corr_length = 0.5 #this is the correlation length between centrality bins

bayes_dtype = [    (s,
                  [(obs, [("mean",float_t,len(cent_list)),
                          ("err",float_t,len(cent_list))]) \
                    for obs, cent_list in obs_cent_list[s].items() ],
                  number_of_models_per_run
                 ) \
                 for s in system_strs
            ]

# The active ones used in Bayes analysis (MCMC)
active_obs_list = {
   sys: list(obs_cent_list[sys].keys()) for sys in system_strs
}

#try exluding PHENIX dN/dy proton from fit
for s in system_strs:
    if s == 'Au-Au-200':
        active_obs_list[s].remove('dN_dy_proton')
        active_obs_list[s].remove('mean_pT_proton')

    if s == 'Pb-Pb-2760':
        active_obs_list[s].remove('dN_dy_Lambda')
        active_obs_list[s].remove('dN_dy_Omega')
        active_obs_list[s].remove('dN_dy_Xi')

print("The active observable list for calibration: " + str(active_obs_list))

def zeta_over_s(T, zmax, T0, width, asym):
    DeltaT = T - T0
    sign = 1 if DeltaT>0 else -1
    x = DeltaT/(width*(1.+asym*sign))
    return zmax/(1.+x**2)
zeta_over_s = np.vectorize(zeta_over_s)

def eta_over_s(T, T_k, alow, ahigh, etas_k):
    if T < T_k:
        y = etas_k + alow*(T-T_k)
    else:
        y = etas_k + ahigh*(T-T_k)
    if y > 0:
        return y
    else:
        return 0.
eta_over_s = np.vectorize(eta_over_s)

def taupi(T, T_k, alow, ahigh, etas_k, bpi):
    return bpi*eta_over_s(T, T_k, alow, ahigh, etas_k)/T
taupi = np.vectorize(taupi)

def tau_fs(e, tau_R, alpha):
    #e stands for e_initial / e_R, dimensionless
    return tau_R * (e**alpha)

# load design for other module
def load_design(system_str, pset='main'): # or validation
    design_file = SystemsInfo[system_str]["main_design_file"] if pset == 'main' \
                  else SystemsInfo[system_str]["validation_design_file"]
    range_file = SystemsInfo[system_str]["main_range_file"] if pset == 'main' \
                  else SystemsInfo[system_str]["validation_range_file"]
    print("Loading {:s} points from {:s}".format(pset, design_file) )
    print("Loading {:s} ranges from {:s}".format(pset, range_file) )
    labels = SystemsInfo[system_str]["labels"]
    # design
    design = pd.read_csv(design_file)
    design = design.drop("idx", axis=1)
    print("Summary of design : ")
    design.describe()
    design_range = pd.read_csv(range_file)
    design_max = design_range['max'].values
    design_min = design_range['min'].values
    return design, design_min, design_max, labels

# A specially transformed design for the emulators
# 0    1        2       3             4
# norm trento_p sigma_k nucleon_width dmin3
#
# 5     6     7
# tau_R alpha eta_over_s_T_kink_in_GeV
#
# 8                             9                              10
# eta_over_s_low_T_slope_in_GeV eta_over_s_high_T_slope_in_GeV eta_over_s_at_kink,
#
# 11              12                        13
# zeta_over_s_max zeta_over_s_T_peak_in_GeV zeta_over_s_width_in_GeV
#
# 14                       15                      16
# zeta_over_s_lambda_asymm shear_relax_time_factor Tswitch

#right now this depends on the ordering of parameters
#we should write a version instead that uses labels in case ordering changes

def transform_design(X):
    #pop out the viscous parameters
    indices = [0, 1, 2, 3, 4, 5, 6, 15, 16]
    new_design_X = X[:, indices]

    #now append the values of eta/s and zeta/s at various temperatures
    num_T = 10
    Temperature_grid = np.linspace(0.135, 0.4, num_T)
    eta_vals = []
    zeta_vals = []
    for pt, T in enumerate(Temperature_grid):
        eta_vals.append( eta_over_s(T, X[:, 7], X[:, 8], X[:, 9], X[:, 10]) )
    for pt, T in enumerate(Temperature_grid):
        zeta_vals.append( zeta_over_s(T, X[:, 11], X[:, 12], X[:, 13], X[:, 14]) )

    eta_vals = np.array(eta_vals).T
    zeta_vals = np.array(zeta_vals).T

    new_design_X = np.concatenate( (new_design_X, eta_vals), axis=1)
    new_design_X = np.concatenate( (new_design_X, zeta_vals), axis=1)
    return new_design_X

def prepare_emu_design(system_str):
    design, design_max, design_min, labels = \
                    load_design(system_str=system_str, pset='main')

    #transformation of design for viscosities
    if do_transform_design:
        print("Note : Transforming design of viscosities")
        #replace this with function that transforms based on labels, not indices
        design = transform_design(design.values)
    else :
        design = design.values

    design_max = np.max(design, axis=0)
    design_min = np.min(design, axis=0)
    return design, design_max, design_min, labels
