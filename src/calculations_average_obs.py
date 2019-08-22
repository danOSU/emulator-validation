#!/usr/bin/env python3

import numpy as np
import h5py
import sys, os, glob
import pickle
import pandas as pd
import logging
from configurations import * 


## Data structure used to save event information
# fully specify numeric data type
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'

# species (name, ID) for identified particle observables
species = [
    ('pion', 211),
    ('kaon', 321),
    ('proton', 2212),
    ('Lambda', 3122),
    ('Sigma0', 3212),
    ('Xi', 3312),
    ('Omega', 3334),
    ('phi', 333),
]
pi_K_p = [
    ('pion', 211),
    ('kaon', 321),
    ('proton', 2212),
]
lambda_omega_xi = [
    ('Lambda', 3122),
    ('Omega', 3334),
    ('Xi', 3312),
]


NpT = 10
Nharmonic = 8
Nharmonic_diff = 5

# results "array" (one element) output format for single event
# to be overwritten for each event
result_dtype=[
('initial_entropy', float_t, 1),
('impact_parameter', float_t, 1),
('npart', float_t, 1),
('ALICE',
    [
        # 1) dNch/deta, eta[-0.5, 0.5], charged
        ('dNch_deta', float_t, 1),
        # 2) dET/deta, eta[-0.6, 0.6]
        ('dET_deta', float_t, 1),
        # 3.1) The Tmunu observables, eta[-0.6, 0.6]
        ('Tmunu', float_t, 10),
        # 3.2) The Tmunu observables, eta[-0.5, 0.5], charged
        ('Tmunu_chg', float_t, 10),
        # 4.1) identified particle yield
        ('dN_dy',   [(name, float_t, 1) for (name,_) in species], 1),
        # 4.2) identified particle <pT>
        ('mean_pT', [(name, float_t, 1) for (name,_) in species], 1),
        # 5.1) pT fluct, pT[0.15, 2], eta[-0.8, 0.8], charged
        ('pT_fluct_chg', [  ('N', int_t, 1),
                            ('sum_pT', float_t, 1),
                            ('sum_pT2', float_t, 1)], 1),
        # 5.2) pT fluct, pT[0.15, 2], eta[-0.8, 0.8], pi, K, p
        ('pT_fluct_pid', [  (name, [    ('N', int_t, 1),
                                        ('sum_pT', float_t, 1),
                                        ('sum_pT2', float_t, 1)], 1 )
                              for (name,_) in pi_K_p    ], 1),
        # 6) Q vector, pT[0.2, 5.0], eta [-0.8, 0.8], charged
        ('flow', [  ('N', int_t, 1),
                    ('Qn', complex_t, Nharmonic)], 1),
        # 7) Q vector, diff-flow eta[-0.8, 0.8], pi, K, p
        # It uses #6 as its reference Q vector
        ('d_flow_chg', [('N', int_t, NpT),
                        ('Qn', complex_t, [NpT, Nharmonic_diff])], 1),
        ('d_flow_pid', [(name, [('N', int_t, NpT),
                                ('Qn', complex_t, [NpT, Nharmonic_diff])], 1)
                        for (name,_) in pi_K_p  ], 1),
    ], 1)
]



## processed observables format
bayes_dtype=[    (s, 
                  [(obs, [("mean",float_t,len(cent_list)),
                          ("err",float_t,len(cent_list))]) \
                    for obs, cent_list in obs_cent_list[s].items() ],
                  number_of_models_per_run                                                                     
                 ) \
                 for s in system_strs
            ]


def list2array(func):
        def func_wrapper(x, w):
                try:
                        x = np.array(x)
                        w = np.array(w)
                except:
                        raise ValueError("cannot interpret input as numpy array...")
                return func(x, w)
        return func_wrapper


def weighted_mean_std(x, w=None):
        if w is None:
                Neff = x.size
                mean = np.mean(x)
                std = np.std(x)/np.sqrt(Neff-1.+1e-9)
        else:
                Neff = np.sum(w)**2/np.sum(w**2)
                mean = np.average(x, weights=w)
                std = ( np.average((x-mean)**2, weights=w)/(Neff-1.+1e-9) ) **.5
        return mean, std


def calculate_dNdeta(ds, exp, cen, idf):
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = np.zeros_like(cenM)
        obs_err = np.zeros_like(cenM)
        for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                nh = np.max([nh, nl+1])
                obs[i], obs_err[i] = weighted_mean_std( ds[exp]['dNch_deta'][nl:nh, idf] )
        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}


def calculate_dETdeta(ds, exp, cen, idf):
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = np.zeros_like(cenM)
        obs_err = np.zeros_like(cenM)
        for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                obs[i], obs_err[i] = weighted_mean_std(ds[exp]['dET_deta'][nl:nh, idf])
        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}

def calculate_dNdy(ds, exp, cen, idf):
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = {s: np.zeros_like(cenM) for (s, _) in species}
        obs_err = {s: np.zeros_like(cenM) for (s, _) in species}
        for (s, _) in species:
                for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                        obs[s][i], obs_err[s][i] = weighted_mean_std(ds[exp]['dN_dy'][s][nl:nh, idf])
        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}

def calculate_mean_pT(ds, exp, cen, idf):
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = {s: np.zeros_like(cenM) for (s, _) in species}
        obs_err = {s: np.zeros_like(cenM) for (s, _) in species}
        for (s, _) in species:
                for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                        obs[s][i], obs_err[s][i] = weighted_mean_std(ds[exp]['mean_pT'][s][nl:nh, idf])
        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}

def calculate_mean_pT_fluct(ds, exp, cen, idf):

        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = np.zeros_like(cenM)
        obs_err = np.zeros_like(cenM)
        for (s, _) in species:

                for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                        N = ds[exp]['pT_fluct_chg']['N'][nl:nh, idf]
                        sum_pT = ds[exp]['pT_fluct_chg']['sum_pT'][nl:nh, idf]
                        sum_pTsq = ds[exp]['pT_fluct_chg']['sum_pT2'][nl:nh, idf]

                        Npairs = .5*N*(N - 1)
                        M = sum_pT.sum() / N.sum()

                        # This is equivalent to the sum over pairs in Eq. (2).  It may be derived
                        # by using that, in general,
                        #
                        #   \sum_{i,j>i} a_i a_j = 1/2 [(\sum_{i} a_i)^2 - \sum_{i} a_i^2].
                        #
                        # That is, the sum over pairs (a_i, a_j) may be re-expressed in terms of
                        # the sum of a_i and sum of squares a_i^2.  Applying this to Eq. (2) and
                        # collecting terms yields the following expression.
                        x = (.5*(sum_pT**2 - sum_pTsq) - M*(N - 1)*sum_pT + M**2*Npairs)/Npairs
                        meanC, stdC = weighted_mean_std(x, Npairs)
                        obs[i] = np.sqrt(meanC)/M
                        obs_err[i] = stdC*.5/np.sqrt(meanC)/M

        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}


def calculate_vn(ds, exp, cen, idf):
        @list2array
        def obs_and_err(qn, m):
                w = m*(m-1.) # is this P_{M,2} in notation of Jonah's Thesis
                if w.sum() == 0.:
                    return 0., 0.
                cn2 = (np.abs(qn)**2 - m)/w # is this is <2> in Jonah's thesis (p.27)
                avg_cn2, std_avg_cn2 = weighted_mean_std(cn2, w)
                vn = np.sqrt(avg_cn2)
                vn_err = std_avg_cn2/2./vn
                return vn, vn_err
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)

        obs = np.zeros([len(cenM), Nharmonic])
        obs_err = np.zeros([len(cenM), Nharmonic])

        for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                M = ds[exp]['flow']['N'][nl:nh, idf]
                for n in range(Nharmonic):
                        Q = ds[exp]['flow']['Qn'][nl:nh, idf, n]
                        obs[i,n], obs_err[i,n] = obs_and_err(Q, M)
        return {'Name': 'vn', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}


def calculate_diff_vn(ds, exp, cenbins, pTbins, idf, pid='chg'):
        Ne = len(ds)
        pTbins = np.array(pTbins)
        cenbins = np.array(cenbins)
        cenM = np.mean(cenbins, axis=1)
        pTM = np.mean(pTbins, axis=1)
        Cindex = (cenbins/100.*Ne).astype(int)

        if pid == 'chg':
                obs = 'd_flow_chg'
                data = ds[exp][:,idf][obs]
        else:
                obs = 'd_flow_pid'
                data = ds[exp][:,idf][obs][s]

        # need soft flow within the same centrality bin first
        # only needs Ncen x [v2, v3]
        vnref = calculate_vn(ds, exp, cenbins, idf)

        # calculate hard vn
        vn = np.zeros([len(cenM), len(pTM), Nharmonic_diff])
        vn_err = np.zeros([len(cenM), len(pTM), Nharmonic_diff])
        for i, (nl, nh) in enumerate(Cindex):
                for j, (pl, ph) in enumerate(pTbins):
                        for n in range(Nharmonic_diff):
                                w = data['N'][nl:nh, j] * ds[exp]['flow']['N'][nl:nh, idf]
                                dn2 = (data['Qn'][nl:nh,j,n].conjugate() * ds[exp]['flow']['Qn'][nl:nh, idf, n]).real / w
                                avg_dn2, std_avg_dn2 = weighted_mean_std(dn2, w)
                                vn[i, j, n] = avg_dn2/vnref['obs'][i,n]
                                vn_err[i, j, n] = std_avg_dn2/vnref['obs'][i,n]
        return {'Name': 'vn2', 'cenM': cenM, 'pTM' : pTM,
                        'obs': vn, 'err': vn_err}



def load_and_print_single_event(filename):
    data = np.fromfile(filename, dtype=result_dtype)
    n_items = len(result_dtype)

    if (n_items > 0):
        for n, item in enumerate(structure):
            tmp_struct=structure[n]
            # If the item has substructure, recurse on it
            if (not isinstance(tmp_struct[1], str)) and (isinstance(tmp_struct[1], Iterable)):
                print(tmp_struct[0])
                print_data_structure(data[tmp_struct[0]],tmp_struct[1])
            # If no substructure, just output the result
            else:
                print(tmp_struct[0],data[tmp_struct[0]])





def load_and_compute_single_design(inputfile, system):
    print('Now analysis for system: ', system)
    entry = np.zeros(1, dtype=np.dtype(bayes_dtype))
    res_unsort = []
    for _ in range(number_of_events_per_design):
        try:
            dum = np.fromfile('{}/{}.dat'.format(inputfile, _), dtype=result_dtype)
            if len(dum) != 0:
                res_unsort.append(dum)
        except:
            continue

    print('read in total events: ', len(res_unsort))

    for idf in [0,3]:
        res = np.array(sorted(res_unsort, key=lambda x: x['ALICE'][idf]['dNch_deta'], reverse=True))

        # dNdeta
        tmp_obs='dNch_deta'
        cenb=np.array(obs_cent_list[system][tmp_obs])
        info = calculate_dNdeta(res, 'ALICE', cenb, idf)
        entry[system][tmp_obs]['mean'][:, idf] = info['obs']
        entry[system][tmp_obs]['err'][:,idf] = info['err']

        # dETdeta
        tmp_obs='dET_deta'
        cenb=np.array(obs_cent_list[system][tmp_obs])
        info = calculate_dETdeta(res, 'ALICE', cenb, idf)
        entry[system][tmp_obs]['mean'][:,idf] = info['obs']
        entry[system][tmp_obs]['err'][:,idf] = info['err']

        # dN(pid)/dy
        for s in ['pion','kaon','proton','Lambda', 'Omega','Xi']:
            cenb=np.array(obs_cent_list[system]['dN_dy_'+s])
            info = calculate_dNdy(res, 'ALICE', cenb, idf)
            entry[system]['dN_dy_'+s]['mean'][:,idf] = info['obs'][s]
            entry[system]['dN_dy_'+s]['err'][:,idf] = info['err'][s]


        # mean-pT
        for s in ['pion','kaon','proton']:
            cenb=np.array(obs_cent_list[system]['mean_pT_'+s])
            info = calculate_mean_pT(res, 'ALICE', cenb, idf)
            entry[system]['mean_pT_'+s]['mean'][:,idf] = info['obs'][s]
            entry[system]['dN_dy_'+s]['err'][:,idf] = info['err'][s]


        # mean-pT-fluct
        tmp_obs='pT_fluct'
        cenb=np.array(obs_cent_list[system][tmp_obs])
        info = calculate_mean_pT_fluct(res, 'ALICE', cenb, idf)
        entry[system][tmp_obs]['mean'][:,idf] = info['obs']
        entry[system][tmp_obs]['err'][:,idf] = info['err']
        
        # vn
        for n in range(2,5):
            tmp_obs='v'+str(n)+'2'
            cenb=np.array(obs_cent_list[system][tmp_obs])
            info = calculate_vn(res, 'ALICE', cenb, idf)
            entry[system][tmp_obs]['mean'][:,idf] = info['obs'][:, n-1]
            entry[system][tmp_obs]['err'][:,idf] = info['err'][:, n-1]
    return entry



if __name__ == '__main__':
    system='Pb-Pb-2760'

    #### reading in main design output
    results = []
    for i in range(n_design_pts_main):
        filename = f_events_main + '/{:d}'.format(i)
        results.append(load_and_compute_single_design(filename, system)[0])
    results = np.array(results)
    results.tofile(f_obs_main)



    #### reading in validation design output
    '''
    results = []
    for i in range(n_design_pts_validation):
        filename = f_events_validation + '/{:d}'.format(i)
        results.append(load_and_compute_single_design(filename, system)[0])
    results = np.array(results)
    results.tofile(f_obs_validation)
    '''

    print('finishing :)')
