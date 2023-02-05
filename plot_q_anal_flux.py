#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created by g39677 at 4/10/21
# aka Buddhaha
# compatibility: if processing output files

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.optimize import curve_fit
'''
initial path:

short description:
'''

#SCRIPT_PATH = '/home/g39677/DATA/DATA/experimental_data/2021_01_sensitivity'
#sys.path.append(SCRIPT_PATH)
#from analyze_dataset import load_and_clean, plot_correlation_matrix ## custom funcion


MARKERS = ['o', "x", "^", ",", "v", "<", ">", ".", "1", "2", "3", "4", "8", "p", "P", "*", "h", "H", "+"]
MARKERS = MARKERS + MARKERS


EXP_LIMIT_SIZE = 0.8 # nm
EXP_LIMIT_DENS = 1e19 # m^-3


## all vars in [cm]
def calc_q(fl_dens, r_c, n_c, z=1):
    return z * fl_dens / (4 * np.pi * r_c * n_c)

def calc_q_from_RCsum(fl_dens, RCsum, z=1):
    return z * fl_dens / (4 * np.pi * RCsum * 1e-6)


def load_desrand_tot(path, step=-1):
    cols = ['t(s)', 'dpaNRT', 'ditot(cm^-3)', 'dvtot(cm^-3)', 'dimet(cm^-3)', 'dvmet(cm^-3)', 'ximet(cm^-3)', \
            'xvmet(cm^-3)', 'rimet(cm)', 'rvmet(cm)', 'r2imet(cm^2)', 'r2vmet(cm^2)', 'r3imet(cm^3)', 'r3vmet(cm^3)', \
            'DiCi', 'DvCv']
    df = pd.read_csv(path, skiprows=1, delim_whitespace=True, names=cols, index_col=False)
    if step == -1:
        step = df.shape[0] - 1
    return df.at[step, 'dvtot(cm^-3)'], df.at[step, 'ditot(cm^-3)']

'''
def plot_swelling_rate_vs_q(DATA,interesting_columns, fig, exp_condition='PWR'):

    df_pwr['Q'] = calc_q(df_pwr['fl_dens'], df_pwr['size(nm)'], df_pwr['density(m^-3)'])#,z=1.05)
    # print(df_pwr.describe)

    coloring = 'temp (C)'
    col = df_pwr[coloring]
    min_, max_ = col.min(), col.max()
    norma = mpl.colors.Normalize(min_, max_)

    i = 0
    for key, grp in df_pwr.groupby(['ref_id']):
        a = plt.scatter(grp['Q'], grp['macro_swelling'] / grp['dose(NRTdpa)'], marker=MARKERS[i], \
                    s=150, label=key,norm=norma,c=grp[coloring])
        #a = ax1.scatter(grp['dose(NRTdpa)'], grp['size(nm)'], marker=MARKERS[i], s=150, label=key, norm=norma,
        #                c=grp[coloring])
        i += 1

    cb = plt.colorbar(a)
    cb.set_label(coloring, fontsize=18)
    cb.ax.tick_params(labelsize=18)

    q_x = np.logspace(-2,3)
    plt.plot(q_x, q_x / ((1 + q_x) ** 2), 'r--')
    plt.text(5e-2, 2e-2, r'$\dot{V} \propto \frac{Q}{(1+Q)^2}$', color='r', fontsize=20)
    return df_pwr
'''

def main():
    #PATH = sys.argv[1]
    data ='merged_out.csv'
    df = pd.read_csv(data)#, names=COLS, delim_whitespace=True)
    print(df.head())
    print(df.columns)

    df['TC'] = df['Tk'] - 273
    df = df[df['dpaNRT_x'] > 70]
    df = df[df['TC'] > 180]
    # [u'pars', u'dpaNRT_x', u'ditot(cm^-3)', u'dvtot(cm^-3)', u'dimet(cm^-3)',
    #        u'dvmet(cm^-3)', u'rimet(cm)', u'rvmet(cm)', u'dV/V(%)', u'dVmet/V(%)',
    #        u'rho_t(cm^-2)', u'Rm(cm)', u'batch', u'Tk', u'gc', u'He', u't']

    #fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 8)) # rows, cols ##, sharex=True
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(9, 6))  # rows, cols ##, sharex=True
    #ax = fig.add_subplot(projection='3d')

    coloring = 'TC'
    col = df[coloring]
    min_, max_ = col.min(), col.max()
    norma = mpl.colors.Normalize(min_, max_)

    #df['QwMET'] = calc_q(2 * np.pi * df['rvmet(cm)'] * df['dvmet(cm^-3)'], df['rvmet(cm)'], df['dvmet(cm^-3)'], z=1)
    #df['QwMET'] = calc_q(df['rho_t(cm^-2)'], df['rvmet(cm)'], df['dvmet(cm^-3)'], z=1)
    ######axs[0].scatter(df['QwMET'], df['dVmet/V(%)'] / df['dpaNRT_x'], marker='x', label='Q w MET',norm=norma,c=col)

    #df['Qtot'] = calc_q_from_RCsum(df['rho_t(cm^-2)'], df['RCsum'], z=1)
    df['Qtot'] = calc_q(df['rho_t(cm^-2)'], df['RCsum'] / df['Csum'], df['dvtot(cm^-3)'])
    b = axs.scatter(df['Qtot'], df['dV/V(%)'] / df['dpaNRT_x'], marker='x', label='simulation',norm=norma,c=col)

    q_x = np.logspace(-8,4)
    ######axs[0].plot(q_x, q_x / ((1 + q_x) ** 2), 'r--')
    axs.plot(q_x, q_x / ((1 + q_x) ** 2), 'r--')
    #axs.text(5e-2, 2e-2, r'$\dot{V} \propto \frac{Q}{(1+Q)^2}$', color='r', fontsize=20)
    ######axs[0].vlines(1, 0, 3)
    axs.vlines(1, 0, 3)

    cb = plt.colorbar(b)
    #cb = plt.colorbar(b)

    ######axs[0].legend()
    axs.legend()
    ######axs[0].set_xscale('log')
    ######axs[0].set_yscale('log')
    ####
    ######axs[0].set_xlabel('Q', fontsize=22)
    axs.set_ylabel(r'void swelling rate (% / dpa)', fontsize=22)
    axs.set_xlabel('Q', fontsize=22)
    ####
    ######axs[0].set_xlim((1e-8, 1e6))
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xlim((1e-8, 1e3))
    axs.set_ylim((1e-4, 3))
    ######axs[0].text(1e1, 0.5, r'$\dot{V} \propto \frac{Q}{(1+Q)^2}$', color='r', fontsize=20)
    axs.text(3, 0.5, r'$\dot{V} \propto \frac{Q}{(1+Q)^2}$', color='r', fontsize=20)

    cb.set_label(r'T($^{\circ}$C)', fontsize=18)
    cb.ax.tick_params(labelsize=18)
    ######axs[0].tick_params(labelsize=18)
    axs.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig('sim_q_anal.png')
    plt.show()



    return


if __name__ == "__main__":
    # execute only if run as a script
    main()