#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created by g39677 at 5/18/21
# aka Buddhaha
# compatibility: if processing output files : CRESCENDO v 4.2

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
from matplotlib.lines import Line2D
import os
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px
import plotly.graph_objects as go


CM_TO_NM = 1e7
CM_M3_TO_M_M3 = 1e6

MARKERS = [".", ",", "v", "^", "<", ">", "1", "2","3", "4", "8", "p", "P", "*", "h", "H", "+", "x"]
MARKERS = 2 * MARKERS


def plot_exp_swelling_curves():
    data = '/home/g39677/DATA/DATA/crescendo_dat/exp_condition_pwr/pwr_exp_all.csv'
    df = pd.read_csv(data)
    df = df.rename(columns={'6': 'ref_id'})
    ### 6: ref_id
    cols = ['ref_id', 'reactor', 'exp_condition_type', 'he_rate(appm/dpa)',
       'dose(NRTdpa)', 'dose_rate', 'temp (C)', 'temp (K)', 'irr_time',
       'time_efpy', 'grade', 'position_group', 'position_name',
       'defect_type', 'size(nm)', 'size_var', 'size_min_max',
       'density(m^-3)', 'density_var', 'micro_swelling', 'macro_swelling']
    cols_s = ['he_rate(appm/dpa)', 'dose(NRTdpa)', 'dose_rate', 'temp (C)', 'irr_time',
       'time_efpy', 'grade', 'position_group','defect_type', 'size(nm)', 'size_var', 'size_min_max',
       'density(m^-3)', 'density_var', 'micro_swelling', 'macro_swelling']

    #df.dropna(subset=['dose(NRTdpa)','density(m^-3)', 'size(nm)'], inplace=True)
    #df = df[(df['ref_id'] != 'bosch543microstructure') & (df['ref_id'] != 'kuleshova2020microstructure') & (df['ref_id'] != 'edwards2003influence')]
    print('available references:', sorted(set(df['ref_id'])))
    #print(df.columns)
    #df['dose_rate']#
    df.astype({'dose_rate': 'float'})
    #df = df[(df['ref_id'] != 'bosch543microstructure') & (df['ref_id'] != 'kuleshova2020microstructure') &
    #        (df['ref_id'] != 'edwards2003influence')& (df['ref_id'] != 'font9_VanRenthergem') &
    #        (df['ref_id'] != 'garner1997irradiation') & (df['ref_id'] != 'EPRI_MRP50') &
    #        (df['ref_id'] != 'EPRI_MRP51') & (df['ref_id'] != 'font8_Panait') & (df['dose_rate'] < 1e-7)]

    # garner1997irradiation : EI847
    df = df[(df['ref_id'] != 'garner1997irradiation') & (df['dose_rate'] < 1e-7)] # (df['reactor'] != 'Tihange 1') &

    #print('after')
    #print(df.shape)
    #print(sorted(set(df['ref_id'])))
    df_c = df[df['defect_type'] == 'cavities']
    #print(sorted(set(df_c['ref_id'])))
    df_fl = df[df['defect_type'] == 'dislocation_loops']
    #print('########### cavs ##############')
    #print(df_c.describe(include='all'))
    #print('########### fl ##############')
    #print(df_fl.describe(include='all'))
    print('refs with cavity size > 5nm', df_c[df_c['size(nm)'] > 5].head(10))

    ## replace NaNs for plotting:
    #
    df['size_var'] = df['size_var'].fillna(0)
    df['density_var'] = df['density_var'].fillna(0)

    fig, axs = plt.subplots(3, 2,sharex=True, figsize=(10,12))

    #axs[0, 0].scatter(df_c['dose(NRTdpa)'], df_c['macro_swelling'], marker='x', label='experimental') # swelling
    axs[0, 0].set_title("swelling %")
    axs[0, 0].legend()

    axs[0, 1].set_title("fl density m^-2")
    axs[0, 1].legend()

    axs[0, 1].set_yscale('log')
    #axs[0, 1].set_ylim((1e14,1e20))

    axs[1, 0].set_title("vac size nm")
    axs[1, 1].set_title("fl size nm")

    axs[2, 0].set_title("vac dens m^-3")
    axs[2, 0].set_yscale('log')
    #axs[2, 0].set_ylim((1e20, 2e26))

    axs[2, 1].set_title("fl dens m^-3")
    axs[2, 1].set_yscale('log')
    #axs[2, 1].set_ylim((1e20, 2e26))

    ######################### PLOT ALL EXP DATA BY REF #########################
    i = 0

    x_axis = 'temp (C)'

    ### calibrate colors
    _color = 'dose(NRTdpa)'
    _min, _max = df[_color].min(), df[_color].max()
    #norma = mpl.colors.LogNorm(_min, _max)
    norma = mpl.colors.Normalize(_min, _max)

    # for label, df1 in df.groupby('exp_condition_type'):
    for label, df1 in df.groupby('ref_id'):
        print(label, df1.shape)
        # ax.scatter(df1['dose(NRTdpa)'], df1['macro_swelling'], marker=MARKERS[i], label=label)
        df_c = df1[df1['defect_type'] == 'cavities']
        df_fl = df1[df1['defect_type'] == 'dislocation_loops']

        a = axs[0, 0].scatter(df_c[x_axis], df_c['macro_swelling'], c=df_c[_color], marker=MARKERS[i],cmap='jet',norm=norma)#, label=label)  # swelling

        y01 = np.pi * df_fl['size(nm)'] * 1e-9 * df_fl['density(m^-3)']
        a = axs[0, 1].scatter(df_fl[x_axis], y01, c=df_fl[_color], marker=MARKERS[i],cmap='jet',norm=norma)#, label=label)  # dislo

        a = axs[1, 0].scatter(df_c[x_axis], df_c['size(nm)'], c=df_c[_color], marker=MARKERS[i],cmap='jet',norm=norma)  # r vac #### errorbar( ..

        a = axs[1, 1].scatter(df_fl[x_axis], df_fl['size(nm)'], c=df_fl[_color], marker=MARKERS[i],cmap='jet',norm=norma)  # r fl #### errorbar( ..

        a = axs[2, 0].scatter(df_c[x_axis], df_c['density(m^-3)'], c=df_c[_color], marker=MARKERS[i],cmap='jet',norm=norma) #### errorbar( ..

        a = axs[2, 1].scatter(df_fl[x_axis], df_fl['density(m^-3)'], c=df_fl[_color], marker=MARKERS[i],cmap='jet',norm=norma) #### errorbar( ..

        i += 1
    ###########################################################################

    #cb = plt.colorbar(a)
    #cb.set_label(_color)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(a, cax=cbar_ax)

    #plt.tight_layout()
    #plt.savefig('multi_scatterplot.png')
    plt.show()



def plot_batch_results(batches, labels):

    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(9.5, 13))

    colors=['k', 'r', 'g', 'b', 'c', 'm', 'y']
    _colors_used = []
    _labels_used = []
    #color = 'k'
    ig = 0
    for i, df_init in enumerate(batches):

        for key, df in df_init.groupby(['gc']):
            color = colors[ig]
            _colors_used.append(color) ## used for custom legend
            ig += 1
            lab = labels[i] + r"$G_{NRT}=$" + "{:.2e} dpa/s".format(key)
            _labels_used.append(lab) ## used for custom legend
            print('label', lab)
            print('color', color)


            print(df.columns)
            print(df.head(3))

            cols = ['pars', 'dpaNRT_x', 'ditot(cm^-3)', 'dvtot(cm^-3)', 'dimet(cm^-3)','dvmet(cm^-3)', 'rimet(cm)',
                    'rvmet(cm)', 'dV/V(%)', 'dVmet/V(%)', 'rho_t(cm^-2)', 'Rm(cm)', 'batch', 'he', 'Tk', 'Eb2i', 'Eb2v',
                    'Zi', 'riv', 'n2dmin', 'cs', 'emi']

            cols = ['pars', 'dpaNRT_x', 'ditot(cm^-3)', 'dvtot(cm^-3)', 'dimet(cm^-3)', 'dvmet(cm^-3)', 'rimet(cm)',
                    'rvmet(cm)', 'dV/V(%)', 'dVmet/V(%)', 'rho_t(cm^-2)', 'Tk','gc']
            df=df[cols]

            '''
            df['ditot(cm^-3)'] = np.log10(df['ditot(cm^-3)'])
            df['dvtot(cm^-3)'] = np.log10(df['dvtot(cm^-3)'])
            df['rho_t(cm^-2)'] = np.log10(df['rho_t(cm^-2)'])
            df['dimet(cm^-3)'] = np.log10(df['dimet(cm^-3)'])
            df['dvmet(cm^-3)'] = np.log10(df['dvmet(cm^-3)'])
        
            df['rimet(cm)'] = df['rimet(cm)'] * 1e7
            df['rvmet(cm)'] = df['rvmet(cm)'] * 1e7
            df = df.rename(columns={'rimet(cm)':'rimet(nm)', 'rvmet(cm)':'rvmet(nm)'})
            '''

            #x_axis = df['dpaNRT_x']
            x_axis = df['Tk'] - 273

            ###################### [0,0] ######################
            axs[0, 0].set_ylabel('void swelling (%)')
            axs[0, 0].plot(x_axis, df['dV/V(%)'], linestyle='--', c=color, label= lab +' model:total')
            #axs[0, 0].plot(x_axis, df['dVmet/V(%)'], linestyle='-', marker='o', c=color, label= lab +' model:>TEM limit')
            axs[0, 0].plot(x_axis, df['dVmet/V(%)'], linestyle='-', c=color, label= lab +' model: >TEM limit')

            ### some fancy shit
            axs[0, 0].hlines(0.35, 200, 400, linestyle='dotted')  # , label='TEM limit')
            axs[0, 0].text(220, 0.45, '0.35%', fontsize=14)
            axs[0, 0].legend()

            ###################### [0,1] ######################
            axs[0, 1].set_ylabel(r'density (m$^{-2}$)')
            axs[0, 1].plot(x_axis, df['rho_t(cm^-2)'] * 1e4, linestyle='--', c=color, label= lab +' model: rho_t(m^-2)')
            y01 = 2 * np.pi * df['rimet(cm)'] * CM_TO_NM * 1e-9 * df['dimet(cm^-3)'] * CM_M3_TO_M_M3
            #axs[0, 1].plot(x_axis, y01, linestyle='-', c=color, label= lab +' model: FL w TEM (m^-2)')
            axs[0, 1].set_yscale('log')
            axs[0, 1].legend()
            #y01 = np.pi * df_fl['size(nm)'] * 1e-9 * df_fl['density(m^-3)']
            #axs[0, 0].plot(x_axis, y01, linestyle='-', marker='o', c=color, label= lab +' model:>TEM limit')

            ###################### [1,0] ######################
            axs[1, 0].set_ylabel(r'size (nm)')
            axs[1, 0].plot(x_axis, 2 * df['rvmet(cm)'] * 1e7, linestyle='-', c=color, label= lab +' model: rvmet(nm)')
            ###################### [1,1] ######################
            axs[1, 1].set_ylabel(r'size (nm)')
            axs[1, 1].plot(x_axis, 2 * df['rimet(cm)'] * 1e7, linestyle='-', c=color, label= lab +' model: rimet(nm)')

            axs[1, 0].legend()
            ##axs[1, 1].legend() ### suppressed for custom legend

            ###################### [2,0] ######################
            axs[2, 0].set_ylabel(r'density (m$^{-3}$)')
            axs[2, 0].plot(x_axis, df['dvmet(cm^-3)'] * CM_M3_TO_M_M3, linestyle='-', c=color, label= lab +' model: dv met(m^-3)')
            axs[2, 0].plot(x_axis, df['dvtot(cm^-3)'] * CM_M3_TO_M_M3, linestyle='--', c=color, label= lab +' model: dv tot(m^-3)')
            ###################### [2,1] ######################
            axs[2, 1].set_ylabel(r'density (m$^{-3}$)')
            axs[2, 1].plot(x_axis, df['dimet(cm^-3)'] * CM_M3_TO_M_M3, linestyle='-', c=color, label= lab +' model: rv met(m^-3)')
            axs[2, 1].plot(x_axis, df['ditot(cm^-3)'] * CM_M3_TO_M_M3, linestyle='--', c=color, label= lab +' model: rv tot(m^-3)')

            axs[2, 0].set_yscale('log')
            axs[2, 1].set_yscale('log')
            axs[2, 0].legend()
            axs[2, 1].legend()
            axs[2, 0].set_xlabel(r'temperature ($^{\circ}$C)')
            axs[2, 1].set_xlabel(r'temperature ($^{\circ}$C)')




    custom_lines = [Line2D([0], [0], color='k', lw=1, linestyle='--'),
                    Line2D([0], [0], color='k', lw=1, linestyle='-')]

    custom_lines_rho = [Line2D([0], [0], color='k', lw=1, linestyle='--')]#,
                    #Line2D([0], [0], color='k', lw=1, linestyle='-'),
                    #    Line2D([0], [0], marker='o', color='w', label='Scatter',
                    #           markerfacecolor='k', markersize=10)]

    axs[0, 0].legend(custom_lines, ['total', 'visible (> 1nm)'])
    axs[0, 1].legend(custom_lines_rho, [r'total dislocation density $\rho_t$'])#, 'visible FL', 'visible FL exp'])

    axs[2, 0].legend(custom_lines, ['total', 'visible (> 1nm)'])
    axs[2, 1].legend(custom_lines, ['total', 'visible (> 1nm)'])

    custom_col_lines = []
    for c in _colors_used:
        custom_col_lines.append(Line2D([0], [0], color=c, lw=2, linestyle='-'))

    axs[1, 0].legend(custom_col_lines, _labels_used)





    #$$#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    ######################### EXP DATA #########################
    data = '/home/g39677/DATA/DATA/crescendo_dat/exp_condition_pwr/pwr_exp_all.csv'
    df = pd.read_csv(data)
    df = df.rename(columns={'6': 'ref_id'})
    ### 6: ref_id
    cols = ['ref_id', 'reactor', 'exp_condition_type', 'he_rate(appm/dpa)',
       'dose(NRTdpa)', 'dose_rate', 'temp (C)', 'temp (K)', 'irr_time',
       'time_efpy', 'grade', 'position_group', 'position_name',
       'defect_type', 'size(nm)', 'size_var', 'size_min_max',
       'density(m^-3)', 'density_var', 'micro_swelling', 'macro_swelling']
    cols_s = ['he_rate(appm/dpa)', 'dose(NRTdpa)', 'dose_rate', 'temp (C)', 'irr_time',
       'time_efpy', 'grade', 'position_group','defect_type', 'size(nm)', 'size_var', 'size_min_max',
       'density(m^-3)', 'density_var', 'micro_swelling', 'macro_swelling']

    #df.dropna(subset=['dose(NRTdpa)','density(m^-3)', 'size(nm)'], inplace=True)
    #df = df[(df['ref_id'] != 'bosch543microstructure') & (df['ref_id'] != 'kuleshova2020microstructure') & (df['ref_id'] != 'edwards2003influence')]
    print('available references:', sorted(set(df['ref_id'])))
    #print(df.columns)
    #df['dose_rate']#
    df.astype({'dose_rate': 'float'})
    #df = df[(df['ref_id'] != 'bosch543microstructure') & (df['ref_id'] != 'kuleshova2020microstructure') &
    #        (df['ref_id'] != 'edwards2003influence')& (df['ref_id'] != 'font9_VanRenthergem') &
    #        (df['ref_id'] != 'garner1997irradiation') & (df['ref_id'] != 'EPRI_MRP50') &
    #        (df['ref_id'] != 'EPRI_MRP51') & (df['ref_id'] != 'font8_Panait') & (df['dose_rate'] < 1e-7)]

    # garner1997irradiation : EI847
    df = df[(df['ref_id'] != 'garner1997irradiation') &
            (df['dose_rate'] < 1.1e-7) &
            (df['dose_rate'] > 0.9e-9) &
            (df['reactor'] != 'Tihange 1')]#
    #print('after')
    #print(df.shape)
    #print(sorted(set(df['ref_id'])))
    df_c = df[df['defect_type'] == 'cavities']
    #print(sorted(set(df_c['ref_id'])))
    df_fl = df[df['defect_type'] == 'dislocation_loops']
    #print('########### cavs ##############')
    #print(df_c.describe(include='all'))
    #print('########### fl ##############')
    #print(df_fl.describe(include='all'))
    print('refs with cavity size > 5nm', df_c[df_c['size(nm)'] > 5].head(10))

    ## replace NaNs for plotting:
    #
    df['size_var'] = df['size_var'].fillna(0)
    df['density_var'] = df['density_var'].fillna(0)


    #axs[0, 0].scatter(df_c['dose(NRTdpa)'], df_c['macro_swelling'], marker='x', label='experimental') # swelling
    axs[0, 0].set_title("void swelling")
    axs[0, 0].set_yscale('log')
    #axs[0, 0].legend()

    axs[0, 1].set_title("dislocation density")
    #axs[0, 1].legend()

    axs[0, 1].set_yscale('log')
    #axs[0, 1].set_ylim((1e14,1e20))

    axs[1, 0].set_title("voids")
    axs[1, 1].set_title("Frank loops")

    #axs[2, 0].set_title("vac dens m^-3")
    axs[2, 0].set_yscale('log')
    #axs[2, 0].set_ylim((1e20, 2e26))

    #axs[2, 1].set_title("fl dens m^-3")
    axs[2, 1].set_yscale('log')
    #axs[2, 1].set_ylim((1e20, 2e26))



    ######################### PLOT ALL EXP DATA BY REF #########################
    i = 0

    x_axis = 'temp (C)'

    ### calibrate colors
    #_color = 'dose(NRTdpa)'
    _color = 'dose_rate'
    _min, _max = df[_color].min(), df[_color].max()
    norma = mpl.colors.LogNorm(_min, _max)
    #norma = mpl.colors.Normalize(_min, _max)
    # for label, df1 in df.groupby('exp_condition_type'):
    for label, df1 in df.groupby('ref_id'):
        print(label, df1.shape)
        # ax.scatter(df1['dose(NRTdpa)'], df1['macro_swelling'], marker=MARKERS[i], label=label)
        df_c = df1[df1['defect_type'] == 'cavities']
        df_fl = df1[df1['defect_type'] == 'dislocation_loops']

        a = axs[0, 0].scatter(df_c[x_axis], df_c['macro_swelling'], c=df_c[_color], marker=MARKERS[i],cmap='jet',norm=norma)#, label=label)  # swelling

        y01 = np.pi * df_fl['size(nm)'] * 1e-9 * df_fl['density(m^-3)']
        #a = axs[0, 1].scatter(df_fl[x_axis], y01, c=df_fl[_color], marker=MARKERS[i],cmap='jet',norm=norma)#, label=label)  # dislo

        a = axs[1, 0].scatter(df_c[x_axis], df_c['size(nm)'], c=df_c[_color], marker=MARKERS[i],cmap='jet',norm=norma)  # r vac #### errorbar( ..

        a = axs[1, 1].scatter(df_fl[x_axis], df_fl['size(nm)'], c=df_fl[_color], marker=MARKERS[i],cmap='jet',norm=norma)  # r fl #### errorbar( ..

        a = axs[2, 0].scatter(df_c[x_axis], df_c['density(m^-3)'], c=df_c[_color], marker=MARKERS[i],cmap='jet',norm=norma) #### errorbar( ..

        a = axs[2, 1].scatter(df_fl[x_axis], df_fl['density(m^-3)'], c=df_fl[_color], marker=MARKERS[i],cmap='jet',norm=norma) #### errorbar( ..

        i += 1
    ###########################################################################
    axs[1,0].set_ylim((0,15))
    plt.tight_layout()
    cb = fig.colorbar(a, ax=axs[1,:], pad=0.05, shrink=0.4, location='bottom')#, loc='lower middle') # , ax=axs[[2, 0]] # , orientation="horizontal"
    cb.set_label('dose rate (dpa/s)')


    ## custom axis for colorbar
    #cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])


    #axs.cax.toggle_label(True)

    #fig.colorbar(a,ax=axs[[2,0]])#, shrink=0.2)
    #fig.colorbar(a, )
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)

    '''
    @TODO: adjust colorbar
    fig.subplots_adjust(right=0.5)
    cbar_ax = fig.add_axes([0.93, 0.05, 0.02, 0.9])
    cb = fig.colorbar(a, cax=cbar_ax, shrink=0.1)
    cb.set_label(_color)
    '''
    #$#$#$$$$$$$$#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    plt.savefig('flux_multi_scatterplot_vs_temp.png')
    plt.show()

    #fig = px.parallel_coordinates(df, color='Tk')#, dimensions=cols)
    #fig.show()
    #sys.exit()

def main():

    '''
    @TODOs
    1) change paths for dfs
    2)


    '''
    #plot_exp_swelling_curves()
    #p01 = '/home/g39677/DATA/DATA/crescendo_dat/2021_05_precipitation/final02/merged_out.csv'
    #p02 = '/home/g39677/DATA/DATA/crescendo_dat/2021_05_precipitation/final03/merged_out.csv'
    #p03 = '/home/g39677/DATA/DATA/crescendo_dat/2021_05_precipitation/final04/merged_out.csv'
    path = '/home/g39677/DATA/DATA/crescendo_dat/from_gaia/gaia_final_curves/flux_dep/merged_out.csv'
    df = pd.read_csv(path)


    print('set(df[gc])', set(df['gc']))
    print(df.head())
    print(df.columns)

    ## 80 dpa comparison
    df80 = df[(df['dpaNRT_x'] > 75) & (df['dpaNRT_x'] < 85)]
    #df80.hist(column='dpaNRT_x')
    ## 120 dpa comparison
    df120 = df[(df['dpaNRT_x'] > 115) & (df['dpaNRT_x'] < 125)]
    #df120.hist(column='dpaNRT_x')
    #plt.show()

    batches = [df80]#, df120]
    #batches = [df120]
    #colors = ['k', 'b']#,'r']
    labels = ['model at 80 dpa, ']#, '120 dpa']#, 'w He']
    #labels = ['model at 120 dpa, ']
    plot_batch_results(batches, labels)

    return


if __name__ == "__main__":
    # execute only if run as a script
    main()

