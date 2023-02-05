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
import os
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px
import plotly.graph_objects as go
'''
initial path:

short description:
'''

def prepare_df_for_surfplot(name = 'merged_out.csv'):
    densrad_data='densrad_out.csv'
    swelling_data='swelling_out.csv'
    dislo_data='dislo_tot_out.csv'
    q_data = 'q_file.csv'

    df1 = pd.read_csv(densrad_data, sep=';')
    df2 = pd.read_csv(swelling_data, sep=';')
    df3 = pd.read_csv(dislo_data, sep=';')
    df4 = pd.read_csv(q_data, sep=';')

    df = pd.merge(pd.merge(df1, df2, on='pars', copy=False), df3, on='pars', copy=False)
    df = pd.merge(df, df4, on='pars', copy=False)
    print(df.columns)
    #cols = ['dV/V(%)', 'dVmet/V(%)', 't(s)_y', 'rho_t(cm^-2)', 'Rm(cm)']
    cols = ['pars', 'dpaNRT_x', 'ditot(cm^-3)', 'dvtot(cm^-3)',
       'dimet(cm^-3)', 'dvmet(cm^-3)', 'rimet(cm)', 'rvmet(cm)',
       'dV/V(%)', 'dVmet/V(%)', 'rho_t(cm^-2)', 'Rm(cm)', 'RCsum', 'Csum']
    df = df[cols]
    print(df.columns)

    row = df.iloc[10]['pars']
    print('####################')

    #parameters = ['he', 'Tk', 'Eb2i', 'Eb2v', 'Zi', 'riv', 'n2dmin', 'cs', 'emi']
    ## final curves version
    parameters = ['Tk', 'gc', 'He','t']

    ###initiate new columns
    df['batch'] = str()
    for par in parameters:
        df[par] = np.nan


    for i, row in df.iterrows():
        '''
        batch = row['pars'].split('/')[1]
        # convert batch to index:
                          'sens01':1,
        batch_to_index = {'sens':0,
                          'sens03':2,
                          'sens04':3,
                          'sens05_mi5':4,
                          'sens06':5,
                          'sens_ST5':6,
                          'sens_eole01': 7,
                          'sens_eole04': 8,
                          'sens_eole05': 9
                          }
        df.loc[i,'batch'] = batch_to_index[batch]
        '''

        ps = row['pars'].split('/')[1].split('_')
        print('##init##')
        # Tk360_gc4.18e-08_He1.00e+00
        print(i, row)
        for p in ps:
            if 'Tk' in p:
                Tk = float(p[2:])
                #print('-------Tk', Tk)
                df.loc[i,'Tk'] = Tk
            elif 'gc' in p:
                gc = float(p[2:])
                #print('-------gc', gc)
                df.loc[i,'gc'] = gc
            elif 'He' in p:
                He = float(p[2:])
                #print('-------He', He)
                df.loc[i,'He'] = He
            elif 't' in p:
                t = float(p[1:])
                #print('-------t', t)
                df.loc[i,'t'] = t

        print(df.iloc[i])

    df.to_csv(name, index=False)

    return os.path.abspath(name)

def compare_w_t(df):

    _x = df['Tk']
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10, 12))

    axs[0, 0].scatter(_x, df['dV/V(%)'], label='dV/V(%)')
    axs[0, 0].scatter(_x, df['dVmet/V(%)'], marker='x', label='dVmet/V(%)')
    axs[0, 0].set_title("swelling %")
    #axs[0, 0].plot(df_sw['t(s)'] * gc, df_sw['dV/V(%)'], linestyle='--', c='k', label='model:total')
    #axs[0, 0].plot(df_sw['t(s)'] * gc, df_sw['dVmet/V(%)'], linestyle='-', marker='o', c='k', label='model:>TEM limit')
    #axs[0, 0].legend()

    #y01 = np.pi * df_fl['size(nm)'] * 1e-9 * df_fl['density(m^-3)']
    axs[0, 1].scatter(_x, df['rho_t(cm^-2)'], marker='x', label='rho_t(cm^-2)')  # dislo
    axs[0, 1].set_title("rho_t(cm^-2)")

    # ax1.fill_between([t_min * gc, t_max * gc] , cw_min, cw_max, facecolor='b', alpha=0.2, label='CW range')
    # ax1.fill_between([t_min * gc, t_max * gc], sa_min, sa_max, facecolor='grey', alpha=0.2, label='SA range')
    ##axs[0, 1].fill_between([t_min * gc, t_max * gc], 5e14, 4e15, facecolor='grey', alpha=0.2, label=r'$\sim \rho_d$ at >10dpa')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_ylim((1e14, 1e20))
    axs[0, 1].legend()

    print('why im not displaying?')
    fig.show()
    print('very very sad')

def main():
    ### uncomment after debugging
    path = prepare_df_for_surfplot()
    #path = '/home/g39677/DATA/DATA/crescendo_dat/gaia_out/merged_out.csv'
    df = pd.read_csv(path)
    print(df.columns)
    print(df.head(3))

    cols = ['pars', 'dpaNRT_x', 'ditot(cm^-3)', 'dvtot(cm^-3)',
       'dimet(cm^-3)', 'dvmet(cm^-3)', 'rimet(cm)', 'rvmet(cm)',
       'dV/V(%)', 'dVmet/V(%)', 'rho_t(cm^-2)', 'Rm(cm)']
    parameters = ['Tk', 'gc', 'He']

    df=df[cols+parameters]
    df['rho_t(cm^-2)'] = np.log10(df['rho_t(cm^-2)'])
    df['dimet(cm^-3)'] = np.log10(df['dimet(cm^-3)'])
    df['dvmet(cm^-3)'] = np.log10(df['dvmet(cm^-3)'])
    df['gc'] = np.log10(df['gc'])
    df['rimet(cm)'] = df['rimet(cm)'] * 1e7
    df['rvmet(cm)'] = df['rvmet(cm)'] * 1e7
    df['Rm(cm)'] = df['Rm(cm)'] * 1e7

    #compare_w_t(df)
    print('size', df.shape)
    print('incomplete:', df[df['dpaNRT_x'] < 70].shape)
    df = df[df['dpaNRT_x'] > 70]
    ### start custom filtering
    df = df[df['Tk'] == 633]
    df = df[df['gc'] < -6.2]
    ### end custom filtering
    print('num of records', df.shape)
    fig = px.parallel_coordinates(df)#, dimensions=cols)
    fig.show()


    df = df.sort_values(['Tk', 'He'], ascending=False)
    #z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

    print(df['dVmet/V(%)'].values.shape[0])
    print('set(df[Tk]), set(df[He])', len(set(df['Tk'])), len(set(df['He'])))

    ## preapre z
    '''
    z = df['dVmet/V(%)'].values.reshape(len(set(df['Tk'])), len(set(df['He'])))
    ## prepare x and y
    x = sorted( df['Tk'].unique().tolist())
    y= sorted( df['He'].unique().tolist())

    fig = go.Figure(data=[go.Surface(z=z)] )  # sorted(, reverse=True)
    #fig.update_layout(title='Mt Bruno Elevation', autosize=False, width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))
    fig.show()
    '''
    #'''
    fig = go.Figure(data=[go.Scatter3d(
        x=df['Tk'] - 273,
        y=df['He'],
        z=df['dVmet/V(%)'],
        #z=df['dV/V(%)'],
        #z=df['dvmet(cm^-3)'],
        #z=df['dimet(cm^-3)'],
        #z=df['rho_t(cm^-2)'],
        mode='markers',
        marker=dict(
            size=12,
            color=df['dpaNRT_x'],  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )])

    fig.update_layout(scene=dict(
        xaxis_title='T(C)',
        yaxis_title='He(appm/dpa)',
        zaxis_title='TEM swellig (%)'),
        )

    fig.show()
    #'''


    return


if __name__ == "__main__":
    # execute only if run as a script
    main()
