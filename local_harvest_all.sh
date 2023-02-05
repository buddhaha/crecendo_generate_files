#!/usr/bin/env bash

get_last_step_out() {
    #get final swelling
    #get final densrad: i, vac for rmet, dtot, dmet
    #get final dislo

    # 1:t(s), 2:dpaNRT, 3:ditot(cm^-3), 4:dvtot(cm^-3), 5:dimet(cm^-3), 6:dvmet(cm^-3), 7:ximet(cm^-3)
    # 8:xvmet(cm^-3), 9:rimet(cm), 10:rvmet(cm), 11:r2imet(cm^2), 12:r2vmet(cm^2), 13:r3imet(cm^3), 14:r3vmet(cm^3)
    # 15:DiCi 16:DvCv
    echo $f;
    #get_last_step_outs
    c_dens=$(tail -n20 $f/densrad_tot.dat | awk -v  OFS=';' '{print $1,$2,$3,$4,$5,$6,$7,$9,$10}')
    echo -e "$f;$c_dens" >> $densrad_out

    c_swelling=$(tail -n20 swelling_tot.dat |awk -v OFS=';' '{print $2,$3,$4}')
    echo -e "$f;$c_swelling" >> $swelling_out

    c_dislo=$(tail -n20 z0dislo.dat | awk -v OFS=';' '{print $1,$2,$3}')
    echo -e "$f;$c_dislo" >> $dislo_tot

}

#init out file
densrad_out='densrad_allsteps_out.csv'
swelling_out='swelling_allsteps_out.csv'
dislo_tot='dislo_tot_allsteps_out.csv'

echo "harvesting results into .." $densrad_out, $swelling_out

densrad_header='pars;t(s);dpaNRT;ditot(cm^-3);dvtot(cm^-3);dimet(cm^-3);dvmet(cm^-3);rimet(cm);rvmet(cm)'
echo -e $densrad_header > $densrad_out

swelling_header='pars;dpaNRT;dV/V(%);dVmet/V(%)'
echo -e $swelling_header > $swelling_out

dislo_tot_header='pars;t(s);rho_t(cm^-2);Rm(cm)'
echo -e $dislo_tot_header > $dislo_tot

#head -n3 z0dislo.dat |tail -n1

# awk -v var="$variable" 'BEGIN {print var}'

#### this line might need to be changed:
for f in $(find . -type d -name 'Tk*'|sort);
do
    echo $f;
    #get_last_step_outs
    c_dens=$(tail -n20 $f/densrad_tot.dat | awk -v par="$f" 'BEGIN{OFS=";"} {print par,$1,$2,$3,$4,$5,$6,$9,$10}')
    echo -e "$c_dens" >> $densrad_out

    c_swelling=$(tail -n20 $f/swelling_tot.dat |awk -v par="$f" 'BEGIN{OFS=";"} {print par,$2,$3,$4}')
    echo -e "$c_swelling" >> $swelling_out

    c_dislo=$(tail -n20 $f/z0dislo.dat | awk -v par="$f" 'BEGIN{OFS=";"} {print par,$1,$2,$3}')
    echo -e "$c_dislo" >> $dislo_tot
done
