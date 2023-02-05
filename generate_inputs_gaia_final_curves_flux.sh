#!/bin/bash
# A sample Bash script

#init variables

## change parameters in chosen input files for final swelling curves;
replace_inputs () {
  #recplace desired values in .cnf files
  # temp
  # he prod
  # dose_rate + time
  sed -i "53s/.*/  $temp /" $name/data.cnf
  sed -i "6s/.*/  $t /" $name/data.cnf
  ##  z (cm)   gc (dpaNRT/s) gcv (dpa/s)  gci (dpa/s)  gs (pa/s): 10 appm
  sed -i "10s/.*/ 1.0000e+00  $gc  $gci $gci  0/" $name/cascades.cnf
}



i=0
#inputs01
#temp=603
for temp in $(seq 473 20 873);
#for temp in 603;
do
    t=1.9e9
    gc=4.18e-8
    gci=$(python -c "print($gc*1e-1)")
    heappm=0
    he=$(python -c "print('{:.2e}'.format(1e-6*$heappm*$gc))")
    echo $temp $gc $gci $heappm $he $t
    ((i=i+1))
    echo index: $i
    name="Tk"$temp"_gc"$gc"_He"$heappm"_t"$t
    echo $name
    mkdir $name
    cp cnf_init/* $name
    replace_inputs

    #inputs02
    t=1.9e9
    gc=6.35e-8
    gci=$(python -c "print($gc*1e-1)")
    heappm=0
    he=$(python -c "print('{:.2e}'.format(1e-6*$heappm*$gc))")
    echo $temp $gc $gci $heappm $he $t
    ((i=i+1))
    echo index: $i
    name="Tk"$temp"_gc"$gc"_He"$heappm"_t"$t
    echo $name
    mkdir $name
    cp cnf_init/* $name
    replace_inputs

    #inputs03: 80dpa comparison
    t=8.0e7
    gc=1e-6
    gci=$(python -c "print($gc*1e-1)")
    heappm=0
    he=$(python -c "print('{:.2e}'.format(1e-6*$heappm*$gc))")
    echo $temp $gc $gci $heappm $he $t
    ((i=i+1))
    echo index: $i
    name="Tk"$temp"_gc"$gc"_He"$heappm"_t"$t
    echo $name
    mkdir $name
    cp cnf_init/* $name
    replace_inputs

    #inputs04: 80dpa comparison
    t=8.0e8
    gc=1e-7
    gci=$(python -c "print($gc*1e-1)")
    heappm=0
    he=$(python -c "print('{:.2e}'.format(1e-6*$heappm*$gc))")
    echo $temp $gc $gci $heappm $he $t
    ((i=i+1))
    echo index: $i
    name="Tk"$temp"_gc"$gc"_He"$heappm"_t"$t
    echo $name
    mkdir $name
    cp cnf_init/* $name
    replace_inputs

    #inputs05: 80dpa comparison
    t=8.0e9
    gc=1e-8
    gci=$(python -c "print($gc*1e-1)")
    heappm=0
    he=$(python -c "print('{:.2e}'.format(1e-6*$heappm*$gc))")
    echo $temp $gc $gci $heappm $he $t
    ((i=i+1))
    echo index: $i
    name="Tk"$temp"_gc"$gc"_He"$heappm"_t"$t
    echo $name
    mkdir $name
    cp cnf_init/* $name
    replace_inputs


    #inputs06: 120 dpa comparison
    t=1.2e8
    gc=1e-6
    gci=$(python -c "print($gc*1e-1)")
    heappm=0
    he=$(python -c "print('{:.2e}'.format(1e-6*$heappm*$gc))")
    echo $temp $gc $gci $heappm $he $t
    ((i=i+1))
    echo index: $i
    name="Tk"$temp"_gc"$gc"_He"$heappm"_t"$t
    echo $name
    mkdir $name
    cp cnf_init/* $name
    replace_inputs

    #inputs07: 120 dpa comparison
    t=1.2e9
    gc=1e-7
    gci=$(python -c "print($gc*1e-1)")
    heappm=0
    he=$(python -c "print('{:.2e}'.format(1e-6*$heappm*$gc))")
    echo $temp $gc $gci $heappm $he $t
    ((i=i+1))
    echo index: $i
    name="Tk"$temp"_gc"$gc"_He"$heappm"_t"$t
    echo $name
    mkdir $name
    cp cnf_init/* $name
    replace_inputs

    #inputs08: 120 dpa comparison
    t=1.2e10
    gc=1e-8
    gci=$(python -c "print($gc*1e-1)")
    heappm=0
    he=$(python -c "print('{:.2e}'.format(1e-6*$heappm*$gc))")
    echo $temp $gc $gci $heappm $he $t
    ((i=i+1))
    echo index: $i
    name="Tk"$temp"_gc"$gc"_He"$heappm"_t"$t
    echo $name
    mkdir $name
    cp cnf_init/* $name
    replace_inputs
done

















