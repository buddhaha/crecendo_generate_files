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
  ##  z (cm)   gc (dpaNRT/s) gcv (dpa/s)  gci (dpa/s)  gs (pa/s): 10 appm
  #sed -i "10s/.*/ 1.0000e+00  $gc  $gci $gci  $he/" $name/cascades.cnf
}



i=0
#seq min step max
#for temp in $(seq 270 130 400)
#for temp in $(seq 543 10 673)
for temp in $(seq 483 20 983)
do
    echo $temp $gc $gci $heappm $he
    ((i=i+1))
    echo index: $i
    name="Tk"$temp"_gc4.18e-08_He0.00"
    echo $name
    mkdir $name
    cp cnf_init/* $name
    replace_inputs
done
