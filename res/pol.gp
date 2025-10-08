set linetype 15 lw 1 lc rgb "black" pointtype 6
set xrange [-60:60]

set term cairolatex size 8cm,6cm
set output "POL_40.tex"

plot "block_avg_40.dat" using 1:5 w l lc "light-red" lw 1.6 title "$\P_1$", "" using 1:6 w l lc "web-green" lw 1.6 title "$\P_2$", "" using 1:7 w l lc "royalblue" lw 1.6 title "$\P_3$"
