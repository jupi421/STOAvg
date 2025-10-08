set linetype 15 lw 1 lc rgb "black" pointtype 6
set xrange [-60:60]

set term cairolatex size 8cm,6cm
set output "OP_20.tex"

plot "block_avg.dat" using 1:2 w l lc "light-red" lw 1.6 title "$\\phi_1$", "" using 1:3 w l lc "web-green" lw 1.6 title "$\\phi_2$", "" using 1:4 w l lc "royalblue" lw 1.6 title "$\\phi_3$"
