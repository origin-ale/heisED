reset session
set datafile columnheaders

set term png font "Libertinus Sans,34" size 1280,960


set style line 1 lc rgb "mediumpurple3" pt 5 dt 1
set style line 2 lc rgb "coral" pt 5 dt 1 
set style data linespoints

set pointsize 2

set xlabel "L"
set xrange [0:20]

set key on font ",28" box height 1 width .5

set output "output/plot_E.png"
set ylabel "E_0"
set yrange [-9:0]
plot "output/J1.00000-new.txt" using 1:2 ls 1 title "Lanczos ED"

set output "output/plot_EL.png"
set ylabel "E_0/L"
set yrange [-0.55:-0.4]
tdlim = - 0.4431
plot "output/J1.00000-new.txt" using 1:3 ls 1 title "Lanczos ED",\
tdlim title "Exact solution in TD lim"

set output "output/plot_durations.png"
set ylabel "Duration (s)"
set yrange [*:100000]
set logscale y
plot "output/J1.00000-new.txt" using 1:4 ls 2 title "Lanczos ED duration" 

