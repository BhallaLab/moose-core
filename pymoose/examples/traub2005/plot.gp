plot 'data/2009_05_06/Vm.plot' u ($0*1e-3):($1*1e3) w l, '../nrn/mydata/Vm.plot' u ($1):($2) w l
plot 'data/2009_04_24/m.plot', 'data/2009_04_22/m.bak.plot'
plot 'cal_xa.plot', 'cal_xa.plot.bak'
plot 'cal_xb.plot', 'cal_xb.plot.bak'
plot 'data/2009_04_25/Ca.plot' u ($0*1e-2):($1*1e3) w l, '../nrn/mydata/Ca.plot' u ($1):($2) w l
plot 'data/2009_04_24/Ca.plot' u ($0*1e-2):($1)
plot 'data/2009_04_25/m_kahp.plot' u ($0*1e-2):($1/9.42e-6), '../nrn/mydata/Vm.plot' u ($1):($3) 
plot 'data/2009_04_25/m_kahp.plot' u ($0*1e-2):($1)
plot 'beta.txt' u ($1*1e3):($2), '../nrn/mydata/Vm.plot' u ($2):($3)
plot  '../nrn/mydata/Vm.plot' u ($2):($3)
plot '~/src/sim/cortical/nrn/dat/spinstell_v_F.dat' w l, 'data/2009_05_03/Vm.plot' u ($0*1e-3):($1*1e3) w l

plot '~/src/sim/cortical/nrn/dat/spinstell_v_F.dat' w l

plot 'data/2009_05_06/Gk_NaF2.plot' u ($0 * 1e-3): ($1) w l, '../nrn/mydata/Vm.plot' u ($1): ($2) w l
plot 'data/2009_05_06/Gk_NaF2.plot'
plot 'data/2009_05_06/Vm.plot'
