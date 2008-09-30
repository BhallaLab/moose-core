//genesis
simobjdump group xtree_fg_req xtree_textfg_req plotfield expanded movealone \
  link savename file version md5sum mod_save_flag x y z
simobjdump geometry size dim shape outside xtree_fg_req xtree_textfg_req x y \
  z
simobjdump kpool DiffConst CoInit Co n nInit mwt nMin vol slave_enable \
  geomname xtree_fg_req xtree_textfg_req x y z
simobjdump kreac kf kb notes xtree_fg_req xtree_textfg_req x y z
simobjdump kenz CoComplexInit CoComplex nComplexInit nComplex vol k1 k2 k3 \
  keepconc usecomplex notes xtree_fg_req xtree_textfg_req link x y z
simundump geometry /kinetics/geometry 0 1.6667e-21 3 sphere "" white black 0 \
  0 0
simundump kpool /kinetics/Ca_input 0 0 0.08 0.08 0.08 0.08 0 0 1 0 \
  /kinetics/geometry 60 black -4 2 0
simundump kpool /kinetics/K_A 0 0 1 1 1 1 0 0 1 0 /kinetics/geometry 0 black \
  1 2 0
simundump kreac /kinetics/kreac 0 1 1 "" white black -2 -1 0
simundump kpool /kinetics/disabled_K_A 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry \
  blue black 1 -2 0
addmsg /kinetics/kreac /kinetics/Ca_input REAC A B 
addmsg /kinetics/kreac /kinetics/K_A REAC A B 
addmsg /kinetics/Ca_input /kinetics/kreac SUBSTRATE n 
addmsg /kinetics/K_A /kinetics/kreac SUBSTRATE n 
addmsg /kinetics/disabled_K_A /kinetics/kreac PRODUCT n 
addmsg /kinetics/kreac /kinetics/disabled_K_A REAC B A 
