//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Sat Jul 14 19:09:00 2012
 
include kkit {argv 1}
 
FASTDT = 0.0001
SIMDT = 0.01
CONTROLDT = 5
PLOTDT = 1
MAXTIME = 1000
TRANSIENT_TIME = 2
VARIABLE_DT_FLAG = 0
DEFAULT_VOL = 1.6667e-21
VERSION = 11.0
setfield /file/modpath value /home2/bhalla/scripts/modules
kparms
 
//genesis

initdump -version 3 -ignoreorphans 1
simobjdump table input output alloced step_mode stepsize x y z
simobjdump xtree path script namemode sizescale
simobjdump xcoredraw xmin xmax ymin ymax
simobjdump xtext editable
simobjdump xgraph xmin xmax ymin ymax overlay
simobjdump xplot pixflags script fg ysquish do_slope wy
simobjdump group xtree_fg_req xtree_textfg_req plotfield expanded movealone \
  link savename file version md5sum mod_save_flag x y z
simobjdump geometry size dim shape outside xtree_fg_req xtree_textfg_req x y \
  z
simobjdump kpool DiffConst CoInit Co n nInit mwt nMin vol slave_enable \
  geomname xtree_fg_req xtree_textfg_req x y z
simobjdump kreac kf kb notes xtree_fg_req xtree_textfg_req x y z
simobjdump kenz CoComplexInit CoComplex nComplexInit nComplex vol k1 k2 k3 \
  keepconc usecomplex notes xtree_fg_req xtree_textfg_req link x y z
simobjdump stim level1 width1 delay1 level2 width2 delay2 baselevel trig_time \
  trig_mode notes xtree_fg_req xtree_textfg_req is_running x y z
simobjdump xtab input output alloced step_mode stepsize notes editfunc \
  xtree_fg_req xtree_textfg_req baselevel last_x last_y is_running x y z
simobjdump kchan perm gmax Vm is_active use_nernst notes xtree_fg_req \
  xtree_textfg_req x y z
simobjdump transport input output alloced step_mode stepsize dt delay clock \
  kf xtree_fg_req xtree_textfg_req x y z
simobjdump proto x y z
simundump geometry /kinetics/geometry 0 1.6667e-21 3 sphere "" white black 0 \
  0 0
simundump kpool /kinetics/S1 0 0 0.5 1.3516 1.3516 0.5 0 0 1 0 \
  /kinetics/geometry blue black -5 1 0
simundump kpool /kinetics/P1 0 0 0 3.4319 3.4319 0 0 0 1 0 /kinetics/geometry \
  25 black 4 2 0
simundump kpool /kinetics/S2 0 0 0.5 0.062449 0.062449 0.5 0 0 1 0 \
  /kinetics/geometry 3 black -5 -2 0
simundump kpool /kinetics/P2 0 0 0 4.9376 4.9376 0 0 0 1 0 /kinetics/geometry \
  62 black 5 -2 0
simundump kpool /kinetics/E1_explicit 0 0 0.1 0.78344 0.78344 0.1 0 0 1 0 \
  /kinetics/geometry 48 black 0 8 0
simundump kenz /kinetics/E1_explicit/explicit 0 0 0.21658 0 0.21658 1 0.01 \
  0.04 0.01 0 0 "" red 48 "" 0 6 0
simundump kpool /kinetics/E2_class 0 0 0.1 1 1 0.1 0 0 1 0 /kinetics/geometry \
  33 black 0 -7 0
simundump kenz /kinetics/E2_class/classical 0 0 0.007746 0 0.007746 1 0.01 \
  0.04 0.04 0 1 "" red 33 "" 0 -5 0
simundump xgraph /graphs/conc1 0 0 1000 0.062449 4.9998 0
simundump xgraph /graphs/conc2 0 0 1000 0 4.9377 0
simundump xplot /graphs/conc1/S1.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xplot /graphs/conc1/S2.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 3 0 0 1
simundump xplot /graphs/conc2/P2.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 62 0 0 1
simundump xplot /graphs/conc2/P1.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 25 0 0 1
simundump xgraph /moregraphs/conc3 0 0 1000 0 1 0
simundump xgraph /moregraphs/conc4 0 0 1000 0 1 0
simundump xcoredraw /edit/draw 0 -7 7 -9 10
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
addmsg /kinetics/E1_explicit/explicit /kinetics/S1 REAC sA B 
addmsg /kinetics/E1_explicit/explicit /kinetics/P1 MM_PRD pA 
addmsg /kinetics/E2_class/classical /kinetics/S2 REAC sA B 
addmsg /kinetics/E2_class/classical /kinetics/P2 MM_PRD pA 
addmsg /kinetics/E1_explicit/explicit /kinetics/E1_explicit REAC eA B 
addmsg /kinetics/E1_explicit /kinetics/E1_explicit/explicit ENZYME n 
addmsg /kinetics/S1 /kinetics/E1_explicit/explicit SUBSTRATE n 
addmsg /kinetics/E2_class /kinetics/E2_class/classical ENZYME n 
addmsg /kinetics/S2 /kinetics/E2_class/classical SUBSTRATE n 
addmsg /kinetics/S1 /graphs/conc1/S1.Co PLOT Co *S1.Co *blue 
addmsg /kinetics/S2 /graphs/conc1/S2.Co PLOT Co *S2.Co *3 
addmsg /kinetics/P2 /graphs/conc2/P2.Co PLOT Co *P2.Co *62 
addmsg /kinetics/P1 /graphs/conc2/P1.Co PLOT Co *P1.Co *25 
enddump
// End of dump

complete_loading
