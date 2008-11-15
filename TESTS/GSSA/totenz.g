//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Sat Nov 15 16:14:37 2008
 
include kkit {argv 1}
 
FASTDT = 0.0001
SIMDT = 0.01
CONTROLDT = 5
PLOTDT = 1
MAXTIME = 100
TRANSIENT_TIME = 2
VARIABLE_DT_FLAG = 0
DEFAULT_VOL = 1.6667e-21
VERSION = 11.0
setfield /file/modpath value /home2/bhalla/scripts/modules
kparms
 
//genesis

initdump -version 3 -ignoreorphans 1
simobjdump doqcsinfo filename accessname accesstype transcriber developer \
  citation species tissue cellcompartment methodology sources \
  model_implementation model_validation x y z
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
simundump geometry /kinetics/geometry 0 1.6667e-21 3 sphere "" white black 4 \
  -4 0
simundump kpool /kinetics/B 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 61 black 1 \
  1 0
simundump kpool /kinetics/A 0 0 1 1 1 1 0 0 1 0 /kinetics/geometry blue black \
  -3 1 0
simundump kpool /kinetics/E 0 0 1 1 1 1 0 0 1 0 /kinetics/geometry 1 black -1 \
  4 0
simundump kenz /kinetics/E/kenz 0 0 0 0 0 1 0.1 0.4 0.1 0 0 "" red 1 "" -1 3 \
  0
simundump kpool /kinetics/tot 0 0 0 1 1 0 0 0 1 0 /kinetics/geometry 24 black \
  -1 -2 0
simundump kenz /kinetics/tot/kenz 0 0 0 0 0 1 0.1 0.4 0.1 0 0 "" red 24 "" -1 \
  -3 0
simundump kreac /kinetics/kreac 0 0.1 0.1 "" white black 1 -3 0
simundump kpool /kinetics/convTot 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 46 \
  black 3 -2 0
simundump kpool /kinetics/Q 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 51 black 1 \
  -4 0
simundump kpool /kinetics/P 0 0 1 1 1 1 0 0 1 0 /kinetics/geometry blue black \
  -3 -4 0
simundump xgraph /graphs/conc1 0 0 100 0 1 0
simundump xgraph /graphs/conc2 0 0 100 0 1 0
simundump xplot /graphs/conc1/A.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xplot /graphs/conc1/B.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 61 0 0 1
simundump xplot /graphs/conc1/E.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 1 0 0 1
simundump xplot /graphs/conc2/tot.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 24 0 0 1
simundump xplot /graphs/conc2/convTot.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 46 0 0 1
simundump xplot /graphs/conc2/Q.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 51 0 0 1
simundump xgraph /moregraphs/conc3 0 0 100 0 1 0
simundump xgraph /moregraphs/conc4 0 0 100 0 1 0
simundump xcoredraw /edit/draw 0 -5 6 -6 6
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
addmsg /kinetics/E/kenz /kinetics/B MM_PRD pA 
addmsg /kinetics/E/kenz /kinetics/A REAC sA B 
addmsg /kinetics/E/kenz /kinetics/E REAC eA B 
addmsg /kinetics/E /kinetics/E/kenz ENZYME n 
addmsg /kinetics/A /kinetics/E/kenz SUBSTRATE n 
addmsg /kinetics/B /kinetics/tot SUMTOTAL n nInit 
addmsg /kinetics/A /kinetics/tot SUMTOTAL n nInit 
addmsg /kinetics/kreac /kinetics/tot REAC A B 
addmsg /kinetics/tot/kenz /kinetics/tot REAC eA B 
addmsg /kinetics/tot /kinetics/tot/kenz ENZYME n 
addmsg /kinetics/P /kinetics/tot/kenz SUBSTRATE n 
addmsg /kinetics/tot /kinetics/kreac SUBSTRATE n 
addmsg /kinetics/convTot /kinetics/kreac PRODUCT n 
addmsg /kinetics/kreac /kinetics/convTot REAC B A 
addmsg /kinetics/tot/kenz /kinetics/Q MM_PRD pA 
addmsg /kinetics/tot/kenz /kinetics/P REAC sA B 
addmsg /kinetics/A /graphs/conc1/A.Co PLOT Co *A.Co *blue 
addmsg /kinetics/B /graphs/conc1/B.Co PLOT Co *B.Co *61 
addmsg /kinetics/E /graphs/conc1/E.Co PLOT Co *E.Co *1 
addmsg /kinetics/tot /graphs/conc2/tot.Co PLOT Co *tot.Co *24 
addmsg /kinetics/convTot /graphs/conc2/convTot.Co PLOT Co *convTot.Co *46 
addmsg /kinetics/Q /graphs/conc2/Q.Co PLOT Co *Q.Co *51 
enddump
// End of dump

complete_loading
