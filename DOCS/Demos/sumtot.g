//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Mon Jul  2 10:40:48 2007
 
include kkit {argv 1}
 
FASTDT = 0.0001
SIMDT = 0.01
CONTROLDT = 5
PLOTDT = 1
MAXTIME = 20
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
simobjdump text str
simundump geometry /kinetics/geometry 0 1.6667e-21 3 sphere "" white black 6 \
  -5 0
simundump kpool /kinetics/B 0 0 0 0.66524 0.66524 0 0 0 1 0 \
  /kinetics/geometry 62 black 1 1 0
simundump kpool /kinetics/A 0 0 1 0.33509 0.33509 1 0 0 1 0 \
  /kinetics/geometry blue black -3 1 0
simundump kreac /kinetics/kreac 0 0.2 0.1 "" white black -1 3 0
simundump kpool /kinetics/tot1 0 0 0 1.0003 1.0003 0 0 0 1 0 \
  /kinetics/geometry 47 black -1 -2 0
simundump kpool /kinetics/tot2 0 0 0 0.66524 0.66524 0 0 0 1 0 \
  /kinetics/geometry 0 black 3 -2 0
simundump kpool /kinetics/C 0 0 0 1.1118 1.1118 0 0 0 1 0 /kinetics/geometry \
  56 black 5 1 0
simundump kreac /kinetics/forward 0 0.1 0 "" white black 3 3 0
simundump xgraph /graphs/conc1 0 0 20 0 1.2 0
simundump xgraph /graphs/conc2 0 0 20 0 1.2 0
simundump xplot /graphs/conc1/A.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xplot /graphs/conc1/B.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 62 0 0 1
simundump xplot /graphs/conc2/tot1.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 47 0 0 1
simundump xplot /graphs/conc2/C.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 56 0 0 1
simundump xgraph /moregraphs/conc3 0 0 20 0 1.2 0
simundump xgraph /moregraphs/conc4 0 0 20 0 1.2 0
simundump xcoredraw /edit/draw 0 -4.55 7.55 -6.05 6.05
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
addmsg /kinetics/kreac /kinetics/B REAC B A 
addmsg /kinetics/kreac /kinetics/A REAC A B 
addmsg /kinetics/A /kinetics/kreac SUBSTRATE n 
addmsg /kinetics/B /kinetics/kreac PRODUCT n 
addmsg /kinetics/A /kinetics/tot1 SUMTOTAL n nInit 
addmsg /kinetics/B /kinetics/tot1 SUMTOTAL n nInit 
addmsg /kinetics/B /kinetics/tot2 SUMTOTAL n nInit 
addmsg /kinetics/forward /kinetics/tot2 REAC A B 
addmsg /kinetics/forward /kinetics/C REAC B A 
addmsg /kinetics/tot2 /kinetics/forward SUBSTRATE n 
addmsg /kinetics/C /kinetics/forward PRODUCT n 
addmsg /kinetics/A /graphs/conc1/A.Co PLOT Co *A.Co *blue 
addmsg /kinetics/B /graphs/conc1/B.Co PLOT Co *B.Co *62 
addmsg /kinetics/tot1 /graphs/conc2/tot1.Co PLOT Co *tot1.Co *47 
addmsg /kinetics/C /graphs/conc2/C.Co PLOT Co *C.Co *56 
enddump
// End of dump

complete_loading
