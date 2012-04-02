//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Mon Apr  2 22:07:13 2012
 
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
simobjdump text str
simundump geometry /kinetics/geometry 0 1.6667e-21 3 sphere "" white black 0 \
  0 0
simundump text /kinetics/notes 0 ""
call /kinetics/notes LOAD \
""
simundump kpool /kinetics/E 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 60 black \
  -1 3 0
simundump text /kinetics/E/notes 0 ""
call /kinetics/E/notes LOAD \
""
simundump kenz /kinetics/E/kenz 0 1 1 1 1 1 0.1 0.4 0.1 0 0 "" red 60 "" -1 2 \
  0
simundump text /kinetics/E/kenz/notes 0 ""
call /kinetics/E/kenz/notes LOAD \
""
simundump kpool /kinetics/S 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 0 black -3 \
  1 0
simundump text /kinetics/S/notes 0 ""
call /kinetics/S/notes LOAD \
""
simundump kpool /kinetics/P 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry blue black \
  1 1 0
simundump text /kinetics/P/notes 0 ""
call /kinetics/P/notes LOAD \
""
simundump text /kinetics/geometry/notes 0 ""
call /kinetics/geometry/notes LOAD \
""
simundump xgraph /graphs/conc1 0 0 100 0 1 0
simundump xgraph /graphs/conc2 0 0 100 0 1 0
simundump xplot /graphs/conc1/S.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 0 0 0 1
simundump xplot /graphs/conc1/E.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 60 0 0 1
simundump xplot /graphs/conc1/P.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xplot /graphs/conc2/kenz.CoComplex 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" red 0 0 1
simundump xgraph /moregraphs/conc3 0 0 100 0 1 0
simundump xgraph /moregraphs/conc4 0 0 100 0 1 0
simundump xcoredraw /edit/draw 0 -5 5 -5 5
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
addmsg /kinetics/E/kenz /kinetics/E REAC eA B 
addmsg /kinetics/E /kinetics/E/kenz ENZYME n 
addmsg /kinetics/S /kinetics/E/kenz SUBSTRATE n 
addmsg /kinetics/E/kenz /kinetics/S REAC sA B 
addmsg /kinetics/E/kenz /kinetics/P MM_PRD pA 
addmsg /kinetics/S /graphs/conc1/S.Co PLOT Co *S.Co *0 
addmsg /kinetics/E /graphs/conc1/E.Co PLOT Co *E.Co *60 
addmsg /kinetics/P /graphs/conc1/P.Co PLOT Co *P.Co *blue 
addmsg /kinetics/E/kenz /graphs/conc2/kenz.CoComplex PLOT CoComplex *kenz.CoComplex *red 
enddump
// End of dump

complete_loading
