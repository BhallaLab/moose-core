//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Thu Jan 31 10:05:56 2013
 
include kkit {argv 1}
 
FASTDT = 0.0001
SIMDT = 0.01
CONTROLDT = 5
PLOTDT = 1
MAXTIME = 100
TRANSIENT_TIME = 2
VARIABLE_DT_FLAG = 0
DEFAULT_VOL = 1e-15
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
simundump geometry /kinetics/geometry 0 9.9998e-16 3 sphere "" white black 11 \
  -7 0
simundump text /kinetics/notes 0 ""
call /kinetics/notes LOAD \
""
simundump group /kinetics/A 0 yellow black x 0 0 "" A defaultfile.g 0 0 0 -2 \
  0 0
simundump text /kinetics/A/notes 0 ""
call /kinetics/A/notes LOAD \
""
simundump kpool /kinetics/A/M1 0 0 0.99998 0.99998 5.9999e+05 5.9999e+05 0 0 \
  6e+05 0 /kinetics/geometry 9 yellow -4 4 0
simundump text /kinetics/A/M1/notes 0 ""
call /kinetics/A/M1/notes LOAD \
""
simundump kpool /kinetics/A/M2 0 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry 1 \
  yellow 0 4 0
simundump text /kinetics/A/M2/notes 0 ""
call /kinetics/A/M2/notes LOAD \
""
simundump kreac /kinetics/A/R1 0 0.1 0.1 "" white yellow -2 6 0
simundump text /kinetics/A/R1/notes 0 ""
call /kinetics/A/R1/notes LOAD \
""
simundump kreac /kinetics/A/R3 0 0.1 1.6667e-07 "" white yellow 0 1 0
simundump text /kinetics/A/R3/notes 0 ""
call /kinetics/A/R3/notes LOAD \
""
simundump kreac /kinetics/A/R4 0 0.1 0.1 "" white yellow 0 7 0
simundump text /kinetics/A/R4/notes 0 ""
call /kinetics/A/R4/notes LOAD \
""
simundump kreac /kinetics/A/R2 0 0.1 0.1 "" white yellow -2 2 0
simundump text /kinetics/A/R2/notes 0 ""
call /kinetics/A/R2/notes LOAD \
""
simundump group /kinetics/B 0 52 black x 0 0 "" B defaultfile.g 0 0 0 -2 -8 0
simundump text /kinetics/B/notes 0 ""
call /kinetics/B/notes LOAD \
""
simundump kpool /kinetics/B/M1 0 0 1 1 1.8e+06 1.8e+06 0 0 1.8e+06 0 \
  /kinetics/geometry 7 52 -4 -3 0
simundump text /kinetics/B/M1/notes 0 ""
call /kinetics/B/M1/notes LOAD \
""
simundump kpool /kinetics/B/M6 0 0 0 0 0 0 0 0 1.8e+06 0 /kinetics/geometry \
  26 52 0 -3 0
simundump text /kinetics/B/M6/notes 0 ""
call /kinetics/B/M6/notes LOAD \
""
simundump kpool /kinetics/B/M3 0 0 0 0 0 0 0 0 1.8e+06 0 /kinetics/geometry \
  blue 52 -2 -3 0
simundump text /kinetics/B/M3/notes 0 ""
call /kinetics/B/M3/notes LOAD \
""
simundump kenz /kinetics/B/M3/R6and7 0 0 0 0 0 6e+05 2.7778e-07 0.4 0.1 0 0 \
  "" red blue "" -2 -5 0
simundump text /kinetics/B/M3/R6and7/notes 0 ""
call /kinetics/B/M3/R6and7/notes LOAD \
""
simundump kreac /kinetics/B/R5 0 0.1 0.1 "" white 52 0 -6 0
simundump text /kinetics/B/R5/notes 0 ""
call /kinetics/B/R5/notes LOAD \
""
simundump group /kinetics/D 0 57 black x 0 0 "" D defaultfile.g 0 0 0 5 0 0
simundump text /kinetics/D/notes 0 ""
call /kinetics/D/notes LOAD \
""
simundump kpool /kinetics/D/M1 0 0 1 1 1.2e+06 1.2e+06 0 0 1.2e+06 0 \
  /kinetics/geometry 7 57 3 4 0
simundump text /kinetics/D/M1/notes 0 ""
call /kinetics/D/M1/notes LOAD \
""
simundump kpool /kinetics/D/M5 0 0 0 0 0 0 0 0 1.2e+06 0 /kinetics/geometry \
  21 57 7 4 0
simundump text /kinetics/D/M5/notes 0 ""
call /kinetics/D/M5/notes LOAD \
""
simundump kreac /kinetics/D/R9 0 0.1 0.1 "" white 57 5 2 0
simundump text /kinetics/D/R9/notes 0 ""
call /kinetics/D/R9/notes LOAD \
""
simundump text /kinetics/geometry/notes 0 ""
call /kinetics/geometry/notes LOAD \
""
simundump group /kinetics/C 0 62 black x 0 0 "" C defaultfile.g 0 0 0 5 -8 0
simundump text /kinetics/C/notes 0 ""
call /kinetics/C/notes LOAD \
""
simundump kpool /kinetics/C/M1 0 0 1 1 3e+06 3e+06 0 0 3e+06 0 \
  /kinetics/geometry 8 62 3 -3 0
simundump text /kinetics/C/M1/notes 0 ""
call /kinetics/C/M1/notes LOAD \
""
simundump kpool /kinetics/C/M4 0 0 0 0 0 0 0 0 3e+06 0 /kinetics/geometry 38 \
  62 7 -3 0
simundump text /kinetics/C/M4/notes 0 ""
call /kinetics/C/M4/notes LOAD \
""
simundump kreac /kinetics/C/R8 0 0.1 0.1 "" white 62 5 -6 0
simundump text /kinetics/C/R8/notes 0 ""
call /kinetics/C/R8/notes LOAD \
""
simundump xgraph /graphs/conc1 0 0 100 0 1 0
simundump xgraph /graphs/conc2 0 0 100 0 1 0
simundump xplot /graphs/conc1/M2.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 1 0 0 1
simundump xplot /graphs/conc1/M3.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xplot /graphs/conc1/M6.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 26 0 0 1
simundump xplot /graphs/conc2/M4.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 38 0 0 1
simundump xplot /graphs/conc2/M5.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 21 0 0 1
simundump xgraph /moregraphs/conc3 0 0 100 0 1 0
simundump xgraph /moregraphs/conc4 0 0 100 0 1 0
simundump xcoredraw /edit/draw 0 -8.2365 12.355 -9.2661 11.325
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
addmsg /kinetics/A/R1 /kinetics/A/M1 REAC A B 
addmsg /kinetics/A/R2 /kinetics/A/M1 REAC B A 
addmsg /kinetics/A/R1 /kinetics/A/M2 REAC B A 
addmsg /kinetics/A/R4 /kinetics/A/M2 REAC A B 
addmsg /kinetics/A/R2 /kinetics/A/M2 REAC A B 
addmsg /kinetics/A/R3 /kinetics/A/M2 REAC A B 
addmsg /kinetics/A/M1 /kinetics/A/R1 SUBSTRATE n 
addmsg /kinetics/A/M2 /kinetics/A/R1 PRODUCT n 
addmsg /kinetics/A/M2 /kinetics/A/R3 SUBSTRATE n 
addmsg /kinetics/B/M3 /kinetics/A/R3 PRODUCT n 
addmsg /kinetics/C/M4 /kinetics/A/R3 PRODUCT n 
addmsg /kinetics/A/M2 /kinetics/A/R4 SUBSTRATE n 
addmsg /kinetics/D/M5 /kinetics/A/R4 PRODUCT n 
addmsg /kinetics/A/M2 /kinetics/A/R2 SUBSTRATE n 
addmsg /kinetics/A/M1 /kinetics/A/R2 PRODUCT n 
addmsg /kinetics/B/M3/R6and7 /kinetics/B/M1 REAC sA B 
addmsg /kinetics/B/R5 /kinetics/B/M6 REAC A B 
addmsg /kinetics/B/M3/R6and7 /kinetics/B/M6 MM_PRD pA 
addmsg /kinetics/A/R3 /kinetics/B/M3 REAC B A 
addmsg /kinetics/B/M3/R6and7 /kinetics/B/M3 REAC eA B 
addmsg /kinetics/B/M3 /kinetics/B/M3/R6and7 ENZYME n 
addmsg /kinetics/B/M1 /kinetics/B/M3/R6and7 SUBSTRATE n 
addmsg /kinetics/B/M6 /kinetics/B/R5 SUBSTRATE n 
addmsg /kinetics/C/M4 /kinetics/B/R5 PRODUCT n 
addmsg /kinetics/D/R9 /kinetics/D/M1 REAC A B 
addmsg /kinetics/A/R4 /kinetics/D/M5 REAC B A 
addmsg /kinetics/D/R9 /kinetics/D/M5 REAC B A 
addmsg /kinetics/D/M1 /kinetics/D/R9 SUBSTRATE n 
addmsg /kinetics/D/M5 /kinetics/D/R9 PRODUCT n 
addmsg /kinetics/C/R8 /kinetics/C/M1 REAC A B 
addmsg /kinetics/A/R3 /kinetics/C/M4 REAC B A 
addmsg /kinetics/B/R5 /kinetics/C/M4 REAC B A 
addmsg /kinetics/C/R8 /kinetics/C/M4 REAC B A 
addmsg /kinetics/C/M1 /kinetics/C/R8 SUBSTRATE n 
addmsg /kinetics/C/M4 /kinetics/C/R8 PRODUCT n 
addmsg /kinetics/A/M2 /graphs/conc1/M2.Co PLOT Co *M2.Co *1 
addmsg /kinetics/B/M3 /graphs/conc1/M3.Co PLOT Co *M3.Co *blue 
addmsg /kinetics/B/M6 /graphs/conc1/M6.Co PLOT Co *M6.Co *26 
addmsg /kinetics/C/M4 /graphs/conc2/M4.Co PLOT Co *M4.Co *38 
addmsg /kinetics/D/M5 /graphs/conc2/M5.Co PLOT Co *M5.Co *21 
enddump
// End of dump

complete_loading
