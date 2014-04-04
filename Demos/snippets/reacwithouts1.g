//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Sat Apr  5 01:05:17 2014
 
include kkit {argv 1}
 
FASTDT = 0.0001
SIMDT = 0.01
CONTROLDT = 5
PLOTDT = 1
MAXTIME = 30
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
simobjdump text str
simundump geometry /kinetics/geometry 0 0.001 3 sphere "" white black 4 -3 0
simundump text /kinetics/notes 0 ""
call /kinetics/notes LOAD \
""
simundump kreac /kinetics/kreac 0 1.25e-12 0 "" white black -1 4 0
simundump text /kinetics/kreac/notes 0 ""
call /kinetics/kreac/notes LOAD \
""
simundump text /kinetics/geometry/notes 0 ""
call /kinetics/geometry/notes LOAD \
""
simundump kpool /kinetics/s1 0 0 1000 1000 6e+20 6e+20 0 0 6e+17 4 \
  /kinetics/geometry blue black 0 1 0
simundump text /kinetics/s1/notes 0 ""
call /kinetics/s1/notes LOAD \
""
simundump kpool /kinetics/s2 0 0 2000 3004.3 1.8026e+21 1.2e+21 0 0 6e+17 0 \
  /kinetics/geometry 58 black -2 1 0
simundump text /kinetics/s2/notes 0 ""
call /kinetics/s2/notes LOAD \
""
simundump kpool /kinetics/s3 0 0 1000 0.55308 3.3185e+17 6e+20 0 0 6e+17 0 \
  /kinetics/geometry 26 black 3 1 0
simundump text /kinetics/s3/notes 0 ""
call /kinetics/s3/notes LOAD \
""
simundump kreac /kinetics/kreac[1] 0 0.25 0 "" white black -1 -2 0
simundump text /kinetics/kreac[1]/notes 0 ""
call /kinetics/kreac[1]/notes LOAD \
""
simundump xgraph /graphs/conc1 0 0 29 0.7084 3004.2 0
simundump xgraph /graphs/conc2 0 0 30 0.7084 997.5 0
simundump xplot /graphs/conc1/s2.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 58 0 0 1
simundump xplot /graphs/conc1/s1.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xplot /graphs/conc1/s3.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 26 0 0 1
simundump xgraph /moregraphs/conc3 0 0 30 0 1 0
simundump xgraph /moregraphs/conc4 0 0 30 0 1 0
simundump xcoredraw /edit/draw 0 -5 5 -5 5
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
addmsg /kinetics/s3 /kinetics/kreac PRODUCT n 
addmsg /kinetics/s2 /kinetics/kreac SUBSTRATE n 
addmsg /kinetics/s1 /kinetics/s1 SUMTOTAL n nInit 
addmsg /kinetics/kreac /kinetics/s2 REAC A B 
addmsg /kinetics/kreac[1] /kinetics/s2 REAC B A 
addmsg /kinetics/kreac /kinetics/s3 REAC B A 
addmsg /kinetics/kreac[1] /kinetics/s3 REAC A B 
addmsg /kinetics/s3 /kinetics/kreac[1] SUBSTRATE n 
addmsg /kinetics/s2 /kinetics/kreac[1] PRODUCT n 
addmsg /kinetics/s2 /graphs/conc1/s2.Co PLOT Co *s2.Co *58 
addmsg /kinetics/s1 /graphs/conc1/s1.Co PLOT Co *s1.Co *blue 
addmsg /kinetics/s3 /graphs/conc1/s3.Co PLOT Co *s3.Co *26 
enddump
// End of dump

complete_loading
