//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Mon Jan 16 21:41:24 2006
 
include kkit {argv 1}
 
FASTDT = 0.0001
SIMDT = 0.01
CONTROLDT = 5
PLOTDT = 1
MAXTIME = 300
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
simundump kpool /kinetics/kpool 0 0 0 0 0 0 0 0 1 2 /kinetics/geometry blue \
  black 1 2 0
simundump xtab /kinetics/xtab 0 -100 0 1 2 0 "" edit_xtab white red 0 0 0 1 \
  -3 2 0
loadtab /kinetics/xtab table 1 100 0 100 \
 2 1.951 1.809 1.5878 1.309 0.99999 0.69103 0.41222 0.19097 0.048953 0 \
 0.048929 0.19098 0.41224 0.69096 1 1.309 1.5878 1.809 1.951 2 1.9511 1.809 \
 1.5878 1.309 0.99998 0.69102 0.41229 0.19113 0.049072 1.1921e-07 0.049026 \
 0.19105 0.41217 0.69078 0.99963 1.3095 1.588 1.8091 1.9511 2 1.9512 1.8093 \
 1.5875 1.3088 0.99996 0.6911 0.41244 0.19124 0.04882 0 0.048969 0.19094 \
 0.41202 0.69061 1.0004 1.3093 1.5879 1.809 1.951 2 1.9512 1.8088 1.5876 \
 1.309 1.0001 0.69128 0.41259 0.19076 0.048878 0 0.048911 0.19083 0.41187 \
 0.69138 1.0003 1.3091 1.5877 1.8089 1.9509 2 1.951 1.8089 1.5878 1.3092 \
 1.0003 0.69145 0.41193 0.19087 0.048934 0 0.048854 0.19072 0.41253 0.6912 \
 1.0001 1.3089 1.5876 1.8088 1.9512 2
simundump xgraph /graphs/conc1 0 0 300 4.1893e-13 2 0
simundump xgraph /graphs/conc2 0 0 300 0 1 0
simundump xplot /graphs/conc1/kpool.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xgraph /moregraphs/conc3 0 0 300 0 1 0
simundump xgraph /moregraphs/conc4 0 0 300 0 1 0
simundump xcoredraw /edit/draw 0 -5 5 -5 5
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
addmsg /kinetics/xtab /kinetics/kpool SLAVE output 
addmsg /kinetics/kpool /graphs/conc1/kpool.Co PLOT Co *kpool.Co *blue 
enddump
// End of dump

setfield /kinetics/xtab table->dx 1
setfield /kinetics/xtab table->invdx 1
complete_loading
