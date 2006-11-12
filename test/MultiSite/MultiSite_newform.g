//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Sun Nov 12 17:34:33 2006
 
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
simundump geometry /kinetics/geometry 0 1.6667e-21 3 sphere "" white black 4 \
  -4 0
simundump kpool /kinetics/Ca 0 0 1 1 1 1 0 0 1 0 /kinetics/geometry 61 black \
  -3 0 0
simundump kpool /kinetics/M_S2 0 0 1 1 1 1 0 0 1 0 /kinetics/geometry blue \
  black -3 2 0
simundump kpool /kinetics/Ca.M_S2 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 53 \
  black 1 2 0
simundump kpool /kinetics/M_S3 0 0 1 1 1 1 0 0 1 0 /kinetics/geometry 8 black \
  -3 -2 0
simundump kreac /kinetics/Ca_bind_M_S2 0 0.1 0.1 "" white black -1 1 0
simundump kpool /kinetics/M_S3* 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 1 \
  black 1 -2 0
simundump kreac /kinetics/phosph_M_S3 0 0.1 0.2 "" white black -1 -1 0
simundump kpool /kinetics/M_S0_ 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 26 \
  black 1 -4 0
simundump kpool /kinetics/M_S0 0 0 1 1 1 1 0 0 1 0 /kinetics/geometry 20 \
  black -3 -4 0
simundump kreac /kinetics/mod_M_S0 0 0.1 0.1 "" white black -1 -3 0
simundump kpool /kinetics/M_S1 0 0 0.4 0.4 0.4 0.4 0 0 1 0 /kinetics/geometry \
  49 black -3 4 0
simundump xgraph /graphs/conc1 0 0 100 0 1 0
simundump xgraph /graphs/conc2 0 0 100 0 1 0
simundump xplot /graphs/conc1/Ca.M_S2.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 53 0 0 1
simundump xplot /graphs/conc1/M_S3*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 1 0 0 1
simundump xplot /graphs/conc2/M_S0_.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 26 0 0 1
simundump xplot /graphs/conc2/M_S1.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 49 0 0 1
simundump xplot /graphs/conc2/Ca.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 61 0 0 1
simundump xgraph /moregraphs/conc3 0 0 100 0 1 0
simundump xgraph /moregraphs/conc4 0 0 100 0 1 0
simundump xcoredraw /edit/draw 0 -5 6 -6 6
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
addmsg /kinetics/Ca_bind_M_S2 /kinetics/Ca REAC A B 
addmsg /kinetics/Ca_bind_M_S2 /kinetics/M_S2 REAC A B 
addmsg /kinetics/Ca_bind_M_S2 /kinetics/Ca.M_S2 REAC B A 
addmsg /kinetics/phosph_M_S3 /kinetics/M_S3 REAC A B 
addmsg /kinetics/M_S2 /kinetics/Ca_bind_M_S2 SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/Ca_bind_M_S2 SUBSTRATE n 
addmsg /kinetics/Ca.M_S2 /kinetics/Ca_bind_M_S2 PRODUCT n 
addmsg /kinetics/phosph_M_S3 /kinetics/M_S3* REAC B A 
addmsg /kinetics/M_S3 /kinetics/phosph_M_S3 SUBSTRATE n 
addmsg /kinetics/M_S3* /kinetics/phosph_M_S3 PRODUCT n 
addmsg /kinetics/mod_M_S0 /kinetics/M_S0_ REAC B A 
addmsg /kinetics/mod_M_S0 /kinetics/M_S0 REAC A B 
addmsg /kinetics/M_S0 /kinetics/mod_M_S0 SUBSTRATE n 
addmsg /kinetics/M_S0_ /kinetics/mod_M_S0 PRODUCT n 
addmsg /kinetics/Ca.M_S2 /graphs/conc1/Ca.M_S2.Co PLOT Co *Ca.M_S2.Co *53 
addmsg /kinetics/M_S3* /graphs/conc1/M_S3*.Co PLOT Co *M_S3*.Co *1 
addmsg /kinetics/M_S0_ /graphs/conc2/M_S0_.Co PLOT Co *M_S0_.Co *26 
addmsg /kinetics/M_S1 /graphs/conc2/M_S1.Co PLOT Co *M_S1.Co *49 
addmsg /kinetics/Ca /graphs/conc2/Ca.Co PLOT Co *Ca.Co *61 
enddump
// End of dump

call /kinetics/mod_M_S0/notes LOAD \
"This is the site that is going to be modulated." \
""
complete_loading
