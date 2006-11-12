//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Sun Nov 12 17:30:54 2006
 
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
simundump geometry /kinetics/geometry 0 1.6667e-21 3 sphere "" white black 7 \
  -6 0
simundump kpool /kinetics/M 0 0 1 1 1 1 0 0 1 0 /kinetics/geometry blue black \
  -3 2 0
simundump kpool /kinetics/Ca.M 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 25 \
  black 1 2 0
simundump kpool /kinetics/M* 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 7 black \
  -3 -2 0
simundump kpool /kinetics/Ca 0 0 1 1 1 1 0 0 1 0 /kinetics/geometry 62 black \
  -3 0 0
simundump kpool /kinetics/Ca.M* 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 0 \
  black 1 -2 0
simundump kreac /kinetics/phosph_Ca.M 0 0.1 0.2 "" white black 2 0 0
simundump kreac /kinetics/phosph_M 0 0.1 0.2 "" white black -5 0 0
simundump kpool /kinetics/Ca.M_ 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 25 \
  black 4 4 0
simundump kpool /kinetics/M_ 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry blue \
  black -6 4 0
simundump kpool /kinetics/M*_ 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 7 black \
  -6 -4 0
simundump kpool /kinetics/Ca.M*_ 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 0 \
  black 4 -4 0
simundump kreac /kinetics/phosph_M_ 0 0.1 0.2 "" white black -8 0 0
simundump kreac /kinetics/phosph_Ca.M_ 0 0.1 0.2 "" white black 6 0 0
simundump kreac /kinetics/Ca_bind_M 0 0.1 0.1 "" white black -1 1 0
simundump kreac /kinetics/Ca_bind_M* 0 0.1 0.1 "" white black -1 -1 0
simundump kreac /kinetics/Ca_bind_M_ 0 0.1 0.1 "" white black -1 5 0
simundump kreac /kinetics/Ca_bind_M*_ 0 0.1 0.1 "" white black -1 -5 0
simundump kreac /kinetics/mod_M 0 0.1 0.1 "" white black -6 2 0
simundump kreac /kinetics/mod_M* 0 0.2 0.1 "" white black -6 -2 0
simundump kreac /kinetics/mod_Ca.M 0 0.05 0.1 "" white black 4 2 0
simundump kreac /kinetics/mod_Ca.M* 0 0.05 0.1 "" white black 4 -2 0
simundump kpool /kinetics/Tot_M_ 0 0 0 0 0 0 0 0 1 0 /kinetics/geometry 47 \
  black -1 -10 0
simundump xgraph /graphs/conc1 0 0 200 0 1 0
simundump xgraph /graphs/conc2 0 0 200 0 1 0
simundump xplot /graphs/conc1/Ca.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 62 0 0 1
simundump xplot /graphs/conc1/M.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xplot /graphs/conc1/Ca.M.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 25 0 0 1
simundump xplot /graphs/conc1/M*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 7 0 0 1
simundump xplot /graphs/conc1/Ca.M*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 0 0 0 1
simundump xplot /graphs/conc2/M_.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xplot /graphs/conc2/Ca.M_.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 25 0 0 1
simundump xplot /graphs/conc2/M*_.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 7 0 0 1
simundump xplot /graphs/conc2/Ca.M*_.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 0 0 0 1
simundump xplot /graphs/conc2/Tot_M_.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 47 0 0 1
simundump xgraph /moregraphs/conc3 0 0 200 0 1 0
simundump xgraph /moregraphs/conc4 0 0 200 0 1 0
simundump xcoredraw /edit/draw 0 -13.144 12.144 -13.483 6.4825
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
addmsg /kinetics/Ca_bind_M /kinetics/M REAC A B 
addmsg /kinetics/phosph_M /kinetics/M REAC A B 
addmsg /kinetics/mod_M /kinetics/M REAC A B 
addmsg /kinetics/Ca_bind_M /kinetics/Ca.M REAC B A 
addmsg /kinetics/phosph_Ca.M /kinetics/Ca.M REAC A B 
addmsg /kinetics/mod_Ca.M /kinetics/Ca.M REAC A B 
addmsg /kinetics/Ca_bind_M* /kinetics/M* REAC A B 
addmsg /kinetics/phosph_M /kinetics/M* REAC B A 
addmsg /kinetics/mod_M* /kinetics/M* REAC A B 
addmsg /kinetics/Ca_bind_M /kinetics/Ca REAC A B 
addmsg /kinetics/Ca_bind_M* /kinetics/Ca REAC A B 
addmsg /kinetics/Ca_bind_M_ /kinetics/Ca REAC A B 
addmsg /kinetics/Ca_bind_M*_ /kinetics/Ca REAC A B 
addmsg /kinetics/Ca_bind_M* /kinetics/Ca.M* REAC B A 
addmsg /kinetics/phosph_Ca.M /kinetics/Ca.M* REAC B A 
addmsg /kinetics/mod_Ca.M* /kinetics/Ca.M* REAC A B 
addmsg /kinetics/Ca.M /kinetics/phosph_Ca.M SUBSTRATE n 
addmsg /kinetics/Ca.M* /kinetics/phosph_Ca.M PRODUCT n 
addmsg /kinetics/M /kinetics/phosph_M SUBSTRATE n 
addmsg /kinetics/M* /kinetics/phosph_M PRODUCT n 
addmsg /kinetics/phosph_Ca.M_ /kinetics/Ca.M_ REAC A B 
addmsg /kinetics/Ca_bind_M_ /kinetics/Ca.M_ REAC B A 
addmsg /kinetics/mod_Ca.M /kinetics/Ca.M_ REAC B A 
addmsg /kinetics/phosph_M_ /kinetics/M_ REAC A B 
addmsg /kinetics/Ca_bind_M_ /kinetics/M_ REAC A B 
addmsg /kinetics/mod_M /kinetics/M_ REAC B A 
addmsg /kinetics/phosph_M_ /kinetics/M*_ REAC B A 
addmsg /kinetics/Ca_bind_M*_ /kinetics/M*_ REAC A B 
addmsg /kinetics/mod_M* /kinetics/M*_ REAC B A 
addmsg /kinetics/phosph_Ca.M_ /kinetics/Ca.M*_ REAC B A 
addmsg /kinetics/Ca_bind_M*_ /kinetics/Ca.M*_ REAC B A 
addmsg /kinetics/mod_Ca.M* /kinetics/Ca.M*_ REAC B A 
addmsg /kinetics/M_ /kinetics/phosph_M_ SUBSTRATE n 
addmsg /kinetics/M*_ /kinetics/phosph_M_ PRODUCT n 
addmsg /kinetics/Ca.M_ /kinetics/phosph_Ca.M_ SUBSTRATE n 
addmsg /kinetics/Ca.M*_ /kinetics/phosph_Ca.M_ PRODUCT n 
addmsg /kinetics/M /kinetics/Ca_bind_M SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/Ca_bind_M SUBSTRATE n 
addmsg /kinetics/Ca.M /kinetics/Ca_bind_M PRODUCT n 
addmsg /kinetics/Ca /kinetics/Ca_bind_M* SUBSTRATE n 
addmsg /kinetics/M* /kinetics/Ca_bind_M* SUBSTRATE n 
addmsg /kinetics/Ca.M* /kinetics/Ca_bind_M* PRODUCT n 
addmsg /kinetics/M_ /kinetics/Ca_bind_M_ SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/Ca_bind_M_ SUBSTRATE n 
addmsg /kinetics/Ca.M_ /kinetics/Ca_bind_M_ PRODUCT n 
addmsg /kinetics/M*_ /kinetics/Ca_bind_M*_ SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/Ca_bind_M*_ SUBSTRATE n 
addmsg /kinetics/Ca.M*_ /kinetics/Ca_bind_M*_ PRODUCT n 
addmsg /kinetics/M /kinetics/mod_M SUBSTRATE n 
addmsg /kinetics/M_ /kinetics/mod_M PRODUCT n 
addmsg /kinetics/M* /kinetics/mod_M* SUBSTRATE n 
addmsg /kinetics/M*_ /kinetics/mod_M* PRODUCT n 
addmsg /kinetics/Ca.M /kinetics/mod_Ca.M SUBSTRATE n 
addmsg /kinetics/Ca.M_ /kinetics/mod_Ca.M PRODUCT n 
addmsg /kinetics/Ca.M* /kinetics/mod_Ca.M* SUBSTRATE n 
addmsg /kinetics/Ca.M*_ /kinetics/mod_Ca.M* PRODUCT n 
addmsg /kinetics/Ca.M_ /kinetics/Tot_M_ SUMTOTAL n nInit 
addmsg /kinetics/M_ /kinetics/Tot_M_ SUMTOTAL n nInit 
addmsg /kinetics/M*_ /kinetics/Tot_M_ SUMTOTAL n nInit 
addmsg /kinetics/Ca.M*_ /kinetics/Tot_M_ SUMTOTAL n nInit 
addmsg /kinetics/Ca /graphs/conc1/Ca.Co PLOT Co *Ca.Co *62 
addmsg /kinetics/M /graphs/conc1/M.Co PLOT Co *M.Co *blue 
addmsg /kinetics/Ca.M /graphs/conc1/Ca.M.Co PLOT Co *Ca.M.Co *25 
addmsg /kinetics/M* /graphs/conc1/M*.Co PLOT Co *M*.Co *7 
addmsg /kinetics/Ca.M* /graphs/conc1/Ca.M*.Co PLOT Co *Ca.M*.Co *0 
addmsg /kinetics/M_ /graphs/conc2/M_.Co PLOT Co *M_.Co *blue 
addmsg /kinetics/Ca.M_ /graphs/conc2/Ca.M_.Co PLOT Co *Ca.M_.Co *25 
addmsg /kinetics/M*_ /graphs/conc2/M*_.Co PLOT Co *M*_.Co *7 
addmsg /kinetics/Ca.M*_ /graphs/conc2/Ca.M*_.Co PLOT Co *Ca.M*_.Co *0 
addmsg /kinetics/Tot_M_ /graphs/conc2/Tot_M_.Co PLOT Co *Tot_M_.Co *47 
enddump
// End of dump

call /kinetics/phosph_Ca.M/notes LOAD \
"As we do not have modulation of this phosph rate in this" \
"specific test model, we must use the same rate here as" \
"for phosph_M." \
""
call /kinetics/phosph_M/notes LOAD \
"This is actually a kinase-phosphatase cycle, but I" \
"am leaving out the kinase and phosphatase enzymes as" \
"they are not regulated in this scenario." \
""
call /kinetics/phosph_M_/notes LOAD \
"Same rate as phosph_M, because there is no modulation of" \
"the phosphorylation reaction in the current scenario." \
""
call /kinetics/phosph_Ca.M_/notes LOAD \
"Same rate as phosph_M, as there is no modulation of " \
"phosphorylation in the current model."
call /kinetics/mod_M/notes LOAD \
"Modify M on the underscore site distinct from phosph" \
"and Ca binding."
call /kinetics/mod_M*/notes LOAD \
"This is case 3, high rate." \
""
call /kinetics/mod_Ca.M/notes LOAD \
"This is case 2, lower rate" \
""
call /kinetics/mod_Ca.M*/notes LOAD \
"This also is case 2, lower rate." \
""
complete_loading
