//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Sat May 11 21:36:15 2013
 
include kkit {argv 1}
 
FASTDT = 0.0001
SIMDT = 0.01
CONTROLDT = 5
PLOTDT = 1
MAXTIME = 100
TRANSIENT_TIME = 2
VARIABLE_DT_FLAG = 0
DEFAULT_VOL = 1.26e-15
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
simundump text /kinetics/geometry/notes 0 ""
call /kinetics/geometry/notes LOAD \
""
simundump group /kinetics/dend 0 yellow black x 0 0 "" dend defaultfile.g 0 0 \
  0 -6 1 0
simundump text /kinetics/dend/notes 0 ""
call /kinetics/dend/notes LOAD \
""
simundump kpool /kinetics/dend/Ca 0 0 0.1 0.090408 68347 75598 0 0 7.5598e+05 \
  0 /kinetics/geometry 52 yellow -3 7 0
simundump text /kinetics/dend/Ca/notes 0 ""
call /kinetics/dend/Ca/notes LOAD \
""
simundump kpool /kinetics/dend/bufCa 0 0 0.1 0.1 75598 75598 0 0 7.5598e+05 4 \
  /kinetics/geometry 58 yellow -7 7 0
simundump text /kinetics/dend/bufCa/notes 0 ""
call /kinetics/dend/bufCa/notes LOAD \
""
simundump kreac /kinetics/dend/pumpCa 0 0.1 0.1 "" white yellow -5 6 0
simundump text /kinetics/dend/pumpCa/notes 0 ""
call /kinetics/dend/pumpCa/notes LOAD \
""
simundump kpool /kinetics/dend/kChan 0 0 1 0.8975 6.7849e+05 7.5598e+05 0 0 \
  7.5598e+05 0 /kinetics/geometry 7 yellow -3 4 0
simundump text /kinetics/dend/kChan/notes 0 ""
call /kinetics/dend/kChan/notes LOAD \
""
simundump kpool /kinetics/dend/kChan_p 0 0 0 0.050489 38168 0 0 0 7.5598e+05 \
  0 /kinetics/geometry 4 yellow -3 0 0
simundump text /kinetics/dend/kChan_p/notes 0 ""
call /kinetics/dend/kChan_p/notes LOAD \
""
simundump kreac /kinetics/dend/phosphatase 0 0.1 0 "" white yellow -2 2 0
simundump text /kinetics/dend/phosphatase/notes 0 ""
call /kinetics/dend/phosphatase/notes LOAD \
""
simundump kpool /kinetics/dend/Ca.kinase 0 0 0 0.29006 2.1928e+05 0 0 0 \
  7.5598e+05 0 /kinetics/geometry 25 yellow -5 2 0
simundump text /kinetics/dend/Ca.kinase/notes 0 ""
call /kinetics/dend/Ca.kinase/notes LOAD \
""
simundump kenz /kinetics/dend/Ca.kinase/enz 0 0 0.051861 0 39206 7.5598e+05 \
  1.3228e-07 0.4 0.1 0 0 "" red 25 "" -4 2 0
simundump text /kinetics/dend/Ca.kinase/enz/notes 0 ""
call /kinetics/dend/Ca.kinase/enz/notes LOAD \
""
simundump kreac /kinetics/dend/turnOnKinase 0 6.6139e-07 0.1 "" white yellow \
  -5 4 0
simundump text /kinetics/dend/turnOnKinase/notes 0 ""
call /kinetics/dend/turnOnKinase/notes LOAD \
""
simundump kpool /kinetics/dend/inact_kinase 0 0 1 0.65771 4.9722e+05 \
  7.5598e+05 0 0 7.5598e+05 0 /kinetics/geometry blue yellow -7 3 0
simundump text /kinetics/dend/inact_kinase/notes 0 ""
call /kinetics/dend/inact_kinase/notes LOAD \
""
simundump group /kinetics/spine 0 61 black x 0 0 "" spine defaultfile.g 0 0 0 \
  10 6 0
simundump text /kinetics/spine/notes 0 ""
call /kinetics/spine/notes LOAD \
""
simundump kpool /kinetics/spine/headGluR 0 0 0.00053079 0.00026226 49.409 100 \
  0 0 1.884e+05 0 /kinetics/geometry 47 61 7 5 0
simundump text /kinetics/spine/headGluR/notes 0 ""
call /kinetics/spine/headGluR/notes LOAD \
""
simundump kpool /kinetics/spine/toPsd 0 0 0 0.048743 9183.2 0 0 0 1.884e+05 0 \
  /kinetics/geometry 36 61 4 3 0
simundump text /kinetics/spine/toPsd/notes 0 ""
call /kinetics/spine/toPsd/notes LOAD \
""
simundump kenz /kinetics/spine/toPsd/enz 0 0 3.1859e-06 0 2.4084 7.5598e+05 \
  2.654e-05 4 1 0 0 "" red 36 "" 5 3 0
simundump text /kinetics/spine/toPsd/enz/notes 0 ""
call /kinetics/spine/toPsd/enz/notes LOAD \
""
simundump kpool /kinetics/spine/Ca 0 0 0.1 0.051242 9654.1 18840 0 0 \
  1.884e+05 0 /kinetics/geometry 52 61 7 7 0
simundump text /kinetics/spine/Ca/notes 0 ""
call /kinetics/spine/Ca/notes LOAD \
""
simundump kpool /kinetics/spine/toPsdInact 0 0 1 0.95122 1.7921e+05 1.884e+05 \
  0 0 1.884e+05 0 /kinetics/geometry blue 61 2 6 0
simundump text /kinetics/spine/toPsdInact/notes 0 ""
call /kinetics/spine/toPsdInact/notes LOAD \
""
simundump kreac /kinetics/spine/turnOnPsd 0 5.3079e-07 0.1 "" white 61 4 5 0
simundump text /kinetics/spine/turnOnPsd/notes 0 ""
call /kinetics/spine/turnOnPsd/notes LOAD \
""
simundump group /kinetics/psd 0 15 black x 0 0 "" psd defaultfile.g 0 0 0 4 0 \
  0
simundump text /kinetics/psd/notes 0 ""
call /kinetics/psd/notes LOAD \
""
simundump kpool /kinetics/psd/psdGluR 0 0 0 0.12775 48.159 0 0 0 376.99 0 \
  /kinetics/geometry 46 15 7 1 0
simundump text /kinetics/psd/psdGluR/notes 0 ""
call /kinetics/psd/psdGluR/notes LOAD \
""
simundump kreac /kinetics/psd/fromPsd 0 0.05 0 "" white 15 9 3 0
simundump text /kinetics/psd/fromPsd/notes 0 ""
call /kinetics/psd/fromPsd/notes LOAD \
""
simundump xgraph /graphs/conc1 0 0 100 0 1 0
simundump xgraph /graphs/conc2 0 0 100 0 0.27273 0
simundump xplot /graphs/conc1/Ca.kinase.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 25 0 0 1
simundump xplot /graphs/conc1/kChan.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 7 0 0 1
simundump xplot /graphs/conc1/kChan_p.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 4 0 0 1
simundump xplot /graphs/conc2/headGluR.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 47 0 0 1
simundump xplot /graphs/conc2/psdGluR.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 46 0 0 1
simundump xplot /graphs/conc2/toPsd.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 36 0 0 1
simundump xplot /graphs/conc2/Ca.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 52 0 0 1
simundump xgraph /moregraphs/conc3 0 0 100 0 1 0
simundump xgraph /moregraphs/conc4 0 0 100 0 1 0
simundump xcoredraw /edit/draw 0 -9 12 -2 9
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
addmsg /kinetics/dend/pumpCa /kinetics/dend/Ca REAC A B 
addmsg /kinetics/dend/turnOnKinase /kinetics/dend/Ca REAC A B 
addmsg /kinetics/dend/pumpCa /kinetics/dend/bufCa REAC B A 
addmsg /kinetics/dend/Ca /kinetics/dend/pumpCa SUBSTRATE n 
addmsg /kinetics/dend/bufCa /kinetics/dend/pumpCa PRODUCT n 
addmsg /kinetics/dend/Ca.kinase/enz /kinetics/dend/kChan REAC sA B 
addmsg /kinetics/dend/phosphatase /kinetics/dend/kChan REAC B A 
addmsg /kinetics/dend/Ca.kinase/enz /kinetics/dend/kChan_p MM_PRD pA 
addmsg /kinetics/dend/phosphatase /kinetics/dend/kChan_p REAC A B 
addmsg /kinetics/dend/kChan_p /kinetics/dend/phosphatase SUBSTRATE n 
addmsg /kinetics/dend/kChan /kinetics/dend/phosphatase PRODUCT n 
addmsg /kinetics/dend/Ca.kinase/enz /kinetics/dend/Ca.kinase REAC eA B 
addmsg /kinetics/dend/turnOnKinase /kinetics/dend/Ca.kinase REAC B A 
addmsg /kinetics/dend/Ca.kinase /kinetics/dend/Ca.kinase/enz ENZYME n 
addmsg /kinetics/dend/kChan /kinetics/dend/Ca.kinase/enz SUBSTRATE n 
addmsg /kinetics/dend/inact_kinase /kinetics/dend/turnOnKinase SUBSTRATE n 
addmsg /kinetics/dend/Ca /kinetics/dend/turnOnKinase SUBSTRATE n 
addmsg /kinetics/dend/Ca.kinase /kinetics/dend/turnOnKinase PRODUCT n 
addmsg /kinetics/dend/turnOnKinase /kinetics/dend/inact_kinase REAC A B 
addmsg /kinetics/spine/toPsd/enz /kinetics/spine/headGluR REAC sA B 
addmsg /kinetics/psd/fromPsd /kinetics/spine/headGluR REAC B A 
addmsg /kinetics/spine/toPsd/enz /kinetics/spine/toPsd REAC eA B 
addmsg /kinetics/spine/turnOnPsd /kinetics/spine/toPsd REAC B A 
addmsg /kinetics/spine/toPsd /kinetics/spine/toPsd/enz ENZYME n 
addmsg /kinetics/spine/headGluR /kinetics/spine/toPsd/enz SUBSTRATE n 
addmsg /kinetics/spine/turnOnPsd /kinetics/spine/Ca REAC A B 
addmsg /kinetics/spine/turnOnPsd /kinetics/spine/toPsdInact REAC A B 
addmsg /kinetics/spine/toPsdInact /kinetics/spine/turnOnPsd SUBSTRATE n 
addmsg /kinetics/spine/Ca /kinetics/spine/turnOnPsd SUBSTRATE n 
addmsg /kinetics/spine/toPsd /kinetics/spine/turnOnPsd PRODUCT n 
addmsg /kinetics/spine/toPsd/enz /kinetics/psd/psdGluR MM_PRD pA 
addmsg /kinetics/psd/fromPsd /kinetics/psd/psdGluR REAC A B 
addmsg /kinetics/psd/psdGluR /kinetics/psd/fromPsd SUBSTRATE n 
addmsg /kinetics/spine/headGluR /kinetics/psd/fromPsd PRODUCT n 
addmsg /kinetics/dend/Ca.kinase /graphs/conc1/Ca.kinase.Co PLOT Co *Ca.kinase.Co *25 
addmsg /kinetics/dend/kChan /graphs/conc1/kChan.Co PLOT Co *kChan.Co *7 
addmsg /kinetics/dend/kChan_p /graphs/conc1/kChan_p.Co PLOT Co *kChan_p.Co *4 
addmsg /kinetics/spine/headGluR /graphs/conc2/headGluR.Co PLOT Co *headGluR.Co *47 
addmsg /kinetics/psd/psdGluR /graphs/conc2/psdGluR.Co PLOT Co *psdGluR.Co *46 
addmsg /kinetics/spine/toPsd /graphs/conc2/toPsd.Co PLOT Co *toPsd.Co *36 
addmsg /kinetics/spine/Ca /graphs/conc2/Ca.Co PLOT Co *Ca.Co *52 
enddump
// End of dump

complete_loading
