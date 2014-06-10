//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Sun Jan 22 17:01:52 2006
 
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
simundump geometry /kinetics/geometry 0 1e-19 3 sphere "" white black -4 5 0
simundump geometry /kinetics/geometry[1] 0 1e-18 3 sphere "" white black 3 5 \
  0
simundump kpool /kinetics/chan 0 0 0 0 0 0 0 0 60 1 /kinetics/geometry blue \
  black 0 0 0
simundump kchan /kinetics/chan/kchan 0 1 0.1 0 1 0 "" brown blue 0 1 0
simundump kpool /kinetics/in 0 0 0 0 0 0 0 0 60 0 /kinetics/geometry 61 black \
  -3 3 0
simundump kpool /kinetics/out 0 0 0.0016667 0.0016667 1 1 0 0 600 0 \
  /kinetics/geometry[1] 1 black 3 3 0
simundump xtab /kinetics/xtab 0 0 0 1 1 0 "" edit_xtab white red 0 0 0 3 3 -2 \
  0
loadtab /kinetics/xtab table 1 100 0 100 \
 1 1.5878 1.951 1.951 1.5878 0.99999 0.41222 0.048953 0.048929 0.41224 1 \
 1.5878 1.951 1.9511 1.5878 0.99998 0.41229 0.049072 0.049026 0.41217 0.99963 \
 1.588 1.9511 1.9512 1.5875 0.99996 0.41244 0.04882 0.048969 0.41202 1.0004 \
 1.5879 1.951 1.9512 1.5876 1.0001 0.41259 0.048878 0.048911 0.41187 1.0003 \
 1.5877 1.9509 1.951 1.5878 1.0003 0.41193 0.048934 0.048854 0.41253 1.0001 \
 1.5876 1.9512 1.951 1.5879 1.0005 0.41208 0.048992 0.048797 0.41238 0.99989 \
 1.5874 1.9511 1.9511 1.5881 0.9997 0.41223 0.049049 0.049049 0.41223 0.9997 \
 1.5881 1.9511 1.9511 1.5874 0.99989 0.41238 0.049106 0.048991 0.41208 \
 0.99952 1.5879 1.951 1.9512 1.5876 1.0001 0.41253 0.048854 0.048934 0.41193 \
 1.0003 1.5878 1.951 1.9509 1.5877 1.0003 0.41187 0.048912 0.048877 0.41178 \
 1.0001
simundump kreac /kinetics/pump 0 1 0 "" white black 0 5 0
simundump xgraph /graphs/conc1 0 0 100 0 0.0015994 0
simundump xgraph /graphs/conc2 0 0 99 0.00081328 0.03252 0
simundump xplot /graphs/conc1/in.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 61 0 0 1
simundump xplot /graphs/conc1/out.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 1 0 0 1
simundump xplot /graphs/conc2/chan.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xgraph /moregraphs/conc3 0 0 100 0 1 0
simundump xgraph /moregraphs/conc4 0 0 100 0 1 0
simundump xcoredraw /edit/draw 0 -6 5 -4 7
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
addmsg /kinetics/xtab /kinetics/chan SLAVE output 
addmsg /kinetics/chan /kinetics/chan/kchan NUMCHAN n 
addmsg /kinetics/out /kinetics/chan/kchan SUBSTRATE n vol 
addmsg /kinetics/in /kinetics/chan/kchan PRODUCT n vol 
addmsg /kinetics/chan/kchan /kinetics/in REAC B A 
addmsg /kinetics/pump /kinetics/in REAC A B 
addmsg /kinetics/chan/kchan /kinetics/out REAC A B 
addmsg /kinetics/pump /kinetics/out REAC B A 
addmsg /kinetics/in /kinetics/pump SUBSTRATE n 
addmsg /kinetics/out /kinetics/pump PRODUCT n 
addmsg /kinetics/in /graphs/conc1/in.Co PLOT Co *in.Co *61 
addmsg /kinetics/out /graphs/conc1/out.Co PLOT Co *out.Co *1 
addmsg /kinetics/chan /graphs/conc2/chan.Co PLOT Co *chan.Co *blue 
enddump
// End of dump

setfield /kinetics/xtab table->dx 1
setfield /kinetics/xtab table->invdx 1
complete_loading
