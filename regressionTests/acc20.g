//genesis
// kkit Version 8 flat dumpfile
 
// Saved on Mon Dec 24 18:33:04 2001
 
include kkit {argv 1}
 
FASTDT = 0.001
SIMDT = 0.01
CONTROLDT = 1
PLOTDT = 0.1
MAXTIME = 50
TRANSIENT_TIME = 10
VARIABLE_DT_FLAG = 1
DEFAULT_VOL = 1.6667e-21
VERSION = 8.0
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
simobjdump kpool CoTotal CoInit Co n nInit nTotal nMin vol slave_enable notes \
  xtree_fg_req xtree_textfg_req x y z
simobjdump kreac kf kb notes xtree_fg_req xtree_textfg_req x y z
simobjdump kenz CoComplexInit CoComplex nComplexInit nComplex vol k1 k2 k3 \
  keepconc usecomplex notes xtree_fg_req xtree_textfg_req link x y z
simobjdump stim level1 width1 delay1 level2 width2 delay2 baselevel trig_time \
  trig_mode notes xtree_fg_req xtree_textfg_req is_running x y z
simobjdump xtab input output alloced step_mode stepsize notes editfunc \
  xtree_fg_req xtree_textfg_req baselevel last_x last_y is_running x y z
simobjdump kchan perm gmax Vm is_active use_nernst notes xtree_fg_req \
  xtree_textfg_req x y z
simobjdump proto x y z
simobjdump linkinfo xtree_fg_req xtree_textfg_req uplink downlink x y z
simobjdump uplink xtree_fg_req xtree_textfg_req x y z
simobjdump downlink xtree_fg_req xtree_textfg_req x y z
simobjdump mirror notes xtree_fg_req x y z
simundump kpool /kinetics/CaM-Ca4 1 20 20 20 20 20 20 0 1 4 "" blue yellow \
  -15 -8 0
simundump group /kinetics/NOSphos 0 47 black x 0 1 "" NOSphos \
  /home2/bhalla/scripts/modules/NOSphos_0.g 0 0 0 -12 -10 0
simundump kpool /kinetics/NOSphos/nNOS 0 5 0.5 0.5 0.5 0.5 5 0 1 0 "" 27 47 \
  -15 -12 0
simundump kpool /kinetics/NOSphos/NOS* 0 0 0 0 0 0 0 0 1 0 "" blue 47 -25 -12 \
  0
simundump kreac /kinetics/NOSphos/Ca-CaMbind_nNOS 0 3.25 0.05 "" "" 47 -20 -9 \
  0
simundump kpool /kinetics/NOSphos/Ca-CaMnNOS 0 0 0 0 0 0 0 0 1 0 "" 57 47 -25 \
  -8 0
simundump kenz /kinetics/NOSphos/Ca-CaMnNOS/kenz 0 0 0 0 0 1 8.3335 66.668 \
  16.667 0 0 "" red 57 "" -28 -8 0
simundump kpool /kinetics/NOSphos/NO 0 0 0 0 0 0 0 0 1 0 "" 47 47 -26 -10 0
simundump kpool /kinetics/NOSphos/cit 0 0 0 0 0 0 0 0 1 0 "" 7 47 -30 -10 0
simundump kpool /kinetics/NOSphos/Larg 0 100 100 100 100 100 100 0 1 0 "" \
  blue 47 -30 -6 0
simundump kpool /kinetics/NOSphos/CaMKIV 0 1 1 1 1 1 1 0 1 4 "" 32 47 -12 -17 \
  0
simundump kenz /kinetics/NOSphos/CaMKIV/kenz 0 0 0 0 0 1 18 72 18 0 0 "" red \
  32 "" -12 -16 0
simundump kpool /kinetics/NOSphos/CaMKIIalpha 0 1 1 1 1 1 1 0 1 4 "" 0 47 -20 \
  -17 0
simundump kenz /kinetics/NOSphos/CaMKIIalpha/kenz 0 0 0 0 0 1 28.5 114 28.5 0 \
  0 "" red 0 "" -20 -16 0
simundump kpool /kinetics/NOSphos/CaMKIalpha 0 1 1 1 1 1 1 0 1 4 "" 62 47 -28 \
  -17 0
simundump kenz /kinetics/NOSphos/CaMKIalpha/kenz 0 0 0 0 0 1 17 68 17 0 0 "" \
  red 62 "" -28 -16 0
simundump kreac /kinetics/NOSphos/dephosporyl 0 13.9 0 "" white 47 -20 -11 0
simundump xgraph /graphs/conc1 0 0 49.991 0 0.43971 1
simundump xgraph /graphs/conc2 0 0 49.991 0 0.030793 1
simundump xplot /graphs/conc1/nNOS.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 27 0 0 1
simundump xplot /graphs/conc2/NOS*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xgraph /moregraphs/conc3 0 0 50 4.5265e-17 100 1
simundump xgraph /moregraphs/conc4 0 0 50 0 100 1
simundump xplot /moregraphs/conc3/Larg.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xplot /moregraphs/conc4/NO.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 47 0 0 1
simundump xcoredraw /edit/draw 0 -32 -10 -19 -4
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
xtextload /file/notes \
"Phosporylation of nNOS by CaM kinases I alpha, II alpha, and IV. " \
"(Hayashi et al., 1999, JBC,274(29):20597-20602.) " \
"This model features the phosphorylation of neuronal NOS alone by the " \
"above mentioned CaM Kinases."
addmsg /kinetics/NOSphos/Ca-CaMbind_nNOS /kinetics/CaM-Ca4 REAC A B 
addmsg /kinetics/NOSphos/Ca-CaMbind_nNOS /kinetics/NOSphos/nNOS REAC A B 
addmsg /kinetics/NOSphos/CaMKIV/kenz /kinetics/NOSphos/nNOS REAC sA B 
addmsg /kinetics/NOSphos/CaMKIIalpha/kenz /kinetics/NOSphos/nNOS REAC sA B 
addmsg /kinetics/NOSphos/CaMKIalpha/kenz /kinetics/NOSphos/nNOS REAC sA B 
addmsg /kinetics/NOSphos/dephosporyl /kinetics/NOSphos/nNOS REAC B A 
addmsg /kinetics/NOSphos/CaMKIV/kenz /kinetics/NOSphos/NOS* MM_PRD pA 
addmsg /kinetics/NOSphos/CaMKIIalpha/kenz /kinetics/NOSphos/NOS* MM_PRD pA 
addmsg /kinetics/NOSphos/CaMKIalpha/kenz /kinetics/NOSphos/NOS* MM_PRD pA 
addmsg /kinetics/NOSphos/dephosporyl /kinetics/NOSphos/NOS* REAC A B 
addmsg /kinetics/NOSphos/nNOS /kinetics/NOSphos/Ca-CaMbind_nNOS SUBSTRATE n 
addmsg /kinetics/CaM-Ca4 /kinetics/NOSphos/Ca-CaMbind_nNOS SUBSTRATE n 
addmsg /kinetics/NOSphos/Ca-CaMnNOS /kinetics/NOSphos/Ca-CaMbind_nNOS PRODUCT n 
addmsg /kinetics/NOSphos/Ca-CaMbind_nNOS /kinetics/NOSphos/Ca-CaMnNOS REAC B A 
addmsg /kinetics/NOSphos/Ca-CaMnNOS/kenz /kinetics/NOSphos/Ca-CaMnNOS REAC eA B 
addmsg /kinetics/NOSphos/Ca-CaMnNOS /kinetics/NOSphos/Ca-CaMnNOS/kenz ENZYME n 
addmsg /kinetics/NOSphos/Larg /kinetics/NOSphos/Ca-CaMnNOS/kenz SUBSTRATE n 
addmsg /kinetics/NOSphos/Ca-CaMnNOS/kenz /kinetics/NOSphos/NO MM_PRD pA 
addmsg /kinetics/NOSphos/Ca-CaMnNOS/kenz /kinetics/NOSphos/cit MM_PRD pA 
addmsg /kinetics/NOSphos/Ca-CaMnNOS/kenz /kinetics/NOSphos/Larg REAC sA B 
addmsg /kinetics/NOSphos/CaMKIV/kenz /kinetics/NOSphos/CaMKIV REAC eA B 
addmsg /kinetics/NOSphos/CaMKIV /kinetics/NOSphos/CaMKIV/kenz ENZYME n 
addmsg /kinetics/NOSphos/nNOS /kinetics/NOSphos/CaMKIV/kenz SUBSTRATE n 
addmsg /kinetics/NOSphos/CaMKIIalpha/kenz /kinetics/NOSphos/CaMKIIalpha REAC eA B 
addmsg /kinetics/NOSphos/CaMKIIalpha /kinetics/NOSphos/CaMKIIalpha/kenz ENZYME n 
addmsg /kinetics/NOSphos/nNOS /kinetics/NOSphos/CaMKIIalpha/kenz SUBSTRATE n 
addmsg /kinetics/NOSphos/CaMKIalpha/kenz /kinetics/NOSphos/CaMKIalpha REAC eA B 
addmsg /kinetics/NOSphos/CaMKIalpha /kinetics/NOSphos/CaMKIalpha/kenz ENZYME n 
addmsg /kinetics/NOSphos/nNOS /kinetics/NOSphos/CaMKIalpha/kenz SUBSTRATE n 
addmsg /kinetics/NOSphos/NOS* /kinetics/NOSphos/dephosporyl SUBSTRATE n 
addmsg /kinetics/NOSphos/nNOS /kinetics/NOSphos/dephosporyl PRODUCT n 
addmsg /kinetics/NOSphos/nNOS /graphs/conc1/nNOS.Co PLOT Co *nNOS.Co *27 
addmsg /kinetics/NOSphos/NOS* /graphs/conc2/NOS*.Co PLOT Co *NOS*.Co *blue 
addmsg /kinetics/NOSphos/Larg /moregraphs/conc3/Larg.Co PLOT Co *Larg.Co *blue 
addmsg /kinetics/NOSphos/NO /moregraphs/conc4/NO.Co PLOT Co *NO.Co *47 
enddump
// End of dump

call /kinetics/NOSphos/nNOS/notes LOAD \
"neuronal Nitric Oxide Synthase." \
"Found in the neurons, activity is dependent on CaM binding," \
"in response to Ca levels." \
"" \
"Molecular weight ~160 kDa." \
"This is unphosporylated form of nNOS."
call /kinetics/NOSphos/NOS*/notes LOAD \
"Phosporylated NOS, by CaM kinases, with lowering of Vmax," \
"but little change in Km for Arg and Kact for CaM." \
"(Hayashi et al., JBC, 1999, 274(29):20597-20602)" \
""
call /kinetics/NOSphos/Ca-CaMbind_nNOS/notes LOAD \
"Those binding CaM have a high Kd, including nNOS, ~<=10nM." \
"The binding of CaM to nNOS has been demonstrated to act as the" \
"trigger necessary for electron transfer and catalytic activity." \
"(Marletta, Biochemistry, 1997;36:12337-12345)."
call /kinetics/NOSphos/Ca-CaMnNOS/kenz/notes LOAD \
"Km for purified NOS is estimated between 1 - 10 uM." \
"(Prog in Neurobiology, 2001, 64: 365-391)" \
"Vmax for unphosporylated NOS, the active form, is " \
"500-1500 nmol/nmol/min (Montellano et al., 1998, JBC,26(12):" \
"1185-1189)." \
"" \
"Hayashi et al., JBC, 1999, 274(29):20597-20602 report" \
"Vmax (nmol/min/mg) of nNOS Unphosporylated at 95.7 (+-) 4.2" \
"" \
""
call /kinetics/NOSphos/CaMKIV/notes LOAD \
"Activity is similar to CaMKIalpha, with " \
"~0.7 mol of 32p/mol of nNOS." \
"(Hayashi et al., 1999,JBC,274(29):20597-20602)"
call /kinetics/NOSphos/CaMKIV/kenz/notes LOAD \
"Hayashi et al., 1999, JBC,274(29):20597-20602." \
"and other reported data from different sources."
call /kinetics/NOSphos/CaMKIIalpha/notes LOAD \
"CaMKIIalpha caused the most rapid phosphorylation of nNOS, with" \
"half-maximal phosphorylation apparent at 3 min and plateau" \
"level at 10 min." \
"(Hayashi et al., 1999,JBC,274(29):20597-20602)." \
"" \
"The Maximal Phosphorylation of nNOS was observed at " \
"~0.4 mol of 32p/mol of nNOS under their expt conditions."
call /kinetics/NOSphos/CaMKIIalpha/kenz/notes LOAD \
"Hayashi et al., 1999, JBC,274(29):20597-20602." \
"and from various other literature datas."
call /kinetics/NOSphos/CaMKIalpha/notes LOAD \
"Phosphorylates nNOS, but not as effective as CaMKIIalpha." \
"(Hayashi et al., 1999,JBC,274(29):20597-20602.)" \
"Report of Plateau Level in their phosphorylation plots reaching" \
"after 100 min.  -- ~0.4 mol of 32p/mol of nNOS." \
""
call /kinetics/NOSphos/CaMKIalpha/kenz/notes LOAD \
"enzyme parameters used from different literature." \
"Hayashi et al., 1999, JBC,274(29):20597-20602."
call /kinetics/NOSphos/dephosporyl/notes LOAD \
"kf -13.9" \
"These rates used to keep the basal level of nNOS at reasonable" \
"experimental levels."
complete_loading
