//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Sat Dec 17 21:43:59 2011
 
include kkit {argv 1}
 
FASTDT = 1e-05
SIMDT = 0.001
CONTROLDT = 10
PLOTDT = 10
MAXTIME = 500
TRANSIENT_TIME = 2
VARIABLE_DT_FLAG = 1
DEFAULT_VOL = 9e-20
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
simundump geometry /kinetics/geometry 0 1e-20 3 sphere "" white black 0 0 0
simundump text /kinetics/notes 0 ""
call /kinetics/notes LOAD \
""
simundump text /kinetics/geometry/notes 0 ""
call /kinetics/geometry/notes LOAD \
""
simundump group /kinetics/PSD 0 0 black x 0 1 "" PSD defaultfile.g 0 0 0 -10 \
  -3 0
simundump text /kinetics/PSD/notes 0 ""
call /kinetics/PSD/notes LOAD \
""
simundump kpool /kinetics/PSD/tot_PSD_R 0 0 0 0 0 0 0 0 6 0 \
  /kinetics/geometry blue 0 -4 7 0
simundump text /kinetics/PSD/tot_PSD_R/notes 0 ""
call /kinetics/PSD/tot_PSD_R/notes LOAD \
""
simundump kpool /kinetics/PSD/CaM_CaN 0 0 0.01 0.01 0.06 0.06 0 0 6 0 \
  /kinetics/geometry blue 0 -4 -5 0
simundump text /kinetics/PSD/CaM_CaN/notes 0 ""
call /kinetics/PSD/CaM_CaN/notes LOAD \
""
simundump kenz /kinetics/PSD/CaM_CaN/CaN1 0 0 0 0 0 6 0.041667 2 0.5 0 0 "" \
  red blue "" -6 -4 0
simundump text /kinetics/PSD/CaM_CaN/CaN1/notes 0 ""
call /kinetics/PSD/CaM_CaN/CaN1/notes LOAD \
""
simundump kenz /kinetics/PSD/CaM_CaN/CaN2 0 0 0 0 0 6 0.041667 2 0.5 0 0 "" \
  red blue "" -2 -4 0
simundump text /kinetics/PSD/CaM_CaN/CaN2/notes 0 ""
call /kinetics/PSD/CaM_CaN/CaN2/notes LOAD \
""
simundump kpool /kinetics/PSD/PP2A 0 0 0.11111 0.11111 0.66666 0.66666 0 0 6 \
  0 /kinetics/geometry 61 0 -4 -2 0
simundump text /kinetics/PSD/PP2A/notes 0 ""
call /kinetics/PSD/PP2A/notes LOAD \
""
simundump kenz /kinetics/PSD/PP2A/P1 0 0 0 0 0 60 8.3333 4 1 0 0 "" red 61 "" \
  -6 -1 0
simundump text /kinetics/PSD/PP2A/P1/notes 0 ""
call /kinetics/PSD/PP2A/P1/notes LOAD \
""
simundump kenz /kinetics/PSD/PP2A/P2 0 0 0 0 0 60 8.3333 4 1 0 0 "" red 61 "" \
  -2 -1 0
simundump text /kinetics/PSD/PP2A/P2/notes 0 ""
call /kinetics/PSD/PP2A/P2/notes LOAD \
""
simundump kpool /kinetics/PSD/actCaMKII 0 0 2.1 2.1 12.6 12.6 0 0 6 0 \
  /kinetics/geometry 35 0 0 7 0
simundump text /kinetics/PSD/actCaMKII/notes 0 ""
call /kinetics/PSD/actCaMKII/notes LOAD \
""
simundump kenz /kinetics/PSD/actCaMKII/CaMKII_1 0 0 0 0 0 6 0.016666 4 1 0 0 \
  "" red 35 "" -6 3 0
simundump text /kinetics/PSD/actCaMKII/CaMKII_1/notes 0 ""
call /kinetics/PSD/actCaMKII/CaMKII_1/notes LOAD \
""
simundump kenz /kinetics/PSD/actCaMKII/CaMKII_2 0 0 0 0 0 6 0.016666 4 1 0 0 \
  "" red 35 "" -2 3 0
simundump text /kinetics/PSD/actCaMKII/CaMKII_2/notes 0 ""
call /kinetics/PSD/actCaMKII/CaMKII_2/notes LOAD \
""
simundump group /kinetics/PSD/PP1_PSD 0 yellow 0 x 0 0 "" PP1_PSD \
  defaultfile.g 0 0 0 7 10 0
simundump text /kinetics/PSD/PP1_PSD/notes 0 ""
call /kinetics/PSD/PP1_PSD/notes LOAD \
""
simundump kreac /kinetics/PSD/move_to_PSD 0 0.0625 1 "" white 0 -8 -5 0
simundump text /kinetics/PSD/move_to_PSD/notes 0 ""
call /kinetics/PSD/move_to_PSD/notes LOAD \
""
simundump kpool /kinetics/PSD/R_S2 0 0 0 0 0 0 0 0 6 0 /kinetics/geometry 4 0 \
  -8 1 0
simundump text /kinetics/PSD/R_S2/notes 0 ""
call /kinetics/PSD/R_S2/notes LOAD \
""
simundump kpool /kinetics/PSD/R_SpS 0 0 0 0 0 0 0 0 6 0 /kinetics/geometry \
  blue 0 -4 1 0
simundump text /kinetics/PSD/R_SpS/notes 0 ""
call /kinetics/PSD/R_SpS/notes LOAD \
""
simundump kpool /kinetics/PSD/R_SpSp 0 0 0 0 0 0 0 0 6 0 /kinetics/geometry \
  28 0 0 1 0
simundump text /kinetics/PSD/R_SpSp/notes 0 ""
call /kinetics/PSD/R_SpSp/notes LOAD \
""
simundump group /kinetics/BULK 0 yellow black x 0 0 "" BULK defaultfile.g 0 0 \
  0 -6 -16 0
simundump text /kinetics/BULK/notes 0 ""
call /kinetics/BULK/notes LOAD \
""
simundump kpool /kinetics/BULK/iR 0 0 2.7778 2.7778 150 150 0 0 54 0 \
  /kinetics/geometry 0 yellow -8 -10 0
simundump text /kinetics/BULK/iR/notes 0 "Same as Fus3\n"
call /kinetics/BULK/iR/notes LOAD \
"Same as Fus3" \
""
simundump xgraph /graphs/conc1 0 0 3550.3 -2.2424 20 0
simundump xgraph /graphs/conc2 0 0 3550.3 0 16 0
simundump xplot /graphs/conc1/tot_PSD_R.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xplot /graphs/conc1/R.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 4 0 0 1
simundump xplot /graphs/conc2/iR.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 0 0 0 1
simundump xplot /graphs/conc2/Rpp.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 28 0 0 1
simundump xgraph /moregraphs/conc3 0 0 3550.3 0 10 0
simundump xgraph /moregraphs/conc4 0 0 3550.3 0 10 0
simundump xcoredraw /edit/draw 0 -12 9 -18 12
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
xtextload /file/notes \
"16 Dec 2011. Completely new PSD version, where the R, Rp and Rpp" \
"all refer to AMPAR bound to two stargazins, and each p is on one" \
"of the stargazins. " \
"" \
"stargazin_psd2.g: tweaked traffic parameters to get bistability." \
"Very bistable." \
"" \
"17 Dec 2011. stargazin_psd3.g: Minor change: reduced kcat for" \
"CaN from 10 to 2." \
"" \
"stargazin_psd4.g: Changed some rates around so that it responds" \
"in a more relevant range for CaMKII activation in the full model." \
"Also renamed the Receptor-Stargazin pools to clarify that the" \
"phospho steps are on the stargazin." \
"" \
"stargazin_psd5.g: Incorporated traffic rates from the " \
"traffillator analysis." \
"" \
"stargazin_psd6.g: Raised CaN Km from 5 to 10 uM." \
""
addmsg /kinetics/PSD/R_S2 /kinetics/PSD/tot_PSD_R SUMTOTAL n nInit 
addmsg /kinetics/PSD/R_SpS /kinetics/PSD/tot_PSD_R SUMTOTAL n nInit 
addmsg /kinetics/PSD/R_SpSp /kinetics/PSD/tot_PSD_R SUMTOTAL n nInit 
addmsg /kinetics/PSD/CaM_CaN/CaN2 /kinetics/PSD/CaM_CaN REAC eA B 
addmsg /kinetics/PSD/CaM_CaN/CaN1 /kinetics/PSD/CaM_CaN REAC eA B 
addmsg /kinetics/PSD/CaM_CaN /kinetics/PSD/CaM_CaN/CaN1 ENZYME n 
addmsg /kinetics/PSD/R_SpS /kinetics/PSD/CaM_CaN/CaN1 SUBSTRATE n 
addmsg /kinetics/PSD/CaM_CaN /kinetics/PSD/CaM_CaN/CaN2 ENZYME n 
addmsg /kinetics/PSD/R_SpSp /kinetics/PSD/CaM_CaN/CaN2 SUBSTRATE n 
addmsg /kinetics/PSD/PP2A/P1 /kinetics/PSD/PP2A REAC eA B 
addmsg /kinetics/PSD/PP2A/P2 /kinetics/PSD/PP2A REAC eA B 
addmsg /kinetics/PSD/PP2A /kinetics/PSD/PP2A/P1 ENZYME n 
addmsg /kinetics/PSD/R_SpS /kinetics/PSD/PP2A/P1 SUBSTRATE n 
addmsg /kinetics/PSD/PP2A /kinetics/PSD/PP2A/P2 ENZYME n 
addmsg /kinetics/PSD/R_SpSp /kinetics/PSD/PP2A/P2 SUBSTRATE n 
addmsg /kinetics/PSD/actCaMKII/CaMKII_1 /kinetics/PSD/actCaMKII REAC eA B 
addmsg /kinetics/PSD/actCaMKII/CaMKII_2 /kinetics/PSD/actCaMKII REAC eA B 
addmsg /kinetics/PSD/actCaMKII /kinetics/PSD/actCaMKII/CaMKII_1 ENZYME n 
addmsg /kinetics/PSD/R_S2 /kinetics/PSD/actCaMKII/CaMKII_1 SUBSTRATE n 
addmsg /kinetics/PSD/actCaMKII /kinetics/PSD/actCaMKII/CaMKII_2 ENZYME n 
addmsg /kinetics/PSD/R_SpS /kinetics/PSD/actCaMKII/CaMKII_2 SUBSTRATE n 
addmsg /kinetics/BULK/iR /kinetics/PSD/move_to_PSD SUBSTRATE n 
addmsg /kinetics/PSD/R_S2 /kinetics/PSD/move_to_PSD PRODUCT n 
addmsg /kinetics/PSD/PP2A/P1 /kinetics/PSD/R_S2 MM_PRD pA 
addmsg /kinetics/PSD/CaM_CaN/CaN1 /kinetics/PSD/R_S2 MM_PRD pA 
addmsg /kinetics/PSD/move_to_PSD /kinetics/PSD/R_S2 REAC B A 
addmsg /kinetics/PSD/actCaMKII/CaMKII_1 /kinetics/PSD/R_S2 REAC sA B 
addmsg /kinetics/PSD/PP2A/P2 /kinetics/PSD/R_SpS MM_PRD pA 
addmsg /kinetics/PSD/PP2A/P1 /kinetics/PSD/R_SpS REAC sA B 
addmsg /kinetics/PSD/CaM_CaN/CaN2 /kinetics/PSD/R_SpS MM_PRD pA 
addmsg /kinetics/PSD/CaM_CaN/CaN1 /kinetics/PSD/R_SpS REAC sA B 
addmsg /kinetics/PSD/actCaMKII/CaMKII_1 /kinetics/PSD/R_SpS MM_PRD pA 
addmsg /kinetics/PSD/actCaMKII/CaMKII_2 /kinetics/PSD/R_SpS REAC sA B 
addmsg /kinetics/PSD/PP2A/P2 /kinetics/PSD/R_SpSp REAC sA B 
addmsg /kinetics/PSD/CaM_CaN/CaN2 /kinetics/PSD/R_SpSp REAC sA B 
addmsg /kinetics/PSD/actCaMKII/CaMKII_2 /kinetics/PSD/R_SpSp MM_PRD pA 
addmsg /kinetics/PSD/move_to_PSD /kinetics/BULK/iR REAC A B 
addmsg /kinetics/PSD/tot_PSD_R /graphs/conc1/tot_PSD_R.Co PLOT Co *tot_PSD_R.Co *blue 
addmsg /kinetics/PSD/R_S2 /graphs/conc1/R.Co PLOT Co *R.Co *4 
addmsg /kinetics/BULK/iR /graphs/conc2/iR.Co PLOT Co *iR.Co *0 
addmsg /kinetics/PSD/R_SpSp /graphs/conc2/Rpp.Co PLOT Co *Rpp.Co *28 
enddump
// End of dump

call /kinetics/BULK/iR/notes LOAD \
"Same as Fus3" \
""
complete_loading
