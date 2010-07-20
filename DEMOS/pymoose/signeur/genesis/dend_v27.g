//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Tue Feb 10 15:37:36 2009
 
include kkit {argv 1}
 
FASTDT = 0.0001
SIMDT = 0.001
CONTROLDT = 1
PLOTDT = 1
MAXTIME = 2000
TRANSIENT_TIME = 2
VARIABLE_DT_FLAG = 1
DEFAULT_VOL = 1e-15
VERSION = 11.0
setfield /file/modpath value /home2/bhalla/scripts/modules
kparms
 
//genesis
echo "dend"
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
simundump geometry /kinetics/geometry 0 1e-15 3 sphere "" white black 0 0 0
simundump geometry /kinetics/geometry[1] 0 1.6667e-21 3 sphere "" white black \
  0 0 0
simundump geometry /kinetics/geometry[2] 0 1e-15 3 sphere "" white black 0 0 \
  0
simundump text /kinetics/notes 0 ""
call /kinetics/notes LOAD \
""
simundump text /kinetics/geometry/notes 0 ""
call /kinetics/geometry/notes LOAD \
""
simundump text /kinetics/geometry[1]/notes 0 ""
call /kinetics/geometry[1]/notes LOAD \
""
simundump text /kinetics/geometry[2]/notes 0 ""
call /kinetics/geometry[2]/notes LOAD \
""
simundump group /kinetics/PKC 0 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 -3.0493 8.2163 0
simundump text /kinetics/PKC/notes 0 ""
call /kinetics/PKC/notes LOAD \
""
simundump kpool /kinetics/PKC/PKC-Ca 0 0 3.7208e-17 3.7208e-17 2.2325e-11 \
  2.2325e-11 0 0 6e+05 0 /kinetics/geometry red black -4.0752 1.5108 0
simundump text /kinetics/PKC/PKC-Ca/notes 0 ""
call /kinetics/PKC/PKC-Ca/notes LOAD \
""
simundump kreac /kinetics/PKC/PKC-act-by-Ca 0 1e-06 0.5 "" white blue -4.0752 \
  -0.12295 0
simundump text /kinetics/PKC/PKC-act-by-Ca/notes 0 \
  "Need est of rate of assoc of Ca and PKC. Assume it is fast\nThe original parameter-searched kf of 439.4 has been\nscaled by 1/6e8 to account for change of units to n. Kf now 8.16e-7, kb=.6085\nRaised kf to 1e-6 to match Ca curve, kb to .5\n"
call /kinetics/PKC/PKC-act-by-Ca/notes LOAD \
"Need est of rate of assoc of Ca and PKC. Assume it is fast" \
"The original parameter-searched kf of 439.4 has been" \
"scaled by 1/6e8 to account for change of units to n. Kf now 8.16e-7, kb=.6085" \
"Raised kf to 1e-6 to match Ca curve, kb to .5" \
""
simundump kreac /kinetics/PKC/PKC-act-by-DAG 0 1.3333e-08 8.6348 "" white \
  blue -2.0612 0.69395 0
simundump text /kinetics/PKC/PKC-act-by-DAG/notes 0 \
  "Need est of rate. Assume it is fast\nObtained from param search\nkf raised 10 X : see Shinomura et al PNAS 88 5149-5153 1991.\nkf changed from 3.865e-7 to 2.0e-7 in line with closer analysis of\nShinomura data.\n26 June 1996: Corrected DAG data: reduce kf 15x from \n2e-7 to 1.333e-8"
call /kinetics/PKC/PKC-act-by-DAG/notes LOAD \
"Need est of rate. Assume it is fast" \
"Obtained from param search" \
"kf raised 10 X : see Shinomura et al PNAS 88 5149-5153 1991." \
"kf changed from 3.865e-7 to 2.0e-7 in line with closer analysis of" \
"Shinomura data." \
"26 June 1996: Corrected DAG data: reduce kf 15x from " \
"2e-7 to 1.333e-8"
simundump kreac /kinetics/PKC/PKC-Ca-to-memb 0 1.2705 3.5026 "" white blue \
  -3.7974 4.2533 0
simundump text /kinetics/PKC/PKC-Ca-to-memb/notes 0 ""
call /kinetics/PKC/PKC-Ca-to-memb/notes LOAD \
""
simundump kreac /kinetics/PKC/PKC-DAG-to-memb 0 1 0.1 "" white blue -2.6168 \
  2.7362 0
simundump text /kinetics/PKC/PKC-DAG-to-memb/notes 0 \
  "Raise kb from .087 to 0.1 to match data from Shinomura et al.\nLower kf from 1.155 to 1.0 to match data from Shinomura et al."
call /kinetics/PKC/PKC-DAG-to-memb/notes LOAD \
"Raise kb from .087 to 0.1 to match data from Shinomura et al." \
"Lower kf from 1.155 to 1.0 to match data from Shinomura et al."
simundump kreac /kinetics/PKC/PKC-act-by-Ca-AA 0 2e-09 0.1 "" white blue \
  -0.78797 3.8157 0
simundump text /kinetics/PKC/PKC-act-by-Ca-AA/notes 0 \
  "Schaechter and Benowitz We have to increase Kf for conc scaling\nChanged kf to 2e-9 on Sept 19, 94. This gives better match.\n"
call /kinetics/PKC/PKC-act-by-Ca-AA/notes LOAD \
"Schaechter and Benowitz We have to increase Kf for conc scaling" \
"Changed kf to 2e-9 on Sept 19, 94. This gives better match." \
""
simundump kreac /kinetics/PKC/PKC-act-by-DAG-AA 0 2 0.2 "" white blue 1.2492 \
  3.2322 0
simundump text /kinetics/PKC/PKC-act-by-DAG-AA/notes 0 \
  "Assume slowish too. Schaechter and Benowitz"
call /kinetics/PKC/PKC-act-by-DAG-AA/notes LOAD \
"Assume slowish too. Schaechter and Benowitz"
simundump kpool /kinetics/PKC/PKC-DAG-AA* 0 0 4.9137e-18 4.9137e-18 \
  2.9482e-12 2.9482e-12 0 0 6e+05 0 /kinetics/geometry cyan blue 0.60098 \
  5.537 0
simundump text /kinetics/PKC/PKC-DAG-AA*/notes 0 ""
call /kinetics/PKC/PKC-DAG-AA*/notes LOAD \
""
simundump kpool /kinetics/PKC/PKC-Ca-AA* 0 0 1.75e-16 1.75e-16 1.05e-10 \
  1.05e-10 0 0 6e+05 0 /kinetics/geometry orange blue -0.60278 6.2956 0
simundump text /kinetics/PKC/PKC-Ca-AA*/notes 0 ""
call /kinetics/PKC/PKC-Ca-AA*/notes LOAD \
""
simundump kpool /kinetics/PKC/PKC-Ca-memb* 0 0 1.3896e-17 1.3896e-17 \
  8.3376e-12 8.3376e-12 0 0 6e+05 0 /kinetics/geometry pink blue -2.7788 \
  6.529 0
simundump text /kinetics/PKC/PKC-Ca-memb*/notes 0 ""
call /kinetics/PKC/PKC-Ca-memb*/notes LOAD \
""
simundump kpool /kinetics/PKC/PKC-DAG-memb* 0 0 9.4352e-21 9.4352e-21 \
  5.6611e-15 5.6611e-15 0 0 6e+05 0 /kinetics/geometry yellow blue -1.8297 \
  5.5078 0
simundump text /kinetics/PKC/PKC-DAG-memb*/notes 0 ""
call /kinetics/PKC/PKC-DAG-memb*/notes LOAD \
""
simundump kpool /kinetics/PKC/PKC-basal* 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry pink blue -4.7465 5.5662 0
simundump text /kinetics/PKC/PKC-basal*/notes 0 ""
call /kinetics/PKC/PKC-basal*/notes LOAD \
""
simundump kreac /kinetics/PKC/PKC-basal-act 0 1 50 "" white blue -4.978 \
  3.0571 0
simundump text /kinetics/PKC/PKC-basal-act/notes 0 \
  "Initial basal levels were set by kf = 1, kb = 20. In model, though, the\nbasal levels of PKC activity are higher."
call /kinetics/PKC/PKC-basal-act/notes LOAD \
"Initial basal levels were set by kf = 1, kb = 20. In model, though, the" \
"basal levels of PKC activity are higher."
simundump kpool /kinetics/PKC/PKC-AA* 0 0 1.8133e-17 1.8133e-17 1.088e-11 \
  1.088e-11 0 0 6e+05 0 /kinetics/geometry cyan blue 1.7816 6.8207 0
simundump text /kinetics/PKC/PKC-AA*/notes 0 ""
call /kinetics/PKC/PKC-AA*/notes LOAD \
""
simundump kreac /kinetics/PKC/PKC-act-by-AA 0 2e-10 0.1 "" white blue -4.9925 \
  -1.8654 0
simundump text /kinetics/PKC/PKC-act-by-AA/notes 0 \
  "Raise kf from 1.667e-10 to 2e-10 to get better match to data."
call /kinetics/PKC/PKC-act-by-AA/notes LOAD \
"Raise kf from 1.667e-10 to 2e-10 to get better match to data."
simundump kpool /kinetics/PKC/PKC-Ca-DAG 0 0 8.4632e-23 8.4632e-23 5.0779e-17 \
  5.0779e-17 0 0 6e+05 0 /kinetics/geometry white blue 0.2306 1.8026 0
simundump text /kinetics/PKC/PKC-Ca-DAG/notes 0 ""
call /kinetics/PKC/PKC-Ca-DAG/notes LOAD \
""
simundump kreac /kinetics/PKC/PKC-n-DAG 0 1e-09 0.1 "" white blue -3.0103 \
  -1.9902 0
simundump text /kinetics/PKC/PKC-n-DAG/notes 0 \
  "kf raised 10 X based on Shinomura et al PNAS 88 5149-5153 1991\ncloser analysis of Shinomura et al: kf now 1e-8 (was 1.66e-8).\nFurther tweak. To get sufficient AA synergy, increase kf to 1.5e-8\n26 June 1996: Corrected DAG levels: reduce kf by 15x from\n1.5e-8 to 1e-9"
call /kinetics/PKC/PKC-n-DAG/notes LOAD \
"kf raised 10 X based on Shinomura et al PNAS 88 5149-5153 1991" \
"closer analysis of Shinomura et al: kf now 1e-8 (was 1.66e-8)." \
"Further tweak. To get sufficient AA synergy, increase kf to 1.5e-8" \
"26 June 1996: Corrected DAG levels: reduce kf by 15x from" \
"1.5e-8 to 1e-9"
simundump kpool /kinetics/PKC/PKC-DAG 0 0 1.161e-16 1.161e-16 6.9661e-11 \
  6.9661e-11 0 0 6e+05 0 /kinetics/geometry white blue -0.99631 -1.0857 0
simundump text /kinetics/PKC/PKC-DAG/notes 0 "CoInit was .0624\n"
call /kinetics/PKC/PKC-DAG/notes LOAD \
"CoInit was .0624" \
""
simundump kreac /kinetics/PKC/PKC-n-DAG-AA 0 3e-08 2 "" white blue -1.2278 \
  -2.9529 0
simundump text /kinetics/PKC/PKC-n-DAG-AA/notes 0 \
  "Reduced kf to 0.66X to match Shinomura et al data.\nInitial: kf = 3.3333e-9\nNew: 2e-9\nNewer: 2e-8\nkb was 0.2\nnow 2."
call /kinetics/PKC/PKC-n-DAG-AA/notes LOAD \
"Reduced kf to 0.66X to match Shinomura et al data." \
"Initial: kf = 3.3333e-9" \
"New: 2e-9" \
"Newer: 2e-8" \
"kb was 0.2" \
"now 2."
simundump kpool /kinetics/PKC/PKC-DAG-AA 0 0 2.5188e-19 2.5188e-19 1.5113e-13 \
  1.5113e-13 0 0 6e+05 0 /kinetics/geometry white blue 0.62413 0.22715 0
simundump text /kinetics/PKC/PKC-DAG-AA/notes 0 ""
call /kinetics/PKC/PKC-DAG-AA/notes LOAD \
""
simundump kpool /kinetics/PKC/PKC-cytosolic 0 0 1 1 6e+05 6e+05 0 0 6e+05 0 \
  /kinetics/geometry white blue -6.1315 0.59711 0
simundump text /kinetics/PKC/PKC-cytosolic/notes 0 \
  "Marquez et al J. Immun 149,2560(92) est 1e6/cell for chromaffin cells\nWe will use 1 uM as our initial concen\n"
call /kinetics/PKC/PKC-cytosolic/notes LOAD \
"Marquez et al J. Immun 149,2560(92) est 1e6/cell for chromaffin cells" \
"We will use 1 uM as our initial concen" \
""
simundump kpool /kinetics/DAG 1 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry \
  green black 0.71777 -7.5617 0
simundump text /kinetics/DAG/notes 0 ""
call /kinetics/DAG/notes LOAD \
""
simundump kpool /kinetics/AA 0 0 12.2 12.2 7.32e+06 7.32e+06 0 0 6e+05 0 \
  /kinetics/geometry darkgreen black -3.2898 -9.3376 0
simundump text /kinetics/AA/notes 0 ""
call /kinetics/AA/notes LOAD \
""
simundump kpool /kinetics/PKC-active 1 0 0.02 2.1195e-16 1.2717e-10 12000 0 0 \
  6e+05 2 /kinetics/geometry red black -0.29091 9.2668 0
simundump text /kinetics/PKC-active/notes 0 ""
call /kinetics/PKC-active/notes LOAD \
""
simundump kenz /kinetics/PKC-active/PKC-act-raf 1 0 0 0 0 6e+05 5e-07 16 4 0 \
  0 "" red yellow "" 7 10 0
simundump text /kinetics/PKC-active/PKC-act-raf/notes 0 \
  "Rate consts from Chen et al Biochem 32, 1032 (1993)\nk3 = k2 = 4\nk1 = 9e-5\nrecalculated gives 1.666e-5, which is not very different.\nLooks like k3 is rate-limiting in this case: there is a huge amount\nof craf locked up in the enz complex. Let us assume a 10x\nhigher Km, ie, lower affinity.  k1 drops by 10x.\nAlso changed k2 to 4x k3.\nLowerd k1 to 1e-6 to balance 10X DAG sensitivity of PKC"
call /kinetics/PKC-active/PKC-act-raf/notes LOAD \
"Rate consts from Chen et al Biochem 32, 1032 (1993)" \
"k3 = k2 = 4" \
"k1 = 9e-5" \
"recalculated gives 1.666e-5, which is not very different." \
"Looks like k3 is rate-limiting in this case: there is a huge amount" \
"of craf locked up in the enz complex. Let us assume a 10x" \
"higher Km, ie, lower affinity.  k1 drops by 10x." \
"Also changed k2 to 4x k3." \
"Lowerd k1 to 1e-6 to balance 10X DAG sensitivity of PKC"
simundump kenz /kinetics/PKC-active/PKC-inact-GAP 1 0 0 0 0 1 1e-05 16 4 0 0 \
  "" red yellow "" 4 13 0
simundump text /kinetics/PKC-active/PKC-inact-GAP/notes 0 \
  "Rate consts copied from PCK-act-raf\nThis reaction inactivates GAP. The idea is from the \nBoguski and McCormick review."
call /kinetics/PKC-active/PKC-inact-GAP/notes LOAD \
"Rate consts copied from PCK-act-raf" \
"This reaction inactivates GAP. The idea is from the " \
"Boguski and McCormick review."
simundump kenz /kinetics/PKC-active/PKC-act-GEF 1 0 0 0 0 1 1e-05 16 4 0 0 "" \
  red yellow "" 9 19 0
simundump text /kinetics/PKC-active/PKC-act-GEF/notes 0 \
  "Rate consts from PKC-act-raf.\nThis reaction activates GEF. It can lead to at least 2X stim of ras, and\na 2X stim of MAPK over and above that obtained via direct phosph of\nc-raf. Note that it is a push-pull reaction, and there is also a contribution\nthrough the phosphorylation and inactivation of GAPs.\nThe original PKC-act-raf rate consts are too fast. We lower K1 by 10 X"
call /kinetics/PKC-active/PKC-act-GEF/notes LOAD \
"Rate consts from PKC-act-raf." \
"This reaction activates GEF. It can lead to at least 2X stim of ras, and" \
"a 2X stim of MAPK over and above that obtained via direct phosph of" \
"c-raf. Note that it is a push-pull reaction, and there is also a contribution" \
"through the phosphorylation and inactivation of GAPs." \
"The original PKC-act-raf rate consts are too fast. We lower K1 by 10 X"
simundump kenz /kinetics/PKC-active/phosph-AC2 1 0 0 0 0 6e+05 1e-06 16 4 0 0 \
  "" red red "" -16 -23 0
simundump text /kinetics/PKC-active/phosph-AC2/notes 0 \
  "Phorbol esters have little effect on AC1 or on the Gs-stimulation of\nAC2. So in this model we are only dealing with the increase in\nbasal activation of AC2 induced by PKC\nk1 = 1.66e-6\nk2 = 16\nk3 =4\n"
call /kinetics/PKC-active/phosph-AC2/notes LOAD \
"Phorbol esters have little effect on AC1 or on the Gs-stimulation of" \
"AC2. So in this model we are only dealing with the increase in" \
"basal activation of AC2 induced by PKC" \
"k1 = 1.66e-6" \
"k2 = 16" \
"k3 =4" \
""
simundump group /kinetics/PLA2 0 darkgreen black x 0 1 "" defaultfile \
  defaultfile.g 0 0 0 -7.3572 -14.209 0
simundump text /kinetics/PLA2/notes 0 \
  "Mail source of data: Leslie and Channon BBA 1045 (1990) pp 261-270.\nFig 6 is Ca curve. Fig 4a is PIP2 curve. Fig 4b is DAG curve. Also see\nWijkander and Sundler JBC 202 (1991) pp873-880;\nDiez and Mong JBC 265(24) p14654;\nLeslie JBC 266(17) (1991) pp11366-11371"
call /kinetics/PLA2/notes LOAD \
"Mail source of data: Leslie and Channon BBA 1045 (1990) pp 261-270." \
"Fig 6 is Ca curve. Fig 4a is PIP2 curve. Fig 4b is DAG curve. Also see" \
"Wijkander and Sundler JBC 202 (1991) pp873-880;" \
"Diez and Mong JBC 265(24) p14654;" \
"Leslie JBC 266(17) (1991) pp11366-11371"
simundump kpool /kinetics/PLA2/PLA2-cytosolic 0 0 0.4 0.4 2.4e+05 2.4e+05 0 0 \
  6e+05 0 /kinetics/geometry yellow darkgreen -11.824 -8.9421 0
simundump text /kinetics/PLA2/PLA2-cytosolic/notes 0 \
  "Calculated cytosolic was 20 nm from Wijkander and Sundler\nHowever, Leslie and Channon use about 400 nM. Need to confirm,\nbut this is the value I use here. Another recalc of W&S gives 1uM"
call /kinetics/PLA2/PLA2-cytosolic/notes LOAD \
"Calculated cytosolic was 20 nm from Wijkander and Sundler" \
"However, Leslie and Channon use about 400 nM. Need to confirm," \
"but this is the value I use here. Another recalc of W&S gives 1uM"
simundump kreac /kinetics/PLA2/PLA2-Ca-act 0 1.6667e-06 0.1 "" white \
  darkgreen -11.097 -11.104 0
simundump text /kinetics/PLA2/PLA2-Ca-act/notes 0 \
  "Leslie and Channon BBA 1045 (1990) 261-270 fig6 pp267."
call /kinetics/PLA2/PLA2-Ca-act/notes LOAD \
"Leslie and Channon BBA 1045 (1990) 261-270 fig6 pp267."
simundump kpool /kinetics/PLA2/PLA2-Ca* 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry yellow darkgreen -8.722 -11.646 0
simundump text /kinetics/PLA2/PLA2-Ca*/notes 0 ""
call /kinetics/PLA2/PLA2-Ca*/notes LOAD \
""
simundump kenz /kinetics/PLA2/PLA2-Ca*/kenz 0 0 0 0 0 6e+05 2.25e-06 21.6 5.4 \
  0 0 "" red yellow "" -6.0553 -11.667 0
simundump text /kinetics/PLA2/PLA2-Ca*/kenz/notes 0 \
  "10 x raise oct22\n12 x oct 24, set k2 = 4 * k3"
call /kinetics/PLA2/PLA2-Ca*/kenz/notes LOAD \
"10 x raise oct22" \
"12 x oct 24, set k2 = 4 * k3"
simundump kreac /kinetics/PLA2/PIP2-PLA2-act 0 2e-09 0.5 "" white darkgreen \
  -11.055 -6.7502 0
simundump text /kinetics/PLA2/PIP2-PLA2-act/notes 0 ""
call /kinetics/PLA2/PIP2-PLA2-act/notes LOAD \
""
simundump kpool /kinetics/PLA2/PIP2-PLA2* 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry cyan darkgreen -8.6803 -6.2919 0
simundump text /kinetics/PLA2/PIP2-PLA2*/notes 0 ""
call /kinetics/PLA2/PIP2-PLA2*/notes LOAD \
""
simundump kenz /kinetics/PLA2/PIP2-PLA2*/kenz 0 0 0 0 0 6e+05 4.6e-06 44.16 \
  11.04 0 0 "" red cyan "" -6.0345 -6.271 0
simundump text /kinetics/PLA2/PIP2-PLA2*/kenz/notes 0 \
  "10 X raise oct 22\n12 X further raise oct 24 to allow for correct conc of enzyme\n"
call /kinetics/PLA2/PIP2-PLA2*/kenz/notes LOAD \
"10 X raise oct 22" \
"12 X further raise oct 24 to allow for correct conc of enzyme" \
""
simundump kreac /kinetics/PLA2/PIP2-Ca-PLA2-act 0 2e-08 0.1 "" white \
  darkgreen -10.097 -7.5002 0
simundump text /kinetics/PLA2/PIP2-Ca-PLA2-act/notes 0 ""
call /kinetics/PLA2/PIP2-Ca-PLA2-act/notes LOAD \
""
simundump kpool /kinetics/PLA2/PIP2-Ca-PLA2* 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry cyan darkgreen -8.3261 -7.896 0
simundump text /kinetics/PLA2/PIP2-Ca-PLA2*/notes 0 ""
call /kinetics/PLA2/PIP2-Ca-PLA2*/notes LOAD \
""
simundump kenz /kinetics/PLA2/PIP2-Ca-PLA2*/kenz 0 0 0 0 0 6e+05 1.5e-05 144 \
  36 0 0 "" red cyan "" -5.972 -7.9794 0
simundump text /kinetics/PLA2/PIP2-Ca-PLA2*/kenz/notes 0 \
  "10 x raise oct 22\n12 x and rescale for k2 = 4 * k3 convention oct 24\nIncrease further to get the match to expt, which was spoilt due\nto large accumulation of PLA2 in the enzyme complexed forms.\nLets raise k3, leaving the others at \nk1 = 1.5e-5 and k2 = 144 since they are large already.\n"
call /kinetics/PLA2/PIP2-Ca-PLA2*/kenz/notes LOAD \
"10 x raise oct 22" \
"12 x and rescale for k2 = 4 * k3 convention oct 24" \
"Increase further to get the match to expt, which was spoilt due" \
"to large accumulation of PLA2 in the enzyme complexed forms." \
"Lets raise k3, leaving the others at " \
"k1 = 1.5e-5 and k2 = 144 since they are large already." \
""
simundump kreac /kinetics/PLA2/DAG-Ca-PLA2-act 0 5e-09 4 "" white darkgreen \
  -10.826 -9.8336 0
simundump text /kinetics/PLA2/DAG-Ca-PLA2-act/notes 0 \
  "27 June 1996\nScaled kf down by 0.015\nfrom 3.33e-7 to 5e-9\nto fit with revised DAG estimates\nand use of mole-fraction to calculate eff on PLA2."
call /kinetics/PLA2/DAG-Ca-PLA2-act/notes LOAD \
"27 June 1996" \
"Scaled kf down by 0.015" \
"from 3.33e-7 to 5e-9" \
"to fit with revised DAG estimates" \
"and use of mole-fraction to calculate eff on PLA2."
simundump kpool /kinetics/PLA2/DAG-Ca-PLA2* 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry pink darkgreen -8.1386 -10.479 0
simundump text /kinetics/PLA2/DAG-Ca-PLA2*/notes 0 ""
call /kinetics/PLA2/DAG-Ca-PLA2*/notes LOAD \
""
simundump kenz /kinetics/PLA2/DAG-Ca-PLA2*/kenz 0 0 0 0 0 6e+05 2.5e-05 240 \
  60 0 0 "" red pink "" -5.9511 -10.354 0
simundump text /kinetics/PLA2/DAG-Ca-PLA2*/kenz/notes 0 \
  "10 X raise oct 22\n12 X raise oct 24 + conversion to k2 =4 * k3"
call /kinetics/PLA2/DAG-Ca-PLA2*/kenz/notes LOAD \
"10 X raise oct 22" \
"12 X raise oct 24 + conversion to k2 =4 * k3"
simundump kpool /kinetics/PLA2/APC 0 0 30 30 1.8e+07 1.8e+07 0 0 6e+05 4 \
  /kinetics/geometry yellow darkgreen -8.2386 -9.9634 0
simundump text /kinetics/PLA2/APC/notes 0 \
  "arachodonylphosphatidylcholine is the favoured substrate\nfrom Wijkander and Sundler, JBC 202 pp 873-880, 1991.\nTheir assay used 30 uM substrate, which is what the kinetics in\nthis model are based on. For the later model we should locate\na more realistic value for APC."
call /kinetics/PLA2/APC/notes LOAD \
"arachodonylphosphatidylcholine is the favoured substrate" \
"from Wijkander and Sundler, JBC 202 pp 873-880, 1991." \
"Their assay used 30 uM substrate, which is what the kinetics in" \
"this model are based on. For the later model we should locate" \
"a more realistic value for APC."
simundump kreac /kinetics/PLA2/Degrade-AA 1 0.4 0 "" white darkgreen -6.1808 \
  -5.2875 0
simundump text /kinetics/PLA2/Degrade-AA/notes 0 \
  "I need to check if the AA degradation pathway really leads back to \nAPC. Anyway, it is a convenient buffered pool to dump it back into.\nFor the purposes of the full model we use a rate of degradation of\n0.2/sec\nRaised decay to 0.4 : see PLA35.g notes for Feb17 "
call /kinetics/PLA2/Degrade-AA/notes LOAD \
"I need to check if the AA degradation pathway really leads back to " \
"APC. Anyway, it is a convenient buffered pool to dump it back into." \
"For the purposes of the full model we use a rate of degradation of" \
"0.2/sec" \
"Raised decay to 0.4 : see PLA35.g notes for Feb17 "
simundump kpool /kinetics/PLA2/PLA2*-Ca 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry orange darkgreen -7.813 -12.687 0
simundump text /kinetics/PLA2/PLA2*-Ca/notes 0 \
  "Phosphorylated form of PLA2. Still need to hook it up using kinases.\nPKA: Wightman et al JBC 257 pp6650 1982\nPKC: Many refs, eg Gronich et al JBC 263 pp 16645, 1988 but see Lin etal\nMAPK: Lin et al, Cell 72 pp 269, 1993.  Show 3x with MAPK but not PKC alone\nDo not know if there is a Ca requirement for active phosphorylated state."
call /kinetics/PLA2/PLA2*-Ca/notes LOAD \
"Phosphorylated form of PLA2. Still need to hook it up using kinases." \
"PKA: Wightman et al JBC 257 pp6650 1982" \
"PKC: Many refs, eg Gronich et al JBC 263 pp 16645, 1988 but see Lin etal" \
"MAPK: Lin et al, Cell 72 pp 269, 1993.  Show 3x with MAPK but not PKC alone" \
"Do not know if there is a Ca requirement for active phosphorylated state."
simundump kenz /kinetics/PLA2/PLA2*-Ca/kenz 0 0 0 0 0 6e+05 5e-05 480 120 0 0 \
  "" red orange "" -6.0814 -12.817 0
simundump text /kinetics/PLA2/PLA2*-Ca/kenz/notes 0 \
  "This form should be 3 to 6 times as fast as the Ca-only form.\nI have scaled by 4x which seems to give a 5x rise.\n10x raise Oct 22\n12 x oct 24, changed k2 = 4 * k3"
call /kinetics/PLA2/PLA2*-Ca/kenz/notes LOAD \
"This form should be 3 to 6 times as fast as the Ca-only form." \
"I have scaled by 4x which seems to give a 5x rise." \
"10x raise Oct 22" \
"12 x oct 24, changed k2 = 4 * k3"
simundump kpool /kinetics/PLA2/PLA2* 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry orange darkgreen -9.025 -14.851 0
simundump text /kinetics/PLA2/PLA2*/notes 0 ""
call /kinetics/PLA2/PLA2*/notes LOAD \
""
simundump kreac /kinetics/PLA2/PLA2*-Ca-act 1 1e-05 0.1 "" white darkgreen \
  -10.086 -12.752 0
simundump text /kinetics/PLA2/PLA2*-Ca-act/notes 0 \
  "To start off, same kinetics as the PLA2-Ca-act direct pathway.\nOops ! Missed out the Ca input to this pathway first time round.\nLet's raise the forward rate about 3x to 5e-6. This will let us reduce the\nrather high rates we have used for the kenz on PLA2*-Ca. In fact, it\nmay be that the rates are not that different, just that this pathway for\ngetting the PLA2 to the memb is more efficien...."
call /kinetics/PLA2/PLA2*-Ca-act/notes LOAD \
"To start off, same kinetics as the PLA2-Ca-act direct pathway." \
"Oops ! Missed out the Ca input to this pathway first time round." \
"Let's raise the forward rate about 3x to 5e-6. This will let us reduce the" \
"rather high rates we have used for the kenz on PLA2*-Ca. In fact, it" \
"may be that the rates are not that different, just that this pathway for" \
"getting the PLA2 to the memb is more efficien...."
simundump kreac /kinetics/PLA2/dephosphorylate-PLA2* 1 0.17 0 "" white \
  darkgreen -13.693 -11.735 0
simundump text /kinetics/PLA2/dephosphorylate-PLA2*/notes 0 ""
call /kinetics/PLA2/dephosphorylate-PLA2*/notes LOAD \
""
simundump kpool /kinetics/MAPK* 1 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry \
  orange yellow 13 1 0
simundump text /kinetics/MAPK*/notes 0 ""
call /kinetics/MAPK*/notes LOAD \
""
simundump kenz /kinetics/MAPK*/MAPK* 0 0 0 0 0 6e+05 3.25e-06 40 10 0 0 "" \
  red orange "" -12 -14 0
simundump text /kinetics/MAPK*/MAPK*/notes 0 \
  "Km = 25uM @ 50 uM ATP and 1mg/ml MBP (huge XS of substrate)\nVmax = 4124 pmol/min/ml at a conc of 125 pmol/ml of enz, so:\nk3 = .5/sec (rate limiting)\nk1 = (k2  + k3)/Km = (.5 + 0)/(25*6e5) = 2e-8 (#/cell)^-1\n#s from Sanghera et al JBC 265 pp 52 , 1990. \nFrom Nemenoff et al JBC 268(3):1960-1964 - using Sanghera's 1e-4 ratio\nof MAPK to protein, we get k3 = 7/sec from 1000 pmol/min/mg fig 5"
call /kinetics/MAPK*/MAPK*/notes LOAD \
"Km = 25uM @ 50 uM ATP and 1mg/ml MBP (huge XS of substrate)" \
"Vmax = 4124 pmol/min/ml at a conc of 125 pmol/ml of enz, so:" \
"k3 = .5/sec (rate limiting)" \
"k1 = (k2  + k3)/Km = (.5 + 0)/(25*6e5) = 2e-8 (#/cell)^-1" \
"#s from Sanghera et al JBC 265 pp 52 , 1990. " \
"From Nemenoff et al JBC 268(3):1960-1964 - using Sanghera's 1e-4 ratio" \
"of MAPK to protein, we get k3 = 7/sec from 1000 pmol/min/mg fig 5"
simundump kenz /kinetics/MAPK*/MAPK*-feedback 1 0 0 0 0 6e+05 3.25e-06 40 10 \
  0 0 "" red orange "" 11 10 0
simundump text /kinetics/MAPK*/MAPK*-feedback/notes 0 \
  "Ueki et al JBC 269(22):15756-15761 show the presence of\nthis step, but not the rate consts, which are derived from\nSanghera et al  JBC 265(1):52-57, 1990, see the deriv in the\nMAPK* notes."
call /kinetics/MAPK*/MAPK*-feedback/notes LOAD \
"Ueki et al JBC 269(22):15756-15761 show the presence of" \
"this step, but not the rate consts, which are derived from" \
"Sanghera et al  JBC 265(1):52-57, 1990, see the deriv in the" \
"MAPK* notes."
simundump kenz /kinetics/MAPK*/phosph_Sos 1 0 0 0 0 6e+05 3.25e-05 40 10 0 0 \
  "" red orange "" 14.005 53.115 0
simundump text /kinetics/MAPK*/phosph_Sos/notes 0 \
  "See Porfiri and McCormick JBC 271:10 pp5871 1996 for the\nexistence of this step. We'll take the rates from the ones\nused for the phosph of Raf by MAPK.\nSep 17 1997: The transient activation curve matches better\nwith k1 up  by 10 x."
call /kinetics/MAPK*/phosph_Sos/notes LOAD \
"See Porfiri and McCormick JBC 271:10 pp5871 1996 for the" \
"existence of this step. We'll take the rates from the ones" \
"used for the phosph of Raf by MAPK." \
"Sep 17 1997: The transient activation curve matches better" \
"with k1 up  by 10 x."
simundump kpool /kinetics/temp-PIP2 1 0 2.5 2.5 1.5e+06 1.5e+06 0 0 6e+05 4 \
  /kinetics/geometry green black -15.796 -7.0473 0
simundump text /kinetics/temp-PIP2/notes 0 \
  "This isn't explicitly present in the M&L model, but is obviously needed.\nI assume its conc is fixed at 1uM for now, which is a bit high. PLA2 is stim\n7x by PIP2 @ 0.5 uM (Leslie and Channon BBA 1045:261(1990) \nLeslie and Channon say PIP2 is present at 0.1 - 0.2mol% range in membs,\nwhich comes to 50 nM. Ref is Majerus et al Cell 37 pp 701-703 1984\nLets use a lower level of 30 nM, same ref...."
call /kinetics/temp-PIP2/notes LOAD \
"This isn't explicitly present in the M&L model, but is obviously needed." \
"I assume its conc is fixed at 1uM for now, which is a bit high. PLA2 is stim" \
"7x by PIP2 @ 0.5 uM (Leslie and Channon BBA 1045:261(1990) " \
"Leslie and Channon say PIP2 is present at 0.1 - 0.2mol% range in membs," \
"which comes to 50 nM. Ref is Majerus et al Cell 37 pp 701-703 1984" \
"Lets use a lower level of 30 nM, same ref...."
simundump kpool /kinetics/IP3 1 0 0.73 0.73 4.38e+05 4.38e+05 0 0 6e+05 0 \
  /kinetics/geometry pink black -0.77375 -4.6555 0
simundump text /kinetics/IP3/notes 0 \
  "Peak IP3 is perhaps 15 uM, basal <= 0.2 uM."
call /kinetics/IP3/notes LOAD \
"Peak IP3 is perhaps 15 uM, basal <= 0.2 uM."
simundump kpool /kinetics/Glu 1 0 0 0 0 0 0 0 6e+05 4 /kinetics/geometry \
  green black -0.79501 13.884 0
simundump text /kinetics/Glu/notes 0 \
  "Varying the amount of (steady state) glu between .01 uM and up, the\nfinal amount of G*GTP complex does not change much. This means that\nthe system should be reasonably robust wr to the amount of glu in the\nsynaptic cleft. It would be nice to know how fast it is removed."
call /kinetics/Glu/notes LOAD \
"Varying the amount of (steady state) glu between .01 uM and up, the" \
"final amount of G*GTP complex does not change much. This means that" \
"the system should be reasonably robust wr to the amount of glu in the" \
"synaptic cleft. It would be nice to know how fast it is removed."
simundump group /kinetics/PLCbeta 1 maroon black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 8.5846 -17.468 0
simundump text /kinetics/PLCbeta/notes 0 "Group for PLC beta"
call /kinetics/PLCbeta/notes LOAD \
"Group for PLC beta"
simundump kreac /kinetics/PLCbeta/Act-PLC-Ca 1 5e-06 1 "" white maroon 3.0709 \
  -16.978 0
simundump text /kinetics/PLCbeta/Act-PLC-Ca/notes 0 \
  "Affinity for Ca = 1uM without AlF, 0.1 with:\n from Smrcka et al science 251 pp 804-807 1991\nso [Ca].kf = kb so kb/kf = 1 * 6e5 = 1/1.66e-6\n\n11 June 1996: Raised affinity to 5e-6 to maintain\nbalance. See notes."
call /kinetics/PLCbeta/Act-PLC-Ca/notes LOAD \
"Affinity for Ca = 1uM without AlF, 0.1 with:" \
" from Smrcka et al science 251 pp 804-807 1991" \
"so [Ca].kf = kb so kb/kf = 1 * 6e5 = 1/1.66e-6" \
"" \
"11 June 1996: Raised affinity to 5e-6 to maintain" \
"balance. See notes."
simundump kpool /kinetics/PLCbeta/PLC 1 0 0.8 0.8 4.8e+05 4.8e+05 0 0 6e+05 0 \
  /kinetics/geometry cyan maroon 10.697 -16.957 0
simundump text /kinetics/PLCbeta/PLC/notes 0 \
  "Total PLC = 0.8 uM see Ryu et al JBC 262 (26) pp 12511 1987"
call /kinetics/PLCbeta/PLC/notes LOAD \
"Total PLC = 0.8 uM see Ryu et al JBC 262 (26) pp 12511 1987"
simundump kreac /kinetics/PLCbeta/Degrade-IP3 1 2.5 0 "" white maroon 2.3125 \
  -7.9705 0
simundump text /kinetics/PLCbeta/Degrade-IP3/notes 0 \
  "The enzyme is IP3 5-phosphomonesterase. about 45K. Actual products\nare Ins(1,4)P2, and cIns(1:2,4,5)P3.  review in Majerus et al Science 234\n1519-1526, 1986.\nMeyer and Stryer 1988 PNAS 85:5051-5055 est decay of IP3 at\n 1-3/sec"
call /kinetics/PLCbeta/Degrade-IP3/notes LOAD \
"The enzyme is IP3 5-phosphomonesterase. about 45K. Actual products" \
"are Ins(1,4)P2, and cIns(1:2,4,5)P3.  review in Majerus et al Science 234" \
"1519-1526, 1986." \
"Meyer and Stryer 1988 PNAS 85:5051-5055 est decay of IP3 at" \
" 1-3/sec"
simundump kpool /kinetics/PLCbeta/Inositol 1 0 0 0 0 0 0 0 6e+05 4 \
  /kinetics/geometry green maroon 4.9653 -8.7416 0
simundump text /kinetics/PLCbeta/Inositol/notes 0 ""
call /kinetics/PLCbeta/Inositol/notes LOAD \
""
simundump kreac /kinetics/PLCbeta/Degrade-DAG 1 0.15 0 "" white maroon \
  -0.95715 -7.261 0
simundump text /kinetics/PLCbeta/Degrade-DAG/notes 0 \
  "These rates are the same as for degrading IP3, but I am sure that they could\nbe improved.\nLets double kf to 0.2, since the amount of DAG in the cell should be <= 1uM.\nNeed to double it again, for the same reason.\nkf now 0.5\n27 June 1996\nkf is now 0.02 to get 50 sec time course\n30 Aug 1997: Raised kf to 0.11 to accomodate PLC_gamma\n27 Mar 1998: kf now 0.15 for PLC_gamma"
call /kinetics/PLCbeta/Degrade-DAG/notes LOAD \
"These rates are the same as for degrading IP3, but I am sure that they could" \
"be improved." \
"Lets double kf to 0.2, since the amount of DAG in the cell should be <= 1uM." \
"Need to double it again, for the same reason." \
"kf now 0.5" \
"27 June 1996" \
"kf is now 0.02 to get 50 sec time course" \
"30 Aug 1997: Raised kf to 0.11 to accomodate PLC_gamma" \
"27 Mar 1998: kf now 0.15 for PLC_gamma"
simundump kpool /kinetics/PLCbeta/PC 1 0 0 0 0 0 0 0 6e+05 4 \
  /kinetics/geometry[1] green maroon 4.9036 -7.1376 0
simundump text /kinetics/PLCbeta/PC/notes 0 \
  "Phosphatidylcholine is the main (around 55%) metabolic product of DAG,\nfollwed by PE (around 25%). Ref is Welsh and Cabot, JCB35:231-245(1987)"
call /kinetics/PLCbeta/PC/notes LOAD \
"Phosphatidylcholine is the main (around 55%) metabolic product of DAG," \
"follwed by PE (around 25%). Ref is Welsh and Cabot, JCB35:231-245(1987)"
simundump kpool /kinetics/PLCbeta/PLC-Ca 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry cyan maroon 7.0147 -12.797 0
simundump text /kinetics/PLCbeta/PLC-Ca/notes 0 ""
call /kinetics/PLCbeta/PLC-Ca/notes LOAD \
""
simundump kenz /kinetics/PLCbeta/PLC-Ca/PLC-Ca 1 0 0 0 0 6e+05 4.2e-06 40 10 \
  0 0 "" red cyan "" -2.2511 -11.697 0
simundump text /kinetics/PLCbeta/PLC-Ca/PLC-Ca/notes 0 \
  "From Sternweis et al Phil Trans R Soc Lond 1992, also matched by Homma et al.\nk1 = 1.5e-5, now 4.2e-6\nk2 = 70/sec; now 40/sec\nk3 = 17.5/sec; now 10/sec\nNote that the wording in Sternweis et al is\nambiguous re the Km."
call /kinetics/PLCbeta/PLC-Ca/PLC-Ca/notes LOAD \
"From Sternweis et al Phil Trans R Soc Lond 1992, also matched by Homma et al." \
"k1 = 1.5e-5, now 4.2e-6" \
"k2 = 70/sec; now 40/sec" \
"k3 = 17.5/sec; now 10/sec" \
"Note that the wording in Sternweis et al is" \
"ambiguous re the Km."
simundump kreac /kinetics/PLCbeta/Act-PLC-by-Gq 1 4.2e-05 1 "" white maroon \
  2.6996 -15.163 0
simundump text /kinetics/PLCbeta/Act-PLC-by-Gq/notes 0 \
  "Affinity for Gq is > 20 nM (Smrcka et al Science251 804-807 1991)\nso [Gq].kf = kb so 40nM * 6e5 = kb/kf = 24e3 so kf = 4.2e-5, kb =1\n"
call /kinetics/PLCbeta/Act-PLC-by-Gq/notes LOAD \
"Affinity for Gq is > 20 nM (Smrcka et al Science251 804-807 1991)" \
"so [Gq].kf = kb so 40nM * 6e5 = kb/kf = 24e3 so kf = 4.2e-5, kb =1" \
""
simundump kreac /kinetics/PLCbeta/Inact-PLC-Gq 1 0.0133 0 "" white maroon \
  11.125 -10.314 0
simundump text /kinetics/PLCbeta/Inact-PLC-Gq/notes 0 \
  "This process is assumed to be directly caused by the inactivation of\nthe G*GTP to G*GDP. Hence, \nkf = .013 /sec = 0.8/min, same as the rate for Inact-G.\nkb = 0 since this is irreversible.\nWe may be\ninterested in studying the role of PLC as a GAP. If so, the kf would be faster here\nthan in Inact-G"
call /kinetics/PLCbeta/Inact-PLC-Gq/notes LOAD \
"This process is assumed to be directly caused by the inactivation of" \
"the G*GTP to G*GDP. Hence, " \
"kf = .013 /sec = 0.8/min, same as the rate for Inact-G." \
"kb = 0 since this is irreversible." \
"We may be" \
"interested in studying the role of PLC as a GAP. If so, the kf would be faster here" \
"than in Inact-G"
simundump kpool /kinetics/PLCbeta/PLC-Ca-Gq 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry cyan maroon 10.629 -13.411 0
simundump text /kinetics/PLCbeta/PLC-Ca-Gq/notes 0 \
  "This should really be labelled PLC-G*GTP-Ca.\nThis is the activated form of the enzyme. Mahama and Linderman assume\nthat the IP3 precursors are not rate-limiting, but I will include those for\ncompleteness as they may be needed later.\n"
call /kinetics/PLCbeta/PLC-Ca-Gq/notes LOAD \
"This should really be labelled PLC-G*GTP-Ca." \
"This is the activated form of the enzyme. Mahama and Linderman assume" \
"that the IP3 precursors are not rate-limiting, but I will include those for" \
"completeness as they may be needed later." \
""
simundump kenz /kinetics/PLCbeta/PLC-Ca-Gq/PLCb-Ca-Gq 0 0 0 0 0 6e+05 8e-05 \
  192 48 0 0 "" red cyan "" 2.9471 -11.078 0
simundump text /kinetics/PLCbeta/PLC-Ca-Gq/PLCb-Ca-Gq/notes 0 \
  "From Sternweis et al, Phil Trans R Soc Lond 1992, and the values from\nother refs eg Homma et al JBC 263(14) pp6592 1988 match.\nk1 = 5e-5/sec\nk2 = 240/sec; now 120/sec\nk3 = 60/sec; now 30/sec\nNote that the wording in Sternweis et al\nis ambiguous wr. to the Km for Gq vs non-Gq states of PLC. \nK1 is still a bit too low. Raise to 7e-5\n9 Jun 1996: k1 was 0.0002, changed to 5e-5"
call /kinetics/PLCbeta/PLC-Ca-Gq/PLCb-Ca-Gq/notes LOAD \
"From Sternweis et al, Phil Trans R Soc Lond 1992, and the values from" \
"other refs eg Homma et al JBC 263(14) pp6592 1988 match." \
"k1 = 5e-5/sec" \
"k2 = 240/sec; now 120/sec" \
"k3 = 60/sec; now 30/sec" \
"Note that the wording in Sternweis et al" \
"is ambiguous wr. to the Km for Gq vs non-Gq states of PLC. " \
"K1 is still a bit too low. Raise to 7e-5" \
"9 Jun 1996: k1 was 0.0002, changed to 5e-5"
simundump kpool /kinetics/PLCbeta/PLC-Gq 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry cyan maroon 15.035 -13.537 0
simundump text /kinetics/PLCbeta/PLC-Gq/notes 0 ""
call /kinetics/PLCbeta/PLC-Gq/notes LOAD \
""
simundump kreac /kinetics/PLCbeta/PLC-bind-Gq 1 4.2e-06 1 "" white maroon \
  14.746 -16.263 0
simundump text /kinetics/PLCbeta/PLC-bind-Gq/notes 0 \
  "this binding does not produce active PLC. This step was needed to\nimplement the described (Smrcka et al) increase in affinity for Ca\nby PLC once Gq was bound.\nThe kinetics are the same as the binding step for Ca-PLC to Gq.\n\nJune 1996:\nChanged the kf to 4.2e-5 to 4.2e-6 to preserve balance around\nthe reactions. "
call /kinetics/PLCbeta/PLC-bind-Gq/notes LOAD \
"this binding does not produce active PLC. This step was needed to" \
"implement the described (Smrcka et al) increase in affinity for Ca" \
"by PLC once Gq was bound." \
"The kinetics are the same as the binding step for Ca-PLC to Gq." \
"" \
"June 1996:" \
"Changed the kf to 4.2e-5 to 4.2e-6 to preserve balance around" \
"the reactions. "
simundump kreac /kinetics/PLCbeta/PLC-Gq-bind-Ca 1 5e-05 1 "" white maroon \
  14.004 -11.254 0
simundump text /kinetics/PLCbeta/PLC-Gq-bind-Ca/notes 0 \
  "this step has a high affinity for Ca, from Smrcka et al. 0.1uM\nso kf /kb = 1/6e4 = 1.666e-5:1. See the Act-PLC-by-Gq reac.\n11 Jun 1996: Raised kf to 5e-5 based on match to conc-eff\ncurves from Smrcka et al."
call /kinetics/PLCbeta/PLC-Gq-bind-Ca/notes LOAD \
"this step has a high affinity for Ca, from Smrcka et al. 0.1uM" \
"so kf /kb = 1/6e4 = 1.666e-5:1. See the Act-PLC-by-Gq reac." \
"11 Jun 1996: Raised kf to 5e-5 based on match to conc-eff" \
"curves from Smrcka et al."
simundump kpool /kinetics/PIP2 1 0 10 10 6e+06 6e+06 0 0 6e+05 4 \
  /kinetics/geometry green black 3.8839 -6.7218 0
simundump text /kinetics/PIP2/notes 0 ""
call /kinetics/PIP2/notes LOAD \
""
simundump kpool /kinetics/BetaGamma 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry yellow black 15.787 -2.6163 0
simundump text /kinetics/BetaGamma/notes 0 \
  "These exist in a nebulous sense in this model, basically only to balance\nthe conservation equations. The details of their reassociation with G-GDP\nare not modeled\nResting level =0.0094, stim level =.0236 from all42.g ish."
call /kinetics/BetaGamma/notes LOAD \
"These exist in a nebulous sense in this model, basically only to balance" \
"the conservation equations. The details of their reassociation with G-GDP" \
"are not modeled" \
"Resting level =0.0094, stim level =.0236 from all42.g ish."
simundump kpool /kinetics/G*GTP 1 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry \
  red black 7.3149 -7.0131 0
simundump text /kinetics/G*GTP/notes 0 \
  "Activated G protein. Berstein et al indicate that about 20-40% of the total\nGq alpha should bind GTP at steady stim. This sim gives more like 65%."
call /kinetics/G*GTP/notes LOAD \
"Activated G protein. Berstein et al indicate that about 20-40% of the total" \
"Gq alpha should bind GTP at steady stim. This sim gives more like 65%."
simundump kpool /kinetics/G*GDP 1 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry \
  yellow black 13.56 -5.6529 0
simundump text /kinetics/G*GDP/notes 0 ""
call /kinetics/G*GDP/notes LOAD \
""
simundump group /kinetics/Gq 1 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 12.745 -1.9437 0
simundump text /kinetics/Gq/notes 0 \
  "We assume GTP is present in fixed amounts, so we leave it out\nof the explicit equations in this model. Normally we would expect it\nto associate along with the G-Receptor-ligand complex.\nMost info is from Berstein et al JBC 267:12 8081-8088 1992\nStructure of rec activation of Gq from Fay et al Biochem 30 5066-5075 1991"
call /kinetics/Gq/notes LOAD \
"We assume GTP is present in fixed amounts, so we leave it out" \
"of the explicit equations in this model. Normally we would expect it" \
"to associate along with the G-Receptor-ligand complex." \
"Most info is from Berstein et al JBC 267:12 8081-8088 1992" \
"Structure of rec activation of Gq from Fay et al Biochem 30 5066-5075 1991"
simundump kreac /kinetics/Gq/RecLigandBinding 1 2.8e-05 10 "" white blue \
  7.3388 -3.179 0
simundump text /kinetics/Gq/RecLigandBinding/notes 0 \
  "kf = kf from text = 1e7 / M / sec = 10 /uM/sec = 10 / 6e5 / # / sec  = 1.67e-5\nkb = kr from text = 60 / sec\nNote that we continue to use uM here since [phenylephrine] is also in uM.\nFrom Martin et al FEBS Lett 316:2 191-196 1993 we have Kd = 600 nM\nAssuming kb = 10/sec, we get kf = 10/(0.6 uM * 6e5) = 2.8e-5 1/sec/#"
call /kinetics/Gq/RecLigandBinding/notes LOAD \
"kf = kf from text = 1e7 / M / sec = 10 /uM/sec = 10 / 6e5 / # / sec  = 1.67e-5" \
"kb = kr from text = 60 / sec" \
"Note that we continue to use uM here since [phenylephrine] is also in uM." \
"From Martin et al FEBS Lett 316:2 191-196 1993 we have Kd = 600 nM" \
"Assuming kb = 10/sec, we get kf = 10/(0.6 uM * 6e5) = 2.8e-5 1/sec/#"
simundump kpool /kinetics/Gq/G-GDP 1 0 1 1 6e+05 6e+05 0 0 6e+05 0 \
  /kinetics/geometry yellow blue 10.68 -2.5729 0
simundump text /kinetics/Gq/G-GDP/notes 0 \
  "From M&L, total Gprot = 1e5molecules/cell\nAt equil, 92340 are here, 400 are in G*GTP, and another 600 are assoc\nwith the PLC and 6475 are as G*GDP. This is OK.\n\nFrom Pang and Sternweis JBC 265:30 18707-12 1990 we get conc est 1.6 uM\nto 0.8 uM. A number of other factors are involved too.\n"
call /kinetics/Gq/G-GDP/notes LOAD \
"From M&L, total Gprot = 1e5molecules/cell" \
"At equil, 92340 are here, 400 are in G*GTP, and another 600 are assoc" \
"with the PLC and 6475 are as G*GDP. This is OK." \
"" \
"From Pang and Sternweis JBC 265:30 18707-12 1990 we get conc est 1.6 uM" \
"to 0.8 uM. A number of other factors are involved too." \
""
simundump kreac /kinetics/Gq/Basal-Act-G 1 0.0001 0 "" white blue 9.805 \
  -4.8225 0
simundump text /kinetics/Gq/Basal-Act-G/notes 0 \
  "kf = kg1 = 0.01/sec, kb = 0. This is the basal exchange of GTP for GDP."
call /kinetics/Gq/Basal-Act-G/notes LOAD \
"kf = kg1 = 0.01/sec, kb = 0. This is the basal exchange of GTP for GDP."
simundump kreac /kinetics/Gq/Trimerize-G 1 1e-05 0 "" white blue 12.255 \
  -4.7831 0
simundump text /kinetics/Gq/Trimerize-G/notes 0 \
  "kf == kg3 = 1e-5 /cell/sec. As usual, there is no back-reaction\nkb = 0"
call /kinetics/Gq/Trimerize-G/notes LOAD \
"kf == kg3 = 1e-5 /cell/sec. As usual, there is no back-reaction" \
"kb = 0"
simundump kreac /kinetics/Gq/Inact-G 1 0.0133 0 "" white blue 10.218 -7.6095 \
  0
simundump text /kinetics/Gq/Inact-G/notes 0 \
  "From Berstein et al JBC 267:12 8081-8088 1992, kcat for GTPase activity\nof Gq is only 0.8/min"
call /kinetics/Gq/Inact-G/notes LOAD \
"From Berstein et al JBC 267:12 8081-8088 1992, kcat for GTPase activity" \
"of Gq is only 0.8/min"
simundump kpool /kinetics/Gq/mGluR 1 0 0.3 0.3 1.8e+05 1.8e+05 0 0 6e+05 0 \
  /kinetics/geometry green blue 6.4638 -1.7623 0
simundump text /kinetics/Gq/mGluR/notes 0 \
  "From M&L, Total # of receptors/cell = 1900\nVol of cell = 1e-15 (10 um cube). Navogadro = 6.023e23\nso conversion from n to conc in uM is n/vol*nA * 1e3 = 1.66e-6\nHowever, for typical synaptic channels the density is likely to be very\nhigh at the synapse. Use an estimate of 0.1 uM for now. this gives\na total of about 60K receptors/cell, which is in line with Fay et at."
call /kinetics/Gq/mGluR/notes LOAD \
"From M&L, Total # of receptors/cell = 1900" \
"Vol of cell = 1e-15 (10 um cube). Navogadro = 6.023e23" \
"so conversion from n to conc in uM is n/vol*nA * 1e3 = 1.66e-6" \
"However, for typical synaptic channels the density is likely to be very" \
"high at the synapse. Use an estimate of 0.1 uM for now. this gives" \
"a total of about 60K receptors/cell, which is in line with Fay et at."
simundump kpool /kinetics/Gq/Rec-Glu 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry green blue 5.8108 -3.7217 0
simundump text /kinetics/Gq/Rec-Glu/notes 0 \
  "This acts like an enzyme to activate the g proteins\nAssume cell has vol 1e-15 m^3 (10 uM cube), conversion factor to\nconc in uM is 6e5\n"
call /kinetics/Gq/Rec-Glu/notes LOAD \
"This acts like an enzyme to activate the g proteins" \
"Assume cell has vol 1e-15 m^3 (10 uM cube), conversion factor to" \
"conc in uM is 6e5" \
""
simundump kpool /kinetics/Gq/Rec-Gq 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry green blue 4.0767 -0.99942 0
simundump text /kinetics/Gq/Rec-Gq/notes 0 \
  "Fraction of Rec-Gq is 44%  of rec, from Fay et al.\nSince this is not the same receptor, this value is a bit doubtful. Still,\nwe adjust the rate consts in Rec-bind-Gq to match."
call /kinetics/Gq/Rec-Gq/notes LOAD \
"Fraction of Rec-Gq is 44%  of rec, from Fay et al." \
"Since this is not the same receptor, this value is a bit doubtful. Still," \
"we adjust the rate consts in Rec-bind-Gq to match."
simundump kreac /kinetics/Gq/Rec-Glu-bind-Gq 1 1e-08 0.0001 "" white blue \
  4.7148 -2.4225 0
simundump text /kinetics/Gq/Rec-Glu-bind-Gq/notes 0 \
  "This is the k1-k2 equivalent for enzyme complex formation in the\nbinding of Rec-Glu to Gq.\nSee Fay et al Biochem 30 5066-5075 1991.\nkf = 5e-5 which is nearly the same as calculated by Fay et al. (4.67e-5)\nkb = .04\n\nJune 1996: Closer reading of Fay et al suggests that \nkb <= 0.0001, so kf = 1e-8 by detailed balance. This\nreaction appears to be neglible."
call /kinetics/Gq/Rec-Glu-bind-Gq/notes LOAD \
"This is the k1-k2 equivalent for enzyme complex formation in the" \
"binding of Rec-Glu to Gq." \
"See Fay et al Biochem 30 5066-5075 1991." \
"kf = 5e-5 which is nearly the same as calculated by Fay et al. (4.67e-5)" \
"kb = .04" \
"" \
"June 1996: Closer reading of Fay et al suggests that " \
"kb <= 0.0001, so kf = 1e-8 by detailed balance. This" \
"reaction appears to be neglible."
simundump kreac /kinetics/Gq/Glu-bind-Rec-Gq 1 2.8e-05 0.1 "" white blue \
  2.386 -3.0371 0
simundump text /kinetics/Gq/Glu-bind-Rec-Gq/notes 0 \
  "From Fay et al\nkb3 = kb = 1.06e-3 which is rather slow.\nk+1 = kf = 2.8e7 /M/sec= 4.67e-5/sec use 5e-5.\nHowever, the Kd from Martin et al may be more appropriate, as this\nis Glu not the system from Fay.\nkf = 2.8e-5, kb = 10\nLet us compromise. since we have the Fay model, keep kf = k+1 = 2.8e-5.\nBut kb (k-3) is .01 * k-1 from Fay. Scaling by .01, kb = .01 * 10 = 0.1"
call /kinetics/Gq/Glu-bind-Rec-Gq/notes LOAD \
"From Fay et al" \
"kb3 = kb = 1.06e-3 which is rather slow." \
"k+1 = kf = 2.8e7 /M/sec= 4.67e-5/sec use 5e-5." \
"However, the Kd from Martin et al may be more appropriate, as this" \
"is Glu not the system from Fay." \
"kf = 2.8e-5, kb = 10" \
"Let us compromise. since we have the Fay model, keep kf = k+1 = 2.8e-5." \
"But kb (k-3) is .01 * k-1 from Fay. Scaling by .01, kb = .01 * 10 = 0.1"
simundump kpool /kinetics/Gq/Rec-Glu-Gq 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry orange blue 4.7416 -5.1166 0
simundump text /kinetics/Gq/Rec-Glu-Gq/notes 0 ""
call /kinetics/Gq/Rec-Glu-Gq/notes LOAD \
""
simundump kreac /kinetics/Gq/Activate-Gq 1 0.01 0 "" white blue 7.0172 \
  -4.6572 0
simundump text /kinetics/Gq/Activate-Gq/notes 0 \
  "This is the kcat==k3 stage of the Rec-Glu ezymatic activation of Gq.\nFrom Berstein et al actiation is at .35 - 0.7/min\nFrom Fay et al Biochem 30 5066-5075 1991 kf = .01/sec\nFrom Nakamura et al J physiol Lond 474:1 35-41 1994 see time courses.\nAlso (Berstein) 15-40% of gprot is in GTP-bound form on stim."
call /kinetics/Gq/Activate-Gq/notes LOAD \
"This is the kcat==k3 stage of the Rec-Glu ezymatic activation of Gq." \
"From Berstein et al actiation is at .35 - 0.7/min" \
"From Fay et al Biochem 30 5066-5075 1991 kf = .01/sec" \
"From Nakamura et al J physiol Lond 474:1 35-41 1994 see time courses." \
"Also (Berstein) 15-40% of gprot is in GTP-bound form on stim."
simundump kreac /kinetics/Gq/Rec-bind-Gq 1 1e-06 1 "" white blue 6.743 \
  -0.088999 0
simundump text /kinetics/Gq/Rec-bind-Gq/notes 0 \
  "Lets try out the same kinetics as the Rec-Glu-bind-Gq\nThis is much too forward. We know that the steady-state\namount of Rec-Gq should be 40% of the total amount of receptor.\nThis is for a different receptor, still we can try to match the value.\nkf = 1e-6 and kb = 1 give 0.333:0.8 which is pretty close.\n"
call /kinetics/Gq/Rec-bind-Gq/notes LOAD \
"Lets try out the same kinetics as the Rec-Glu-bind-Gq" \
"This is much too forward. We know that the steady-state" \
"amount of Rec-Gq should be 40% of the total amount of receptor." \
"This is for a different receptor, still we can try to match the value." \
"kf = 1e-6 and kb = 1 give 0.333:0.8 which is pretty close." \
""
simundump kpool /kinetics/Gq/mGluRAntag 1 0 0 0 0 0 0 0 6e+05 4 \
  /kinetics/geometry seagreen blue 0.60216 -2.3091 0
simundump text /kinetics/Gq/mGluRAntag/notes 0 \
  "I am implementing this as acting only on the Rec-Gq complex, based on\na more complete model PLC_Gq48.g\nwhich showed that the binding to the rec alone contributed only a small amount."
call /kinetics/Gq/mGluRAntag/notes LOAD \
"I am implementing this as acting only on the Rec-Gq complex, based on" \
"a more complete model PLC_Gq48.g" \
"which showed that the binding to the rec alone contributed only a small amount."
simundump kreac /kinetics/Gq/Antag-bind-Rec-Gq 1 0.0001 0.01 "" white blue \
  2.1399 -4.2806 0
simundump text /kinetics/Gq/Antag-bind-Rec-Gq/notes 0 \
  "The rate consts give a total binding affinity of only "
call /kinetics/Gq/Antag-bind-Rec-Gq/notes LOAD \
"The rate consts give a total binding affinity of only "
simundump kpool /kinetics/Gq/Blocked-rec-Gq 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry seagreen blue 2.4602 -5.9815 0
simundump text /kinetics/Gq/Blocked-rec-Gq/notes 0 ""
call /kinetics/Gq/Blocked-rec-Gq/notes LOAD \
""
simundump group /kinetics/MAPK 0 brown black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 14.616 11.191 0
simundump text /kinetics/MAPK/notes 0 ""
call /kinetics/MAPK/notes LOAD \
""
simundump kpool /kinetics/MAPK/craf-1 0 0 0.5 0.5 3e+05 3e+05 0 0 6e+05 0 \
  /kinetics/geometry pink brown 6 8 0
simundump text /kinetics/MAPK/craf-1/notes 0 \
  "Couldn't find any ref to the actual conc of craf-1 but I\nshould try Strom et al Oncogene 5 pp 345\nIn line with the other kinases in the cascade, I estimate the conc to be\n0.2 uM. To init we use 0.15, which is close to equil"
call /kinetics/MAPK/craf-1/notes LOAD \
"Couldn't find any ref to the actual conc of craf-1 but I" \
"should try Strom et al Oncogene 5 pp 345" \
"In line with the other kinases in the cascade, I estimate the conc to be" \
"0.2 uM. To init we use 0.15, which is close to equil"
simundump kpool /kinetics/MAPK/craf-1* 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry pink brown 9 8 0
simundump text /kinetics/MAPK/craf-1*/notes 0 ""
call /kinetics/MAPK/craf-1*/notes LOAD \
""
simundump kpool /kinetics/MAPK/MAPKK 0 0 0.5 0.5 3e+05 3e+05 0 0 6e+05 0 \
  /kinetics/geometry pink brown 5 4 0
simundump text /kinetics/MAPK/MAPKK/notes 0 \
  "Conc is from Seger et al JBC 267:20 pp14373 (1992)\nmwt is 45/46 Kd\nWe assume that phosphorylation on both ser and thr is needed for\nactiviation. See Kyriakis et al Nature 358 417 1992\nInit conc of total is 0.18\n"
call /kinetics/MAPK/MAPKK/notes LOAD \
"Conc is from Seger et al JBC 267:20 pp14373 (1992)" \
"mwt is 45/46 Kd" \
"We assume that phosphorylation on both ser and thr is needed for" \
"activiation. See Kyriakis et al Nature 358 417 1992" \
"Init conc of total is 0.18" \
""
simundump kpool /kinetics/MAPK/MAPK 0 0 3 3 1.8e+06 1.8e+06 0 0 6e+05 0 \
  /kinetics/geometry pink brown 5 1 0
simundump text /kinetics/MAPK/MAPK/notes 0 \
  "conc is from Sanghera et al JBC 265 pp 52 (1990)\nA second calculation gives 3.1 uM, from same paper.\nThey est MAPK is 1e-4x total protein, and protein is 15% of cell wt,\nso MAPK is 1.5e-5g/ml = 0.36uM. which is closer to our first estimate.\nLets use this."
call /kinetics/MAPK/MAPK/notes LOAD \
"conc is from Sanghera et al JBC 265 pp 52 (1990)" \
"A second calculation gives 3.1 uM, from same paper." \
"They est MAPK is 1e-4x total protein, and protein is 15% of cell wt," \
"so MAPK is 1.5e-5g/ml = 0.36uM. which is closer to our first estimate." \
"Lets use this."
simundump kpool /kinetics/MAPK/craf-1** 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry hotpink brown 12 8 0
simundump text /kinetics/MAPK/craf-1**/notes 0 \
  "Negative feedback by MAPK* by hyperphosphorylating craf-1* gives\nrise to this pool.\nUeki et al JBC 269(22):15756-15761, 1994\n"
call /kinetics/MAPK/craf-1**/notes LOAD \
"Negative feedback by MAPK* by hyperphosphorylating craf-1* gives" \
"rise to this pool." \
"Ueki et al JBC 269(22):15756-15761, 1994" \
""
simundump kpool /kinetics/MAPK/MAPK-tyr 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry orange brown 9 1 0
simundump text /kinetics/MAPK/MAPK-tyr/notes 0 \
  "Haystead et al FEBS Lett. 306(1) pp 17-22 show that phosphorylation\nis strictly sequential, first tyr185 then thr183."
call /kinetics/MAPK/MAPK-tyr/notes LOAD \
"Haystead et al FEBS Lett. 306(1) pp 17-22 show that phosphorylation" \
"is strictly sequential, first tyr185 then thr183."
simundump kpool /kinetics/MAPK/MAPKK* 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry pink brown 13 4 0
simundump text /kinetics/MAPK/MAPKK*/notes 0 \
  "MAPKK phosphorylates MAPK on both the tyr and thr residues, first\ntyr then thr. Refs: Seger et al JBC267:20 pp 14373 1992\nThe MAPKK itself is phosphorylated on ser as well as thr residues.\nLet us assume that the ser goes first, and that the sequential phosphorylation\nis needed. See Kyriakis et al Nature 358 417-421 1992"
call /kinetics/MAPK/MAPKK*/notes LOAD \
"MAPKK phosphorylates MAPK on both the tyr and thr residues, first" \
"tyr then thr. Refs: Seger et al JBC267:20 pp 14373 1992" \
"The MAPKK itself is phosphorylated on ser as well as thr residues." \
"Let us assume that the ser goes first, and that the sequential phosphorylation" \
"is needed. See Kyriakis et al Nature 358 417-421 1992"
simundump kenz /kinetics/MAPK/MAPKK*/MAPKKtyr 0 0 0 0 0 6e+05 5.4e-05 1.2 0.3 \
  0 0 "" red pink "" 7 3 0
simundump text /kinetics/MAPK/MAPKK*/MAPKKtyr/notes 0 \
  "The actual MAPKK is 2 forms from Seger et al JBC 267:20 14373(1992)\nVmax = 150nmol/min/mg\nFrom Haystead et al FEBS 306(1):17-22 we get Km=46.6nM for at least one\nof the phosphs.\nPutting these together:\nk3=0.15/sec, scale to get k2=0.6.\nk1=0.75/46.6nM=2.7e-5"
call /kinetics/MAPK/MAPKK*/MAPKKtyr/notes LOAD \
"The actual MAPKK is 2 forms from Seger et al JBC 267:20 14373(1992)" \
"Vmax = 150nmol/min/mg" \
"From Haystead et al FEBS 306(1):17-22 we get Km=46.6nM for at least one" \
"of the phosphs." \
"Putting these together:" \
"k3=0.15/sec, scale to get k2=0.6." \
"k1=0.75/46.6nM=2.7e-5"
simundump kenz /kinetics/MAPK/MAPKK*/MAPKKthr 1 0 0 0 0 6e+05 5.4e-05 1.2 0.3 \
  0 0 "" red pink "" 11 3 0
simundump text /kinetics/MAPK/MAPKK*/MAPKKthr/notes 0 \
  "Rate consts same as for MAPKKtyr."
call /kinetics/MAPK/MAPKK*/MAPKKthr/notes LOAD \
"Rate consts same as for MAPKKtyr."
simundump kpool /kinetics/MAPK/MAPKK-ser 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry pink brown 9 4 0
simundump text /kinetics/MAPK/MAPKK-ser/notes 0 \
  "Intermediately phophorylated, assumed inactive, form of MAPKK"
call /kinetics/MAPK/MAPKK-ser/notes LOAD \
"Intermediately phophorylated, assumed inactive, form of MAPKK"
simundump kpool /kinetics/MAPK/Raf-GTP-Ras 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[2] 53 brown 4 6 0
simundump text /kinetics/MAPK/Raf-GTP-Ras/notes 0 ""
call /kinetics/MAPK/Raf-GTP-Ras/notes LOAD \
""
simundump kenz /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1 0 0 0 0 0 6e+05 \
  1.5713e-05 1.2 0.3 0 0 "" red 53 "" 7 6 0
simundump text /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1/notes 0 \
  "Based on acc79.g from Ajay and Bhalla 2007."
call /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1/notes LOAD \
"Based on acc79.g from Ajay and Bhalla 2007."
simundump kenz /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2 0 0 0 0 0 6e+05 \
  1.5713e-05 1.2 0.3 0 0 "" red 53 "" 11 6 0
simundump text /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2/notes 0 \
  "Based on acc79.g\n"
call /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2/notes LOAD \
"Based on acc79.g" \
""
simundump kreac /kinetics/MAPK/Ras-act-unphosph-raf 0 1e-05 1 "" white brown \
  4 9 0
simundump text /kinetics/MAPK/Ras-act-unphosph-raf/notes 0 ""
call /kinetics/MAPK/Ras-act-unphosph-raf/notes LOAD \
""
simundump kreac /kinetics/MAPK/Ras-act-craf 1 1.6667e-05 0.5 "" white brown 3 \
  10 0
simundump text /kinetics/MAPK/Ras-act-craf/notes 0 \
  "Assume the binding is fast and limited only by the amount of \nRas* available. So kf=kb/[craf-1]\nIf kb is 1/sec, then kf = 1/0.2 uM = 1/(0.2 * 6e5) = 8.3e-6\nLater: Raise it by 10 X to 4e-5\nFrom Hallberg et al JBC 269:6 3913-3916 1994, 3% of cellular Raf is\ncomplexed with Ras. So we raise kb 4x to 4\nThis step needed to memb-anchor and activate Raf: Leevers et al Nature\n369 411-414\n(I don't...."
call /kinetics/MAPK/Ras-act-craf/notes LOAD \
"Assume the binding is fast and limited only by the amount of " \
"Ras* available. So kf=kb/[craf-1]" \
"If kb is 1/sec, then kf = 1/0.2 uM = 1/(0.2 * 6e5) = 8.3e-6" \
"Later: Raise it by 10 X to 4e-5" \
"From Hallberg et al JBC 269:6 3913-3916 1994, 3% of cellular Raf is" \
"complexed with Ras. So we raise kb 4x to 4" \
"This step needed to memb-anchor and activate Raf: Leevers et al Nature" \
"369 411-414" \
"(I don't...."
simundump kpool /kinetics/MAPK/Raf*-GTP-Ras 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry red brown 4 5 0
simundump text /kinetics/MAPK/Raf*-GTP-Ras/notes 0 ""
call /kinetics/MAPK/Raf*-GTP-Ras/notes LOAD \
""
simundump kenz /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1 1 0 0 0 0 1 \
  1.5714e-05 1.2 0.3 0 0 "" red red "" 7 5 0
simundump text /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1/notes 0 \
  "Kinetics are the same as for the craf-1* activity, ie.,\nk1=1.1e-6, k2=.42, k3 =0.105\nThese are based on Force et al PNAS USA 91 1270-1274 1994.\nThese parms cannot reach the observed 4X stim of MAPK. So lets\nincrease the affinity, ie, raise k1 10X to 1.1e-5\nLets take it back down to where it was.\nBack up to 5X: 5.5e-6"
call /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1/notes LOAD \
"Kinetics are the same as for the craf-1* activity, ie.," \
"k1=1.1e-6, k2=.42, k3 =0.105" \
"These are based on Force et al PNAS USA 91 1270-1274 1994." \
"These parms cannot reach the observed 4X stim of MAPK. So lets" \
"increase the affinity, ie, raise k1 10X to 1.1e-5" \
"Lets take it back down to where it was." \
"Back up to 5X: 5.5e-6"
simundump kenz /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2 1 0 0 0 0 1 \
  1.5714e-05 1.2 0.3 0 0 "" red red "" 11 5 0
simundump text /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2/notes 0 \
  "Same kinetics as other c-raf activated forms. See \nForce et al PNAS 91 1270-1274 1994.\nk1 = 1.1e-6, k2 = .42, k3 = 1.05\nraise k1 to 5.5e-6\n"
call /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2/notes LOAD \
"Same kinetics as other c-raf activated forms. See " \
"Force et al PNAS 91 1270-1274 1994." \
"k1 = 1.1e-6, k2 = .42, k3 = 1.05" \
"raise k1 to 5.5e-6" \
""
simundump kpool /kinetics/MKP-1 1 0 0.024 0.024 14400 14400 0 0 6e+05 0 \
  /kinetics/geometry hotpink black 5.0816 2.4407 0
simundump text /kinetics/MKP-1/notes 0 \
  "MKP-1 dephosphoryates and inactivates MAPK in vivo: Sun et al Cell 75 \n487-493 1993. Levels of MKP-1\nare regulated, and rise in 1 hour. \nKinetics from Charles et al PNAS 90:5292-5296 1993. They refer\nto Charles et al Oncogene 7 187-190 who show that half-life of MKP1/3CH134\nis 40 min. 80% deph of MAPK in 20 min\nSep 17 1997: CoInit now 0.4x to 0.0032. See parm searches\nfrom jun96 on."
call /kinetics/MKP-1/notes LOAD \
"MKP-1 dephosphoryates and inactivates MAPK in vivo: Sun et al Cell 75 " \
"487-493 1993. Levels of MKP-1" \
"are regulated, and rise in 1 hour. " \
"Kinetics from Charles et al PNAS 90:5292-5296 1993. They refer" \
"to Charles et al Oncogene 7 187-190 who show that half-life of MKP1/3CH134" \
"is 40 min. 80% deph of MAPK in 20 min" \
"Sep 17 1997: CoInit now 0.4x to 0.0032. See parm searches" \
"from jun96 on."
simundump kenz /kinetics/MKP-1/MKP1-tyr-deph 1 0 0 0 0 6e+05 0.00025001 16 4 \
  0 0 "" red hotpink "" 7 2 0
simundump text /kinetics/MKP-1/MKP1-tyr-deph/notes 0 \
  "The original kinetics have been modified to obey the k2 = 4 * k3 rule,\nwhile keeping kcat and Km fixed. As noted in the NOTES, the only constraining\ndata point is the time course of MAPK dephosphorylation, which this\nmodel satisfies. It would be nice to have more accurate estimates of\nrate consts and MKP-1 levels from the literature. \nEffective Km : 67 nM\nkcat = 1.43 umol/min/mg\nRaised k...."
call /kinetics/MKP-1/MKP1-tyr-deph/notes LOAD \
"The original kinetics have been modified to obey the k2 = 4 * k3 rule," \
"while keeping kcat and Km fixed. As noted in the NOTES, the only constraining" \
"data point is the time course of MAPK dephosphorylation, which this" \
"model satisfies. It would be nice to have more accurate estimates of" \
"rate consts and MKP-1 levels from the literature. " \
"Effective Km : 67 nM" \
"kcat = 1.43 umol/min/mg" \
"Raised kcat from 1 to 4, following acc79.g (Ajay and Bhalla07)"
simundump kenz /kinetics/MKP-1/MKP1-thr-deph 1 0 0 0 0 6e+05 0.00025001 16 4 \
  0 0 "" red hotpink "" 11 2 0
simundump text /kinetics/MKP-1/MKP1-thr-deph/notes 0 "See MKP1-tyr-deph"
call /kinetics/MKP-1/MKP1-thr-deph/notes LOAD \
"See MKP1-tyr-deph"
simundump kpool /kinetics/PPhosphatase2A 1 0 1 1 6e+05 6e+05 0 0 6e+05 0 \
  /kinetics/geometry hotpink yellow 9 9 0
simundump text /kinetics/PPhosphatase2A/notes 0 \
  "Refs: Pato et al Biochem J 293:35-41(93);\nTakai&Mieskes Biochem J 275:233-239\nk1=1.46e-4, k2=1000,k3=250. these use\nkcat values for calponin. Also, units of kcat may be in min!\nrevert to Vmax base:\nk3=6, k2=25,k1=3.3e-6 or 6,6,1e-6\nCoInit assumed 0.1 uM.\nSee NOTES for MAPK_Ras50.g. CoInit now 0.08\nSep 17 1997: Raise CoInt 1.4x to 0.224, see parm\nsearches from jun 96 on.\n"
call /kinetics/PPhosphatase2A/notes LOAD \
"Refs: Pato et al Biochem J 293:35-41(93);" \
"Takai&Mieskes Biochem J 275:233-239" \
"k1=1.46e-4, k2=1000,k3=250. these use" \
"kcat values for calponin. Also, units of kcat may be in min!" \
"revert to Vmax base:" \
"k3=6, k2=25,k1=3.3e-6 or 6,6,1e-6" \
"CoInit assumed 0.1 uM." \
"See NOTES for MAPK_Ras50.g. CoInit now 0.08" \
"Sep 17 1997: Raise CoInt 1.4x to 0.224, see parm" \
"searches from jun 96 on." \
""
simundump kenz /kinetics/PPhosphatase2A/craf-deph 1 0 0 0 0 6e+05 3.1935e-06 \
  24 6 0 0 "" red hotpink "" 7 9 0
simundump text /kinetics/PPhosphatase2A/craf-deph/notes 0 \
  "See parent PPhosphatase2A for parms\n"
call /kinetics/PPhosphatase2A/craf-deph/notes LOAD \
"See parent PPhosphatase2A for parms" \
""
simundump kenz /kinetics/PPhosphatase2A/MAPKK-deph 1 0 0 0 0 6e+05 3.1935e-06 \
  24 6 0 0 "" red hotpink "" 11 7 0
simundump text /kinetics/PPhosphatase2A/MAPKK-deph/notes 0 \
  "See: Kyriakis et al Nature 358 pp 417-421 1992\nAhn et al Curr Op Cell Biol 4:992-999 1992 for this pathway.\nSee parent PPhosphatase2A for parms."
call /kinetics/PPhosphatase2A/MAPKK-deph/notes LOAD \
"See: Kyriakis et al Nature 358 pp 417-421 1992" \
"Ahn et al Curr Op Cell Biol 4:992-999 1992 for this pathway." \
"See parent PPhosphatase2A for parms."
simundump kenz /kinetics/PPhosphatase2A/MAPKK-deph-ser 1 0 0 0 0 6e+05 \
  3.1935e-06 24 6 0 0 "" red hotpink "" 7 7 0
simundump text /kinetics/PPhosphatase2A/MAPKK-deph-ser/notes 0 ""
call /kinetics/PPhosphatase2A/MAPKK-deph-ser/notes LOAD \
""
simundump kenz /kinetics/PPhosphatase2A/craf**-deph 1 0 0 0 0 1 3.1935e-06 24 \
  6 0 0 "" red hotpink "" 11 9 0
simundump text /kinetics/PPhosphatase2A/craf**-deph/notes 0 \
  "Ueki et al JBC 269(22) pp 15756-15761 1994 show hyperphosphorylation of\ncraf, so this is there to dephosphorylate it. Identity of phosphatase is not\nknown to me, but it may be PP2A like the rest, so I have made it so."
call /kinetics/PPhosphatase2A/craf**-deph/notes LOAD \
"Ueki et al JBC 269(22) pp 15756-15761 1994 show hyperphosphorylation of" \
"craf, so this is there to dephosphorylate it. Identity of phosphatase is not" \
"known to me, but it may be PP2A like the rest, so I have made it so."
simundump group /kinetics/Ras 1 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 14.513 16.351 0
simundump text /kinetics/Ras/notes 0 \
  "Ras has now gotten to be a big enough component of the model to\ndeserve its own group. The main refs are\nBoguski and McCormick Nature 366 643-654 '93 Major review\nEccleston et al JBC 268:36 pp 27012-19\nOrita et al JBC 268:34 2554246"
call /kinetics/Ras/notes LOAD \
"Ras has now gotten to be a big enough component of the model to" \
"deserve its own group. The main refs are" \
"Boguski and McCormick Nature 366 643-654 '93 Major review" \
"Eccleston et al JBC 268:36 pp 27012-19" \
"Orita et al JBC 268:34 2554246"
simundump kreac /kinetics/Ras/bg-act-GEF 1 1e-05 1 "" white blue 13.468 \
  14.838 0
simundump text /kinetics/Ras/bg-act-GEF/notes 0 \
  "SoS/GEF is present at 50 nM ie 3e4/cell. BetaGamma maxes out at 9e4.\nAssume we have 1/3 of the GEF active when the BetaGamma is 1.5e4.\nso 1e4 * kb = 2e4 * 1.5e4 * kf, so kf/kb = 3e-5. The rate of this equil should\nbe reasonably fast, say 1/sec\n"
call /kinetics/Ras/bg-act-GEF/notes LOAD \
"SoS/GEF is present at 50 nM ie 3e4/cell. BetaGamma maxes out at 9e4." \
"Assume we have 1/3 of the GEF active when the BetaGamma is 1.5e4." \
"so 1e4 * kb = 2e4 * 1.5e4 * kf, so kf/kb = 3e-5. The rate of this equil should" \
"be reasonably fast, say 1/sec" \
""
simundump kpool /kinetics/Ras/GEF-Gprot-bg 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry hotpink blue 12 16 0
simundump text /kinetics/Ras/GEF-Gprot-bg/notes 0 \
  "Guanine nucleotide exchange factor. This activates raf by exchanging bound\nGDP with GTP. I have left the GDP/GTP out of this reaction, it would be\ntrivial to put them in. See Boguski & McCormick.\nPossible candidate molecules: RasGRF, smgGDS, Vav (in dispute). \nrasGRF: Kcat= 1.2/min    Km = 680 nM\nsmgGDS: Kcat: 0.37 /min, Km = 220 nM.\nvav: Turnover up over baseline by 10X, \n"
call /kinetics/Ras/GEF-Gprot-bg/notes LOAD \
"Guanine nucleotide exchange factor. This activates raf by exchanging bound" \
"GDP with GTP. I have left the GDP/GTP out of this reaction, it would be" \
"trivial to put them in. See Boguski & McCormick." \
"Possible candidate molecules: RasGRF, smgGDS, Vav (in dispute). " \
"rasGRF: Kcat= 1.2/min    Km = 680 nM" \
"smgGDS: Kcat: 0.37 /min, Km = 220 nM." \
"vav: Turnover up over baseline by 10X, " \
""
simundump kenz /kinetics/Ras/GEF-Gprot-bg/GEF-bg_act-ras 1 0 0 0 0 6e+05 \
  3.3e-07 0.08 0.02 0 0 "" red hotpink "" 9 16 0
simundump text /kinetics/Ras/GEF-Gprot-bg/GEF-bg_act-ras/notes 0 \
  "Kinetics based on the activation of Gq by the receptor complex in the\nGq model (in turn based on the Mahama and Linderman model)\nk1 = 2e-5, k2 = 1e-10, k3 = 10 (I do not know why they even bother with k2).\nLets put k1 at 2e-6 to get a reasonable equilibrium\nMore specific values from, eg.g: Orita et al JBC 268(34) 25542-25546\nfrom rasGRF and smgGDS: k1=3.3e-7; k2 = 0.08, k3 = 0.02\n"
call /kinetics/Ras/GEF-Gprot-bg/GEF-bg_act-ras/notes LOAD \
"Kinetics based on the activation of Gq by the receptor complex in the" \
"Gq model (in turn based on the Mahama and Linderman model)" \
"k1 = 2e-5, k2 = 1e-10, k3 = 10 (I do not know why they even bother with k2)." \
"Lets put k1 at 2e-6 to get a reasonable equilibrium" \
"More specific values from, eg.g: Orita et al JBC 268(34) 25542-25546" \
"from rasGRF and smgGDS: k1=3.3e-7; k2 = 0.08, k3 = 0.02" \
""
simundump kreac /kinetics/Ras/dephosph-GEF 1 1 0 "" white blue 9.0702 17.881 \
  0
simundump text /kinetics/Ras/dephosph-GEF/notes 0 ""
call /kinetics/Ras/dephosph-GEF/notes LOAD \
""
simundump kpool /kinetics/Ras/inact-GEF 1 0 0.1 0.1 60000 60000 0 0 6e+05 0 \
  /kinetics/geometry hotpink blue 12 18 0
simundump text /kinetics/Ras/inact-GEF/notes 0 \
  "Assume that SoS is present only at 50 nM.\nRevised to 100 nM to get equil to experimentally known levels."
call /kinetics/Ras/inact-GEF/notes LOAD \
"Assume that SoS is present only at 50 nM." \
"Revised to 100 nM to get equil to experimentally known levels."
simundump kpool /kinetics/Ras/GEF* 1 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry \
  hotpink blue 6 15 0
simundump text /kinetics/Ras/GEF*/notes 0 \
  "phosphorylated and thereby activated form of GEF. See, e.g.\nOrita et al JBC 268:34 25542-25546 1993, Gulbins et al.\nIt is not clear whether there is major specificity for tyr or ser/thr."
call /kinetics/Ras/GEF*/notes LOAD \
"phosphorylated and thereby activated form of GEF. See, e.g." \
"Orita et al JBC 268:34 25542-25546 1993, Gulbins et al." \
"It is not clear whether there is major specificity for tyr or ser/thr."
simundump kenz /kinetics/Ras/GEF*/GEF*-act-ras 1 0 0 0 0 6e+05 3.3e-07 0.08 \
  0.02 0 0 "" red hotpink "" 9 15 0
simundump text /kinetics/Ras/GEF*/GEF*-act-ras/notes 0 \
  "Kinetics same as GEF-bg-act-ras\n"
call /kinetics/Ras/GEF*/GEF*-act-ras/notes LOAD \
"Kinetics same as GEF-bg-act-ras" \
""
simundump kpool /kinetics/Ras/GTP-Ras 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry orange blue 12 14 0
simundump text /kinetics/Ras/GTP-Ras/notes 0 \
  "Only a very small fraction (7% unstim, 15% stim) of ras is GTP-bound.\nGibbs et al JBC 265(33) 20437\n"
call /kinetics/Ras/GTP-Ras/notes LOAD \
"Only a very small fraction (7% unstim, 15% stim) of ras is GTP-bound." \
"Gibbs et al JBC 265(33) 20437" \
""
simundump kpool /kinetics/Ras/GDP-Ras 1 0 0.5 0.5 3e+05 3e+05 0 0 6e+05 0 \
  /kinetics/geometry pink blue 6 14 0
simundump text /kinetics/Ras/GDP-Ras/notes 0 \
  "GDP bound form. See Rosen et al Neuron 12 1207-1221 June 1994.\nthe activation loop is based on Boguski and McCormick Nature 366 643-654 93\nAssume Ras is present at about the same level as craf-1, 0.2 uM.\nHallberg et al JBC 269:6 3913-3916 1994 estimate upto 5-10% of cellular\nRaf is assoc with Ras. Given that only 5-10% of Ras is GTP-bound, we\nneed similar amounts of Ras as Raf."
call /kinetics/Ras/GDP-Ras/notes LOAD \
"GDP bound form. See Rosen et al Neuron 12 1207-1221 June 1994." \
"the activation loop is based on Boguski and McCormick Nature 366 643-654 93" \
"Assume Ras is present at about the same level as craf-1, 0.2 uM." \
"Hallberg et al JBC 269:6 3913-3916 1994 estimate upto 5-10% of cellular" \
"Raf is assoc with Ras. Given that only 5-10% of Ras is GTP-bound, we" \
"need similar amounts of Ras as Raf."
simundump kreac /kinetics/Ras/Ras-intrinsic-GTPase 1 0.0001 0 "" white blue \
  9.0979 13.5 0
simundump text /kinetics/Ras/Ras-intrinsic-GTPase/notes 0 \
  "This is extremely slow (1e-4), but it is significant as so little GAP actually\ngets complexed with it that the total GTP turnover rises only by\n2-3 X (see Gibbs et al, JBC 265(33) 20437-20422) and \nEccleston et al JBC 268(36) 27012-27019\nkf = 1e-4\n"
call /kinetics/Ras/Ras-intrinsic-GTPase/notes LOAD \
"This is extremely slow (1e-4), but it is significant as so little GAP actually" \
"gets complexed with it that the total GTP turnover rises only by" \
"2-3 X (see Gibbs et al, JBC 265(33) 20437-20422) and " \
"Eccleston et al JBC 268(36) 27012-27019" \
"kf = 1e-4" \
""
simundump kreac /kinetics/Ras/dephosph-GAP 1 0.1 0 "" white blue 4 11 0
simundump text /kinetics/Ras/dephosph-GAP/notes 0 \
  "Assume a reasonably good rate for dephosphorylating it, 1/sec"
call /kinetics/Ras/dephosph-GAP/notes LOAD \
"Assume a reasonably good rate for dephosphorylating it, 1/sec"
simundump kpool /kinetics/Ras/GAP* 1 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry \
  red blue 2 12 0
simundump text /kinetics/Ras/GAP*/notes 0 ""
call /kinetics/Ras/GAP*/notes LOAD \
""
simundump kpool /kinetics/Ras/GAP 1 0 0.01 0.01 6000 6000 0 0 6e+05 0 \
  /kinetics/geometry red blue 6 12 0
simundump text /kinetics/Ras/GAP/notes 0 \
  "GTPase-activating proteins. See Boguski and McCormick.\nTurn off Ras by helping to hydrolyze bound GTP. \nThis one is probably NF1, ie.,  Neurofibromin as it is inhibited by AA and lipids,\nand expressed in neural cells. p120-GAP is also a possible candidate, but\nis less regulated. Both may exist at similar levels.\nSee Eccleston et al JBC 268(36) pp27012-19\nLevel=.002"
call /kinetics/Ras/GAP/notes LOAD \
"GTPase-activating proteins. See Boguski and McCormick." \
"Turn off Ras by helping to hydrolyze bound GTP. " \
"This one is probably NF1, ie.,  Neurofibromin as it is inhibited by AA and lipids," \
"and expressed in neural cells. p120-GAP is also a possible candidate, but" \
"is less regulated. Both may exist at similar levels." \
"See Eccleston et al JBC 268(36) pp27012-19" \
"Level=.002"
simundump kenz /kinetics/Ras/GAP/GAP-inact-ras 1 0 0 0 0 6e+05 8.2476e-05 40 \
  10 0 0 "" red red "" 9 12 0
simundump text /kinetics/Ras/GAP/GAP-inact-ras/notes 0 \
  "From Eccleston et al JBC 268(36)pp27012-19 get Kd < 2uM, kcat - 10/sec\nFrom Martin et al Cell 63 843-849 1990 get Kd ~ 250 nM, kcat = 20/min\nI will go with the Eccleston figures as there are good error bars (10%). In general\nthe values are reasonably close.\nk1 = 1.666e-3/sec, k2 = 1000/sec, k3 = 10/sec (note k3 is rate-limiting)\n5 Nov 2002: Changed ratio term to 4 from 100. Now we have\nk...."
call /kinetics/Ras/GAP/GAP-inact-ras/notes LOAD \
"From Eccleston et al JBC 268(36)pp27012-19 get Kd < 2uM, kcat - 10/sec" \
"From Martin et al Cell 63 843-849 1990 get Kd ~ 250 nM, kcat = 20/min" \
"I will go with the Eccleston figures as there are good error bars (10%). In general" \
"the values are reasonably close." \
"k1 = 1.666e-3/sec, k2 = 1000/sec, k3 = 10/sec (note k3 is rate-limiting)" \
"5 Nov 2002: Changed ratio term to 4 from 100. Now we have" \
"k1=8.25e-5; k2=40, k3=10. k3 is still rate-limiting."
simundump kpool /kinetics/Ras/inact-GEF* 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry hotpink blue 15 18 0
simundump text /kinetics/Ras/inact-GEF*/notes 0 \
  "Phosphorylation-inactivated form of GEF. See\nHordijk et al JBC 269:5 3534-3538 1994\nand \nBuregering et al EMBO J 12:11 4211-4220 1993\n"
call /kinetics/Ras/inact-GEF*/notes LOAD \
"Phosphorylation-inactivated form of GEF. See" \
"Hordijk et al JBC 269:5 3534-3538 1994" \
"and " \
"Buregering et al EMBO J 12:11 4211-4220 1993" \
""
simundump kreac /kinetics/Ras/CaM-bind-GEF 1 0.00033333 1 "" white blue 9 20 \
  0
simundump text /kinetics/Ras/CaM-bind-GEF/notes 0 \
  "Nov 2008: Updated based on acc79.g from Ajay and Bhalla 2007.\n\nWe have no numbers for this. It is probably between\nthe two extremes represented by the CaMKII phosph states,\nand I have used guesses based on this.\nkf=1e-4\nkb=1\nThe reaction is based on Farnsworth et al Nature 376 524-527\n1995"
call /kinetics/Ras/CaM-bind-GEF/notes LOAD \
"Nov 2008: Updated based on acc79.g from Ajay and Bhalla 2007." \
"" \
"We have no numbers for this. It is probably between" \
"the two extremes represented by the CaMKII phosph states," \
"and I have used guesses based on this." \
"kf=1e-4" \
"kb=1" \
"The reaction is based on Farnsworth et al Nature 376 524-527" \
"1995"
simundump kpool /kinetics/Ras/CaM-GEF 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry pink blue 6 17 0
simundump text /kinetics/Ras/CaM-GEF/notes 0 \
  "See Farnsworth et al Nature 376 524-527 1995"
call /kinetics/Ras/CaM-GEF/notes LOAD \
"See Farnsworth et al Nature 376 524-527 1995"
simundump kenz /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras 1 0 0 0 0 6e+05 1.65e-06 \
  0.4 0.1 0 0 "" red pink "" 9 17 0
simundump text /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras/notes 0 \
  "Kinetics same as GEF-bg_act-ras\n"
call /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras/notes LOAD \
"Kinetics same as GEF-bg_act-ras" \
""
simundump kreac /kinetics/Ras/dephosph-inact-GEF* 1 1 0 "" white blue 13 20 0
simundump text /kinetics/Ras/dephosph-inact-GEF*/notes 0 ""
call /kinetics/Ras/dephosph-inact-GEF*/notes LOAD \
""
simundump kpool /kinetics/PKA-active 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry yellow black -33.585 -12.858 0
simundump text /kinetics/PKA-active/notes 0 ""
call /kinetics/PKA-active/notes LOAD \
""
simundump kenz /kinetics/PKA-active/PKA-phosph-GEF 1 0 0 0 0 6e+05 1e-05 36 9 \
  0 0 "" red yellow "" 13 21 0
simundump text /kinetics/PKA-active/PKA-phosph-GEF/notes 0 \
  "This pathway inhibits Ras when cAMP is elevated. See:\nHordijk et al JBC 269:5 3534-3538 1994\nBurgering et al EMBO J 12:11 4211-4220 1993\nThe rates are the same as used in PKA-phosph-I1"
call /kinetics/PKA-active/PKA-phosph-GEF/notes LOAD \
"This pathway inhibits Ras when cAMP is elevated. See:" \
"Hordijk et al JBC 269:5 3534-3538 1994" \
"Burgering et al EMBO J 12:11 4211-4220 1993" \
"The rates are the same as used in PKA-phosph-I1"
simundump kenz /kinetics/PKA-active/PKA-phosph-I1 1 0 0 0 0 6e+05 1e-05 36 9 \
  0 0 "" red yellow "" -36.894 -17.114 0
simundump text /kinetics/PKA-active/PKA-phosph-I1/notes 0 \
  "#s from Bramson et al CRC crit rev Biochem\n15:2 93-124. They have a huge list of peptide substrates\nand I have chosen high-ish rates.\nThese consts give too much PKA activity, so lower Vmax 1/3.\nNow, k1 = 3e-5, k2 = 36, k3 = 9 (still pretty fast).\nAlso lower Km 1/3  so k1 = 1e-5\nCohen et al FEBS Lett 76:182-86 1977 say rate =30% PKA act on \nphosphokinase beta.\n"
call /kinetics/PKA-active/PKA-phosph-I1/notes LOAD \
"#s from Bramson et al CRC crit rev Biochem" \
"15:2 93-124. They have a huge list of peptide substrates" \
"and I have chosen high-ish rates." \
"These consts give too much PKA activity, so lower Vmax 1/3." \
"Now, k1 = 3e-5, k2 = 36, k3 = 9 (still pretty fast)." \
"Also lower Km 1/3  so k1 = 1e-5" \
"Cohen et al FEBS Lett 76:182-86 1977 say rate =30% PKA act on " \
"phosphokinase beta." \
""
simundump kenz /kinetics/PKA-active/phosph-PDE 1 0 0 0 0 6e+05 1e-05 36 9 0 0 \
  "" red yellow "" -30.934 -13.317 0
simundump text /kinetics/PKA-active/phosph-PDE/notes 0 \
  "Same rates as PKA-phosph-I1"
call /kinetics/PKA-active/phosph-PDE/notes LOAD \
"Same rates as PKA-phosph-I1"
simundump kpool /kinetics/CaM-Ca4 1 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry \
  blue yellow -22.075 -2.8669 0
simundump text /kinetics/CaM-Ca4/notes 0 ""
call /kinetics/CaM-Ca4/notes LOAD \
""
simundump kpool /kinetics/Shc*.Sos.Grb2 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry brown yellow 9 27 0
simundump text /kinetics/Shc*.Sos.Grb2/notes 0 ""
call /kinetics/Shc*.Sos.Grb2/notes LOAD \
""
simundump kenz /kinetics/Shc*.Sos.Grb2/Sos.Ras_GEF 1 0 0 0 0 6e+05 3.3e-07 \
  0.08 0.02 0 0 "" red brown "" 8.737 25.888 0
simundump text /kinetics/Shc*.Sos.Grb2/Sos.Ras_GEF/notes 0 ""
call /kinetics/Shc*.Sos.Grb2/Sos.Ras_GEF/notes LOAD \
""
simundump group /kinetics/EGFR 1 yellow black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 7.0249 39.57 0
simundump text /kinetics/EGFR/notes 0 ""
call /kinetics/EGFR/notes LOAD \
""
simundump kpool /kinetics/EGFR/EGFR 1 0 0.16667 0.16667 1e+05 1e+05 0 0 6e+05 \
  0 /kinetics/geometry red yellow 1.9551 39.853 0
simundump text /kinetics/EGFR/EGFR/notes 0 \
  "Berkers et al JBC 266 say 22K hi aff recs.\nSherrill and Kyte Biochemistry 35 use range 4-200 nM.\nThese match, lets use them."
call /kinetics/EGFR/EGFR/notes LOAD \
"Berkers et al JBC 266 say 22K hi aff recs." \
"Sherrill and Kyte Biochemistry 35 use range 4-200 nM." \
"These match, lets use them."
simundump kreac /kinetics/EGFR/act_EGFR 1 7e-06 0.25 "" white yellow 4.4894 \
  38.493 0
simundump text /kinetics/EGFR/act_EGFR/notes 0 \
  "Affinity of EGFR for EGF is complex: depends on [EGFR].\nWe'll assume fixed [EGFR] and use exptal\naffinity ~20 nM (see Sherrill and Kyte\nBiochem 1996 35 5705-5718, Berkers et al JBC 266:2 922-927\n1991, Sorokin et al JBC 269:13 9752-9759 1994). \nTau =~2 min (Davis et al JBC 263:11 5373-5379 1988)\nor Berkers Kass = 6.2e5/M/sec, Kdiss=3.5e-4/sec.\nSherrill and Kyte have Hill Coeff=1.7\n"
call /kinetics/EGFR/act_EGFR/notes LOAD \
"Affinity of EGFR for EGF is complex: depends on [EGFR]." \
"We'll assume fixed [EGFR] and use exptal" \
"affinity ~20 nM (see Sherrill and Kyte" \
"Biochem 1996 35 5705-5718, Berkers et al JBC 266:2 922-927" \
"1991, Sorokin et al JBC 269:13 9752-9759 1994). " \
"Tau =~2 min (Davis et al JBC 263:11 5373-5379 1988)" \
"or Berkers Kass = 6.2e5/M/sec, Kdiss=3.5e-4/sec." \
"Sherrill and Kyte have Hill Coeff=1.7" \
""
simundump kpool /kinetics/EGFR/L.EGFR 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry red yellow 6.2195 36.599 0
simundump text /kinetics/EGFR/L.EGFR/notes 0 \
  "This is terribly simplified: there are many interesting\nintermediate stages, including dimerization and assoc\nwith adapter molecules like Shc, that contribute to the\nactivation of the EGFR."
call /kinetics/EGFR/L.EGFR/notes LOAD \
"This is terribly simplified: there are many interesting" \
"intermediate stages, including dimerization and assoc" \
"with adapter molecules like Shc, that contribute to the" \
"activation of the EGFR."
simundump kenz /kinetics/EGFR/L.EGFR/phosph_PLC_g 1 0 0 0 0 6e+05 5e-06 0.8 \
  0.2 0 0 "" red red "" 6.3358 35.082 0
simundump text /kinetics/EGFR/L.EGFR/phosph_PLC_g/notes 0 \
  "Hsu et al JBC 266:1 603-608 1991\nKm = 385 +- 100 uM, Vm = 5.1 +-1 pmol/min/ug for PLC-771.\nOther sites have similar range, but are not stim as much\nby EGF.\nk1 = 2.8e-2/385/6e5 = 1.2e-10. Phenomenally slow.\nBut Sherrill and Kyte say turnover # for angiotensin II is\n5/min for cell extt, and 2/min for placental. Also see\nOkada et al for Shc rates which are much faster."
call /kinetics/EGFR/L.EGFR/phosph_PLC_g/notes LOAD \
"Hsu et al JBC 266:1 603-608 1991" \
"Km = 385 +- 100 uM, Vm = 5.1 +-1 pmol/min/ug for PLC-771." \
"Other sites have similar range, but are not stim as much" \
"by EGF." \
"k1 = 2.8e-2/385/6e5 = 1.2e-10. Phenomenally slow." \
"But Sherrill and Kyte say turnover # for angiotensin II is" \
"5/min for cell extt, and 2/min for placental. Also see" \
"Okada et al for Shc rates which are much faster."
simundump kenz /kinetics/EGFR/L.EGFR/phosph_Shc 1 0 0 0 0 6e+05 2e-06 0.8 0.2 \
  0 0 "" red red "" 9.0331 36.49 0
simundump text /kinetics/EGFR/L.EGFR/phosph_Shc/notes 0 \
  "Rates from Okada et al JBC 270:35 pp 20737 1995\nKm = 0.70 to 0.85 uM, Vmax = 4.4 to 5.0 pmol/min. Unfortunately\nthe amount of enzyme is not known, the prep is only\npartially purified.\nTime course of phosph is max within 30 sec, falls back within\n20 min. Ref: Sasaoka et al JBC 269:51 32621 1994.\nUse k3 = 0.1 based on this tau.\n"
call /kinetics/EGFR/L.EGFR/phosph_Shc/notes LOAD \
"Rates from Okada et al JBC 270:35 pp 20737 1995" \
"Km = 0.70 to 0.85 uM, Vmax = 4.4 to 5.0 pmol/min. Unfortunately" \
"the amount of enzyme is not known, the prep is only" \
"partially purified." \
"Time course of phosph is max within 30 sec, falls back within" \
"20 min. Ref: Sasaoka et al JBC 269:51 32621 1994." \
"Use k3 = 0.1 based on this tau." \
""
simundump kpool /kinetics/EGFR/EGF 1 0 0 0 0 0 0 0 6e+05 4 /kinetics/geometry \
  red yellow 2.2719 36.309 0
simundump text /kinetics/EGFR/EGF/notes 0 ""
call /kinetics/EGFR/EGF/notes LOAD \
""
simundump kpool /kinetics/EGFR/SHC 1 0 0.5 0.5 3e+05 3e+05 0 0 6e+05 0 \
  /kinetics/geometry orange yellow 8.3857 33.936 0
simundump text /kinetics/EGFR/SHC/notes 0 \
  "There are 2 isoforms: 52 KDa and 46 KDa (See Okada et al\nJBC 270:35 pp 20737 1995). They are acted up on by the EGFR\nin very similar ways, and apparently both bind Grb2 similarly,\nso we'll bundle them together here.\nSasaoka et al JBC 269:51 pp 32621 1994 show immunoprecs where\nit looks like there is at least as much Shc as Grb2. So\nwe'll tentatively say there is 1 uM of Shc."
call /kinetics/EGFR/SHC/notes LOAD \
"There are 2 isoforms: 52 KDa and 46 KDa (See Okada et al" \
"JBC 270:35 pp 20737 1995). They are acted up on by the EGFR" \
"in very similar ways, and apparently both bind Grb2 similarly," \
"so we'll bundle them together here." \
"Sasaoka et al JBC 269:51 pp 32621 1994 show immunoprecs where" \
"it looks like there is at least as much Shc as Grb2. So" \
"we'll tentatively say there is 1 uM of Shc."
simundump kpool /kinetics/EGFR/SHC* 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry orange yellow 12.832 33.029 0
simundump text /kinetics/EGFR/SHC*/notes 0 ""
call /kinetics/EGFR/SHC*/notes LOAD \
""
simundump kreac /kinetics/EGFR/dephosph_Shc 1 0.0016667 0 "" white yellow \
  9.7373 31.442 0
simundump text /kinetics/EGFR/dephosph_Shc/notes 0 \
  "Time course of decline of phosph is 20 min. Part of this is\nthe turnoff time of the EGFR itself. Lets assume a tau of\n10 min for this dephosph. It may be wildly off."
call /kinetics/EGFR/dephosph_Shc/notes LOAD \
"Time course of decline of phosph is 20 min. Part of this is" \
"the turnoff time of the EGFR itself. Lets assume a tau of" \
"10 min for this dephosph. It may be wildly off."
simundump kpool /kinetics/EGFR/Internal_L.EGFR 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry red yellow 6.3061 41.93 0
simundump text /kinetics/EGFR/Internal_L.EGFR/notes 0 ""
call /kinetics/EGFR/Internal_L.EGFR/notes LOAD \
""
simundump kreac /kinetics/EGFR/Internalize 1 0.002 0.00033 "" white yellow \
  4.5213 39.863 0
simundump text /kinetics/EGFR/Internalize/notes 0 \
  "See Helin and Beguinot JBC 266:13 1991 pg 8363-8368.\nIn Fig 3 they have internalization tau about 10 min, \nequil at about 20% EGF available. So kf = 4x kb, and\n1/(kf + kb) = 600 sec so kb = 1/3K = 3.3e-4,\nand kf = 1.33e-3. This doesn't take into account the\nunbound receptor, so we need to push the kf up a bit, to\n0.002"
call /kinetics/EGFR/Internalize/notes LOAD \
"See Helin and Beguinot JBC 266:13 1991 pg 8363-8368." \
"In Fig 3 they have internalization tau about 10 min, " \
"equil at about 20% EGF available. So kf = 4x kb, and" \
"1/(kf + kb) = 600 sec so kb = 1/3K = 3.3e-4," \
"and kf = 1.33e-3. This doesn't take into account the" \
"unbound receptor, so we need to push the kf up a bit, to" \
"0.002"
simundump group /kinetics/Sos 1 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 19.547 34.811 0
simundump text /kinetics/Sos/notes 0 ""
call /kinetics/Sos/notes LOAD \
""
simundump kreac /kinetics/Sos/Shc_bind_Sos.Grb2 1 8.333e-07 0.1 "" white blue \
  10.23 29.891 0
simundump text /kinetics/Sos/Shc_bind_Sos.Grb2/notes 0 \
  "Sasaoka et al JBC 269:51 pp 32621 1994, table on pg\n32623 indicates that this pathway accounts for about \n50% of the GEF activation. (88% - 39%). Error is large,\nabout 20%. Fig 1 is most useful in constraining rates.\n\nChook et al JBC 271:48 pp 30472, 1996 say that the Kd is\n0.2 uM for Shc binding to EGFR. The Kd for Grb direct binding\nis 0.7, so we'll ignore it."
call /kinetics/Sos/Shc_bind_Sos.Grb2/notes LOAD \
"Sasaoka et al JBC 269:51 pp 32621 1994, table on pg" \
"32623 indicates that this pathway accounts for about " \
"50% of the GEF activation. (88% - 39%). Error is large," \
"about 20%. Fig 1 is most useful in constraining rates." \
"" \
"Chook et al JBC 271:48 pp 30472, 1996 say that the Kd is" \
"0.2 uM for Shc binding to EGFR. The Kd for Grb direct binding" \
"is 0.7, so we'll ignore it."
simundump kpool /kinetics/Sos/Sos*.Grb2 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry orange blue 12.274 41.661 0
simundump text /kinetics/Sos/Sos*.Grb2/notes 0 ""
call /kinetics/Sos/Sos*.Grb2/notes LOAD \
""
simundump kreac /kinetics/Sos/Grb2_bind_Sos* 1 4.1667e-08 0.0168 "" white \
  blue 10.533 38.235 0
simundump text /kinetics/Sos/Grb2_bind_Sos*/notes 0 \
  "Same rates as Grb2_bind_Sos: Porfiri and McCormick JBC\n271:10 pp 5871 1996 show that the binding is not affected\nby the phosph."
call /kinetics/Sos/Grb2_bind_Sos*/notes LOAD \
"Same rates as Grb2_bind_Sos: Porfiri and McCormick JBC" \
"271:10 pp 5871 1996 show that the binding is not affected" \
"by the phosph."
simundump kpool /kinetics/Sos/Grb2 1 0 1 1 6e+05 6e+05 0 0 6e+05 0 \
  /kinetics/geometry orange blue 14.742 35.301 0
simundump text /kinetics/Sos/Grb2/notes 0 \
  "There is probably a lot of it in the cell: it is also known\nas Ash (abundant src homology protein I think). Also \nWaters et al JBC 271:30 18224 1996 say that only a small\nfraction of cellular Grb is precipitated out when SoS is\nprecipitated. As most of the Sos seems to be associated\nwith Grb2, it would seem like there is a lot of the latter.\nSay 1 uM. I haven't been able to find a decent...."
call /kinetics/Sos/Grb2/notes LOAD \
"There is probably a lot of it in the cell: it is also known" \
"as Ash (abundant src homology protein I think). Also " \
"Waters et al JBC 271:30 18224 1996 say that only a small" \
"fraction of cellular Grb is precipitated out when SoS is" \
"precipitated. As most of the Sos seems to be associated" \
"with Grb2, it would seem like there is a lot of the latter." \
"Say 1 uM. I haven't been able to find a decent...."
simundump kpool /kinetics/Sos/Sos.Grb2 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry orange blue 13.988 30.097 0
simundump text /kinetics/Sos/Sos.Grb2/notes 0 ""
call /kinetics/Sos/Sos.Grb2/notes LOAD \
""
simundump kpool /kinetics/Sos/Sos* 1 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry \
  red blue 15.421 40.215 0
simundump text /kinetics/Sos/Sos*/notes 0 ""
call /kinetics/Sos/Sos*/notes LOAD \
""
simundump kreac /kinetics/Sos/dephosph_Sos 1 0.001 0 "" white blue 13.185 \
  37.153 0
simundump text /kinetics/Sos/dephosph_Sos/notes 0 \
  "The only clue I have to these rates is from the time\ncourses of the EGF activation, which is around 1 to 5 min.\nThe dephosph would be expected to be of the same order,\nperhaps a bit longer. Lets use 0.002 which is about 8 min.\nSep 17: The transient activation curve matches better with\nkf = 0.001"
call /kinetics/Sos/dephosph_Sos/notes LOAD \
"The only clue I have to these rates is from the time" \
"courses of the EGF activation, which is around 1 to 5 min." \
"The dephosph would be expected to be of the same order," \
"perhaps a bit longer. Lets use 0.002 which is about 8 min." \
"Sep 17: The transient activation curve matches better with" \
"kf = 0.001"
simundump kreac /kinetics/Sos/Grb2_bind_Sos 1 4.1667e-08 0.0168 "" white blue \
  16.422 33.133 0
simundump text /kinetics/Sos/Grb2_bind_Sos/notes 0 \
  "As there are 2 SH3 domains, this reaction could be 2nd order.\nI have a Kd of 22 uM from peptide binding (Lemmon et al \nJBC 269:50 pg 31653). However, Chook et al JBC 271:48 pg30472\nsay it is 0.4uM with purified proteins, so we believe them.\nThey say it is 1:1 binding."
call /kinetics/Sos/Grb2_bind_Sos/notes LOAD \
"As there are 2 SH3 domains, this reaction could be 2nd order." \
"I have a Kd of 22 uM from peptide binding (Lemmon et al " \
"JBC 269:50 pg 31653). However, Chook et al JBC 271:48 pg30472" \
"say it is 0.4uM with purified proteins, so we believe them." \
"They say it is 1:1 binding."
simundump kpool /kinetics/Sos/Sos 1 0 0.1 0.1 60000 60000 0 0 6e+05 0 \
  /kinetics/geometry red blue 17.381 36.794 0
simundump text /kinetics/Sos/Sos/notes 0 \
  "I have tried using low (0.02 uM) initial concs, but these\ngive a very flat response to EGF stim although the overall\nactivation of Ras is not too bad. I am reverting to 0.1 \nbecause we expect a sharp initial response, followed by\na decline.\nSep 17 1997: The transient activation curve looks better with\n[Sos] = 0.05.\nApr 26 1998: Some error there, it is better where it was\nat 0.1"
call /kinetics/Sos/Sos/notes LOAD \
"I have tried using low (0.02 uM) initial concs, but these" \
"give a very flat response to EGF stim although the overall" \
"activation of Ras is not too bad. I am reverting to 0.1 " \
"because we expect a sharp initial response, followed by" \
"a decline." \
"Sep 17 1997: The transient activation curve looks better with" \
"[Sos] = 0.05." \
"Apr 26 1998: Some error there, it is better where it was" \
"at 0.1"
simundump group /kinetics/PLC_g 1 darkgreen black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 0.44974 33.831 0
simundump text /kinetics/PLC_g/notes 0 ""
call /kinetics/PLC_g/notes LOAD \
""
simundump kpool /kinetics/PLC_g/PLC_g 1 0 0.82 0.82 4.92e+05 4.92e+05 0 0 \
  6e+05 0 /kinetics/geometry pink darkgreen 0.07993 31.598 0
simundump text /kinetics/PLC_g/PLC_g/notes 0 \
  "Amount from Homma et al JBC 263:14 6592-6598 1988"
call /kinetics/PLC_g/PLC_g/notes LOAD \
"Amount from Homma et al JBC 263:14 6592-6598 1988"
simundump kreac /kinetics/PLC_g/Ca_act_PLC_g 1 0.0003 10 "" white darkgreen \
  -1.4451 28.194 0
simundump text /kinetics/PLC_g/Ca_act_PLC_g/notes 0 \
  "Nice curves from Homma et al JBC 263:14 6592-6598 1988 \nFig 5c. The activity falls above 10 uM, but that is too high\nto reach physiologically anyway, so we'll ignore the higher\npts and match the lower ones only. Half-max at 1 uM.\nBut  Wahl et al JBC 267:15 10447-10456 1992 have half-max\nat 56 nM which is what I'll use."
call /kinetics/PLC_g/Ca_act_PLC_g/notes LOAD \
"Nice curves from Homma et al JBC 263:14 6592-6598 1988 " \
"Fig 5c. The activity falls above 10 uM, but that is too high" \
"to reach physiologically anyway, so we'll ignore the higher" \
"pts and match the lower ones only. Half-max at 1 uM." \
"But  Wahl et al JBC 267:15 10447-10456 1992 have half-max" \
"at 56 nM which is what I'll use."
simundump kreac /kinetics/PLC_g/Ca_act_PLC_g* 1 2e-05 10 "" white darkgreen \
  2.7901 29.8 0
simundump text /kinetics/PLC_g/Ca_act_PLC_g*/notes 0 \
  "Again, we refer to Homma et al and Wahl et al, for preference\nusing Wahl. Half-Max of the phosph form is at 316 nM.\nUse kb of 10 as this is likely to be pretty fast.\nDid some curve comparisons, and instead of 316 nM giving\na kf of 5.27e-5, we will use 8e-5 for kf. \n16 Sep 97. As we are now phosphorylating the Ca-bound form,\nequils have shifted. kf should now be 2e-5 to match the\ncurves."
call /kinetics/PLC_g/Ca_act_PLC_g*/notes LOAD \
"Again, we refer to Homma et al and Wahl et al, for preference" \
"using Wahl. Half-Max of the phosph form is at 316 nM." \
"Use kb of 10 as this is likely to be pretty fast." \
"Did some curve comparisons, and instead of 316 nM giving" \
"a kf of 5.27e-5, we will use 8e-5 for kf. " \
"16 Sep 97. As we are now phosphorylating the Ca-bound form," \
"equils have shifted. kf should now be 2e-5 to match the" \
"curves."
simundump kreac /kinetics/PLC_g/dephosph_PLC_g 1 0.05 0 "" white darkgreen \
  4.5589 32.225 0
simundump text /kinetics/PLC_g/dephosph_PLC_g/notes 0 ""
call /kinetics/PLC_g/dephosph_PLC_g/notes LOAD \
""
simundump kpool /kinetics/PLC_g/PLC_G* 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry pink darkgreen 7.1385 31.319 0
simundump text /kinetics/PLC_g/PLC_G*/notes 0 ""
call /kinetics/PLC_g/PLC_G*/notes LOAD \
""
simundump kpool /kinetics/PLC_g/Ca.PLC_g 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry pink darkgreen 2.0998 27.462 0
simundump text /kinetics/PLC_g/Ca.PLC_g/notes 0 ""
call /kinetics/PLC_g/Ca.PLC_g/notes LOAD \
""
simundump kenz /kinetics/PLC_g/Ca.PLC_g/PIP2_hydrolysis 1 0 0 0 0 6e+05 \
  1.2e-06 56 14 0 1 "" red pink "" -0.76478 -0.35259 0
simundump text /kinetics/PLC_g/Ca.PLC_g/PIP2_hydrolysis/notes 0 \
  "Mainly Homma et al JBC 263:14 1988 pp 6592, but these\nparms are the Ca-stimulated form. It is not clear whether\nthe enzyme is activated by tyrosine phosphorylation at this\npoint or not. Wahl et al JBC 267:15 10447-10456 1992 say\nthat the Ca_stim and phosph form has 7X higher affinity \nfor substrate than control. This is close to Wahl's\nfigure 7, which I am using as reference."
call /kinetics/PLC_g/Ca.PLC_g/PIP2_hydrolysis/notes LOAD \
"Mainly Homma et al JBC 263:14 1988 pp 6592, but these" \
"parms are the Ca-stimulated form. It is not clear whether" \
"the enzyme is activated by tyrosine phosphorylation at this" \
"point or not. Wahl et al JBC 267:15 10447-10456 1992 say" \
"that the Ca_stim and phosph form has 7X higher affinity " \
"for substrate than control. This is close to Wahl's" \
"figure 7, which I am using as reference."
simundump kpool /kinetics/PLC_g/Ca.PLC_g* 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry pink darkgreen 6 28 0
simundump text /kinetics/PLC_g/Ca.PLC_g*/notes 0 ""
call /kinetics/PLC_g/Ca.PLC_g*/notes LOAD \
""
simundump kenz /kinetics/PLC_g/Ca.PLC_g*/PIP2_hydrolysis 1 0 0 0 0 6e+05 \
  2.4e-05 228 57 0 1 "" red pink "" 2.3983 -1.0044 0
simundump text /kinetics/PLC_g/Ca.PLC_g*/PIP2_hydrolysis/notes 0 \
  "Mainly Homma et al JBC 263:14 1988 pp 6592, but these\nparms are the Ca-stimulated form. It is not clear whether\nthe enzyme is activated by tyrosine phosphorylation at this\npoint or not. Wahl et al JBC 267:15 10447-10456 1992 say\nthat this has 7X higher affinity for substrate than control."
call /kinetics/PLC_g/Ca.PLC_g*/PIP2_hydrolysis/notes LOAD \
"Mainly Homma et al JBC 263:14 1988 pp 6592, but these" \
"parms are the Ca-stimulated form. It is not clear whether" \
"the enzyme is activated by tyrosine phosphorylation at this" \
"point or not. Wahl et al JBC 267:15 10447-10456 1992 say" \
"that this has 7X higher affinity for substrate than control."
simundump group /kinetics/CaMKII 1 purple black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 -22.401 3.9743 0
simundump text /kinetics/CaMKII/notes 0 \
  "Main reference here is the review by Hanson and Schulman, Ann Rev Biochem\n1992 vol 61 pp 559-601. Most of the mechanistic details and a few constants\nare derived from there. Many kinetics are from Hanson and Schulman JBC\n267:24 17216-17224 1992.\nThe enzs look a terrible mess. Actually it is just 3 reactions for diff sites,\nby 4 states of CaMKII, defined by the phosph state."
call /kinetics/CaMKII/notes LOAD \
"Main reference here is the review by Hanson and Schulman, Ann Rev Biochem" \
"1992 vol 61 pp 559-601. Most of the mechanistic details and a few constants" \
"are derived from there. Many kinetics are from Hanson and Schulman JBC" \
"267:24 17216-17224 1992." \
"The enzs look a terrible mess. Actually it is just 3 reactions for diff sites," \
"by 4 states of CaMKII, defined by the phosph state."
simundump kpool /kinetics/CaMKII/CaMKII 1 0 2 2 1.2e+06 1.2e+06 0 0 6e+05 0 \
  /kinetics/geometry palegreen purple -23.819 3.271 0
simundump text /kinetics/CaMKII/CaMKII/notes 0 \
  "Huge conc of CaMKII. In PSD it is 20-40% of protein, so we assume it is around\n2.5% of protein in spine as a whole. This level is so high it is unlikely to matter\nmuch if we are off a bit. This comes to about 70 uM."
call /kinetics/CaMKII/CaMKII/notes LOAD \
"Huge conc of CaMKII. In PSD it is 20-40% of protein, so we assume it is around" \
"2.5% of protein in spine as a whole. This level is so high it is unlikely to matter" \
"much if we are off a bit. This comes to about 70 uM."
simundump kpool /kinetics/CaMKII/CaMKII-CaM 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry palegreen purple -27.443 3.0376 0
simundump text /kinetics/CaMKII/CaMKII-CaM/notes 0 ""
call /kinetics/CaMKII/CaMKII-CaM/notes LOAD \
""
simundump kpool /kinetics/CaMKII/CaMKII-thr286*-CaM 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry palegreen purple -27.703 1.6156 0
simundump text /kinetics/CaMKII/CaMKII-thr286*-CaM/notes 0 \
  "From Hanson and Schulman, the thr286 is responsible for autonomous activation\nof CaMKII."
call /kinetics/CaMKII/CaMKII-thr286*-CaM/notes LOAD \
"From Hanson and Schulman, the thr286 is responsible for autonomous activation" \
"of CaMKII."
simundump kpool /kinetics/CaMKII/CaMKII*** 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry cyan purple -27.616 -1.6238 0
simundump text /kinetics/CaMKII/CaMKII***/notes 0 \
  "From Hanson and Schulman, the CaMKII does a lot of autophosphorylation\njust after the CaM is released. This prevents further CaM binding and renders\nthe enzyme quite independent of Ca."
call /kinetics/CaMKII/CaMKII***/notes LOAD \
"From Hanson and Schulman, the CaMKII does a lot of autophosphorylation" \
"just after the CaM is released. This prevents further CaM binding and renders" \
"the enzyme quite independent of Ca."
simundump kreac /kinetics/CaMKII/CaMKII-bind-CaM 1 8.3333e-05 5 "" white \
  purple -23.298 1.5267 0
simundump text /kinetics/CaMKII/CaMKII-bind-CaM/notes 0 \
  "This is tricky. There is some cooperativity here arising from interactions\nbetween the subunits of the CAMKII holoenzyme. However, the\nstoichiometry is 1. \nKb/Kf = 6e4 #/cell. Rate is fast (see Hanson et al Neuron 12 943-956 1994)\nso lets say kb = 10. This gives kf = 1.6667e-4\nH&S AnnRev Biochem 92 give tau for dissoc as 0.2 sec at low Ca, 0.4 at high.\nLow Ca = 100 nM = physiol."
call /kinetics/CaMKII/CaMKII-bind-CaM/notes LOAD \
"This is tricky. There is some cooperativity here arising from interactions" \
"between the subunits of the CAMKII holoenzyme. However, the" \
"stoichiometry is 1. " \
"Kb/Kf = 6e4 #/cell. Rate is fast (see Hanson et al Neuron 12 943-956 1994)" \
"so lets say kb = 10. This gives kf = 1.6667e-4" \
"H&S AnnRev Biochem 92 give tau for dissoc as 0.2 sec at low Ca, 0.4 at high." \
"Low Ca = 100 nM = physiol."
simundump kreac /kinetics/CaMKII/CaMK-thr286-bind-CaM 1 0.001667 0.1 "" white \
  purple -23.277 0.92147 0
simundump text /kinetics/CaMKII/CaMK-thr286-bind-CaM/notes 0 \
  "Affinity is up 1000X. Time to release is about 20 sec, so the kb is OK at 0.1\nThis makes Kf around 1.6666e-3\n"
call /kinetics/CaMKII/CaMK-thr286-bind-CaM/notes LOAD \
"Affinity is up 1000X. Time to release is about 20 sec, so the kb is OK at 0.1" \
"This makes Kf around 1.6666e-3" \
""
simundump kpool /kinetics/CaMKII/CaMKII-thr286 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry red purple -27.551 -0.09309 0
simundump text /kinetics/CaMKII/CaMKII-thr286/notes 0 \
  "I am not sure if we need to endow this one with a lot of enzs. It is likely\nto be a short-lived intermediate, since it will be phosphorylated further\nas soon as the CAM falls off."
call /kinetics/CaMKII/CaMKII-thr286/notes LOAD \
"I am not sure if we need to endow this one with a lot of enzs. It is likely" \
"to be a short-lived intermediate, since it will be phosphorylated further" \
"as soon as the CAM falls off."
simundump kpool /kinetics/CaMKII/CaMK-thr306 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry palegreen purple -27.539 -3.2652 0
simundump text /kinetics/CaMKII/CaMK-thr306/notes 0 \
  "This forms due to basal autophosphorylation, but I think it has to be\nconsidered as a pathway even if some CaM is floating around. In either\ncase it will tend to block further binding of CaM, and will not display any\nenzyme activity. See Hanson and Schulman JBC 267:24 pp17216-17224 1992"
call /kinetics/CaMKII/CaMK-thr306/notes LOAD \
"This forms due to basal autophosphorylation, but I think it has to be" \
"considered as a pathway even if some CaM is floating around. In either" \
"case it will tend to block further binding of CaM, and will not display any" \
"enzyme activity. See Hanson and Schulman JBC 267:24 pp17216-17224 1992"
simundump kreac /kinetics/CaMKII/basal-activity 1 0.003 0 "" white purple \
  -25.369 -0.16228 0
simundump text /kinetics/CaMKII/basal-activity/notes 0 \
  "This reaction represents one of the big unknowns in CaMK-II\nbiochemistry: what maintains the basal level of phosphorylation\non thr 286 ? See Hanson and Schulman Ann Rev Biochem 1992\n61:559-601, specially pg 580, for review. I have not been able to\nfind any compelling mechanism in the literature, but fortunately\nthe level of basal activity is well documented. "
call /kinetics/CaMKII/basal-activity/notes LOAD \
"This reaction represents one of the big unknowns in CaMK-II" \
"biochemistry: what maintains the basal level of phosphorylation" \
"on thr 286 ? See Hanson and Schulman Ann Rev Biochem 1992" \
"61:559-601, specially pg 580, for review. I have not been able to" \
"find any compelling mechanism in the literature, but fortunately" \
"the level of basal activity is well documented. "
simundump kpool /kinetics/CaMKII/tot_CaM_CaMKII 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry green purple -31.715 3.2973 0
simundump text /kinetics/CaMKII/tot_CaM_CaMKII/notes 0 ""
call /kinetics/CaMKII/tot_CaM_CaMKII/notes LOAD \
""
simundump kenz /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_305 1 0 0 0 0 6e+05 \
  4.4014e-07 24 6 0 0 "" red green "" -29.551 0.6145 0
simundump text /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_305/notes 0 \
  "Rates from autocamtide phosphorylation, from \nHanson and Schulman JBC 267:24 17216-17224 1992.\nJan 1 1998: Speed up 12x to match fig 5."
call /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_305/notes LOAD \
"Rates from autocamtide phosphorylation, from " \
"Hanson and Schulman JBC 267:24 17216-17224 1992." \
"Jan 1 1998: Speed up 12x to match fig 5."
simundump kenz /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_286 1 0 0 0 0 6e+05 \
  3.6678e-08 2 0.5 0 0 "" red green "" -25.596 2.816 0
simundump text /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_286/notes 0 ""
call /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_286/notes LOAD \
""
simundump kpool /kinetics/CaMKII/tot_autonomous_CaMKII 1 0 0 0 0 0 0 0 6e+05 \
  0 /kinetics/geometry green purple -32.064 2.3272 0
simundump text /kinetics/CaMKII/tot_autonomous_CaMKII/notes 0 ""
call /kinetics/CaMKII/tot_autonomous_CaMKII/notes LOAD \
""
simundump kenz /kinetics/CaMKII/tot_autonomous_CaMKII/auton_305 1 0 0 0 0 \
  6e+05 2.8571e-07 24 6 0 0 "" red green "" -29.736 -0.41162 0
simundump text /kinetics/CaMKII/tot_autonomous_CaMKII/auton_305/notes 0 \
  "See Hanson and Schulman again, for afterburst rates of\nphosph."
call /kinetics/CaMKII/tot_autonomous_CaMKII/auton_305/notes LOAD \
"See Hanson and Schulman again, for afterburst rates of" \
"phosph."
simundump kenz /kinetics/CaMKII/tot_autonomous_CaMKII/auton_286 1 0 0 0 0 \
  6e+05 2.381e-08 2 0.5 0 0 "" red green "" -25.473 1.9951 0
simundump text /kinetics/CaMKII/tot_autonomous_CaMKII/auton_286/notes 0 ""
call /kinetics/CaMKII/tot_autonomous_CaMKII/auton_286/notes LOAD \
""
simundump kpool /kinetics/PP1-active 1 0 1.8 1.8 1.08e+06 1.08e+06 0 0 6e+05 \
  0 /kinetics/geometry cyan yellow -31.448 0.13975 0
simundump text /kinetics/PP1-active/notes 0 \
  "Cohen et al Meth Enz 159 390-408 is main source of info\nconc  = 1.8 uM"
call /kinetics/PP1-active/notes LOAD \
"Cohen et al Meth Enz 159 390-408 is main source of info" \
"conc  = 1.8 uM"
simundump kenz /kinetics/PP1-active/Deph-thr286 1 0 0 0 0 6e+05 5.72e-07 1.4 \
  0.35 0 0 "" red cyan "" -31.097 1.7813 0
simundump text /kinetics/PP1-active/Deph-thr286/notes 0 \
  "The rates are from Stralfors et al Eur J Biochem 149 295-303 giving\nVmax = 5.7 umol/min giving k3 = 3.5/sec and k2 = 14.\nFoulkes et al Eur J Biochem 132 309-313 1983 give Km = 5.1 uM so\nk1 becomes 5.72e-6\nSimonelli 1984 (Grad Thesis, CUNY) showed that other substrates\nare about 1/10 rate of phosphorylase a, so we reduce k1,k2,k3 by 10\nto 5.72e-7, 1.4, 0.35"
call /kinetics/PP1-active/Deph-thr286/notes LOAD \
"The rates are from Stralfors et al Eur J Biochem 149 295-303 giving" \
"Vmax = 5.7 umol/min giving k3 = 3.5/sec and k2 = 14." \
"Foulkes et al Eur J Biochem 132 309-313 1983 give Km = 5.1 uM so" \
"k1 becomes 5.72e-6" \
"Simonelli 1984 (Grad Thesis, CUNY) showed that other substrates" \
"are about 1/10 rate of phosphorylase a, so we reduce k1,k2,k3 by 10" \
"to 5.72e-7, 1.4, 0.35"
simundump kenz /kinetics/PP1-active/Deph-thr305 1 0 0 0 0 6e+05 5.72e-07 1.4 \
  0.35 0 0 "" red cyan "" -30.313 -1.1052 0
simundump text /kinetics/PP1-active/Deph-thr305/notes 0 ""
call /kinetics/PP1-active/Deph-thr305/notes LOAD \
""
simundump kenz /kinetics/PP1-active/Deph-thr306 1 0 0 0 0 6e+05 5.72e-07 1.4 \
  0.35 0 0 "" red cyan "" -25.538 3.7223 0
simundump text /kinetics/PP1-active/Deph-thr306/notes 0 "See Cohen et al"
call /kinetics/PP1-active/Deph-thr306/notes LOAD \
"See Cohen et al"
simundump kenz /kinetics/PP1-active/Deph-thr286c 1 0 0 0 0 6e+05 5.72e-07 1.4 \
  0.35 0 0 "" red cyan "" -30.334 -2.8165 0
simundump text /kinetics/PP1-active/Deph-thr286c/notes 0 ""
call /kinetics/PP1-active/Deph-thr286c/notes LOAD \
""
simundump kenz /kinetics/PP1-active/Deph_thr286b 1 0 0 0 0 6e+05 5.72e-07 1.4 \
  0.35 0 0 "" red cyan "" -24.758 -1.1185 0
simundump text /kinetics/PP1-active/Deph_thr286b/notes 0 ""
call /kinetics/PP1-active/Deph_thr286b/notes LOAD \
""
simundump group /kinetics/CaM 1 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 -45.327 -3.6101 0
simundump text /kinetics/CaM/notes 0 ""
call /kinetics/CaM/notes LOAD \
""
simundump kpool /kinetics/CaM/CaM 1 0 5 5 3e+06 3e+06 0 0 6e+05 0 \
  /kinetics/geometry pink blue -45.344 4.1096 0
simundump text /kinetics/CaM/CaM/notes 0 \
  "There is a LOT of this in the cell: upto 1% of total protein mass. (Alberts et al)\nSay 25 uM. Meyer et al Science 256 1199-1202 1992 refer to studies saying\nit is comparable to CaMK levels. \n"
call /kinetics/CaM/CaM/notes LOAD \
"There is a LOT of this in the cell: upto 1% of total protein mass. (Alberts et al)" \
"Say 25 uM. Meyer et al Science 256 1199-1202 1992 refer to studies saying" \
"it is comparable to CaMK levels. " \
""
simundump kreac /kinetics/CaM/CaM-TR2-bind-Ca 1 2e-10 72 "" white blue \
  -43.165 3.4688 0
simundump text /kinetics/CaM/CaM-TR2-bind-Ca/notes 0 \
  "Lets use the fast rate consts here. Since the rates are so different, I am not\nsure whether the order is relevant. These correspond to the TR2C fragment.\nWe use the Martin et al rates here, plus the Drabicowski binding consts.\nAll are scaled by 3X to cell temp.\nkf = 2e-10 kb = 72\nStemmer & Klee: K1=.9, K2=1.1. Assume 1.0uM for both. kb/kf=3.6e11.\nIf kb=72, kf = 2e-10 (Exactly the same !)...."
call /kinetics/CaM/CaM-TR2-bind-Ca/notes LOAD \
"Lets use the fast rate consts here. Since the rates are so different, I am not" \
"sure whether the order is relevant. These correspond to the TR2C fragment." \
"We use the Martin et al rates here, plus the Drabicowski binding consts." \
"All are scaled by 3X to cell temp." \
"kf = 2e-10 kb = 72" \
"Stemmer & Klee: K1=.9, K2=1.1. Assume 1.0uM for both. kb/kf=3.6e11." \
"If kb=72, kf = 2e-10 (Exactly the same !)...."
simundump kreac /kinetics/CaM/CaM-TR2-Ca2-bind-Ca 1 6e-06 10 "" white blue \
  -44.169 1.6152 0
simundump text /kinetics/CaM/CaM-TR2-Ca2-bind-Ca/notes 0 \
  "K3 = 21.5, K4 = 2.8. Assuming that the K4 step happens first, we get\nkb/kf = 2.8 uM = 1.68e6 so kf =6e-6 assuming kb = 10\n"
call /kinetics/CaM/CaM-TR2-Ca2-bind-Ca/notes LOAD \
"K3 = 21.5, K4 = 2.8. Assuming that the K4 step happens first, we get" \
"kb/kf = 2.8 uM = 1.68e6 so kf =6e-6 assuming kb = 10" \
""
simundump kreac /kinetics/CaM/CaM-Ca3-bind-Ca 1 7.75e-07 10 "" white blue \
  -45.727 -1.3505 0
simundump text /kinetics/CaM/CaM-Ca3-bind-Ca/notes 0 \
  "Use K3 = 21.5 uM here from Stemmer and Klee table 3.\nkb/kf =21.5 * 6e5 so kf = 7.75e-7, kb = 10"
call /kinetics/CaM/CaM-Ca3-bind-Ca/notes LOAD \
"Use K3 = 21.5 uM here from Stemmer and Klee table 3." \
"kb/kf =21.5 * 6e5 so kf = 7.75e-7, kb = 10"
simundump kpool /kinetics/CaM/CaM-TR2-Ca2 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry pink blue -40.328 2.6476 0
simundump text /kinetics/CaM/CaM-TR2-Ca2/notes 0 \
  "This is the intermediate where the TR2 end (the high-affinity end) has\nbound the Ca but the TR1 end has not."
call /kinetics/CaM/CaM-TR2-Ca2/notes LOAD \
"This is the intermediate where the TR2 end (the high-affinity end) has" \
"bound the Ca but the TR1 end has not."
simundump kpool /kinetics/CaM/CaM-Ca3 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry hotpink blue -41.838 -0.21314 0
simundump text /kinetics/CaM/CaM-Ca3/notes 0 ""
call /kinetics/CaM/CaM-Ca3/notes LOAD \
""
simundump group /kinetics/PP1 1 yellow black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 -46.362 -10.235 0
simundump text /kinetics/PP1/notes 0 ""
call /kinetics/PP1/notes LOAD \
""
simundump kpool /kinetics/PP1/I1 1 0 1.8 1.8 1.08e+06 1.08e+06 0 0 6e+05 0 \
  /kinetics/geometry orange yellow -38.013 -14.351 0
simundump text /kinetics/PP1/I1/notes 0 \
  "I1 is a 'mixed' inhibitor, but at high enz concs it looks like a non-compet\ninhibitor (Foulkes et al Eur J Biochem 132 309-313 9183).\nWe treat it as non-compet, so it just turns the enz off\nwithout interacting with the binding site.\nCohen et al ann rev bioch refer to results where conc is \n1.5 to 1.8 uM. In order to get complete inhib of PP1, which is at 1.8 uM,\nwe need >= 1.8 uM.\n\n"
call /kinetics/PP1/I1/notes LOAD \
"I1 is a 'mixed' inhibitor, but at high enz concs it looks like a non-compet" \
"inhibitor (Foulkes et al Eur J Biochem 132 309-313 9183)." \
"We treat it as non-compet, so it just turns the enz off" \
"without interacting with the binding site." \
"Cohen et al ann rev bioch refer to results where conc is " \
"1.5 to 1.8 uM. In order to get complete inhib of PP1, which is at 1.8 uM," \
"we need >= 1.8 uM." \
"" \
""
simundump kpool /kinetics/PP1/I1* 1 0 0.001 0.001 600 600 0 0 6e+05 0 \
  /kinetics/geometry orange yellow -42.158 -14.319 0
simundump text /kinetics/PP1/I1*/notes 0 "Dephosph is mainly by PP2B"
call /kinetics/PP1/I1*/notes LOAD \
"Dephosph is mainly by PP2B"
simundump kreac /kinetics/PP1/Inact-PP1 1 8.3333e-05 0.1 "" white yellow \
  -45.403 -12.417 0
simundump text /kinetics/PP1/Inact-PP1/notes 0 \
  "K inhib = 1nM from Cohen Ann Rev Bioch 1989, \n4 nM from Foukes et al \nAssume 2 nM. kf /kb = 8.333e-4"
call /kinetics/PP1/Inact-PP1/notes LOAD \
"K inhib = 1nM from Cohen Ann Rev Bioch 1989, " \
"4 nM from Foukes et al " \
"Assume 2 nM. kf /kb = 8.333e-4"
simundump kpool /kinetics/PP1/PP1-I1* 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry brown yellow -43.747 -8.9641 0
simundump text /kinetics/PP1/PP1-I1*/notes 0 ""
call /kinetics/PP1/PP1-I1*/notes LOAD \
""
simundump kpool /kinetics/PP1/PP1-I1 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry brown yellow -36.339 -8.6879 0
simundump text /kinetics/PP1/PP1-I1/notes 0 ""
call /kinetics/PP1/PP1-I1/notes LOAD \
""
simundump kreac /kinetics/PP1/dissoc-PP1-I1 1 1 0 "" white yellow -42.498 \
  -12.141 0
simundump text /kinetics/PP1/dissoc-PP1-I1/notes 0 \
  "Let us assume that the equil in this case is very far over to the\nright. This is probably safe.\n"
call /kinetics/PP1/dissoc-PP1-I1/notes LOAD \
"Let us assume that the equil in this case is very far over to the" \
"right. This is probably safe." \
""
simundump kpool /kinetics/PP2A 1 0 0.12 0.12 72000 72000 0 0 6e+05 0 \
  /kinetics/geometry red black -36.52 -3.6325 0
simundump text /kinetics/PP2A/notes 0 ""
call /kinetics/PP2A/notes LOAD \
""
simundump kenz /kinetics/PP2A/PP2A-dephosph-I1 1 0 0 0 0 6e+05 6.6e-06 25 6 0 \
  0 "" red red "" -38.954 -10.663 0
simundump text /kinetics/PP2A/PP2A-dephosph-I1/notes 0 \
  "PP2A does most of the dephosph of I1 at basal Ca levels. See\nthe review by Cohen in Ann Rev Biochem 1989.\nFor now, lets halve Km. k1 was 3.3e-6, now 6.6e-6\n"
call /kinetics/PP2A/PP2A-dephosph-I1/notes LOAD \
"PP2A does most of the dephosph of I1 at basal Ca levels. See" \
"the review by Cohen in Ann Rev Biochem 1989." \
"For now, lets halve Km. k1 was 3.3e-6, now 6.6e-6" \
""
simundump kenz /kinetics/PP2A/PP2A-dephosph-PP1-I* 1 0 0 0 0 6e+05 6.6e-06 25 \
  6 0 0 "" red red "" -36.521 -4.7733 0
simundump text /kinetics/PP2A/PP2A-dephosph-PP1-I*/notes 0 \
  "k1 changed from 3.3e-6 to 6.6e-6\n"
call /kinetics/PP2A/PP2A-dephosph-PP1-I*/notes LOAD \
"k1 changed from 3.3e-6 to 6.6e-6" \
""
simundump kpool /kinetics/CaNAB-Ca4 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry tan yellow -24.923 -8.5717 0
simundump text /kinetics/CaNAB-Ca4/notes 0 ""
call /kinetics/CaNAB-Ca4/notes LOAD \
""
simundump kenz /kinetics/CaNAB-Ca4/dephosph_inhib1_noCaM 1 0 0 0 0 6e+05 \
  5.7e-08 0.136 0.034 0 0 "" red tan "" -35.539 -10.496 0
simundump text /kinetics/CaNAB-Ca4/dephosph_inhib1_noCaM/notes 0 \
  "The rates here are so slow I do not know if we should even bother\nwith this enz reacn. These numbers are from Liu and Storm.\nOther refs suggest that the Km stays the same\nbut the Vmax goes to 10% of the CaM stim levels. \nPrev: k1=2.2e-9, k2 = 0.0052, k3 = 0.0013\nNew : k1=5.7e-8, k2=.136, k3=.034"
call /kinetics/CaNAB-Ca4/dephosph_inhib1_noCaM/notes LOAD \
"The rates here are so slow I do not know if we should even bother" \
"with this enz reacn. These numbers are from Liu and Storm." \
"Other refs suggest that the Km stays the same" \
"but the Vmax goes to 10% of the CaM stim levels. " \
"Prev: k1=2.2e-9, k2 = 0.0052, k3 = 0.0013" \
"New : k1=5.7e-8, k2=.136, k3=.034"
simundump group /kinetics/PP2B 1 red3 black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 -20.052 -5.8546 0
simundump text /kinetics/PP2B/notes 0 \
  "Also called Calcineurin.\nMajor sources of info:\nCohen, P Ann Rev Biochem 1989 58:453-508\nMumby and Walker Physiol Rev 73:4 673-699\nStemmer and Klee Biochem 33 1994 6859-6866\nLiu and Storm JBC 264:22 1989 12800-12804\nThis model is unusual: There is actually more expt info than I want to\nput in the model at this time.\nPhosph: Hashimoto and Soderling JBC 1989 264:28 16624-16629 (Not used)"
call /kinetics/PP2B/notes LOAD \
"Also called Calcineurin." \
"Major sources of info:" \
"Cohen, P Ann Rev Biochem 1989 58:453-508" \
"Mumby and Walker Physiol Rev 73:4 673-699" \
"Stemmer and Klee Biochem 33 1994 6859-6866" \
"Liu and Storm JBC 264:22 1989 12800-12804" \
"This model is unusual: There is actually more expt info than I want to" \
"put in the model at this time." \
"Phosph: Hashimoto and Soderling JBC 1989 264:28 16624-16629 (Not used)"
simundump kpool /kinetics/PP2B/CaNAB 1 0 0.5 1 6e+05 3e+05 0 0 6e+05 0 \
  /kinetics/geometry tan red3 -18.702 -8.4456 0
simundump text /kinetics/PP2B/CaNAB/notes 0 \
  "We assume that the A and B subunits of PP2B are always bound under\nphysiol conditions.\nUp to 1% of brain protein = 25 uM. I need to work out how it is distributed\nbetween cytosolic and particulate fractions.\nTallant and Cheung '83 Biochem 22 3630-3635 have conc in many \nspecies, average for mammalian brain is around 1 uM.\n10 Feb 2009.\nHalved conc to 0.5 uM."
call /kinetics/PP2B/CaNAB/notes LOAD \
"We assume that the A and B subunits of PP2B are always bound under" \
"physiol conditions." \
"Up to 1% of brain protein = 25 uM. I need to work out how it is distributed" \
"between cytosolic and particulate fractions." \
"Tallant and Cheung '83 Biochem 22 3630-3635 have conc in many " \
"species, average for mammalian brain is around 1 uM." \
"10 Feb 2009." \
"Halved conc to 0.5 uM."
simundump kpool /kinetics/PP2B/CaNAB-Ca2 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry tan red3 -21.258 -8.5373 0
simundump text /kinetics/PP2B/CaNAB-Ca2/notes 0 ""
call /kinetics/PP2B/CaNAB-Ca2/notes LOAD \
""
simundump kreac /kinetics/PP2B/Ca-bind-CaNAB-Ca2 1 1e-11 1 "" white red3 \
  -22.826 -9.7525 0
simundump text /kinetics/PP2B/Ca-bind-CaNAB-Ca2/notes 0 \
  "This process is probably much more complicated and involves CaM.\nHowever, as I can't find detailed info I am bundling this into a\nsingle step.\nBased on Steemer and Klee pg 6863, the Kact is 0.5 uM.\nkf/kb = 1/(0.5 * 6e5)^2 = 1.11e-11"
call /kinetics/PP2B/Ca-bind-CaNAB-Ca2/notes LOAD \
"This process is probably much more complicated and involves CaM." \
"However, as I can't find detailed info I am bundling this into a" \
"single step." \
"Based on Steemer and Klee pg 6863, the Kact is 0.5 uM." \
"kf/kb = 1/(0.5 * 6e5)^2 = 1.11e-11"
simundump kreac /kinetics/PP2B/Ca-bind-CaNAB 1 2.7778e-09 0.1 "" white red3 \
  -20.125 -9.8899 0
simundump text /kinetics/PP2B/Ca-bind-CaNAB/notes 0 \
  "going on the experience with CaM, we put the fast (high affinity)\nsites first. We only know (Stemmer and Klee) that the affinity is < 70 nM.\nAssuming 10 nM at first, we get\nkf = 2.78e-8, kb = 1.\nTry 20 nM.\nkf = 7e-9, kb = 1\n\n"
call /kinetics/PP2B/Ca-bind-CaNAB/notes LOAD \
"going on the experience with CaM, we put the fast (high affinity)" \
"sites first. We only know (Stemmer and Klee) that the affinity is < 70 nM." \
"Assuming 10 nM at first, we get" \
"kf = 2.78e-8, kb = 1." \
"Try 20 nM." \
"kf = 7e-9, kb = 1" \
"" \
""
simundump kreac /kinetics/PP2B/CaMCa4-bind-CaNAB 1 0.001 1 "" white red3 \
  -27.639 -7.6415 0
simundump text /kinetics/PP2B/CaMCa4-bind-CaNAB/notes 0 ""
call /kinetics/PP2B/CaMCa4-bind-CaNAB/notes LOAD \
""
simundump group /kinetics/PKA 0 blue blue x 0 0 "" defaultfile defaultfile.g \
  0 0 0 -41.943 -20.667 0
simundump text /kinetics/PKA/notes 0 \
  "General ref: Taylor et al Ann Rev Biochem 1990 59:971-1005\n"
call /kinetics/PKA/notes LOAD \
"General ref: Taylor et al Ann Rev Biochem 1990 59:971-1005" \
""
simundump kpool /kinetics/PKA/R2C2 0 0 0.5 0.5 3e+05 3e+05 0 0 6e+05 0 \
  /kinetics/geometry white blue -46.656 -27.74 0
simundump text /kinetics/PKA/R2C2/notes 0 \
  "This is the R2C2 complex, consisting of 2 catalytic (C)\nsubunits, and the R-dimer. See Taylor et al Ann Rev Biochem\n1990 59:971-1005 for a review.\nThe Doskeland and Ogreid review is better for numbers.\nAmount of PKA is about .5 uM."
call /kinetics/PKA/R2C2/notes LOAD \
"This is the R2C2 complex, consisting of 2 catalytic (C)" \
"subunits, and the R-dimer. See Taylor et al Ann Rev Biochem" \
"1990 59:971-1005 for a review." \
"The Doskeland and Ogreid review is better for numbers." \
"Amount of PKA is about .5 uM."
simundump kpool /kinetics/PKA/R2C2-cAMP 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry white blue -43.694 -27.272 0
simundump text /kinetics/PKA/R2C2-cAMP/notes 0 "CoInit was .0624\n"
call /kinetics/PKA/R2C2-cAMP/notes LOAD \
"CoInit was .0624" \
""
simundump kreac /kinetics/PKA/cAMP-bind-site-B1 0 9e-05 33 "" white blue \
  -44.279 -31.015 0
simundump text /kinetics/PKA/cAMP-bind-site-B1/notes 0 \
  "Hasler et al FASEB J 6:2734-2741 1992 say Kd =1e-7M\nfor type II, 5.6e-8 M for type I. Take mean\nwhich comes to 2e-13 #/cell\nSmith et al PNAS USA 78:3 1591-1595 1981 have better data.\nFirst kf/kb=2.1e7/M = 3.5e-5 (#/cell).\nOgreid and Doskeland Febs Lett 129:2 287-292 1981 have figs\nsuggesting time course of complete assoc is < 1 min."
call /kinetics/PKA/cAMP-bind-site-B1/notes LOAD \
"Hasler et al FASEB J 6:2734-2741 1992 say Kd =1e-7M" \
"for type II, 5.6e-8 M for type I. Take mean" \
"which comes to 2e-13 #/cell" \
"Smith et al PNAS USA 78:3 1591-1595 1981 have better data." \
"First kf/kb=2.1e7/M = 3.5e-5 (#/cell)." \
"Ogreid and Doskeland Febs Lett 129:2 287-292 1981 have figs" \
"suggesting time course of complete assoc is < 1 min."
simundump kreac /kinetics/PKA/cAMP-bind-site-B2 1 9e-05 33 "" white blue \
  -41.937 -29.767 0
simundump text /kinetics/PKA/cAMP-bind-site-B2/notes 0 \
  "For now let us set this to the same Km (1e-7M) as\nsite B. This gives kf/kb = .7e-7M * 1e6 / (6e5^2) : 1/(6e5^2)\n= 2e-13:2.77e-12\nSmith et al have better values. They say that this is\ncooperative, so the consts are now kf/kb =8.3e-4"
call /kinetics/PKA/cAMP-bind-site-B2/notes LOAD \
"For now let us set this to the same Km (1e-7M) as" \
"site B. This gives kf/kb = .7e-7M * 1e6 / (6e5^2) : 1/(6e5^2)" \
"= 2e-13:2.77e-12" \
"Smith et al have better values. They say that this is" \
"cooperative, so the consts are now kf/kb =8.3e-4"
simundump kreac /kinetics/PKA/cAMP-bind-site-A1 1 0.000125 110 "" white blue \
  -39.251 -30.952 0
simundump text /kinetics/PKA/cAMP-bind-site-A1/notes 0 ""
call /kinetics/PKA/cAMP-bind-site-A1/notes LOAD \
""
simundump kreac /kinetics/PKA/cAMP-bind-site-A2 1 0.000125 32.5 "" white blue \
  -35.964 -29.521 0
simundump text /kinetics/PKA/cAMP-bind-site-A2/notes 0 ""
call /kinetics/PKA/cAMP-bind-site-A2/notes LOAD \
""
simundump kpool /kinetics/PKA/R2C2-cAMP2 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry white blue -40.146 -26.43 0
simundump text /kinetics/PKA/R2C2-cAMP2/notes 0 ""
call /kinetics/PKA/R2C2-cAMP2/notes LOAD \
""
simundump kpool /kinetics/PKA/R2C2-cAMP3 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry white blue -37.46 -27.49 0
simundump text /kinetics/PKA/R2C2-cAMP3/notes 0 ""
call /kinetics/PKA/R2C2-cAMP3/notes LOAD \
""
simundump kpool /kinetics/PKA/R2C2-cAMP4 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry white blue -35.074 -25.879 0
simundump text /kinetics/PKA/R2C2-cAMP4/notes 0 ""
call /kinetics/PKA/R2C2-cAMP4/notes LOAD \
""
simundump kpool /kinetics/PKA/R2C-cAMP4 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry white blue -37.357 -24.745 0
simundump text /kinetics/PKA/R2C-cAMP4/notes 0 ""
call /kinetics/PKA/R2C-cAMP4/notes LOAD \
""
simundump kpool /kinetics/PKA/R2-cAMP4 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry white blue -43.694 -25.182 0
simundump text /kinetics/PKA/R2-cAMP4/notes 0 \
  "Starts at 0.15 for the test of fig 6 in Smith et al, but we aren't using\nthat paper any more."
call /kinetics/PKA/R2-cAMP4/notes LOAD \
"Starts at 0.15 for the test of fig 6 in Smith et al, but we aren't using" \
"that paper any more."
simundump kreac /kinetics/PKA/Release-C1 1 60 3e-05 "" white blue -35.361 \
  -22.877 0
simundump text /kinetics/PKA/Release-C1/notes 0 \
  "This has to be fast, as the activation of PKA by cAMP\nis also fast.\nkf was 10\n"
call /kinetics/PKA/Release-C1/notes LOAD \
"This has to be fast, as the activation of PKA by cAMP" \
"is also fast." \
"kf was 10" \
""
simundump kreac /kinetics/PKA/Release-C2 1 60 3e-05 "" white blue -40.232 \
  -24.155 0
simundump text /kinetics/PKA/Release-C2/notes 0 ""
call /kinetics/PKA/Release-C2/notes LOAD \
""
simundump kpool /kinetics/PKA/PKA-inhibitor 1 0 0.25 0.25 1.5e+05 1.5e+05 0 0 \
  6e+05 0 /kinetics/geometry cyan blue -44.714 -23.288 0
simundump text /kinetics/PKA/PKA-inhibitor/notes 0 \
  "About 25% of PKA C subunit is dissociated in resting cells without\nhaving any noticable activity.\nDoskeland and Ogreid Int J biochem 13 pp1-19 suggest that this is\nbecause there is a corresponding amount of inhibitor protein."
call /kinetics/PKA/PKA-inhibitor/notes LOAD \
"About 25% of PKA C subunit is dissociated in resting cells without" \
"having any noticable activity." \
"Doskeland and Ogreid Int J biochem 13 pp1-19 suggest that this is" \
"because there is a corresponding amount of inhibitor protein."
simundump kreac /kinetics/PKA/inhib-PKA 1 0.0001 1 "" white blue -41.921 \
  -22.664 0
simundump text /kinetics/PKA/inhib-PKA/notes 0 \
  "This has to be set to zero for matching the expts in vitro. In vivo\nwe need to consider the inhibition though.\nkf = 1e-5\nkb = 1\n"
call /kinetics/PKA/inhib-PKA/notes LOAD \
"This has to be set to zero for matching the expts in vitro. In vivo" \
"we need to consider the inhibition though." \
"kf = 1e-5" \
"kb = 1" \
""
simundump kpool /kinetics/PKA/inhibited-PKA 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry cyan blue -45.341 -21.689 0
simundump text /kinetics/PKA/inhibited-PKA/notes 0 ""
call /kinetics/PKA/inhibited-PKA/notes LOAD \
""
simundump kpool /kinetics/cAMP 1 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry \
  green black -30.156 -32.591 0
simundump text /kinetics/cAMP/notes 0 \
  "The conc of this has been a problem. Schaecter and Benowitz use 50 uM,\nbut Shinomura et al have < 5. So I have altered the cAMP-dependent \nrates in the PKA model to reflect this."
call /kinetics/cAMP/notes LOAD \
"The conc of this has been a problem. Schaecter and Benowitz use 50 uM," \
"but Shinomura et al have < 5. So I have altered the cAMP-dependent " \
"rates in the PKA model to reflect this."
simundump group /kinetics/AC 1 blue blue x 0 0 "" defaultfile defaultfile.g 0 \
  0 0 -17.529 -17.47 0
simundump text /kinetics/AC/notes 0 ""
call /kinetics/AC/notes LOAD \
""
simundump kpool /kinetics/AC/ATP 1 0 5000 5000 3e+09 3e+09 0 0 6e+05 4 \
  /kinetics/geometry red blue -18.042 -18.868 0
simundump text /kinetics/AC/ATP/notes 0 \
  "ATP is present in all cells between 2 and 10 mM. See Lehninger."
call /kinetics/AC/ATP/notes LOAD \
"ATP is present in all cells between 2 and 10 mM. See Lehninger."
simundump kpool /kinetics/AC/AC1-CaM 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry orange blue -20.483 -17.259 0
simundump text /kinetics/AC/AC1-CaM/notes 0 \
  "This version of cyclase is Calmodulin activated.\nGs stims it but betagamma inhibits."
call /kinetics/AC/AC1-CaM/notes LOAD \
"This version of cyclase is Calmodulin activated." \
"Gs stims it but betagamma inhibits."
simundump kenz /kinetics/AC/AC1-CaM/kenz 1 0 0 0 0 6e+05 7.5e-06 72 18 0 1 "" \
  red orange "" -20.52 -18.394 0
simundump text /kinetics/AC/AC1-CaM/kenz/notes 0 ""
call /kinetics/AC/AC1-CaM/kenz/notes LOAD \
""
simundump kpool /kinetics/AC/AC1 1 0 0.02 0.02 12000 12000 0 0 6e+05 0 \
  /kinetics/geometry orange blue -24.247 -15.394 0
simundump text /kinetics/AC/AC1/notes 0 "Starting conc at 20 nM."
call /kinetics/AC/AC1/notes LOAD \
"Starting conc at 20 nM."
simundump kreac /kinetics/AC/CaM-bind-AC1 1 8.3333e-05 1 "" white blue \
  -22.762 -15.59 0
simundump text /kinetics/AC/CaM-bind-AC1/notes 0 \
  "Half-max at 20 nM CaM (Tang et al JBC 266:13 8595-8603 1991\nkb/kf = 20 nM = 12000 #/cell\nso kf = kb/12000 = kb * 8.333e-5\n"
call /kinetics/AC/CaM-bind-AC1/notes LOAD \
"Half-max at 20 nM CaM (Tang et al JBC 266:13 8595-8603 1991" \
"kb/kf = 20 nM = 12000 #/cell" \
"so kf = kb/12000 = kb * 8.333e-5" \
""
simundump kpool /kinetics/AC/AC2* 1 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry \
  yellow blue -18.647 -22.52 0
simundump text /kinetics/AC/AC2*/notes 0 \
  "This version is activated by Gs and by a betagamma and phosphorylation."
call /kinetics/AC/AC2*/notes LOAD \
"This version is activated by Gs and by a betagamma and phosphorylation."
simundump kenz /kinetics/AC/AC2*/kenz 1 0 0 0 0 6e+05 2.9e-06 28 7 0 1 "" red \
  yellow "" -18.774 -21.663 0
simundump text /kinetics/AC/AC2*/kenz/notes 0 \
  "Reduced Km to match expt data for basal activation of AC2 by PKC.\nNow k1 = 2.9e-6, k2 = 72, k3 = 18\n"
call /kinetics/AC/AC2*/kenz/notes LOAD \
"Reduced Km to match expt data for basal activation of AC2 by PKC." \
"Now k1 = 2.9e-6, k2 = 72, k3 = 18" \
""
simundump kpool /kinetics/AC/AC2-Gs 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry yellow blue -21.486 -21.709 0
simundump text /kinetics/AC/AC2-Gs/notes 0 ""
call /kinetics/AC/AC2-Gs/notes LOAD \
""
simundump kenz /kinetics/AC/AC2-Gs/kenz 1 0 0 0 0 6e+05 7.5e-06 72 18 0 1 "" \
  red yellow "" -21.564 -20.701 0
simundump text /kinetics/AC/AC2-Gs/kenz/notes 0 ""
call /kinetics/AC/AC2-Gs/kenz/notes LOAD \
""
simundump kpool /kinetics/AC/AC2 1 0 0.015 0.015 9000 9000 0 0 6e+05 0 \
  /kinetics/geometry yellow blue -17.606 -24.303 0
simundump text /kinetics/AC/AC2/notes 0 "Starting at 0.015 uM."
call /kinetics/AC/AC2/notes LOAD \
"Starting at 0.015 uM."
simundump kreac /kinetics/AC/dephosph-AC2 1 0.1 0 "" white blue -19.759 \
  -25.108 0
simundump text /kinetics/AC/dephosph-AC2/notes 0 "Random rate."
call /kinetics/AC/dephosph-AC2/notes LOAD \
"Random rate."
simundump kpool /kinetics/AC/AC1-Gs 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry orange blue -22.92 -16.507 0
simundump text /kinetics/AC/AC1-Gs/notes 0 ""
call /kinetics/AC/AC1-Gs/notes LOAD \
""
simundump kenz /kinetics/AC/AC1-Gs/kenz 1 0 0 0 0 1 7.5e-06 72 18 0 1 "" red \
  orange "" -21.945 -17.655 0
simundump text /kinetics/AC/AC1-Gs/kenz/notes 0 ""
call /kinetics/AC/AC1-Gs/kenz/notes LOAD \
""
simundump kreac /kinetics/AC/Gs-bind-AC2 1 0.00083333 1 "" white blue -20.17 \
  -27.142 0
simundump text /kinetics/AC/Gs-bind-AC2/notes 0 \
  "Half-max at around 3nM = kb/kf from fig 5 in \nFeinstein et al PNAS USA 88 10173-10177 1991\nkf = kb/1800 = 5.56e-4 kb\nOfer's thesis data indicates it is more like 2 nM.\nkf = kb/1200 = 8.33e-4\n"
call /kinetics/AC/Gs-bind-AC2/notes LOAD \
"Half-max at around 3nM = kb/kf from fig 5 in " \
"Feinstein et al PNAS USA 88 10173-10177 1991" \
"kf = kb/1800 = 5.56e-4 kb" \
"Ofer's thesis data indicates it is more like 2 nM." \
"kf = kb/1200 = 8.33e-4" \
""
simundump kreac /kinetics/AC/Gs-bind-AC1 1 0.00021 1 "" white blue -24.879 \
  -16.883 0
simundump text /kinetics/AC/Gs-bind-AC1/notes 0 \
  "Half-max 8nM from Tang et al JBC266:13 8595-8603\nkb/kf = 8 nM = 4800#/cell so kf = kb * 2.08e-4"
call /kinetics/AC/Gs-bind-AC1/notes LOAD \
"Half-max 8nM from Tang et al JBC266:13 8595-8603" \
"kb/kf = 8 nM = 4800#/cell so kf = kb * 2.08e-4"
simundump kpool /kinetics/AC/AMP 1 0 0.54248 0.54248 3.2549e+05 3.2549e+05 0 \
  0 6e+05 0 /kinetics/geometry[1] pink blue -23.649 -17.47 0
simundump text /kinetics/AC/AMP/notes 0 ""
call /kinetics/AC/AMP/notes LOAD \
""
simundump kpool /kinetics/AC/AC2*-Gs 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry green blue -20.142 -22.141 0
simundump text /kinetics/AC/AC2*-Gs/notes 0 ""
call /kinetics/AC/AC2*-Gs/notes LOAD \
""
simundump kenz /kinetics/AC/AC2*-Gs/kenz 1 0 0 0 0 1 7.5e-06 216 54 0 1 "" \
  red green "" -20.066 -21.087 0
simundump text /kinetics/AC/AC2*-Gs/kenz/notes 0 ""
call /kinetics/AC/AC2*-Gs/kenz/notes LOAD \
""
simundump kreac /kinetics/AC/Gs-bind-AC2* 1 0.0013888 1 "" white blue -20.343 \
  -23.991 0
simundump text /kinetics/AC/Gs-bind-AC2*/notes 0 \
  "kb/kf = 1.2 nM\nso kf = kb/720 = 1.3888 * kb."
call /kinetics/AC/Gs-bind-AC2*/notes LOAD \
"kb/kf = 1.2 nM" \
"so kf = kb/720 = 1.3888 * kb."
simundump kpool /kinetics/AC/cAMP-PDE 1 0 0.45 0.45 2.7e+05 2.7e+05 0 0 6e+05 \
  0 /kinetics/geometry green blue -26.712 -15.696 0
simundump text /kinetics/AC/cAMP-PDE/notes 0 \
  "The levels of the PDE are not known at this time. However,\nenough\nkinetic info and info about steady-state levels of cAMP\netc are around\nto make it possible to estimate this."
call /kinetics/AC/cAMP-PDE/notes LOAD \
"The levels of the PDE are not known at this time. However," \
"enough" \
"kinetic info and info about steady-state levels of cAMP" \
"etc are around" \
"to make it possible to estimate this."
simundump kenz /kinetics/AC/cAMP-PDE/PDE 1 0 0 0 0 6e+05 4.2e-06 40 10 0 0 "" \
  red green "" -26.821 -23.131 0
simundump text /kinetics/AC/cAMP-PDE/PDE/notes 0 \
  "Best rates are from Conti et al Biochem 34 7979-7987 1995.\nThough these\nare for the Sertoli cell form, it looks like they carry\nnicely into\nalternatively spliced brain form. See Sette et al\nJBC 269:28 18271-18274\nKm ~2 uM, Vmax est ~ 10 umol/min/mg for pure form.\nBrain protein is 93 kD but this was 67.\nSo k3 ~10, k2 ~40, k1 ~4.2e-6"
call /kinetics/AC/cAMP-PDE/PDE/notes LOAD \
"Best rates are from Conti et al Biochem 34 7979-7987 1995." \
"Though these" \
"are for the Sertoli cell form, it looks like they carry" \
"nicely into" \
"alternatively spliced brain form. See Sette et al" \
"JBC 269:28 18271-18274" \
"Km ~2 uM, Vmax est ~ 10 umol/min/mg for pure form." \
"Brain protein is 93 kD but this was 67." \
"So k3 ~10, k2 ~40, k1 ~4.2e-6"
simundump kpool /kinetics/AC/cAMP-PDE* 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry green blue -26.685 -17.78 0
simundump text /kinetics/AC/cAMP-PDE*/notes 0 \
  "This form has about 2X activity as plain PDE. See Sette et al JBC 269:28\n18271-18274 1994."
call /kinetics/AC/cAMP-PDE*/notes LOAD \
"This form has about 2X activity as plain PDE. See Sette et al JBC 269:28" \
"18271-18274 1994."
simundump kenz /kinetics/AC/cAMP-PDE*/PDE* 1 0 0 0 0 6e+05 8.4e-06 80 20 0 0 \
  "" red green "" -25.438 -22.305 0
simundump text /kinetics/AC/cAMP-PDE*/PDE*/notes 0 \
  "This form has about twice the activity of the unphosphorylated form. See\nSette et al JBC 269:28 18271-18274 1994.\nWe'll ignore cGMP effects for now."
call /kinetics/AC/cAMP-PDE*/PDE*/notes LOAD \
"This form has about twice the activity of the unphosphorylated form. See" \
"Sette et al JBC 269:28 18271-18274 1994." \
"We'll ignore cGMP effects for now."
simundump kreac /kinetics/AC/dephosph-PDE 1 0.1 0 "" white blue -28.587 \
  -18.842 0
simundump text /kinetics/AC/dephosph-PDE/notes 0 \
  "The rates for this are poorly constrained. In adipocytes (probably a\ndifferent PDE) the dephosphorylation is complete within 15 min, but\nthere are no intermediate time points so it could be much faster. Identity\nof phosphatase etc is still unknown."
call /kinetics/AC/dephosph-PDE/notes LOAD \
"The rates for this are poorly constrained. In adipocytes (probably a" \
"different PDE) the dephosphorylation is complete within 15 min, but" \
"there are no intermediate time points so it could be much faster. Identity" \
"of phosphatase etc is still unknown."
simundump kpool /kinetics/AC/PDE1 1 0 2 2 1.2e+06 1.2e+06 0 0 6e+05 0 \
  /kinetics/geometry green blue -30.493 -12.115 0
simundump text /kinetics/AC/PDE1/notes 0 \
  "CaM-Dependent PDE. Amount calculated from total rate in\nbrain vs. specific rate. "
call /kinetics/AC/PDE1/notes LOAD \
"CaM-Dependent PDE. Amount calculated from total rate in" \
"brain vs. specific rate. "
simundump kenz /kinetics/AC/PDE1/PDE1 1 0 0 0 0 6e+05 3.5e-07 6.67 1.667 0 0 \
  "" red green "" -27.426 -22.069 0
simundump text /kinetics/AC/PDE1/PDE1/notes 0 \
  "Rate is 1/6 of the CaM stim form. We'll just reduce\nall lf k1, k2, k3 so that the Vmax goes down 1/6."
call /kinetics/AC/PDE1/PDE1/notes LOAD \
"Rate is 1/6 of the CaM stim form. We'll just reduce" \
"all lf k1, k2, k3 so that the Vmax goes down 1/6."
simundump kpool /kinetics/AC/CaM.PDE1 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry green blue -30.493 -14.85 0
simundump text /kinetics/AC/CaM.PDE1/notes 0 \
  "Activity up 6x following Ca-CaM binding."
call /kinetics/AC/CaM.PDE1/notes LOAD \
"Activity up 6x following Ca-CaM binding."
simundump kenz /kinetics/AC/CaM.PDE1/CaM.PDE1 1 0 0 0 0 6e+05 2.1e-06 40 10 0 \
  0 "" red green "" -28.333 -21.282 0
simundump text /kinetics/AC/CaM.PDE1/CaM.PDE1/notes 0 \
  "Max activity ~10umol/min/mg in presence of lots of CaM.\nAffinity is low, 40 uM.\nk3 = 10, k2 = 40, k1 = (50/40) / 6e5."
call /kinetics/AC/CaM.PDE1/CaM.PDE1/notes LOAD \
"Max activity ~10umol/min/mg in presence of lots of CaM." \
"Affinity is low, 40 uM." \
"k3 = 10, k2 = 40, k1 = (50/40) / 6e5."
simundump kreac /kinetics/AC/CaM_bind_PDE1 1 0.0012 5 "" white blue -27.28 \
  -13.293 0
simundump text /kinetics/AC/CaM_bind_PDE1/notes 0 \
  "For olf epi PDE1, affinity is 7 nM. Assume same for brain.\nReaction should be pretty fast. Assume kb = 5/sec.\nThen kf = 5 / (0.007 * 6e5) = 1.2e-3"
call /kinetics/AC/CaM_bind_PDE1/notes LOAD \
"For olf epi PDE1, affinity is 7 nM. Assume same for brain." \
"Reaction should be pretty fast. Assume kb = 5/sec." \
"Then kf = 5 / (0.007 * 6e5) = 1.2e-3"
simundump kpool /kinetics/Gs-alpha 1 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry \
  red black -23.677 -28.03 0
simundump text /kinetics/Gs-alpha/notes 0 ""
call /kinetics/Gs-alpha/notes LOAD \
""
simundump kpool /kinetics/Ca 1 0 0.08 0.08 48000 48000 0 0 6e+05 0 \
  /kinetics/geometry red black -37.661 -0.21314 0
simundump text /kinetics/Ca/notes 0 ""
call /kinetics/Ca/notes LOAD \
""
simundump kpool /kinetics/Ca_input 0 0 0.08 0.08 48000 48000 0 0 6e+05 4 \
  /kinetics/geometry 61 black -35 4 0
simundump text /kinetics/Ca_input/notes 0 ""
call /kinetics/Ca_input/notes LOAD \
""
simundump kreac /kinetics/Ca_stoch 0 100 100 "" white black -38 2 0
simundump text /kinetics/Ca_stoch/notes 0 ""
call /kinetics/Ca_stoch/notes LOAD \
""
simundump kpool /kinetics/CaM_Ca_n-CaNAB 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry darkblue yellow -30.524 -6.6543 0
simundump text /kinetics/CaM_Ca_n-CaNAB/notes 0 ""
call /kinetics/CaM_Ca_n-CaNAB/notes LOAD \
""
simundump kenz /kinetics/CaM_Ca_n-CaNAB/dephosph_inhib1 1 0 0 0 0 6e+05 \
  5.7e-07 1.36 0.34 0 0 "" red darkblue "" -42.742 -17.357 0
simundump text /kinetics/CaM_Ca_n-CaNAB/dephosph_inhib1/notes 0 ""
call /kinetics/CaM_Ca_n-CaNAB/dephosph_inhib1/notes LOAD \
""
simundump kenz /kinetics/CaM_Ca_n-CaNAB/dephosph-PP1-I* 1 0 0 0 0 6e+05 \
  5.7e-07 1.36 0.34 0 0 "" white darkblue "" -41.24 -6.4435 0
simundump text /kinetics/CaM_Ca_n-CaNAB/dephosph-PP1-I*/notes 0 ""
call /kinetics/CaM_Ca_n-CaNAB/dephosph-PP1-I*/notes LOAD \
""
simundump xgraph /graphs/conc1 0 0 2000 0 0.01801 0
simundump xgraph /graphs/conc2 0 0 2000 0 12.2 0
simundump xplot /graphs/conc1/MAPK*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" orange 0 0 1
simundump xplot /graphs/conc1/PKC-active.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" red 0 0 1
simundump xplot /graphs/conc1/GTP-Ras.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" orange 0 0 1
simundump xplot /graphs/conc1/CaM-Ca4.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xplot /graphs/conc1/PKA-active.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" yellow 0 0 1
simundump xplot /graphs/conc1/cAMP.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" green 0 0 1
simundump xplot /graphs/conc1/PP1-active.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" cyan 0 0 1
simundump xplot /graphs/conc1/Raf-GTP-Ras*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" red 0 0 1
simundump xplot /graphs/conc2/tot_CaM_CaMKII.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" green 0 0 1
simundump xplot /graphs/conc2/tot_autonomous_CaMKII.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" green 0 0 1
simundump xplot /graphs/conc2/AA.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" darkgreen 0 0 1
simundump xplot /graphs/conc2/DAG.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" green 0 0 1
simundump xplot /graphs/conc2/Ca.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" red 0 0 1
simundump xplot /graphs/conc2/AC2*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" yellow 0 0 1
simundump xplot /graphs/conc2/CaM(Ca)n-CaNAB.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" darkblue 0 0 1
simundump xgraph /moregraphs/conc3 0 0 2000 -1.1176e-08 1 0
simundump xgraph /moregraphs/conc4 0 0 2000 0 1 0
simundump xcoredraw /edit/draw 0 -23.31 -5.4656 -23.566 -0.76346
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
xtextload /file/notes \
"10 Feb 2009." \
"dend_v27.g: Based on dend_v26.g. Halved CaNAB (PP2B) initial conc" \
"from 1 to 0.5" \
"" \
"4 Feb 2009" \
"dend_v26.g: Based on dendv21.g, altered Ca-bind-CaNAB Kb from" \
"1 to 0.1 to match rate in spine model." \
"" \
"24 Nov 2008" \
"dend_v21.g: Reduced MAPK from 3.6 uM to 3.0" \
"" \
"18 Nov 2008." \
"dend_v16.g Fixed rates for MKP-1." \
"dend_v17.g Added Raf-GTP-Ras" \
"dend_v18.g Fixed assorted rates and shifted positions around." \
"dend_v19.g Fixed naming and some rates." \
"dend_v20.g More fixes to rates. Fixed Raf-GTP-Ras.2. Buffered Ca_input." \
"" \
"" \
"17 Nov 2008" \
"Slowly converting over to using better values for MKP-1 and " \
"other enzymes. For now MKP-1 is all I have fixed." \
"" \
"15 Nov 2008. Slowed down some very high rates for Ca_bind_CaNAB" \
"and inhib_PP1. Fixed bad volumes for AC/AMP and PLCbeta/PC. " \
"Saved as v2." \
"v13: (skipping over many on an earlier branch): Reduced CaMKII" \
"to 2 uM, CaM to 5 uM, and cleaned out neurogranin from CaM model." \
"v14: more plots." \
"" \
"1 Sep 2003: Based on nonscaf_syn1c.g, which in turn is just " \
"like nonscaf_syn1b.g" \
"with all plots but MAPK* stripped out." \
"The current version provides the Ca input through a fast" \
"(10 msec) time-course reaction step so that the Ca can undergo" \
"fluctuations." \
"1 Oct 2003: Identical to nonscaf_syn1e.g, added a few plots." \
"9 Dec 2003: Identical to nonscaf_syn1f.g, added a plot for C"
addmsg /kinetics/PKC/PKC-act-by-Ca /kinetics/PKC/PKC-Ca REAC B A 
addmsg /kinetics/PKC/PKC-act-by-DAG /kinetics/PKC/PKC-Ca REAC A B 
addmsg /kinetics/PKC/PKC-Ca-to-memb /kinetics/PKC/PKC-Ca REAC A B 
addmsg /kinetics/PKC/PKC-act-by-Ca-AA /kinetics/PKC/PKC-Ca REAC A B 
addmsg /kinetics/PKC/PKC-cytosolic /kinetics/PKC/PKC-act-by-Ca SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/PKC/PKC-act-by-Ca SUBSTRATE n 
addmsg /kinetics/PKC/PKC-Ca /kinetics/PKC/PKC-act-by-Ca PRODUCT n 
addmsg /kinetics/DAG /kinetics/PKC/PKC-act-by-DAG SUBSTRATE n 
addmsg /kinetics/PKC/PKC-Ca /kinetics/PKC/PKC-act-by-DAG SUBSTRATE n 
addmsg /kinetics/PKC/PKC-Ca-DAG /kinetics/PKC/PKC-act-by-DAG PRODUCT n 
addmsg /kinetics/PKC/PKC-Ca /kinetics/PKC/PKC-Ca-to-memb SUBSTRATE n 
addmsg /kinetics/PKC/PKC-Ca-memb* /kinetics/PKC/PKC-Ca-to-memb PRODUCT n 
addmsg /kinetics/PKC/PKC-Ca-DAG /kinetics/PKC/PKC-DAG-to-memb SUBSTRATE n 
addmsg /kinetics/PKC/PKC-DAG-memb* /kinetics/PKC/PKC-DAG-to-memb PRODUCT n 
addmsg /kinetics/PKC/PKC-Ca /kinetics/PKC/PKC-act-by-Ca-AA SUBSTRATE n 
addmsg /kinetics/AA /kinetics/PKC/PKC-act-by-Ca-AA SUBSTRATE n 
addmsg /kinetics/PKC/PKC-Ca-AA* /kinetics/PKC/PKC-act-by-Ca-AA PRODUCT n 
addmsg /kinetics/PKC/PKC-DAG-AA* /kinetics/PKC/PKC-act-by-DAG-AA PRODUCT n 
addmsg /kinetics/PKC/PKC-DAG-AA /kinetics/PKC/PKC-act-by-DAG-AA SUBSTRATE n 
addmsg /kinetics/PKC/PKC-act-by-DAG-AA /kinetics/PKC/PKC-DAG-AA* REAC B A 
addmsg /kinetics/PKC/PKC-act-by-Ca-AA /kinetics/PKC/PKC-Ca-AA* REAC B A 
addmsg /kinetics/PKC/PKC-Ca-to-memb /kinetics/PKC/PKC-Ca-memb* REAC B A 
addmsg /kinetics/PKC/PKC-DAG-to-memb /kinetics/PKC/PKC-DAG-memb* REAC B A 
addmsg /kinetics/PKC/PKC-basal-act /kinetics/PKC/PKC-basal* REAC B A 
addmsg /kinetics/PKC/PKC-cytosolic /kinetics/PKC/PKC-basal-act SUBSTRATE n 
addmsg /kinetics/PKC/PKC-basal* /kinetics/PKC/PKC-basal-act PRODUCT n 
addmsg /kinetics/PKC/PKC-act-by-AA /kinetics/PKC/PKC-AA* REAC B A 
addmsg /kinetics/AA /kinetics/PKC/PKC-act-by-AA SUBSTRATE n 
addmsg /kinetics/PKC/PKC-AA* /kinetics/PKC/PKC-act-by-AA PRODUCT n 
addmsg /kinetics/PKC/PKC-cytosolic /kinetics/PKC/PKC-act-by-AA SUBSTRATE n 
addmsg /kinetics/PKC/PKC-act-by-DAG /kinetics/PKC/PKC-Ca-DAG REAC B A 
addmsg /kinetics/PKC/PKC-DAG-to-memb /kinetics/PKC/PKC-Ca-DAG REAC A B 
addmsg /kinetics/PKC/PKC-cytosolic /kinetics/PKC/PKC-n-DAG SUBSTRATE n 
addmsg /kinetics/DAG /kinetics/PKC/PKC-n-DAG SUBSTRATE n 
addmsg /kinetics/PKC/PKC-DAG /kinetics/PKC/PKC-n-DAG PRODUCT n 
addmsg /kinetics/PKC/PKC-n-DAG /kinetics/PKC/PKC-DAG REAC B A 
addmsg /kinetics/PKC/PKC-n-DAG-AA /kinetics/PKC/PKC-DAG REAC A B 
addmsg /kinetics/PKC/PKC-DAG /kinetics/PKC/PKC-n-DAG-AA SUBSTRATE n 
addmsg /kinetics/AA /kinetics/PKC/PKC-n-DAG-AA SUBSTRATE n 
addmsg /kinetics/PKC/PKC-DAG-AA /kinetics/PKC/PKC-n-DAG-AA PRODUCT n 
addmsg /kinetics/PKC/PKC-n-DAG-AA /kinetics/PKC/PKC-DAG-AA REAC B A 
addmsg /kinetics/PKC/PKC-act-by-DAG-AA /kinetics/PKC/PKC-DAG-AA REAC A B 
addmsg /kinetics/PKC/PKC-act-by-Ca /kinetics/PKC/PKC-cytosolic REAC A B 
addmsg /kinetics/PKC/PKC-basal-act /kinetics/PKC/PKC-cytosolic REAC A B 
addmsg /kinetics/PKC/PKC-act-by-AA /kinetics/PKC/PKC-cytosolic REAC A B 
addmsg /kinetics/PKC/PKC-n-DAG /kinetics/PKC/PKC-cytosolic REAC A B 
addmsg /kinetics/PKC/PKC-act-by-DAG /kinetics/DAG REAC A B 
addmsg /kinetics/PKC/PKC-n-DAG /kinetics/DAG REAC A B 
addmsg /kinetics/PLA2/DAG-Ca-PLA2-act /kinetics/DAG REAC A B 
addmsg /kinetics/PLCbeta/PLC-Ca-Gq/PLCb-Ca-Gq /kinetics/DAG MM_PRD pA 
addmsg /kinetics/PLCbeta/Degrade-DAG /kinetics/DAG REAC A B 
addmsg /kinetics/PLCbeta/PLC-Ca/PLC-Ca /kinetics/DAG MM_PRD pA 
addmsg /kinetics/PLC_g/Ca.PLC_g/PIP2_hydrolysis /kinetics/DAG MM_PRD pA 
addmsg /kinetics/PLC_g/Ca.PLC_g*/PIP2_hydrolysis /kinetics/DAG MM_PRD pA 
addmsg /kinetics/PKC/PKC-act-by-Ca-AA /kinetics/AA REAC A B 
addmsg /kinetics/PKC/PKC-act-by-AA /kinetics/AA REAC A B 
addmsg /kinetics/PKC/PKC-n-DAG-AA /kinetics/AA REAC A B 
addmsg /kinetics/PLA2/PLA2-Ca*/kenz /kinetics/AA MM_PRD pA 
addmsg /kinetics/PLA2/PIP2-PLA2*/kenz /kinetics/AA MM_PRD pA 
addmsg /kinetics/PLA2/PIP2-Ca-PLA2*/kenz /kinetics/AA MM_PRD pA 
addmsg /kinetics/PLA2/DAG-Ca-PLA2*/kenz /kinetics/AA MM_PRD pA 
addmsg /kinetics/PLA2/PLA2*-Ca/kenz /kinetics/AA MM_PRD pA 
addmsg /kinetics/PLA2/Degrade-AA /kinetics/AA REAC A B 
addmsg /kinetics/PKC/PKC-DAG-AA* /kinetics/PKC-active SUMTOTAL n nInit 
addmsg /kinetics/PKC/PKC-Ca-memb* /kinetics/PKC-active SUMTOTAL n nInit 
addmsg /kinetics/PKC/PKC-Ca-AA* /kinetics/PKC-active SUMTOTAL n nInit 
addmsg /kinetics/PKC/PKC-DAG-memb* /kinetics/PKC-active SUMTOTAL n nInit 
addmsg /kinetics/PKC/PKC-basal* /kinetics/PKC-active SUMTOTAL n nInit 
addmsg /kinetics/PKC/PKC-AA* /kinetics/PKC-active SUMTOTAL n nInit 
addmsg /kinetics/PKC-active/PKC-act-raf /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active/PKC-inact-GAP /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active/PKC-act-GEF /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active/phosph-AC2 /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-act-raf ENZYME n 
addmsg /kinetics/MAPK/craf-1 /kinetics/PKC-active/PKC-act-raf SUBSTRATE n 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-inact-GAP ENZYME n 
addmsg /kinetics/Ras/GAP /kinetics/PKC-active/PKC-inact-GAP SUBSTRATE n 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-act-GEF ENZYME n 
addmsg /kinetics/Ras/inact-GEF /kinetics/PKC-active/PKC-act-GEF SUBSTRATE n 
addmsg /kinetics/PKC-active /kinetics/PKC-active/phosph-AC2 ENZYME n 
addmsg /kinetics/AC/AC2 /kinetics/PKC-active/phosph-AC2 SUBSTRATE n 
addmsg /kinetics/PLA2/PLA2-Ca-act /kinetics/PLA2/PLA2-cytosolic REAC A B 
addmsg /kinetics/PLA2/PIP2-PLA2-act /kinetics/PLA2/PLA2-cytosolic REAC A B 
addmsg /kinetics/MAPK*/MAPK* /kinetics/PLA2/PLA2-cytosolic REAC sA B 
addmsg /kinetics/PLA2/dephosphorylate-PLA2* /kinetics/PLA2/PLA2-cytosolic REAC B A 
addmsg /kinetics/PLA2/PLA2-cytosolic /kinetics/PLA2/PLA2-Ca-act SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/PLA2/PLA2-Ca-act SUBSTRATE n 
addmsg /kinetics/PLA2/PLA2-Ca* /kinetics/PLA2/PLA2-Ca-act PRODUCT n 
addmsg /kinetics/PLA2/PLA2-Ca-act /kinetics/PLA2/PLA2-Ca* REAC B A 
addmsg /kinetics/PLA2/PLA2-Ca*/kenz /kinetics/PLA2/PLA2-Ca* REAC eA B 
addmsg /kinetics/PLA2/PIP2-Ca-PLA2-act /kinetics/PLA2/PLA2-Ca* REAC A B 
addmsg /kinetics/PLA2/DAG-Ca-PLA2-act /kinetics/PLA2/PLA2-Ca* REAC A B 
addmsg /kinetics/PLA2/PLA2-Ca* /kinetics/PLA2/PLA2-Ca*/kenz ENZYME n 
addmsg /kinetics/PLA2/APC /kinetics/PLA2/PLA2-Ca*/kenz SUBSTRATE n 
addmsg /kinetics/temp-PIP2 /kinetics/PLA2/PIP2-PLA2-act SUBSTRATE n 
addmsg /kinetics/PLA2/PLA2-cytosolic /kinetics/PLA2/PIP2-PLA2-act SUBSTRATE n 
addmsg /kinetics/PLA2/PIP2-PLA2* /kinetics/PLA2/PIP2-PLA2-act PRODUCT n 
addmsg /kinetics/PLA2/PIP2-PLA2-act /kinetics/PLA2/PIP2-PLA2* REAC B A 
addmsg /kinetics/PLA2/PIP2-PLA2*/kenz /kinetics/PLA2/PIP2-PLA2* REAC eA B 
addmsg /kinetics/PLA2/PIP2-PLA2* /kinetics/PLA2/PIP2-PLA2*/kenz ENZYME n 
addmsg /kinetics/PLA2/APC /kinetics/PLA2/PIP2-PLA2*/kenz SUBSTRATE n 
addmsg /kinetics/temp-PIP2 /kinetics/PLA2/PIP2-Ca-PLA2-act SUBSTRATE n 
addmsg /kinetics/PLA2/PLA2-Ca* /kinetics/PLA2/PIP2-Ca-PLA2-act SUBSTRATE n 
addmsg /kinetics/PLA2/PIP2-Ca-PLA2* /kinetics/PLA2/PIP2-Ca-PLA2-act PRODUCT n 
addmsg /kinetics/PLA2/PIP2-Ca-PLA2-act /kinetics/PLA2/PIP2-Ca-PLA2* REAC B A 
addmsg /kinetics/PLA2/PIP2-Ca-PLA2*/kenz /kinetics/PLA2/PIP2-Ca-PLA2* REAC eA B 
addmsg /kinetics/PLA2/PIP2-Ca-PLA2* /kinetics/PLA2/PIP2-Ca-PLA2*/kenz ENZYME n 
addmsg /kinetics/PLA2/APC /kinetics/PLA2/PIP2-Ca-PLA2*/kenz SUBSTRATE n 
addmsg /kinetics/DAG /kinetics/PLA2/DAG-Ca-PLA2-act SUBSTRATE n 
addmsg /kinetics/PLA2/PLA2-Ca* /kinetics/PLA2/DAG-Ca-PLA2-act SUBSTRATE n 
addmsg /kinetics/PLA2/DAG-Ca-PLA2* /kinetics/PLA2/DAG-Ca-PLA2-act PRODUCT n 
addmsg /kinetics/PLA2/DAG-Ca-PLA2-act /kinetics/PLA2/DAG-Ca-PLA2* REAC B A 
addmsg /kinetics/PLA2/DAG-Ca-PLA2*/kenz /kinetics/PLA2/DAG-Ca-PLA2* REAC eA B 
addmsg /kinetics/PLA2/DAG-Ca-PLA2* /kinetics/PLA2/DAG-Ca-PLA2*/kenz ENZYME n 
addmsg /kinetics/PLA2/APC /kinetics/PLA2/DAG-Ca-PLA2*/kenz SUBSTRATE n 
addmsg /kinetics/PLA2/PLA2-Ca*/kenz /kinetics/PLA2/APC REAC sA B 
addmsg /kinetics/PLA2/PIP2-PLA2*/kenz /kinetics/PLA2/APC REAC sA B 
addmsg /kinetics/PLA2/PIP2-Ca-PLA2*/kenz /kinetics/PLA2/APC REAC sA B 
addmsg /kinetics/PLA2/DAG-Ca-PLA2*/kenz /kinetics/PLA2/APC REAC sA B 
addmsg /kinetics/PLA2/PLA2*-Ca/kenz /kinetics/PLA2/APC REAC sA B 
addmsg /kinetics/PLA2/Degrade-AA /kinetics/PLA2/APC REAC B A 
addmsg /kinetics/AA /kinetics/PLA2/Degrade-AA SUBSTRATE n 
addmsg /kinetics/PLA2/APC /kinetics/PLA2/Degrade-AA PRODUCT n 
addmsg /kinetics/PLA2/PLA2*-Ca/kenz /kinetics/PLA2/PLA2*-Ca REAC eA B 
addmsg /kinetics/PLA2/PLA2*-Ca-act /kinetics/PLA2/PLA2*-Ca REAC B A 
addmsg /kinetics/PLA2/PLA2*-Ca /kinetics/PLA2/PLA2*-Ca/kenz ENZYME n 
addmsg /kinetics/PLA2/APC /kinetics/PLA2/PLA2*-Ca/kenz SUBSTRATE n 
addmsg /kinetics/MAPK*/MAPK* /kinetics/PLA2/PLA2* MM_PRD pA 
addmsg /kinetics/PLA2/PLA2*-Ca-act /kinetics/PLA2/PLA2* REAC A B 
addmsg /kinetics/PLA2/dephosphorylate-PLA2* /kinetics/PLA2/PLA2* REAC A B 
addmsg /kinetics/PLA2/PLA2* /kinetics/PLA2/PLA2*-Ca-act SUBSTRATE n 
addmsg /kinetics/PLA2/PLA2*-Ca /kinetics/PLA2/PLA2*-Ca-act PRODUCT n 
addmsg /kinetics/Ca /kinetics/PLA2/PLA2*-Ca-act SUBSTRATE n 
addmsg /kinetics/PLA2/PLA2* /kinetics/PLA2/dephosphorylate-PLA2* SUBSTRATE n 
addmsg /kinetics/PLA2/PLA2-cytosolic /kinetics/PLA2/dephosphorylate-PLA2* PRODUCT n 
addmsg /kinetics/MAPK*/MAPK* /kinetics/MAPK* REAC eA B 
addmsg /kinetics/MAPK*/MAPK*-feedback /kinetics/MAPK* REAC eA B 
addmsg /kinetics/MAPK/MAPKK*/MAPKKthr /kinetics/MAPK* MM_PRD pA 
addmsg /kinetics/MKP-1/MKP1-thr-deph /kinetics/MAPK* REAC sA B 
addmsg /kinetics/MAPK*/phosph_Sos /kinetics/MAPK* REAC eA B 
addmsg /kinetics/MAPK* /kinetics/MAPK*/MAPK* ENZYME n 
addmsg /kinetics/PLA2/PLA2-cytosolic /kinetics/MAPK*/MAPK* SUBSTRATE n 
addmsg /kinetics/MAPK* /kinetics/MAPK*/MAPK*-feedback ENZYME n 
addmsg /kinetics/MAPK/craf-1* /kinetics/MAPK*/MAPK*-feedback SUBSTRATE n 
addmsg /kinetics/MAPK* /kinetics/MAPK*/phosph_Sos ENZYME n 
addmsg /kinetics/Sos/Sos /kinetics/MAPK*/phosph_Sos SUBSTRATE n 
addmsg /kinetics/PLA2/PIP2-PLA2-act /kinetics/temp-PIP2 REAC A B 
addmsg /kinetics/PLA2/PIP2-Ca-PLA2-act /kinetics/temp-PIP2 REAC A B 
addmsg /kinetics/PLCbeta/PLC-Ca-Gq/PLCb-Ca-Gq /kinetics/IP3 MM_PRD pA 
addmsg /kinetics/PLCbeta/Degrade-IP3 /kinetics/IP3 REAC A B 
addmsg /kinetics/PLCbeta/PLC-Ca/PLC-Ca /kinetics/IP3 MM_PRD pA 
addmsg /kinetics/PLC_g/Ca.PLC_g/PIP2_hydrolysis /kinetics/IP3 MM_PRD pA 
addmsg /kinetics/PLC_g/Ca.PLC_g*/PIP2_hydrolysis /kinetics/IP3 MM_PRD pA 
addmsg /kinetics/Gq/RecLigandBinding /kinetics/Glu REAC A B 
addmsg /kinetics/Gq/Glu-bind-Rec-Gq /kinetics/Glu REAC A B 
addmsg /kinetics/Ca /kinetics/PLCbeta/Act-PLC-Ca SUBSTRATE n 
addmsg /kinetics/PLCbeta/PLC /kinetics/PLCbeta/Act-PLC-Ca SUBSTRATE n 
addmsg /kinetics/PLCbeta/PLC-Ca /kinetics/PLCbeta/Act-PLC-Ca PRODUCT n 
addmsg /kinetics/PLCbeta/Act-PLC-Ca /kinetics/PLCbeta/PLC REAC A B 
addmsg /kinetics/PLCbeta/PLC-bind-Gq /kinetics/PLCbeta/PLC REAC A B 
addmsg /kinetics/IP3 /kinetics/PLCbeta/Degrade-IP3 SUBSTRATE n 
addmsg /kinetics/PLCbeta/Inositol /kinetics/PLCbeta/Degrade-IP3 PRODUCT n 
addmsg /kinetics/PLCbeta/Degrade-IP3 /kinetics/PLCbeta/Inositol REAC B A 
addmsg /kinetics/PLCbeta/PC /kinetics/PLCbeta/Degrade-DAG PRODUCT n 
addmsg /kinetics/DAG /kinetics/PLCbeta/Degrade-DAG SUBSTRATE n 
addmsg /kinetics/PLCbeta/Degrade-DAG /kinetics/PLCbeta/PC REAC B A 
addmsg /kinetics/PLCbeta/Act-PLC-Ca /kinetics/PLCbeta/PLC-Ca REAC B A 
addmsg /kinetics/PLCbeta/Act-PLC-by-Gq /kinetics/PLCbeta/PLC-Ca REAC A B 
addmsg /kinetics/PLCbeta/Inact-PLC-Gq /kinetics/PLCbeta/PLC-Ca REAC B A 
addmsg /kinetics/PLCbeta/PLC-Ca/PLC-Ca /kinetics/PLCbeta/PLC-Ca REAC eA B 
addmsg /kinetics/PLCbeta/PLC-Ca /kinetics/PLCbeta/PLC-Ca/PLC-Ca ENZYME n 
addmsg /kinetics/PIP2 /kinetics/PLCbeta/PLC-Ca/PLC-Ca SUBSTRATE n 
addmsg /kinetics/G*GTP /kinetics/PLCbeta/Act-PLC-by-Gq SUBSTRATE n 
addmsg /kinetics/PLCbeta/PLC-Ca /kinetics/PLCbeta/Act-PLC-by-Gq SUBSTRATE n 
addmsg /kinetics/PLCbeta/PLC-Ca-Gq /kinetics/PLCbeta/Act-PLC-by-Gq PRODUCT n 
addmsg /kinetics/G*GDP /kinetics/PLCbeta/Inact-PLC-Gq PRODUCT n 
addmsg /kinetics/PLCbeta/PLC-Ca-Gq /kinetics/PLCbeta/Inact-PLC-Gq SUBSTRATE n 
addmsg /kinetics/PLCbeta/PLC-Ca /kinetics/PLCbeta/Inact-PLC-Gq PRODUCT n 
addmsg /kinetics/PLCbeta/PLC-Ca-Gq/PLCb-Ca-Gq /kinetics/PLCbeta/PLC-Ca-Gq REAC eA B 
addmsg /kinetics/PLCbeta/Act-PLC-by-Gq /kinetics/PLCbeta/PLC-Ca-Gq REAC B A 
addmsg /kinetics/PLCbeta/Inact-PLC-Gq /kinetics/PLCbeta/PLC-Ca-Gq REAC A B 
addmsg /kinetics/PLCbeta/PLC-Gq-bind-Ca /kinetics/PLCbeta/PLC-Ca-Gq REAC B A 
addmsg /kinetics/PLCbeta/PLC-Ca-Gq /kinetics/PLCbeta/PLC-Ca-Gq/PLCb-Ca-Gq ENZYME n 
addmsg /kinetics/PIP2 /kinetics/PLCbeta/PLC-Ca-Gq/PLCb-Ca-Gq SUBSTRATE n 
addmsg /kinetics/PLCbeta/PLC-bind-Gq /kinetics/PLCbeta/PLC-Gq REAC B A 
addmsg /kinetics/PLCbeta/PLC-Gq-bind-Ca /kinetics/PLCbeta/PLC-Gq REAC A B 
addmsg /kinetics/PLCbeta/PLC /kinetics/PLCbeta/PLC-bind-Gq SUBSTRATE n 
addmsg /kinetics/G*GTP /kinetics/PLCbeta/PLC-bind-Gq SUBSTRATE n 
addmsg /kinetics/PLCbeta/PLC-Gq /kinetics/PLCbeta/PLC-bind-Gq PRODUCT n 
addmsg /kinetics/Ca /kinetics/PLCbeta/PLC-Gq-bind-Ca SUBSTRATE n 
addmsg /kinetics/PLCbeta/PLC-Gq /kinetics/PLCbeta/PLC-Gq-bind-Ca SUBSTRATE n 
addmsg /kinetics/PLCbeta/PLC-Ca-Gq /kinetics/PLCbeta/PLC-Gq-bind-Ca PRODUCT n 
addmsg /kinetics/PLCbeta/PLC-Ca-Gq/PLCb-Ca-Gq /kinetics/PIP2 REAC sA B 
addmsg /kinetics/PLCbeta/PLC-Ca/PLC-Ca /kinetics/PIP2 REAC sA B 
addmsg /kinetics/PLC_g/Ca.PLC_g*/PIP2_hydrolysis /kinetics/PIP2 REAC sA B 
addmsg /kinetics/PLC_g/Ca.PLC_g/PIP2_hydrolysis /kinetics/PIP2 REAC sA B 
addmsg /kinetics/Gq/Trimerize-G /kinetics/BetaGamma REAC A B 
addmsg /kinetics/Gq/Basal-Act-G /kinetics/BetaGamma REAC B A 
addmsg /kinetics/Gq/Activate-Gq /kinetics/BetaGamma REAC B A 
addmsg /kinetics/Ras/bg-act-GEF /kinetics/BetaGamma REAC A B 
addmsg /kinetics/PLCbeta/Act-PLC-by-Gq /kinetics/G*GTP REAC A B 
addmsg /kinetics/PLCbeta/PLC-bind-Gq /kinetics/G*GTP REAC A B 
addmsg /kinetics/Gq/Inact-G /kinetics/G*GTP REAC A B 
addmsg /kinetics/Gq/Basal-Act-G /kinetics/G*GTP REAC B A 
addmsg /kinetics/Gq/Activate-Gq /kinetics/G*GTP REAC B A 
addmsg /kinetics/PLCbeta/Inact-PLC-Gq /kinetics/G*GDP REAC B A 
addmsg /kinetics/Gq/Inact-G /kinetics/G*GDP REAC B A 
addmsg /kinetics/Gq/Trimerize-G /kinetics/G*GDP REAC A B 
addmsg /kinetics/Gq/mGluR /kinetics/Gq/RecLigandBinding SUBSTRATE n 
addmsg /kinetics/Glu /kinetics/Gq/RecLigandBinding SUBSTRATE n 
addmsg /kinetics/Gq/Rec-Glu /kinetics/Gq/RecLigandBinding PRODUCT n 
addmsg /kinetics/Gq/Trimerize-G /kinetics/Gq/G-GDP REAC B A 
addmsg /kinetics/Gq/Basal-Act-G /kinetics/Gq/G-GDP REAC A B 
addmsg /kinetics/Gq/Rec-Glu-bind-Gq /kinetics/Gq/G-GDP REAC A B 
addmsg /kinetics/Gq/Rec-bind-Gq /kinetics/Gq/G-GDP REAC A B 
addmsg /kinetics/Gq/G-GDP /kinetics/Gq/Basal-Act-G SUBSTRATE n 
addmsg /kinetics/G*GTP /kinetics/Gq/Basal-Act-G PRODUCT n 
addmsg /kinetics/BetaGamma /kinetics/Gq/Basal-Act-G PRODUCT n 
addmsg /kinetics/G*GDP /kinetics/Gq/Trimerize-G SUBSTRATE n 
addmsg /kinetics/BetaGamma /kinetics/Gq/Trimerize-G SUBSTRATE n 
addmsg /kinetics/Gq/G-GDP /kinetics/Gq/Trimerize-G PRODUCT n 
addmsg /kinetics/G*GTP /kinetics/Gq/Inact-G SUBSTRATE n 
addmsg /kinetics/G*GDP /kinetics/Gq/Inact-G PRODUCT n 
addmsg /kinetics/Gq/RecLigandBinding /kinetics/Gq/mGluR REAC A B 
addmsg /kinetics/Gq/Rec-bind-Gq /kinetics/Gq/mGluR REAC A B 
addmsg /kinetics/Gq/RecLigandBinding /kinetics/Gq/Rec-Glu REAC B A 
addmsg /kinetics/Gq/Rec-Glu-bind-Gq /kinetics/Gq/Rec-Glu REAC A B 
addmsg /kinetics/Gq/Activate-Gq /kinetics/Gq/Rec-Glu REAC B A 
addmsg /kinetics/Gq/Glu-bind-Rec-Gq /kinetics/Gq/Rec-Gq REAC A B 
addmsg /kinetics/Gq/Rec-bind-Gq /kinetics/Gq/Rec-Gq REAC B A 
addmsg /kinetics/Gq/Antag-bind-Rec-Gq /kinetics/Gq/Rec-Gq REAC A B 
addmsg /kinetics/Gq/G-GDP /kinetics/Gq/Rec-Glu-bind-Gq SUBSTRATE n 
addmsg /kinetics/Gq/Rec-Glu /kinetics/Gq/Rec-Glu-bind-Gq SUBSTRATE n 
addmsg /kinetics/Gq/Rec-Glu-Gq /kinetics/Gq/Rec-Glu-bind-Gq PRODUCT n 
addmsg /kinetics/Glu /kinetics/Gq/Glu-bind-Rec-Gq SUBSTRATE n 
addmsg /kinetics/Gq/Rec-Glu-Gq /kinetics/Gq/Glu-bind-Rec-Gq PRODUCT n 
addmsg /kinetics/Gq/Rec-Gq /kinetics/Gq/Glu-bind-Rec-Gq SUBSTRATE n 
addmsg /kinetics/Gq/Rec-Glu-bind-Gq /kinetics/Gq/Rec-Glu-Gq REAC B A 
addmsg /kinetics/Gq/Glu-bind-Rec-Gq /kinetics/Gq/Rec-Glu-Gq REAC B A 
addmsg /kinetics/Gq/Activate-Gq /kinetics/Gq/Rec-Glu-Gq REAC A B 
addmsg /kinetics/Gq/Rec-Glu-Gq /kinetics/Gq/Activate-Gq SUBSTRATE n 
addmsg /kinetics/G*GTP /kinetics/Gq/Activate-Gq PRODUCT n 
addmsg /kinetics/BetaGamma /kinetics/Gq/Activate-Gq PRODUCT n 
addmsg /kinetics/Gq/Rec-Glu /kinetics/Gq/Activate-Gq PRODUCT n 
addmsg /kinetics/Gq/G-GDP /kinetics/Gq/Rec-bind-Gq SUBSTRATE n 
addmsg /kinetics/Gq/mGluR /kinetics/Gq/Rec-bind-Gq SUBSTRATE n 
addmsg /kinetics/Gq/Rec-Gq /kinetics/Gq/Rec-bind-Gq PRODUCT n 
addmsg /kinetics/Gq/Antag-bind-Rec-Gq /kinetics/Gq/mGluRAntag REAC A B 
addmsg /kinetics/Gq/Rec-Gq /kinetics/Gq/Antag-bind-Rec-Gq SUBSTRATE n 
addmsg /kinetics/Gq/mGluRAntag /kinetics/Gq/Antag-bind-Rec-Gq SUBSTRATE n 
addmsg /kinetics/Gq/Blocked-rec-Gq /kinetics/Gq/Antag-bind-Rec-Gq PRODUCT n 
addmsg /kinetics/Gq/Antag-bind-Rec-Gq /kinetics/Gq/Blocked-rec-Gq REAC B A 
addmsg /kinetics/PKC-active/PKC-act-raf /kinetics/MAPK/craf-1 REAC sA B 
addmsg /kinetics/PPhosphatase2A/craf-deph /kinetics/MAPK/craf-1 MM_PRD pA 
addmsg /kinetics/MAPK/Ras-act-unphosph-raf /kinetics/MAPK/craf-1 REAC A B 
addmsg /kinetics/PKC-active/PKC-act-raf /kinetics/MAPK/craf-1* MM_PRD pA 
addmsg /kinetics/MAPK*/MAPK*-feedback /kinetics/MAPK/craf-1* REAC sA B 
addmsg /kinetics/PPhosphatase2A/craf-deph /kinetics/MAPK/craf-1* REAC sA B 
addmsg /kinetics/PPhosphatase2A/craf**-deph /kinetics/MAPK/craf-1* MM_PRD pA 
addmsg /kinetics/MAPK/Ras-act-craf /kinetics/MAPK/craf-1* REAC A B 
addmsg /kinetics/PPhosphatase2A/MAPKK-deph-ser /kinetics/MAPK/MAPKK MM_PRD pA 
addmsg /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1 /kinetics/MAPK/MAPKK REAC sA B 
addmsg /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1 /kinetics/MAPK/MAPKK REAC sA B 
addmsg /kinetics/MAPK/MAPKK*/MAPKKtyr /kinetics/MAPK/MAPK REAC sA B 
addmsg /kinetics/MKP-1/MKP1-tyr-deph /kinetics/MAPK/MAPK MM_PRD pA 
addmsg /kinetics/MAPK*/MAPK*-feedback /kinetics/MAPK/craf-1** MM_PRD pA 
addmsg /kinetics/PPhosphatase2A/craf**-deph /kinetics/MAPK/craf-1** REAC sA B 
addmsg /kinetics/MAPK/MAPKK*/MAPKKtyr /kinetics/MAPK/MAPK-tyr MM_PRD pA 
addmsg /kinetics/MAPK/MAPKK*/MAPKKthr /kinetics/MAPK/MAPK-tyr REAC sA B 
addmsg /kinetics/MKP-1/MKP1-tyr-deph /kinetics/MAPK/MAPK-tyr REAC sA B 
addmsg /kinetics/MKP-1/MKP1-thr-deph /kinetics/MAPK/MAPK-tyr MM_PRD pA 
addmsg /kinetics/MAPK/MAPKK*/MAPKKtyr /kinetics/MAPK/MAPKK* REAC eA B 
addmsg /kinetics/MAPK/MAPKK*/MAPKKthr /kinetics/MAPK/MAPKK* REAC eA B 
addmsg /kinetics/PPhosphatase2A/MAPKK-deph /kinetics/MAPK/MAPKK* REAC sA B 
addmsg /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2 /kinetics/MAPK/MAPKK* MM_PRD pA 
addmsg /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2 /kinetics/MAPK/MAPKK* MM_PRD pA 
addmsg /kinetics/MAPK/MAPKK* /kinetics/MAPK/MAPKK*/MAPKKtyr ENZYME n 
addmsg /kinetics/MAPK/MAPK /kinetics/MAPK/MAPKK*/MAPKKtyr SUBSTRATE n 
addmsg /kinetics/MAPK/MAPKK* /kinetics/MAPK/MAPKK*/MAPKKthr ENZYME n 
addmsg /kinetics/MAPK/MAPK-tyr /kinetics/MAPK/MAPKK*/MAPKKthr SUBSTRATE n 
addmsg /kinetics/PPhosphatase2A/MAPKK-deph /kinetics/MAPK/MAPKK-ser MM_PRD pA 
addmsg /kinetics/PPhosphatase2A/MAPKK-deph-ser /kinetics/MAPK/MAPKK-ser REAC sA B 
addmsg /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1 /kinetics/MAPK/MAPKK-ser MM_PRD pA 
addmsg /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2 /kinetics/MAPK/MAPKK-ser REAC sA B 
addmsg /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1 /kinetics/MAPK/MAPKK-ser MM_PRD pA 
addmsg /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2 /kinetics/MAPK/MAPKK-ser REAC sA B 
addmsg /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1 /kinetics/MAPK/Raf-GTP-Ras REAC eA B 
addmsg /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2 /kinetics/MAPK/Raf-GTP-Ras REAC eA B 
addmsg /kinetics/MAPK/Ras-act-unphosph-raf /kinetics/MAPK/Raf-GTP-Ras REAC B A 
addmsg /kinetics/MAPK/Raf-GTP-Ras /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1 ENZYME n 
addmsg /kinetics/MAPK/MAPKK /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1 SUBSTRATE n 
addmsg /kinetics/MAPK/Raf-GTP-Ras /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2 ENZYME n 
addmsg /kinetics/MAPK/MAPKK-ser /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2 SUBSTRATE n 
addmsg /kinetics/Ras/GTP-Ras /kinetics/MAPK/Ras-act-unphosph-raf SUBSTRATE n 
addmsg /kinetics/MAPK/Raf-GTP-Ras /kinetics/MAPK/Ras-act-unphosph-raf PRODUCT n 
addmsg /kinetics/MAPK/craf-1 /kinetics/MAPK/Ras-act-unphosph-raf SUBSTRATE n 
addmsg /kinetics/MAPK/Raf*-GTP-Ras /kinetics/MAPK/Ras-act-craf PRODUCT n 
addmsg /kinetics/MAPK/craf-1* /kinetics/MAPK/Ras-act-craf SUBSTRATE n 
addmsg /kinetics/Ras/GTP-Ras /kinetics/MAPK/Ras-act-craf SUBSTRATE n 
addmsg /kinetics/MAPK/Ras-act-craf /kinetics/MAPK/Raf*-GTP-Ras REAC B A 
addmsg /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1 /kinetics/MAPK/Raf*-GTP-Ras REAC eA B 
addmsg /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2 /kinetics/MAPK/Raf*-GTP-Ras REAC eA B 
addmsg /kinetics/MAPK/Raf*-GTP-Ras /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1 ENZYME n 
addmsg /kinetics/MAPK/MAPKK /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1 SUBSTRATE n 
addmsg /kinetics/MAPK/Raf*-GTP-Ras /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2 ENZYME n 
addmsg /kinetics/MAPK/MAPKK-ser /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2 SUBSTRATE n 
addmsg /kinetics/MKP-1/MKP1-tyr-deph /kinetics/MKP-1 REAC eA B 
addmsg /kinetics/MKP-1/MKP1-thr-deph /kinetics/MKP-1 REAC eA B 
addmsg /kinetics/MKP-1 /kinetics/MKP-1/MKP1-tyr-deph ENZYME n 
addmsg /kinetics/MAPK/MAPK-tyr /kinetics/MKP-1/MKP1-tyr-deph SUBSTRATE n 
addmsg /kinetics/MKP-1 /kinetics/MKP-1/MKP1-thr-deph ENZYME n 
addmsg /kinetics/MAPK* /kinetics/MKP-1/MKP1-thr-deph SUBSTRATE n 
addmsg /kinetics/PPhosphatase2A/craf-deph /kinetics/PPhosphatase2A REAC eA B 
addmsg /kinetics/PPhosphatase2A/MAPKK-deph /kinetics/PPhosphatase2A REAC eA B 
addmsg /kinetics/PPhosphatase2A/MAPKK-deph-ser /kinetics/PPhosphatase2A REAC eA B 
addmsg /kinetics/PPhosphatase2A/craf**-deph /kinetics/PPhosphatase2A REAC eA B 
addmsg /kinetics/PPhosphatase2A /kinetics/PPhosphatase2A/craf-deph ENZYME n 
addmsg /kinetics/MAPK/craf-1* /kinetics/PPhosphatase2A/craf-deph SUBSTRATE n 
addmsg /kinetics/PPhosphatase2A /kinetics/PPhosphatase2A/MAPKK-deph ENZYME n 
addmsg /kinetics/MAPK/MAPKK* /kinetics/PPhosphatase2A/MAPKK-deph SUBSTRATE n 
addmsg /kinetics/PPhosphatase2A /kinetics/PPhosphatase2A/MAPKK-deph-ser ENZYME n 
addmsg /kinetics/MAPK/MAPKK-ser /kinetics/PPhosphatase2A/MAPKK-deph-ser SUBSTRATE n 
addmsg /kinetics/PPhosphatase2A /kinetics/PPhosphatase2A/craf**-deph ENZYME n 
addmsg /kinetics/MAPK/craf-1** /kinetics/PPhosphatase2A/craf**-deph SUBSTRATE n 
addmsg /kinetics/BetaGamma /kinetics/Ras/bg-act-GEF SUBSTRATE n 
addmsg /kinetics/Ras/inact-GEF /kinetics/Ras/bg-act-GEF SUBSTRATE n 
addmsg /kinetics/Ras/GEF-Gprot-bg /kinetics/Ras/bg-act-GEF PRODUCT n 
addmsg /kinetics/Ras/GEF-Gprot-bg/GEF-bg_act-ras /kinetics/Ras/GEF-Gprot-bg REAC eA B 
addmsg /kinetics/Ras/bg-act-GEF /kinetics/Ras/GEF-Gprot-bg REAC B A 
addmsg /kinetics/Ras/GEF-Gprot-bg /kinetics/Ras/GEF-Gprot-bg/GEF-bg_act-ras ENZYME n 
addmsg /kinetics/Ras/GDP-Ras /kinetics/Ras/GEF-Gprot-bg/GEF-bg_act-ras SUBSTRATE n 
addmsg /kinetics/Ras/GEF* /kinetics/Ras/dephosph-GEF SUBSTRATE n 
addmsg /kinetics/Ras/inact-GEF /kinetics/Ras/dephosph-GEF PRODUCT n 
addmsg /kinetics/Ras/bg-act-GEF /kinetics/Ras/inact-GEF REAC A B 
addmsg /kinetics/PKC-active/PKC-act-GEF /kinetics/Ras/inact-GEF REAC sA B 
addmsg /kinetics/Ras/dephosph-GEF /kinetics/Ras/inact-GEF REAC B A 
addmsg /kinetics/PKA-active/PKA-phosph-GEF /kinetics/Ras/inact-GEF REAC sA B 
addmsg /kinetics/Ras/CaM-bind-GEF /kinetics/Ras/inact-GEF REAC A B 
addmsg /kinetics/Ras/dephosph-inact-GEF* /kinetics/Ras/inact-GEF REAC B A 
addmsg /kinetics/PKC-active/PKC-act-GEF /kinetics/Ras/GEF* MM_PRD pA 
addmsg /kinetics/Ras/dephosph-GEF /kinetics/Ras/GEF* REAC A B 
addmsg /kinetics/Ras/GEF*/GEF*-act-ras /kinetics/Ras/GEF* REAC eA B 
addmsg /kinetics/Ras/GEF* /kinetics/Ras/GEF*/GEF*-act-ras ENZYME n 
addmsg /kinetics/Ras/GDP-Ras /kinetics/Ras/GEF*/GEF*-act-ras SUBSTRATE n 
addmsg /kinetics/Ras/GEF-Gprot-bg/GEF-bg_act-ras /kinetics/Ras/GTP-Ras MM_PRD pA 
addmsg /kinetics/Ras/GAP/GAP-inact-ras /kinetics/Ras/GTP-Ras REAC sA B 
addmsg /kinetics/Ras/Ras-intrinsic-GTPase /kinetics/Ras/GTP-Ras REAC A B 
addmsg /kinetics/Ras/GEF*/GEF*-act-ras /kinetics/Ras/GTP-Ras MM_PRD pA 
addmsg /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras /kinetics/Ras/GTP-Ras MM_PRD pA 
addmsg /kinetics/MAPK/Ras-act-craf /kinetics/Ras/GTP-Ras REAC A B 
addmsg /kinetics/Shc*.Sos.Grb2/Sos.Ras_GEF /kinetics/Ras/GTP-Ras MM_PRD pA 
addmsg /kinetics/MAPK/Ras-act-unphosph-raf /kinetics/Ras/GTP-Ras REAC A B 
addmsg /kinetics/Ras/GEF-Gprot-bg/GEF-bg_act-ras /kinetics/Ras/GDP-Ras REAC sA B 
addmsg /kinetics/Ras/GAP/GAP-inact-ras /kinetics/Ras/GDP-Ras MM_PRD pA 
addmsg /kinetics/Ras/Ras-intrinsic-GTPase /kinetics/Ras/GDP-Ras REAC B A 
addmsg /kinetics/Ras/GEF*/GEF*-act-ras /kinetics/Ras/GDP-Ras REAC sA B 
addmsg /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras /kinetics/Ras/GDP-Ras REAC sA B 
addmsg /kinetics/Shc*.Sos.Grb2/Sos.Ras_GEF /kinetics/Ras/GDP-Ras REAC sA B 
addmsg /kinetics/Ras/GTP-Ras /kinetics/Ras/Ras-intrinsic-GTPase SUBSTRATE n 
addmsg /kinetics/Ras/GDP-Ras /kinetics/Ras/Ras-intrinsic-GTPase PRODUCT n 
addmsg /kinetics/Ras/GAP* /kinetics/Ras/dephosph-GAP SUBSTRATE n 
addmsg /kinetics/Ras/GAP /kinetics/Ras/dephosph-GAP PRODUCT n 
addmsg /kinetics/PKC-active/PKC-inact-GAP /kinetics/Ras/GAP* MM_PRD pA 
addmsg /kinetics/Ras/dephosph-GAP /kinetics/Ras/GAP* REAC A B 
addmsg /kinetics/Ras/GAP/GAP-inact-ras /kinetics/Ras/GAP REAC eA B 
addmsg /kinetics/PKC-active/PKC-inact-GAP /kinetics/Ras/GAP REAC sA B 
addmsg /kinetics/Ras/dephosph-GAP /kinetics/Ras/GAP REAC B A 
addmsg /kinetics/Ras/GAP /kinetics/Ras/GAP/GAP-inact-ras ENZYME n 
addmsg /kinetics/Ras/GTP-Ras /kinetics/Ras/GAP/GAP-inact-ras SUBSTRATE n 
addmsg /kinetics/PKA-active/PKA-phosph-GEF /kinetics/Ras/inact-GEF* MM_PRD pA 
addmsg /kinetics/Ras/dephosph-inact-GEF* /kinetics/Ras/inact-GEF* REAC A B 
addmsg /kinetics/Ras/inact-GEF /kinetics/Ras/CaM-bind-GEF SUBSTRATE n 
addmsg /kinetics/Ras/CaM-GEF /kinetics/Ras/CaM-bind-GEF PRODUCT n 
addmsg /kinetics/CaM-Ca4 /kinetics/Ras/CaM-bind-GEF SUBSTRATE n 
addmsg /kinetics/Ras/CaM-bind-GEF /kinetics/Ras/CaM-GEF REAC B A 
addmsg /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras /kinetics/Ras/CaM-GEF REAC eA B 
addmsg /kinetics/Ras/CaM-GEF /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras ENZYME n 
addmsg /kinetics/Ras/GDP-Ras /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras SUBSTRATE n 
addmsg /kinetics/Ras/inact-GEF* /kinetics/Ras/dephosph-inact-GEF* SUBSTRATE n 
addmsg /kinetics/Ras/inact-GEF /kinetics/Ras/dephosph-inact-GEF* PRODUCT n 
addmsg /kinetics/PKA-active/PKA-phosph-GEF /kinetics/PKA-active REAC eA B 
addmsg /kinetics/PKA-active/PKA-phosph-I1 /kinetics/PKA-active REAC eA B 
addmsg /kinetics/PKA/Release-C1 /kinetics/PKA-active REAC B A 
addmsg /kinetics/PKA/Release-C2 /kinetics/PKA-active REAC B A 
addmsg /kinetics/PKA/inhib-PKA /kinetics/PKA-active REAC A B 
addmsg /kinetics/PKA-active/phosph-PDE /kinetics/PKA-active REAC eA B 
addmsg /kinetics/PKA-active /kinetics/PKA-active/PKA-phosph-GEF ENZYME n 
addmsg /kinetics/Ras/inact-GEF /kinetics/PKA-active/PKA-phosph-GEF SUBSTRATE n 
addmsg /kinetics/PKA-active /kinetics/PKA-active/PKA-phosph-I1 ENZYME n 
addmsg /kinetics/PP1/I1 /kinetics/PKA-active/PKA-phosph-I1 SUBSTRATE n 
addmsg /kinetics/PKA-active /kinetics/PKA-active/phosph-PDE ENZYME n 
addmsg /kinetics/AC/cAMP-PDE /kinetics/PKA-active/phosph-PDE SUBSTRATE n 
addmsg /kinetics/Ras/CaM-bind-GEF /kinetics/CaM-Ca4 REAC A B 
addmsg /kinetics/CaMKII/CaMKII-bind-CaM /kinetics/CaM-Ca4 REAC A B 
addmsg /kinetics/CaMKII/CaMK-thr286-bind-CaM /kinetics/CaM-Ca4 REAC A B 
addmsg /kinetics/CaM/CaM-Ca3-bind-Ca /kinetics/CaM-Ca4 REAC B A 
addmsg /kinetics/PP2B/CaMCa4-bind-CaNAB /kinetics/CaM-Ca4 REAC A B 
addmsg /kinetics/AC/CaM-bind-AC1 /kinetics/CaM-Ca4 REAC A B 
addmsg /kinetics/AC/CaM_bind_PDE1 /kinetics/CaM-Ca4 REAC A B 
addmsg /kinetics/Sos/Shc_bind_Sos.Grb2 /kinetics/Shc*.Sos.Grb2 REAC B A 
addmsg /kinetics/Shc*.Sos.Grb2/Sos.Ras_GEF /kinetics/Shc*.Sos.Grb2 REAC eA B 
addmsg /kinetics/Shc*.Sos.Grb2 /kinetics/Shc*.Sos.Grb2/Sos.Ras_GEF ENZYME n 
addmsg /kinetics/Ras/GDP-Ras /kinetics/Shc*.Sos.Grb2/Sos.Ras_GEF SUBSTRATE n 
addmsg /kinetics/EGFR/act_EGFR /kinetics/EGFR/EGFR REAC A B 
addmsg /kinetics/EGFR/EGFR /kinetics/EGFR/act_EGFR SUBSTRATE n 
addmsg /kinetics/EGFR/EGF /kinetics/EGFR/act_EGFR SUBSTRATE n 
addmsg /kinetics/EGFR/L.EGFR /kinetics/EGFR/act_EGFR PRODUCT n 
addmsg /kinetics/EGFR/act_EGFR /kinetics/EGFR/L.EGFR REAC B A 
addmsg /kinetics/EGFR/L.EGFR/phosph_PLC_g /kinetics/EGFR/L.EGFR REAC eA B 
addmsg /kinetics/EGFR/L.EGFR/phosph_Shc /kinetics/EGFR/L.EGFR REAC eA B 
addmsg /kinetics/EGFR/Internalize /kinetics/EGFR/L.EGFR REAC A B 
addmsg /kinetics/EGFR/L.EGFR /kinetics/EGFR/L.EGFR/phosph_PLC_g ENZYME n 
addmsg /kinetics/PLC_g/Ca.PLC_g /kinetics/EGFR/L.EGFR/phosph_PLC_g SUBSTRATE n 
addmsg /kinetics/EGFR/L.EGFR /kinetics/EGFR/L.EGFR/phosph_Shc ENZYME n 
addmsg /kinetics/EGFR/SHC /kinetics/EGFR/L.EGFR/phosph_Shc SUBSTRATE n 
addmsg /kinetics/EGFR/act_EGFR /kinetics/EGFR/EGF REAC A B 
addmsg /kinetics/EGFR/dephosph_Shc /kinetics/EGFR/SHC REAC B A 
addmsg /kinetics/EGFR/L.EGFR/phosph_Shc /kinetics/EGFR/SHC REAC sA B 
addmsg /kinetics/EGFR/dephosph_Shc /kinetics/EGFR/SHC* REAC A B 
addmsg /kinetics/Sos/Shc_bind_Sos.Grb2 /kinetics/EGFR/SHC* REAC A B 
addmsg /kinetics/EGFR/L.EGFR/phosph_Shc /kinetics/EGFR/SHC* MM_PRD pA 
addmsg /kinetics/EGFR/SHC* /kinetics/EGFR/dephosph_Shc SUBSTRATE n 
addmsg /kinetics/EGFR/SHC /kinetics/EGFR/dephosph_Shc PRODUCT n 
addmsg /kinetics/EGFR/Internalize /kinetics/EGFR/Internal_L.EGFR REAC B A 
addmsg /kinetics/EGFR/L.EGFR /kinetics/EGFR/Internalize SUBSTRATE n 
addmsg /kinetics/EGFR/Internal_L.EGFR /kinetics/EGFR/Internalize PRODUCT n 
addmsg /kinetics/Sos/Sos.Grb2 /kinetics/Sos/Shc_bind_Sos.Grb2 SUBSTRATE n 
addmsg /kinetics/EGFR/SHC* /kinetics/Sos/Shc_bind_Sos.Grb2 SUBSTRATE n 
addmsg /kinetics/Shc*.Sos.Grb2 /kinetics/Sos/Shc_bind_Sos.Grb2 PRODUCT n 
addmsg /kinetics/Sos/Grb2_bind_Sos* /kinetics/Sos/Sos*.Grb2 REAC B A 
addmsg /kinetics/Sos/Sos* /kinetics/Sos/Grb2_bind_Sos* SUBSTRATE n 
addmsg /kinetics/Sos/Grb2 /kinetics/Sos/Grb2_bind_Sos* SUBSTRATE n 
addmsg /kinetics/Sos/Sos*.Grb2 /kinetics/Sos/Grb2_bind_Sos* PRODUCT n 
addmsg /kinetics/Sos/Grb2_bind_Sos /kinetics/Sos/Grb2 REAC A B 
addmsg /kinetics/Sos/Grb2_bind_Sos* /kinetics/Sos/Grb2 REAC A B 
addmsg /kinetics/Sos/Grb2_bind_Sos /kinetics/Sos/Sos.Grb2 REAC B A 
addmsg /kinetics/Sos/Shc_bind_Sos.Grb2 /kinetics/Sos/Sos.Grb2 REAC A B 
addmsg /kinetics/MAPK*/phosph_Sos /kinetics/Sos/Sos* MM_PRD pA 
addmsg /kinetics/Sos/Grb2_bind_Sos* /kinetics/Sos/Sos* REAC A B 
addmsg /kinetics/Sos/dephosph_Sos /kinetics/Sos/Sos* REAC A B 
addmsg /kinetics/Sos/Sos* /kinetics/Sos/dephosph_Sos SUBSTRATE n 
addmsg /kinetics/Sos/Sos /kinetics/Sos/dephosph_Sos PRODUCT n 
addmsg /kinetics/Sos/Grb2 /kinetics/Sos/Grb2_bind_Sos SUBSTRATE n 
addmsg /kinetics/Sos/Sos.Grb2 /kinetics/Sos/Grb2_bind_Sos PRODUCT n 
addmsg /kinetics/Sos/Sos /kinetics/Sos/Grb2_bind_Sos SUBSTRATE n 
addmsg /kinetics/Sos/Grb2_bind_Sos /kinetics/Sos/Sos REAC A B 
addmsg /kinetics/MAPK*/phosph_Sos /kinetics/Sos/Sos REAC sA B 
addmsg /kinetics/Sos/dephosph_Sos /kinetics/Sos/Sos REAC B A 
addmsg /kinetics/PLC_g/Ca_act_PLC_g /kinetics/PLC_g/PLC_g REAC A B 
addmsg /kinetics/PLC_g/PLC_g /kinetics/PLC_g/Ca_act_PLC_g SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/PLC_g/Ca_act_PLC_g SUBSTRATE n 
addmsg /kinetics/PLC_g/Ca.PLC_g /kinetics/PLC_g/Ca_act_PLC_g PRODUCT n 
addmsg /kinetics/Ca /kinetics/PLC_g/Ca_act_PLC_g* SUBSTRATE n 
addmsg /kinetics/PLC_g/PLC_G* /kinetics/PLC_g/Ca_act_PLC_g* SUBSTRATE n 
addmsg /kinetics/PLC_g/Ca.PLC_g* /kinetics/PLC_g/Ca_act_PLC_g* PRODUCT n 
addmsg /kinetics/PLC_g/Ca.PLC_g* /kinetics/PLC_g/dephosph_PLC_g SUBSTRATE n 
addmsg /kinetics/PLC_g/Ca.PLC_g /kinetics/PLC_g/dephosph_PLC_g PRODUCT n 
addmsg /kinetics/PLC_g/Ca_act_PLC_g* /kinetics/PLC_g/PLC_G* REAC A B 
addmsg /kinetics/PLC_g/Ca_act_PLC_g /kinetics/PLC_g/Ca.PLC_g REAC B A 
addmsg /kinetics/EGFR/L.EGFR/phosph_PLC_g /kinetics/PLC_g/Ca.PLC_g REAC sA B 
addmsg /kinetics/PLC_g/dephosph_PLC_g /kinetics/PLC_g/Ca.PLC_g REAC B A 
addmsg /kinetics/PLC_g/Ca.PLC_g /kinetics/PLC_g/Ca.PLC_g/PIP2_hydrolysis ENZYME n 
addmsg /kinetics/PIP2 /kinetics/PLC_g/Ca.PLC_g/PIP2_hydrolysis SUBSTRATE n 
addmsg /kinetics/PLC_g/Ca_act_PLC_g* /kinetics/PLC_g/Ca.PLC_g* REAC B A 
addmsg /kinetics/EGFR/L.EGFR/phosph_PLC_g /kinetics/PLC_g/Ca.PLC_g* MM_PRD pA 
addmsg /kinetics/PLC_g/dephosph_PLC_g /kinetics/PLC_g/Ca.PLC_g* REAC A B 
addmsg /kinetics/PLC_g/Ca.PLC_g* /kinetics/PLC_g/Ca.PLC_g*/PIP2_hydrolysis ENZYME n 
addmsg /kinetics/PIP2 /kinetics/PLC_g/Ca.PLC_g*/PIP2_hydrolysis SUBSTRATE n 
addmsg /kinetics/CaMKII/CaMKII-bind-CaM /kinetics/CaMKII/CaMKII REAC A B 
addmsg /kinetics/PP1-active/Deph-thr306 /kinetics/CaMKII/CaMKII MM_PRD pA 
addmsg /kinetics/PP1-active/Deph_thr286b /kinetics/CaMKII/CaMKII MM_PRD pA 
addmsg /kinetics/CaMKII/basal-activity /kinetics/CaMKII/CaMKII REAC A B 
addmsg /kinetics/CaMKII/CaMKII-bind-CaM /kinetics/CaMKII/CaMKII-CaM REAC B A 
addmsg /kinetics/PP1-active/Deph-thr286 /kinetics/CaMKII/CaMKII-CaM MM_PRD pA 
addmsg /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_286 /kinetics/CaMKII/CaMKII-CaM REAC sA B 
addmsg /kinetics/CaMKII/tot_autonomous_CaMKII/auton_286 /kinetics/CaMKII/CaMKII-CaM REAC sA B 
addmsg /kinetics/CaMKII/CaMK-thr286-bind-CaM /kinetics/CaMKII/CaMKII-thr286*-CaM REAC B A 
addmsg /kinetics/PP1-active/Deph-thr286 /kinetics/CaMKII/CaMKII-thr286*-CaM REAC sA B 
addmsg /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_286 /kinetics/CaMKII/CaMKII-thr286*-CaM MM_PRD pA 
addmsg /kinetics/CaMKII/tot_autonomous_CaMKII/auton_286 /kinetics/CaMKII/CaMKII-thr286*-CaM MM_PRD pA 
addmsg /kinetics/PP1-active/Deph-thr305 /kinetics/CaMKII/CaMKII*** REAC sA B 
addmsg /kinetics/PP1-active/Deph-thr286c /kinetics/CaMKII/CaMKII*** REAC sA B 
addmsg /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_305 /kinetics/CaMKII/CaMKII*** MM_PRD pA 
addmsg /kinetics/CaMKII/tot_autonomous_CaMKII/auton_305 /kinetics/CaMKII/CaMKII*** MM_PRD pA 
addmsg /kinetics/CaM-Ca4 /kinetics/CaMKII/CaMKII-bind-CaM SUBSTRATE n 
addmsg /kinetics/CaMKII/CaMKII /kinetics/CaMKII/CaMKII-bind-CaM SUBSTRATE n 
addmsg /kinetics/CaMKII/CaMKII-CaM /kinetics/CaMKII/CaMKII-bind-CaM PRODUCT n 
addmsg /kinetics/CaMKII/CaMKII-thr286 /kinetics/CaMKII/CaMK-thr286-bind-CaM SUBSTRATE n 
addmsg /kinetics/CaM-Ca4 /kinetics/CaMKII/CaMK-thr286-bind-CaM SUBSTRATE n 
addmsg /kinetics/CaMKII/CaMKII-thr286*-CaM /kinetics/CaMKII/CaMK-thr286-bind-CaM PRODUCT n 
addmsg /kinetics/CaMKII/CaMK-thr286-bind-CaM /kinetics/CaMKII/CaMKII-thr286 REAC A B 
addmsg /kinetics/PP1-active/Deph-thr305 /kinetics/CaMKII/CaMKII-thr286 MM_PRD pA 
addmsg /kinetics/PP1-active/Deph_thr286b /kinetics/CaMKII/CaMKII-thr286 REAC sA B 
addmsg /kinetics/CaMKII/basal-activity /kinetics/CaMKII/CaMKII-thr286 REAC B A 
addmsg /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_305 /kinetics/CaMKII/CaMKII-thr286 REAC sA B 
addmsg /kinetics/CaMKII/tot_autonomous_CaMKII/auton_305 /kinetics/CaMKII/CaMKII-thr286 REAC sA B 
addmsg /kinetics/PP1-active/Deph-thr306 /kinetics/CaMKII/CaMK-thr306 REAC sA B 
addmsg /kinetics/PP1-active/Deph-thr286c /kinetics/CaMKII/CaMK-thr306 MM_PRD pA 
addmsg /kinetics/CaMKII/CaMKII /kinetics/CaMKII/basal-activity SUBSTRATE n 
addmsg /kinetics/CaMKII/CaMKII-thr286 /kinetics/CaMKII/basal-activity PRODUCT n 
addmsg /kinetics/CaMKII/CaMKII-CaM /kinetics/CaMKII/tot_CaM_CaMKII SUMTOTAL n nInit 
addmsg /kinetics/CaMKII/CaMKII-thr286*-CaM /kinetics/CaMKII/tot_CaM_CaMKII SUMTOTAL n nInit 
addmsg /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_305 /kinetics/CaMKII/tot_CaM_CaMKII REAC eA B 
addmsg /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_286 /kinetics/CaMKII/tot_CaM_CaMKII REAC eA B 
addmsg /kinetics/CaMKII/tot_CaM_CaMKII /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_305 ENZYME n 
addmsg /kinetics/CaMKII/CaMKII-thr286 /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_305 SUBSTRATE n 
addmsg /kinetics/CaMKII/tot_CaM_CaMKII /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_286 ENZYME n 
addmsg /kinetics/CaMKII/CaMKII-CaM /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_286 SUBSTRATE n 
addmsg /kinetics/CaMKII/CaMKII-thr286 /kinetics/CaMKII/tot_autonomous_CaMKII SUMTOTAL n nInit 
addmsg /kinetics/CaMKII/CaMKII*** /kinetics/CaMKII/tot_autonomous_CaMKII SUMTOTAL n nInit 
addmsg /kinetics/CaMKII/tot_autonomous_CaMKII/auton_305 /kinetics/CaMKII/tot_autonomous_CaMKII REAC eA B 
addmsg /kinetics/CaMKII/tot_autonomous_CaMKII/auton_286 /kinetics/CaMKII/tot_autonomous_CaMKII REAC eA B 
addmsg /kinetics/CaMKII/tot_autonomous_CaMKII /kinetics/CaMKII/tot_autonomous_CaMKII/auton_305 ENZYME n 
addmsg /kinetics/CaMKII/CaMKII-thr286 /kinetics/CaMKII/tot_autonomous_CaMKII/auton_305 SUBSTRATE n 
addmsg /kinetics/CaMKII/tot_autonomous_CaMKII /kinetics/CaMKII/tot_autonomous_CaMKII/auton_286 ENZYME n 
addmsg /kinetics/CaMKII/CaMKII-CaM /kinetics/CaMKII/tot_autonomous_CaMKII/auton_286 SUBSTRATE n 
addmsg /kinetics/PP1-active/Deph-thr286 /kinetics/PP1-active REAC eA B 
addmsg /kinetics/PP1-active/Deph-thr305 /kinetics/PP1-active REAC eA B 
addmsg /kinetics/PP1-active/Deph-thr306 /kinetics/PP1-active REAC eA B 
addmsg /kinetics/PP1-active/Deph_thr286b /kinetics/PP1-active REAC eA B 
addmsg /kinetics/PP1-active/Deph-thr286c /kinetics/PP1-active REAC eA B 
addmsg /kinetics/PP1/Inact-PP1 /kinetics/PP1-active REAC A B 
addmsg /kinetics/PP1/dissoc-PP1-I1 /kinetics/PP1-active REAC B A 
addmsg /kinetics/PP1-active /kinetics/PP1-active/Deph-thr286 ENZYME n 
addmsg /kinetics/CaMKII/CaMKII-thr286*-CaM /kinetics/PP1-active/Deph-thr286 SUBSTRATE n 
addmsg /kinetics/PP1-active /kinetics/PP1-active/Deph-thr305 ENZYME n 
addmsg /kinetics/CaMKII/CaMKII*** /kinetics/PP1-active/Deph-thr305 SUBSTRATE n 
addmsg /kinetics/PP1-active /kinetics/PP1-active/Deph-thr306 ENZYME n 
addmsg /kinetics/CaMKII/CaMK-thr306 /kinetics/PP1-active/Deph-thr306 SUBSTRATE n 
addmsg /kinetics/PP1-active /kinetics/PP1-active/Deph-thr286c ENZYME n 
addmsg /kinetics/CaMKII/CaMKII*** /kinetics/PP1-active/Deph-thr286c SUBSTRATE n 
addmsg /kinetics/PP1-active /kinetics/PP1-active/Deph_thr286b ENZYME n 
addmsg /kinetics/CaMKII/CaMKII-thr286 /kinetics/PP1-active/Deph_thr286b SUBSTRATE n 
addmsg /kinetics/CaM/CaM-TR2-bind-Ca /kinetics/CaM/CaM REAC A B 
addmsg /kinetics/CaM/CaM /kinetics/CaM/CaM-TR2-bind-Ca SUBSTRATE n 
addmsg /kinetics/CaM/CaM-TR2-Ca2 /kinetics/CaM/CaM-TR2-bind-Ca PRODUCT n 
addmsg /kinetics/Ca /kinetics/CaM/CaM-TR2-bind-Ca SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/CaM/CaM-TR2-bind-Ca SUBSTRATE n 
addmsg /kinetics/CaM/CaM-TR2-Ca2 /kinetics/CaM/CaM-TR2-Ca2-bind-Ca SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/CaM/CaM-TR2-Ca2-bind-Ca SUBSTRATE n 
addmsg /kinetics/CaM/CaM-Ca3 /kinetics/CaM/CaM-TR2-Ca2-bind-Ca PRODUCT n 
addmsg /kinetics/CaM/CaM-Ca3 /kinetics/CaM/CaM-Ca3-bind-Ca SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/CaM/CaM-Ca3-bind-Ca SUBSTRATE n 
addmsg /kinetics/CaM-Ca4 /kinetics/CaM/CaM-Ca3-bind-Ca PRODUCT n 
addmsg /kinetics/CaM/CaM-TR2-bind-Ca /kinetics/CaM/CaM-TR2-Ca2 REAC B A 
addmsg /kinetics/CaM/CaM-TR2-Ca2-bind-Ca /kinetics/CaM/CaM-TR2-Ca2 REAC A B 
addmsg /kinetics/CaM/CaM-TR2-Ca2-bind-Ca /kinetics/CaM/CaM-Ca3 REAC B A 
addmsg /kinetics/CaM/CaM-Ca3-bind-Ca /kinetics/CaM/CaM-Ca3 REAC A B 
addmsg /kinetics/PKA-active/PKA-phosph-I1 /kinetics/PP1/I1 REAC sA B 
addmsg /kinetics/CaNAB-Ca4/dephosph_inhib1_noCaM /kinetics/PP1/I1 MM_PRD pA 
addmsg /kinetics/CaM_Ca_n-CaNAB/dephosph_inhib1 /kinetics/PP1/I1 MM_PRD pA 
addmsg /kinetics/PP2A/PP2A-dephosph-I1 /kinetics/PP1/I1 MM_PRD pA 
addmsg /kinetics/PP1/dissoc-PP1-I1 /kinetics/PP1/I1 REAC B A 
addmsg /kinetics/PP1/Inact-PP1 /kinetics/PP1/I1* REAC A B 
addmsg /kinetics/PKA-active/PKA-phosph-I1 /kinetics/PP1/I1* MM_PRD pA 
addmsg /kinetics/CaNAB-Ca4/dephosph_inhib1_noCaM /kinetics/PP1/I1* REAC sA B 
addmsg /kinetics/CaM_Ca_n-CaNAB/dephosph_inhib1 /kinetics/PP1/I1* REAC sA B 
addmsg /kinetics/PP2A/PP2A-dephosph-I1 /kinetics/PP1/I1* REAC sA B 
addmsg /kinetics/PP1/PP1-I1* /kinetics/PP1/Inact-PP1 PRODUCT n 
addmsg /kinetics/PP1/I1* /kinetics/PP1/Inact-PP1 SUBSTRATE n 
addmsg /kinetics/PP1-active /kinetics/PP1/Inact-PP1 SUBSTRATE n 
addmsg /kinetics/PP1/Inact-PP1 /kinetics/PP1/PP1-I1* REAC B A 
addmsg /kinetics/PP2A/PP2A-dephosph-PP1-I* /kinetics/PP1/PP1-I1* REAC sA B 
addmsg /kinetics/CaM_Ca_n-CaNAB/dephosph-PP1-I* /kinetics/PP1/PP1-I1* REAC sA B 
addmsg /kinetics/PP1/dissoc-PP1-I1 /kinetics/PP1/PP1-I1 REAC A B 
addmsg /kinetics/PP2A/PP2A-dephosph-PP1-I* /kinetics/PP1/PP1-I1 MM_PRD pA 
addmsg /kinetics/CaM_Ca_n-CaNAB/dephosph-PP1-I* /kinetics/PP1/PP1-I1 MM_PRD pA 
addmsg /kinetics/PP1/PP1-I1 /kinetics/PP1/dissoc-PP1-I1 SUBSTRATE n 
addmsg /kinetics/PP1-active /kinetics/PP1/dissoc-PP1-I1 PRODUCT n 
addmsg /kinetics/PP1/I1 /kinetics/PP1/dissoc-PP1-I1 PRODUCT n 
addmsg /kinetics/PP2A/PP2A-dephosph-I1 /kinetics/PP2A REAC eA B 
addmsg /kinetics/PP2A/PP2A-dephosph-PP1-I* /kinetics/PP2A REAC eA B 
addmsg /kinetics/PP2A /kinetics/PP2A/PP2A-dephosph-I1 ENZYME n 
addmsg /kinetics/PP1/I1* /kinetics/PP2A/PP2A-dephosph-I1 SUBSTRATE n 
addmsg /kinetics/PP2A /kinetics/PP2A/PP2A-dephosph-PP1-I* ENZYME n 
addmsg /kinetics/PP1/PP1-I1* /kinetics/PP2A/PP2A-dephosph-PP1-I* SUBSTRATE n 
addmsg /kinetics/CaNAB-Ca4/dephosph_inhib1_noCaM /kinetics/CaNAB-Ca4 REAC eA B 
addmsg /kinetics/PP2B/Ca-bind-CaNAB-Ca2 /kinetics/CaNAB-Ca4 REAC B A 
addmsg /kinetics/PP2B/CaMCa4-bind-CaNAB /kinetics/CaNAB-Ca4 REAC A B 
addmsg /kinetics/CaNAB-Ca4 /kinetics/CaNAB-Ca4/dephosph_inhib1_noCaM ENZYME n 
addmsg /kinetics/PP1/I1* /kinetics/CaNAB-Ca4/dephosph_inhib1_noCaM SUBSTRATE n 
addmsg /kinetics/PP2B/Ca-bind-CaNAB /kinetics/PP2B/CaNAB REAC A B 
addmsg /kinetics/PP2B/Ca-bind-CaNAB /kinetics/PP2B/CaNAB-Ca2 REAC B A 
addmsg /kinetics/PP2B/Ca-bind-CaNAB-Ca2 /kinetics/PP2B/CaNAB-Ca2 REAC A B 
addmsg /kinetics/CaNAB-Ca4 /kinetics/PP2B/Ca-bind-CaNAB-Ca2 PRODUCT n 
addmsg /kinetics/Ca /kinetics/PP2B/Ca-bind-CaNAB-Ca2 SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/PP2B/Ca-bind-CaNAB-Ca2 SUBSTRATE n 
addmsg /kinetics/PP2B/CaNAB-Ca2 /kinetics/PP2B/Ca-bind-CaNAB-Ca2 SUBSTRATE n 
addmsg /kinetics/PP2B/CaNAB /kinetics/PP2B/Ca-bind-CaNAB SUBSTRATE n 
addmsg /kinetics/PP2B/CaNAB-Ca2 /kinetics/PP2B/Ca-bind-CaNAB PRODUCT n 
addmsg /kinetics/Ca /kinetics/PP2B/Ca-bind-CaNAB SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/PP2B/Ca-bind-CaNAB SUBSTRATE n 
addmsg /kinetics/CaM-Ca4 /kinetics/PP2B/CaMCa4-bind-CaNAB SUBSTRATE n 
addmsg /kinetics/CaNAB-Ca4 /kinetics/PP2B/CaMCa4-bind-CaNAB SUBSTRATE n 
addmsg /kinetics/CaM_Ca_n-CaNAB /kinetics/PP2B/CaMCa4-bind-CaNAB PRODUCT n 
addmsg /kinetics/PKA/cAMP-bind-site-B1 /kinetics/PKA/R2C2 REAC A B 
addmsg /kinetics/PKA/cAMP-bind-site-B1 /kinetics/PKA/R2C2-cAMP REAC B A 
addmsg /kinetics/PKA/cAMP-bind-site-B2 /kinetics/PKA/R2C2-cAMP REAC A B 
addmsg /kinetics/PKA/R2C2 /kinetics/PKA/cAMP-bind-site-B1 SUBSTRATE n 
addmsg /kinetics/PKA/R2C2-cAMP /kinetics/PKA/cAMP-bind-site-B1 PRODUCT n 
addmsg /kinetics/cAMP /kinetics/PKA/cAMP-bind-site-B1 SUBSTRATE n 
addmsg /kinetics/PKA/R2C2-cAMP /kinetics/PKA/cAMP-bind-site-B2 SUBSTRATE n 
addmsg /kinetics/cAMP /kinetics/PKA/cAMP-bind-site-B2 SUBSTRATE n 
addmsg /kinetics/PKA/R2C2-cAMP2 /kinetics/PKA/cAMP-bind-site-B2 PRODUCT n 
addmsg /kinetics/PKA/R2C2-cAMP2 /kinetics/PKA/cAMP-bind-site-A1 SUBSTRATE n 
addmsg /kinetics/cAMP /kinetics/PKA/cAMP-bind-site-A1 SUBSTRATE n 
addmsg /kinetics/PKA/R2C2-cAMP3 /kinetics/PKA/cAMP-bind-site-A1 PRODUCT n 
addmsg /kinetics/cAMP /kinetics/PKA/cAMP-bind-site-A2 SUBSTRATE n 
addmsg /kinetics/PKA/R2C2-cAMP3 /kinetics/PKA/cAMP-bind-site-A2 SUBSTRATE n 
addmsg /kinetics/PKA/R2C2-cAMP4 /kinetics/PKA/cAMP-bind-site-A2 PRODUCT n 
addmsg /kinetics/PKA/cAMP-bind-site-B2 /kinetics/PKA/R2C2-cAMP2 REAC B A 
addmsg /kinetics/PKA/cAMP-bind-site-A1 /kinetics/PKA/R2C2-cAMP2 REAC A B 
addmsg /kinetics/PKA/cAMP-bind-site-A1 /kinetics/PKA/R2C2-cAMP3 REAC B A 
addmsg /kinetics/PKA/cAMP-bind-site-A2 /kinetics/PKA/R2C2-cAMP3 REAC A B 
addmsg /kinetics/PKA/cAMP-bind-site-A2 /kinetics/PKA/R2C2-cAMP4 REAC B A 
addmsg /kinetics/PKA/Release-C1 /kinetics/PKA/R2C2-cAMP4 REAC A B 
addmsg /kinetics/PKA/Release-C1 /kinetics/PKA/R2C-cAMP4 REAC B A 
addmsg /kinetics/PKA/Release-C2 /kinetics/PKA/R2C-cAMP4 REAC A B 
addmsg /kinetics/PKA/Release-C2 /kinetics/PKA/R2-cAMP4 REAC B A 
addmsg /kinetics/PKA/R2C2-cAMP4 /kinetics/PKA/Release-C1 SUBSTRATE n 
addmsg /kinetics/PKA-active /kinetics/PKA/Release-C1 PRODUCT n 
addmsg /kinetics/PKA/R2C-cAMP4 /kinetics/PKA/Release-C1 PRODUCT n 
addmsg /kinetics/PKA/R2C-cAMP4 /kinetics/PKA/Release-C2 SUBSTRATE n 
addmsg /kinetics/PKA-active /kinetics/PKA/Release-C2 PRODUCT n 
addmsg /kinetics/PKA/R2-cAMP4 /kinetics/PKA/Release-C2 PRODUCT n 
addmsg /kinetics/PKA/inhib-PKA /kinetics/PKA/PKA-inhibitor REAC A B 
addmsg /kinetics/PKA-active /kinetics/PKA/inhib-PKA SUBSTRATE n 
addmsg /kinetics/PKA/PKA-inhibitor /kinetics/PKA/inhib-PKA SUBSTRATE n 
addmsg /kinetics/PKA/inhibited-PKA /kinetics/PKA/inhib-PKA PRODUCT n 
addmsg /kinetics/PKA/inhib-PKA /kinetics/PKA/inhibited-PKA REAC B A 
addmsg /kinetics/PKA/cAMP-bind-site-B1 /kinetics/cAMP REAC A B 
addmsg /kinetics/PKA/cAMP-bind-site-B2 /kinetics/cAMP REAC A B 
addmsg /kinetics/PKA/cAMP-bind-site-A1 /kinetics/cAMP REAC A B 
addmsg /kinetics/PKA/cAMP-bind-site-A2 /kinetics/cAMP REAC A B 
addmsg /kinetics/AC/AC2*/kenz /kinetics/cAMP MM_PRD pA 
addmsg /kinetics/AC/AC2-Gs/kenz /kinetics/cAMP MM_PRD pA 
addmsg /kinetics/AC/AC1-CaM/kenz /kinetics/cAMP MM_PRD pA 
addmsg /kinetics/AC/AC1-Gs/kenz /kinetics/cAMP MM_PRD pA 
addmsg /kinetics/AC/AC2*-Gs/kenz /kinetics/cAMP MM_PRD pA 
addmsg /kinetics/AC/cAMP-PDE*/PDE* /kinetics/cAMP REAC sA B 
addmsg /kinetics/AC/cAMP-PDE/PDE /kinetics/cAMP REAC sA B 
addmsg /kinetics/AC/PDE1/PDE1 /kinetics/cAMP REAC sA B 
addmsg /kinetics/AC/CaM.PDE1/CaM.PDE1 /kinetics/cAMP REAC sA B 
addmsg /kinetics/AC/AC2*/kenz /kinetics/AC/ATP REAC sA B 
addmsg /kinetics/AC/AC2-Gs/kenz /kinetics/AC/ATP REAC sA B 
addmsg /kinetics/AC/AC1-CaM/kenz /kinetics/AC/ATP REAC sA B 
addmsg /kinetics/AC/AC1-Gs/kenz /kinetics/AC/ATP REAC sA B 
addmsg /kinetics/AC/AC2*-Gs/kenz /kinetics/AC/ATP REAC sA B 
addmsg /kinetics/AC/CaM-bind-AC1 /kinetics/AC/AC1-CaM REAC B A 
addmsg /kinetics/AC/AC1-CaM /kinetics/AC/AC1-CaM/kenz ENZYME n 
addmsg /kinetics/AC/ATP /kinetics/AC/AC1-CaM/kenz SUBSTRATE n 
addmsg /kinetics/AC/CaM-bind-AC1 /kinetics/AC/AC1 REAC A B 
addmsg /kinetics/AC/Gs-bind-AC1 /kinetics/AC/AC1 REAC A B 
addmsg /kinetics/CaM-Ca4 /kinetics/AC/CaM-bind-AC1 SUBSTRATE n 
addmsg /kinetics/AC/AC1-CaM /kinetics/AC/CaM-bind-AC1 PRODUCT n 
addmsg /kinetics/AC/AC1 /kinetics/AC/CaM-bind-AC1 SUBSTRATE n 
addmsg /kinetics/PKC-active/phosph-AC2 /kinetics/AC/AC2* MM_PRD pA 
addmsg /kinetics/AC/dephosph-AC2 /kinetics/AC/AC2* REAC A B 
addmsg /kinetics/AC/Gs-bind-AC2* /kinetics/AC/AC2* REAC A B 
addmsg /kinetics/AC/AC2* /kinetics/AC/AC2*/kenz ENZYME n 
addmsg /kinetics/AC/ATP /kinetics/AC/AC2*/kenz SUBSTRATE n 
addmsg /kinetics/AC/Gs-bind-AC2 /kinetics/AC/AC2-Gs REAC B A 
addmsg /kinetics/AC/AC2-Gs /kinetics/AC/AC2-Gs/kenz ENZYME n 
addmsg /kinetics/AC/ATP /kinetics/AC/AC2-Gs/kenz SUBSTRATE n 
addmsg /kinetics/AC/Gs-bind-AC2 /kinetics/AC/AC2 REAC A B 
addmsg /kinetics/PKC-active/phosph-AC2 /kinetics/AC/AC2 REAC sA B 
addmsg /kinetics/AC/dephosph-AC2 /kinetics/AC/AC2 REAC B A 
addmsg /kinetics/AC/AC2* /kinetics/AC/dephosph-AC2 SUBSTRATE n 
addmsg /kinetics/AC/AC2 /kinetics/AC/dephosph-AC2 PRODUCT n 
addmsg /kinetics/AC/Gs-bind-AC1 /kinetics/AC/AC1-Gs REAC B A 
addmsg /kinetics/AC/AC1-Gs /kinetics/AC/AC1-Gs/kenz ENZYME n 
addmsg /kinetics/AC/ATP /kinetics/AC/AC1-Gs/kenz SUBSTRATE n 
addmsg /kinetics/AC/AC2 /kinetics/AC/Gs-bind-AC2 SUBSTRATE n 
addmsg /kinetics/AC/AC2-Gs /kinetics/AC/Gs-bind-AC2 PRODUCT n 
addmsg /kinetics/Gs-alpha /kinetics/AC/Gs-bind-AC2 SUBSTRATE n 
addmsg /kinetics/Gs-alpha /kinetics/AC/Gs-bind-AC1 SUBSTRATE n 
addmsg /kinetics/AC/AC1 /kinetics/AC/Gs-bind-AC1 SUBSTRATE n 
addmsg /kinetics/AC/AC1-Gs /kinetics/AC/Gs-bind-AC1 PRODUCT n 
addmsg /kinetics/AC/cAMP-PDE*/PDE* /kinetics/AC/AMP MM_PRD pA 
addmsg /kinetics/AC/cAMP-PDE/PDE /kinetics/AC/AMP MM_PRD pA 
addmsg /kinetics/AC/CaM.PDE1/CaM.PDE1 /kinetics/AC/AMP MM_PRD pA 
addmsg /kinetics/AC/PDE1/PDE1 /kinetics/AC/AMP MM_PRD pA 
addmsg /kinetics/AC/Gs-bind-AC2* /kinetics/AC/AC2*-Gs REAC B A 
addmsg /kinetics/AC/AC2*-Gs /kinetics/AC/AC2*-Gs/kenz ENZYME n 
addmsg /kinetics/AC/ATP /kinetics/AC/AC2*-Gs/kenz SUBSTRATE n 
addmsg /kinetics/Gs-alpha /kinetics/AC/Gs-bind-AC2* SUBSTRATE n 
addmsg /kinetics/AC/AC2*-Gs /kinetics/AC/Gs-bind-AC2* PRODUCT n 
addmsg /kinetics/AC/AC2* /kinetics/AC/Gs-bind-AC2* SUBSTRATE n 
addmsg /kinetics/AC/cAMP-PDE/PDE /kinetics/AC/cAMP-PDE REAC eA B 
addmsg /kinetics/AC/dephosph-PDE /kinetics/AC/cAMP-PDE REAC B A 
addmsg /kinetics/PKA-active/phosph-PDE /kinetics/AC/cAMP-PDE REAC sA B 
addmsg /kinetics/AC/cAMP-PDE /kinetics/AC/cAMP-PDE/PDE ENZYME n 
addmsg /kinetics/cAMP /kinetics/AC/cAMP-PDE/PDE SUBSTRATE n 
addmsg /kinetics/AC/cAMP-PDE*/PDE* /kinetics/AC/cAMP-PDE* REAC eA B 
addmsg /kinetics/AC/dephosph-PDE /kinetics/AC/cAMP-PDE* REAC A B 
addmsg /kinetics/PKA-active/phosph-PDE /kinetics/AC/cAMP-PDE* MM_PRD pA 
addmsg /kinetics/AC/cAMP-PDE* /kinetics/AC/cAMP-PDE*/PDE* ENZYME n 
addmsg /kinetics/cAMP /kinetics/AC/cAMP-PDE*/PDE* SUBSTRATE n 
addmsg /kinetics/AC/cAMP-PDE* /kinetics/AC/dephosph-PDE SUBSTRATE n 
addmsg /kinetics/AC/cAMP-PDE /kinetics/AC/dephosph-PDE PRODUCT n 
addmsg /kinetics/AC/PDE1/PDE1 /kinetics/AC/PDE1 REAC eA B 
addmsg /kinetics/AC/CaM_bind_PDE1 /kinetics/AC/PDE1 REAC A B 
addmsg /kinetics/AC/PDE1 /kinetics/AC/PDE1/PDE1 ENZYME n 
addmsg /kinetics/cAMP /kinetics/AC/PDE1/PDE1 SUBSTRATE n 
addmsg /kinetics/AC/CaM.PDE1/CaM.PDE1 /kinetics/AC/CaM.PDE1 REAC eA B 
addmsg /kinetics/AC/CaM_bind_PDE1 /kinetics/AC/CaM.PDE1 REAC B A 
addmsg /kinetics/AC/CaM.PDE1 /kinetics/AC/CaM.PDE1/CaM.PDE1 ENZYME n 
addmsg /kinetics/cAMP /kinetics/AC/CaM.PDE1/CaM.PDE1 SUBSTRATE n 
addmsg /kinetics/AC/PDE1 /kinetics/AC/CaM_bind_PDE1 SUBSTRATE n 
addmsg /kinetics/AC/CaM.PDE1 /kinetics/AC/CaM_bind_PDE1 PRODUCT n 
addmsg /kinetics/CaM-Ca4 /kinetics/AC/CaM_bind_PDE1 SUBSTRATE n 
addmsg /kinetics/AC/Gs-bind-AC2 /kinetics/Gs-alpha REAC A B 
addmsg /kinetics/AC/Gs-bind-AC1 /kinetics/Gs-alpha REAC A B 
addmsg /kinetics/AC/Gs-bind-AC2* /kinetics/Gs-alpha REAC A B 
addmsg /kinetics/PKC/PKC-act-by-Ca /kinetics/Ca REAC A B 
addmsg /kinetics/PLA2/PLA2-Ca-act /kinetics/Ca REAC A B 
addmsg /kinetics/PLA2/PLA2*-Ca-act /kinetics/Ca REAC A B 
addmsg /kinetics/PLCbeta/Act-PLC-Ca /kinetics/Ca REAC A B 
addmsg /kinetics/PLCbeta/PLC-Gq-bind-Ca /kinetics/Ca REAC A B 
addmsg /kinetics/PLC_g/Ca_act_PLC_g /kinetics/Ca REAC A B 
addmsg /kinetics/PLC_g/Ca_act_PLC_g* /kinetics/Ca REAC A B 
addmsg /kinetics/CaM/CaM-TR2-bind-Ca /kinetics/Ca REAC A B 
addmsg /kinetics/CaM/CaM-TR2-bind-Ca /kinetics/Ca REAC A B 
addmsg /kinetics/CaM/CaM-TR2-Ca2-bind-Ca /kinetics/Ca REAC A B 
addmsg /kinetics/CaM/CaM-Ca3-bind-Ca /kinetics/Ca REAC A B 
addmsg /kinetics/PP2B/Ca-bind-CaNAB-Ca2 /kinetics/Ca REAC A B 
addmsg /kinetics/PP2B/Ca-bind-CaNAB-Ca2 /kinetics/Ca REAC A B 
addmsg /kinetics/PP2B/Ca-bind-CaNAB /kinetics/Ca REAC A B 
addmsg /kinetics/PP2B/Ca-bind-CaNAB /kinetics/Ca REAC A B 
addmsg /kinetics/Ca_stoch /kinetics/Ca REAC B A 
addmsg /kinetics/Ca_stoch /kinetics/Ca_input REAC A B 
addmsg /kinetics/Ca_input /kinetics/Ca_stoch SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/Ca_stoch PRODUCT n 
addmsg /kinetics/CaM_Ca_n-CaNAB/dephosph_inhib1 /kinetics/CaM_Ca_n-CaNAB REAC eA B 
addmsg /kinetics/CaM_Ca_n-CaNAB/dephosph-PP1-I* /kinetics/CaM_Ca_n-CaNAB REAC eA B 
addmsg /kinetics/PP2B/CaMCa4-bind-CaNAB /kinetics/CaM_Ca_n-CaNAB REAC B A 
addmsg /kinetics/CaM_Ca_n-CaNAB /kinetics/CaM_Ca_n-CaNAB/dephosph_inhib1 ENZYME n 
addmsg /kinetics/PP1/I1* /kinetics/CaM_Ca_n-CaNAB/dephosph_inhib1 SUBSTRATE n 
addmsg /kinetics/CaM_Ca_n-CaNAB /kinetics/CaM_Ca_n-CaNAB/dephosph-PP1-I* ENZYME n 
addmsg /kinetics/PP1/PP1-I1* /kinetics/CaM_Ca_n-CaNAB/dephosph-PP1-I* SUBSTRATE n 
addmsg /kinetics/MAPK* /graphs/conc1/MAPK*.Co PLOT Co *MAPK*.Co *orange 
addmsg /kinetics/PKC-active /graphs/conc1/PKC-active.Co PLOT Co *PKC-active.Co *red 
addmsg /kinetics/Ras/GTP-Ras /graphs/conc1/GTP-Ras.Co PLOT Co *GTP-Ras.Co *orange 
addmsg /kinetics/CaM-Ca4 /graphs/conc1/CaM-Ca4.Co PLOT Co *CaM-Ca4.Co *blue 
addmsg /kinetics/PKA-active /graphs/conc1/PKA-active.Co PLOT Co *PKA-active.Co *yellow 
addmsg /kinetics/cAMP /graphs/conc1/cAMP.Co PLOT Co *cAMP.Co *green 
addmsg /kinetics/PP1-active /graphs/conc1/PP1-active.Co PLOT Co *PP1-active.Co *cyan 
addmsg /kinetics/MAPK/Raf*-GTP-Ras /graphs/conc1/Raf-GTP-Ras*.Co PLOT Co *Raf-GTP-Ras*.Co *red 
addmsg /kinetics/CaMKII/tot_CaM_CaMKII /graphs/conc2/tot_CaM_CaMKII.Co PLOT Co *tot_CaM_CaMKII.Co *green 
addmsg /kinetics/CaMKII/tot_autonomous_CaMKII /graphs/conc2/tot_autonomous_CaMKII.Co PLOT Co *tot_autonomous_CaMKII.Co *green 
addmsg /kinetics/AA /graphs/conc2/AA.Co PLOT Co *AA.Co *darkgreen 
addmsg /kinetics/DAG /graphs/conc2/DAG.Co PLOT Co *DAG.Co *green 
addmsg /kinetics/Ca /graphs/conc2/Ca.Co PLOT Co *Ca.Co *red 
addmsg /kinetics/AC/AC2* /graphs/conc2/AC2*.Co PLOT Co *AC2*.Co *yellow 
addmsg /kinetics/CaM_Ca_n-CaNAB /graphs/conc2/CaM(Ca)n-CaNAB.Co PLOT Co *CaM(Ca)n-CaNAB.Co *darkblue 
enddump
// End of dump

call /kinetics/PKC/PKC-act-by-Ca/notes LOAD \
"Need est of rate of assoc of Ca and PKC. Assume it is fast" \
"The original parameter-searched kf of 439.4 has been" \
"scaled by 1/6e8 to account for change of units to n. Kf now 8.16e-7, kb=.6085" \
"Raised kf to 1e-6 to match Ca curve, kb to .5" \
""
call /kinetics/PKC/PKC-act-by-DAG/notes LOAD \
"Need est of rate. Assume it is fast" \
"Obtained from param search" \
"kf raised 10 X : see Shinomura et al PNAS 88 5149-5153 1991." \
"kf changed from 3.865e-7 to 2.0e-7 in line with closer analysis of" \
"Shinomura data." \
"26 June 1996: Corrected DAG data: reduce kf 15x from " \
"2e-7 to 1.333e-8"
call /kinetics/PKC/PKC-DAG-to-memb/notes LOAD \
"Raise kb from .087 to 0.1 to match data from Shinomura et al." \
"Lower kf from 1.155 to 1.0 to match data from Shinomura et al."
call /kinetics/PKC/PKC-act-by-Ca-AA/notes LOAD \
"Schaechter and Benowitz We have to increase Kf for conc scaling" \
"Changed kf to 2e-9 on Sept 19, 94. This gives better match." \
""
call /kinetics/PKC/PKC-act-by-DAG-AA/notes LOAD \
"Assume slowish too. Schaechter and Benowitz"
call /kinetics/PKC/PKC-basal-act/notes LOAD \
"Initial basal levels were set by kf = 1, kb = 20. In model, though, the" \
"basal levels of PKC activity are higher."
call /kinetics/PKC/PKC-act-by-AA/notes LOAD \
"Raise kf from 1.667e-10 to 2e-10 to get better match to data."
call /kinetics/PKC/PKC-n-DAG/notes LOAD \
"kf raised 10 X based on Shinomura et al PNAS 88 5149-5153 1991" \
"closer analysis of Shinomura et al: kf now 1e-8 (was 1.66e-8)." \
"Further tweak. To get sufficient AA synergy, increase kf to 1.5e-8" \
"26 June 1996: Corrected DAG levels: reduce kf by 15x from" \
"1.5e-8 to 1e-9"
call /kinetics/PKC/PKC-DAG/notes LOAD \
"CoInit was .0624" \
""
call /kinetics/PKC/PKC-n-DAG-AA/notes LOAD \
"Reduced kf to 0.66X to match Shinomura et al data." \
"Initial: kf = 3.3333e-9" \
"New: 2e-9" \
"Newer: 2e-8" \
"kb was 0.2" \
"now 2."
call /kinetics/PKC/PKC-cytosolic/notes LOAD \
"Marquez et al J. Immun 149,2560(92) est 1e6/cell for chromaffin cells" \
"We will use 1 uM as our initial concen" \
""
call /kinetics/PKC-active/PKC-act-raf/notes LOAD \
"Rate consts from Chen et al Biochem 32, 1032 (1993)" \
"k3 = k2 = 4" \
"k1 = 9e-5" \
"recalculated gives 1.666e-5, which is not very different." \
"Looks like k3 is rate-limiting in this case: there is a huge amount" \
"of craf locked up in the enz complex. Let us assume a 10x" \
"higher Km, ie, lower affinity.  k1 drops by 10x." \
"Also changed k2 to 4x k3." \
"Lowerd k1 to 1e-6 to balance 10X DAG sensitivity of PKC"
call /kinetics/PKC-active/PKC-inact-GAP/notes LOAD \
"Rate consts copied from PCK-act-raf" \
"This reaction inactivates GAP. The idea is from the " \
"Boguski and McCormick review."
call /kinetics/PKC-active/PKC-act-GEF/notes LOAD \
"Rate consts from PKC-act-raf." \
"This reaction activates GEF. It can lead to at least 2X stim of ras, and" \
"a 2X stim of MAPK over and above that obtained via direct phosph of" \
"c-raf. Note that it is a push-pull reaction, and there is also a contribution" \
"through the phosphorylation and inactivation of GAPs." \
"The original PKC-act-raf rate consts are too fast. We lower K1 by 10 X"
call /kinetics/PKC-active/phosph-AC2/notes LOAD \
"Phorbol esters have little effect on AC1 or on the Gs-stimulation of" \
"AC2. So in this model we are only dealing with the increase in" \
"basal activation of AC2 induced by PKC" \
"k1 = 1.66e-6" \
"k2 = 16" \
"k3 =4" \
""
call /kinetics/PLA2/notes LOAD \
"Mail source of data: Leslie and Channon BBA 1045 (1990) pp 261-270." \
"Fig 6 is Ca curve. Fig 4a is PIP2 curve. Fig 4b is DAG curve. Also see" \
"Wijkander and Sundler JBC 202 (1991) pp873-880;" \
"Diez and Mong JBC 265(24) p14654;" \
"Leslie JBC 266(17) (1991) pp11366-11371"
call /kinetics/PLA2/PLA2-cytosolic/notes LOAD \
"Calculated cytosolic was 20 nm from Wijkander and Sundler" \
"However, Leslie and Channon use about 400 nM. Need to confirm," \
"but this is the value I use here. Another recalc of W&S gives 1uM"
call /kinetics/PLA2/PLA2-Ca-act/notes LOAD \
"Leslie and Channon BBA 1045 (1990) 261-270 fig6 pp267."
call /kinetics/PLA2/PLA2-Ca*/kenz/notes LOAD \
"10 x raise oct22" \
"12 x oct 24, set k2 = 4 * k3"
call /kinetics/PLA2/PIP2-PLA2*/kenz/notes LOAD \
"10 X raise oct 22" \
"12 X further raise oct 24 to allow for correct conc of enzyme" \
""
call /kinetics/PLA2/PIP2-Ca-PLA2*/kenz/notes LOAD \
"10 x raise oct 22" \
"12 x and rescale for k2 = 4 * k3 convention oct 24" \
"Increase further to get the match to expt, which was spoilt due" \
"to large accumulation of PLA2 in the enzyme complexed forms." \
"Lets raise k3, leaving the others at " \
"k1 = 1.5e-5 and k2 = 144 since they are large already." \
""
call /kinetics/PLA2/DAG-Ca-PLA2-act/notes LOAD \
"27 June 1996" \
"Scaled kf down by 0.015" \
"from 3.33e-7 to 5e-9" \
"to fit with revised DAG estimates" \
"and use of mole-fraction to calculate eff on PLA2."
call /kinetics/PLA2/DAG-Ca-PLA2*/kenz/notes LOAD \
"10 X raise oct 22" \
"12 X raise oct 24 + conversion to k2 =4 * k3"
call /kinetics/PLA2/APC/notes LOAD \
"arachodonylphosphatidylcholine is the favoured substrate" \
"from Wijkander and Sundler, JBC 202 pp 873-880, 1991." \
"Their assay used 30 uM substrate, which is what the kinetics in" \
"this model are based on. For the later model we should locate" \
"a more realistic value for APC."
call /kinetics/PLA2/Degrade-AA/notes LOAD \
"I need to check if the AA degradation pathway really leads back to " \
"APC. Anyway, it is a convenient buffered pool to dump it back into." \
"For the purposes of the full model we use a rate of degradation of" \
"0.2/sec" \
"Raised decay to 0.4 : see PLA35.g notes for Feb17 "
call /kinetics/PLA2/PLA2*-Ca/notes LOAD \
"Phosphorylated form of PLA2. Still need to hook it up using kinases." \
"PKA: Wightman et al JBC 257 pp6650 1982" \
"PKC: Many refs, eg Gronich et al JBC 263 pp 16645, 1988 but see Lin etal" \
"MAPK: Lin et al, Cell 72 pp 269, 1993.  Show 3x with MAPK but not PKC alone" \
"Do not know if there is a Ca requirement for active phosphorylated state."
call /kinetics/PLA2/PLA2*-Ca/kenz/notes LOAD \
"This form should be 3 to 6 times as fast as the Ca-only form." \
"I have scaled by 4x which seems to give a 5x rise." \
"10x raise Oct 22" \
"12 x oct 24, changed k2 = 4 * k3"
call /kinetics/PLA2/PLA2*-Ca-act/notes LOAD \
"To start off, same kinetics as the PLA2-Ca-act direct pathway." \
"Oops ! Missed out the Ca input to this pathway first time round." \
"Let's raise the forward rate about 3x to 5e-6. This will let us reduce the" \
"rather high rates we have used for the kenz on PLA2*-Ca. In fact, it" \
"may be that the rates are not that different, just that this pathway for" \
"getting the PLA2 to the memb is more efficien...."
call /kinetics/MAPK*/MAPK*/notes LOAD \
"Km = 25uM @ 50 uM ATP and 1mg/ml MBP (huge XS of substrate)" \
"Vmax = 4124 pmol/min/ml at a conc of 125 pmol/ml of enz, so:" \
"k3 = .5/sec (rate limiting)" \
"k1 = (k2  + k3)/Km = (.5 + 0)/(25*6e5) = 2e-8 (#/cell)^-1" \
"#s from Sanghera et al JBC 265 pp 52 , 1990. " \
"From Nemenoff et al JBC 268(3):1960-1964 - using Sanghera's 1e-4 ratio" \
"of MAPK to protein, we get k3 = 7/sec from 1000 pmol/min/mg fig 5"
call /kinetics/MAPK*/MAPK*-feedback/notes LOAD \
"Ueki et al JBC 269(22):15756-15761 show the presence of" \
"this step, but not the rate consts, which are derived from" \
"Sanghera et al  JBC 265(1):52-57, 1990, see the deriv in the" \
"MAPK* notes."
call /kinetics/MAPK*/phosph_Sos/notes LOAD \
"See Porfiri and McCormick JBC 271:10 pp5871 1996 for the" \
"existence of this step. We'll take the rates from the ones" \
"used for the phosph of Raf by MAPK." \
"Sep 17 1997: The transient activation curve matches better" \
"with k1 up  by 10 x."
call /kinetics/temp-PIP2/notes LOAD \
"This isn't explicitly present in the M&L model, but is obviously needed." \
"I assume its conc is fixed at 1uM for now, which is a bit high. PLA2 is stim" \
"7x by PIP2 @ 0.5 uM (Leslie and Channon BBA 1045:261(1990) " \
"Leslie and Channon say PIP2 is present at 0.1 - 0.2mol% range in membs," \
"which comes to 50 nM. Ref is Majerus et al Cell 37 pp 701-703 1984" \
"Lets use a lower level of 30 nM, same ref...."
call /kinetics/IP3/notes LOAD \
"Peak IP3 is perhaps 15 uM, basal <= 0.2 uM."
call /kinetics/Glu/notes LOAD \
"Varying the amount of (steady state) glu between .01 uM and up, the" \
"final amount of G*GTP complex does not change much. This means that" \
"the system should be reasonably robust wr to the amount of glu in the" \
"synaptic cleft. It would be nice to know how fast it is removed."
call /kinetics/PLCbeta/notes LOAD \
"Group for PLC beta"
call /kinetics/PLCbeta/Act-PLC-Ca/notes LOAD \
"Affinity for Ca = 1uM without AlF, 0.1 with:" \
" from Smrcka et al science 251 pp 804-807 1991" \
"so [Ca].kf = kb so kb/kf = 1 * 6e5 = 1/1.66e-6" \
"" \
"11 June 1996: Raised affinity to 5e-6 to maintain" \
"balance. See notes."
call /kinetics/PLCbeta/PLC/notes LOAD \
"Total PLC = 0.8 uM see Ryu et al JBC 262 (26) pp 12511 1987"
call /kinetics/PLCbeta/Degrade-IP3/notes LOAD \
"The enzyme is IP3 5-phosphomonesterase. about 45K. Actual products" \
"are Ins(1,4)P2, and cIns(1:2,4,5)P3.  review in Majerus et al Science 234" \
"1519-1526, 1986." \
"Meyer and Stryer 1988 PNAS 85:5051-5055 est decay of IP3 at" \
" 1-3/sec"
call /kinetics/PLCbeta/Degrade-DAG/notes LOAD \
"These rates are the same as for degrading IP3, but I am sure that they could" \
"be improved." \
"Lets double kf to 0.2, since the amount of DAG in the cell should be <= 1uM." \
"Need to double it again, for the same reason." \
"kf now 0.5" \
"27 June 1996" \
"kf is now 0.02 to get 50 sec time course" \
"30 Aug 1997: Raised kf to 0.11 to accomodate PLC_gamma" \
"27 Mar 1998: kf now 0.15 for PLC_gamma"
call /kinetics/PLCbeta/PC/notes LOAD \
"Phosphatidylcholine is the main (around 55%) metabolic product of DAG," \
"follwed by PE (around 25%). Ref is Welsh and Cabot, JCB35:231-245(1987)"
call /kinetics/PLCbeta/PLC-Ca/PLC-Ca/notes LOAD \
"From Sternweis et al Phil Trans R Soc Lond 1992, also matched by Homma et al." \
"k1 = 1.5e-5, now 4.2e-6" \
"k2 = 70/sec; now 40/sec" \
"k3 = 17.5/sec; now 10/sec" \
"Note that the wording in Sternweis et al is" \
"ambiguous re the Km."
call /kinetics/PLCbeta/Act-PLC-by-Gq/notes LOAD \
"Affinity for Gq is > 20 nM (Smrcka et al Science251 804-807 1991)" \
"so [Gq].kf = kb so 40nM * 6e5 = kb/kf = 24e3 so kf = 4.2e-5, kb =1" \
""
call /kinetics/PLCbeta/Inact-PLC-Gq/notes LOAD \
"This process is assumed to be directly caused by the inactivation of" \
"the G*GTP to G*GDP. Hence, " \
"kf = .013 /sec = 0.8/min, same as the rate for Inact-G." \
"kb = 0 since this is irreversible." \
"We may be" \
"interested in studying the role of PLC as a GAP. If so, the kf would be faster here" \
"than in Inact-G"
call /kinetics/PLCbeta/PLC-Ca-Gq/notes LOAD \
"This should really be labelled PLC-G*GTP-Ca." \
"This is the activated form of the enzyme. Mahama and Linderman assume" \
"that the IP3 precursors are not rate-limiting, but I will include those for" \
"completeness as they may be needed later." \
""
call /kinetics/PLCbeta/PLC-Ca-Gq/PLCb-Ca-Gq/notes LOAD \
"From Sternweis et al, Phil Trans R Soc Lond 1992, and the values from" \
"other refs eg Homma et al JBC 263(14) pp6592 1988 match." \
"k1 = 5e-5/sec" \
"k2 = 240/sec; now 120/sec" \
"k3 = 60/sec; now 30/sec" \
"Note that the wording in Sternweis et al" \
"is ambiguous wr. to the Km for Gq vs non-Gq states of PLC. " \
"K1 is still a bit too low. Raise to 7e-5" \
"9 Jun 1996: k1 was 0.0002, changed to 5e-5"
call /kinetics/PLCbeta/PLC-bind-Gq/notes LOAD \
"this binding does not produce active PLC. This step was needed to" \
"implement the described (Smrcka et al) increase in affinity for Ca" \
"by PLC once Gq was bound." \
"The kinetics are the same as the binding step for Ca-PLC to Gq." \
"" \
"June 1996:" \
"Changed the kf to 4.2e-5 to 4.2e-6 to preserve balance around" \
"the reactions. "
call /kinetics/PLCbeta/PLC-Gq-bind-Ca/notes LOAD \
"this step has a high affinity for Ca, from Smrcka et al. 0.1uM" \
"so kf /kb = 1/6e4 = 1.666e-5:1. See the Act-PLC-by-Gq reac." \
"11 Jun 1996: Raised kf to 5e-5 based on match to conc-eff" \
"curves from Smrcka et al."
call /kinetics/BetaGamma/notes LOAD \
"These exist in a nebulous sense in this model, basically only to balance" \
"the conservation equations. The details of their reassociation with G-GDP" \
"are not modeled" \
"Resting level =0.0094, stim level =.0236 from all42.g ish."
call /kinetics/G*GTP/notes LOAD \
"Activated G protein. Berstein et al indicate that about 20-40% of the total" \
"Gq alpha should bind GTP at steady stim. This sim gives more like 65%."
call /kinetics/Gq/notes LOAD \
"We assume GTP is present in fixed amounts, so we leave it out" \
"of the explicit equations in this model. Normally we would expect it" \
"to associate along with the G-Receptor-ligand complex." \
"Most info is from Berstein et al JBC 267:12 8081-8088 1992" \
"Structure of rec activation of Gq from Fay et al Biochem 30 5066-5075 1991"
call /kinetics/Gq/RecLigandBinding/notes LOAD \
"kf = kf from text = 1e7 / M / sec = 10 /uM/sec = 10 / 6e5 / # / sec  = 1.67e-5" \
"kb = kr from text = 60 / sec" \
"Note that we continue to use uM here since [phenylephrine] is also in uM." \
"From Martin et al FEBS Lett 316:2 191-196 1993 we have Kd = 600 nM" \
"Assuming kb = 10/sec, we get kf = 10/(0.6 uM * 6e5) = 2.8e-5 1/sec/#"
call /kinetics/Gq/G-GDP/notes LOAD \
"From M&L, total Gprot = 1e5molecules/cell" \
"At equil, 92340 are here, 400 are in G*GTP, and another 600 are assoc" \
"with the PLC and 6475 are as G*GDP. This is OK." \
"" \
"From Pang and Sternweis JBC 265:30 18707-12 1990 we get conc est 1.6 uM" \
"to 0.8 uM. A number of other factors are involved too." \
""
call /kinetics/Gq/Basal-Act-G/notes LOAD \
"kf = kg1 = 0.01/sec, kb = 0. This is the basal exchange of GTP for GDP."
call /kinetics/Gq/Trimerize-G/notes LOAD \
"kf == kg3 = 1e-5 /cell/sec. As usual, there is no back-reaction" \
"kb = 0"
call /kinetics/Gq/Inact-G/notes LOAD \
"From Berstein et al JBC 267:12 8081-8088 1992, kcat for GTPase activity" \
"of Gq is only 0.8/min"
call /kinetics/Gq/mGluR/notes LOAD \
"From M&L, Total # of receptors/cell = 1900" \
"Vol of cell = 1e-15 (10 um cube). Navogadro = 6.023e23" \
"so conversion from n to conc in uM is n/vol*nA * 1e3 = 1.66e-6" \
"However, for typical synaptic channels the density is likely to be very" \
"high at the synapse. Use an estimate of 0.1 uM for now. this gives" \
"a total of about 60K receptors/cell, which is in line with Fay et at."
call /kinetics/Gq/Rec-Glu/notes LOAD \
"This acts like an enzyme to activate the g proteins" \
"Assume cell has vol 1e-15 m^3 (10 uM cube), conversion factor to" \
"conc in uM is 6e5" \
""
call /kinetics/Gq/Rec-Gq/notes LOAD \
"Fraction of Rec-Gq is 44%  of rec, from Fay et al." \
"Since this is not the same receptor, this value is a bit doubtful. Still," \
"we adjust the rate consts in Rec-bind-Gq to match."
call /kinetics/Gq/Rec-Glu-bind-Gq/notes LOAD \
"This is the k1-k2 equivalent for enzyme complex formation in the" \
"binding of Rec-Glu to Gq." \
"See Fay et al Biochem 30 5066-5075 1991." \
"kf = 5e-5 which is nearly the same as calculated by Fay et al. (4.67e-5)" \
"kb = .04" \
"" \
"June 1996: Closer reading of Fay et al suggests that " \
"kb <= 0.0001, so kf = 1e-8 by detailed balance. This" \
"reaction appears to be neglible."
call /kinetics/Gq/Glu-bind-Rec-Gq/notes LOAD \
"From Fay et al" \
"kb3 = kb = 1.06e-3 which is rather slow." \
"k+1 = kf = 2.8e7 /M/sec= 4.67e-5/sec use 5e-5." \
"However, the Kd from Martin et al may be more appropriate, as this" \
"is Glu not the system from Fay." \
"kf = 2.8e-5, kb = 10" \
"Let us compromise. since we have the Fay model, keep kf = k+1 = 2.8e-5." \
"But kb (k-3) is .01 * k-1 from Fay. Scaling by .01, kb = .01 * 10 = 0.1"
call /kinetics/Gq/Activate-Gq/notes LOAD \
"This is the kcat==k3 stage of the Rec-Glu ezymatic activation of Gq." \
"From Berstein et al actiation is at .35 - 0.7/min" \
"From Fay et al Biochem 30 5066-5075 1991 kf = .01/sec" \
"From Nakamura et al J physiol Lond 474:1 35-41 1994 see time courses." \
"Also (Berstein) 15-40% of gprot is in GTP-bound form on stim."
call /kinetics/Gq/Rec-bind-Gq/notes LOAD \
"Lets try out the same kinetics as the Rec-Glu-bind-Gq" \
"This is much too forward. We know that the steady-state" \
"amount of Rec-Gq should be 40% of the total amount of receptor." \
"This is for a different receptor, still we can try to match the value." \
"kf = 1e-6 and kb = 1 give 0.333:0.8 which is pretty close." \
""
call /kinetics/Gq/mGluRAntag/notes LOAD \
"I am implementing this as acting only on the Rec-Gq complex, based on" \
"a more complete model PLC_Gq48.g" \
"which showed that the binding to the rec alone contributed only a small amount."
call /kinetics/Gq/Antag-bind-Rec-Gq/notes LOAD \
"The rate consts give a total binding affinity of only "
call /kinetics/MAPK/craf-1/notes LOAD \
"Couldn't find any ref to the actual conc of craf-1 but I" \
"should try Strom et al Oncogene 5 pp 345" \
"In line with the other kinases in the cascade, I estimate the conc to be" \
"0.2 uM. To init we use 0.15, which is close to equil"
call /kinetics/MAPK/MAPKK/notes LOAD \
"Conc is from Seger et al JBC 267:20 pp14373 (1992)" \
"mwt is 45/46 Kd" \
"We assume that phosphorylation on both ser and thr is needed for" \
"activiation. See Kyriakis et al Nature 358 417 1992" \
"Init conc of total is 0.18" \
""
call /kinetics/MAPK/MAPK/notes LOAD \
"conc is from Sanghera et al JBC 265 pp 52 (1990)" \
"A second calculation gives 3.1 uM, from same paper." \
"They est MAPK is 1e-4x total protein, and protein is 15% of cell wt," \
"so MAPK is 1.5e-5g/ml = 0.36uM. which is closer to our first estimate." \
"Lets use this."
call /kinetics/MAPK/craf-1**/notes LOAD \
"Negative feedback by MAPK* by hyperphosphorylating craf-1* gives" \
"rise to this pool." \
"Ueki et al JBC 269(22):15756-15761, 1994" \
""
call /kinetics/MAPK/MAPK-tyr/notes LOAD \
"Haystead et al FEBS Lett. 306(1) pp 17-22 show that phosphorylation" \
"is strictly sequential, first tyr185 then thr183."
call /kinetics/MAPK/MAPKK*/notes LOAD \
"MAPKK phosphorylates MAPK on both the tyr and thr residues, first" \
"tyr then thr. Refs: Seger et al JBC267:20 pp 14373 1992" \
"The MAPKK itself is phosphorylated on ser as well as thr residues." \
"Let us assume that the ser goes first, and that the sequential phosphorylation" \
"is needed. See Kyriakis et al Nature 358 417-421 1992"
call /kinetics/MAPK/MAPKK*/MAPKKtyr/notes LOAD \
"The actual MAPKK is 2 forms from Seger et al JBC 267:20 14373(1992)" \
"Vmax = 150nmol/min/mg" \
"From Haystead et al FEBS 306(1):17-22 we get Km=46.6nM for at least one" \
"of the phosphs." \
"Putting these together:" \
"k3=0.15/sec, scale to get k2=0.6." \
"k1=0.75/46.6nM=2.7e-5"
call /kinetics/MAPK/MAPKK*/MAPKKthr/notes LOAD \
"Rate consts same as for MAPKKtyr."
call /kinetics/MAPK/MAPKK-ser/notes LOAD \
"Intermediately phophorylated, assumed inactive, form of MAPKK"
call /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1/notes LOAD \
"Based on acc79.g from Ajay and Bhalla 2007."
call /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2/notes LOAD \
"Based on acc79.g" \
""
call /kinetics/MAPK/Ras-act-craf/notes LOAD \
"Assume the binding is fast and limited only by the amount of " \
"Ras* available. So kf=kb/[craf-1]" \
"If kb is 1/sec, then kf = 1/0.2 uM = 1/(0.2 * 6e5) = 8.3e-6" \
"Later: Raise it by 10 X to 4e-5" \
"From Hallberg et al JBC 269:6 3913-3916 1994, 3% of cellular Raf is" \
"complexed with Ras. So we raise kb 4x to 4" \
"This step needed to memb-anchor and activate Raf: Leevers et al Nature" \
"369 411-414" \
"(I don't...."
call /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1/notes LOAD \
"Kinetics are the same as for the craf-1* activity, ie.," \
"k1=1.1e-6, k2=.42, k3 =0.105" \
"These are based on Force et al PNAS USA 91 1270-1274 1994." \
"These parms cannot reach the observed 4X stim of MAPK. So lets" \
"increase the affinity, ie, raise k1 10X to 1.1e-5" \
"Lets take it back down to where it was." \
"Back up to 5X: 5.5e-6"
call /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2/notes LOAD \
"Same kinetics as other c-raf activated forms. See " \
"Force et al PNAS 91 1270-1274 1994." \
"k1 = 1.1e-6, k2 = .42, k3 = 1.05" \
"raise k1 to 5.5e-6" \
""
call /kinetics/MKP-1/notes LOAD \
"MKP-1 dephosphoryates and inactivates MAPK in vivo: Sun et al Cell 75 " \
"487-493 1993. Levels of MKP-1" \
"are regulated, and rise in 1 hour. " \
"Kinetics from Charles et al PNAS 90:5292-5296 1993. They refer" \
"to Charles et al Oncogene 7 187-190 who show that half-life of MKP1/3CH134" \
"is 40 min. 80% deph of MAPK in 20 min" \
"Sep 17 1997: CoInit now 0.4x to 0.0032. See parm searches" \
"from jun96 on."
call /kinetics/MKP-1/MKP1-tyr-deph/notes LOAD \
"The original kinetics have been modified to obey the k2 = 4 * k3 rule," \
"while keeping kcat and Km fixed. As noted in the NOTES, the only constraining" \
"data point is the time course of MAPK dephosphorylation, which this" \
"model satisfies. It would be nice to have more accurate estimates of" \
"rate consts and MKP-1 levels from the literature. " \
"Effective Km : 67 nM" \
"kcat = 1.43 umol/min/mg" \
"Raised kcat from 1 to 4, following acc79.g (Ajay and Bhalla07)"
call /kinetics/MKP-1/MKP1-thr-deph/notes LOAD \
"See MKP1-tyr-deph"
call /kinetics/PPhosphatase2A/notes LOAD \
"Refs: Pato et al Biochem J 293:35-41(93);" \
"Takai&Mieskes Biochem J 275:233-239" \
"k1=1.46e-4, k2=1000,k3=250. these use" \
"kcat values for calponin. Also, units of kcat may be in min!" \
"revert to Vmax base:" \
"k3=6, k2=25,k1=3.3e-6 or 6,6,1e-6" \
"CoInit assumed 0.1 uM." \
"See NOTES for MAPK_Ras50.g. CoInit now 0.08" \
"Sep 17 1997: Raise CoInt 1.4x to 0.224, see parm" \
"searches from jun 96 on." \
""
call /kinetics/PPhosphatase2A/craf-deph/notes LOAD \
"See parent PPhosphatase2A for parms" \
""
call /kinetics/PPhosphatase2A/MAPKK-deph/notes LOAD \
"See: Kyriakis et al Nature 358 pp 417-421 1992" \
"Ahn et al Curr Op Cell Biol 4:992-999 1992 for this pathway." \
"See parent PPhosphatase2A for parms."
call /kinetics/PPhosphatase2A/craf**-deph/notes LOAD \
"Ueki et al JBC 269(22) pp 15756-15761 1994 show hyperphosphorylation of" \
"craf, so this is there to dephosphorylate it. Identity of phosphatase is not" \
"known to me, but it may be PP2A like the rest, so I have made it so."
call /kinetics/Ras/notes LOAD \
"Ras has now gotten to be a big enough component of the model to" \
"deserve its own group. The main refs are" \
"Boguski and McCormick Nature 366 643-654 '93 Major review" \
"Eccleston et al JBC 268:36 pp 27012-19" \
"Orita et al JBC 268:34 2554246"
call /kinetics/Ras/bg-act-GEF/notes LOAD \
"SoS/GEF is present at 50 nM ie 3e4/cell. BetaGamma maxes out at 9e4." \
"Assume we have 1/3 of the GEF active when the BetaGamma is 1.5e4." \
"so 1e4 * kb = 2e4 * 1.5e4 * kf, so kf/kb = 3e-5. The rate of this equil should" \
"be reasonably fast, say 1/sec" \
""
call /kinetics/Ras/GEF-Gprot-bg/notes LOAD \
"Guanine nucleotide exchange factor. This activates raf by exchanging bound" \
"GDP with GTP. I have left the GDP/GTP out of this reaction, it would be" \
"trivial to put them in. See Boguski & McCormick." \
"Possible candidate molecules: RasGRF, smgGDS, Vav (in dispute). " \
"rasGRF: Kcat= 1.2/min    Km = 680 nM" \
"smgGDS: Kcat: 0.37 /min, Km = 220 nM." \
"vav: Turnover up over baseline by 10X, " \
""
call /kinetics/Ras/GEF-Gprot-bg/GEF-bg_act-ras/notes LOAD \
"Kinetics based on the activation of Gq by the receptor complex in the" \
"Gq model (in turn based on the Mahama and Linderman model)" \
"k1 = 2e-5, k2 = 1e-10, k3 = 10 (I do not know why they even bother with k2)." \
"Lets put k1 at 2e-6 to get a reasonable equilibrium" \
"More specific values from, eg.g: Orita et al JBC 268(34) 25542-25546" \
"from rasGRF and smgGDS: k1=3.3e-7; k2 = 0.08, k3 = 0.02" \
""
call /kinetics/Ras/inact-GEF/notes LOAD \
"Assume that SoS is present only at 50 nM." \
"Revised to 100 nM to get equil to experimentally known levels."
call /kinetics/Ras/GEF*/notes LOAD \
"phosphorylated and thereby activated form of GEF. See, e.g." \
"Orita et al JBC 268:34 25542-25546 1993, Gulbins et al." \
"It is not clear whether there is major specificity for tyr or ser/thr."
call /kinetics/Ras/GEF*/GEF*-act-ras/notes LOAD \
"Kinetics same as GEF-bg-act-ras" \
""
call /kinetics/Ras/GTP-Ras/notes LOAD \
"Only a very small fraction (7% unstim, 15% stim) of ras is GTP-bound." \
"Gibbs et al JBC 265(33) 20437" \
""
call /kinetics/Ras/GDP-Ras/notes LOAD \
"GDP bound form. See Rosen et al Neuron 12 1207-1221 June 1994." \
"the activation loop is based on Boguski and McCormick Nature 366 643-654 93" \
"Assume Ras is present at about the same level as craf-1, 0.2 uM." \
"Hallberg et al JBC 269:6 3913-3916 1994 estimate upto 5-10% of cellular" \
"Raf is assoc with Ras. Given that only 5-10% of Ras is GTP-bound, we" \
"need similar amounts of Ras as Raf."
call /kinetics/Ras/Ras-intrinsic-GTPase/notes LOAD \
"This is extremely slow (1e-4), but it is significant as so little GAP actually" \
"gets complexed with it that the total GTP turnover rises only by" \
"2-3 X (see Gibbs et al, JBC 265(33) 20437-20422) and " \
"Eccleston et al JBC 268(36) 27012-27019" \
"kf = 1e-4" \
""
call /kinetics/Ras/dephosph-GAP/notes LOAD \
"Assume a reasonably good rate for dephosphorylating it, 1/sec"
call /kinetics/Ras/GAP/notes LOAD \
"GTPase-activating proteins. See Boguski and McCormick." \
"Turn off Ras by helping to hydrolyze bound GTP. " \
"This one is probably NF1, ie.,  Neurofibromin as it is inhibited by AA and lipids," \
"and expressed in neural cells. p120-GAP is also a possible candidate, but" \
"is less regulated. Both may exist at similar levels." \
"See Eccleston et al JBC 268(36) pp27012-19" \
"Level=.002"
call /kinetics/Ras/GAP/GAP-inact-ras/notes LOAD \
"From Eccleston et al JBC 268(36)pp27012-19 get Kd < 2uM, kcat - 10/sec" \
"From Martin et al Cell 63 843-849 1990 get Kd ~ 250 nM, kcat = 20/min" \
"I will go with the Eccleston figures as there are good error bars (10%). In general" \
"the values are reasonably close." \
"k1 = 1.666e-3/sec, k2 = 1000/sec, k3 = 10/sec (note k3 is rate-limiting)" \
"5 Nov 2002: Changed ratio term to 4 from 100. Now we have" \
"k1=8.25e-5; k2=40, k3=10. k3 is still rate-limiting."
call /kinetics/Ras/inact-GEF*/notes LOAD \
"Phosphorylation-inactivated form of GEF. See" \
"Hordijk et al JBC 269:5 3534-3538 1994" \
"and " \
"Buregering et al EMBO J 12:11 4211-4220 1993" \
""
call /kinetics/Ras/CaM-bind-GEF/notes LOAD \
"Nov 2008: Updated based on acc79.g from Ajay and Bhalla 2007." \
"" \
"We have no numbers for this. It is probably between" \
"the two extremes represented by the CaMKII phosph states," \
"and I have used guesses based on this." \
"kf=1e-4" \
"kb=1" \
"The reaction is based on Farnsworth et al Nature 376 524-527" \
"1995"
call /kinetics/Ras/CaM-GEF/notes LOAD \
"See Farnsworth et al Nature 376 524-527 1995"
call /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras/notes LOAD \
"Kinetics same as GEF-bg_act-ras" \
""
call /kinetics/PKA-active/PKA-phosph-GEF/notes LOAD \
"This pathway inhibits Ras when cAMP is elevated. See:" \
"Hordijk et al JBC 269:5 3534-3538 1994" \
"Burgering et al EMBO J 12:11 4211-4220 1993" \
"The rates are the same as used in PKA-phosph-I1"
call /kinetics/PKA-active/PKA-phosph-I1/notes LOAD \
"#s from Bramson et al CRC crit rev Biochem" \
"15:2 93-124. They have a huge list of peptide substrates" \
"and I have chosen high-ish rates." \
"These consts give too much PKA activity, so lower Vmax 1/3." \
"Now, k1 = 3e-5, k2 = 36, k3 = 9 (still pretty fast)." \
"Also lower Km 1/3  so k1 = 1e-5" \
"Cohen et al FEBS Lett 76:182-86 1977 say rate =30% PKA act on " \
"phosphokinase beta." \
""
call /kinetics/PKA-active/phosph-PDE/notes LOAD \
"Same rates as PKA-phosph-I1"
call /kinetics/EGFR/EGFR/notes LOAD \
"Berkers et al JBC 266 say 22K hi aff recs." \
"Sherrill and Kyte Biochemistry 35 use range 4-200 nM." \
"These match, lets use them."
call /kinetics/EGFR/act_EGFR/notes LOAD \
"Affinity of EGFR for EGF is complex: depends on [EGFR]." \
"We'll assume fixed [EGFR] and use exptal" \
"affinity ~20 nM (see Sherrill and Kyte" \
"Biochem 1996 35 5705-5718, Berkers et al JBC 266:2 922-927" \
"1991, Sorokin et al JBC 269:13 9752-9759 1994). " \
"Tau =~2 min (Davis et al JBC 263:11 5373-5379 1988)" \
"or Berkers Kass = 6.2e5/M/sec, Kdiss=3.5e-4/sec." \
"Sherrill and Kyte have Hill Coeff=1.7" \
""
call /kinetics/EGFR/L.EGFR/notes LOAD \
"This is terribly simplified: there are many interesting" \
"intermediate stages, including dimerization and assoc" \
"with adapter molecules like Shc, that contribute to the" \
"activation of the EGFR."
call /kinetics/EGFR/L.EGFR/phosph_PLC_g/notes LOAD \
"Hsu et al JBC 266:1 603-608 1991" \
"Km = 385 +- 100 uM, Vm = 5.1 +-1 pmol/min/ug for PLC-771." \
"Other sites have similar range, but are not stim as much" \
"by EGF." \
"k1 = 2.8e-2/385/6e5 = 1.2e-10. Phenomenally slow." \
"But Sherrill and Kyte say turnover # for angiotensin II is" \
"5/min for cell extt, and 2/min for placental. Also see" \
"Okada et al for Shc rates which are much faster."
call /kinetics/EGFR/L.EGFR/phosph_Shc/notes LOAD \
"Rates from Okada et al JBC 270:35 pp 20737 1995" \
"Km = 0.70 to 0.85 uM, Vmax = 4.4 to 5.0 pmol/min. Unfortunately" \
"the amount of enzyme is not known, the prep is only" \
"partially purified." \
"Time course of phosph is max within 30 sec, falls back within" \
"20 min. Ref: Sasaoka et al JBC 269:51 32621 1994." \
"Use k3 = 0.1 based on this tau." \
""
call /kinetics/EGFR/SHC/notes LOAD \
"There are 2 isoforms: 52 KDa and 46 KDa (See Okada et al" \
"JBC 270:35 pp 20737 1995). They are acted up on by the EGFR" \
"in very similar ways, and apparently both bind Grb2 similarly," \
"so we'll bundle them together here." \
"Sasaoka et al JBC 269:51 pp 32621 1994 show immunoprecs where" \
"it looks like there is at least as much Shc as Grb2. So" \
"we'll tentatively say there is 1 uM of Shc."
call /kinetics/EGFR/dephosph_Shc/notes LOAD \
"Time course of decline of phosph is 20 min. Part of this is" \
"the turnoff time of the EGFR itself. Lets assume a tau of" \
"10 min for this dephosph. It may be wildly off."
call /kinetics/EGFR/Internalize/notes LOAD \
"See Helin and Beguinot JBC 266:13 1991 pg 8363-8368." \
"In Fig 3 they have internalization tau about 10 min, " \
"equil at about 20% EGF available. So kf = 4x kb, and" \
"1/(kf + kb) = 600 sec so kb = 1/3K = 3.3e-4," \
"and kf = 1.33e-3. This doesn't take into account the" \
"unbound receptor, so we need to push the kf up a bit, to" \
"0.002"
call /kinetics/Sos/Shc_bind_Sos.Grb2/notes LOAD \
"Sasaoka et al JBC 269:51 pp 32621 1994, table on pg" \
"32623 indicates that this pathway accounts for about " \
"50% of the GEF activation. (88% - 39%). Error is large," \
"about 20%. Fig 1 is most useful in constraining rates." \
"" \
"Chook et al JBC 271:48 pp 30472, 1996 say that the Kd is" \
"0.2 uM for Shc binding to EGFR. The Kd for Grb direct binding" \
"is 0.7, so we'll ignore it."
call /kinetics/Sos/Grb2_bind_Sos*/notes LOAD \
"Same rates as Grb2_bind_Sos: Porfiri and McCormick JBC" \
"271:10 pp 5871 1996 show that the binding is not affected" \
"by the phosph."
call /kinetics/Sos/Grb2/notes LOAD \
"There is probably a lot of it in the cell: it is also known" \
"as Ash (abundant src homology protein I think). Also " \
"Waters et al JBC 271:30 18224 1996 say that only a small" \
"fraction of cellular Grb is precipitated out when SoS is" \
"precipitated. As most of the Sos seems to be associated" \
"with Grb2, it would seem like there is a lot of the latter." \
"Say 1 uM. I haven't been able to find a decent...."
call /kinetics/Sos/dephosph_Sos/notes LOAD \
"The only clue I have to these rates is from the time" \
"courses of the EGF activation, which is around 1 to 5 min." \
"The dephosph would be expected to be of the same order," \
"perhaps a bit longer. Lets use 0.002 which is about 8 min." \
"Sep 17: The transient activation curve matches better with" \
"kf = 0.001"
call /kinetics/Sos/Grb2_bind_Sos/notes LOAD \
"As there are 2 SH3 domains, this reaction could be 2nd order." \
"I have a Kd of 22 uM from peptide binding (Lemmon et al " \
"JBC 269:50 pg 31653). However, Chook et al JBC 271:48 pg30472" \
"say it is 0.4uM with purified proteins, so we believe them." \
"They say it is 1:1 binding."
call /kinetics/Sos/Sos/notes LOAD \
"I have tried using low (0.02 uM) initial concs, but these" \
"give a very flat response to EGF stim although the overall" \
"activation of Ras is not too bad. I am reverting to 0.1 " \
"because we expect a sharp initial response, followed by" \
"a decline." \
"Sep 17 1997: The transient activation curve looks better with" \
"[Sos] = 0.05." \
"Apr 26 1998: Some error there, it is better where it was" \
"at 0.1"
call /kinetics/PLC_g/PLC_g/notes LOAD \
"Amount from Homma et al JBC 263:14 6592-6598 1988"
call /kinetics/PLC_g/Ca_act_PLC_g/notes LOAD \
"Nice curves from Homma et al JBC 263:14 6592-6598 1988 " \
"Fig 5c. The activity falls above 10 uM, but that is too high" \
"to reach physiologically anyway, so we'll ignore the higher" \
"pts and match the lower ones only. Half-max at 1 uM." \
"But  Wahl et al JBC 267:15 10447-10456 1992 have half-max" \
"at 56 nM which is what I'll use."
call /kinetics/PLC_g/Ca_act_PLC_g*/notes LOAD \
"Again, we refer to Homma et al and Wahl et al, for preference" \
"using Wahl. Half-Max of the phosph form is at 316 nM." \
"Use kb of 10 as this is likely to be pretty fast." \
"Did some curve comparisons, and instead of 316 nM giving" \
"a kf of 5.27e-5, we will use 8e-5 for kf. " \
"16 Sep 97. As we are now phosphorylating the Ca-bound form," \
"equils have shifted. kf should now be 2e-5 to match the" \
"curves."
call /kinetics/PLC_g/Ca.PLC_g/PIP2_hydrolysis/notes LOAD \
"Mainly Homma et al JBC 263:14 1988 pp 6592, but these" \
"parms are the Ca-stimulated form. It is not clear whether" \
"the enzyme is activated by tyrosine phosphorylation at this" \
"point or not. Wahl et al JBC 267:15 10447-10456 1992 say" \
"that the Ca_stim and phosph form has 7X higher affinity " \
"for substrate than control. This is close to Wahl's" \
"figure 7, which I am using as reference."
call /kinetics/PLC_g/Ca.PLC_g*/PIP2_hydrolysis/notes LOAD \
"Mainly Homma et al JBC 263:14 1988 pp 6592, but these" \
"parms are the Ca-stimulated form. It is not clear whether" \
"the enzyme is activated by tyrosine phosphorylation at this" \
"point or not. Wahl et al JBC 267:15 10447-10456 1992 say" \
"that this has 7X higher affinity for substrate than control."
call /kinetics/CaMKII/notes LOAD \
"Main reference here is the review by Hanson and Schulman, Ann Rev Biochem" \
"1992 vol 61 pp 559-601. Most of the mechanistic details and a few constants" \
"are derived from there. Many kinetics are from Hanson and Schulman JBC" \
"267:24 17216-17224 1992." \
"The enzs look a terrible mess. Actually it is just 3 reactions for diff sites," \
"by 4 states of CaMKII, defined by the phosph state."
call /kinetics/CaMKII/CaMKII/notes LOAD \
"Huge conc of CaMKII. In PSD it is 20-40% of protein, so we assume it is around" \
"2.5% of protein in spine as a whole. This level is so high it is unlikely to matter" \
"much if we are off a bit. This comes to about 70 uM."
call /kinetics/CaMKII/CaMKII-thr286*-CaM/notes LOAD \
"From Hanson and Schulman, the thr286 is responsible for autonomous activation" \
"of CaMKII."
call /kinetics/CaMKII/CaMKII***/notes LOAD \
"From Hanson and Schulman, the CaMKII does a lot of autophosphorylation" \
"just after the CaM is released. This prevents further CaM binding and renders" \
"the enzyme quite independent of Ca."
call /kinetics/CaMKII/CaMKII-bind-CaM/notes LOAD \
"This is tricky. There is some cooperativity here arising from interactions" \
"between the subunits of the CAMKII holoenzyme. However, the" \
"stoichiometry is 1. " \
"Kb/Kf = 6e4 #/cell. Rate is fast (see Hanson et al Neuron 12 943-956 1994)" \
"so lets say kb = 10. This gives kf = 1.6667e-4" \
"H&S AnnRev Biochem 92 give tau for dissoc as 0.2 sec at low Ca, 0.4 at high." \
"Low Ca = 100 nM = physiol."
call /kinetics/CaMKII/CaMK-thr286-bind-CaM/notes LOAD \
"Affinity is up 1000X. Time to release is about 20 sec, so the kb is OK at 0.1" \
"This makes Kf around 1.6666e-3" \
""
call /kinetics/CaMKII/CaMKII-thr286/notes LOAD \
"I am not sure if we need to endow this one with a lot of enzs. It is likely" \
"to be a short-lived intermediate, since it will be phosphorylated further" \
"as soon as the CAM falls off."
call /kinetics/CaMKII/CaMK-thr306/notes LOAD \
"This forms due to basal autophosphorylation, but I think it has to be" \
"considered as a pathway even if some CaM is floating around. In either" \
"case it will tend to block further binding of CaM, and will not display any" \
"enzyme activity. See Hanson and Schulman JBC 267:24 pp17216-17224 1992"
call /kinetics/CaMKII/basal-activity/notes LOAD \
"This reaction represents one of the big unknowns in CaMK-II" \
"biochemistry: what maintains the basal level of phosphorylation" \
"on thr 286 ? See Hanson and Schulman Ann Rev Biochem 1992" \
"61:559-601, specially pg 580, for review. I have not been able to" \
"find any compelling mechanism in the literature, but fortunately" \
"the level of basal activity is well documented. "
call /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_305/notes LOAD \
"Rates from autocamtide phosphorylation, from " \
"Hanson and Schulman JBC 267:24 17216-17224 1992." \
"Jan 1 1998: Speed up 12x to match fig 5."
call /kinetics/CaMKII/tot_autonomous_CaMKII/auton_305/notes LOAD \
"See Hanson and Schulman again, for afterburst rates of" \
"phosph."
call /kinetics/PP1-active/notes LOAD \
"Cohen et al Meth Enz 159 390-408 is main source of info" \
"conc  = 1.8 uM"
call /kinetics/PP1-active/Deph-thr286/notes LOAD \
"The rates are from Stralfors et al Eur J Biochem 149 295-303 giving" \
"Vmax = 5.7 umol/min giving k3 = 3.5/sec and k2 = 14." \
"Foulkes et al Eur J Biochem 132 309-313 1983 give Km = 5.1 uM so" \
"k1 becomes 5.72e-6" \
"Simonelli 1984 (Grad Thesis, CUNY) showed that other substrates" \
"are about 1/10 rate of phosphorylase a, so we reduce k1,k2,k3 by 10" \
"to 5.72e-7, 1.4, 0.35"
call /kinetics/PP1-active/Deph-thr306/notes LOAD \
"See Cohen et al"
call /kinetics/CaM/CaM/notes LOAD \
"There is a LOT of this in the cell: upto 1% of total protein mass. (Alberts et al)" \
"Say 25 uM. Meyer et al Science 256 1199-1202 1992 refer to studies saying" \
"it is comparable to CaMK levels. " \
""
call /kinetics/CaM/CaM-TR2-bind-Ca/notes LOAD \
"Lets use the fast rate consts here. Since the rates are so different, I am not" \
"sure whether the order is relevant. These correspond to the TR2C fragment." \
"We use the Martin et al rates here, plus the Drabicowski binding consts." \
"All are scaled by 3X to cell temp." \
"kf = 2e-10 kb = 72" \
"Stemmer & Klee: K1=.9, K2=1.1. Assume 1.0uM for both. kb/kf=3.6e11." \
"If kb=72, kf = 2e-10 (Exactly the same !)...."
call /kinetics/CaM/CaM-TR2-Ca2-bind-Ca/notes LOAD \
"K3 = 21.5, K4 = 2.8. Assuming that the K4 step happens first, we get" \
"kb/kf = 2.8 uM = 1.68e6 so kf =6e-6 assuming kb = 10" \
""
call /kinetics/CaM/CaM-Ca3-bind-Ca/notes LOAD \
"Use K3 = 21.5 uM here from Stemmer and Klee table 3." \
"kb/kf =21.5 * 6e5 so kf = 7.75e-7, kb = 10"
call /kinetics/CaM/CaM-TR2-Ca2/notes LOAD \
"This is the intermediate where the TR2 end (the high-affinity end) has" \
"bound the Ca but the TR1 end has not."
call /kinetics/PP1/I1/notes LOAD \
"I1 is a 'mixed' inhibitor, but at high enz concs it looks like a non-compet" \
"inhibitor (Foulkes et al Eur J Biochem 132 309-313 9183)." \
"We treat it as non-compet, so it just turns the enz off" \
"without interacting with the binding site." \
"Cohen et al ann rev bioch refer to results where conc is " \
"1.5 to 1.8 uM. In order to get complete inhib of PP1, which is at 1.8 uM," \
"we need >= 1.8 uM." \
"" \
""
call /kinetics/PP1/I1*/notes LOAD \
"Dephosph is mainly by PP2B"
call /kinetics/PP1/Inact-PP1/notes LOAD \
"K inhib = 1nM from Cohen Ann Rev Bioch 1989, " \
"4 nM from Foukes et al " \
"Assume 2 nM. kf /kb = 8.333e-4"
call /kinetics/PP1/dissoc-PP1-I1/notes LOAD \
"Let us assume that the equil in this case is very far over to the" \
"right. This is probably safe." \
""
call /kinetics/PP2A/PP2A-dephosph-I1/notes LOAD \
"PP2A does most of the dephosph of I1 at basal Ca levels. See" \
"the review by Cohen in Ann Rev Biochem 1989." \
"For now, lets halve Km. k1 was 3.3e-6, now 6.6e-6" \
""
call /kinetics/PP2A/PP2A-dephosph-PP1-I*/notes LOAD \
"k1 changed from 3.3e-6 to 6.6e-6" \
""
call /kinetics/CaNAB-Ca4/dephosph_inhib1_noCaM/notes LOAD \
"The rates here are so slow I do not know if we should even bother" \
"with this enz reacn. These numbers are from Liu and Storm." \
"Other refs suggest that the Km stays the same" \
"but the Vmax goes to 10% of the CaM stim levels. " \
"Prev: k1=2.2e-9, k2 = 0.0052, k3 = 0.0013" \
"New : k1=5.7e-8, k2=.136, k3=.034"
call /kinetics/PP2B/notes LOAD \
"Also called Calcineurin." \
"Major sources of info:" \
"Cohen, P Ann Rev Biochem 1989 58:453-508" \
"Mumby and Walker Physiol Rev 73:4 673-699" \
"Stemmer and Klee Biochem 33 1994 6859-6866" \
"Liu and Storm JBC 264:22 1989 12800-12804" \
"This model is unusual: There is actually more expt info than I want to" \
"put in the model at this time." \
"Phosph: Hashimoto and Soderling JBC 1989 264:28 16624-16629 (Not used)"
call /kinetics/PP2B/CaNAB/notes LOAD \
"We assume that the A and B subunits of PP2B are always bound under" \
"physiol conditions." \
"Up to 1% of brain protein = 25 uM. I need to work out how it is distributed" \
"between cytosolic and particulate fractions." \
"Tallant and Cheung '83 Biochem 22 3630-3635 have conc in many " \
"species, average for mammalian brain is around 1 uM." \
"10 Feb 2009." \
"Halved conc to 0.5 uM."
call /kinetics/PP2B/Ca-bind-CaNAB-Ca2/notes LOAD \
"This process is probably much more complicated and involves CaM." \
"However, as I can't find detailed info I am bundling this into a" \
"single step." \
"Based on Steemer and Klee pg 6863, the Kact is 0.5 uM." \
"kf/kb = 1/(0.5 * 6e5)^2 = 1.11e-11"
call /kinetics/PP2B/Ca-bind-CaNAB/notes LOAD \
"going on the experience with CaM, we put the fast (high affinity)" \
"sites first. We only know (Stemmer and Klee) that the affinity is < 70 nM." \
"Assuming 10 nM at first, we get" \
"kf = 2.78e-8, kb = 1." \
"Try 20 nM." \
"kf = 7e-9, kb = 1" \
"" \
""
call /kinetics/PKA/notes LOAD \
"General ref: Taylor et al Ann Rev Biochem 1990 59:971-1005" \
""
call /kinetics/PKA/R2C2/notes LOAD \
"This is the R2C2 complex, consisting of 2 catalytic (C)" \
"subunits, and the R-dimer. See Taylor et al Ann Rev Biochem" \
"1990 59:971-1005 for a review." \
"The Doskeland and Ogreid review is better for numbers." \
"Amount of PKA is about .5 uM."
call /kinetics/PKA/R2C2-cAMP/notes LOAD \
"CoInit was .0624" \
""
call /kinetics/PKA/cAMP-bind-site-B1/notes LOAD \
"Hasler et al FASEB J 6:2734-2741 1992 say Kd =1e-7M" \
"for type II, 5.6e-8 M for type I. Take mean" \
"which comes to 2e-13 #/cell" \
"Smith et al PNAS USA 78:3 1591-1595 1981 have better data." \
"First kf/kb=2.1e7/M = 3.5e-5 (#/cell)." \
"Ogreid and Doskeland Febs Lett 129:2 287-292 1981 have figs" \
"suggesting time course of complete assoc is < 1 min."
call /kinetics/PKA/cAMP-bind-site-B2/notes LOAD \
"For now let us set this to the same Km (1e-7M) as" \
"site B. This gives kf/kb = .7e-7M * 1e6 / (6e5^2) : 1/(6e5^2)" \
"= 2e-13:2.77e-12" \
"Smith et al have better values. They say that this is" \
"cooperative, so the consts are now kf/kb =8.3e-4"
call /kinetics/PKA/R2-cAMP4/notes LOAD \
"Starts at 0.15 for the test of fig 6 in Smith et al, but we aren't using" \
"that paper any more."
call /kinetics/PKA/Release-C1/notes LOAD \
"This has to be fast, as the activation of PKA by cAMP" \
"is also fast." \
"kf was 10" \
""
call /kinetics/PKA/PKA-inhibitor/notes LOAD \
"About 25% of PKA C subunit is dissociated in resting cells without" \
"having any noticable activity." \
"Doskeland and Ogreid Int J biochem 13 pp1-19 suggest that this is" \
"because there is a corresponding amount of inhibitor protein."
call /kinetics/PKA/inhib-PKA/notes LOAD \
"This has to be set to zero for matching the expts in vitro. In vivo" \
"we need to consider the inhibition though." \
"kf = 1e-5" \
"kb = 1" \
""
call /kinetics/cAMP/notes LOAD \
"The conc of this has been a problem. Schaecter and Benowitz use 50 uM," \
"but Shinomura et al have < 5. So I have altered the cAMP-dependent " \
"rates in the PKA model to reflect this."
call /kinetics/AC/ATP/notes LOAD \
"ATP is present in all cells between 2 and 10 mM. See Lehninger."
call /kinetics/AC/AC1-CaM/notes LOAD \
"This version of cyclase is Calmodulin activated." \
"Gs stims it but betagamma inhibits."
call /kinetics/AC/AC1/notes LOAD \
"Starting conc at 20 nM."
call /kinetics/AC/CaM-bind-AC1/notes LOAD \
"Half-max at 20 nM CaM (Tang et al JBC 266:13 8595-8603 1991" \
"kb/kf = 20 nM = 12000 #/cell" \
"so kf = kb/12000 = kb * 8.333e-5" \
""
call /kinetics/AC/AC2*/notes LOAD \
"This version is activated by Gs and by a betagamma and phosphorylation."
call /kinetics/AC/AC2*/kenz/notes LOAD \
"Reduced Km to match expt data for basal activation of AC2 by PKC." \
"Now k1 = 2.9e-6, k2 = 72, k3 = 18" \
""
call /kinetics/AC/AC2/notes LOAD \
"Starting at 0.015 uM."
call /kinetics/AC/dephosph-AC2/notes LOAD \
"Random rate."
call /kinetics/AC/Gs-bind-AC2/notes LOAD \
"Half-max at around 3nM = kb/kf from fig 5 in " \
"Feinstein et al PNAS USA 88 10173-10177 1991" \
"kf = kb/1800 = 5.56e-4 kb" \
"Ofer's thesis data indicates it is more like 2 nM." \
"kf = kb/1200 = 8.33e-4" \
""
call /kinetics/AC/Gs-bind-AC1/notes LOAD \
"Half-max 8nM from Tang et al JBC266:13 8595-8603" \
"kb/kf = 8 nM = 4800#/cell so kf = kb * 2.08e-4"
call /kinetics/AC/Gs-bind-AC2*/notes LOAD \
"kb/kf = 1.2 nM" \
"so kf = kb/720 = 1.3888 * kb."
call /kinetics/AC/cAMP-PDE/notes LOAD \
"The levels of the PDE are not known at this time. However," \
"enough" \
"kinetic info and info about steady-state levels of cAMP" \
"etc are around" \
"to make it possible to estimate this."
call /kinetics/AC/cAMP-PDE/PDE/notes LOAD \
"Best rates are from Conti et al Biochem 34 7979-7987 1995." \
"Though these" \
"are for the Sertoli cell form, it looks like they carry" \
"nicely into" \
"alternatively spliced brain form. See Sette et al" \
"JBC 269:28 18271-18274" \
"Km ~2 uM, Vmax est ~ 10 umol/min/mg for pure form." \
"Brain protein is 93 kD but this was 67." \
"So k3 ~10, k2 ~40, k1 ~4.2e-6"
call /kinetics/AC/cAMP-PDE*/notes LOAD \
"This form has about 2X activity as plain PDE. See Sette et al JBC 269:28" \
"18271-18274 1994."
call /kinetics/AC/cAMP-PDE*/PDE*/notes LOAD \
"This form has about twice the activity of the unphosphorylated form. See" \
"Sette et al JBC 269:28 18271-18274 1994." \
"We'll ignore cGMP effects for now."
call /kinetics/AC/dephosph-PDE/notes LOAD \
"The rates for this are poorly constrained. In adipocytes (probably a" \
"different PDE) the dephosphorylation is complete within 15 min, but" \
"there are no intermediate time points so it could be much faster. Identity" \
"of phosphatase etc is still unknown."
call /kinetics/AC/PDE1/notes LOAD \
"CaM-Dependent PDE. Amount calculated from total rate in" \
"brain vs. specific rate. "
call /kinetics/AC/PDE1/PDE1/notes LOAD \
"Rate is 1/6 of the CaM stim form. We'll just reduce" \
"all lf k1, k2, k3 so that the Vmax goes down 1/6."
call /kinetics/AC/CaM.PDE1/notes LOAD \
"Activity up 6x following Ca-CaM binding."
call /kinetics/AC/CaM.PDE1/CaM.PDE1/notes LOAD \
"Max activity ~10umol/min/mg in presence of lots of CaM." \
"Affinity is low, 40 uM." \
"k3 = 10, k2 = 40, k1 = (50/40) / 6e5."
call /kinetics/AC/CaM_bind_PDE1/notes LOAD \
"For olf epi PDE1, affinity is 7 nM. Assume same for brain." \
"Reaction should be pretty fast. Assume kb = 5/sec." \
"Then kf = 5 / (0.007 * 6e5) = 1.2e-3"
complete_loading
