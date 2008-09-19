//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Thu Sep 18 13:17:43 2008
 
include kkit {argv 1}
 
FASTDT = 2e-05
SIMDT = 0.001
CONTROLDT = 10
PLOTDT = 1
MAXTIME = 2500
TRANSIENT_TIME = 2
VARIABLE_DT_FLAG = 1
DEFAULT_VOL = 1.257e-16
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
simundump geometry /kinetics/dendgeom 0 1.257e-16 3 cylinder "" white black 8 \
  7 0
simundump group /kinetics/PKC 0 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 8.9507 -3.7837 0
simundump kpool /kinetics/PKC/PKC-Ca 0 -1e-12 3.7208e-17 3.7208e-17 \
  2.8062e-12 2.8062e-12 0 0 75420 0 /kinetics/dendgeom red black 7.9248 \
  -10.489 0
simundump kreac /kinetics/PKC/PKC-act-by-Ca 0 7.9555e-06 0.5 "" white blue \
  7.9248 -12.123 0
simundump kreac /kinetics/PKC/PKC-act-by-DAG 0 1.0607e-07 8.6348 "" white \
  blue 9.9388 -11.306 0
simundump kreac /kinetics/PKC/PKC-Ca-to-memb 0 1.2705 3.5026 "" white blue \
  8.2026 -7.7467 0
simundump kreac /kinetics/PKC/PKC-DAG-to-memb 0 1 0.1 "" white blue 9.3832 \
  -9.2638 0
simundump kreac /kinetics/PKC/PKC-act-by-Ca-AA 0 1.5911e-08 0.1 "" white blue \
  11.212 -8.1843 0
simundump kreac /kinetics/PKC/PKC-act-by-DAG-AA 0 2 0.2 "" white blue 13.249 \
  -8.7678 0
simundump kpool /kinetics/PKC/PKC-DAG-AA* 0 -1e-12 4.9137e-18 4.9137e-18 \
  3.7059e-13 3.7059e-13 0 0 75420 0 /kinetics/dendgeom cyan blue 12.601 \
  -6.463 0
simundump kpool /kinetics/PKC/PKC-Ca-AA* 0 -1e-12 1.7501e-16 1.7501e-16 \
  1.3199e-11 1.3199e-11 0 0 75420 0 /kinetics/dendgeom orange blue 11.397 \
  -5.7044 0
simundump kpool /kinetics/PKC/PKC-Ca-memb* 0 -1e-12 1.3896e-17 1.3896e-17 \
  1.048e-12 1.048e-12 0 0 75420 0 /kinetics/dendgeom pink blue 9.2212 -5.471 \
  0
simundump kpool /kinetics/PKC/PKC-DAG-memb* 0 -1e-12 9.4352e-21 9.4352e-21 \
  7.116e-16 7.116e-16 0 0 75420 0 /kinetics/dendgeom yellow blue 10.17 \
  -6.4922 0
simundump kpool /kinetics/PKC/PKC-basal* 0 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom pink blue 7.2535 -6.4338 0
simundump kreac /kinetics/PKC/PKC-basal-act 0 1 50 "" white blue 7.022 \
  -8.9429 0
simundump kpool /kinetics/PKC/PKC-AA* 0 -1e-12 1.8133e-17 1.8133e-17 \
  1.3676e-12 1.3676e-12 0 0 75420 0 /kinetics/dendgeom cyan blue 13.782 \
  -5.1793 0
simundump kreac /kinetics/PKC/PKC-act-by-AA 0 1.5911e-09 0.1 "" white blue \
  7.0075 -13.865 0
simundump kpool /kinetics/PKC/PKC-Ca-DAG 0 -1e-12 8.4631e-23 8.4631e-23 \
  6.3829e-18 6.3829e-18 0 0 75420 0 /kinetics/dendgeom white blue 12.231 \
  -10.197 0
simundump kreac /kinetics/PKC/PKC-n-DAG 0 7.9555e-09 0.1 "" white blue 8.9897 \
  -13.99 0
simundump kpool /kinetics/PKC/PKC-DAG 0 -1e-12 1.161e-16 1.161e-16 8.7563e-12 \
  8.7563e-12 0 0 75420 0 /kinetics/dendgeom white blue 11.004 -13.086 0
simundump kreac /kinetics/PKC/PKC-n-DAG-AA 0 2.3866e-07 2 "" white blue \
  10.772 -14.953 0
simundump kpool /kinetics/PKC/PKC-DAG-AA 0 -1e-12 2.5188e-19 2.5188e-19 \
  1.8997e-14 1.8997e-14 0 0 75420 0 /kinetics/dendgeom white blue 12.624 \
  -11.773 0
simundump kpool /kinetics/PKC/PKC-cytosolic 0 -1e-12 1 1 75420 75420 0 0 \
  75420 0 /kinetics/dendgeom white blue 5.8685 -11.403 0
simundump kpool /kinetics/AA 0 -1e-12 0 0 0 0 0 0 75420 0 /kinetics/dendgeom \
  darkgreen black 8.7102 -21.338 0
simundump group /kinetics/PLA2 0 darkgreen black x 0 1 "" defaultfile \
  defaultfile.g 0 0 0 4.6428 -26.209 0
simundump kpool /kinetics/PLA2/PLA2-cytosolic 0 -1e-12 0.4 0.4 30168 30168 0 \
  0 75420 0 /kinetics/dendgeom yellow darkgreen 0.176 -20.942 0
simundump kreac /kinetics/PLA2/PLA2-Ca-act 0 1.3259e-05 0.1 "" white \
  darkgreen 0.903 -23.104 0
simundump kpool /kinetics/PLA2/PLA2-Ca* 0 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom yellow darkgreen 3.278 -23.646 0
simundump kenz /kinetics/PLA2/PLA2-Ca*/kenz 0 0 0 0 0 75420 1.79e-05 21.6 5.4 \
  0 0 "" red yellow "" 5.9447 -23.667 0
simundump kreac /kinetics/PLA2/PIP2-PLA2-act 0 1.5911e-08 0.5 "" white \
  darkgreen 0.945 -18.75 0
simundump kpool /kinetics/PLA2/PIP2-PLA2* 0 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom cyan darkgreen 3.3197 -18.292 0
simundump kenz /kinetics/PLA2/PIP2-PLA2*/kenz 0 0 0 0 0 75420 3.6595e-05 \
  44.16 11.04 0 0 "" red cyan "" 5.9655 -18.271 0
simundump kreac /kinetics/PLA2/PIP2-Ca-PLA2-act 0 1.5911e-07 0.1 "" white \
  darkgreen 1.903 -19.5 0
simundump kpool /kinetics/PLA2/PIP2-Ca-PLA2* 0 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom cyan darkgreen 3.6739 -19.896 0
simundump kenz /kinetics/PLA2/PIP2-Ca-PLA2*/kenz 0 0 0 0 0 75420 0.00011933 \
  144 36 0 0 "" red cyan "" 6.028 -19.979 0
simundump kreac /kinetics/PLA2/DAG-Ca-PLA2-act 0 3.9777e-08 4 "" white \
  darkgreen 1.174 -21.834 0
simundump kpool /kinetics/PLA2/DAG-Ca-PLA2* 0 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom pink darkgreen 3.8614 -22.479 0
simundump kenz /kinetics/PLA2/DAG-Ca-PLA2*/kenz 0 0 0 0 0 75420 0.00019889 \
  240 60 0 0 "" red pink "" 6.0489 -22.354 0
simundump kpool /kinetics/PLA2/APC 0 -1e-12 30 30 2.2626e+06 2.2626e+06 0 0 \
  75420 4 /kinetics/dendgeom yellow darkgreen 3.7614 -21.963 0
simundump kreac /kinetics/PLA2/Degrade-AA 1 0.4 0 "" white darkgreen 5.8192 \
  -17.288 0
simundump kpool /kinetics/PLA2/PLA2*-Ca 0 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom orange darkgreen 4.187 -24.687 0
simundump kenz /kinetics/PLA2/PLA2*-Ca/kenz 0 0 0 0 0 75420 0.00039777 480 \
  120 0 0 "" red orange "" 5.9186 -24.817 0
simundump kpool /kinetics/PLA2/PLA2* 1 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom orange darkgreen 2.975 -26.851 0
simundump kreac /kinetics/PLA2/PLA2*-Ca-act 1 7.9555e-05 0.1 "" white \
  darkgreen 1.914 -24.752 0
simundump kreac /kinetics/PLA2/dephosphorylate-PLA2* 1 0.17 0 "" white \
  darkgreen -1.693 -23.735 0
simundump kpool /kinetics/temp-PIP2 1 -1e-12 2.5 2.5 1.8855e+05 1.8855e+05 0 \
  0 75420 4 /kinetics/dendgeom green black -3.796 -19.047 0
simundump group /kinetics/MAPK 0 brown black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 27 -1 0
simundump kpool /kinetics/MAPK/craf-1 0 0 0.5 0.5 37710 37710 0 0 75420 0 \
  /kinetics/dendgeom pink brown 18 -4 0
simundump kpool /kinetics/MAPK/craf-1* 0 0 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom pink brown 21 -4 0
simundump kpool /kinetics/MAPK/MAPKK 0 0 0.5 0.5 37710 37710 0 0 75420 0 \
  /kinetics/dendgeom pink brown 18 -8 0
simundump kpool /kinetics/MAPK/MAPK 0 -1e-12 3.6 3.6 2.7151e+05 2.7151e+05 0 \
  0 75420 0 /kinetics/dendgeom pink brown 18 -11 0
simundump kpool /kinetics/MAPK/craf-1** 1 0 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom hotpink brown 24 -4 0
simundump kpool /kinetics/MAPK/MAPK-tyr 1 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom orange brown 21 -11 0
simundump kpool /kinetics/MAPK/MAPKK* 0 0 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom pink brown 25 -8 0
simundump kenz /kinetics/MAPK/MAPKK*/MAPKKtyr 0 0 0 0 0 75420 0.00042959 1.2 \
  0.3 0 0 "" red pink "" 20 -9 0
simundump kenz /kinetics/MAPK/MAPKK*/MAPKKthr 1 0 0 0 0 75420 0.00042959 1.2 \
  0.3 0 0 "" red pink "" 25 -9 0
simundump kpool /kinetics/MAPK/MAPKK-ser 1 0 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom pink brown 21 -8 0
simundump kpool /kinetics/MAPK/Raf-GTP-Ras 0 0 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom 55 brown 17 -6 0
simundump kenz /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1 0 0 0 0 0 75420 \
  0.00012501 1.2 0.3 0 0 "" red 55 "" 20 -6 0
simundump kenz /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2 0 0 0 0 0 75420 \
  0.00012501 1.2 0.3 0 0 "" red 55 "" 23 -6 0
simundump kpool /kinetics/MAPK/Raf*-GTP-Ras 1 0 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom red brown 17 -7 0
simundump kenz /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1 1 0 0 0 0 0.1257 \
  0.00012501 1.2 0.3 0 0 "" red red "" 20 -7 0
simundump kenz /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2 1 0 0 0 0 0.1257 \
  0.00012501 1.2 0.3 0 0 "" red red "" 23 -7 0
simundump kreac /kinetics/Ras-act-craf 1 0.00013259 0.5 "" white black 15 -3 \
  0
simundump group /kinetics/Ras 1 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 26.513 4.351 0
simundump kreac /kinetics/Ras/dephosph-GEF 1 1 0 "" white blue 21.07 5.881 0
simundump kpool /kinetics/Ras/inact-GEF 1 -1e-12 0.1 0.1 7542 7542 0 0 75420 \
  0 /kinetics/dendgeom hotpink blue 24.453 6.352 0
simundump kenz /kinetics/Ras/inact-GEF/basal_GEF_activity 0 0 0 0 0 75420 \
  1.3126e-07 0.08 0.02 0 0 "" red hotpink "" 23 5 0
simundump kpool /kinetics/Ras/GEF* 1 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom hotpink blue 18.448 5.246 0
simundump kenz /kinetics/Ras/GEF*/GEF*-act-ras 1 0 0 0 0 75420 2.6253e-06 \
  0.08 0.02 0 0 "" red hotpink "" 19.085 4.086 0
simundump kpool /kinetics/Ras/GTP-Ras 1 0 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom orange blue 24.564 1.084 0
simundump kpool /kinetics/Ras/GDP-Ras 1 0 0.5 0.5 37710 37710 0 0 75420 0 \
  /kinetics/dendgeom pink blue 18.131 2.165 0
simundump kreac /kinetics/Ras/Ras-intrinsic-GTPase 1 0.0001 0 "" white blue \
  21.098 1.5 0
simundump kreac /kinetics/Ras/dephosph-GAP 1 0.1 0 "" white blue 16.023 3.524 \
  0
simundump kpool /kinetics/Ras/GAP* 1 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom red blue 13.35 2.349 0
simundump kpool /kinetics/Ras/GAP 1 -1e-12 0.01 0.01 754.2 754.2 0 0 75420 0 \
  /kinetics/dendgeom red blue 18.655 0.338 0
simundump kenz /kinetics/Ras/GAP/GAP-inact-ras 1 0 0 0 0 75420 0.00065613 40 \
  10 0 0 "" red red "" 21.012 0.403 0
simundump kreac /kinetics/Ras/CaM-bind-GEF 1 0.0026518 1 "" white blue 14.486 \
  9.679 0
simundump kpool /kinetics/Ras/CaM-GEF 1 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom pink blue 17.345 7.58 0
simundump kenz /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras 1 0 0 0 0 75420 \
  1.3127e-05 0.4 0.1 0 0 "" red pink "" 17.022 6.657 0
simundump kpool /kinetics/Ca 1 -1e-12 0.08 0.08 6033.6 6033.6 0 0 75420 0 \
  /kinetics/dendgeom red black -3 -13 0
simundump kpool /kinetics/DAG 1 -1e-12 12.2 12.2 9.2012e+05 9.2012e+05 0 0 \
  75420 4 /kinetics/dendgeom green black 12.718 -19.562 0
simundump kpool /kinetics/PKC-active 1 -1e-12 0.3 2.1196e-16 1.5986e-11 22626 \
  0 0 75420 2 /kinetics/dendgeom red black 12 -3 0
simundump kenz /kinetics/PKC-active/PKC-act-raf 1 0 0 0 0 75420 3.9777e-06 16 \
  4 0 0 "" red yellow "" 18 -2 0
simundump kenz /kinetics/PKC-active/PKC-inact-GAP 1 0 0 0 0 0.1257 7.9555e-05 \
  16 4 0 0 "" red yellow "" 16 1 0
simundump kenz /kinetics/PKC-active/PKC-act-GEF 1 0 0 0 0 0.1257 7.9555e-05 \
  16 4 0 0 "" red yellow "" 21 8 0
simundump kpool /kinetics/MAPK* 1 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom orange yellow 25 -11 0
simundump kenz /kinetics/MAPK*/MAPK* 0 0 0 0 0 75420 2.5856e-05 40 10 0 0 "" \
  red orange "" 0 -26 0
simundump kenz /kinetics/MAPK*/MAPK*-feedback 1 0 0 0 0 75420 2.5856e-05 40 \
  10 0 0 "" red orange "" 22 -2 0
simundump kpool /kinetics/IP3 1 -1e-12 0.73001 0.73001 55057 55057 0 0 75420 \
  4 /kinetics/dendgeom pink black 11.226 -16.656 0
simundump kpool /kinetics/MKP-1 1 -1e-12 0.024 0.024 1810.1 1810.1 0 0 75420 \
  0 /kinetics/dendgeom hotpink black 17 -10 0
simundump kenz /kinetics/MKP-1/MKP1-tyr-deph 1 0 0 0 0 75420 0.0019889 16 4 0 \
  0 "" red hotpink "" 18 -9 0
simundump kenz /kinetics/MKP-1/MKP1-thr-deph 1 0 0 0 0 75420 0.0019889 16 4 0 \
  0 "" red hotpink "" 23 -9 0
simundump kpool /kinetics/PPhosphatase2A 1 -1e-12 1 1 75420 75420 0 0 75420 0 \
  /kinetics/dendgeom hotpink yellow 21 -3 0
simundump kenz /kinetics/PPhosphatase2A/craf-deph 1 0 0 0 0 75420 2.5406e-05 \
  24 6 0 0 "" red hotpink "" 20 -2 0
simundump kenz /kinetics/PPhosphatase2A/MAPKK-deph 1 0 0 0 0 75420 2.5406e-05 \
  24 6 0 0 "" red hotpink "" 23 -5 0
simundump kenz /kinetics/PPhosphatase2A/MAPKK-deph-ser 1 0 0 0 0 75420 \
  2.5406e-05 24 6 0 0 "" red hotpink "" 20 -5 0
simundump kenz /kinetics/PPhosphatase2A/craf**-deph 1 0 0 0 0 0.1257 \
  2.5406e-05 24 6 0 0 "" red hotpink "" 24 -2 0
simundump group /kinetics/CaM 1 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 -1 -9 0
simundump kpool /kinetics/CaM/CaM 1 -1e-12 20 20 1.5084e+06 1.5084e+06 0 0 \
  75420 0 /kinetics/dendgeom pink blue -1.017 -1.28 0
simundump kreac /kinetics/CaM/CaM-TR2-bind-Ca 1 1.2658e-08 72 "" white blue \
  1.162 -1.921 0
simundump kreac /kinetics/CaM/CaM-TR2-Ca2-bind-Ca 1 4.7733e-05 10 "" white \
  blue 0.158 -3.7747 0
simundump kreac /kinetics/CaM/CaM-Ca3-bind-Ca 1 6.1655e-06 10 "" white blue \
  -1.4 -6.7404 0
simundump kpool /kinetics/CaM/CaM-Ca3 1 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom hotpink blue 2.489 -5.603 0
simundump kpool /kinetics/CaM/CaM-TR2-Ca2 1 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom pink blue 3.999 -2.7423 0
simundump kpool /kinetics/CaM/CaM-Ca4 1 -1e-12 0 0 0 0 0 0 75420 0 \
  /kinetics/dendgeom blue blue 4 -8 0
simundump kreac /kinetics/Ras-act-unphosph-raf 0 7.9555e-05 1 "" white black \
  16 -4 0
simundump kpool /kinetics/Ca_input 0 -1e-12 0.08 0.08 6033.6 6033.6 0 0 75420 \
  4 /kinetics/dendgeom 61 black -10 -7 0
simundump kreac /kinetics/Ca_diff 0 2 2 "" white black -9 -12 0
simundump doqcsinfo /kinetics/doqcsinfo 0 Ajay_Bhalla_2007_Bistable.g \
  Ajay_Bhalla_2007_Bistable network "Upinder S. Bhalla, NCBS" \
  "Upinder S. Bhalla, NCBS" "citation here" Rat "Hippocampal CA1" Dendrite \
  "Semi-Quantitative match to experiments" \
  "Ajay_Bhalla_bistable_model ( Under Review )" \
  "Exact GENESIS implementation" "Replicates original data " -1 7 0
simundump xgraph /graphs/conc1 0 0 5280.9 0 0.9 0
simundump xgraph /graphs/conc2 0 0 5280.9 0 40 0
simundump xplot /graphs/conc1/PKC-Ca.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" red 0 0 1
simundump xplot /graphs/conc2/PKC-active.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" red 0 0 1
simundump xgraph /moregraphs/conc3 0 0 5280.9 -1.1176e-08 1 0
simundump xgraph /moregraphs/conc4 0 0 5280.9 0 1 0
simundump xcoredraw /edit/draw 0 -12 29 -28.851 11.679
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
xtextload /file/notes \
"27 Feb 2006" \
"Version for testing Ca threshold of model. Modified with Ca_input" \
"to avoid sharp input transients." \
"28 Feb 2006: bis_PKC_MAPK_dend3.g" \
"Raised affinity of CaM and GEF 36-fold to account for Ca-input to" \
"MAPK cascade." \
"bis_PKC_MAPK_dend4.g: Set diffusion constants. Most are -1 um^2/s" \
"so as to indicate that they can be assigned to the same global" \
"value. Some are set to 0 as they are in a scaffold and probably" \
"do not diffuse much." \
"bis_PKC_MAPK_dend5.g: Speed up CaM-GEF-act-ras kcat 10x to 0.2/sec" \
"for faster MAPK response to Ca stimulus. This is in line with" \
"the PKM model. Also altered CaM affinity for CaM-GEF-act-ras" \
"downward, again in line with PKM model." \
""
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
addmsg /kinetics/PKC/PKC-act-by-Ca-AA /kinetics/AA REAC A B 
addmsg /kinetics/PKC/PKC-act-by-AA /kinetics/AA REAC A B 
addmsg /kinetics/PKC/PKC-n-DAG-AA /kinetics/AA REAC A B 
addmsg /kinetics/PLA2/PLA2-Ca*/kenz /kinetics/AA MM_PRD pA 
addmsg /kinetics/PLA2/PIP2-PLA2*/kenz /kinetics/AA MM_PRD pA 
addmsg /kinetics/PLA2/PIP2-Ca-PLA2*/kenz /kinetics/AA MM_PRD pA 
addmsg /kinetics/PLA2/DAG-Ca-PLA2*/kenz /kinetics/AA MM_PRD pA 
addmsg /kinetics/PLA2/PLA2*-Ca/kenz /kinetics/AA MM_PRD pA 
addmsg /kinetics/PLA2/Degrade-AA /kinetics/AA REAC A B 
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
addmsg /kinetics/PLA2/PIP2-PLA2-act /kinetics/temp-PIP2 REAC A B 
addmsg /kinetics/PLA2/PIP2-Ca-PLA2-act /kinetics/temp-PIP2 REAC A B 
addmsg /kinetics/PKC-active/PKC-act-raf /kinetics/MAPK/craf-1 REAC sA B 
addmsg /kinetics/PPhosphatase2A/craf-deph /kinetics/MAPK/craf-1 MM_PRD pA 
addmsg /kinetics/Ras-act-unphosph-raf /kinetics/MAPK/craf-1 REAC A B 
addmsg /kinetics/PKC-active/PKC-act-raf /kinetics/MAPK/craf-1* MM_PRD pA 
addmsg /kinetics/MAPK*/MAPK*-feedback /kinetics/MAPK/craf-1* REAC sA B 
addmsg /kinetics/PPhosphatase2A/craf-deph /kinetics/MAPK/craf-1* REAC sA B 
addmsg /kinetics/PPhosphatase2A/craf**-deph /kinetics/MAPK/craf-1* MM_PRD pA 
addmsg /kinetics/Ras-act-craf /kinetics/MAPK/craf-1* REAC A B 
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
addmsg /kinetics/Ras-act-unphosph-raf /kinetics/MAPK/Raf-GTP-Ras REAC B A 
addmsg /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1 /kinetics/MAPK/Raf-GTP-Ras REAC eA B 
addmsg /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2 /kinetics/MAPK/Raf-GTP-Ras REAC eA B 
addmsg /kinetics/MAPK/Raf-GTP-Ras /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1 ENZYME n 
addmsg /kinetics/MAPK/MAPKK /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1 SUBSTRATE n 
addmsg /kinetics/MAPK/Raf-GTP-Ras /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2 ENZYME n 
addmsg /kinetics/MAPK/MAPKK-ser /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2 SUBSTRATE n 
addmsg /kinetics/Ras-act-craf /kinetics/MAPK/Raf*-GTP-Ras REAC B A 
addmsg /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1 /kinetics/MAPK/Raf*-GTP-Ras REAC eA B 
addmsg /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2 /kinetics/MAPK/Raf*-GTP-Ras REAC eA B 
addmsg /kinetics/MAPK/Raf*-GTP-Ras /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1 ENZYME n 
addmsg /kinetics/MAPK/MAPKK /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1 SUBSTRATE n 
addmsg /kinetics/MAPK/Raf*-GTP-Ras /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2 ENZYME n 
addmsg /kinetics/MAPK/MAPKK-ser /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2 SUBSTRATE n 
addmsg /kinetics/MAPK/craf-1* /kinetics/Ras-act-craf SUBSTRATE n 
addmsg /kinetics/Ras/GTP-Ras /kinetics/Ras-act-craf SUBSTRATE n 
addmsg /kinetics/MAPK/Raf*-GTP-Ras /kinetics/Ras-act-craf PRODUCT n 
addmsg /kinetics/Ras/GEF* /kinetics/Ras/dephosph-GEF SUBSTRATE n 
addmsg /kinetics/Ras/inact-GEF /kinetics/Ras/dephosph-GEF PRODUCT n 
addmsg /kinetics/PKC-active/PKC-act-GEF /kinetics/Ras/inact-GEF REAC sA B 
addmsg /kinetics/Ras/dephosph-GEF /kinetics/Ras/inact-GEF REAC B A 
addmsg /kinetics/Ras/CaM-bind-GEF /kinetics/Ras/inact-GEF REAC A B 
addmsg /kinetics/Ras/inact-GEF/basal_GEF_activity /kinetics/Ras/inact-GEF REAC eA B 
addmsg /kinetics/Ras/inact-GEF /kinetics/Ras/inact-GEF/basal_GEF_activity ENZYME n 
addmsg /kinetics/Ras/GDP-Ras /kinetics/Ras/inact-GEF/basal_GEF_activity SUBSTRATE n 
addmsg /kinetics/PKC-active/PKC-act-GEF /kinetics/Ras/GEF* MM_PRD pA 
addmsg /kinetics/Ras/dephosph-GEF /kinetics/Ras/GEF* REAC A B 
addmsg /kinetics/Ras/GEF*/GEF*-act-ras /kinetics/Ras/GEF* REAC eA B 
addmsg /kinetics/Ras/GEF* /kinetics/Ras/GEF*/GEF*-act-ras ENZYME n 
addmsg /kinetics/Ras/GDP-Ras /kinetics/Ras/GEF*/GEF*-act-ras SUBSTRATE n 
addmsg /kinetics/Ras/GAP/GAP-inact-ras /kinetics/Ras/GTP-Ras REAC sA B 
addmsg /kinetics/Ras/Ras-intrinsic-GTPase /kinetics/Ras/GTP-Ras REAC A B 
addmsg /kinetics/Ras/GEF*/GEF*-act-ras /kinetics/Ras/GTP-Ras MM_PRD pA 
addmsg /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras /kinetics/Ras/GTP-Ras MM_PRD pA 
addmsg /kinetics/Ras-act-craf /kinetics/Ras/GTP-Ras REAC A B 
addmsg /kinetics/Ras-act-unphosph-raf /kinetics/Ras/GTP-Ras REAC A B 
addmsg /kinetics/Ras/inact-GEF/basal_GEF_activity /kinetics/Ras/GTP-Ras MM_PRD pA 
addmsg /kinetics/Ras/GAP/GAP-inact-ras /kinetics/Ras/GDP-Ras MM_PRD pA 
addmsg /kinetics/Ras/Ras-intrinsic-GTPase /kinetics/Ras/GDP-Ras REAC B A 
addmsg /kinetics/Ras/GEF*/GEF*-act-ras /kinetics/Ras/GDP-Ras REAC sA B 
addmsg /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras /kinetics/Ras/GDP-Ras REAC sA B 
addmsg /kinetics/Ras/inact-GEF/basal_GEF_activity /kinetics/Ras/GDP-Ras REAC sA B 
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
addmsg /kinetics/Ras/inact-GEF /kinetics/Ras/CaM-bind-GEF SUBSTRATE n 
addmsg /kinetics/Ras/CaM-GEF /kinetics/Ras/CaM-bind-GEF PRODUCT n 
addmsg /kinetics/CaM/CaM-Ca4 /kinetics/Ras/CaM-bind-GEF SUBSTRATE n 
addmsg /kinetics/Ras/CaM-bind-GEF /kinetics/Ras/CaM-GEF REAC B A 
addmsg /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras /kinetics/Ras/CaM-GEF REAC eA B 
addmsg /kinetics/Ras/CaM-GEF /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras ENZYME n 
addmsg /kinetics/Ras/GDP-Ras /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras SUBSTRATE n 
addmsg /kinetics/PKC/PKC-act-by-Ca /kinetics/Ca REAC A B 
addmsg /kinetics/PLA2/PLA2-Ca-act /kinetics/Ca REAC A B 
addmsg /kinetics/PLA2/PLA2*-Ca-act /kinetics/Ca REAC A B 
addmsg /kinetics/CaM/CaM-TR2-bind-Ca /kinetics/Ca REAC A B 
addmsg /kinetics/CaM/CaM-TR2-bind-Ca /kinetics/Ca REAC A B 
addmsg /kinetics/CaM/CaM-TR2-Ca2-bind-Ca /kinetics/Ca REAC A B 
addmsg /kinetics/CaM/CaM-Ca3-bind-Ca /kinetics/Ca REAC A B 
addmsg /kinetics/Ca_diff /kinetics/Ca REAC B A 
addmsg /kinetics/PKC/PKC-act-by-DAG /kinetics/DAG REAC A B 
addmsg /kinetics/PKC/PKC-n-DAG /kinetics/DAG REAC A B 
addmsg /kinetics/PLA2/DAG-Ca-PLA2-act /kinetics/DAG REAC A B 
addmsg /kinetics/PKC/PKC-DAG-AA* /kinetics/PKC-active SUMTOTAL n nInit 
addmsg /kinetics/PKC/PKC-Ca-memb* /kinetics/PKC-active SUMTOTAL n nInit 
addmsg /kinetics/PKC/PKC-Ca-AA* /kinetics/PKC-active SUMTOTAL n nInit 
addmsg /kinetics/PKC/PKC-DAG-memb* /kinetics/PKC-active SUMTOTAL n nInit 
addmsg /kinetics/PKC/PKC-basal* /kinetics/PKC-active SUMTOTAL n nInit 
addmsg /kinetics/PKC/PKC-AA* /kinetics/PKC-active SUMTOTAL n nInit 
addmsg /kinetics/PKC-active/PKC-act-raf /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active/PKC-inact-GAP /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active/PKC-act-GEF /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-act-raf ENZYME n 
addmsg /kinetics/MAPK/craf-1 /kinetics/PKC-active/PKC-act-raf SUBSTRATE n 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-inact-GAP ENZYME n 
addmsg /kinetics/Ras/GAP /kinetics/PKC-active/PKC-inact-GAP SUBSTRATE n 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-act-GEF ENZYME n 
addmsg /kinetics/Ras/inact-GEF /kinetics/PKC-active/PKC-act-GEF SUBSTRATE n 
addmsg /kinetics/MAPK*/MAPK* /kinetics/MAPK* REAC eA B 
addmsg /kinetics/MAPK*/MAPK*-feedback /kinetics/MAPK* REAC eA B 
addmsg /kinetics/MAPK/MAPKK*/MAPKKthr /kinetics/MAPK* MM_PRD pA 
addmsg /kinetics/MKP-1/MKP1-thr-deph /kinetics/MAPK* REAC sA B 
addmsg /kinetics/MAPK* /kinetics/MAPK*/MAPK* ENZYME n 
addmsg /kinetics/PLA2/PLA2-cytosolic /kinetics/MAPK*/MAPK* SUBSTRATE n 
addmsg /kinetics/MAPK* /kinetics/MAPK*/MAPK*-feedback ENZYME n 
addmsg /kinetics/MAPK/craf-1* /kinetics/MAPK*/MAPK*-feedback SUBSTRATE n 
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
addmsg /kinetics/CaM/CaM-Ca4 /kinetics/CaM/CaM-Ca3-bind-Ca PRODUCT n 
addmsg /kinetics/CaM/CaM-TR2-Ca2-bind-Ca /kinetics/CaM/CaM-Ca3 REAC B A 
addmsg /kinetics/CaM/CaM-Ca3-bind-Ca /kinetics/CaM/CaM-Ca3 REAC A B 
addmsg /kinetics/CaM/CaM-TR2-bind-Ca /kinetics/CaM/CaM-TR2-Ca2 REAC B A 
addmsg /kinetics/CaM/CaM-TR2-Ca2-bind-Ca /kinetics/CaM/CaM-TR2-Ca2 REAC A B 
addmsg /kinetics/Ras/CaM-bind-GEF /kinetics/CaM/CaM-Ca4 REAC A B 
addmsg /kinetics/CaM/CaM-Ca3-bind-Ca /kinetics/CaM/CaM-Ca4 REAC B A 
addmsg /kinetics/MAPK/craf-1 /kinetics/Ras-act-unphosph-raf SUBSTRATE n 
addmsg /kinetics/MAPK/Raf-GTP-Ras /kinetics/Ras-act-unphosph-raf PRODUCT n 
addmsg /kinetics/Ras/GTP-Ras /kinetics/Ras-act-unphosph-raf SUBSTRATE n 
addmsg /kinetics/Ca_diff /kinetics/Ca_input REAC A B 
addmsg /kinetics/Ca_input /kinetics/Ca_diff SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/Ca_diff PRODUCT n 
addmsg /kinetics/PKC/PKC-Ca /graphs/conc1/PKC-Ca.Co PLOT Co *PKC-Ca.Co *red 
addmsg /kinetics/PKC-active /graphs/conc2/PKC-active.Co PLOT Co *PKC-active.Co *red 
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
call /kinetics/temp-PIP2/notes LOAD \
"This isn't explicitly present in the M&L model, but is obviously needed." \
"I assume its conc is fixed at 1uM for now, which is a bit high. PLA2 is stim" \
"7x by PIP2 @ 0.5 uM (Leslie and Channon BBA 1045:261(1990) " \
"Leslie and Channon say PIP2 is present at 0.1 - 0.2mol% range in membs," \
"which comes to 50 nM. Ref is Majerus et al Cell 37 pp 701-703 1984" \
"Lets use a lower level of 30 nM, same ref...."
call /kinetics/MAPK/craf-1/notes LOAD \
"Couldn't find any ref to the actual conc of craf-1 but I" \
"should try Strom et al Oncogene 5 pp 345" \
"In line with the other kinases in the cascade, I estimate the conc to be" \
"0.2 uM. To init we use 0.15, which is close to equil" \
"16 May 2003: Changing to synaptic levels. Increasing 2.5 fold to 0.5 uM." \
"See Mihaly et al 1991 Brain Res 547(2):309-14" \
"and " \
"Morice et al 1999 Eur J Neurosci 11(6):1995-2006"
call /kinetics/MAPK/MAPKK/notes LOAD \
"Conc is from Seger et al JBC 267:20 pp14373 (1992)" \
"mwt is 45/46 Kd" \
"We assume that phosphorylation on both ser and thr is needed for" \
"activiation. See Kyriakis et al Nature 358 417 1992" \
"Init conc of total is 0.18" \
"Ortiz et al 1995 J Neurosci 15(2):1285-1297 suggest that levels are" \
"higher in hippocampus than other brain regions, and further elevated " \
"in synapses. Estimate 3x higher levels than before, at 0.5 uM." \
"Similar results from Schipper et al 1999 Neuroscience 93(2):585-595" \
"but again lacking in quantitation."
call /kinetics/MAPK/MAPK/notes LOAD \
"conc is from Sanghera et al JBC 265 pp 52 (1990)" \
"A second calculation gives 3.1 uM, from same paper." \
"They est MAPK is 1e-4x total protein, and protein is 15% of cell wt," \
"so MAPK is 1.5e-5g/ml = 0.36uM. which is closer to our first estimate." \
"Lets use this." \
"Updated 16 May 2003." \
"Ortiz et al 1995 J Neurosci 15(2):1285-1297 provide estimates of " \
"ERK2 levels in hippocampus: 1009 ng/mg. This comes to about 3.6uM, which" \
"may still be an underestimate of synaptic levels."
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
"Kinetics are the same as for the craf_1* activity, ie., " \
"k1=5.5e-6, k2=0.42, k3 = 0.105" \
"These are basedo n Force et al PNAS USA 91 1270-1274, 1994.," \
"but k1 is scaled up 5x (ie., Km is scaled down 5x to the value used here" \
"and for craf_1* activity: Km = 0.1591)."
call /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2/notes LOAD \
"Kinetics are the same as for the craf_1* activity, ie., " \
"k1=5.5e-6, k2=0.42, k3 = 0.105" \
"These are basedo n Force et al PNAS USA 91 1270-1274, 1994.," \
"but k1 is scaled up 5x (ie., Km is scaled down 5x to the value used here" \
"and for craf_1* activity: Km = 0.1591)."
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
call /kinetics/Ras-act-craf/notes LOAD \
"Assume the binding is fast and limited only by the amount of " \
"Ras* available. So kf=kb/[craf-1]" \
"If kb is 1/sec, then kf = 1/0.2 uM = 1/(0.2 * 6e5) = 8.3e-6" \
"Later: Raise it by 10 X to 4e-5" \
"From Hallberg et al JBC 269:6 3913-3916 1994, 3% of cellular Raf is" \
"complexed with Ras. So we raise kb 4x to 4" \
"This step needed to memb-anchor and activate Raf: Leevers et al Nature" \
"369 411-414" \
"May 16, 2003" \
"Changed Ras and Raf to synaptic levels, an increase of about 2x for each." \
"To maintain the percentage of complexed Raf, reduced the kf by 2.4 fold" \
"to 10." \
""
call /kinetics/Ras/notes LOAD \
"Ras has now gotten to be a big enough component of the model to" \
"deserve its own group. The main refs are" \
"Boguski and McCormick Nature 366 643-654 '93 Major review" \
"Eccleston et al JBC 268:36 pp 27012-19" \
"Orita et al JBC 268:34 2554246"
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
"Level=.002" \
"16 May 2003: Increased level to 0.0036, in line with other concentration" \
"raises at the synapse."
call /kinetics/Ras/GAP/GAP-inact-ras/notes LOAD \
"From Eccleston et al JBC 268(36)pp27012-19 get Kd < 2uM, kcat - 10/sec" \
"From Martin et al Cell 63 843-849 1990 get Kd ~ 250 nM, kcat = 20/min" \
"I will go with the Eccleston figures as there are good error bars (10%). In general" \
"the values are reasonably close." \
"k1 = 1.666e-3/sec, k2 = 1000/sec, k3 = 10/sec (note k3 is rate-limiting)" \
"5 Nov 2002: Changed ratio term to 4 from 100. Now we have" \
"k1=8.25e-5; k2=40, k3=10. k3 is still rate-limiting."
call /kinetics/Ras/CaM-bind-GEF/notes LOAD \
"We have no numbers for this. It is probably between" \
"the two extremes represented by the CaMKII phosph states," \
"and I have used guesses based on this." \
"kf=1e-4" \
"kb=1" \
"The reaction is based on Farnsworth et al Nature 376 524-527" \
"1995" \
"28 Feb 2006: Increased affinity 36-fold to account for Ca" \
"input to MAPK cascade, possibly folding in other pathway" \
"inputs." \
"21 April 2006: Altered affinity to same level as " \
"pkm_mapk21.g model to prevent spontaneous turnon." \
"Kf = 200, Kb = 1." \
"" \
"" \
""
call /kinetics/Ras/CaM-GEF/notes LOAD \
"See Farnsworth et al Nature 376 524-527 1995"
call /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras/notes LOAD \
"Kinetics same as GEF-bg_act-ras" \
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
call /kinetics/IP3/notes LOAD \
"Peak IP3 is perhaps 15 uM, basal <= 0.2 uM."
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
"kcat = 1.43 umol/min/mg"
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
call /kinetics/Ras-act-unphosph-raf/notes LOAD \
"18 May 2003. This reaction is here to" \
"provide basal activity for MAPK as well" \
"as the potential for direct EGF stimulus" \
"without PKC activation." \
"Based on model from FB/fb28c.g: the model" \
"used for MKP-1 turnover. The rates there" \
"were constrained by basal activity values."
call /kinetics/doqcsinfo/notes LOAD \
"This is a model of ERKII signaling which is bistable due to feedback." \
"The feedback occurs through ERKII phosphorylation of phospholipase A2 (PLA2), leading to increased production of arachidonic acid (AA), which activates protein kinase C (PKC) which activates c-Raf which is upstream of ERKII.<br> " \
"The model is a highly simplified variant of more detailed bistable models of MAPK signaling (<a href=http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?db=pubmed&cmd=Retrieve&dopt=AbstractPlus&list_uids=9888852&query_hl=4&itool=pubmed_docsum>Bhalla US, Iyengar R. Science. 1999 Jan 15;283(5400):381-7</a>, <a href=http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?db=pubmed&cmd=Retrieve&dopt=AbstractPlus&list_uids=15548210&query_hl=2&itool=pubmed_docsum>Ajay SM, Bhalla US. Eur J Neurosci. 2004 Nov;20(10):2671-80</a>)"
complete_loading
