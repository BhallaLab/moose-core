//  DOQCS : http://doqcs.ncbs.res.in/ 
//  Accession Name = 3d_fold_model 
//  Accession Number = 8 
//  Transcriber = Upinder S. Bhalla, NCBS 
//  Developer = Upinder S. Bhalla, NCBS 
//  Species = Generic mammalian 
//  Tissue = NIH 3T3 Expression 
//  Cell Compartment = Surface - Nucleus 
//  Notes =  This model is an annotated version of the synaptic signaling network.<br>The primary reference is <a href= http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?cmd=Retrieve&db=PubMed&list_uids=9888852&dopt=Abstract >Bhalla US and Iyengar R. Science (1999) 283(5400):381-7</a> but several of the model pathways have been updated <a href =http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?cmd=Retrieve&db=pubmed&dopt=Abstract&list_uids=12169734&query_hl=2>Bhalla US, Ram PT, Iyengar R. Science. 2002 Aug 9;297(5583):1018-23</a>. 
 
 //genesis
// kkit Version 11 flat dumpfile
 
// Saved on Thu Dec  8 14:02:10 2005
 
include kkit {argv 1}
 
FASTDT = 0.005
SIMDT = 0.005
CONTROLDT = 10
PLOTDT = 10
MAXTIME = 4000
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
simundump geometry /kinetics/geometry 0 1e-15 3 sphere "" white black 15 21 0
simundump geometry /kinetics/geometry[1] 0 1e-15 3 sphere "" white black 0 0 \
  0
simundump group /kinetics/PKC 0 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 -3.0493 8.2163 0
simundump kpool /kinetics/PKC/PKC-Ca 0 0 3.7208e-17 3.7208e-17 2.2325e-11 \
  2.2325e-11 0 0 6e+05 0 /kinetics/geometry[1] red black -4.0752 1.5108 0
simundump kreac /kinetics/PKC/PKC-act-by-Ca 0 1e-06 0.5 "" white blue -4.0752 \
  -0.12295 0
simundump kreac /kinetics/PKC/PKC-act-by-DAG 0 1.3333e-08 8.6348 "" white \
  blue -2.0612 0.69395 0
simundump kreac /kinetics/PKC/PKC-Ca-to-memb 0 1.2705 3.5026 "" white blue \
  -3.7974 4.2533 0
simundump kreac /kinetics/PKC/PKC-DAG-to-memb 0 1 0.1 "" white blue -2.6168 \
  2.7362 0
simundump kreac /kinetics/PKC/PKC-act-by-Ca-AA 0 2e-09 0.1 "" white blue \
  -0.78797 3.8157 0
simundump kreac /kinetics/PKC/PKC-act-by-DAG-AA 0 2 0.2 "" white blue 1.2492 \
  3.2322 0
simundump kpool /kinetics/PKC/PKC-DAG-AA* 0 0 4.9137e-18 4.9137e-18 \
  2.9482e-12 2.9482e-12 0 0 6e+05 0 /kinetics/geometry[1] cyan blue 0.60098 \
  5.537 0
simundump kpool /kinetics/PKC/PKC-Ca-AA* 0 0 1.75e-16 1.75e-16 1.05e-10 \
  1.05e-10 0 0 6e+05 0 /kinetics/geometry[1] orange blue -0.60278 6.2956 0
simundump kpool /kinetics/PKC/PKC-Ca-memb* 0 0 1.3896e-17 1.3896e-17 \
  8.3376e-12 8.3376e-12 0 0 6e+05 0 /kinetics/geometry[1] pink blue -2.7788 \
  6.529 0
simundump kpool /kinetics/PKC/PKC-DAG-memb* 0 0 9.4352e-21 9.4352e-21 \
  5.6611e-15 5.6611e-15 0 0 6e+05 0 /kinetics/geometry[1] yellow blue -1.8297 \
  5.5078 0
simundump kpool /kinetics/PKC/PKC-basal* 0 0 0.02 0.02 12000 12000 0 0 6e+05 \
  0 /kinetics/geometry[1] pink blue -4.7465 5.5662 0
simundump kreac /kinetics/PKC/PKC-basal-act 0 1 50 "" white blue -4.978 \
  3.0571 0
simundump kpool /kinetics/PKC/PKC-AA* 0 0 1.8133e-17 1.8133e-17 1.088e-11 \
  1.088e-11 0 0 6e+05 0 /kinetics/geometry[1] cyan blue 1.7816 6.8207 0
simundump kreac /kinetics/PKC/PKC-act-by-AA 0 2e-10 0.1 "" white blue -4.9925 \
  -1.8654 0
simundump kpool /kinetics/PKC/PKC-Ca-DAG 0 0 8.4632e-23 8.4632e-23 5.0779e-17 \
  5.0779e-17 0 0 6e+05 0 /kinetics/geometry[1] white blue 0.2306 1.8026 0
simundump kreac /kinetics/PKC/PKC-n-DAG 0 1e-09 0.1 "" white blue -3.0103 \
  -1.9902 0
simundump kpool /kinetics/PKC/PKC-DAG 0 0 1.161e-16 1.161e-16 6.9661e-11 \
  6.9661e-11 0 0 6e+05 0 /kinetics/geometry[1] white blue 3 -5 0
simundump kreac /kinetics/PKC/PKC-n-DAG-AA 0 3e-08 2 "" white blue -1.2278 \
  -2.9529 0
simundump kpool /kinetics/PKC/PKC-DAG-AA 0 0 2.5188e-19 2.5188e-19 1.5113e-13 \
  1.5113e-13 0 0 6e+05 0 /kinetics/geometry[1] white blue 0.62413 0.22715 0
simundump kpool /kinetics/PKC/PKC-cytosolic 0 0 1 1 6e+05 6e+05 0 0 6e+05 0 \
  /kinetics/geometry[1] white blue -6.1315 0.59711 0
simundump kpool /kinetics/DAG 1 0 11.661 11.661 6.9966e+06 6.9966e+06 0 0 \
  6e+05 6 /kinetics/geometry[1] green black -3.5151 -4.3314 0
simundump kpool /kinetics/Ca 1 0 0.08 0.08 48000 48000 0 0 6e+05 6 \
  /kinetics/geometry[1] red black -8.3874 -2.7634 0
simundump kpool /kinetics/AA 0 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry[1] \
  darkgreen black -3.2898 -9.3376 0
simundump kpool /kinetics/PKC-active 0 0 0.02 0.02 12000 1.2733e-10 0 0 6e+05 \
  2 /kinetics/geometry[1] yellow black 2.1325 8.477 0
simundump kenz /kinetics/PKC-active/PKC-act-raf 1 0 0 0 0 6e+05 5e-07 16 4 0 \
  0 "" red yellow "" 6.2532 10.549 0
simundump kenz /kinetics/PKC-active/PKC-inact-GAP 1 0 0 0 0 1 1e-05 16 4 0 0 \
  "" red yellow "" 3.4391 11.804 0
simundump kenz /kinetics/PKC-active/PKC-act-GEF 1 0 0 0 0 1 1e-05 16 4 0 0 "" \
  red yellow "" -0.24791 17.264 0
simundump group /kinetics/MAPK 0 brown black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 14.616 11.191 0
simundump kpool /kinetics/MAPK/craf-1 0 0 0.2 0.2 1.2e+05 1.2e+05 0 0 6e+05 0 \
  /kinetics/geometry[1] pink brown 6.326 8.1168 0
simundump kpool /kinetics/MAPK/craf-1* 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] pink brown 9.2401 7.7115 0
simundump kpool /kinetics/MAPK/MAPKK 0 0 0.18 0.18 1.08e+05 1.08e+05 0 0 \
  6e+05 0 /kinetics/geometry[1] pink brown 6.3315 3.9894 0
simundump kpool /kinetics/MAPK/MAPK 0 0 0.36 0.36 2.16e+05 2.16e+05 0 0 6e+05 \
  0 /kinetics/geometry[1] pink brown 6.0656 1.0863 0
simundump kpool /kinetics/MAPK/craf-1** 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] hotpink brown 12.464 7.9022 0
simundump kpool /kinetics/MAPK/MAPK-tyr 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] orange brown 8.4147 0.82034 0
simundump kpool /kinetics/MAPK/MAPKK* 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] pink brown 12.548 4.0256 0
simundump kenz /kinetics/MAPK/MAPKK*/MAPKKtyr 0 0 0 0 0 6e+05 2.7e-05 0.6 \
  0.15 0 0 "" red pink "" 8.8914 3.5531 0
simundump kenz /kinetics/MAPK/MAPKK*/MAPKKthr 1 0 0 0 0 6e+05 2.7e-05 0.6 \
  0.15 0 0 "" red pink "" 12.961 3.0363 0
simundump kpool /kinetics/MAPK/MAPKK-ser 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] pink brown 9.2652 4.1657 0
simundump kpool /kinetics/MAPK/Raf-GTP-Ras* 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] red brown 4.9054 6.7814 0
simundump kenz /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.1 1 0 0 0 0 1 5.5e-06 \
  0.42 0.105 0 0 "" red red "" 7.6179 6.2189 0
simundump kenz /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.2 1 0 0 0 0 1 5.5e-06 \
  0.42 0.105 0 0 "" red red "" 10.698 6.0688 0
simundump kpool /kinetics/MAPK* 0 0 0 0 0 0 0 0 6e+05 2 /kinetics/geometry[1] \
  orange yellow 13.735 0.41378 0
simundump kenz /kinetics/MAPK*/MAPK*-feedback 1 0 0 0 0 6e+05 3.25e-06 40 10 \
  0 0 "" red orange "" 10.387 10.668 0
simundump kenz /kinetics/MAPK*/MAPK* 0 0 0 0 0 6e+05 6.5e-06 80 20 0 0 "" red \
  orange "" -12.005 -14.94 0
simundump kpool /kinetics/MKP-1 1 0 0.0032 0.0032 1920 1920 0 0 6e+05 0 \
  /kinetics/geometry[1] hotpink black 5.0816 2.4407 0
simundump kenz /kinetics/MKP-1/MKP1-tyr-deph 1 0 0 0 0 6e+05 0.000125 4 1 0 0 \
  "" red hotpink "" 6.2781 3.0684 0
simundump kenz /kinetics/MKP-1/MKP1-thr-deph 1 0 0 0 0 6e+05 0.000125 4 1 0 0 \
  "" red hotpink "" 10.789 2.9311 0
simundump kreac /kinetics/Ras-act-craf 1 4e-05 0.5 "" white black 3.5614 \
  10.091 0
simundump kpool /kinetics/PPhosphatase2A 1 0 0.224 0.224 1.344e+05 1.344e+05 \
  0 0 6e+05 0 /kinetics/geometry[1] hotpink yellow 9.3898 9.1309 0
simundump kenz /kinetics/PPhosphatase2A/craf-deph 1 0 0 0 0 6e+05 3.3e-06 25 \
  6 0 0 "" red hotpink "" 7.8013 10.215 0
simundump kenz /kinetics/PPhosphatase2A/MAPKK-deph 1 0 0 0 0 6e+05 3.3e-06 25 \
  6 0 0 "" red hotpink "" 13.159 6.0736 0
simundump kenz /kinetics/PPhosphatase2A/MAPKK-deph-ser 1 0 0 0 0 6e+05 \
  3.3e-06 25 6 0 0 "" red hotpink "" 4.8651 5.9208 0
simundump kenz /kinetics/PPhosphatase2A/craf**-deph 1 0 0 0 0 1 3.3e-06 25 6 \
  0 0 "" red hotpink "" 12.446 9.9054 0
simundump group /kinetics/PLA2 0 darkgreen black x 0 1 "" defaultfile \
  defaultfile.g 0 0 0 -7.3572 -14.209 0
simundump kpool /kinetics/PLA2/PLA2-cytosolic 0 0 0.4 0.4 2.4e+05 2.4e+05 0 0 \
  6e+05 0 /kinetics/geometry[1] yellow darkgreen -11.824 -8.9421 0
simundump kreac /kinetics/PLA2/PLA2-Ca-act 0 1.6667e-06 0.1 "" white \
  darkgreen -11.097 -11.104 0
simundump kpool /kinetics/PLA2/PLA2-Ca* 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] yellow darkgreen -8.722 -11.646 0
simundump kenz /kinetics/PLA2/PLA2-Ca*/kenz 0 0 0 0 0 6e+05 2.25e-06 21.6 5.4 \
  0 0 "" red yellow "" -6.0553 -11.667 0
simundump kreac /kinetics/PLA2/PIP2-PLA2-act 0 2e-09 0.5 "" white darkgreen \
  -11.055 -6.7502 0
simundump kpool /kinetics/PLA2/PIP2-PLA2* 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] cyan darkgreen -8.6803 -6.2919 0
simundump kenz /kinetics/PLA2/PIP2-PLA2*/kenz 0 0 0 0 0 6e+05 4.6e-06 44.16 \
  11.04 0 0 "" red cyan "" -6.0345 -6.271 0
simundump kreac /kinetics/PLA2/PIP2-Ca-PLA2-act 0 2e-08 0.1 "" white \
  darkgreen -10.097 -7.5002 0
simundump kpool /kinetics/PLA2/PIP2-Ca-PLA2* 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] cyan darkgreen -8.3261 -7.896 0
simundump kenz /kinetics/PLA2/PIP2-Ca-PLA2*/kenz 0 0 0 0 0 6e+05 1.5e-05 144 \
  36 0 0 "" red cyan "" -5.972 -7.9794 0
simundump kreac /kinetics/PLA2/DAG-Ca-PLA2-act 0 5e-09 4 "" white darkgreen \
  -10.826 -9.8336 0
simundump kpool /kinetics/PLA2/DAG-Ca-PLA2* 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] pink darkgreen -8.1386 -10.479 0
simundump kenz /kinetics/PLA2/DAG-Ca-PLA2*/kenz 0 0 0 0 0 6e+05 2.5e-05 240 \
  60 0 0 "" red pink "" -5.9511 -10.354 0
simundump kpool /kinetics/PLA2/APC 0 0 30 30 1.8e+07 1.8e+07 0 0 6e+05 5 \
  /kinetics/geometry[1] yellow darkgreen -8.2386 -9.9634 0
simundump kreac /kinetics/PLA2/Degrade-AA 1 0.4 0 "" white darkgreen -6.1808 \
  -5.2875 0
simundump kpool /kinetics/PLA2/PLA2*-Ca 0 0 0 0 0 0 0 0 6e+05 1 \
  /kinetics/geometry[1] orange darkgreen -7.813 -12.687 0
simundump kenz /kinetics/PLA2/PLA2*-Ca/kenz 0 0 0 0 0 6e+05 5e-05 480 120 0 0 \
  "" red orange "" -6.0814 -12.817 0
simundump kpool /kinetics/PLA2/PLA2* 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] orange darkgreen -9.025 -14.851 0
simundump kreac /kinetics/PLA2/PLA2*-Ca-act 1 1e-05 0.1 "" white darkgreen \
  -10.086 -12.752 0
simundump kreac /kinetics/PLA2/dephosphorylate-PLA2* 1 0.17 0 "" white \
  darkgreen -13.693 -11.735 0
simundump kpool /kinetics/temp-PIP2 1 0 2.5 2.5 1.5e+06 1.5e+06 0 0 6e+05 6 \
  /kinetics/geometry[1] green black -15.796 -7.0473 0
simundump group /kinetics/Ras 1 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 14.513 16.351 0
simundump kreac /kinetics/Ras/dephosph-GEF 1 1 0 "" white blue 9.0702 17.881 \
  0
simundump kpool /kinetics/Ras/inact-GEF 1 0 0.1 0.1 60000 60000 0 0 6e+05 0 \
  /kinetics/geometry[1] hotpink blue 12.453 18.352 0
simundump kpool /kinetics/Ras/GEF* 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] hotpink blue 6.4483 17.246 0
simundump kenz /kinetics/Ras/GEF*/GEF*-act-ras 1 0 0 0 0 6e+05 3.3e-07 0.08 \
  0.02 0 0 "" red hotpink "" 7.0855 16.086 0
simundump kpool /kinetics/Ras/GTP-Ras 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] orange blue 12.564 13.084 0
simundump kpool /kinetics/Ras/GDP-Ras 1 0 0.2 0.2 1.2e+05 1.2e+05 0 0 6e+05 0 \
  /kinetics/geometry[1] pink blue 6.1309 14.165 0
simundump kreac /kinetics/Ras/Ras-intrinsic-GTPase 1 1e-04 0 "" white blue \
  9.0979 13.5 0
simundump kreac /kinetics/Ras/dephosph-GAP 1 0.1 0 "" white blue 4.0234 \
  15.524 0
simundump kpool /kinetics/Ras/GAP* 1 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry[1] red blue 1.3498 14.349 0
simundump kpool /kinetics/Ras/GAP 1 0 0.002 0.002 1200 1200 0 0 6e+05 0 \
  /kinetics/geometry[1] red blue 6.6549 12.338 0
simundump kenz /kinetics/Ras/GAP/GAP-inact-ras 1 0 0 0 0 6e+05 0.001666 1000 \
  10 0 0 "" red red "" 9.0121 12.403 0
simundump doqcsinfo /kinetics/doqcsinfo 0 db8.g 3d_fold_model network \
  "Upinder S. Bhalla, NCBS" "Upinder S. Bhalla, NCBS" "citation here" \
  "Generic Mammalian" "NIH 3T3 Expression" "Surface - Nucleus" \
  "Quantitative match to experiments" \
  "<a href = http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?cmd=Retrieve&db=PubMed&list_uids=9888852&dopt=Abstract>Bhalla US and Iyengar R. Science (1999) 283(5400):381-7</a>( Peer-reviewed publication )" \
  "Exact GENESIS implementation" \
  "Replicates original data , Quantitatively predicts new data" 15 23 0
simundump xgraph /graphs/conc1 0 0 3750 0.011195 0.08583 0
simundump xgraph /graphs/conc2 0 0 4000 0 1 0
simundump xplot /graphs/conc1/GTP-Ras.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" orange 0 0 1
simundump xplot /graphs/conc1/Raf-GTP-Ras*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" red 0 0 1
simundump xplot /graphs/conc1/MAPKK*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" pink 0 0 1
simundump xplot /graphs/conc1/MAPK*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" orange 0 0 1
simundump xplot /graphs/conc1/AA.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" darkgreen 0 0 1
simundump xplot /graphs/conc1/PKC-active.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" yellow 0 0 1
simundump xplot /graphs/conc1/PLA2*-Ca.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" orange 0 0 1
simundump xplot /graphs/conc1/PLA2*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" orange 0 0 1
simundump xplot /graphs/conc2/MKP-1.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" hotpink 0 0 1
simundump xplot /graphs/conc2/PPhosphatase2A.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" hotpink 0 0 1
simundump xgraph /moregraphs/conc3 0 0 4000 0 1 0
simundump xgraph /moregraphs/conc4 0 0 4000 0 1 0
simundump xcoredraw /edit/draw 0 -19.609 18.429 -17.04 24.181
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
xtextload /file/notes \
"Based on fb2.g" \
"Buffered DAG to 11.661 uM." \
"Added plots."
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
addmsg /kinetics/PKC/PKC-act-by-Ca /kinetics/Ca REAC A B 
addmsg /kinetics/PLA2/PLA2-Ca-act /kinetics/Ca REAC A B 
addmsg /kinetics/PLA2/PLA2*-Ca-act /kinetics/Ca REAC A B 
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
addmsg /kinetics/PKC-active/PKC-act-raf /kinetics/PKC-active CONSERVE nComplex nComplexInit 
addmsg /kinetics/PKC-active/PKC-inact-GAP /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active/PKC-inact-GAP /kinetics/PKC-active CONSERVE nComplex nComplexInit 
addmsg /kinetics/PKC-active/PKC-act-GEF /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active/PKC-act-GEF /kinetics/PKC-active CONSERVE nComplex nComplexInit 
addmsg /kinetics/PKC-active/PKC-act-raf /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-act-raf ENZYME n 
addmsg /kinetics/MAPK/craf-1 /kinetics/PKC-active/PKC-act-raf SUBSTRATE n 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-inact-GAP ENZYME n 
addmsg /kinetics/Ras/GAP /kinetics/PKC-active/PKC-inact-GAP SUBSTRATE n 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-act-GEF ENZYME n 
addmsg /kinetics/Ras/inact-GEF /kinetics/PKC-active/PKC-act-GEF SUBSTRATE n 
addmsg /kinetics/PKC-active/PKC-act-raf /kinetics/MAPK/craf-1 REAC sA B 
addmsg /kinetics/PPhosphatase2A/craf-deph /kinetics/MAPK/craf-1 MM_PRD pA 
addmsg /kinetics/PKC-active/PKC-act-raf /kinetics/MAPK/craf-1* MM_PRD pA 
addmsg /kinetics/MAPK*/MAPK*-feedback /kinetics/MAPK/craf-1* REAC sA B 
addmsg /kinetics/PPhosphatase2A/craf-deph /kinetics/MAPK/craf-1* REAC sA B 
addmsg /kinetics/PPhosphatase2A/craf**-deph /kinetics/MAPK/craf-1* MM_PRD pA 
addmsg /kinetics/Ras-act-craf /kinetics/MAPK/craf-1* REAC A B 
addmsg /kinetics/PPhosphatase2A/MAPKK-deph-ser /kinetics/MAPK/MAPKK MM_PRD pA 
addmsg /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.1 /kinetics/MAPK/MAPKK REAC sA B 
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
addmsg /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.2 /kinetics/MAPK/MAPKK* MM_PRD pA 
addmsg /kinetics/MAPK/MAPKK* /kinetics/MAPK/MAPKK*/MAPKKtyr ENZYME n 
addmsg /kinetics/MAPK/MAPK /kinetics/MAPK/MAPKK*/MAPKKtyr SUBSTRATE n 
addmsg /kinetics/MAPK/MAPKK* /kinetics/MAPK/MAPKK*/MAPKKthr ENZYME n 
addmsg /kinetics/MAPK/MAPK-tyr /kinetics/MAPK/MAPKK*/MAPKKthr SUBSTRATE n 
addmsg /kinetics/PPhosphatase2A/MAPKK-deph /kinetics/MAPK/MAPKK-ser MM_PRD pA 
addmsg /kinetics/PPhosphatase2A/MAPKK-deph-ser /kinetics/MAPK/MAPKK-ser REAC sA B 
addmsg /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.1 /kinetics/MAPK/MAPKK-ser MM_PRD pA 
addmsg /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.2 /kinetics/MAPK/MAPKK-ser REAC sA B 
addmsg /kinetics/Ras-act-craf /kinetics/MAPK/Raf-GTP-Ras* REAC B A 
addmsg /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.1 /kinetics/MAPK/Raf-GTP-Ras* REAC eA B 
addmsg /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.2 /kinetics/MAPK/Raf-GTP-Ras* REAC eA B 
addmsg /kinetics/MAPK/Raf-GTP-Ras* /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.1 ENZYME n 
addmsg /kinetics/MAPK/MAPKK /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.1 SUBSTRATE n 
addmsg /kinetics/MAPK/Raf-GTP-Ras* /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.2 ENZYME n 
addmsg /kinetics/MAPK/MAPKK-ser /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.2 SUBSTRATE n 
addmsg /kinetics/MAPK*/MAPK*-feedback /kinetics/MAPK* REAC eA B 
addmsg /kinetics/MAPK/MAPKK*/MAPKKthr /kinetics/MAPK* MM_PRD pA 
addmsg /kinetics/MKP-1/MKP1-thr-deph /kinetics/MAPK* REAC sA B 
addmsg /kinetics/MAPK*/MAPK* /kinetics/MAPK* REAC eA B 
addmsg /kinetics/MAPK* /kinetics/MAPK*/MAPK*-feedback ENZYME n 
addmsg /kinetics/MAPK/craf-1* /kinetics/MAPK*/MAPK*-feedback SUBSTRATE n 
addmsg /kinetics/MAPK* /kinetics/MAPK*/MAPK* ENZYME n 
addmsg /kinetics/PLA2/PLA2-cytosolic /kinetics/MAPK*/MAPK* SUBSTRATE n 
addmsg /kinetics/MKP-1/MKP1-tyr-deph /kinetics/MKP-1 REAC eA B 
addmsg /kinetics/MKP-1/MKP1-thr-deph /kinetics/MKP-1 REAC eA B 
addmsg /kinetics/MKP-1 /kinetics/MKP-1/MKP1-tyr-deph ENZYME n 
addmsg /kinetics/MAPK/MAPK-tyr /kinetics/MKP-1/MKP1-tyr-deph SUBSTRATE n 
addmsg /kinetics/MKP-1 /kinetics/MKP-1/MKP1-thr-deph ENZYME n 
addmsg /kinetics/MAPK* /kinetics/MKP-1/MKP1-thr-deph SUBSTRATE n 
addmsg /kinetics/MAPK/Raf-GTP-Ras* /kinetics/Ras-act-craf PRODUCT n 
addmsg /kinetics/MAPK/craf-1* /kinetics/Ras-act-craf SUBSTRATE n 
addmsg /kinetics/Ras/GTP-Ras /kinetics/Ras-act-craf SUBSTRATE n 
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
addmsg /kinetics/PLA2/PLA2-Ca-act /kinetics/PLA2/PLA2-cytosolic REAC A B 
addmsg /kinetics/PLA2/PIP2-PLA2-act /kinetics/PLA2/PLA2-cytosolic REAC A B 
addmsg /kinetics/PLA2/PIP2-PLA2* /kinetics/PLA2/PLA2-cytosolic CONSERVE n nInit 
addmsg /kinetics/PLA2/PIP2-Ca-PLA2* /kinetics/PLA2/PLA2-cytosolic CONSERVE n nInit 
addmsg /kinetics/PLA2/DAG-Ca-PLA2* /kinetics/PLA2/PLA2-cytosolic CONSERVE n nInit 
addmsg /kinetics/PLA2/PLA2-Ca* /kinetics/PLA2/PLA2-cytosolic CONSERVE n nInit 
addmsg /kinetics/PLA2/PLA2*-Ca /kinetics/PLA2/PLA2-cytosolic CONSERVE n nInit 
addmsg /kinetics/MAPK*/MAPK* /kinetics/PLA2/PLA2-cytosolic CONSERVE nComplex nComplexInit 
addmsg /kinetics/PLA2/PLA2*-Ca/kenz /kinetics/PLA2/PLA2-cytosolic CONSERVE nComplex nComplexInit 
addmsg /kinetics/PLA2/PLA2-Ca*/kenz /kinetics/PLA2/PLA2-cytosolic CONSERVE nComplex nComplexInit 
addmsg /kinetics/PLA2/DAG-Ca-PLA2*/kenz /kinetics/PLA2/PLA2-cytosolic CONSERVE nComplex nComplexInit 
addmsg /kinetics/PLA2/PIP2-Ca-PLA2*/kenz /kinetics/PLA2/PLA2-cytosolic CONSERVE nComplex nComplexInit 
addmsg /kinetics/PLA2/PIP2-PLA2*/kenz /kinetics/PLA2/PLA2-cytosolic CONSERVE nComplex nComplexInit 
addmsg /kinetics/MAPK*/MAPK* /kinetics/PLA2/PLA2-cytosolic REAC sA B 
addmsg /kinetics/PLA2/PLA2* /kinetics/PLA2/PLA2-cytosolic CONSERVE n nInit 
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
addmsg /kinetics/Ras/GEF* /kinetics/Ras/dephosph-GEF SUBSTRATE n 
addmsg /kinetics/Ras/inact-GEF /kinetics/Ras/dephosph-GEF PRODUCT n 
addmsg /kinetics/PKC-active/PKC-act-GEF /kinetics/Ras/inact-GEF REAC sA B 
addmsg /kinetics/Ras/dephosph-GEF /kinetics/Ras/inact-GEF REAC B A 
addmsg /kinetics/PKC-active/PKC-act-GEF /kinetics/Ras/GEF* MM_PRD pA 
addmsg /kinetics/Ras/dephosph-GEF /kinetics/Ras/GEF* REAC A B 
addmsg /kinetics/Ras/GEF*/GEF*-act-ras /kinetics/Ras/GEF* REAC eA B 
addmsg /kinetics/Ras/GEF* /kinetics/Ras/GEF*/GEF*-act-ras ENZYME n 
addmsg /kinetics/Ras/GDP-Ras /kinetics/Ras/GEF*/GEF*-act-ras SUBSTRATE n 
addmsg /kinetics/Ras/GAP/GAP-inact-ras /kinetics/Ras/GTP-Ras REAC sA B 
addmsg /kinetics/Ras/Ras-intrinsic-GTPase /kinetics/Ras/GTP-Ras REAC A B 
addmsg /kinetics/Ras/GEF*/GEF*-act-ras /kinetics/Ras/GTP-Ras MM_PRD pA 
addmsg /kinetics/Ras-act-craf /kinetics/Ras/GTP-Ras REAC A B 
addmsg /kinetics/Ras/GAP/GAP-inact-ras /kinetics/Ras/GDP-Ras MM_PRD pA 
addmsg /kinetics/Ras/Ras-intrinsic-GTPase /kinetics/Ras/GDP-Ras REAC B A 
addmsg /kinetics/Ras/GEF*/GEF*-act-ras /kinetics/Ras/GDP-Ras REAC sA B 
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
addmsg /kinetics/Ras/GTP-Ras /graphs/conc1/GTP-Ras.Co PLOT Co *GTP-Ras.Co *orange 
addmsg /kinetics/MAPK/Raf-GTP-Ras* /graphs/conc1/Raf-GTP-Ras*.Co PLOT Co *Raf-GTP-Ras*.Co *red 
addmsg /kinetics/MAPK/MAPKK* /graphs/conc1/MAPKK*.Co PLOT Co *MAPKK*.Co *pink 
addmsg /kinetics/MAPK* /graphs/conc1/MAPK*.Co PLOT Co *MAPK*.Co *orange 
addmsg /kinetics/AA /graphs/conc1/AA.Co PLOT Co *AA.Co *darkgreen 
addmsg /kinetics/PKC-active /graphs/conc1/PKC-active.Co PLOT Co *PKC-active.Co *yellow 
addmsg /kinetics/PLA2/PLA2*-Ca /graphs/conc1/PLA2*-Ca.Co PLOT Co *PLA2*-Ca.Co *orange 
addmsg /kinetics/PLA2/PLA2* /graphs/conc1/PLA2*.Co PLOT Co *PLA2*.Co *orange 
addmsg /kinetics/MKP-1 /graphs/conc2/MKP-1.Co PLOT Co *MKP-1.Co *hotpink 
addmsg /kinetics/PPhosphatase2A /graphs/conc2/PPhosphatase2A.Co PLOT Co *PPhosphatase2A.Co *hotpink 
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
call /kinetics/DAG/notes LOAD \
"Baseline is 11.661 from running combined model to" \
"steady state. See ALL/NOTES for 23 Apr 1998."
call /kinetics/PKC-active/notes LOAD \
"Conc of PKC in brain is about 1 uM (?)"
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
call /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.1/notes LOAD \
"Kinetics are the same as for the craf-1* activity, ie.," \
"k1=1.1e-6, k2=.42, k3 =0.105" \
"These are based on Force et al PNAS USA 91 1270-1274 1994." \
"These parms cannot reach the observed 4X stim of MAPK. So lets" \
"increase the affinity, ie, raise k1 10X to 1.1e-5" \
"Lets take it back down to where it was." \
"Back up to 5X: 5.5e-6"
call /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.2/notes LOAD \
"Same kinetics as other c-raf activated forms. See " \
"Force et al PNAS 91 1270-1274 1994." \
"k1 = 1.1e-6, k2 = .42, k3 = 1.05" \
"raise k1 to 5.5e-6" \
""
call /kinetics/MAPK*/notes LOAD \
"This version is phosphorylated on both the tyr and thr residues and" \
"is active: refs" \
"The rate consts are very hard to nail down. Combine Sanghera et al" \
"JBC 265(1) :52-57 with Nemenoff et al JBC 93 pp 1960 to get" \
"k3=10/sec = k2 (from Nemenoff Vmax) and k1 = (k2 + k3)/Km = 1.3e-6" \
"Or: k3 = 10, k2 = 40, k1 = 3.25e-6"
call /kinetics/MAPK*/MAPK*-feedback/notes LOAD \
"Ueki et al JBC 269(22):15756-15761 show the presence of" \
"this step, but not the rate consts, which are derived from" \
"Sanghera et al  JBC 265(1):52-57, 1990, see the deriv in the" \
"MAPK* notes."
call /kinetics/MAPK*/MAPK*/notes LOAD \
"Km = 25uM @ 50 uM ATP and 1mg/ml MBP (huge XS of substrate)" \
"Vmax = 4124 pmol/min/ml at a conc of 125 pmol/ml of enz, so:" \
"k3 = .5/sec (rate limiting)" \
"k1 = (k2  + k3)/Km = (.5 + 0)/(25*6e5) = 2e-8 (#/cell)^-1" \
"#s from Sanghera et al JBC 265 pp 52 , 1990. " \
"From Nemenoff et al JBC 268(3):1960-1964 - using Sanghera's 1e-4 ratio" \
"of MAPK to protein, we get k3 = 7/sec from 1000 pmol/min/mg fig 5"
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
call /kinetics/Ras-act-craf/notes LOAD \
"Assume the binding is fast and is limited only by the amount of" \
"Ras* available. So kf = kb/[craf-1] " \
"If kb is 1/sec, then kf = 1/0.2 uM = 1/(0.2 * 6e5) = 8.3e-6" \
"Later: Raise it by 10 X to 4e-5" \
"From Hallberg et al JBC 269:6 3913-3916 1994, 3% of cellular Raf is" \
"complexed with Ras. So we raise kb 4 x to 4" \
"This step needed to memb-anchor and activate Raf: Leevers et al Nature" \
"369 411-414"
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
"Level=.002"
call /kinetics/Ras/GAP/GAP-inact-ras/notes LOAD \
"From Eccleston et al JBC 268(36)pp27012-19 get Kd < 2uM, kcat - 10/sec" \
"From Martin et al Cell 63 843-849 1990 get Kd ~ 250 nM, kcat = 20/min" \
"I will go with the Eccleston figures as there are good error bars (10%). In general" \
"the values are reasonably close." \
"k1 = 1.666e-3/sec, k2 = 1000/sec, k3 = 10/sec (note k3 is rate-limiting)"
call /kinetics/doqcsinfo/notes LOAD \
"This model is based closely on the one from <a href=http://www.ncbi.nlm.nih.gov:80/entrez/query.fcgi?cmd=Retrieve&db=PubMed&list_uids=9888852&dopt=Abstract>Bhalla US and Iyengar R. Science (1999) 283(5400):381-7</a>. This is a stripped down version with only the essential components of the feedback loop:PKC, Ras and the MAPK cascade, and PLA2 in the synapse. The upregulation of MKP-1 is not incorporated in this model since it is treated as a fixed regulatory input. This model was used to develop figures 2 - 4 which show the bistable region of parameter space when the regulatory inputs Ca, PP2A and MKP-1 are varied."
complete_loading
