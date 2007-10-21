//genesis
// kkit Version 9 flat dumpfile
 
// Saved on Mon Sep 22 21:26:16 2003
 
include kkit {argv 1}
 
FASTDT = 0.0001
SIMDT = 0.001
CONTROLDT = 10
PLOTDT = 10
MAXTIME = 2500
TRANSIENT_TIME = 2
VARIABLE_DT_FLAG = 1
DEFAULT_VOL = 1.6667e-21
VERSION = 9.0
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
simobjdump transport input output alloced step_mode stepsize dt delay clock \
  kf xtree_fg_req xtree_textfg_req x y z
simobjdump proto x y z
simobjdump linkinfo xtree_fg_req xtree_textfg_req uplink downlink x y z
simobjdump uplink xtree_fg_req xtree_textfg_req x y z
simobjdump downlink xtree_fg_req xtree_textfg_req x y z
simobjdump mirror notes xtree_fg_req x y z
simundump group /kinetics/PKC 0 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 -3.0493 8.2163 0
simundump kpool /kinetics/PKC/PKC-Ca 0 0.28281 3.7207e-17 3.7207e-17 \
  3.7207e-17 3.7207e-17 0.28281 0 0.99999 0 "" red black -4.0752 1.5108 0
simundump kreac /kinetics/PKC/PKC-act-by-Ca 0 0.60001 0.5 "" white blue \
  -4.0752 -0.12295 0
simundump kreac /kinetics/PKC/PKC-act-by-DAG 0 0.0079999 8.6348 "" white blue \
  -2.0612 0.69395 0
simundump kreac /kinetics/PKC/PKC-Ca-to-memb 0 1.2705 3.5026 "" white blue \
  -3.7974 4.2533 0
simundump kreac /kinetics/PKC/PKC-DAG-to-memb 0 1 0.1 "" white blue -2.6168 \
  2.7362 0
simundump kreac /kinetics/PKC/PKC-act-by-Ca-AA 0 0.0012 0.1 "" white blue \
  -0.78797 3.8157 0
simundump kreac /kinetics/PKC/PKC-act-by-DAG-AA 0 2 0.2 "" white blue 1.2492 \
  3.2322 0
simundump kpool /kinetics/PKC/PKC-DAG-AA* 0 0.12061 4.9136e-18 4.9136e-18 \
  4.9136e-18 4.9136e-18 0.12061 0 0.99999 0 "" cyan blue 0.60098 5.537 0
simundump kpool /kinetics/PKC/PKC-Ca-AA* 0 0.16962 1.75e-16 1.75e-16 1.75e-16 \
  1.75e-16 0.16962 0 0.99999 0 "" orange blue -0.60278 6.2956 0
simundump kpool /kinetics/PKC/PKC-Ca-memb* 0 0.10258 1.3895e-17 1.3895e-17 \
  1.3895e-17 1.3895e-17 0.10258 0 0.99999 0 "" pink blue -2.7788 6.529 0
simundump kpool /kinetics/PKC/PKC-DAG-memb* 0 0.023753 9.4351e-21 9.4351e-21 \
  9.435e-21 9.435e-21 0.023753 0 0.99999 0 "" yellow blue -1.8297 5.5078 0
simundump kpool /kinetics/PKC/PKC-basal* 0 0 0 0 0 0 0 0 0.99999 0 "" pink \
  blue -4.7465 5.5662 0
simundump kreac /kinetics/PKC/PKC-basal-act 0 1 50 "" white blue -4.978 \
  3.0571 0
simundump kpool /kinetics/PKC/PKC-AA* 0 1 1.8133e-17 1.8133e-17 1.8133e-17 \
  1.8133e-17 0.99999 0 0.99999 0 "" cyan blue 1.7816 6.8207 0
simundump kreac /kinetics/PKC/PKC-act-by-AA 0 0.00012 0.1 "" white blue \
  -4.9925 -1.8654 0
simundump kpool /kinetics/PKC/PKC-Ca-DAG 0 0.0017993 8.4631e-23 8.4631e-23 \
  8.463e-23 8.463e-23 0.0017993 0 0.99999 0 "" white blue 0.2306 1.8026 0
simundump kreac /kinetics/PKC/PKC-n-DAG 0 0.00060001 0.1 "" white blue \
  -3.0103 -1.9902 0
simundump kpool /kinetics/PKC/PKC-DAG 0 0.08533 1.161e-16 1.161e-16 1.161e-16 \
  1.161e-16 0.085329 0 0.99999 0 "" white blue -0.99631 -1.0857 0
simundump kreac /kinetics/PKC/PKC-n-DAG-AA 0 0.018 2 "" white blue -1.2278 \
  -2.9529 0
simundump kpool /kinetics/PKC/PKC-DAG-AA 0 0.012092 2.5189e-19 2.5189e-19 \
  2.5189e-19 2.5189e-19 0.012092 0 0.99999 0 "" white blue 0.62413 0.22715 0
simundump kpool /kinetics/PKC/PKC-cytosolic 0 1.5489 1 1 0.99999 0.99999 \
  1.5489 0 0.99999 0 "" white blue -6.1315 0.59711 0
simundump kpool /kinetics/DAG 1 1.6667e-06 0 0 0 0 1.6667e-06 0 0.99999 0 "" \
  green black 0.71777 -7.5617 0
simundump kpool /kinetics/AA 0 7 0 0 0 0 6.9999 0 0.99999 0 "" darkgreen \
  black -3.2898 -9.3376 0
simundump kpool /kinetics/PKC-active 1 2.1195e-16 0 0 0 0 2.1195e-16 0 \
  0.99999 2 "" red black 0 9 0
simundump kenz /kinetics/PKC-active/PKC-act-raf 1 0 0 0 0 0.99999 0.3 16 4 0 \
  0 "" red yellow "" 6 10 0
simundump kenz /kinetics/PKC-active/PKC-inact-GAP 1 0 0 0 0 1.6667e-06 6.0001 \
  16 4 0 0 "" red yellow "" 4 13 0
simundump kenz /kinetics/PKC-active/PKC-act-GEF 1 0 0 0 0 1.6667e-06 6.0001 \
  16 4 0 0 "" red yellow "" 9 20 0
simundump kenz /kinetics/PKC-active/PKC-phosph-neurogranin 1 0 0 0 0 0.99999 \
  0.102 2.34 0.58 0 0 "" red red "" -38.873 5.5986 0
simundump kenz /kinetics/PKC-active/PKC-phosph-ng-CaM 1 0 0 0 0 0.99999 \
  0.061201 1.4 0.35 0 0 "" red red "" -42.309 8.3838 0
simundump kenz /kinetics/PKC-active/phosph-AC2 1 0 0 0 0 0.99999 0.60001 16 4 \
  0 0 "" red red "" -6.6822 -11.269 0
simundump group /kinetics/PLA2 0 darkgreen black x 0 1 "" defaultfile \
  defaultfile.g 0 0 0 -7.3572 -14.209 0
simundump kpool /kinetics/PLA2/PLA2-cytosolic 0 0.4 0.4 0.4 0.4 0.4 0.4 0 \
  0.99999 0 "" yellow darkgreen -11.824 -8.9421 0
simundump kreac /kinetics/PLA2/PLA2-Ca-act 0 1 0.1 "" white darkgreen -11.097 \
  -11.104 0
simundump kpool /kinetics/PLA2/PLA2-Ca* 0 1 0 0 0 0 0.99999 0 0.99999 0 "" \
  yellow darkgreen -8.722 -11.646 0
simundump kenz /kinetics/PLA2/PLA2-Ca*/kenz 0 0 0 0 0 0.99999 1.35 21.6 5.4 0 \
  0 "" red yellow "" -6.0553 -11.667 0
simundump kreac /kinetics/PLA2/PIP2-PLA2-act 0 0.0012 0.5 "" white darkgreen \
  -11.055 -6.7502 0
simundump kpool /kinetics/PLA2/PIP2-PLA2* 0 1 0 0 0 0 0.99999 0 0.99999 0 "" \
  cyan darkgreen -8.6803 -6.2919 0
simundump kenz /kinetics/PLA2/PIP2-PLA2*/kenz 0 0 0 0 0 0.99999 2.7601 44.16 \
  11.04 0 0 "" red cyan "" -6.0345 -6.271 0
simundump kreac /kinetics/PLA2/PIP2-Ca-PLA2-act 0 0.012 0.1 "" white \
  darkgreen -10.097 -7.5002 0
simundump kpool /kinetics/PLA2/PIP2-Ca-PLA2* 0 1 0 0 0 0 0.99999 0 0.99999 0 \
  "" cyan darkgreen -8.3261 -7.896 0
simundump kenz /kinetics/PLA2/PIP2-Ca-PLA2*/kenz 0 0 0 0 0 0.99999 9.0001 144 \
  36 0 0 "" red cyan "" -5.972 -7.9794 0
simundump kreac /kinetics/PLA2/DAG-Ca-PLA2-act 0 0.003 4 "" white darkgreen \
  -10.826 -9.8336 0
simundump kpool /kinetics/PLA2/DAG-Ca-PLA2* 0 1 0 0 0 0 0.99999 0 0.99999 0 \
  "" pink darkgreen -8.1386 -10.479 0
simundump kenz /kinetics/PLA2/DAG-Ca-PLA2*/kenz 0 0 0 0 0 0.99999 15 240 60 0 \
  0 "" red pink "" -5.9511 -10.354 0
simundump kpool /kinetics/PLA2/APC 0 30 30 30 30 30 30 0 0.99999 4 "" yellow \
  darkgreen -8.2386 -9.9634 0
simundump kreac /kinetics/PLA2/Degrade-AA 1 0.4 0 "" white darkgreen -6.1808 \
  -5.2875 0
simundump kpool /kinetics/PLA2/PLA2*-Ca 0 1 0 0 0 0 0.99999 0 0.99999 0 "" \
  orange darkgreen -7.813 -12.687 0
simundump kenz /kinetics/PLA2/PLA2*-Ca/kenz 0 0 0 0 0 0.99999 30 480 120 0 0 \
  "" red orange "" -6.0814 -12.817 0
simundump kpool /kinetics/PLA2/PLA2* 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" orange darkgreen -9.025 -14.851 0
simundump kreac /kinetics/PLA2/PLA2*-Ca-act 1 6.0001 0.1 "" white darkgreen \
  -10.086 -12.752 0
simundump kreac /kinetics/PLA2/dephosphorylate-PLA2* 1 0.17 0 "" white \
  darkgreen -13.693 -11.735 0
simundump kpool /kinetics/MAPK* 1 1.6667e-06 0 0 0 0 1.6667e-06 0 0.99999 0 \
  "" orange yellow 13 1 0
simundump kenz /kinetics/MAPK*/MAPK* 0 0 0 0 0 0.99999 1.95 40 10 0 0 "" red \
  orange "" -12 -14 0
simundump kenz /kinetics/MAPK*/MAPK*-feedback 1 0 0 0 0 0.99999 1.95 40 10 0 \
  0 "" red orange "" 10 10 0
simundump kenz /kinetics/MAPK*/phosph_Sos 1 0 0 0 0 0.99999 19.5 40 10 0 0 "" \
  red orange "" 18 39 0
simundump kpool /kinetics/temp-PIP2 1 25 2.5 2.5 2.5 2.5 25 0 0.99999 4 "" \
  green black -15.796 -7.0473 0
simundump kpool /kinetics/IP3 1 1.71 0.73 0.73 0.72999 0.72999 1.71 0 0.99999 \
  0 "" pink black -0.77375 -4.6555 0
simundump kpool /kinetics/Glu 1 1000 0 0 0 0 999.99 0 0.99999 4 "" green \
  black -0.79501 13.884 0
simundump group /kinetics/PLCbeta 1 maroon black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 8.5846 -17.468 0
simundump kreac /kinetics/PLCbeta/Act-PLC-Ca 1 3 1 "" white maroon 3.0709 \
  -16.978 0
simundump kpool /kinetics/PLCbeta/PLC 1 0.8 0.8 0.8 0.79999 0.79999 0.79999 0 \
  0.99999 0 "" cyan maroon 10.697 -16.957 0
simundump kreac /kinetics/PLCbeta/Degrade-IP3 1 2.5 0 "" white maroon 2.3125 \
  -7.9705 0
simundump kpool /kinetics/PLCbeta/Inositol 1 0 0 0 0 0 0 0 0.99999 4 "" green \
  maroon 4.9653 -8.7416 0
simundump kreac /kinetics/PLCbeta/Degrade-DAG 1 0.15 0 "" white maroon \
  -0.95715 -7.261 0
simundump kpool /kinetics/PLCbeta/PC 1 1 0 0 0 0 1.6667e-06 0 1.6667e-06 4 "" \
  green maroon 4.9036 -7.1376 0
simundump kpool /kinetics/PLCbeta/PLC-Ca 1 0.061191 0 0 0 0 0.06119 0 0.99999 \
  0 "" cyan maroon 7.0147 -12.797 0
simundump kenz /kinetics/PLCbeta/PLC-Ca/PLC-Ca 1 0 0 0 0 0.99999 2.52 40 10 0 \
  0 "" red cyan "" -2.2511 -11.697 0
simundump kreac /kinetics/PLCbeta/Act-PLC-by-Gq 1 25.2 1 "" white maroon \
  2.6996 -15.163 0
simundump kreac /kinetics/PLCbeta/Inact-PLC-Gq 1 0.0133 0 "" white maroon \
  11.125 -10.314 0
simundump kpool /kinetics/PLCbeta/PLC-Ca-Gq 0 0.008698 0 0 0 0 0.0086979 0 \
  0.99999 0 "" cyan maroon 10.629 -13.411 0
simundump kenz /kinetics/PLCbeta/PLC-Ca-Gq/PLCb-Ca-Gq 0 0 0 0 0 0.99999 48 \
  192 48 0 0 "" red cyan "" 2.9471 -11.078 0
simundump kpool /kinetics/PLCbeta/PLC-Gq 1 0.063589 0 0 0 0 0.063588 0 \
  0.99999 0 "" cyan maroon 15.035 -13.537 0
simundump kreac /kinetics/PLCbeta/PLC-bind-Gq 1 2.52 1 "" white maroon 14.746 \
  -16.263 0
simundump kreac /kinetics/PLCbeta/PLC-Gq-bind-Ca 1 30 1 "" white maroon \
  14.004 -11.254 0
simundump kpool /kinetics/PIP2 1 10 10 10 9.9999 9.9999 9.9999 0 0.99999 4 "" \
  green black 3.8839 -6.7218 0
simundump kpool /kinetics/BetaGamma 1 1.6 0 0 0 0 1.6 0 0.99999 0 "" yellow \
  black 15.787 -2.6163 0
simundump kpool /kinetics/G*GTP 1 5.0001 0 0 0 0 5 0 0.99999 0 "" red black \
  7.3149 -7.0131 0
simundump kpool /kinetics/G*GDP 1 0.1667 0 0 0 0 0.1667 0 0.99999 0 "" yellow \
  black 13.56 -5.6529 0
simundump group /kinetics/Gq 1 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 12.745 -1.9437 0
simundump kreac /kinetics/Gq/RecLigandBinding 1 16.8 10 "" white blue 7.3388 \
  -3.179 0
simundump kpool /kinetics/Gq/G-GDP 1 1.6 1 1 0.99999 0.99999 1.6 0 0.99999 0 \
  "" yellow blue 10.68 -2.5729 0
simundump kreac /kinetics/Gq/Basal-Act-G 1 1e-04 0 "" white blue 9.805 \
  -4.8225 0
simundump kreac /kinetics/Gq/Trimerize-G 1 6.0001 0 "" white blue 12.255 \
  -4.7831 0
simundump kreac /kinetics/Gq/Inact-G 1 0.0133 0 "" white blue 10.218 -7.6095 \
  0
simundump kpool /kinetics/Gq/mGluR 1 0.3 0.3 0.3 0.3 0.3 0.3 0 0.99999 0 "" \
  green blue 6.4638 -1.7623 0
simundump kpool /kinetics/Gq/Rec-Glu 1 0.8 0 0 0 0 0.79999 0 0.99999 0 "" \
  green blue 5.8108 -3.7217 0
simundump kpool /kinetics/Gq/Rec-Gq 1 1.6667e-06 0 0 0 0 1.6667e-06 0 0.99999 \
  0 "" green blue 4.0767 -0.99942 0
simundump kreac /kinetics/Gq/Rec-Glu-bind-Gq 1 0.0060001 1e-04 "" white blue \
  4.7148 -2.4225 0
simundump kreac /kinetics/Gq/Glu-bind-Rec-Gq 1 16.8 0.1 "" white blue 2.386 \
  -3.0371 0
simundump kpool /kinetics/Gq/Rec-Glu-Gq 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" orange blue 4.7416 -5.1166 0
simundump kreac /kinetics/Gq/Activate-Gq 1 0.01 0 "" white blue 7.0172 \
  -4.6572 0
simundump kreac /kinetics/Gq/Rec-bind-Gq 1 0.60001 1 "" white blue 6.743 \
  -0.088999 0
simundump kpool /kinetics/Gq/mGluRAntag 1 100 0 0 0 0 99.999 0 0.99999 4 "" \
  seagreen blue 0.60216 -2.3091 0
simundump kreac /kinetics/Gq/Antag-bind-Rec-Gq 1 60.001 0.01 "" white blue \
  2.1399 -4.2806 0
simundump kpool /kinetics/Gq/Blocked-rec-Gq 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" seagreen blue 2.4602 -5.9815 0
simundump group /kinetics/MAPK 0 brown black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 15 11 0
simundump kpool /kinetics/MAPK/craf-1 0 0.50001 0.50001 0.50001 0.5 0.5 0.5 0 \
  0.99999 0 "" pink brown 6 8 0
simundump kpool /kinetics/MAPK/craf-1* 0 0.2 0 0 0 0 0.2 0 0.99999 0 "" pink \
  brown 9 8 0
simundump kpool /kinetics/MAPK/MAPKK 0 0.50001 0.50001 0.50001 0.5 0.5 0.5 0 \
  0.99999 0 "" pink brown 6 4 0
simundump kpool /kinetics/MAPK/MAPK 0 3.6 3.6 3.6 3.6 3.6 3.6 0 0.99999 0 "" \
  pink brown 6 1 0
simundump kpool /kinetics/MAPK/craf-1** 1 0.2 0 0 0 0 0.2 0 0.99999 0 "" \
  hotpink brown 12 8 0
simundump kpool /kinetics/MAPK/MAPK-tyr 1 0.36 0 0 0 0 0.36 0 0.99999 0 "" \
  orange brown 9 1 0
simundump kpool /kinetics/MAPK/MAPKK* 0 0.18 0 0 0 0 0.18 0 0.99999 0 "" pink \
  brown 13 4 0
simundump kenz /kinetics/MAPK/MAPKK*/MAPKKtyr 0 0 0 0 0 0.99999 32.4 1.2 0.3 \
  0 0 "" red pink "" 8 3 0
simundump kenz /kinetics/MAPK/MAPKK*/MAPKKthr 1 0 0 0 0 0.99999 32.4 1.2 0.3 \
  0 0 "" red pink "" 13 3 0
simundump kpool /kinetics/MAPK/MAPKK-ser 1 0.18 0 0 0 0 0.18 0 0.99999 0 "" \
  pink brown 9 4 0
simundump kpool /kinetics/MAPK/Raf-GTP-Ras 0 0 0 0 0 0 0 0 0.99999 0 "" 55 \
  brown 5 6 0
simundump kenz /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.1 0 0 0 0 0 0.99999 \
  9.4285 1.2 0.3 0 0 "" red 55 "" 8 6 0
simundump kenz /kinetics/MAPK/Raf-GTP-Ras/Raf-GTP-Ras.2 0 0 0 0 0 0.99999 \
  9.4285 1.2 0.3 0 0 "" red 55 "" 11 6 0
simundump kpool /kinetics/MAPK/Raf*-GTP-Ras 1 0.0104 0 0 0 0 0.0104 0 0.99999 \
  0 "" red brown 5 5 0
simundump kenz /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.1 1 0 0 0 0 \
  1.6667e-06 9.4285 1.2 0.3 0 0 "" red red "" 8 5 0
simundump kenz /kinetics/MAPK/Raf*-GTP-Ras/Raf*-GTP-Ras.2 1 0 0 0 0 \
  1.6667e-06 9.4285 1.2 0.3 0 0 "" red red "" 11 5 0
simundump kpool /kinetics/MKP-1 1 0.1 0.024 0.024 0.024 0.024 0.099999 0 \
  0.99999 0 "" hotpink black 5 2 0
simundump kenz /kinetics/MKP-1/MKP1-tyr-deph 1 0 0 0 0 0.99999 150 16 4 0 0 \
  "" red hotpink "" 6 3 0
simundump kenz /kinetics/MKP-1/MKP1-thr-deph 1 0 0 0 0 0.99999 150 16 4 0 0 \
  "" red hotpink "" 11 3 0
simundump kreac /kinetics/Ras-act-craf 1 10 0.5 "" white black 3 9 0
simundump kpool /kinetics/PPhosphatase2A 1 1 1 1 0.99999 0.99999 0.99999 0 \
  0.99999 0 "" hotpink yellow 9 9 0
simundump kenz /kinetics/PPhosphatase2A/craf-deph 1 0 0 0 0 0.99999 1.9161 24 \
  6 0 0 "" red hotpink "" 8 10 0
simundump kenz /kinetics/PPhosphatase2A/MAPKK-deph 1 0 0 0 0 0.99999 1.9161 \
  24 6 0 0 "" red hotpink "" 11 7 0
simundump kenz /kinetics/PPhosphatase2A/MAPKK-deph-ser 1 0 0 0 0 0.99999 \
  1.9161 24 6 0 0 "" red hotpink "" 8 7 0
simundump kenz /kinetics/PPhosphatase2A/craf**-deph 1 0 0 0 0 1.6667e-06 \
  1.9161 24 6 0 0 "" red hotpink "" 12 10 0
simundump group /kinetics/Ras 1 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 14.513 16.351 0
simundump kreac /kinetics/Ras/bg-act-GEF 1 6.0001 1 "" white blue 13.468 \
  14.838 0
simundump kpool /kinetics/Ras/GEF-Gprot-bg 1 0.1 0 0 0 0 0.099999 0 0.99999 0 \
  "" hotpink blue 10.373 17.271 0
simundump kenz /kinetics/Ras/GEF-Gprot-bg/GEF-bg_act-ras 1 0 0 0 0 0.99999 \
  0.198 0.08 0.02 0 0 "" red hotpink "" 10.402 16.523 0
simundump kreac /kinetics/Ras/dephosph-GEF 1 1 0 "" white blue 9.0702 17.881 \
  0
simundump kpool /kinetics/Ras/inact-GEF 1 0.50001 0.1 0.1 0.099999 0.099999 \
  0.5 0 0.99999 0 "" hotpink blue 12.453 18.352 0
simundump kpool /kinetics/Ras/GEF* 1 0.1 0 0 0 0 0.099999 0 0.99999 0 "" \
  hotpink blue 6.4483 17.246 0
simundump kenz /kinetics/Ras/GEF*/GEF*-act-ras 1 0 0 0 0 0.99999 0.198 0.08 \
  0.02 0 0 "" red hotpink "" 7.0855 16.086 0
simundump kpool /kinetics/Ras/GTP-Ras 1 0.2 0 0 0 0 0.2 0 0.99999 0 "" orange \
  blue 12.564 13.084 0
simundump kpool /kinetics/Ras/GDP-Ras 1 0.50001 0.50001 0.50001 0.5 0.5 0.5 0 \
  0.99999 0 "" pink blue 6.1309 14.165 0
simundump kreac /kinetics/Ras/Ras-intrinsic-GTPase 1 1e-04 0 "" white blue \
  9.0979 13.5 0
simundump kreac /kinetics/Ras/dephosph-GAP 1 0.1 0 "" white blue 4.0234 \
  15.524 0
simundump kpool /kinetics/Ras/GAP* 1 0.050001 0 0 0 0 0.05 0 0.99999 0 "" red \
  blue 1.3498 14.349 0
simundump kpool /kinetics/Ras/GAP 1 0.01 0.01 0.01 0.0099999 0.0099999 \
  0.0099999 0 0.99999 0 "" red blue 6.6549 12.338 0
simundump kenz /kinetics/Ras/GAP/GAP-inact-ras 1 0 0 0 0 0.99999 49.486 40 10 \
  0 0 "" red red "" 9.0121 12.403 0
simundump kpool /kinetics/Ras/inact-GEF* 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" hotpink blue 15.029 19.16 0
simundump kreac /kinetics/Ras/CaM-bind-GEF 1 60.001 1 "" white blue 2.4861 \
  21.679 0
simundump kpool /kinetics/Ras/CaM-GEF 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" pink blue 5.3451 19.58 0
simundump kenz /kinetics/Ras/CaM-GEF/CaM-GEF-act-ras 1 0 0 0 0 0.99999 0.198 \
  0.08 0.02 0 0 "" red pink "" 5.0223 18.657 0
simundump kreac /kinetics/Ras/dephosph-inact-GEF* 1 1 0 "" white blue 14.431 \
  21.995 0
simundump kpool /kinetics/PKA-active 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" yellow black -33.585 -12.858 0
simundump kenz /kinetics/PKA-active/PKA-phosph-GEF 1 0 0 0 0 0.99999 6.0001 \
  36 9 0 0 "" red yellow "" 10.464 21.469 0
simundump kenz /kinetics/PKA-active/PKA-phosph-I1 1 0 0 0 0 0.99999 6.0001 36 \
  9 0 0 "" red yellow "" -36.894 -17.114 0
simundump kenz /kinetics/PKA-active/phosph-PDE 1 0 0 0 0 0.99999 6.0001 36 9 \
  0 0 "" red yellow "" -30.934 -13.317 0
simundump kpool /kinetics/CaM-Ca4 1 5.0001 0 0 0 0 5 0 0.99999 0 "" blue \
  yellow -22.075 -2.8669 0
simundump kpool /kinetics/Shc*.Sos.Grb2 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" brown yellow 11.263 27.112 0
simundump kenz /kinetics/Shc*.Sos.Grb2/Sos.Ras_GEF 1 0 0 0 0 0.99999 0.198 \
  0.08 0.02 0 0 "" red brown "" 11.266 24.47 0
simundump group /kinetics/EGFR 1 yellow black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 7.0249 39.57 0
simundump kpool /kinetics/EGFR/EGFR 1 0.16667 0.16667 0.16667 0.16667 0.16667 \
  0.16667 0 0.99999 0 "" red yellow 1.9551 39.853 0
simundump kreac /kinetics/EGFR/act_EGFR 1 4.2001 0.25 "" white yellow 4.4894 \
  38.493 0
simundump kpool /kinetics/EGFR/L.EGFR 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" red yellow 6.2195 36.599 0
simundump kenz /kinetics/EGFR/L.EGFR/phosph_PLC_g 1 0 0 0 0 0.99999 3 0.8 0.2 \
  0 0 "" red red "" 6.3358 35.082 0
simundump kenz /kinetics/EGFR/L.EGFR/phosph_Shc 1 0 0 0 0 0.99999 1.2 0.8 0.2 \
  0 0 "" red red "" 9.0331 36.49 0
simundump kpool /kinetics/EGFR/EGF 1 1 0 0 0 0 0.99999 0 0.99999 4 "" red \
  yellow 2.2719 36.309 0
simundump kpool /kinetics/EGFR/SHC 1 1 0.50001 0.50001 0.5 0.5 0.99999 0 \
  0.99999 0 "" orange yellow 8.3857 33.936 0
simundump kpool /kinetics/EGFR/SHC* 1 1.6667e-06 0 0 0 0 1.6667e-06 0 0.99999 \
  0 "" orange yellow 12.832 33.029 0
simundump kreac /kinetics/EGFR/dephosph_Shc 1 0.0016667 0 "" white yellow \
  9.7373 31.442 0
simundump kpool /kinetics/EGFR/Internal_L.EGFR 1 1.6667e-06 0 0 0 0 \
  1.6667e-06 0 0.99999 0 "" red yellow 6.3061 41.93 0
simundump kreac /kinetics/EGFR/Internalize 1 0.002 0.00033 "" white yellow \
  4.5213 39.863 0
simundump group /kinetics/Sos 1 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 19.547 34.811 0
simundump kreac /kinetics/Sos/Shc_bind_Sos.Grb2 1 0.49998 0.1 "" white blue \
  10.23 29.891 0
simundump kpool /kinetics/Sos/Sos*.Grb2 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" orange blue 12.274 41.661 0
simundump kreac /kinetics/Sos/Grb2_bind_Sos* 1 0.025 0.0168 "" white blue \
  10.533 38.235 0
simundump kpool /kinetics/Sos/Grb2 1 1 1 1 0.99999 0.99999 0.99999 0 0.99999 \
  0 "" orange blue 14.742 35.301 0
simundump kpool /kinetics/Sos/Sos.Grb2 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" orange blue 13.988 30.097 0
simundump kpool /kinetics/Sos/Sos* 1 1.6667e-06 0 0 0 0 1.6667e-06 0 0.99999 \
  0 "" red blue 15.421 40.215 0
simundump kreac /kinetics/Sos/dephosph_Sos 1 0.001 0 "" white blue 13.185 \
  37.153 0
simundump kreac /kinetics/Sos/Grb2_bind_Sos 1 0.025 0.0168 "" white blue \
  16.422 33.133 0
simundump kpool /kinetics/Sos/Sos 1 0.1 0.1 0.1 0.099999 0.099999 0.099999 0 \
  0.99999 0 "" red blue 17.381 36.794 0
simundump group /kinetics/PLC_g 1 darkgreen black x 0 0 "" defaultfile \
  /home2/bhalla/scripts/modules/defaultfile_0.g 0 0 0 0.44974 33.831 0
simundump kpool /kinetics/PLC_g/PLC_g 1 0.82 0.82 0.82 0.81999 0.81999 \
  0.81999 0 0.99999 0 "" pink darkgreen 0.07993 31.598 0
simundump kreac /kinetics/PLC_g/Ca_act_PLC_g 1 180 10 "" white darkgreen \
  -1.4451 28.194 0
simundump kreac /kinetics/PLC_g/Ca_act_PLC_g* 1 12 10 "" white darkgreen \
  2.7901 29.8 0
simundump kreac /kinetics/PLC_g/dephosph_PLC_g 1 0.05 0 "" white darkgreen \
  4.5589 32.225 0
simundump kpool /kinetics/PLC_g/PLC_G* 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" pink darkgreen 7.1385 31.319 0
simundump kpool /kinetics/PLC_g/Ca.PLC_g 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" pink darkgreen 2.0998 27.462 0
simundump kenz /kinetics/PLC_g/Ca.PLC_g/PIP2_hydrolysis 1 0 0 0 0 0.99999 \
  0.72001 56 14 0 1 "" red pink "" -0.76478 -0.35259 0
simundump kpool /kinetics/PLC_g/Ca.PLC_g* 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" pink darkgreen 7.1087 26.972 0
simundump kenz /kinetics/PLC_g/Ca.PLC_g*/PIP2_hydrolysis 1 0 0 0 0 0.99999 \
  14.4 228 57 0 1 "" red pink "" 3.507 -2.0324 0
simundump group /kinetics/CaMKII 1 purple black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 -22.401 3.9743 0
simundump kpool /kinetics/CaMKII/CaMKII 1 135 70 70 69.999 69.999 135 0 \
  0.99999 0 "" palegreen purple -23.819 3.271 0
simundump kpool /kinetics/CaMKII/CaMKII-CaM 1 0 0 0 0 0 0 0 0.99999 0 "" \
  palegreen purple -27.443 3.0376 0
simundump kpool /kinetics/CaMKII/CaMKII-thr286*-CaM 1 1.6667e-06 0 0 0 0 \
  1.6667e-06 0 0.99999 0 "" palegreen purple -27.703 1.6156 0
simundump kpool /kinetics/CaMKII/CaMKII*** 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" cyan purple -27.616 -1.6238 0
simundump kreac /kinetics/CaMKII/CaMKII-bind-CaM 1 50 5 "" white purple \
  -23.298 1.5267 0
simundump kreac /kinetics/CaMKII/CaMK-thr286-bind-CaM 1 1000.2 0.1 "" white \
  purple -23.277 0.92147 0
simundump kpool /kinetics/CaMKII/CaMKII-thr286 1 1.6667e-06 0 0 0 0 \
  1.6667e-06 0 0.99999 0 "" red purple -27.551 -0.09309 0
simundump kpool /kinetics/CaMKII/CaMK-thr306 1 1.6667e-06 0 0 0 0 1.6667e-06 \
  0 0.99999 0 "" palegreen purple -27.539 -3.2652 0
simundump kreac /kinetics/CaMKII/basal-activity 1 0.003 0 "" white purple \
  -25.369 -0.16228 0
simundump kpool /kinetics/CaMKII/tot_CaM_CaMKII 1 0 0 0 0 0 0 0 0.99999 0 "" \
  green purple -31.715 3.2973 0
simundump kenz /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_305 1 0 0 0 0 0.99999 \
  0.26409 24 6 0 0 "" red green "" -29.551 0.6145 0
simundump kenz /kinetics/CaMKII/tot_CaM_CaMKII/CaM_act_286 1 0 0 0 0 0.99999 \
  0.022007 2 0.5 0 0 "" red green "" -25.596 2.816 0
simundump kpool /kinetics/CaMKII/tot_autonomous_CaMKII 1 0 0 0 0 0 0 0 \
  0.99999 0 "" green purple -32.064 2.3272 0
simundump kenz /kinetics/CaMKII/tot_autonomous_CaMKII/auton_305 1 0 0 0 0 \
  0.99999 0.17142 24 6 0 0 "" red green "" -29.736 -0.41162 0
simundump kenz /kinetics/CaMKII/tot_autonomous_CaMKII/auton_286 1 0 0 0 0 \
  0.99999 0.014286 2 0.5 0 0 "" red green "" -25.473 1.9951 0
simundump kpool /kinetics/PP1-active 1 1.8 1.8 1.8 1.8 1.8 1.8 0 0.99999 0 "" \
  cyan yellow -31.448 0.13975 0
simundump kenz /kinetics/PP1-active/Deph-thr286 1 0 0 0 0 0.99999 0.3432 1.4 \
  0.35 0 0 "" red cyan "" -31.097 1.7813 0
simundump kenz /kinetics/PP1-active/Deph-thr305 1 0 0 0 0 0.99999 0.3432 1.4 \
  0.35 0 0 "" red cyan "" -30.313 -1.1052 0
simundump kenz /kinetics/PP1-active/Deph-thr306 1 0 0 0 0 0.99999 0.3432 1.4 \
  0.35 0 0 "" red cyan "" -25.538 3.7223 0
simundump kenz /kinetics/PP1-active/Deph-thr286c 1 0 0 0 0 0.99999 0.3432 1.4 \
  0.35 0 0 "" red cyan "" -30.334 -2.8165 0
simundump kenz /kinetics/PP1-active/Deph_thr286b 1 0 0 0 0 0.99999 0.3432 1.4 \
  0.35 0 0 "" red cyan "" -24.758 -1.1185 0
simundump group /kinetics/CaM 1 blue black x 0 0 "" defaultfile defaultfile.g \
  0 0 0 -45.327 -3.6101 0
simundump kpool /kinetics/CaM/CaM 1 25 20 20 20 20 25 0 0.99999 0 "" pink \
  blue -45.344 4.1096 0
simundump kreac /kinetics/CaM/CaM-TR2-bind-Ca 1 72.002 72 "" white blue \
  -43.165 3.4688 0
simundump kreac /kinetics/CaM/CaM-TR2-Ca2-bind-Ca 1 3.6 10 "" white blue \
  -44.169 1.6152 0
simundump kreac /kinetics/CaM/CaM-Ca3-bind-Ca 1 0.46501 10 "" white blue \
  -45.727 -1.3505 0
simundump kpool /kinetics/CaM/neurogranin-CaM 1 1.6667e-06 0 0 0 0 1.6667e-06 \
  0 0.99999 0 "" red blue -54.938 -4.1384 0
simundump kreac /kinetics/CaM/neurogranin-bind-CaM 1 0.3 1 "" white blue \
  -50.958 -3.7013 0
simundump kpool /kinetics/CaM/neurogranin* 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" red blue -47.87 -6.8058 0
simundump kpool /kinetics/CaM/neurogranin 1 10 10 10 9.9999 9.9999 9.9999 0 \
  0.99999 0 "" red blue -52.568 -6.8161 0
simundump kreac /kinetics/CaM/dephosph-neurogranin 1 0.005 0 "" white blue \
  -45.755 -5.1411 0
simundump kpool /kinetics/CaM-Ca3 1 1.6667e-06 0 0 0 0 1.6667e-06 0 0.99999 0 \
  "" hotpink yellow -41.838 -0.21314 0
simundump kpool /kinetics/CaM-TR2-Ca2 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" pink yellow -40.328 2.6476 0
simundump kpool /kinetics/CaM(Ca)n-CaNAB 1 0 0 0 0 0 0 0 0.99999 0 "" \
  darkblue yellow -30.524 -6.6543 0
simundump kenz /kinetics/CaM(Ca)n-CaNAB/dephosph_neurogranin 1 0 0 0 0 \
  0.99999 0.33361 2.67 0.67 0 0 "" red darkblue "" -51.047 -9.2181 0
simundump kenz /kinetics/CaM(Ca)n-CaNAB/dephosph_inhib1 1 0 0 0 0 0.99999 \
  0.342 1.36 0.34 0 0 "" red darkblue "" -42.742 -17.357 0
simundump kenz /kinetics/CaM(Ca)n-CaNAB/dephosph-PP1-I* 1 0 0 0 0 0.99999 \
  0.342 1.36 0.34 0 0 "" white darkblue "" -41.24 -6.4435 0
simundump group /kinetics/PP1 1 yellow black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 -46.362 -10.235 0
simundump kpool /kinetics/PP1/I1 1 1.8 1.8 1.8 1.8 1.8 1.8 0 0.99999 0 "" \
  orange yellow -38.013 -14.351 0
simundump kpool /kinetics/PP1/I1* 1 0.1 0.001 0.001 0.00099999 0.00099999 \
  0.099999 0 0.99999 0 "" orange yellow -42.158 -14.319 0
simundump kreac /kinetics/PP1/Inact-PP1 1 499.98 0.1 "" white yellow -45.403 \
  -12.417 0
simundump kpool /kinetics/PP1/PP1-I1* 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" brown yellow -43.747 -8.9641 0
simundump kpool /kinetics/PP1/PP1-I1 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" brown yellow -36.339 -8.6879 0
simundump kreac /kinetics/PP1/dissoc-PP1-I1 1 1 0 "" white yellow -42.498 \
  -12.141 0
simundump kpool /kinetics/PP2A 1 0.17 0.12 0.12 0.12 0.12 0.17 0 0.99999 0 "" \
  red black -36.52 -3.6325 0
simundump kenz /kinetics/PP2A/PP2A-dephosph-I1 1 0 0 0 0 0.99999 3.96 25 6 0 \
  0 "" red red "" -38.954 -10.663 0
simundump kenz /kinetics/PP2A/PP2A-dephosph-PP1-I* 1 0 0 0 0 0.99999 3.96 25 \
  6 0 0 "" red red "" -36.521 -4.7733 0
simundump kpool /kinetics/CaNAB-Ca4 1 1.6667e-06 0 0 0 0 1.6667e-06 0 0.99999 \
  0 "" tan yellow -24.923 -8.5717 0
simundump kenz /kinetics/CaNAB-Ca4/dephosph_inhib1_noCaM 1 0 0 0 0 0.99999 \
  0.0342 0.136 0.034 0 0 "" red tan "" -35.539 -10.496 0
simundump group /kinetics/PP2B 1 red3 black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 -20.052 -5.8546 0
simundump kpool /kinetics/PP2B/CaNAB 1 25 1 1 0.99999 0.99999 25 0 0.99999 0 \
  "" tan red3 -18.702 -8.4456 0
simundump kpool /kinetics/PP2B/CaNAB-Ca2 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" tan red3 -21.258 -8.5373 0
simundump kreac /kinetics/PP2B/Ca-bind-CaNAB-Ca2 1 3.6 1 "" white red3 \
  -22.826 -9.7525 0
simundump kreac /kinetics/PP2B/Ca-bind-CaNAB 1 10009 1 "" white red3 -20.125 \
  -9.8899 0
simundump kreac /kinetics/PP2B/CaM-Ca2-bind-CaNAB 1 0.24001 1 "" white red3 \
  -26.962 -9.3156 0
simundump kpool /kinetics/PP2B/CaMCa3-CaNAB 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" blue red3 -29.667 -10.754 0
simundump kpool /kinetics/PP2B/CaMCa2-CANAB 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" blue red3 -27.526 -10.689 0
simundump kpool /kinetics/PP2B/CaMCa4-CaNAB 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" blue red3 -30.997 -9.6161 0
simundump kreac /kinetics/PP2B/CaMCa3-bind-CaNAB 1 2.2381 1 "" white red3 \
  -27.594 -8.6073 0
simundump kreac /kinetics/PP2B/CaMCa4-bind-CaNAB 1 600.01 1 "" white red3 \
  -27.639 -7.6415 0
simundump group /kinetics/PKA 0 blue blue x 0 0 "" defaultfile defaultfile.g \
  0 0 0 -41.943 -20.667 0
simundump kpool /kinetics/PKA/R2C2 0 0.50001 0.50001 0.50001 0.5 0.5 0.5 0 \
  0.99999 0 "" white blue -46.656 -27.74 0
simundump kpool /kinetics/PKA/R2C2-cAMP 0 1 0 0 0 0 0.99999 0 0.99999 0 "" \
  white blue -43.694 -27.272 0
simundump kreac /kinetics/PKA/cAMP-bind-site-B1 0 54.001 33 "" white blue \
  -44.279 -31.015 0
simundump kreac /kinetics/PKA/cAMP-bind-site-B2 1 54.001 33 "" white blue \
  -41.937 -29.767 0
simundump kreac /kinetics/PKA/cAMP-bind-site-A1 1 75 110 "" white blue \
  -39.251 -30.952 0
simundump kreac /kinetics/PKA/cAMP-bind-site-A2 1 75 32.5 "" white blue \
  -35.964 -29.521 0
simundump kpool /kinetics/PKA/R2C2-cAMP2 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" white blue -40.146 -26.43 0
simundump kpool /kinetics/PKA/R2C2-cAMP3 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" white blue -37.46 -27.49 0
simundump kpool /kinetics/PKA/R2C2-cAMP4 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" white blue -35.074 -25.879 0
simundump kpool /kinetics/PKA/R2C-cAMP4 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" white blue -37.357 -24.745 0
simundump kpool /kinetics/PKA/R2-cAMP4 1 0.15 0 0 0 0 0.15 0 0.99999 0 "" \
  white blue -43.694 -25.182 0
simundump kreac /kinetics/PKA/Release-C1 1 60 18 "" white blue -35.361 \
  -22.877 0
simundump kreac /kinetics/PKA/Release-C2 1 60 18 "" white blue -40.232 \
  -24.155 0
simundump kpool /kinetics/PKA/PKA-inhibitor 1 1 0.25 0.25 0.25 0.25 0.99999 0 \
  0.99999 0 "" cyan blue -44.714 -23.288 0
simundump kreac /kinetics/PKA/inhib-PKA 1 60.001 1 "" white blue -41.921 \
  -22.664 0
simundump kpool /kinetics/PKA/inhibited-PKA 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" cyan blue -45.341 -21.689 0
simundump kpool /kinetics/cAMP 1 10 0 0 0 0 9.9999 0 0.99999 0 "" green black \
  -30.156 -32.591 0
simundump group /kinetics/AC 1 blue blue x 0 0 "" defaultfile defaultfile.g 0 \
  0 0 -17.529 -17.47 0
simundump kpool /kinetics/AC/ATP 1 5000.1 5000.1 5000.1 5000 5000 5000 0 \
  0.99999 4 "" red blue -18.042 -18.868 0
simundump kpool /kinetics/AC/AC1-CaM 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" orange blue -20.483 -17.259 0
simundump kenz /kinetics/AC/AC1-CaM/kenz 1 0 0 0 0 0.99999 4.5 72 18 0 1 "" \
  red orange "" -20.52 -18.394 0
simundump kpool /kinetics/AC/AC1 1 1 0.02 0.02 0.02 0.02 0.99999 0 0.99999 0 \
  "" orange blue -24.247 -15.394 0
simundump kreac /kinetics/AC/CaM-bind-AC1 1 50 1 "" white blue -22.762 -15.59 \
  0
simundump kpool /kinetics/AC/AC2* 1 0.01085 0 0 0 0 0.01085 0 0.99999 0 "" \
  yellow blue -18.647 -22.52 0
simundump kenz /kinetics/AC/AC2*/kenz 1 0 0 0 0 0.99999 1.74 28 7 0 1 "" red \
  yellow "" -18.774 -21.663 0
simundump kpool /kinetics/AC/AC2-Gs 1 1.6667e-06 0 0 0 0 1.6667e-06 0 0.99999 \
  0 "" yellow blue -21.486 -21.709 0
simundump kenz /kinetics/AC/AC2-Gs/kenz 1 0 0 0 0 0.99999 4.5 72 18 0 1 "" \
  red yellow "" -21.564 -20.701 0
simundump kpool /kinetics/AC/AC2 1 0.015 0.015 0.015 0.015 0.015 0.015 0 \
  0.99999 0 "" yellow blue -17.606 -24.303 0
simundump kreac /kinetics/AC/dephosph-AC2 1 0.1 0 "" white blue -19.759 \
  -25.108 0
simundump kpool /kinetics/AC/AC1-Gs 1 1.6667e-06 0 0 0 0 1.6667e-06 0 0.99999 \
  0 "" orange blue -22.92 -16.507 0
simundump kenz /kinetics/AC/AC1-Gs/kenz 1 0 0 0 0 1.6667e-06 4.5 72 18 0 1 "" \
  red orange "" -21.945 -17.655 0
simundump kreac /kinetics/AC/Gs-bind-AC2 1 500 1 "" white blue -20.17 -27.142 \
  0
simundump kreac /kinetics/AC/Gs-bind-AC1 1 126 1 "" white blue -24.879 \
  -16.883 0
simundump kpool /kinetics/AC/AMP 1 3.2548e+05 3.2548e+05 3.2548e+05 0.54248 \
  0.54248 0.54248 0 1.6667e-06 0 "" pink blue -23.649 -17.47 0
simundump kpool /kinetics/AC/AC2*-Gs 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" green blue -20.142 -22.141 0
simundump kenz /kinetics/AC/AC2*-Gs/kenz 1 0 0 0 0 1.6667e-06 4.5 216 54 0 1 \
  "" red green "" -20.066 -21.087 0
simundump kreac /kinetics/AC/Gs-bind-AC2* 1 833.29 1 "" white blue -20.343 \
  -23.991 0
simundump kpool /kinetics/AC/cAMP-PDE 1 1 0.45 0.45 0.45 0.45 0.99999 0 \
  0.99999 0 "" green blue -26.712 -15.696 0
simundump kenz /kinetics/AC/cAMP-PDE/PDE 1 0 0 0 0 0.99999 2.52 40 10 0 0 "" \
  red green "" -26.821 -23.131 0
simundump kpool /kinetics/AC/cAMP-PDE* 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" green blue -26.685 -17.78 0
simundump kenz /kinetics/AC/cAMP-PDE*/PDE* 1 0 0 0 0 0.99999 5.0401 80 20 0 0 \
  "" red green "" -25.438 -22.305 0
simundump kreac /kinetics/AC/dephosph-PDE 1 0.1 0 "" white blue -28.587 \
  -18.842 0
simundump kpool /kinetics/AC/PDE1 1 2 2 2 2 2 2 0 0.99999 0 "" green blue \
  -30.493 -12.115 0
simundump kenz /kinetics/AC/PDE1/PDE1 1 0 0 0 0 0.99999 0.21 6.67 1.667 0 0 \
  "" red green "" -27.426 -22.069 0
simundump kpool /kinetics/AC/CaM.PDE1 1 1.6667e-06 0 0 0 0 1.6667e-06 0 \
  0.99999 0 "" green blue -30.493 -14.85 0
simundump kenz /kinetics/AC/CaM.PDE1/CaM.PDE1 1 0 0 0 0 0.99999 1.26 40 10 0 \
  0 "" red green "" -28.333 -21.282 0
simundump kreac /kinetics/AC/CaM_bind_PDE1 1 720.01 5 "" white blue -27.28 \
  -13.293 0
simundump kpool /kinetics/Gs-alpha 1 0.01 0 0 0 0 0.0099999 0 0.99999 0 "" \
  red black -23.677 -28.03 0
simundump kpool /kinetics/Ca 1 0.08 0.08 0.08 0.079999 0.079999 0.079999 0 \
  0.99999 0 "" red black -37.661 -0.21314 0
simundump kreac /kinetics/Ras-act-unphosph-raf 0 6.0001 1 "" white black 4 8 \
  0
simundump xtab /kinetics/Ca_tab 0 -3000 5.0186 1 2 0 "" edit_xtab white red 0 \
  0 0 1 -31 6 0
loadtab /kinetics/Ca_tab table 0 1000 0 100 \
 0.0799992 0.549994 0.975264 1.36006 1.70825 2.02329 2.30835 2.5663 2.79969 \
 3.01087 3.20196 3.37486 3.53131 3.67286 3.80095 3.91686 4.02173 4.11661 \
 4.20248 4.28017 4.35047 4.41407 4.47162 4.5237 4.57082 4.61345 4.65204 \
 4.68694 4.71853 4.74711 4.77297 4.79637 4.81755 4.83671 4.85404 4.86973 \
 4.88392 4.89676 4.90838 4.9189 4.92841 4.93702 4.94481 4.95185 4.95823 \
 4.96399 4.96922 4.97395 4.97822 4.98208 4.98558 4.98875 4.99162 4.99422 \
 4.99656 4.99868 5.00061 5.00234 5.00392 5.00534 5.00663 5.00778 5.00884 \
 5.00979 5.01066 5.01144 5.01215 5.01278 5.01336 5.01388 5.01436 5.01479 \
 5.01518 5.01553 5.01585 5.01614 5.01639 5.01663 5.01684 5.01704 5.01721 \
 5.01737 5.01751 5.01764 5.01775 5.01786 5.01796 5.01804 5.01812 5.01819 \
 5.01826 5.01832 5.01837 5.01842 5.01846 5.01849 5.01853 5.01856 5.01859 \
 5.01862 5.01864 4.54867 4.12341 3.73864 3.39047 3.07544 2.79038 2.53245 \
 2.29908 2.0879 1.89683 1.72393 1.56748 1.42594 1.29785 1.18195 1.07709 \
 0.982208 0.896351 0.818665 0.748371 0.684768 0.627216 0.575142 0.528022 \
 0.485387 0.44681 0.411904 0.380318 0.351739 0.32588 0.302481 0.281309 \
 0.262152 0.244818 0.229133 0.214941 0.2021 0.19048 0.179967 0.170454 \
 0.161846 0.154057 0.14701 0.140633 0.134863 0.129642 0.124918 0.120643 \
 0.116775 0.113276 0.110109 0.107244 0.104651 0.102305 0.100182 0.0982616 \
 0.0965237 0.0949512 0.0935283 0.0922409 0.0910759 0.0900219 0.089068 \
 0.088205 0.0874241 0.0867176 0.0860782 0.0854997 0.0849763 0.0845027 \
 0.084074 0.0836863 0.0833354 0.0830179 0.0827307 0.0824707 0.0822355 \
 0.0820227 0.0818302 0.081656 0.0814983 0.0813556 0.0812265 0.0811097 \
 0.0810041 0.0809084 0.0808219 0.0807436 0.0806727 0.0806086 0.0805506 \
 0.0804982 0.0804507 0.0804077 0.0803689 0.0803336 0.0803019 0.0802731 \
 0.080247 
loadtab -continue \
 0.0802234 0.0802021 0.0801828 0.0801653 0.0801495 0.0801352 0.0801222 \
 0.0801105 0.0801 0.0800903 0.0800816 0.0800739 0.0800668 0.0800603 0.0800545 \
 0.0800492 0.0800444 0.0800402 0.0800363 0.0800328 0.0800295 0.0800266 \
 0.0800241 0.0800216 0.0800195 0.0800176 0.0800159 0.0800143 0.0800129 \
 0.0800115 0.0800103 0.0800093 0.0800083 0.0800074 0.0800066 0.080006 \
 0.0800053 0.0800048 0.0800042 0.0800038 0.0800033 0.0800029 0.0800025 \
 0.0800022 0.080002 0.0800016 0.0800014 0.0800012 0.0800011 0.0800009 \
 0.0800008 0.0800005 0.0800004 0.0800003 0.0800002 0.0800001 0.08 0.08 \
 0.0799999 0.0799999 0.0799998 0.0799998 0.0799996 0.0799996 0.0799995 \
 0.0799995 0.0799995 0.0799994 0.0799994 0.0799994 0.0799994 0.0799994 \
 0.0799994 0.0799993 0.0799993 0.0799993 0.0799993 0.0799993 0.0799993 \
 0.0799993 0.0799993 0.0799993 0.0799993 0.0799993 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 
loadtab -continue \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 
loadtab -continue \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 
loadtab -continue \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 0.0799992 \
 0.0799992 0.0799992 0.0799992 0.0799992 
loadtab -continue \
 0.0799992
simundump kpool /kinetics/Ca_input 0 0 0 0 0 0 0 0 0.99999 2 "" 62 black -36 \
  4 0
simundump kreac /kinetics/Ca_stoch 0 100 100 "" white black -38 2 0
simundump xgraph /graphs/conc1 0 0 2500 0 46.331 0
simundump xgraph /graphs/conc2 0 0 2500 0 1 0
simundump xplot /graphs/conc1/CaMKII-CaM.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" palegreen 0 0 1
simundump xplot /graphs/conc1/Ca.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" red 0 0 1
simundump xplot /graphs/conc1/AA.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" darkgreen 0 0 1
simundump xplot /graphs/conc1/IP3.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" pink 0 0 1
simundump xplot /graphs/conc1/CaMKII-thr286*-CaM.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" palegreen 0 0 1
simundump xplot /graphs/conc1/CaMKII-thr286.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" red 0 0 1
simundump xplot /graphs/conc1/CaMKII***.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" cyan 0 0 1
simundump xplot /graphs/conc2/MAPK*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" orange 0 0 1
simundump xplot /graphs/conc2/PKA-active.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" yellow 0 0 1
simundump xplot /graphs/conc2/PKC-active.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" red 0 0 1
simundump xplot /graphs/conc2/PP1-active.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" cyan 0 0 1
simundump xplot /graphs/conc2/MKP-1.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" hotpink 0 0 1
simundump xplot /graphs/conc2/CaM-Ca4.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xgraph /moregraphs/conc3 0 0 2500 -1.1176e-08 1 0
simundump xgraph /moregraphs/conc4 0 0 2500 0 1 0
simundump xcoredraw /edit/draw 0 -43.443 8.0515 -21.089 30.428
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
xtextload /file/notes \
"Almost samee as nonscaf_syn3c.g, shifted Ca stimulus back" \
"one stage via an intermediate pool which feeds into Ca via a" \
"rather rapid reaction step. This ensures that the" \
"Ca stimulus itself will undergo stochastic fluctuations approximating" \
"the buffered condition of free Ca in the cell." \
"The kinetics of the Ca stimulus have been accelerated to " \
"100 msec to more closely match those in Sabatini et al Neuron 2003" \
"and Majewska et al J Neurosci 2000"
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
addmsg /kinetics/PKC-active/PKC-phosph-neurogranin /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active/PKC-phosph-ng-CaM /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active/phosph-AC2 /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-act-raf ENZYME n 
addmsg /kinetics/MAPK/craf-1 /kinetics/PKC-active/PKC-act-raf SUBSTRATE n 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-inact-GAP ENZYME n 
addmsg /kinetics/Ras/GAP /kinetics/PKC-active/PKC-inact-GAP SUBSTRATE n 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-act-GEF ENZYME n 
addmsg /kinetics/Ras/inact-GEF /kinetics/PKC-active/PKC-act-GEF SUBSTRATE n 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-phosph-neurogranin ENZYME n 
addmsg /kinetics/CaM/neurogranin /kinetics/PKC-active/PKC-phosph-neurogranin SUBSTRATE n 
addmsg /kinetics/PKC-active /kinetics/PKC-active/PKC-phosph-ng-CaM ENZYME n 
addmsg /kinetics/CaM/neurogranin-CaM /kinetics/PKC-active/PKC-phosph-ng-CaM SUBSTRATE n 
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
addmsg /kinetics/MKP-1/MKP1-tyr-deph /kinetics/MKP-1 REAC eA B 
addmsg /kinetics/MKP-1/MKP1-thr-deph /kinetics/MKP-1 REAC eA B 
addmsg /kinetics/MKP-1 /kinetics/MKP-1/MKP1-tyr-deph ENZYME n 
addmsg /kinetics/MAPK/MAPK-tyr /kinetics/MKP-1/MKP1-tyr-deph SUBSTRATE n 
addmsg /kinetics/MKP-1 /kinetics/MKP-1/MKP1-thr-deph ENZYME n 
addmsg /kinetics/MAPK* /kinetics/MKP-1/MKP1-thr-deph SUBSTRATE n 
addmsg /kinetics/MAPK/craf-1* /kinetics/Ras-act-craf SUBSTRATE n 
addmsg /kinetics/Ras/GTP-Ras /kinetics/Ras-act-craf SUBSTRATE n 
addmsg /kinetics/MAPK/Raf*-GTP-Ras /kinetics/Ras-act-craf PRODUCT n 
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
addmsg /kinetics/Ras-act-craf /kinetics/Ras/GTP-Ras REAC A B 
addmsg /kinetics/Shc*.Sos.Grb2/Sos.Ras_GEF /kinetics/Ras/GTP-Ras MM_PRD pA 
addmsg /kinetics/Ras-act-unphosph-raf /kinetics/Ras/GTP-Ras REAC A B 
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
addmsg /kinetics/CaM/neurogranin-bind-CaM /kinetics/CaM/CaM REAC A B 
addmsg /kinetics/PKC-active/PKC-phosph-ng-CaM /kinetics/CaM/CaM MM_PRD pA 
addmsg /kinetics/CaM/CaM /kinetics/CaM/CaM-TR2-bind-Ca SUBSTRATE n 
addmsg /kinetics/CaM-TR2-Ca2 /kinetics/CaM/CaM-TR2-bind-Ca PRODUCT n 
addmsg /kinetics/Ca /kinetics/CaM/CaM-TR2-bind-Ca SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/CaM/CaM-TR2-bind-Ca SUBSTRATE n 
addmsg /kinetics/CaM-TR2-Ca2 /kinetics/CaM/CaM-TR2-Ca2-bind-Ca SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/CaM/CaM-TR2-Ca2-bind-Ca SUBSTRATE n 
addmsg /kinetics/CaM-Ca3 /kinetics/CaM/CaM-TR2-Ca2-bind-Ca PRODUCT n 
addmsg /kinetics/CaM-Ca3 /kinetics/CaM/CaM-Ca3-bind-Ca SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/CaM/CaM-Ca3-bind-Ca SUBSTRATE n 
addmsg /kinetics/CaM-Ca4 /kinetics/CaM/CaM-Ca3-bind-Ca PRODUCT n 
addmsg /kinetics/CaM/neurogranin-bind-CaM /kinetics/CaM/neurogranin-CaM REAC B A 
addmsg /kinetics/PKC-active/PKC-phosph-ng-CaM /kinetics/CaM/neurogranin-CaM REAC sA B 
addmsg /kinetics/CaM/neurogranin /kinetics/CaM/neurogranin-bind-CaM SUBSTRATE n 
addmsg /kinetics/CaM/neurogranin-CaM /kinetics/CaM/neurogranin-bind-CaM PRODUCT n 
addmsg /kinetics/CaM/CaM /kinetics/CaM/neurogranin-bind-CaM SUBSTRATE n 
addmsg /kinetics/CaM(Ca)n-CaNAB/dephosph_neurogranin /kinetics/CaM/neurogranin* REAC sA B 
addmsg /kinetics/PKC-active/PKC-phosph-neurogranin /kinetics/CaM/neurogranin* MM_PRD pA 
addmsg /kinetics/PKC-active/PKC-phosph-ng-CaM /kinetics/CaM/neurogranin* MM_PRD pA 
addmsg /kinetics/CaM/dephosph-neurogranin /kinetics/CaM/neurogranin* REAC A B 
addmsg /kinetics/CaM(Ca)n-CaNAB/dephosph_neurogranin /kinetics/CaM/neurogranin MM_PRD pA 
addmsg /kinetics/CaM/neurogranin-bind-CaM /kinetics/CaM/neurogranin REAC A B 
addmsg /kinetics/PKC-active/PKC-phosph-neurogranin /kinetics/CaM/neurogranin REAC sA B 
addmsg /kinetics/CaM/dephosph-neurogranin /kinetics/CaM/neurogranin REAC B A 
addmsg /kinetics/CaM/neurogranin* /kinetics/CaM/dephosph-neurogranin SUBSTRATE n 
addmsg /kinetics/CaM/neurogranin /kinetics/CaM/dephosph-neurogranin PRODUCT n 
addmsg /kinetics/CaM/CaM-TR2-Ca2-bind-Ca /kinetics/CaM-Ca3 REAC B A 
addmsg /kinetics/CaM/CaM-Ca3-bind-Ca /kinetics/CaM-Ca3 REAC A B 
addmsg /kinetics/PP2B/CaMCa3-bind-CaNAB /kinetics/CaM-Ca3 REAC A B 
addmsg /kinetics/CaM/CaM-TR2-bind-Ca /kinetics/CaM-TR2-Ca2 REAC B A 
addmsg /kinetics/CaM/CaM-TR2-Ca2-bind-Ca /kinetics/CaM-TR2-Ca2 REAC A B 
addmsg /kinetics/PP2B/CaM-Ca2-bind-CaNAB /kinetics/CaM-TR2-Ca2 REAC A B 
addmsg /kinetics/CaM(Ca)n-CaNAB/dephosph_neurogranin /kinetics/CaM(Ca)n-CaNAB REAC eA B 
addmsg /kinetics/CaM(Ca)n-CaNAB/dephosph_inhib1 /kinetics/CaM(Ca)n-CaNAB REAC eA B 
addmsg /kinetics/CaM(Ca)n-CaNAB/dephosph-PP1-I* /kinetics/CaM(Ca)n-CaNAB REAC eA B 
addmsg /kinetics/PP2B/CaMCa4-CaNAB /kinetics/CaM(Ca)n-CaNAB SUMTOTAL n nInit 
addmsg /kinetics/PP2B/CaMCa3-CaNAB /kinetics/CaM(Ca)n-CaNAB SUMTOTAL n nInit 
addmsg /kinetics/PP2B/CaMCa2-CANAB /kinetics/CaM(Ca)n-CaNAB SUMTOTAL n nInit 
addmsg /kinetics/CaM(Ca)n-CaNAB /kinetics/CaM(Ca)n-CaNAB/dephosph_neurogranin ENZYME n 
addmsg /kinetics/CaM/neurogranin* /kinetics/CaM(Ca)n-CaNAB/dephosph_neurogranin SUBSTRATE n 
addmsg /kinetics/CaM(Ca)n-CaNAB /kinetics/CaM(Ca)n-CaNAB/dephosph_inhib1 ENZYME n 
addmsg /kinetics/PP1/I1* /kinetics/CaM(Ca)n-CaNAB/dephosph_inhib1 SUBSTRATE n 
addmsg /kinetics/CaM(Ca)n-CaNAB /kinetics/CaM(Ca)n-CaNAB/dephosph-PP1-I* ENZYME n 
addmsg /kinetics/PP1/PP1-I1* /kinetics/CaM(Ca)n-CaNAB/dephosph-PP1-I* SUBSTRATE n 
addmsg /kinetics/PKA-active/PKA-phosph-I1 /kinetics/PP1/I1 REAC sA B 
addmsg /kinetics/CaNAB-Ca4/dephosph_inhib1_noCaM /kinetics/PP1/I1 MM_PRD pA 
addmsg /kinetics/CaM(Ca)n-CaNAB/dephosph_inhib1 /kinetics/PP1/I1 MM_PRD pA 
addmsg /kinetics/PP2A/PP2A-dephosph-I1 /kinetics/PP1/I1 MM_PRD pA 
addmsg /kinetics/PP1/dissoc-PP1-I1 /kinetics/PP1/I1 REAC B A 
addmsg /kinetics/PP1/Inact-PP1 /kinetics/PP1/I1* REAC A B 
addmsg /kinetics/PKA-active/PKA-phosph-I1 /kinetics/PP1/I1* MM_PRD pA 
addmsg /kinetics/CaNAB-Ca4/dephosph_inhib1_noCaM /kinetics/PP1/I1* REAC sA B 
addmsg /kinetics/CaM(Ca)n-CaNAB/dephosph_inhib1 /kinetics/PP1/I1* REAC sA B 
addmsg /kinetics/PP2A/PP2A-dephosph-I1 /kinetics/PP1/I1* REAC sA B 
addmsg /kinetics/PP1/PP1-I1* /kinetics/PP1/Inact-PP1 PRODUCT n 
addmsg /kinetics/PP1/I1* /kinetics/PP1/Inact-PP1 SUBSTRATE n 
addmsg /kinetics/PP1-active /kinetics/PP1/Inact-PP1 SUBSTRATE n 
addmsg /kinetics/PP1/Inact-PP1 /kinetics/PP1/PP1-I1* REAC B A 
addmsg /kinetics/PP2A/PP2A-dephosph-PP1-I* /kinetics/PP1/PP1-I1* REAC sA B 
addmsg /kinetics/CaM(Ca)n-CaNAB/dephosph-PP1-I* /kinetics/PP1/PP1-I1* REAC sA B 
addmsg /kinetics/PP1/dissoc-PP1-I1 /kinetics/PP1/PP1-I1 REAC A B 
addmsg /kinetics/PP2A/PP2A-dephosph-PP1-I* /kinetics/PP1/PP1-I1 MM_PRD pA 
addmsg /kinetics/CaM(Ca)n-CaNAB/dephosph-PP1-I* /kinetics/PP1/PP1-I1 MM_PRD pA 
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
addmsg /kinetics/PP2B/CaM-Ca2-bind-CaNAB /kinetics/CaNAB-Ca4 REAC A B 
addmsg /kinetics/PP2B/CaMCa4-bind-CaNAB /kinetics/CaNAB-Ca4 REAC A B 
addmsg /kinetics/PP2B/CaMCa3-bind-CaNAB /kinetics/CaNAB-Ca4 REAC A B 
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
addmsg /kinetics/CaNAB-Ca4 /kinetics/PP2B/CaM-Ca2-bind-CaNAB SUBSTRATE n 
addmsg /kinetics/CaM-TR2-Ca2 /kinetics/PP2B/CaM-Ca2-bind-CaNAB SUBSTRATE n 
addmsg /kinetics/PP2B/CaMCa2-CANAB /kinetics/PP2B/CaM-Ca2-bind-CaNAB PRODUCT n 
addmsg /kinetics/PP2B/CaMCa3-bind-CaNAB /kinetics/PP2B/CaMCa3-CaNAB REAC B A 
addmsg /kinetics/PP2B/CaM-Ca2-bind-CaNAB /kinetics/PP2B/CaMCa2-CANAB REAC B A 
addmsg /kinetics/PP2B/CaMCa4-bind-CaNAB /kinetics/PP2B/CaMCa4-CaNAB REAC B A 
addmsg /kinetics/PP2B/CaMCa3-CaNAB /kinetics/PP2B/CaMCa3-bind-CaNAB PRODUCT n 
addmsg /kinetics/CaM-Ca3 /kinetics/PP2B/CaMCa3-bind-CaNAB SUBSTRATE n 
addmsg /kinetics/CaNAB-Ca4 /kinetics/PP2B/CaMCa3-bind-CaNAB SUBSTRATE n 
addmsg /kinetics/PP2B/CaMCa4-CaNAB /kinetics/PP2B/CaMCa4-bind-CaNAB PRODUCT n 
addmsg /kinetics/CaM-Ca4 /kinetics/PP2B/CaMCa4-bind-CaNAB SUBSTRATE n 
addmsg /kinetics/CaNAB-Ca4 /kinetics/PP2B/CaMCa4-bind-CaNAB SUBSTRATE n 
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
addmsg /kinetics/MAPK/craf-1 /kinetics/Ras-act-unphosph-raf SUBSTRATE n 
addmsg /kinetics/MAPK/Raf-GTP-Ras /kinetics/Ras-act-unphosph-raf PRODUCT n 
addmsg /kinetics/Ras/GTP-Ras /kinetics/Ras-act-unphosph-raf SUBSTRATE n 
addmsg /kinetics/Ca_stoch /kinetics/Ca_input REAC A B 
addmsg /kinetics/Ca_tab /kinetics/Ca_input SUMTOTAL output output 
addmsg /kinetics/Ca_input /kinetics/Ca_stoch SUBSTRATE n 
addmsg /kinetics/Ca /kinetics/Ca_stoch PRODUCT n 
addmsg /kinetics/CaMKII/CaMKII-CaM /graphs/conc1/CaMKII-CaM.Co PLOT Co *CaMKII-CaM.Co *palegreen 
addmsg /kinetics/Ca /graphs/conc1/Ca.Co PLOT Co *Ca.Co *red 
addmsg /kinetics/AA /graphs/conc1/AA.Co PLOT Co *AA.Co *darkgreen 
addmsg /kinetics/IP3 /graphs/conc1/IP3.Co PLOT Co *IP3.Co *pink 
addmsg /kinetics/CaMKII/CaMKII-thr286*-CaM /graphs/conc1/CaMKII-thr286*-CaM.Co PLOT Co *CaMKII-thr286*-CaM.Co *palegreen 
addmsg /kinetics/CaMKII/CaMKII-thr286 /graphs/conc1/CaMKII-thr286.Co PLOT Co *CaMKII-thr286.Co *red 
addmsg /kinetics/CaMKII/CaMKII*** /graphs/conc1/CaMKII***.Co PLOT Co *CaMKII***.Co *cyan 
addmsg /kinetics/MAPK* /graphs/conc2/MAPK*.Co PLOT Co *MAPK*.Co *orange 
addmsg /kinetics/PKA-active /graphs/conc2/PKA-active.Co PLOT Co *PKA-active.Co *yellow 
addmsg /kinetics/PKC-active /graphs/conc2/PKC-active.Co PLOT Co *PKC-active.Co *red 
addmsg /kinetics/PP1-active /graphs/conc2/PP1-active.Co PLOT Co *PP1-active.Co *cyan 
addmsg /kinetics/MKP-1 /graphs/conc2/MKP-1.Co PLOT Co *MKP-1.Co *hotpink 
addmsg /kinetics/CaM-Ca4 /graphs/conc2/CaM-Ca4.Co PLOT Co *CaM-Ca4.Co *blue 
enddump
// End of dump

setfield /kinetics/Ca_tab table->dx 0.01
setfield /kinetics/Ca_tab table->invdx 100
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
call /kinetics/PKC-active/PKC-phosph-neurogranin/notes LOAD \
"Rates from Huang et al ABB 305:2 570-580 1993"
call /kinetics/PKC-active/PKC-phosph-ng-CaM/notes LOAD \
"Rates are 60% those of PKC-phosph-neurogranin. See" \
"Huang et al ABB 305:2 570-580 1993"
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
call /kinetics/Ras/inact-GEF*/notes LOAD \
"Phosphorylation-inactivated form of GEF. See" \
"Hordijk et al JBC 269:5 3534-3538 1994" \
"and " \
"Buregering et al EMBO J 12:11 4211-4220 1993" \
""
call /kinetics/Ras/CaM-bind-GEF/notes LOAD \
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
call /kinetics/CaM/neurogranin-bind-CaM/notes LOAD \
"Surprisingly, no direct info on rates from neurogranin at this time." \
"These rates are based on GAP-43 binding studies. As GAP-43 and " \
"neurogranin share near identity in the CaM/PKC binding regions, and also" \
"similarity in phosph and dephosph rates, I am borrowing GAP-43 " \
"kinetic info." \
"See Alexander et al JBC 262:13 6108-6113 1987"
call /kinetics/CaM/neurogranin*/notes LOAD \
"The phosph form does not bind CaM (Huang et al ABB 305:2 570-580 1993)"
call /kinetics/CaM/neurogranin/notes LOAD \
"Also known as RC3 and p17 and BICKS." \
"Conc in brain  >> 2 uM from Martzen and Slemmon J neurosci 64 92-100 1995" \
"but others say less without any #s. Conc in dend spines is much higher than " \
"overall, so it could be anywhere from 2 uM to 50. We will estimate" \
"10 uM as a starting point." \
"Gerendasy et al JBC 269:35 22420-22426 1994 have a skeleton model (no" \
"numbers) indicating CaM-Ca(n) binding ...."
call /kinetics/CaM/dephosph-neurogranin/notes LOAD \
"This is put in to keep the basal levels of neurogranin* experimentally" \
"reasonable. From various papers, specially Ramakers et al JBC 270:23 1995" \
"13892-13898," \
" it looks like the basal level of phosph is between 20 and 40%. I est" \
"around 25 % The kf of 0.005 gives around this level at basal PKC" \
"activity levels of 0.1 uM active PKC."
call /kinetics/CaM-TR2-Ca2/notes LOAD \
"This is the intermediate where the TR2 end (the high-affinity end) has" \
"bound the Ca but the TR1 end has not."
call /kinetics/CaM(Ca)n-CaNAB/dephosph_neurogranin/notes LOAD \
"From Seki et al ABB 316(2):673-679"
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
"species, average for mammalian brain is around 1 uM."
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
call /kinetics/PP2B/CaM-Ca2-bind-CaNAB/notes LOAD \
"Disabled. See notes for PP2B7.g" \
""
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
call /kinetics/Ras-act-unphosph-raf/notes LOAD \
"18 May 2003. This reaction is here to" \
"provide basal activity for MAPK as well" \
"as the potential for direct EGF stimulus" \
"without PKC activation." \
"Based on model from FB/fb28c.g: the model" \
"used for MKP-1 turnover. The rates there" \
"were constrained by basal activity values."
call /kinetics/Ca_tab/notes LOAD \
"This stimulus was generated using a target amplitude of 5 uM Ca," \
"with tau 0.1 seconds for each simulated pulse at 100Hz for 1 second." \
"The simulated volume is 1.5e-18 m^3 but this table works in terms" \
"of number of molecules. The stimulus parameters are loosely based on" \
"the Calcium parameters of Sabatini et al Neuron 2002 and" \
"Majewska et al J. Neurosci 2000"
complete_loading
