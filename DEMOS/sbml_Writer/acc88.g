//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Thu Feb 12 16:49:18 2009
 
include kkit {argv 1}
 
FASTDT = 2e-06
SIMDT = 0.001
CONTROLDT = 1
PLOTDT = 10
MAXTIME = 1000
TRANSIENT_TIME = 2
VARIABLE_DT_FLAG = 1
DEFAULT_VOL = 1e-15
VERSION = 11.0
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
simobjdump doqcsinfo filename accessname accesstype transcriber developer \
  citation species tissue cellcompartment methodology sources \
  model_implementation model_validation editfunc
simundump geometry /kinetics/geometry 0 1e-15 3 sphere "" white black 0 0 0
simundump text /kinetics/notes 0 ""
call /kinetics/notes LOAD \
""
simundump text /kinetics/geometry/notes 0 ""
call /kinetics/geometry/notes LOAD \
""
simundump kpool /kinetics/PDK1 0 0 1 1 6e+05 6e+05 0 0 6e+05 0 \
  /kinetics/geometry 37 black 24 28 0
simundump text /kinetics/PDK1/notes 0 ""
call /kinetics/PDK1/notes LOAD \
""
simundump kenz /kinetics/PDK1/S6K_phospho 0 0 0 0 0 6e+05 8.3333e-07 4 1 0 0 \
  "" red 37 "" 24 26 0
simundump text /kinetics/PDK1/S6K_phospho/notes 0 ""
call /kinetics/PDK1/S6K_phospho/notes LOAD \
""
simundump kpool /kinetics/PP2A 0 0 0.15 0.15 90000 90000 0 0 6e+05 0 \
  /kinetics/geometry 4 black 24 12 0
simundump text /kinetics/PP2A/notes 0 "Protein Phosphatase"
call /kinetics/PP2A/notes LOAD \
"Protein Phosphatase"
simundump kenz /kinetics/PP2A/dephos_clus_S6K 0 0 0 0 0 5.9998e+05 9.4696e-07 \
  4 1 0 0 "" red 25 "" 10 0 0
simundump text /kinetics/PP2A/dephos_clus_S6K/notes 0 ""
call /kinetics/PP2A/dephos_clus_S6K/notes LOAD \
""
simundump kenz /kinetics/PP2A/dephos_S6K 0 0 0 0 0 5.9998e+05 9.4696e-07 4 1 \
  0 0 "" red 25 "" 22 9 0
simundump text /kinetics/PP2A/dephos_S6K/notes 0 ""
call /kinetics/PP2A/dephos_S6K/notes LOAD \
""
simundump kenz /kinetics/PP2A/dephosp_S6K 0 0 0 0 0 5.9998e+05 9.4696e-07 4 1 \
  0 0 "" red 25 "" 26 9 0
simundump text /kinetics/PP2A/dephosp_S6K/notes 0 ""
call /kinetics/PP2A/dephosp_S6K/notes LOAD \
""
simundump kpool /kinetics/Rheb-GTP 0 0 1 1 6e+05 6e+05 0 0 6e+05 0 \
  /kinetics/geometry 28 black 32 11 0
simundump text /kinetics/Rheb-GTP/notes 0 ""
call /kinetics/Rheb-GTP/notes LOAD \
""
simundump group /kinetics/mTOR_model 0 8 black x 0 0 "" mTOR_model \
  defaultfile.g 0 cb4f9df2959ba465dbdd4a6b7a857a4c 0 20 9 0
simundump text /kinetics/mTOR_model/notes 0 ""
call /kinetics/mTOR_model/notes LOAD \
""
simundump kreac /kinetics/mTOR_model/Rheb-GTP_bind_TORclx 0 1e-05 3 "" white \
  8 31 2 0
simundump text /kinetics/mTOR_model/Rheb-GTP_bind_TORclx/notes 0 ""
call /kinetics/mTOR_model/Rheb-GTP_bind_TORclx/notes LOAD \
""
simundump kpool /kinetics/mTOR_model/TOR-clx 0 0 0.6 0.6 3.6e+05 3.6e+05 0 0 \
  6e+05 0 /kinetics/geometry 25 8 23 -2 0
simundump text /kinetics/mTOR_model/TOR-clx/notes 0 ""
call /kinetics/mTOR_model/TOR-clx/notes LOAD \
""
simundump kpool /kinetics/TOR_Rheb-GTP_clx 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry 44 black 31 -5 0
simundump text /kinetics/TOR_Rheb-GTP_clx/notes 0 ""
call /kinetics/TOR_Rheb-GTP_clx/notes LOAD \
""
simundump kenz /kinetics/TOR_Rheb-GTP_clx/S6K_phospho 0 0 0 0 0 6e+05 \
  6.2502e-07 0.24 0.06 0 0 "" red 42 "" 31 -7 0
simundump text /kinetics/TOR_Rheb-GTP_clx/S6K_phospho/notes 0 ""
call /kinetics/TOR_Rheb-GTP_clx/S6K_phospho/notes LOAD \
""
simundump group /kinetics/S6Kinase 0 6 black x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 21 20 0
simundump text /kinetics/S6Kinase/notes 0 ""
call /kinetics/S6Kinase/notes LOAD \
""
simundump kpool /kinetics/S6Kinase/S6K* 0 0 0 0 0 0 0 0 5.9998e+05 0 \
  /kinetics/geometry 46 6 16 15 0
simundump text /kinetics/S6Kinase/S6K*/notes 0 "\n\n"
call /kinetics/S6Kinase/S6K*/notes LOAD \
"" \
"" \
""
simundump kpool /kinetics/S6Kinase/S6K 0 0 1.25 1.25 7.5e+05 7.5e+05 0 0 \
  6e+05 0 /kinetics/geometry Pink 6 3 9 0
simundump text /kinetics/S6Kinase/S6K/notes 0 ""
call /kinetics/S6Kinase/S6K/notes LOAD \
""
simundump kpool /kinetics/S6Kinase/S6K_thr-412 0 0 0 0 0 0 0 0 5.9998e+05 0 \
  /kinetics/geometry 48 6 15 -8 0
simundump text /kinetics/S6Kinase/S6K_thr-412/notes 0 ""
call /kinetics/S6Kinase/S6K_thr-412/notes LOAD \
""
simundump kenz /kinetics/S6Kinase/S6K_thr-412/S6_phos 0 0 0 0 0 5.9998e+05 \
  3.3334e-07 0.04 0.01 0 0 "" red 48 "" 4 -12 0
simundump text /kinetics/S6Kinase/S6K_thr-412/S6_phos/notes 0 ""
call /kinetics/S6Kinase/S6K_thr-412/S6_phos/notes LOAD \
""
simundump kreac /kinetics/S6Kinase/S6_dephosph 0 0.01 0 "" white 6 27 16 0
simundump text /kinetics/S6Kinase/S6_dephosph/notes 0 ""
call /kinetics/S6Kinase/S6_dephosph/notes LOAD \
""
simundump kpool /kinetics/S6Kinase/40S_inact 0 0 0.02 0.02 12000 12000 0 0 \
  6e+05 4 /kinetics/geometry 3 6 29 23 0
simundump text /kinetics/S6Kinase/40S_inact/notes 0 \
  "Inactivated form of S6K\n"
call /kinetics/S6Kinase/40S_inact/notes LOAD \
"Inactivated form of S6K" \
""
simundump kpool /kinetics/S6K_tot 0 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry \
  44 black 10 14 0
simundump text /kinetics/S6K_tot/notes 0 ""
call /kinetics/S6K_tot/notes LOAD \
""
simundump kpool /kinetics/40S 0 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry 7 \
  black 0 17 0
simundump text /kinetics/40S/notes 0 "Activated form of S6\n"
call /kinetics/40S/notes LOAD \
"Activated form of S6" \
""
simundump kpool /kinetics/S6K_thr-252 0 0 0 0 0 0 0 0 6e+05 0 \
  /kinetics/geometry 6 black 7 28 0
simundump text /kinetics/S6K_thr-252/notes 0 ""
call /kinetics/S6K_thr-252/notes LOAD \
""
simundump kenz /kinetics/S6K_thr-252/S6_phospho 0 0 0 0 0 6e+05 3.3333e-06 \
  0.4 0.1 0 0 "" red 4 "" 7 27 0
simundump text /kinetics/S6K_thr-252/S6_phospho/notes 0 ""
call /kinetics/S6K_thr-252/S6_phospho/notes LOAD \
""
simundump kpool /kinetics/MAPK* 1 0 0 0 0 0 0 0 6e+05 0 /kinetics/geometry \
  orange yellow 13 23 0
simundump text /kinetics/MAPK*/notes 0 ""
call /kinetics/MAPK*/notes LOAD \
""
simundump kenz /kinetics/MAPK*/cluster_phospho_S6K 0 0 0 0 0 6e+05 4.817e-07 \
  4 1 0 0 "" red red "" 13 20 0
simundump text /kinetics/MAPK*/cluster_phospho_S6K/notes 0 ""
call /kinetics/MAPK*/cluster_phospho_S6K/notes LOAD \
""
simundump kreac /kinetics/basal_S6K 0 0.01 0 "" white black 4 2 0
simundump text /kinetics/basal_S6K/notes 0 ""
call /kinetics/basal_S6K/notes LOAD \
""
simundump kpool /kinetics/40S_basal 0 0 0.0001 0.0001 60 60 0 0 6e+05 0 \
  /kinetics/geometry 44 black 33 14 0
simundump text /kinetics/40S_basal/notes 0 ""
call /kinetics/40S_basal/notes LOAD \
""
simundump kenz /kinetics/40S_basal/basal 0 0 0 0 0 6e+05 3.3333e-06 0.4 0.1 0 \
  0 "" red 44 "" 33 15 0
simundump text /kinetics/40S_basal/basal/notes 0 ""
call /kinetics/40S_basal/basal/notes LOAD \
""
simundump doqcsinfo /kinetics/doqcsinfo 0 S6K.g S6K Network \
  "Pragati Jain, NCBS" "Pragati Jain, NCBS" "citation here" \
  "General Mammalian" "HEK293 cells" "Surface, Cytosol" Qualitative \
  "<a href=http://www.ploscompbiol.org/article/info:doi/10.1371/journal.pcbi.1000287>Jain P, and Bhalla, U.S. PLoS Comput Biol. 2009 Feb;5(2).</a>( Peer-reviewed publication )" \
  "Exact GENESIS implementation" "Approximates original data " show_dumpdb
simundump text /kinetics/doqcsinfo/notes 0 \
  "Rheb-GTP activates TOR, and along with convergent input from MAPK, leads to \nS6K activation and formation of the active 40S for translation\n"
call /kinetics/doqcsinfo/notes LOAD \
"Rheb-GTP activates TOR, and along with convergent input from MAPK, leads to " \
"S6K activation and formation of the active 40S for translation" \
""
simundump xgraph /graphs/conc1 0 0 10070 0 0.88766 0
simundump xgraph /graphs/conc2 0 0 7890 0 0.47175 0
simundump xplot /graphs/conc1/40S.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 50 0 0 1
simundump xplot /graphs/conc1/S6K_tot.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 44 0 0 1
simundump xplot /graphs/conc1/MAPK*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" orange 0 0 1
simundump xplot /graphs/conc1/S6K_thr-252.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 6 0 0 1
simundump xplot /graphs/conc2/S6K_thr-412.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 48 0 0 1
simundump xplot /graphs/conc2/S6K*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 46 0 0 1
simundump xgraph /moregraphs/conc3 0 -28345 12513 0.39027 0.78996 0
simundump xgraph /moregraphs/conc4 0 0 12000 0 0.71395 0
simundump xplot /moregraphs/conc4/TOR_Rheb-GTP_clx.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 44 0 0 1
simundump xplot /moregraphs/conc4/Rheb-GTP.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 28 0 0 1
simundump xcoredraw /edit/draw 0 -2 35 -14 30
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
xtextload /file/notes \
"" \
"" \
" "
addmsg /kinetics/PDK1/S6K_phospho /kinetics/PDK1 REAC eA B 
addmsg /kinetics/PDK1 /kinetics/PDK1/S6K_phospho ENZYME n 
addmsg /kinetics/S6Kinase/S6K_thr-412 /kinetics/PDK1/S6K_phospho SUBSTRATE n 
addmsg /kinetics/PP2A/dephos_S6K /kinetics/PP2A REAC eA B 
addmsg /kinetics/PP2A/dephos_clus_S6K /kinetics/PP2A REAC eA B 
addmsg /kinetics/PP2A/dephosp_S6K /kinetics/PP2A REAC eA B 
addmsg /kinetics/PP2A /kinetics/PP2A/dephos_clus_S6K ENZYME n 
addmsg /kinetics/S6Kinase/S6K* /kinetics/PP2A/dephos_clus_S6K SUBSTRATE n 
addmsg /kinetics/PP2A /kinetics/PP2A/dephos_S6K ENZYME n 
addmsg /kinetics/S6Kinase/S6K_thr-412 /kinetics/PP2A/dephos_S6K SUBSTRATE n 
addmsg /kinetics/PP2A /kinetics/PP2A/dephosp_S6K ENZYME n 
addmsg /kinetics/S6K_thr-252 /kinetics/PP2A/dephosp_S6K SUBSTRATE n 
addmsg /kinetics/mTOR_model/Rheb-GTP_bind_TORclx /kinetics/Rheb-GTP REAC A B 
addmsg /kinetics/Rheb-GTP /kinetics/mTOR_model/Rheb-GTP_bind_TORclx SUBSTRATE n 
addmsg /kinetics/mTOR_model/TOR-clx /kinetics/mTOR_model/Rheb-GTP_bind_TORclx SUBSTRATE n 
addmsg /kinetics/TOR_Rheb-GTP_clx /kinetics/mTOR_model/Rheb-GTP_bind_TORclx PRODUCT n 
addmsg /kinetics/mTOR_model/Rheb-GTP_bind_TORclx /kinetics/mTOR_model/TOR-clx REAC A B 
addmsg /kinetics/mTOR_model/Rheb-GTP_bind_TORclx /kinetics/TOR_Rheb-GTP_clx REAC B A 
addmsg /kinetics/TOR_Rheb-GTP_clx/S6K_phospho /kinetics/TOR_Rheb-GTP_clx REAC eA B 
addmsg /kinetics/TOR_Rheb-GTP_clx /kinetics/TOR_Rheb-GTP_clx/S6K_phospho ENZYME n 
addmsg /kinetics/S6Kinase/S6K* /kinetics/TOR_Rheb-GTP_clx/S6K_phospho SUBSTRATE n 
addmsg /kinetics/MAPK*/cluster_phospho_S6K /kinetics/S6Kinase/S6K* MM_PRD pA 
addmsg /kinetics/TOR_Rheb-GTP_clx/S6K_phospho /kinetics/S6Kinase/S6K* REAC sA B 
addmsg /kinetics/PP2A/dephos_S6K /kinetics/S6Kinase/S6K* MM_PRD pA 
addmsg /kinetics/PP2A/dephos_clus_S6K /kinetics/S6Kinase/S6K* REAC sA B 
addmsg /kinetics/basal_S6K /kinetics/S6Kinase/S6K* REAC B A 
addmsg /kinetics/MAPK*/cluster_phospho_S6K /kinetics/S6Kinase/S6K REAC sA B 
addmsg /kinetics/PP2A/dephos_clus_S6K /kinetics/S6Kinase/S6K MM_PRD pA 
addmsg /kinetics/basal_S6K /kinetics/S6Kinase/S6K REAC A B 
addmsg /kinetics/TOR_Rheb-GTP_clx/S6K_phospho /kinetics/S6Kinase/S6K_thr-412 MM_PRD pA 
addmsg /kinetics/PDK1/S6K_phospho /kinetics/S6Kinase/S6K_thr-412 REAC sA B 
addmsg /kinetics/PP2A/dephosp_S6K /kinetics/S6Kinase/S6K_thr-412 MM_PRD pA 
addmsg /kinetics/PP2A/dephos_S6K /kinetics/S6Kinase/S6K_thr-412 REAC sA B 
addmsg /kinetics/S6Kinase/S6K_thr-412/S6_phos /kinetics/S6Kinase/S6K_thr-412 REAC eA B 
addmsg /kinetics/S6Kinase/S6K_thr-412 /kinetics/S6Kinase/S6K_thr-412/S6_phos ENZYME n 
addmsg /kinetics/S6Kinase/40S_inact /kinetics/S6Kinase/S6K_thr-412/S6_phos SUBSTRATE n 
addmsg /kinetics/40S /kinetics/S6Kinase/S6_dephosph SUBSTRATE n 
addmsg /kinetics/S6Kinase/40S_inact /kinetics/S6Kinase/S6_dephosph PRODUCT n 
addmsg /kinetics/S6Kinase/S6_dephosph /kinetics/S6Kinase/40S_inact REAC B A 
addmsg /kinetics/S6K_thr-252/S6_phospho /kinetics/S6Kinase/40S_inact REAC sA B 
addmsg /kinetics/40S_basal/basal /kinetics/S6Kinase/40S_inact REAC sA B 
addmsg /kinetics/S6Kinase/S6K_thr-412/S6_phos /kinetics/S6Kinase/40S_inact REAC sA B 
addmsg /kinetics/S6Kinase/S6K_thr-412 /kinetics/S6K_tot SUMTOTAL n nInit 
addmsg /kinetics/S6K_thr-252 /kinetics/S6K_tot SUMTOTAL n nInit 
addmsg /kinetics/S6Kinase/S6_dephosph /kinetics/40S REAC A B 
addmsg /kinetics/S6K_thr-252/S6_phospho /kinetics/40S MM_PRD pA 
addmsg /kinetics/40S_basal/basal /kinetics/40S MM_PRD pA 
addmsg /kinetics/S6Kinase/S6K_thr-412/S6_phos /kinetics/40S MM_PRD pA 
addmsg /kinetics/PDK1/S6K_phospho /kinetics/S6K_thr-252 MM_PRD pA 
addmsg /kinetics/PP2A/dephosp_S6K /kinetics/S6K_thr-252 REAC sA B 
addmsg /kinetics/S6K_thr-252/S6_phospho /kinetics/S6K_thr-252 REAC eA B 
addmsg /kinetics/S6K_thr-252 /kinetics/S6K_thr-252/S6_phospho ENZYME n 
addmsg /kinetics/S6Kinase/40S_inact /kinetics/S6K_thr-252/S6_phospho SUBSTRATE n 
addmsg /kinetics/MAPK*/cluster_phospho_S6K /kinetics/MAPK* REAC eA B 
addmsg /kinetics/MAPK* /kinetics/MAPK*/cluster_phospho_S6K ENZYME n 
addmsg /kinetics/S6Kinase/S6K /kinetics/MAPK*/cluster_phospho_S6K SUBSTRATE n 
addmsg /kinetics/S6Kinase/S6K /kinetics/basal_S6K SUBSTRATE n 
addmsg /kinetics/S6Kinase/S6K* /kinetics/basal_S6K PRODUCT n 
addmsg /kinetics/40S_basal/basal /kinetics/40S_basal REAC eA B 
addmsg /kinetics/40S_basal /kinetics/40S_basal/basal ENZYME n 
addmsg /kinetics/S6Kinase/40S_inact /kinetics/40S_basal/basal SUBSTRATE n 
addmsg /kinetics/40S /graphs/conc1/40S.Co PLOT Co *40S.Co *50 
addmsg /kinetics/S6K_tot /graphs/conc1/S6K_tot.Co PLOT Co *S6K_tot.Co *44 
addmsg /kinetics/MAPK* /graphs/conc1/MAPK*.Co PLOT Co *MAPK*.Co *orange 
addmsg /kinetics/S6K_thr-252 /graphs/conc1/S6K_thr-252.Co PLOT Co *S6K_thr-252.Co *6 
addmsg /kinetics/S6Kinase/S6K_thr-412 /graphs/conc2/S6K_thr-412.Co PLOT Co *S6K_thr-412.Co *48 
addmsg /kinetics/S6Kinase/S6K* /graphs/conc2/S6K*.Co PLOT Co *S6K*.Co *46 
addmsg /kinetics/TOR_Rheb-GTP_clx /moregraphs/conc4/TOR_Rheb-GTP_clx.Co PLOT Co *TOR_Rheb-GTP_clx.Co *44 
addmsg /kinetics/Rheb-GTP /moregraphs/conc4/Rheb-GTP.Co PLOT Co *Rheb-GTP.Co *28 
enddump
// End of dump

call /kinetics/PP2A/notes LOAD \
"Protein Phosphatase"
call /kinetics/S6Kinase/S6K*/notes LOAD \
"" \
"" \
""
call /kinetics/S6Kinase/40S_inact/notes LOAD \
"Inactivated form of S6K" \
""
call /kinetics/40S/notes LOAD \
"Activated form of S6" \
""
call /kinetics/doqcsinfo/notes LOAD \
"Rheb-GTP activates TOR, and along with convergent input from MAPK, leads to " \
"S6K activation and formation of the active 40S for translation" \
""
complete_loading
//writeSBML acc88.xml /kinetics
//step {MAXTIME} -t
//do_save_all_plots acc88.plot
//quit
