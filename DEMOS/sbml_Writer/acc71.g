//genesis
// kkit Version 11 flat dumpfile
 
// Saved on Fri May 12 21:12:12 2006
 
include kkit {argv 1}
 
FASTDT = 1e-06
SIMDT = 1e-06
CONTROLDT = 0.1
PLOTDT = 0.01
MAXTIME = 0.6
TRANSIENT_TIME = 2
VARIABLE_DT_FLAG = 1
DEFAULT_VOL = 5.2357e-16
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
simundump geometry /kinetics/geometry 0 5.236e-16 3 sphere "" white black 2 \
  14 0
simundump kpool /kinetics/Pi_ 0 0 0 0 0 0 0 0 3.1416e+05 0 /kinetics/geometry \
  56 black -2 6 0
simundump kreac /kinetics/GTP_hydrolysis[1] 0 19.5 0.22 "" white black -2 13 \
  0
simundump kreac /kinetics/GTP_hydrolysis[2] 0 40 3.4377e-10 "" white black 3 \
  6 0
simundump kreac /kinetics/Ras_activation 0 418 5.5 "" white black -6 6 0
simundump kreac /kinetics/GAP_dissociation 0 46.5 3.8197e-06 "" white black 3 \
  -6 0
simundump kreac /kinetics/NF1_binding 0 3.8197e-06 6.36 "" white black -6 -6 \
  0
simundump kpool /kinetics/RasGTP 0 0 1 1 3.1416e+05 3.1416e+05 0 0 3.1416e+05 \
  0 /kinetics/geometry 24 black -6 -14 0
simundump kpool /kinetics/RasGTP-NF1 0 0 0 0 0 0 0 0 3.1416e+05 0 \
  /kinetics/geometry 14 black -6 -1 0
simundump kpool /kinetics/RasGTP-NF1* 0 0 0 0 0 0 0 0 3.1416e+05 0 \
  /kinetics/geometry 6 black -6 13 0
simundump kpool /kinetics/RasGDP-NF1_Pi 0 0 0 0 0 0 0 0 3.1416e+05 0 \
  /kinetics/geometry 11 black 3 13 0
simundump kpool /kinetics/RasGDP_NF1 0 0 0 0 0 0 0 0 3.1416e+05 0 \
  /kinetics/geometry 48 black 3 0 0
simundump kpool /kinetics/RasGDP 0 0 0 0 0 0 0 0 3.1416e+05 0 \
  /kinetics/geometry 61 black 3 -14 0
simundump doqcsinfo /kinetics/doqcsinfo 0 \
  /home/pragati/models/RasGTPase/25Feb2006/RasGTPase.g RasGTPase pathway \
  "Sharat J. Vayttaden and Pragati Jain, NCBS" \
  "Robert A. Phillips, Jackie L. Hunter, John F. Eccleston, and Martin R. Webb" \
  "" "General Mammalian" "E.coli Expression system" Cytosol, Qualitative \
  "<a href= http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?cmd=Retrieve&db=pubmed&dopt=Abstract&list_uids=12667087&query_hl=2&itool=pubmed_docsum>Phillips RA et al Biochemistry. 2003 Apr 8;42(13):3956-65</a>. (Peer-reviewed publication)" \
  "Exact GENESIS implementation" "Approximates original data " 4 14 0
simundump kpool /kinetics/NF1 0 0 9.9994 9.9994 3.1414e+06 3.1414e+06 0 0 \
  3.1416e+05 0 /kinetics/geometry 54 black -2 -6 0
simundump xgraph /graphs/conc1 0 0 0.6 0 0.78639 0
simundump xgraph /graphs/conc2 0 0 0.6 0 1 0
simundump xplot /graphs/conc1/GAP.Ras.GTP*.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 6 0 0 1
simundump xplot /graphs/conc2/Ras.GTP.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 24 0 0 1
simundump xplot /graphs/conc2/Pi_.Co 3 524288 \
  "delete_plot.w <s> <d>; edit_plot.D <w>" 56 0 0 1
simundump xgraph /moregraphs/conc3 0 0 0.6 0 0.00019729 0
simundump xgraph /moregraphs/conc4 0 0 0.6 0 0.00019729 0
simundump xcoredraw /edit/draw 0 -8 6 -16 16
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
xtextload /file/notes \
"Ras GTPase Activation by GTPase Activating Proteins" \
"Phillips RA et al (2003)" \
"Biochemistry 42(13):3956-65 " \
"" \
"Volume of NIH/3T3 cells roughly approximated as half a picoliter" \
"and size is approximated as 10 uM radius based on " \
"personal communication by Rainer Heintzmann" \
""
addmsg /kinetics/GTP_hydrolysis[2] /kinetics/Pi_ REAC B A 
addmsg /kinetics/RasGDP-NF1_Pi /kinetics/GTP_hydrolysis[1] PRODUCT n 
addmsg /kinetics/RasGTP-NF1* /kinetics/GTP_hydrolysis[1] SUBSTRATE n 
addmsg /kinetics/RasGDP-NF1_Pi /kinetics/GTP_hydrolysis[2] SUBSTRATE n 
addmsg /kinetics/Pi_ /kinetics/GTP_hydrolysis[2] PRODUCT n 
addmsg /kinetics/RasGDP_NF1 /kinetics/GTP_hydrolysis[2] PRODUCT n 
addmsg /kinetics/RasGTP-NF1 /kinetics/Ras_activation SUBSTRATE n 
addmsg /kinetics/RasGTP-NF1* /kinetics/Ras_activation PRODUCT n 
addmsg /kinetics/RasGDP_NF1 /kinetics/GAP_dissociation SUBSTRATE n 
addmsg /kinetics/RasGDP /kinetics/GAP_dissociation PRODUCT n 
addmsg /kinetics/NF1 /kinetics/GAP_dissociation PRODUCT n 
addmsg /kinetics/RasGTP-NF1 /kinetics/NF1_binding PRODUCT n 
addmsg /kinetics/RasGTP /kinetics/NF1_binding SUBSTRATE n 
addmsg /kinetics/NF1 /kinetics/NF1_binding SUBSTRATE n 
addmsg /kinetics/NF1_binding /kinetics/RasGTP REAC A B 
addmsg /kinetics/NF1_binding /kinetics/RasGTP-NF1 REAC B A 
addmsg /kinetics/Ras_activation /kinetics/RasGTP-NF1 REAC A B 
addmsg /kinetics/Ras_activation /kinetics/RasGTP-NF1* REAC B A 
addmsg /kinetics/GTP_hydrolysis[1] /kinetics/RasGTP-NF1* REAC A B 
addmsg /kinetics/GTP_hydrolysis[1] /kinetics/RasGDP-NF1_Pi REAC B A 
addmsg /kinetics/GTP_hydrolysis[2] /kinetics/RasGDP-NF1_Pi REAC A B 
addmsg /kinetics/GTP_hydrolysis[2] /kinetics/RasGDP_NF1 REAC B A 
addmsg /kinetics/GAP_dissociation /kinetics/RasGDP_NF1 REAC A B 
addmsg /kinetics/GAP_dissociation /kinetics/RasGDP REAC B A 
addmsg /kinetics/GAP_dissociation /kinetics/NF1 REAC B A 
addmsg /kinetics/NF1_binding /kinetics/NF1 REAC A B 
addmsg /kinetics/RasGTP-NF1* /graphs/conc1/GAP.Ras.GTP*.Co PLOT Co *GAP.Ras.GTP*.Co *6 
addmsg /kinetics/RasGTP /graphs/conc2/Ras.GTP.Co PLOT Co *Ras.GTP.Co *24 
addmsg /kinetics/Pi_ /graphs/conc2/Pi_.Co PLOT Co *Pi_.Co *56 
enddump
// End of dump

call /kinetics/Pi_/notes LOAD \
"Inorganic Phosphate is released (Pi_)" \
"" \
""
call /kinetics/GTP_hydrolysis[1]/notes LOAD \
"First step in hydrolysis of GTP bound to" \
"Ras complexed with NF1 - a mammalian GAP" \
"" \
"Kf = 19.5 /sec" \
"Kb = 0.22 /sec" \
"" \
"Table 3, Phillips RA et al 2003" \
"Biochemistry 42: 3956-3965"
call /kinetics/GTP_hydrolysis[2]/notes LOAD \
"Second step in hydrolysis of GTP bound to" \
"Ras is complexed with NF1 - a mammalian GAP" \
"" \
"Kf = 40 /sec" \
"Kb = 108 /M/sec = 1.08e-04 /uM/sec" \
"" \
"Phillips RA et al 2003" \
"Biochemistry 42: 3956-3965"
call /kinetics/Ras_activation/notes LOAD \
"Activation of Ras by GAP (i.e NF1)" \
"" \
"Kf = 418 /sec" \
"Kb = 5.5 /sec" \
"" \
"Table 3, Phillips RA et al 2003" \
"Biochemistry 42: 3956-3965"
call /kinetics/GAP_dissociation/notes LOAD \
"Dissociation of NF1 from Ras.GDP" \
"NF1 is a mammalian GAP" \
"" \
"Kf = 46.5 /sec" \
"Kb = 1.2 /sec/uM" \
"" \
"Table 3, Phillips RA et al 2003" \
"Biochemistry 42: 3956-3965"
call /kinetics/NF1_binding/notes LOAD \
"Binding of NF1 to Ras.GTP" \
"NF1 is a mammalian GAP" \
"" \
"Kd = 5.3 uM" \
"" \
"Table 3, Phillips RA et al 2003 " \
"Biochemistry 42: 3956-3965"
call /kinetics/RasGTP/notes LOAD \
"The concentration of Ras GTP is 1 um." \
"Ref: Fig-2 " \
"Phillips RA et al " \
"Biochemistry 2003, 42, 3956-3965." \
""
call /kinetics/RasGTP-NF1/notes LOAD \
"NF1 bound to N-ras complexed to GTP" \
"NF1 is a mammalian GAP" \
"" \
"Phillips RA et al 2003 " \
"Biochemistry 42: 3956-3965"
call /kinetics/RasGTP-NF1*/notes LOAD \
"NF1 bound to activated N-ras complexed to GTP" \
"NF1 is a mammalian GAP" \
"" \
"Phillips RA et al 2003" \
"Biochemistry 42: 3956-3965"
call /kinetics/RasGDP-NF1_Pi/notes LOAD \
"NF1 bound to N-ras complexed to GDP.Pi" \
"NF1 is a GAP" \
"" \
"Phillips RA et al 2003" \
"Biochemistry 42: 3956-3965"
call /kinetics/RasGDP_NF1/notes LOAD \
"NF1 bound to N-ras complexed to GDP" \
"NF1 is a mammalian GAP" \
"" \
"Phillips RA et al 2003" \
"Biochemistry 42: 3956-3965"
call /kinetics/RasGDP/notes LOAD \
"Inactive Ras GDP is formed."
call /kinetics/doqcsinfo/notes LOAD \
"Ras is an important regulator of cell growth in all eukaryotic cells. The model " \
"represent hydrolysis of active Ras-bound GTP to give inactive Ras-bound GDP catalyzed " \
"by GTPase activating proteins i.e NF1. The inactive Ras-bound GDP turns signalling " \
"protein off."
call /kinetics/NF1/notes LOAD \
"Neurofibromin a mammalian GAP" \
"NF1 conc = 10 um pp.3959" \
"" \
"Refer: Phillips RA et al 2003" \
"Biochemistry 42: 3956-3965" \
""
complete_loading

//step {MAXTIME} -t
//do_save_all_plots acc71.plot
//writeSBML siji_acc71 /kinetics
//quit

