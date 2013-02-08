//genesis
// kkit Version 6 dumpfile
 
// Saved on Wed Feb 24 10:09:51 1999
with some garbage in the beginning /*  this one has a multi-line 
                                       C like comment */
// this should fail the test
// include kkit {argv 1}
 
FASTDT = 0.005
SIMDT = 0.005
CONTROLDT = 10
PLOTDT = 10
MAXTIME = 4000
kparms
 
//genesis

initdump -version 3 -ignoreorphans 1
simobjdump table input output alloced step_mode stepsize
simobjdump xtree path script namemode sizescale
simobjdump xcoredraw xmin xmax ymin ymax
simobjdump xtext editable
simobjdump xgraph xmin xmax ymin ymax overlay
simobjdump xplot pixflags script fg ysquish do_slope wy
simobjdump kpool CoTotal CoInit Co CoRemaining n nInit nTotal nMin vol \
  slave_enable keepconc notes xtree_fg_req xtree_textfg_req
simobjdump xtab input output alloced step_mode stepsize notes editfunc \
  xtree_fg_req xtree_textfg_req plotfield manageclass baselevel last_x last_y \
  is_running
simobjdump group notes editfunc xtree_fg_req xtree_textfg_req plotfield \
  manageclass expanded movealone
simobjdump kreac A B kf kb is_active oldmsgmode notes editfunc xtree_fg_req \
  xtree_textfg_req plotfield manageclass
simobjdump kenz CoComplexInit CoComplex nComplexInit nComplex vol sA pA eA B \
  k1 k2 k3 keepconc usecomplex ksum oldmsgmode notes editfunc xtree_fg_req \
  xtree_textfg_req plotfield manageclass link_to_manage
simundump group /kinetics/PKC 0 "" edit_group blue black x group 0 0
simundump kpool /kinetics/PKC/PKC-Ca 0 0.28282 3.7208e-17 3.7208e-17 \
  2.7867e-07 2.2325e-11 2.2325e-11 1.6969e+05 0 6e+05 0 1 "" red black
simundump kreac /kinetics/PKC/PKC-act-by-Ca 0 34687 34687 1e-06 0.5 1 0 \
  "Need est of rate of assoc of Ca and PKC. Assume it is fast\nThe original parameter-searched kf of 439.4 has been\nscaled by 1/6e8 to account for change of units to n. Kf now 8.16e-7, kb=.6085\nRaised kf to 1e-6 to match Ca curve, kb to .5\n" \
  edit_reac white blue A group
simundump kreac /kinetics/PKC/PKC-act-by-DAG 0 83248 83248 1.3333e-08 8.6348 \
  1 0 \
  "Need est of rate. Assume it is fast\nObtained from param search\nkf raised 10 X : see Shinomura et al PNAS 88 5149-5153 1991.\nkf changed from 3.865e-7 to 2.0e-7 in line with closer analysis of\nShinomura data.\n26 June 1996: Corrected DAG data: reduce kf 15x from \n2e-7 to 1.333e-8" \
  edit_reac white blue A group
simundump kreac /kinetics/PKC/PKC-Ca-to-memb 0 88137 88137 1.2705 3.5026 1 0 \
  "" edit_reac white blue A group
simundump kreac /kinetics/PKC/PKC-DAG-to-memb 0 9641 9641 1 0.1 1 0 \
  "Raise kb from .087 to 0.1 to match data from Shinomura et al.\nLower kf from 1.155 to 1.0 to match data from Shinomura et al." \
  edit_reac white blue A group
simundump kreac /kinetics/PKC/PKC-act-by-Ca-AA 0 4162.4 4162.4 2e-09 0.1 1 0 \
  "Schaechter and Benowitz We have to increase Kf for conc scaling\nChanged kf to 2e-9 on Sept 19, 94. This gives better match.\n" \
  edit_reac white blue A group
simundump kreac /kinetics/PKC/PKC-act-by-DAG-AA 0 46827 46827 2 0.2 1 0 \
  "Assume slowish too. Schaechter and Benowitz" edit_reac white blue A group
simundump kpool /kinetics/PKC/PKC-DAG-AA* 0 0.12061 4.9137e-18 4.9137e-18 0 \
  2.9482e-12 2.9482e-12 72366 0 6e+05 0 1 "" cyan blue
simundump kpool /kinetics/PKC/PKC-Ca-AA* 0 0.16962 1.75e-16 1.75e-16 \
  1.6707e-07 1.05e-10 1.05e-10 1.0177e+05 0 6e+05 0 1 "" orange blue
simundump kpool /kinetics/PKC/PKC-Ca-memb* 0 0.10258 1.3896e-17 1.3896e-17 \
  1.0107e-07 8.3376e-12 8.3376e-12 61549 0 6e+05 0 1 "" pink blue
simundump kpool /kinetics/PKC/PKC-DAG-memb* 0 0.023753 9.4352e-21 9.4352e-21 \
  0 5.6611e-15 5.6611e-15 14252 0 6e+05 0 1 "" yellow blue
simundump kpool /kinetics/PKC/PKC-basal* 0 0.0432 0.02 0.02 6.8788e-08 12000 \
  12000 25920 0 6e+05 0 1 "" pink blue
simundump kreac /kinetics/PKC/PKC-basal-act 0 57812 57812 1 50 1 0 \
  "Initial basal levels were set by kf = 1, kb = 20. In model, though, the\nbasal levels of PKC activity are higher." \
  edit_reac white blue A group
simundump kpool /kinetics/PKC/PKC-AA* 0 1 1.8133e-17 1.8133e-17 1.657e-06 \
  1.088e-11 1.088e-11 6e+05 0 6e+05 0 1 "" cyan blue
simundump kreac /kinetics/PKC/PKC-act-by-AA 0 346.87 346.87 2e-10 0.1 1 0 \
  "Raise kf from 1.667e-10 to 2e-10 to get better match to data." edit_reac \
  white blue A group
simundump kpool /kinetics/PKC/PKC-Ca-DAG 0 0.0017993 8.4632e-23 8.4632e-23 0 \
  5.0779e-17 5.0779e-17 1079.6 0 6e+05 0 1 "" white blue
simundump kreac /kinetics/PKC/PKC-n-DAG 0 5203 5203 1e-09 0.1 1 0 \
  "kf raised 10 X based on Shinomura et al PNAS 88 5149-5153 1991\ncloser analysis of Shinomura et al: kf now 1e-8 (was 1.66e-8).\nFurther tweak. To get sufficient AA synergy, increase kf to 1.5e-8\n26 June 1996: Corrected DAG levels: reduce kf by 15x from\n1.5e-8 to 1e-9" \
  edit_reac white blue A group
simundump kpool /kinetics/PKC/PKC-DAG 0 0.08533 1.161e-16 1.161e-16 0 \
  6.9661e-11 6.9661e-11 51198 0 6e+05 0 1 "CoInit was .0624\n" white blue
simundump kreac /kinetics/PKC/PKC-n-DAG-AA 0 46827 46827 3e-08 2 1 0 \
  "Reduced kf to 0.66X to match Shinomura et al data.\nInitial: kf = 3.3333e-9\nNew: 2e-9\nNewer: 2e-8\nkb was 0.2\nnow 2." \
  edit_reac white blue A group
simundump kpool /kinetics/PKC/PKC-DAG-AA 0 0.012093 2.5188e-19 2.5188e-19 0 \
  1.5113e-13 1.5113e-13 7255.5 0 6e+05 0 1 "" white blue
simundump kpool /kinetics/PKC/PKC-cytosolic 0 1.5489 1 1 2.421e-06 6e+05 \
  6e+05 9.2935e+05 0 6e+05 0 1 \
  "Marquez et al J. Immun 149,2560(92) est 1e6/cell for chromaffin cells\nWe will use 1 uM as our initial concen\n" \
  white blue
simundump kpool /kinetics/DAG 1 100 11.661 11.661 2.5e-06 6.9966e+06 \
  6.9966e+06 6e+07 0 6e+05 6 0 \
  "Baseline is 11.661 from running combined model to\nsteady state. See ALL/NOTES for 23 Apr 1998." \
  green black
simundump kpool /kinetics/Ca 1 1 0.08 0.08 0 48000 48000 6e+05 0 6e+05 6 0 "" \
  red black
simundump kpool /kinetics/AA 0 1 0 0 -1.0975 0 0 6e+05 0 6e+05 0 1 "" \
  darkgreen black
simundump kpool /kinetics/PKC-active 0 0.02 0 0.02 0 12000 12000 12000 0 \
  6e+05 2 1 "Conc of PKC in brain is about 1 uM (?)" yellow black
simundump kenz /kinetics/PKC-active/PKC-act-raf 1 0 0 0 0 6e+05 0 0 0 3238.9 \
  5e-07 16 4 0 0 20 0 \
  "Rate consts from Chen et al Biochem 32, 1032 (1993)\nk3 = k2 = 4\nk1 = 9e-5\nrecalculated gives 1.666e-5, which is not very different.\nLooks like k3 is rate-limiting in this case: there is a huge amount\nof craf locked up in the enz complex. Let us assume a 10x\nhigher Km, ie, lower affinity.  k1 drops by 10x.\nAlso changed k2 to 4x k3.\nLowerd k1 to 1e-6 to balance 10X DAG sensitivity of PKC" \
  edit_enz red yellow CoComplex kpool enzproto_to_pool_link
simundump kenz /kinetics/PKC-active/PKC-inact-GAP 1 0 0 0 0 0 3.9525e-322 \
  9.8813e-323 4.9407e-322 0 1e-05 16 4 0 0 20 0 \
  "Rate consts copied from PCK-act-raf\nThis reaction inactivates GAP. The idea is from the \nBoguski and McCormick review." \
  edit_enz red yellow CoComplex kpool enzproto_to_pool_link
simundump kenz /kinetics/PKC-active/PKC-act-GEF 1 0 0 0 0 0 3.9525e-322 \
  9.8813e-323 4.9407e-322 0 1e-05 16 4 0 0 20 0 \
  "Rate consts from PKC-act-raf.\nThis reaction activates GEF. It can lead to at least 2X stim of ras, and\na 2X stim of MAPK over and above that obtained via direct phosph of\nc-raf. Note that it is a push-pull reaction, and there is also a contribution\nthrough the phosphorylation and inactivation of GAPs.\nThe original PKC-act-raf rate consts are too fast. We lower K1 by 10 X" \
  edit_enz red yellow CoComplex kpool enzproto_to_pool_link
simundump group /kinetics/MAPK 0 "" edit_group brown black x group 0 0
simundump kpool /kinetics/MAPK/craf-1 0 0.2 0.2 0.2 5.7847e-40 1.2e+05 \
  1.2e+05 1.2e+05 0 6e+05 0 1 \
  "Couldn't find any ref to the actual conc of craf-1 but I\nshould try Strom et al Oncogene 5 pp 345\nIn line with the other kinases in the cascade, I estimate the conc to be\n0.2 uM. To init we use 0.15, which is close to equil" \
  pink brown
simundump kpool /kinetics/MAPK/craf-1* 0 0.2 0 0 4.2867e-36 0 0 1.2e+05 0 \
  6e+05 0 1 "" pink brown
simundump kpool /kinetics/MAPK/MAPKK 0 0.18 0.18 0.18 0 1.08e+05 1.08e+05 \
  1.08e+05 0 6e+05 0 1 \
  "Conc is from Seger et al JBC 267:20 pp14373 (1992)\nmwt is 45/46 Kd\nWe assume that phosphorylation on both ser and thr is needed for\nactiviation. See Kyriakis et al Nature 358 417 1992\nInit conc of total is 0.18\n" \
  pink brown
simundump kpool /kinetics/MAPK/MAPK 0 0.36 0.36 0.36 0 2.16e+05 2.16e+05 \
  2.16e+05 0 6e+05 0 1 \
  "conc is from Sanghera et al JBC 265 pp 52 (1990)\nA second calculation gives 3.1 uM, from same paper.\nThey est MAPK is 1e-4x total protein, and protein is 15% of cell wt,\nso MAPK is 1.5e-5g/ml = 0.36uM. which is closer to our first estimate.\nLets use this." \
  pink brown
simundump kpool /kinetics/MAPK/craf-1** 1 0.2 0 0 4.2867e-36 0 0 1.2e+05 0 \
  6e+05 0 0 \
  "Negative feedback by MAPK* by hyperphosphorylating craf-1* gives\nrise to this pool.\nUeki et al JBC 269(22):15756-15761, 1994\n" \
  hotpink brown
simundump kpool /kinetics/MAPK/MAPK-tyr 1 0.36 0 0 7.7162e-36 0 0 2.16e+05 0 \
  6e+05 0 0 \
  "Haystead et al FEBS Lett. 306(1) pp 17-22 show that phosphorylation\nis strictly sequential, first tyr185 then thr183." \
  orange brown
simundump kpool /kinetics/MAPK/MAPKK* 0 0.18 0 0 3.858e-36 0 0 1.08e+05 0 \
  6e+05 0 1 \
  "MAPKK phosphorylates MAPK on both the tyr and thr residues, first\ntyr then thr. Refs: Seger et al JBC267:20 pp 14373 1992\nThe MAPKK itself is phosphorylated on ser as well as thr residues.\nLet us assume that the ser goes first, and that the sequential phosphorylation\nis needed. See Kyriakis et al Nature 358 417-421 1992" \
  pink brown
simundump kenz /kinetics/MAPK/MAPKK*/MAPKKtyr 0 0 0 0 0 6e+05 0 0 0 0 2.7e-05 \
  0.6 0.15 0 0 0.75 0 \
  "The actual MAPKK is 2 forms from Seger et al JBC 267:20 14373(1992)\nVmax = 150nmol/min/mg\nFrom Haystead et al FEBS 306(1):17-22 we get Km=46.6nM for at least one\nof the phosphs.\nPutting these together:\nk3=0.15/sec, scale to get k2=0.6.\nk1=0.75/46.6nM=2.7e-5" \
  edit_enz red pink CoComplex kpool enzproto_to_pool_link
simundump kenz /kinetics/MAPK/MAPKK*/MAPKKthr 1 0 0 0 0 6e+05 0 0 0 0 2.7e-05 \
  0.6 0.15 0 0 0.75 0 "Rate consts same as for MAPKKtyr." edit_enz red pink \
  CoComplex kpool enzproto_to_pool_link
simundump kpool /kinetics/MAPK/MAPKK-ser 1 0.18 0 0 3.858e-36 0 0 1.08e+05 0 \
  6e+05 0 0 "Intermediately phophorylated, assumed inactive, form of MAPKK" \
  pink brown
simundump kpool /kinetics/MAPK/Raf-GTP-Ras* 1 0.0104 0 0 2.229e-37 0 0 6240 0 \
  6e+05 0 0 "" red brown
simundump kenz /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.1 1 0 0 0 0 0 0 0 0 0 \
  5.5e-06 0.42 0.105 0 0 0.525 0 \
  "Kinetics are the same as for the craf-1* activity, ie.,\nk1=1.1e-6, k2=.42, k3 =0.105\nThese are based on Force et al PNAS USA 91 1270-1274 1994.\nThese parms cannot reach the observed 4X stim of MAPK. So lets\nincrease the affinity, ie, raise k1 10X to 1.1e-5\nLets take it back down to where it was.\nBack up to 5X: 5.5e-6" \
  edit_enz red red CoComplex kpool enzproto_to_pool_link
simundump kenz /kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.2 1 0 0 0 0 0 0 0 0 0 \
  5.5e-06 0.42 0.105 0 0 0.525 0 \
  "Same kinetics as other c-raf activated forms. See \nForce et al PNAS 91 1270-1274 1994.\nk1 = 1.1e-6, k2 = .42, k3 = 1.05\nraise k1 to 5.5e-6\n" \
  edit_enz red red CoComplex kpool enzproto_to_pool_link
simundump kpool /kinetics/MAPK* 0 1 0 0 1.6667e-06 0 0 6e+05 0 6e+05 2 1 \
  "This version is phosphorylated on both the tyr and thr residues and\nis active: refs\nThe rate consts are very hard to nail down. Combine Sanghera et al\nJBC 265(1) :52-57 with Nemenoff et al JBC 93 pp 1960 to get\nk3=10/sec = k2 (from Nemenoff Vmax) and k1 = (k2 + k3)/Km = 1.3e-6\nOr: k3 = 10, k2 = 40, k1 = 3.25e-6" \
  orange yellow
simundump kenz /kinetics/MAPK*/MAPK*-feedback 1 0 0 0 0 6e+05 0 0 0 0 \
  3.25e-06 40 10 0 0 50 0 \
  "Ueki et al JBC 269(22):15756-15761 show the presence of\nthis step, but not the rate consts, which are derived from\nSanghera et al  JBC 265(1):52-57, 1990, see the deriv in the\nMAPK* notes." \
  edit_enz red orange CoComplex kpool enzproto_to_pool_link
simundump kenz /kinetics/MAPK*/MAPK* 0 0 0 0 0 6e+05 0 0 0 0 6.5e-06 80 20 0 \
  0 100 0 \
  "Km = 25uM @ 50 uM ATP and 1mg/ml MBP (huge XS of substrate)\nVmax = 4124 pmol/min/ml at a conc of 125 pmol/ml of enz, so:\nk3 = .5/sec (rate limiting)\nk1 = (k2  + k3)/Km = (.5 + 0)/(25*6e5) = 2e-8 (#/cell)^-1\n#s from Sanghera et al JBC 265 pp 52 , 1990. \nFrom Nemenoff et al JBC 268(3):1960-1964 - using Sanghera's 1e-4 ratio\nof MAPK to protein, we get k3 = 7/sec from 1000 pmol/min/mg fig 5" \
  edit_enz red orange CoComplex kpool enzproto_to_pool_link
simundump kpool /kinetics/MKP-1 1 0.008 0.0032 0.0032 7.073e-37 1920 1920 \
  4800 0 6e+05 0 1 \
  "MKP-1 dephosphoryates and inactivates MAPK in vivo: Sun et al Cell 75 \n487-493 1993. Levels of MKP-1\nare regulated, and rise in 1 hour. \nKinetics from Charles et al PNAS 90:5292-5296 1993. They refer\nto Charles et al Oncogene 7 187-190 who show that half-life of MKP1/3CH134\nis 40 min. 80% deph of MAPK in 20 min\nSep 17 1997: CoInit now 0.4x to 0.0032. See parm searches\nfrom jun96 on." \
  hotpink black
simundump kenz /kinetics/MKP-1/MKP1-tyr-deph 1 0 0 0 0 6e+05 0 0 0 0 0.000125 \
  4 1 0 0 5 0 \
  "The original kinetics have been modified to obey the k2 = 4 * k3 rule,\nwhile keeping kcat and Km fixed. As noted in the NOTES, the only constraining\ndata point is the time course of MAPK dephosphorylation, which this\nmodel satisfies. It would be nice to have more accurate estimates of\nrate consts and MKP-1 levels from the literature. \nEffective Km : 67 nM\nkcat = 1.43 umol/min/mg" \
  edit_enz red hotpink CoComplex kpool enzproto_to_pool_link
simundump kenz /kinetics/MKP-1/MKP1-thr-deph 1 0 0 0 0 6e+05 0 0 0 0 0.000125 \
  4 1 0 0 5 0 "See MKP1-tyr-deph" edit_enz red hotpink CoComplex kpool \
  enzproto_to_pool_link
simundump kreac /kinetics/Ras-act-craf 1 0 0 4e-05 0.5 0 0 \
  "Assume the binding is fast and is limited only by the amount of\nRas* available. So kf = kb/[craf-1] \nIf kb is 1/sec, then kf = 1/0.2 uM = 1/(0.2 * 6e5) = 8.3e-6\nLater: Raise it by 10 X to 4e-5\nFrom Hallberg et al JBC 269:6 3913-3916 1994, 3% of cellular Raf is\ncomplexed with Ras. So we raise kb 4 x to 4\nThis step needed to memb-anchor and activate Raf: Leevers et al Nature\n369 411-414" \
  edit_reac white black A group
simundump kpool /kinetics/PPhosphatase2A 1 0.224 0.224 0.224 0 1.344e+05 \
  1.344e+05 1.344e+05 0 6e+05 0 0 \
  "Refs: Pato et al Biochem J 293:35-41(93);\nTakai&Mieskes Biochem J 275:233-239\nk1=1.46e-4, k2=1000,k3=250. these use\nkcat values for calponin. Also, units of kcat may be in min!\nrevert to Vmax base:\nk3=6, k2=25,k1=3.3e-6 or 6,6,1e-6\nCoInit assumed 0.1 uM.\nSee NOTES for MAPK_Ras50.g. CoInit now 0.08\nSep 17 1997: Raise CoInt 1.4x to 0.224, see parm\nsearches from jun 96 on.\n" \
  hotpink yellow
simundump kenz /kinetics/PPhosphatase2A/craf-deph 1 0 0 0 0 6e+05 0 0 0 0 \
  3.3e-06 25 6 0 0 31 0 "See parent PPhosphatase2A for parms\n" edit_enz red \
  hotpink CoComplex kpool enzproto_to_pool_link
simundump kenz /kinetics/PPhosphatase2A/MAPKK-deph 1 0 0 0 0 6e+05 0 0 0 0 \
  3.3e-06 25 6 0 0 31 0 \
  "See: Kyriakis et al Nature 358 pp 417-421 1992\nAhn et al Curr Op Cell Biol 4:992-999 1992 for this pathway.\nSee parent PPhosphatase2A for parms." \
  edit_enz red hotpink CoComplex kpool enzproto_to_pool_link
simundump kenz /kinetics/PPhosphatase2A/MAPKK-deph-ser 1 0 0 0 0 6e+05 0 0 0 \
  0 3.3e-06 25 6 0 0 31 0 "" edit_enz red hotpink CoComplex kpool \
  enzproto_to_pool_link
simundump kenz /kinetics/PPhosphatase2A/craf**-deph 1 0 0 0 0 0 0 0 0 0 \
  3.3e-06 25 6 0 0 31 0 \
  "Ueki et al JBC 269(22) pp 15756-15761 1994 show hyperphosphorylation of\ncraf, so this is there to dephosphorylate it. Identity of phosphatase is not\nknown to me, but it may be PP2A like the rest, so I have made it so." \
  edit_enz red hotpink CoComplex kpool enzproto_to_pool_link
simundump group /kinetics/PLA2 0 \
  "Mail source of data: Leslie and Channon BBA 1045 (1990) pp 261-270.\nFig 6 is Ca curve. Fig 4a is PIP2 curve. Fig 4b is DAG curve. Also see\nWijkander and Sundler JBC 202 (1991) pp873-880;\nDiez and Mong JBC 265(24) p14654;\nLeslie JBC 266(17) (1991) pp11366-11371" \
  edit_group darkgreen black x group 0 1
simundump kpool /kinetics/PLA2/PLA2-cytosolic 0 0.4 0.4 0.4 2.331e-07 2.4e+05 \
  2.4e+05 2.4e+05 0 6e+05 0 1 \
  "Calculated cytosolic was 20 nm from Wijkander and Sundler\nHowever, Leslie and Channon use about 400 nM. Need to confirm,\nbut this is the value I use here. Another recalc of W&S gives 1uM" \
  yellow darkgreen
simundump kreac /kinetics/PLA2/PLA2-Ca-act 0 498.41 0 1.6667e-06 0.1 1 0 \
  "Leslie and Channon BBA 1045 (1990) 261-270 fig6 pp267." edit_reac white \
  darkgreen A group
simundump kpool /kinetics/PLA2/PLA2-Ca* 0 1 0 0 1.6528e-06 0 0 6e+05 0 6e+05 \
  0 1 "" yellow darkgreen
simundump kenz /kinetics/PLA2/PLA2-Ca*/kenz 0 0 0 0 0 6e+05 1.6153e+05 40381 \
  2.0191e+05 2.0185e+05 2.25e-06 21.6 5.4 0 0 27 0 \
  "10 x raise oct22\n12 x oct 24, set k2 = 4 * k3" edit_enz red yellow \
  CoComplex kpool enzproto_to_pool_link
simundump kreac /kinetics/PLA2/PIP2-PLA2-act 0 4622.4 4682.4 2e-09 0.5 1 0 "" \
  edit_reac white darkgreen A group
simundump kpool /kinetics/PLA2/PIP2-PLA2* 0 1 0 0 1.641e-06 0 0 6e+05 0 6e+05 \
  0 1 "" cyan darkgreen
simundump kenz /kinetics/PLA2/PIP2-PLA2*/kenz 0 0 0 0 0 6e+05 6.1234e+05 \
  1.5309e+05 7.6543e+05 7.6546e+05 4.6e-06 44.16 11.04 0 0 55.2 0 \
  "10 X raise oct 22\n12 X further raise oct 24 to allow for correct conc of enzyme\n" \
  edit_enz red cyan CoComplex kpool enzproto_to_pool_link
simundump kreac /kinetics/PLA2/PIP2-Ca-PLA2-act 0 1896.2 1495.2 2e-08 0.1 1 0 \
  "" edit_reac white darkgreen A group
simundump kpool /kinetics/PLA2/PIP2-Ca-PLA2* 0 1 0 0 1.614e-06 0 0 6e+05 0 \
  6e+05 0 1 "" cyan darkgreen
simundump kenz /kinetics/PLA2/PIP2-Ca-PLA2*/kenz 0 0 0 0 0 6e+05 4.0959e+06 \
  1.024e+06 5.1199e+06 5.1197e+06 1.5e-05 144 36 0 0 180 0 \
  "10 x raise oct 22\n12 x and rescale for k2 = 4 * k3 convention oct 24\nIncrease further to get the match to expt, which was spoilt due\nto large accumulation of PLA2 in the enzyme complexed forms.\nLets raise k3, leaving the others at \nk1 = 1.5e-5 and k2 = 144 since they are large already.\n" \
  edit_enz red cyan CoComplex kpool enzproto_to_pool_link
simundump kreac /kinetics/PLA2/DAG-Ca-PLA2-act 0 1504.3 1495.2 5e-09 4 1 0 \
  "27 June 1996\nScaled kf down by 0.015\nfrom 3.33e-7 to 5e-9\nto fit with revised DAG estimates\nand use of mole-fraction to calculate eff on PLA2." \
  edit_reac white darkgreen A group
simundump kpool /kinetics/PLA2/DAG-Ca-PLA2* 0 1 0 0 1.6656e-06 0 0 6e+05 0 \
  6e+05 0 1 "" pink darkgreen
simundump kenz /kinetics/PLA2/DAG-Ca-PLA2*/kenz 0 0 0 0 0 6e+05 1.3539e+05 \
  33848 1.6924e+05 1.6923e+05 2.5e-05 240 60 0 0 300 0 \
  "10 X raise oct 22\n12 X raise oct 24 + conversion to k2 =4 * k3" edit_enz \
  red pink CoComplex kpool enzproto_to_pool_link
simundump kpool /kinetics/PLA2/APC 0 30 30 30 0 1.8e+07 1.8e+07 1.8e+07 0 \
  6e+05 5 1 \
  "arachodonylphosphatidylcholine is the favoured substrate\nfrom Wijkander and Sundler, JBC 202 pp 873-880, 1991.\nTheir assay used 30 uM substrate, which is what the kinetics in\nthis model are based on. For the later model we should locate\na more realistic value for APC." \
  yellow darkgreen
simundump kreac /kinetics/PLA2/Degrade-AA 1 0 1.2585e+06 0.4 0 1 0 \
  "I need to check if the AA degradation pathway really leads back to \nAPC. Anyway, it is a convenient buffered pool to dump it back into.\nFor the purposes of the full model we use a rate of degradation of\n0.2/sec\nRaised decay to 0.4 : see PLA35.g notes for Feb17 " \
  edit_reac white darkgreen A group
simundump kpool /kinetics/PLA2/PLA2*-Ca 0 1 0 0 1.6667e-06 0 0 6e+05 0 6e+05 \
  1 1 \
  "Phosphorylated form of PLA2. Still need to hook it up using kinases.\nPKA: Wightman et al JBC 257 pp6650 1982\nPKC: Many refs, eg Gronich et al JBC 263 pp 16645, 1988 but see Lin etal\nMAPK: Lin et al, Cell 72 pp 269, 1993.  Show 3x with MAPK but not PKC alone\nDo not know if there is a Ca requirement for active phosphorylated state." \
  orange darkgreen
simundump kenz /kinetics/PLA2/PLA2*-Ca/kenz 0 0 0 0 0 6e+05 0 0 0 0 5e-05 480 \
  120 0 0 600 0 \
  "This form should be 3 to 6 times as fast as the Ca-only form.\nI have scaled by 4x which seems to give a 5x rise.\n10x raise Oct 22\n12 x oct 24, changed k2 = 4 * k3" \
  edit_enz red orange CoComplex kpool enzproto_to_pool_link
simundump kpool /kinetics/PLA2/PLA2* 1 1.6667e-06 0 0 2.7778e-12 0 0 1 0 \
  6e+05 0 0 "" orange darkgreen
simundump kreac /kinetics/PLA2/PLA2*-Ca-act 1 0 0 1e-05 0.1 1 0 \
  "To start off, same kinetics as the PLA2-Ca-act direct pathway.\nOops ! Missed out the Ca input to this pathway first time round.\nLet's raise the forward rate about 3x to 5e-6. This will let us reduce the\nrather high rates we have used for the kenz on PLA2*-Ca. In fact, it\nmay be that the rates are not that different, just that this pathway for\ngetting the PLA2 to the memb is more efficien...." \
  edit_reac white darkgreen A group
simundump kreac /kinetics/PLA2/dephosphorylate-PLA2* 1 0 0 0.17 0 1 0 "" \
  edit_reac white darkgreen A group
simundump kpool /kinetics/temp-PIP2 1 25 2.5 2.5 0 1.5e+06 1.5e+06 1.5e+07 0 \
  6e+05 6 1 \
  "This isn't explicitly present in the M&L model, but is obviously needed.\nI assume its conc is fixed at 1uM for now, which is a bit high. PLA2 is stim\n7x by PIP2 @ 0.5 uM (Leslie and Channon BBA 1045:261(1990) \nLeslie and Channon say PIP2 is present at 0.1 - 0.2mol% range in membs,\nwhich comes to 50 nM. Ref is Majerus et al Cell 37 pp 701-703 1984\nLets use a lower level of 30 nM, same ref...." \
  green black
simundump group /kinetics/Ras 1 \
  "Ras has now gotten to be a big enough component of the model to\ndeserve its own group. The main refs are\nBoguski and McCormick Nature 366 643-654 '93 Major review\nEccleston et al JBC 268:36 pp 27012-19\nOrita et al JBC 268:34 2554246" \
  edit_group blue black x group 0 0
simundump kreac /kinetics/Ras/dephosph-GEF 1 0 9.9695e-11 1 0 1 0 "" \
  edit_reac white blue A group
simundump kpool /kinetics/Ras/inact-GEF 1 0.1 0.1 0.1 0 60000 60000 60000 0 \
  6e+05 0 1 \
  "Assume that SoS is present only at 50 nM.\nRevised to 100 nM to get equil to experimentally known levels." \
  hotpink blue
simundump kpool /kinetics/Ras/GEF* 1 0.1 0 0 1.6667e-07 0 0 60000 0 6e+05 0 0 \
  "phosphorylated and thereby activated form of GEF. See, e.g.\nOrita et al JBC 268:34 25542-25546 1993, Gulbins et al.\nIt is not clear whether there is major specificity for tyr or ser/thr." \
  hotpink blue
simundump kenz /kinetics/Ras/GEF*/GEF*-act-ras 1 0 0 0 0 6e+05 3.1152e-12 \
  7.7881e-13 3.8941e-12 3.8916e-12 3.3e-07 0.08 0.02 0 0 0.1 0 \
  "Kinetics same as GEF-bg-act-ras\n" edit_enz red hotpink CoComplex kpool \
  enzproto_to_pool_link
simundump kpool /kinetics/Ras/GTP-Ras 1 0.2 0 0 3.3058e-07 0 0 1.2e+05 0 \
  6e+05 0 0 \
  "Only a very small fraction (7% unstim, 15% stim) of ras is GTP-bound.\nGibbs et al JBC 265(33) 20437\n" \
  orange blue
simundump kpool /kinetics/Ras/GDP-Ras 1 0.2 0.2 0.2 4.7575e-09 1.2e+05 \
  1.2e+05 1.2e+05 0 6e+05 0 1 \
  "GDP bound form. See Rosen et al Neuron 12 1207-1221 June 1994.\nthe activation loop is based on Boguski and McCormick Nature 366 643-654 93\nAssume Ras is present at about the same level as craf-1, 0.2 uM.\nHallberg et al JBC 269:6 3913-3916 1994 estimate upto 5-10% of cellular\nRaf is assoc with Ras. Given that only 5-10% of Ras is GTP-bound, we\nneed similar amounts of Ras as Raf." \
  pink blue
simundump kreac /kinetics/Ras/Ras-intrinsic-GTPase 1 0 0.098726 1e-04 0 1 0 \
  "This is extremely slow (1e-4), but it is significant as so little GAP actually\ngets complexed with it that the total GTP turnover rises only by\n2-3 X (see Gibbs et al, JBC 265(33) 20437-20422) and \nEccleston et al JBC 268(36) 27012-27019\nkf = 1e-4\n" \
  edit_reac white blue A group
simundump kreac /kinetics/Ras/dephosph-GAP 1 0 9.9981e-12 0.1 0 1 0 \
  "Assume a reasonably good rate for dephosphorylating it, 1/sec" edit_reac \
  white blue A group
simundump kpool /kinetics/Ras/GAP* 1 0.05 0 0 8.3333e-08 0 0 30000 0 6e+05 0 \
  1 "" red blue
simundump kpool /kinetics/Ras/GAP 1 0.002 0.002 0.002 3.9215e-10 1200 1200 \
  1200 0 6e+05 0 1 \
  "GTPase-activating proteins. See Boguski and McCormick.\nTurn off Ras by helping to hydrolyze bound GTP. \nThis one is probably NF1, ie.,  Neurofibromin as it is inhibited by AA and lipids,\nand expressed in neural cells. p120-GAP is also a possible candidate, but\nis less regulated. Both may exist at similar levels.\nSee Eccleston et al JBC 268(36) pp27012-19\nLevel=.002" \
  red blue
simundump kenz /kinetics/Ras/GAP/GAP-inact-ras 1 0 0 0 0 6e+05 1724.3 17.243 \
  1741.5 1741.5 0.001666 1000 10 0 0 1010 0 \
  "From Eccleston et al JBC 268(36)pp27012-19 get Kd < 2uM, kcat - 10/sec\nFrom Martin et al Cell 63 843-849 1990 get Kd ~ 250 nM, kcat = 20/min\nI will go with the Eccleston figures as there are good error bars (10%). In general\nthe values are reasonably close.\nk1 = 1.666e-3/sec, k2 = 1000/sec, k3 = 10/sec (note k3 is rate-limiting)" \
  edit_enz red red CoComplex kpool enzproto_to_pool_link
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
simundump xcoredraw /edit/draw 0 -13.918 -2.9433 -15.8 1.1447
simundump xtree /edit/draw/tree 0 /kinetics/##[] \
  "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
call /edit/draw/tree ADDMSGARROW \
             select                 all       all       all    none -1000 0 \
  "" "" ""
call /edit/draw/tree ADDMSGARROW \
     /kinetics/##[] /kinetics/##[TYPE=group]       all       all    none -1001 0 \
  "echo.p dragging <S> onto group <D>; thing_to_group_add.p <S> <D>" "" ""
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kpool] /kinetics/##[TYPE=kpool]  SUMTOTAL     kpool darkgreen -1002 0 \
  "echo.p dragging <S> to <D> for sumtotal" "pool_to_pool_add_sumtot.p <S> <D>" "pool_to_pool_drop_sumtot.p <S> <D>"
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kpool] /kinetics/##[TYPE=kpool]  CONSERVE     kpool    blue -1003 0 \
  "echo.p dragging <S> to <D> for conserve" "pool_to_pool_add_consv.p <S> <D>" "pool_to_pool_drop_consv.p <S> <D>"
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kreac] /kinetics/##[TYPE=kpool]      REAC     kpool    none -1004 0 \
  "echo.p dragging <S> to <D> for product" "reac_to_pool_add.p <S> <D>" "reac_to_pool_drop.p <S> <D>"
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kpool] /kinetics/##[TYPE=kreac] SUBSTRATE     kreac   green -1005 0 \
  "echo.p dragging <S> to <D> for substrate" "pool_to_reac_add.p <S> <D>" "pool_to_reac_drop.p <S> <D>"
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kpool] /kinetics/##[TYPE=kreac]        KF     kreac  purple -1006 0 \
  "" "" ""
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kpool] /kinetics/##[TYPE=kreac]        KB     kreac hotpink -1007 0 \
  "" "" ""
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kpool] /kinetics/##[TYPE=kreac]   PRODUCT     kreac   green -1008 1 \
  "" "" ""
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kenz] /kinetics/##[TYPE=kpool]    MM_PRD     kpool     red -1009 0 \
  "echo.p dragging <S> to <D> for enz product" "enz_to_pool_add.p <S> <D>" "enz_to_pool_drop.p <S> <D>"
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kpool] /kinetics/##[TYPE=kenz] SUBSTRATE      kenz     red -1010 0 \
  "echo.p dragging <S> to <D> for enz substrate" "pool_to_enz_add.p <S> <D>" "pool_to_enz_drop.p <S> <D>"
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kenz] /kinetics/##[TYPE=kpool]  SUMTOTAL     kpool darkgreen -1011 0 \
  "" "" ""
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kenz] /kinetics/##[TYPE=kpool]  CONSERVE     kpool    blue -1012 0 \
  "" "" ""
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kpool] /kinetics/##[TYPE=kenz]  INTRAMOL      kenz   white -1013 0 \
  "" "" ""
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=stim] /kinetics/##[TYPE=kpool]     SLAVE     kpool  orange -1014 0 \
  "echo.p dragging <S> to <D> for stimulus" "stim_to_pool_add.p <S> <D>" "stim_to_pool_drop.p <S> <D>"
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=stim] /kinetics/##[TYPE=stim]     INPUT      stim  orange -1015 0 \
  "echo.p dragging <S> to <D> for stim" "stim_to_stim_add.p <S> <D>" "stim_to_stim_drop.p <S> <D>"
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=xtab] /kinetics/##[TYPE=kpool]     SLAVE     kpool  orange -1016 0 \
  "echo.p dragging <S> to <D> for xtab" "xtab_to_pool_add.p <S> <D>" "xtab_to_pool_drop.p <S> <D>"
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=xtab] /kinetics/##[TYPE=xtab]     INPUT      xtab  orange -1017 0 \
  "echo.p dragging <S> to <D> for xtab" "xtab_to_xtab_add.p <S> <D>" "xtab_to_xtab_drop.p <S> <D>"
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kpool] /kinetics/##[TYPE=xtab]     INPUT      xtab hotpink -1018 0 \
  "echo.p dragging <S> to <D> for xtab" "pool_to_xtab_add.p <S> <D>" "pool_to_xtab_drop.p <S> <D>"
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kchan] /kinetics/##[TYPE=kpool]      REAC     kpool    none -1019 0 \
  "echo.p dragging <S> to <D> for chan product" "chan_to_pool_add.p <S> <D>" "chan_to_pool_drop.p <S> <D>"
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kpool] /kinetics/##[TYPE=kchan]   PRODUCT     kchan  orange -1020 1 \
  "" "" ""
call /edit/draw/tree ADDMSGARROW \
/kinetics/##[TYPE=kpool] /kinetics/##[TYPE=kchan] SUBSTRATE     kchan  orange -1021 0 \
  "echo.p dragging <S> to <D> for chan substrate" "pool_to_chan_add.p <S> <D>" "pool_to_chan_drop.p <S> <D>"
call /edit/draw/tree UNDUMP 89 \
 "/kinetics/PKC" -3.04929 8.21631 0 \
 "/kinetics/PKC/PKC-Ca" -4.07517 1.51085 0 \
 "/kinetics/PKC/PKC-act-by-Ca" -4.07517 -0.122947 0 \
 "/kinetics/PKC/PKC-act-by-DAG" -2.06118 0.693953 0 \
 "/kinetics/PKC/PKC-Ca-to-memb" -3.79738 4.25331 0 \
 "/kinetics/PKC/PKC-DAG-to-memb" -2.61677 2.73621 0 \
 "/kinetics/PKC/PKC-act-by-Ca-AA" -0.78797 3.81568 0 \
 "/kinetics/PKC/PKC-act-by-DAG-AA" 1.24916 3.23218 0 \
 "/kinetics/PKC/PKC-DAG-AA*" 0.600985 5.53701 0 \
 "/kinetics/PKC/PKC-Ca-AA*" -0.602776 6.29557 0 \
 "/kinetics/PKC/PKC-Ca-memb*" -2.77881 6.52897 0 \
 "/kinetics/PKC/PKC-DAG-memb*" -1.82969 5.50784 0 \
 "/kinetics/PKC/PKC-basal*" -4.7465 5.56619 0 \
 "/kinetics/PKC/PKC-basal-act" -4.97799 3.05713 0 \
 "/kinetics/PKC/PKC-AA*" 1.78159 6.82072 0 \
 "/kinetics/PKC/PKC-act-by-AA" -4.99245 -1.86537 0 \
 "/kinetics/PKC/PKC-Ca-DAG" 0.230597 1.8026 0 \
 "/kinetics/PKC/PKC-n-DAG" -3.0103 -1.99016 0 \
 "/kinetics/PKC/PKC-DAG" -0.996314 -1.08573 0 \
 "/kinetics/PKC/PKC-n-DAG-AA" -1.22781 -2.95294 0 \
 "/kinetics/PKC/PKC-DAG-AA" 0.624134 0.227153 0 \
 "/kinetics/PKC/PKC-cytosolic" -6.13155 0.597108 0 \
 "/kinetics/DAG" -3.51509 -4.33144 0 \
 "/kinetics/Ca" -8.3874 -2.76345 0 \
 "/kinetics/AA" -3.28976 -9.33758 0 \
 "/kinetics/PKC-active" 2.13247 8.47704 0 \
 "/kinetics/PKC-active/PKC-act-raf" 6.2532 10.5492 0 \
 "/kinetics/PKC-active/PKC-inact-GAP" 3.43912 11.8042 0 \
 "/kinetics/PKC-active/PKC-act-GEF" -0.247906 17.2639 0 \
 "/kinetics/MAPK" 14.6165 11.1912 0 \
 "/kinetics/MAPK/craf-1" 6.32605 8.11685 0 \
 "/kinetics/MAPK/craf-1*" 9.2401 7.71145 0 \
 "/kinetics/MAPK/MAPKK" 6.33153 3.98944 0 \
 "/kinetics/MAPK/MAPK" 6.06559 1.08628 0 \
 "/kinetics/MAPK/craf-1**" 12.4638 7.90223 0 \
 "/kinetics/MAPK/MAPK-tyr" 8.4147 0.82034 0 \
 "/kinetics/MAPK/MAPKK*" 12.5477 4.02564 0 \
 "/kinetics/MAPK/MAPKK*/MAPKKtyr" 8.89145 3.55309 0 \
 "/kinetics/MAPK/MAPKK*/MAPKKthr" 12.9613 3.03628 0 \
 "/kinetics/MAPK/MAPKK-ser" 9.26524 4.16569 0 \
 "/kinetics/MAPK/Raf-GTP-Ras*" 4.90545 6.78143 0 \
 "/kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.1" 7.61789 6.21889 0 \
 "/kinetics/MAPK/Raf-GTP-Ras*/Raf-GTP-Ras*.2" 10.6978 6.06883 0 \
 "/kinetics/MAPK*" 13.7349 0.413779 0 \
 "/kinetics/MAPK*/MAPK*-feedback" 10.3875 10.6685 0 \
 "/kinetics/MAPK*/MAPK*" -12.0046 -14.9398 0 \
 "/kinetics/MKP-1" 5.0816 2.44073 0 \
 "/kinetics/MKP-1/MKP1-tyr-deph" 6.27806 3.06838 0 \
 "/kinetics/MKP-1/MKP1-thr-deph" 10.7893 2.93108 0 \
 "/kinetics/Ras-act-craf" 3.5614 10.091 0 \
 "/kinetics/PPhosphatase2A" 9.3898 9.13091 0 \
 "/kinetics/PPhosphatase2A/craf-deph" 7.80129 10.2154 0 \
 "/kinetics/PPhosphatase2A/MAPKK-deph" 13.159 6.07359 0 \
 "/kinetics/PPhosphatase2A/MAPKK-deph-ser" 4.8651 5.92081 0 \
 "/kinetics/PPhosphatase2A/craf**-deph" 12.4456 9.90537 0 \
 "/kinetics/PLA2" -7.35718 -14.2086 0 \
 "/kinetics/PLA2/PLA2-cytosolic" -11.8243 -8.94209 0 \
 "/kinetics/PLA2/PLA2-Ca-act" -11.097 -11.1044 0 \
 "/kinetics/PLA2/PLA2-Ca*" -8.72196 -11.646 0 \
 "/kinetics/PLA2/PLA2-Ca*/kenz" -6.0553 -11.6669 0 \
 "/kinetics/PLA2/PIP2-PLA2-act" -11.0553 -6.75022 0 \
 "/kinetics/PLA2/PIP2-PLA2*" -8.6803 -6.29188 0 \
 "/kinetics/PLA2/PIP2-PLA2*/kenz" -6.03446 -6.27105 0 \
 "/kinetics/PLA2/PIP2-Ca-PLA2-act" -10.097 -7.50022 0 \
 "/kinetics/PLA2/PIP2-Ca-PLA2*" -8.32613 -7.89605 0 \
 "/kinetics/PLA2/PIP2-Ca-PLA2*/kenz" -5.97196 -7.97938 0 \
 "/kinetics/PLA2/DAG-Ca-PLA2-act" -10.8261 -9.83355 0 \
 "/kinetics/PLA2/DAG-Ca-PLA2*" -8.13863 -10.4794 0 \
 "/kinetics/PLA2/DAG-Ca-PLA2*/kenz" -5.95113 -10.3544 0 \
 "/kinetics/PLA2/APC" -8.23862 -9.96337 0 \
 "/kinetics/PLA2/Degrade-AA" -6.1808 -5.2875 0 \
 "/kinetics/PLA2/PLA2*-Ca" -7.81296 -12.6868 0 \
 "/kinetics/PLA2/PLA2*-Ca/kenz" -6.08142 -12.8166 0 \
 "/kinetics/PLA2/PLA2*" -9.02503 -14.8512 0 \
 "/kinetics/PLA2/PLA2*-Ca-act" -10.0856 -12.7517 0 \
 "/kinetics/PLA2/dephosphorylate-PLA2*" -13.6928 -11.7346 0 \
 "/kinetics/temp-PIP2" -15.7959 -7.04735 0 \
 "/kinetics/Ras" 14.5128 16.351 0 \
 "/kinetics/Ras/dephosph-GEF" 9.07016 17.8808 0 \
 "/kinetics/Ras/inact-GEF" 12.4531 18.3522 0 \
 "/kinetics/Ras/GEF*" 6.4483 17.246 0 \
 "/kinetics/Ras/GEF*/GEF*-act-ras" 7.0855 16.086 0 \
 "/kinetics/Ras/GTP-Ras" 12.564 13.0836 0 \
 "/kinetics/Ras/GDP-Ras" 6.13087 14.1651 0 \
 "/kinetics/Ras/Ras-intrinsic-GTPase" 9.09789 13.4996 0 \
 "/kinetics/Ras/dephosph-GAP" 4.02345 15.5238 0 \
 "/kinetics/Ras/GAP*" 1.3498 14.349 0 \
 "/kinetics/Ras/GAP" 6.6549 12.3377 0 \
 "/kinetics/Ras/GAP/GAP-inact-ras" 9.01211 12.4032 0
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
addmsg /kinetics/PKC-active/PKC-act-raf /kinetics/PKC-active REAC sA B 
addmsg /kinetics/PKC-active/PKC-act-raf /kinetics/PKC-active CONSERVE nComplex nComplexInit 
addmsg /kinetics/PKC-active/PKC-inact-GAP /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active/PKC-inact-GAP /kinetics/PKC-active CONSERVE nComplex nComplexInit 
addmsg /kinetics/PKC-active/PKC-act-GEF /kinetics/PKC-active REAC eA B 
addmsg /kinetics/PKC-active/PKC-act-GEF /kinetics/PKC-active CONSERVE nComplex nComplexInit 
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

call /edit/draw/tree RESET
reset
