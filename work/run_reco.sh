#! /bin/bash
export USER="$(id -u -n)"
export LOGNAME=${USER}
export HOME=/sphenix/u/${LOGNAME}


#source /opt/sphenix/core/bin/sphenix_setup.sh new
#source /sphenix/u/mitrankova/.login_TPC_SA
#source /sphenix/u/mitrankova/.login_PR_TPC_SA
#source /sphenix/u/mitrankova/.login_Hits_Data
source /sphenix/u/mitrankova/.login_crossing_vertex_si_assoc
#printenv

nEvents="$1"


#runnumber=79528
runnumber=79513
collision="run3pp"
cdbtag="ana561_2025p013_v001"
OutDir=/sphenix/tg/tg01/hf/mitrankova/Hadrons
name="HITS_pp"

#runnumber=76905
#collision="run3auau"
#cdbtag="ana514_nocdbtag_v001"
#OutDir=/sphenix/tg/tg01/hf/mitrankova/PatternReco/76905_new_8_380_no_mv_rot_Btest
#OutDir=/sphenix/tg/tg01/hf/mitrankova/79513_ifc_auau
#OutDir=/sphenix/tg/tg01/hf/mitrankova/76905_tpc_vertex
#name="HITS_6x6_1mrad_AuAu"

#runnumber=82626
#collision="run3oo"
#cdbtag="ana537_nocdbtag_v001"
#OutDir=/sphenix/tg/tg01/hf/mitrankova/PatternReco/82626_00
#name="HITS_OO"

#runnumber=75405
#collision="run3auau"
#cdbtag="ana514_nocdbtag_v001"
#OutDir=/sphenix/tg/tg01/hf/mitrankova/PatternReco/75405_new
#name="HITS_AuAuFieldOn"

#runnumber=75391
#collision="run3auau"
#cdbtag="ana514_nocdbtag_v001"
#OutDir=/sphenix/tg/tg01/hf/mitrankova/PatternReco/75391_new
#name="HITS_AuAuFieldOn"

segment="$2"
nSkip="$3"

echo ========================================================
echo Running Fun4All_raw_hit_Poly_reco_pp_V0.C
echo "  nEvents             = ${nEvents}"
echo "  nSkip               = ${nSkip}"
echo "  Run Number          = ${runnumber}"
echo "  Segment             = ${segment}"
echo "  OutDir              = ${OutDir}"
echo "  collision           = ${collision}"
echo "  cdbtag              = ${cdbtag}"
echo "  name                = ${name}"
echo ========================================================


 root.exe -l -b << EOF
        .x Fun4All_raw_hit_Poly_reco_pp_V0.C(${nEvents},${runnumber}, ${segment},"${OutDir}",${nSkip},"${collision}","${cdbtag}","${name}")
EOF

echo all done
