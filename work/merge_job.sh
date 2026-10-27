#! /bin/bash
# One condor merge job.
#   merge_job.sh <macro> <group list>   step 1: hadd the files of one group  -> ${OutDir}/merge_groups/<macro>/<group>.root (uncompressed)
#   merge_job.sh <macro> final          step 2: hadd all group files          -> ${OutDir}/<macro>_merged.root
export USER="$(id -u -n)"
export LOGNAME=${USER}
export HOME=/sphenix/u/${LOGNAME}
source /sphenix/u/mitrankova/.login_crossing_vertex_si_assoc

macro="$1"
what="$2"
OutDir=/sphenix/tg/tg01/hf/mitrankova/Hadrons_FullRun/analysis
GroupDir=${OutDir}/merge_groups/${macro}
mkdir -p ${GroupDir}

if [ "${what}" = "final" ]; then
  echo "final merge of $(ls ${GroupDir}/*.root | wc -l) group files for ${macro}"
  hadd -f -k ${OutDir}/${macro}_merged.root ${GroupDir}/*.root || exit 1
  echo "written ${OutDir}/${macro}_merged.root"
else
  out=${GroupDir}/$(basename ${what} .txt).root
  echo "merging $(wc -l < ${what}) files of ${what}"
  hadd -f0 -k ${out} @${what} || exit 1
  echo "written ${out}"
fi
