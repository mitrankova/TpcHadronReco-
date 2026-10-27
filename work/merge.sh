#! /bin/bash
# Fast two-step merge of the per-file outputs of one macro (or of all four):
#   ./merge.sh MakeK0sPairHistograms        ->  ${OutDir}/MakeK0sPairHistograms_merged.root
#   ./merge.sh                              ->  all four
# Step 1: groups of ${Group} files are merged in parallel (${NPar} hadd at a time), uncompressed (-f0).
# Step 2: the ~40 group files are merged into the final file (compressed).
source /sphenix/u/mitrankova/.login_crossing_vertex_si_assoc
OutDir=/sphenix/tg/tg01/hf/mitrankova/Hadrons_FullRun/analysis
macros=${1:-"MakeK0sPairHistograms StudyV0TopologyQA MakePromptHadronHistograms StudyPromptHadronTopologyQA"}
Group=100
NPar=8

for m in ${macros}; do
  tmp=${OutDir}/${m}_merge_tmp
  rm -rf ${tmp}; mkdir -p ${tmp}
  ls ${OutDir}/${m}/${m}_*.root > ${tmp}/all.txt
  echo "${m}: $(wc -l < ${tmp}/all.txt) files"
  [ -s ${tmp}/all.txt ] || continue

  split -l ${Group} -d -a 3 --additional-suffix=.txt ${tmp}/all.txt ${tmp}/group_
  ls ${tmp}/group_*.txt | xargs -P ${NPar} -I{} sh -c 'hadd -f0 -k "${1%.txt}.root" @"$1" > "${1%.txt}.log" 2>&1' _ {}

  hadd -f -k ${OutDir}/${m}_merged.root ${tmp}/group_*.root && rm -rf ${tmp}
done