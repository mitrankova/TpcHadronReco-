#! /bin/bash
# Step 0 (interactive, seconds): split the per-file outputs of each macro into groups of ${Group} files
# and write the index merge_lists/groups.txt ("<macro> <group list>" per line) used by merge_groups.job.
#   ./make_merge_lists.sh            (all four macros)
#   ./make_merge_lists.sh MakeK0sPairHistograms
OutDir=/sphenix/tg/tg01/hf/mitrankova/Hadrons_FullRun/analysis
macros=${1:-"MakeK0sPairHistograms StudyV0TopologyQA MakePromptHadronHistograms StudyPromptHadronTopologyQA"}
Group=100
ListDir=$(pwd)/merge_lists

rm -rf ${ListDir}; mkdir -p ${ListDir}
for m in ${macros}; do
  ls ${OutDir}/${m}/${m}_*.root > ${ListDir}/${m}_all.txt 2>/dev/null
  n=$(wc -l < ${ListDir}/${m}_all.txt)
  [ "${n}" -gt 0 ] || { echo "${m}: no files"; continue; }
  split -l ${Group} -d -a 3 --additional-suffix=.txt ${ListDir}/${m}_all.txt ${ListDir}/${m}_group_
  for g in ${ListDir}/${m}_group_*.txt; do echo "${m} ${g}" >> ${ListDir}/groups.txt; done
  echo "${m}: ${n} files -> $(ls ${ListDir}/${m}_group_*.txt | wc -l) groups"
done
