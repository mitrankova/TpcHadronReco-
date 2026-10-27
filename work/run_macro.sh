#! /bin/bash
# One condor job: run one analysis macro on ONE hadron file.
#   run_macro.sh <macro> <input file>
#   e.g. run_macro.sh MakeK0sPairHistograms /sphenix/tg/tg01/hf/mitrankova/Hadrons_FullRun/hadron_sitpc_HITS_pp_79513_00012.root
# Output: ${OutDir}/<macro>/<macro>_<input file name>.root
export USER="$(id -u -n)"
export LOGNAME=${USER}
export HOME=/sphenix/u/${LOGNAME}
source /sphenix/u/mitrankova/.login_crossing_vertex_si_assoc

macro="$1"
infile="$2"
OutDir=/sphenix/tg/tg01/hf/mitrankova/Hadrons_FullRun/analysis
workdir=$(pwd)

if [ ! -s "${infile}" ]; then
  echo "skip empty file ${infile}"
  exit 0
fi

tag=$(basename "${infile}" .root)
outname=${macro}_${tag}.root
scratch=${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}/${macro}_${tag}
mkdir -p "${scratch}" "${OutDir}/${macro}"

cp "${workdir}/${macro}.C" "${scratch}/"
cd "${scratch}"
echo "${macro} on ${infile}"

# macros take (inputDir, filePattern, outputDir, outputName): the pattern is just the one file name
# "+" = compiled with ACLiC, much faster than the interpreter for the event loops
root.exe -l -b -q "${macro}.C+(\"$(dirname ${infile})\",\"$(basename ${infile})\",\"${scratch}\",\"${outname}\")"

cp "${scratch}/${outname}" "${OutDir}/${macro}/" && echo "written ${OutDir}/${macro}/${outname}"
rm -rf "${scratch}"
