// Remove the duplicate cycles of non-mergeable objects (TNamed / TObjString notes such as
// "track_class;1 ... ;3890") that hadd leaves in a merged file: one copy per input file.
// For every such object only the highest cycle is kept. Histograms and trees are not touched.
//   root -l -b -q 'CleanMetaCycles.C("MakePromptHadronHistograms_merged.root")'
#include <TClass.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TKey.h>
#include <TList.h>
#include <TString.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

static void CleanDirectory(TDirectory* dir, int& nRemoved)
{
  std::map<std::string, int> nCycles;    // object name -> number of cycles
  std::map<std::string, short> maxCycle; // object name -> highest cycle
  std::vector<std::string> subDirs;

  TIter next(dir->GetListOfKeys());
  while (TKey* key = static_cast<TKey*>(next()))
  {
    TClass* cl = TClass::GetClass(key->GetClassName());
    const std::string name = key->GetName();
    if (cl && cl->InheritsFrom(TDirectory::Class()))
    {
      if (nCycles.count("dir:" + name) == 0) subDirs.push_back(name);
      nCycles["dir:" + name] = 1;
      continue;
    }
    if (cl && (cl->InheritsFrom("TH1") || cl->InheritsFrom("TTree"))) continue;
    ++nCycles[name];
    if (key->GetCycle() > maxCycle[name]) maxCycle[name] = key->GetCycle();
  }

  for (const auto& entry : nCycles)
  {
    const std::string& name = entry.first;
    if (name.rfind("dir:", 0) == 0 || entry.second < 2) continue;

    TObject* obj = dir->Get(TString::Format("%s;%d", name.c_str(), maxCycle[name]));
    if (!obj) continue;
    TObject* keep = obj->Clone();

    dir->cd();
    dir->Delete(TString::Format("%s;*", name.c_str()));
    keep->Write(name.c_str());
    delete keep;

    nRemoved += entry.second - 1;
    std::cout << "  " << dir->GetPath() << " " << name << ": kept 1 of " << entry.second << " cycles" << std::endl;
  }

  for (const auto& sub : subDirs)
  {
    TDirectory* d = dir->GetDirectory(sub.c_str());
    if (d) CleanDirectory(d, nRemoved);
  }
}
//MakePromptHadronHistograms_merged
//MakeK0sPairHistograms_merged.root
//StudyV0TopologyQA_merged
//StudyPromptHadronTopologyQA_merged

void CleanMetaCycles(const char* fileName = "StudyPromptHadronTopologyQA_merged.root")
{
  TFile f(fileName, "UPDATE");
  if (f.IsZombie())
  {
    std::cerr << "cannot open " << fileName << std::endl;
    return;
  }
  int nRemoved = 0;
  CleanDirectory(&f, nRemoved);
  f.Close();
  std::cout << fileName << ": removed " << nRemoved << " duplicate cycles" << std::endl;
}
