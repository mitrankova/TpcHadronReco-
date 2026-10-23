// Species-selected invariant-mass plots from the TpcV0CandidateTree output.
//
// pairTree / likeSignPairTree store ALL mass hypotheses for every written pair; a
// pair is written if it passes ANY species.  Each species must therefore be selected
// with its candidate_mask bit and drawn with ITS mass branch:
//
//   bit  species    opposite sign (pairTree)  like sign (likeSignPairTree)
//     1  K0S        mass_Kshort               mass_Kshort
//     2  Lambda     mass_Lambda               mass_P1Pi2 / mass_Pi1P2
//     4  anti-Lam.  mass_AntiLambda           mass_P1Pi2 / mass_Pi1P2
//     8  phi        mass_Phi                  mass_Phi
//    16  D0         mass_D0                   mass_K1Pi2
//    32  anti-D0    mass_AntiD0               mass_Pi1K2
//    64  J/psi      mass_Jpsi                 mass_Jpsi
//
// Input (first argument), read as a TChain of all matching files:
//   - one file                 "hadron_sitpc_xxx.root"
//   - a wildcard pattern       "/sphenix/tg/tg01/hf/mitrankova/Hadrons/hadron_sitpc_*.root"
//     (wildcards in the FILE name only, not in directory names)
//   - a list file (.list/.txt) one ROOT file per line; empty lines and lines starting with # skipped
//   - several of these, comma separated
// Every file is opened once and skipped (with a message) if it cannot be opened, is a zombie,
// was recovered (job killed while writing), or has no pairTree.
//
// Usage:
//   root -l 'PlotHadronMasses.C("Hadrons/hadron_sitpc_*.root")'
//   root -l 'PlotHadronMasses.C("files.list", "has_si1 && has_si2 && pv_source == 1")'
//   root -l 'PlotHadronMasses.C("Hadrons/hadron_sitpc_*.root", "", "hadron_masses.pdf", "both")'
//   root -l -e '.L PlotHadronMasses.C' -e 'ScanD0Bin("Hadrons/hadron_sitpc_*.root", 1.84, 1.85)'
//
// files.list can be made with:  ls /sphenix/tg/tg01/hf/mitrankova/Hadrons/hadron_sitpc_*.root > files.list

#include <TAxis.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TChainElement.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TObjArray.h>
#include <TString.h>
#include <TStyle.h>
#include <TTree.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace
{
  bool ends_with(const std::string &s, const std::string &suffix)
  {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
  }

  std::string trim(const std::string &s)
  {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos)
    {
      return "";
    }
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
  }

  // Expand one input token (file, wildcard pattern, or list file) into file names.
  void expand_input(const std::string &token, std::vector<std::string> &files)
  {
    if (token.empty())
    {
      return;
    }
    if (ends_with(token, ".list") || ends_with(token, ".txt"))
    {
      std::ifstream in(token);
      if (!in)
      {
        std::cerr << "cannot open list file " << token << std::endl;
        return;
      }
      std::string line;
      while (std::getline(in, line))
      {
        line = trim(line);
        if (!line.empty() && line[0] != '#')
        {
          expand_input(line, files);  // a list line may itself be a pattern
        }
      }
      return;
    }
    if (token.find_first_of("*?[") != std::string::npos)
    {
      // TChain expands wildcards in the file name; read back the expanded names
      TChain expander("pairTree");
      expander.Add(token.c_str());
      TIter next(expander.GetListOfFiles());
      while (auto *element = dynamic_cast<TChainElement *>(next()))
      {
        files.emplace_back(element->GetTitle());
      }
      if (expander.GetListOfFiles()->GetEntries() == 0)
      {
        std::cerr << "no file matches " << token << std::endl;
      }
      return;
    }
    files.push_back(token);
  }

  // Build the pairTree / likeSignPairTree chains from the input string.
  // Returns the number of files added.
  int build_chains(const char *input, TChain *os_chain, TChain *ls_chain)
  {
    std::vector<std::string> files;
    std::stringstream tokens(input ? input : "");
    std::string token;
    while (std::getline(tokens, token, ','))
    {
      expand_input(trim(token), files);
    }

    std::set<std::string> seen;
    int added = 0;
    int skipped = 0;
    Long64_t n_os = 0;
    Long64_t n_ls = 0;
    for (const auto &name : files)
    {
      if (!seen.insert(name).second)
      {
        continue;  // same file given twice
      }
      TFile *f = TFile::Open(name.c_str(), "READ");
      if (!f || f->IsZombie() || f->TestBit(TFile::kRecovered))
      {
        std::cerr << "skip (unreadable / zombie / recovered): " << name << std::endl;
        ++skipped;
        delete f;
        continue;
      }
      auto *os = dynamic_cast<TTree *>(f->Get("pairTree"));
      if (!os)
      {
        std::cerr << "skip (no pairTree): " << name << std::endl;
        ++skipped;
        f->Close();
        delete f;
        continue;
      }
      n_os += os->GetEntries();
      auto *ls = dynamic_cast<TTree *>(f->Get("likeSignPairTree"));
      if (ls)
      {
        n_ls += ls->GetEntries();
        ls_chain->Add(name.c_str());
      }
      os_chain->Add(name.c_str());
      ++added;
      f->Close();
      delete f;
    }
    std::cout << "input: " << added << " files (" << skipped << " skipped), pairTree " << n_os
              << " entries, likeSignPairTree " << n_ls << " entries" << std::endl;
    return added;
  }

  // Fill h from tree with expression expr under the selection; returns entries.
  Long64_t fill(TChain *tree, TH1D *h, const char *expr, const TString &selection)
  {
    if (!tree || tree->GetNtrees() == 0)
    {
      return 0;
    }
    return tree->Project(h->GetName(), expr, selection, "goff");
  }

  Long64_t count(TChain *tree, const TString &selection)
  {
    return (tree && tree->GetNtrees() > 0) ? tree->GetEntries(selection) : 0;
  }

  // reference_mass: PDG mass [GeV] for the dotted line; pdg_label: the same value as printed text
  void draw_pair(TCanvas *c, int pad, TH1D *os, TH1D *ls, double reference_mass,
                 const char *pdg_label, const char *title)
  {
    c->cd(pad);
    os->SetTitle(title);
    os->SetLineColor(kBlue + 1);
    os->SetLineWidth(2);
    ls->SetLineColor(kRed + 1);
    ls->SetLineWidth(2);
    ls->SetLineStyle(2);
    const double ymax = 1.25 * std::max(os->GetMaximum(), ls->GetMaximum());
    os->SetMaximum(ymax > 0 ? ymax : 1.0);
    os->SetMinimum(0);
    os->Draw("hist");
    ls->Draw("hist same");
    if (reference_mass > 0)
    {
      // PDG mass: dotted line over the full height + its value next to the line
      const double ytop = os->GetMaximum();
      auto *line = new TLine(reference_mass, 0, reference_mass, ytop);
      line->SetLineColor(kGray + 2);
      line->SetLineStyle(3);
      line->SetLineWidth(2);
      line->Draw();

      const double xmin = os->GetXaxis()->GetXmin();
      const double xmax = os->GetXaxis()->GetXmax();
      const bool label_left = reference_mass > xmin + 0.6 * (xmax - xmin);
      const double offset = 0.01 * (xmax - xmin);
      auto *label = new TLatex(label_left ? reference_mass - offset : reference_mass + offset,
                               0.93 * ytop,
                               Form("m_{PDG} = %s GeV", pdg_label));
      label->SetTextAlign(label_left ? 33 : 13);  // right- or left-aligned, top
      label->SetTextSize(0.035);
      label->SetTextColor(kGray + 3);
      label->Draw();
    }
    // below the PDG-mass label, which sits at the top of the frame
    auto *legend = new TLegend(0.55, 0.62, 0.88, 0.75);
    legend->SetBorderSize(0);
    legend->AddEntry(os, Form("opposite sign (%.0f)", os->GetEntries()), "l");
    legend->AddEntry(ls, Form("like sign (%.0f)", ls->GetEntries()), "l");
    legend->Draw();
  }
}  // namespace

// List the D0 / anti-D0 entries in a mass interval, to check a spike: the same
// (evt, track_id1, track_id2) twice would be a real duplicate; the same evt with one
// common track and a near-identical partner (same pt, eta, phi) would be a track clone.
// With a chain, run and evt identify the event; the input is read as in PlotHadronMasses.
void ScanD0Bin(const char *input, const double mass_min = 1.84, const double mass_max = 1.85,
               const char *tree_name = "pairTree")
{
  auto *os_chain = new TChain("pairTree");
  auto *ls_chain = new TChain("likeSignPairTree");
  if (build_chains(input, os_chain, ls_chain) == 0)
  {
    std::cerr << "no input files" << std::endl;
    return;
  }
  const bool like_sign = std::string(tree_name) == "likeSignPairTree";
  TChain *tree = like_sign ? ls_chain : os_chain;
  if (tree->GetNtrees() == 0)
  {
    std::cerr << "no " << tree_name << " in the input" << std::endl;
    return;
  }
  const char *m_d0 = like_sign ? "mass_K1Pi2" : "mass_D0";
  const char *m_ad0 = like_sign ? "mass_Pi1K2" : "mass_AntiD0";
  const TString selection = TString::Format(
      "((candidate_mask & 16) && %s > %g && %s < %g) || ((candidate_mask & 32) && %s > %g && %s < %g)",
      m_d0, mass_min, m_d0, mass_max, m_ad0, mass_min, m_ad0, mass_max);
  tree->SetScanField(0);
  tree->Scan(TString::Format("run:evt:track_id1:track_id2:crossing1:candidate_mask:%s:%s:"
                             "primary_pt1:primary_pt2:pv_z:dca_xy1:dca_xy2",
                             m_d0, m_ad0),
             selection, "colsize=11 precision=5");
}

//"both"
void PlotHadronMasses(const char *input = "/sphenix/tg/tg01/hf/mitrankova/Hadrons/hadron_sitpc_HITS_pp_79513_1*.root",
                      const char *extra_selection = "",
                      const char *output_pdf = "hadron_masses_new_si_aligned_si_fit_7780K.pdf",
                      const char *d0_mode = "unique")
{
  gStyle->SetOptStat(0);

  auto *os_tree = new TChain("pairTree");
  auto *ls_tree = new TChain("likeSignPairTree");
  if (build_chains(input, os_tree, ls_tree) == 0)
  {
    std::cerr << "no input files for " << input << std::endl;
    return;
  }

  const TString extra = (extra_selection && std::string(extra_selection).size() > 0)
                            ? TString::Format(" && (%s)", extra_selection)
                            : TString("");
  auto sel = [&extra](const char *mask) { return TString::Format("(%s)%s", mask, extra.Data()); };

  // ---- phi -> K+ K-   (PDG 1.019461, threshold 0.98735)
  auto *phi_os = new TH1D("phi_os", "", 50, 0.98, 1.08);
  auto *phi_ls = new TH1D("phi_ls", "", 50, 0.98, 1.08);
  fill(os_tree, phi_os, "mass_Phi", sel("(candidate_mask & 8) != 0"));
  fill(ls_tree, phi_ls, "mass_Phi", sel("(candidate_mask & 8) != 0"));

  // ---- D0 + anti-D0 -> K pi   (PDG 1.86484)
  // Without PID a pair has two K/pi assignments.  A pair whose BOTH assignments fall in
  // the window carries both bits (16 and 32) and enters twice, at two different masses:
  // the right assignment and its reflection.  This is not a duplicated pair (each pair is
  // written once), but entries != pairs.
  //   d0_mode = "both"   : both assignments (standard without PID; OS and LS treated alike)
  //   d0_mode = "unique" : only pairs with exactly one assignment in the window
  const bool d0_unique = std::string(d0_mode) == "unique";
  const char *d0_only = d0_unique ? "(candidate_mask & 48) == 16" : "(candidate_mask & 16) != 0";
  const char *ad0_only = d0_unique ? "(candidate_mask & 48) == 32" : "(candidate_mask & 32) != 0";
  auto *d0_os = new TH1D("d0_os", "", 35, 1.70, 2.05);
  auto *d0_ls = new TH1D("d0_ls", "", 35, 1.70, 2.05);
  auto *tmp = new TH1D("d0_tmp", "", 35, 1.70, 2.05);
  fill(os_tree, d0_os, "mass_D0", sel(d0_only));
  fill(os_tree, tmp, "mass_AntiD0", sel(ad0_only));
  d0_os->Add(tmp);
  tmp->Reset();
  fill(ls_tree, d0_ls, "mass_K1Pi2", sel(d0_only));
  fill(ls_tree, tmp, "mass_Pi1K2", sel(ad0_only));
  d0_ls->Add(tmp);
  const Long64_t d0_pairs_os = count(os_tree, sel("(candidate_mask & 48) != 0"));
  const Long64_t d0_both_os = count(os_tree, sel("(candidate_mask & 48) == 48"));
  const Long64_t d0_pairs_ls = count(ls_tree, sel("(candidate_mask & 48) != 0"));
  const Long64_t d0_both_ls = count(ls_tree, sel("(candidate_mask & 48) == 48"));

  // ---- K0S -> pi+ pi-   (PDG 0.497611)
  auto *k0_os = new TH1D("k0_os", "", 50, 0.40, 0.60);
  auto *k0_ls = new TH1D("k0_ls", "", 50, 0.40, 0.60);
  fill(os_tree, k0_os, "mass_Kshort", sel("(candidate_mask & 1) != 0"));
  fill(ls_tree, k0_ls, "mass_Kshort", sel("(candidate_mask & 1) != 0"));

  // ---- Lambda + anti-Lambda -> p pi   (PDG 1.115683)
  auto *lam_os = new TH1D("lam_os", "", 40, 1.08, 1.16);
  auto *lam_ls = new TH1D("lam_ls", "", 40, 1.08, 1.16);
  tmp = new TH1D("lam_tmp", "", 40, 1.08, 1.16);
  fill(os_tree, lam_os, "mass_Lambda", sel("(candidate_mask & 2) != 0"));
  fill(os_tree, tmp, "mass_AntiLambda", sel("(candidate_mask & 4) != 0"));
  lam_os->Add(tmp);
  tmp->Reset();
  // like sign: both proton assignments (only the one inside the window passed)
  fill(ls_tree, lam_ls, "mass_P1Pi2", sel("(candidate_mask & 6) != 0 && mass_P1Pi2 > 1.08 && mass_P1Pi2 < 1.16"));
  fill(ls_tree, tmp, "mass_Pi1P2", sel("(candidate_mask & 6) != 0 && mass_Pi1P2 > 1.08 && mass_Pi1P2 < 1.16"));
  lam_ls->Add(tmp);

  auto *c = new TCanvas("c_hadron_masses", "hadron masses", 1200, 900);
  c->Divide(2, 2);
  // PDG (RPP 2024) masses
  draw_pair(c, 1, phi_os, phi_ls, 1.019461, "1.019461",
            "#phi #rightarrow K^{+}K^{-};m_{KK} [GeV];pairs / 2 MeV");
  draw_pair(c, 2, d0_os, d0_ls, 1.86484, "1.86484",
            "D^{0} + #bar{D}^{0} #rightarrow K#pi;m_{K#pi} [GeV];pairs / 10 MeV");
  draw_pair(c, 3, k0_os, k0_ls, 0.497611, "0.497611",
            "K^{0}_{S} #rightarrow #pi^{+}#pi^{-};m_{#pi#pi} [GeV];pairs / 4 MeV");
  draw_pair(c, 4, lam_os, lam_ls, 1.115683, "1.115683",
            "#Lambda + #bar{#Lambda} #rightarrow p#pi;m_{p#pi} [GeV];pairs / 2 MeV");
  c->SaveAs(output_pdf);

  std::cout << "selection: candidate_mask bit" << extra << std::endl
            << "  phi      OS " << phi_os->GetEntries() << "  LS " << phi_ls->GetEntries() << std::endl
            << "  D0+aD0   OS " << d0_os->GetEntries() << "  LS " << d0_ls->GetEntries()
            << "  (mode " << d0_mode << "; pairs OS " << d0_pairs_os << " of which both assignments "
            << d0_both_os << ", LS " << d0_pairs_ls << " / " << d0_both_ls << ")" << std::endl
            << "  K0S      OS " << k0_os->GetEntries() << "  LS " << k0_ls->GetEntries() << std::endl
            << "  Lam+aLam OS " << lam_os->GetEntries() << "  LS " << lam_ls->GetEntries() << std::endl;
}