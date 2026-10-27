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
#include <TTreeFormula.h>

#include <algorithm>
#include <cmath>
#include <limits>
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
    legend->SetFillStyle(0);  // transparent: does not hide the histograms
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

// ============================================================
// Best cuts per species (mvtxS300), applied in ONE pass over each tree
// ============================================================
// The previous version filled every histogram with TTree::Project and long string selections:
// ~21 full passes over both chains, the Lambda_c mass written out 3 times per assignment, and
// "max(0.0, ...)", which TTreeFormula does not know (interpreted call per entry).  It also had
// three selection bugs:
//   * phi "PID" required one track > 300 and the other < 300 (the D0 K/pi assignment), not two kaons
//   * the K0S / Lambda "DIRA" used pca_x/y/z as the flight vector, but pca_* is the ABSOLUTE decay
//     point (not relative to the collision vertex, which sits at |z| up to ~15 cm): cos of the pair
//     momentum with the position vector, not the pointing angle.  The tree has the pointing angle
//     w.r.t. the collision vertex: cosThetaReco (= DIRA), and Lproj = |decay point - vertex|.
//   * Lambda_c: the pK Dalitz window 1.43-2.15 GeV with a 2.0-2.6 GeV axis left only 2.00-2.15 filled
// Now: one loop per tree with SetBranchAddress (only the needed branches read), the extra selection
// evaluated by one TTreeFormula.
//
// K0S     : pairDCA < 0.05 cm, Lproj > 0.05 cm, DIRA > v0DiraMin (default 0.985)
//           DIRA = cos(angle between the flight vector and the pair momentum v0_p), diraMode:
//             "vertex" (default): flight = (pca_x, pca_y, pca_z) - (pv_x, pv_y, pv_z), 3D
//             "xy"              : the same in the transverse plane (z of the vertex not needed)
//             "tree"            : the cosThetaReco branch of the tree
//             "origin"          : flight = (pca_x, pca_y, pca_z) from (0, 0, 0) (the earlier formula)
//           Page 2 of the pdf: K0S mass with / without the DIRA cut and the DIRA distributions of
//           the K0S peak and sidebands for every definition (which one separates signal).
// Lambda  : the same topology; plotted 1.08-1.16 GeV = the reconstruction window (set_lambda_selection)
// phi     : reference = MakeK0sPairHistograms promptMesons/primary_dz_0p3_pid:
//           pT(primary) > 0.15, ntpc >= 10, |dca_xy|, |dca_z| < 0.5 cm, |primary_pca_dz| < 0.3 cm,
//           kaon dE/dx > 300 BOTH tracks (phiPid "dedx300"), or the momentum-dependent band
//           (phiPid "nsigma": both K-compatible, >= 1 identified)
// D0      : d0v_status == 1, L_xy 80-700 um, DIRA > 0.985, d0*d0 < 0, d0v_pair_dca < 20 um,
//           |dca_xy| < 0.02 cm + dE/dx per K pi assignment (d0Pid 1: K and pi compatible, 2: K identified)
// Lambda_c: pK sub-system (two-body proxy): v0_pt > 1.5 GeV, |dca_xy| < 0.05 cm, |primary_pca_dz| < 0.3 cm,
//           |prompt_pairDCA| < 0.04 cm + dE/dx per p K assignment (lcPid 1: p identified + K compatible).
//           plotted over 2.0-2.5 GeV without a Dalitz window.  m(pK) of a Lambda_c -> p K pi decay lies
//           in 1.43-2.15 GeV, it does NOT peak at 2.286 GeV; a Lambda_c peak needs the pK pi triplet.
//
// dE/dx bands: <dE/dx>(beta gamma) fitted to the pion (K0S) and proton (Lambda) bands,
//   f = A/beta^K + C ln(beta gamma) + D, resolution 19 %  (as MakePromptHadronHistograms.C)
namespace
{
  constexpr double kPionMass = 0.13957039;
  constexpr double kKaonMass = 0.493677;
  constexpr double kProtonMass = 0.938272088;
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

  double dedxExpected(const double p, const double m)
  {
    const double bg = std::max(p / m, 0.05);
    const double beta = bg / std::sqrt(1.0 + bg * bg);
    return 232.916 / std::pow(beta, 2.138) + 64.414 * std::log(bg) - 133.213;
  }
  double nsig(const double dedx, const double p, const double m)
  {
    const double e = dedxExpected(p, m);
    return (e > 0 && std::isfinite(dedx) && dedx > 0 && std::isfinite(p) && p > 0) ? (dedx - e) / (0.19 * e) : 999.0;
  }
  bool kCompat(double dedx, double p) { return std::fabs(nsig(dedx, p, kKaonMass)) < 2.0; }
  bool kIdent(double dedx, double p) { return kCompat(dedx, p) && nsig(dedx, p, kPionMass) > 3.0 && nsig(dedx, p, kProtonMass) < -2.0; }
  bool piCompat(double dedx, double p) { return std::fabs(nsig(dedx, p, kPionMass)) < 3.0; }
  bool pIdent(double dedx, double p) { return std::fabs(nsig(dedx, p, kProtonMass)) < 2.0 && nsig(dedx, p, kKaonMass) > 2.0; }

  double mass2(double px1, double py1, double pz1, double m1, double px2, double py2, double pz2, double m2)
  {
    const double e1 = std::sqrt(px1 * px1 + py1 * py1 + pz1 * pz1 + m1 * m1);
    const double e2 = std::sqrt(px2 * px2 + py2 * py2 + pz2 * pz2 + m2 * m2);
    const double px = px1 + px2, py = py1 + py2, pz = pz1 + pz2;
    return std::sqrt(std::max(0.0, (e1 + e2) * (e1 + e2) - px * px - py * py - pz * pz));
  }

  struct Hists
  {
    TH1D *phiFree{nullptr}, *phiConstr{nullptr}, *d0{nullptr}, *k0{nullptr}, *lam{nullptr}, *lc{nullptr};
    Long64_t d0Pairs{0}, d0Both{0};
  };

  bool bindBranch(TChain *t, const char *name, void *addr, const bool required = true)
  {
    if (!t->GetBranch(name))
    {
      if (required)
      {
        std::cerr << "missing branch " << name << " in " << t->GetName() << std::endl;
      }
      return false;
    }
    t->SetBranchStatus(name, 1);
    t->SetBranchAddress(name, addr);
    return true;
  }

  // one pass over one tree; like = likeSignPairTree
  void runTree(TChain *t, const bool like, Hists &h, const TString &extra, const bool d0Unique,
               const std::string &phiPid, const int d0Pid, const int lcPid, const double v0DiraMin,
               const std::string &diraMode, TH1D *k0NoDira, TH1D *diraPeak[4], TH1D *diraSide[4])
  {
    if (!t || t->GetNtrees() == 0)
    {
      return;
    }
    UInt_t mask = 0, ntpc1 = 999, ntpc2 = 999;
    Int_t d0vStatus = 0;
    Float_t charge1 = 0, charge2 = 0, dedx1 = 0, dedx2 = 0;
    Float_t ppx1 = 0, ppy1 = 0, ppz1 = 0, ppx2 = 0, ppy2 = 0, ppz2 = 0;
    Float_t dxy1 = 0, dxy2 = 0, dz1 = 0, dz2 = 0, pcaDz = 0, promptPairDca = kNaN, v0pt = 0;
    Float_t pairDca = 0, lproj = 0, cosTheta = 0;
    Float_t pcaX = 0, pcaY = 0, pcaZ = 0, v0px = 0, v0py = 0, v0pz = 0, pvX = 0, pvY = 0, pvZ = 0;
    Float_t mPhi = 0, mPhiC = kNaN, mD0 = 0, mAD0 = 0, mK1Pi2 = 0, mPi1K2 = 0, mK0 = 0, mLam = 0, mALam = 0, mP1Pi2 = 0, mPi1P2 = 0;
    Float_t lxy = kNaN, dira = kNaN, dprod = kNaN, d0vPairDca = kNaN;

    t->SetBranchStatus("*", 0);
    bool ok = true;
    ok &= bindBranch(t, "candidate_mask", &mask);
    ok &= bindBranch(t, "charge1", &charge1);
    ok &= bindBranch(t, "charge2", &charge2);
    ok &= bindBranch(t, "dedx_1", &dedx1);
    ok &= bindBranch(t, "dedx_2", &dedx2);
    ok &= bindBranch(t, "primary_px1", &ppx1);
    ok &= bindBranch(t, "primary_py1", &ppy1);
    ok &= bindBranch(t, "primary_pz1", &ppz1);
    ok &= bindBranch(t, "primary_px2", &ppx2);
    ok &= bindBranch(t, "primary_py2", &ppy2);
    ok &= bindBranch(t, "primary_pz2", &ppz2);
    ok &= bindBranch(t, "dca_xy1", &dxy1);
    ok &= bindBranch(t, "dca_xy2", &dxy2);
    ok &= bindBranch(t, "dca_z1", &dz1);
    ok &= bindBranch(t, "dca_z2", &dz2);
    ok &= bindBranch(t, "primary_pca_dz", &pcaDz);
    ok &= bindBranch(t, "v0_pt", &v0pt);
    ok &= bindBranch(t, "pairDCA", &pairDca);
    ok &= bindBranch(t, "Lproj", &lproj);
    ok &= bindBranch(t, "cosThetaReco", &cosTheta);
    ok &= bindBranch(t, "pca_x", &pcaX);
    ok &= bindBranch(t, "pca_y", &pcaY);
    ok &= bindBranch(t, "pca_z", &pcaZ);
    ok &= bindBranch(t, "v0_px", &v0px);
    ok &= bindBranch(t, "v0_py", &v0py);
    ok &= bindBranch(t, "v0_pz", &v0pz);
    const bool havePv = bindBranch(t, "pv_x", &pvX, false) && bindBranch(t, "pv_y", &pvY, false) && bindBranch(t, "pv_z", &pvZ, false);
    if (!havePv && (diraMode == "vertex" || diraMode == "xy"))
    {
      std::cerr << t->GetName() << ": no pv_x/y/z - DIRA w.r.t. (0, 0, 0)" << std::endl;
    }
    ok &= bindBranch(t, "mass_Phi", &mPhi);
    ok &= bindBranch(t, "mass_Kshort", &mK0);
    ok &= bindBranch(t, "mass_Lambda", &mLam);
    ok &= bindBranch(t, "mass_AntiLambda", &mALam);
    ok &= bindBranch(t, "mass_P1Pi2", &mP1Pi2);
    ok &= bindBranch(t, "mass_Pi1P2", &mPi1P2);
    ok &= bindBranch(t, "mass_D0", &mD0);
    ok &= bindBranch(t, "mass_AntiD0", &mAD0);
    ok &= bindBranch(t, "mass_K1Pi2", &mK1Pi2);
    ok &= bindBranch(t, "mass_Pi1K2", &mPi1K2);
    if (!ok)
    {
      return;
    }
    bindBranch(t, "prompt_pairDCA", &promptPairDca, false);
    bindBranch(t, "mass_Phi_primary_constrained", &mPhiC, false);
    bindBranch(t, "ntpc_clusters1", &ntpc1, false);
    bindBranch(t, "ntpc_clusters2", &ntpc2, false);
    const bool haveD0v = bindBranch(t, "d0v_status", &d0vStatus, false) &&
                         bindBranch(t, "d0v_lxy_signed", &lxy, false) && bindBranch(t, "d0v_dira", &dira, false) &&
                         bindBranch(t, "d0v_dca_product", &dprod, false) && bindBranch(t, "d0v_pair_dca", &d0vPairDca, false);
    if (!haveD0v)
    {
      std::cerr << t->GetName() << ": no d0v_* branches - no D0 candidates (the D0 cuts need the D0 collision vertex)" << std::endl;
    }

    // extra selection: one formula (its branches are switched on), evaluated once per entry
    TTreeFormula *extraFormula = nullptr;
    if (extra.Length() > 0)
    {
      t->SetBranchStatus("*", 1);  // let the formula find its branches
      extraFormula = new TTreeFormula("extra", extra, t);
      if (extraFormula->GetNdim() == 0)
      {
        std::cerr << "invalid extra selection: " << extra << std::endl;
        delete extraFormula;
        return;
      }
      t->SetNotify(extraFormula);
    }

    const Long64_t n = t->GetEntries();
    int treeNumber = -1;
    for (Long64_t i = 0; i < n; ++i)
    {
      const Long64_t local = t->LoadTree(i);
      if (local < 0)
      {
        break;
      }
      if (extraFormula && t->GetTreeNumber() != treeNumber)
      {
        treeNumber = t->GetTreeNumber();
        extraFormula->UpdateFormulaLeaves();
      }
      t->GetEntry(i);
      if (extraFormula && extraFormula->EvalInstance() == 0)
      {
        continue;
      }
      const double p1 = std::sqrt(ppx1 * ppx1 + ppy1 * ppy1 + ppz1 * ppz1);
      const double p2 = std::sqrt(ppx2 * ppx2 + ppy2 * ppy2 + ppz2 * ppz2);
      const double pt1 = std::hypot(ppx1, ppy1), pt2 = std::hypot(ppx2, ppy2);
      const double maxDcaXY = std::max(std::fabs(dxy1), std::fabs(dxy2));
      const double maxDcaZ = std::max(std::fabs(dz1), std::fabs(dz2));

      // ---- K0S / Lambda: pairDCA < 0.05 cm, Lproj > 0.05 cm, DIRA > 0.985
      // DIRA definitions: [0] vertex 3D, [1] vertex xy, [2] tree cosThetaReco, [3] origin 3D
      double dirAll[4];
      {
        const double vx = havePv && std::isfinite(pvX) ? pvX : 0.0;
        const double vy = havePv && std::isfinite(pvY) ? pvY : 0.0;
        const double vz = havePv && std::isfinite(pvZ) ? pvZ : 0.0;
        const auto cos3 = [](double ax, double ay, double az, double bx, double by, double bz)
        {
          const double na = std::sqrt(ax * ax + ay * ay + az * az), nb = std::sqrt(bx * bx + by * by + bz * bz);
          return (na > 0 && nb > 0) ? (ax * bx + ay * by + az * bz) / (na * nb) : kNaN;
        };
        dirAll[0] = cos3(pcaX - vx, pcaY - vy, pcaZ - vz, v0px, v0py, v0pz);
        dirAll[1] = cos3(pcaX - vx, pcaY - vy, 0.0, v0px, v0py, 0.0);
        dirAll[2] = cosTheta;
        dirAll[3] = cos3(pcaX, pcaY, pcaZ, v0px, v0py, v0pz);
      }
      const int diraIndex = diraMode == "xy" ? 1 : (diraMode == "tree" ? 2 : (diraMode == "origin" ? 3 : 0));
      const double dira = dirAll[diraIndex];
      const bool v0TopoNoDira = pairDca < 0.05 && lproj > 0.05;
      const bool v0Topo = v0TopoNoDira && std::isfinite(dira) && dira > v0DiraMin;
      if (v0Topo && (mask & 1U))
      {
        h.k0->Fill(mK0);
      }
      if (!like && v0TopoNoDira && (mask & 1U))
      {
        if (k0NoDira) k0NoDira->Fill(mK0);
        const bool peak = mK0 > 0.480 && mK0 < 0.505;
        const bool side = (mK0 > 0.420 && mK0 < 0.445) || (mK0 > 0.545 && mK0 < 0.570);
        for (int k = 0; k < 4; ++k)
        {
          if (peak && diraPeak[k]) diraPeak[k]->Fill(std::isfinite(dirAll[k]) ? dirAll[k] : -2.0);
          if (side && diraSide[k]) diraSide[k]->Fill(std::isfinite(dirAll[k]) ? dirAll[k] : -2.0);
        }
      }
      if (v0Topo && (mask & 6U))
      {
        if (!like)
        {
          if (mask & 2U) h.lam->Fill(mLam);
          if (mask & 4U) h.lam->Fill(mALam);
        }
        else
        {
          // like sign: both proton assignments inside the window
          if (mP1Pi2 > 1.08 && mP1Pi2 < 1.16) h.lam->Fill(mP1Pi2);
          if (mPi1P2 > 1.08 && mPi1P2 < 1.16) h.lam->Fill(mPi1P2);
        }
      }

      // ---- phi
      if ((mask & 8U) && std::min(pt1, pt2) > 0.15 && std::min(ntpc1, ntpc2) >= 10 && maxDcaXY < 0.5 && maxDcaZ < 0.5 &&
          std::fabs(pcaDz) < 0.30)
      {
        bool pid = false;
        if (phiPid == "nsigma")
        {
          pid = kCompat(dedx1, p1) && kCompat(dedx2, p2) && (kIdent(dedx1, p1) || kIdent(dedx2, p2));
        }
        else
        {
          pid = dedx1 > 300 && dedx2 > 300;  // reference: kaon dE/dx > 300 on BOTH tracks
        }
        if (pid)
        {
          h.phiFree->Fill(mPhi);
          if (h.phiConstr && std::isfinite(mPhiC))
          {
            h.phiConstr->Fill(mPhiC);
          }
        }
      }

      // ---- D0: slot 16 = D0 (OS: negative track is the kaon) / K1Pi2 (LS: track 1 is the kaon)
      //          slot 32 = anti-D0 / Pi1K2
      if (haveD0v && (mask & 48U) && d0vStatus == 1 && lxy >= 0.008 && lxy <= 0.070 && dira > 0.985 && dprod < 0.0 &&
          d0vPairDca < 0.002 && std::fabs(dxy1) < 0.02 && std::fabs(dxy2) < 0.02)
      {
        const bool track1KaonInSlot16 = like ? true : (charge1 < 0);
        const auto pidOk = [&](const bool track1Kaon)
        {
          const double dK = track1Kaon ? dedx1 : dedx2, pK = track1Kaon ? p1 : p2;
          const double dPi = track1Kaon ? dedx2 : dedx1, pPi = track1Kaon ? p2 : p1;
          if (d0Pid <= 0) return true;
          return piCompat(dPi, pPi) && (d0Pid == 1 ? kCompat(dK, pK) : kIdent(dK, pK));
        };
        const bool s16 = (mask & 16U) && pidOk(track1KaonInSlot16);
        const bool s32 = (mask & 32U) && pidOk(!track1KaonInSlot16);
        if (s16 || s32)
        {
          ++h.d0Pairs;
          if (s16 && s32) ++h.d0Both;
          const bool fill16 = s16 && (!d0Unique || !s32);
          const bool fill32 = s32 && (!d0Unique || !s16);
          if (fill16) h.d0->Fill(like ? mK1Pi2 : mD0);
          if (fill32) h.d0->Fill(like ? mPi1K2 : mAD0);
        }
      }

      // ---- Lambda_c pK sub-system (two-body proxy), both assignments, PID per assignment
      if (v0pt > 1.5 && std::fabs(dxy1) < 0.05 && std::fabs(dxy2) < 0.05 && std::fabs(pcaDz) < 0.30 &&
          std::isfinite(promptPairDca) && std::fabs(promptPairDca) < 0.04)
      {
        for (int a = 0; a < 2; ++a)
        {
          const bool t1p = a == 0;  // track 1 = proton
          const double dP = t1p ? dedx1 : dedx2, pP = t1p ? p1 : p2;
          const double dK = t1p ? dedx2 : dedx1, pK = t1p ? p2 : p1;
          if (lcPid > 0 && !(pIdent(dP, pP) && (lcPid == 1 ? kCompat(dK, pK) : kIdent(dK, pK))))
          {
            continue;
          }
          const double m = t1p ? mass2(ppx1, ppy1, ppz1, kProtonMass, ppx2, ppy2, ppz2, kKaonMass)
                               : mass2(ppx1, ppy1, ppz1, kKaonMass, ppx2, ppy2, ppz2, kProtonMass);
          h.lc->Fill(m);  // histogram range 2.0-2.5 GeV (no Dalitz window: it would end the plot at 2.15)
        }
      }
    }
    if (extraFormula)
    {
      t->SetNotify(nullptr);
      delete extraFormula;
    }
  }
}  // namespace
///sphenix/tg/tg01/hf/mitrankova/Hadrons/hadron_sitpc_HITS_pp_mvtxS300_79513_*.root
//hadron_masses_mvtxS300
void PlotHadronMasses(const char *input = "/sphenix/tg/tg01/hf/mitrankova/Hadrons_FullRun/hadron_sitpc_HITS_pp_mvtxS300_79513_1*.root",
                      const char *extra_selection = "",
                      const char *output_pdf = "hadron_masses_mvtxS300_FullRun_1.pdf",
                      const char *d0_mode = "unique",
                      const char *phiPid = "dedx300",  // "dedx300" (reference) or "nsigma" (momentum-dependent band)
                      const int d0Pid = 1,               // 0 none, 1 K and pi compatible, 2 K identified
                      const int lcPid = 1,               // 0 none, 1 p identified + K compatible, 2 p and K identified
                      const double v0DiraMin = 0.985,    // K0S / Lambda: DIRA > v0DiraMin
                      const char *diraMode = "vertex")   // "vertex" (pca - pv, 3D), "xy", "tree" (cosThetaReco), "origin"
{
  gStyle->SetOptStat(0);

  auto *os_tree = new TChain("pairTree");
  auto *ls_tree = new TChain("likeSignPairTree");
  if (build_chains(input, os_tree, ls_tree) == 0)
  {
    std::cerr << "no input files for " << input << std::endl;
    return;
  }
  const TString extra = (extra_selection && std::string(extra_selection).size() > 0) ? TString(extra_selection) : TString("");
  const bool d0Unique = std::string(d0_mode) == "unique";

  Hists os, ls;
  os.phiFree = new TH1D("phi_os_free", "", 100, 0.98, 1.08);
  os.phiConstr = new TH1D("phi_os_constr", "", 100, 0.98, 1.08);
  ls.phiFree = new TH1D("phi_ls_free", "", 100, 0.98, 1.08);
  os.d0 = new TH1D("d0_os", "", 35, 1.70, 2.05);
  ls.d0 = new TH1D("d0_ls", "", 35, 1.70, 2.05);
  os.k0 = new TH1D("k0_os", "", 50, 0.40, 0.60);
  ls.k0 = new TH1D("k0_ls", "", 50, 0.40, 0.60);
  os.lam = new TH1D("lam_os", "", 32, 1.08, 1.16);
  ls.lam = new TH1D("lam_ls", "", 32, 1.08, 1.16);
  os.lc = new TH1D("lc_os", "", 25, 2.00, 2.50);
  ls.lc = new TH1D("lc_ls", "", 25, 2.00, 2.50);

  // DIRA diagnostics (opposite sign K0S after pairDCA + Lproj): mass without DIRA, DIRA in peak / sidebands
  auto *k0NoDira = new TH1D("k0_os_nodira", "", 50, 0.40, 0.60);
  const char *diraName[4] = {"vertex", "xy", "tree", "origin"};
  const char *diraTitle[4] = {"flight = pca - pv (3D)", "flight = pca - pv (xy)", "cosThetaReco (tree)", "flight = pca from (0,0,0)"};
  TH1D *diraPeak[4], *diraSide[4];
  for (int k = 0; k < 4; ++k)
  {
    diraPeak[k] = new TH1D(Form("dira_peak_%s", diraName[k]), Form("K^{0}_{S} DIRA: %s;DIRA;pairs (normalised)", diraTitle[k]), 100, -1.0, 1.0);
    diraSide[k] = new TH1D(Form("dira_side_%s", diraName[k]), "", 100, -1.0, 1.0);
  }
  runTree(os_tree, false, os, extra, d0Unique, phiPid, d0Pid, lcPid, v0DiraMin, diraMode, k0NoDira, diraPeak, diraSide);
  TH1D *noneP[4] = {nullptr, nullptr, nullptr, nullptr};
  runTree(ls_tree, true, ls, extra, d0Unique, phiPid, d0Pid, lcPid, v0DiraMin, diraMode, nullptr, noneP, noneP);

  // ---- Canvas: 3 rows x 2 cols
  auto *c = new TCanvas("c_hadron_masses", "hadron masses", 1200, 1350);
  c->Divide(2, 3);

  // Pad 1: phi - free mass (blue) + PV-constrained (red dashed), like sign (grey dotted)
  c->cd(1);
  os.phiFree->SetTitle(Form("#phi #rightarrow K^{+}K^{-} (|#Deltaz_{PCA}|<0.3 cm, %s)"
                            ";m_{KK} [GeV];pairs / 1 MeV",
                            std::string(phiPid) == "nsigma" ? "dE/dx bands" : "dE/dx>300 both"));
  os.phiFree->SetLineColor(kBlue + 1);
  os.phiFree->SetLineWidth(2);
  os.phiConstr->SetLineColor(kRed + 1);
  os.phiConstr->SetLineWidth(2);
  os.phiConstr->SetLineStyle(2);
  ls.phiFree->SetLineColor(kGray + 1);
  ls.phiFree->SetLineWidth(1);
  ls.phiFree->SetLineStyle(3);
  const double phi_ymax = 1.25 * std::max({os.phiFree->GetMaximum(), os.phiConstr->GetMaximum(), ls.phiFree->GetMaximum()});
  os.phiFree->SetMaximum(phi_ymax > 0 ? phi_ymax : 1.0);
  os.phiFree->SetMinimum(0);
  os.phiFree->Draw("hist");
  os.phiConstr->Draw("hist same");
  ls.phiFree->Draw("hist same");
  {
    const double ytop = os.phiFree->GetMaximum();
    auto *line = new TLine(1.019461, 0, 1.019461, ytop);
    line->SetLineColor(kGray + 2);
    line->SetLineStyle(3);
    line->SetLineWidth(2);
    line->Draw();
    auto *lbl = new TLatex(1.019461 - 0.001, 0.93 * ytop, "m_{PDG} = 1.019461 GeV");
    lbl->SetTextAlign(33);
    lbl->SetTextSize(0.035);
    lbl->SetTextColor(kGray + 3);
    lbl->Draw();
    auto *leg = new TLegend(0.52, 0.55, 0.88, 0.75);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);  // transparent: does not hide the histograms
    leg->AddEntry(os.phiFree, Form("OS free (%.0f)", os.phiFree->GetEntries()), "l");
    leg->AddEntry(os.phiConstr, Form("OS constr. (%.0f)", os.phiConstr->GetEntries()), "l");
    leg->AddEntry(ls.phiFree, Form("LS free (%.0f)", ls.phiFree->GetEntries()), "l");
    leg->Draw();
  }

  draw_pair(c, 2, os.d0, ls.d0, 1.86484, "1.86484",
            Form("D^{0}+#bar{D}^{0} (L_{xy} 80-700 #mum, DIRA>0.985, d_{pair}<20 #mum, %s)"
                 ";m_{K#pi} [GeV];pairs / 10 MeV",
                 d0Pid == 0 ? "no PID" : (d0Pid == 1 ? "K/#pi dE/dx" : "K identified")));
  draw_pair(c, 3, os.k0, ls.k0, 0.497611, "0.497611",
            Form("K^{0}_{S} (pairDCA<0.05 cm, L_{proj}>0.05 cm, DIRA>%.3f)"
                 ";m_{#pi#pi} [GeV];pairs / 4 MeV", v0DiraMin));
  draw_pair(c, 4, os.lam, ls.lam, 1.115683, "1.115683",
            Form("#Lambda+#bar{#Lambda} (pairDCA<0.05 cm, L_{proj}>0.05 cm, DIRA>%.3f)"
                 ";m_{p#pi} [GeV];pairs / 2.5 MeV", v0DiraMin));
  // Lambda_c: pK sub-mass in 2.0-2.5 GeV (a Lambda_c -> p K pi decay has m(pK) < 2.15 GeV, so this
  // range shows the pK combinatorial; the PDG line marks the Lambda_c mass for reference)
  draw_pair(c, 5, os.lc, ls.lc, 2.28646, "2.28646",
            Form("#Lambda_{c} #rightarrow pK#pi: m_{pK} sub-mass (p_{T}^{pK}>1.5 GeV, d_{pair}<0.04 cm, %s)"
                 ";m_{pK} [GeV];pairs / 20 MeV",
                 lcPid == 0 ? "no PID" : (lcPid == 1 ? "p id., K compat." : "p, K identified")));

  c->cd(6);
  auto *info = new TLatex(0.05, 0.92,
                          "#splitline{Best cuts (mvtxS300):}"
                          "{#splitline{#phi: n_{TPC}#geq10, |DCA|<0.5 cm, |#Deltaz_{PCA}|<0.3 cm, dE/dx PID}"
                          "{#splitline{K^{0}_{S}: pairDCA<0.05 cm, L_{proj}>0.05 cm, DIRA (cos#theta_{point})>0.985}"
                          "{#splitline{#Lambda: same topology, 1.08-1.16 GeV (reco window)}"
                          "{#splitline{D^{0}: L_{xy}[80,700]#mum, DIRA>0.985, d_{pair}<20#mum, K/#pi dE/dx}"
                          "{#Lambda_{c}: m_{pK} 2.0-2.5 GeV, p_{T}>1.5 GeV, d_{pair}<0.04 cm, p/K dE/dx}}}}}");
  info->SetNDC();
  info->SetTextSize(0.034);
  info->Draw();

  const TString pdf(output_pdf);
  c->Print(pdf + "(");

  // ---- page 2: DIRA diagnostics
  auto *c2 = new TCanvas("c_dira", "DIRA", 1200, 900);
  c2->Divide(3, 2);
  c2->cd(1);
  k0NoDira->SetTitle(Form("K^{0}_{S} OS: pairDCA, L_{proj} cuts, without / with DIRA>%.3f (%s);m_{#pi#pi} [GeV];pairs", v0DiraMin, diraMode));
  k0NoDira->SetLineColor(kGray + 2);
  k0NoDira->SetMinimum(0);
  k0NoDira->Draw("hist");
  os.k0->Draw("hist same");
  std::cout << "DIRA (" << diraMode << ") > " << v0DiraMin << ": K0S OS " << os.k0->GetEntries() << " of " << k0NoDira->GetEntries()
            << " after pairDCA + Lproj; fraction with DIRA > cut, K0S peak (0.480-0.505) / sidebands:" << std::endl;
  for (int k = 0; k < 4; ++k)
  {
    c2->cd(k + 2);
    gPad->SetLogy();
    const double np = diraPeak[k]->Integral(), ns = diraSide[k]->Integral();
    const double fp = np > 0 ? diraPeak[k]->Integral(diraPeak[k]->FindBin(v0DiraMin + 1e-6), diraPeak[k]->GetNbinsX() + 1) / np : 0;
    const double fs = ns > 0 ? diraSide[k]->Integral(diraSide[k]->FindBin(v0DiraMin + 1e-6), diraSide[k]->GetNbinsX() + 1) / ns : 0;
    std::cout << Form("  %-26s peak %.3f  sidebands %.3f", diraTitle[k], fp, fs) << std::endl;
    if (np > 0) diraPeak[k]->Scale(1.0 / np);
    if (ns > 0) diraSide[k]->Scale(1.0 / ns);
    diraPeak[k]->SetLineColor(kBlue + 1);
    diraSide[k]->SetLineColor(kRed + 1);
    diraSide[k]->SetLineStyle(2);
    diraPeak[k]->SetMaximum(2.0);
    diraPeak[k]->SetMinimum(1e-4);
    diraPeak[k]->Draw("hist");
    diraSide[k]->Draw("hist same");
    auto *t = new TLatex(0.15, 0.82, Form("frac > %.3f: peak %.2f, side %.2f", v0DiraMin, fp, fs));
    t->SetNDC();
    t->SetTextSize(0.045);
    t->Draw();
  }
  c2->Print(pdf + ")");

  std::cout << "selection: best cuts per species" << (extra.Length() ? " && (" + std::string(extra.Data()) + ")" : std::string()) << "\n"
            << "  phi free         OS " << os.phiFree->GetEntries() << "  LS " << ls.phiFree->GetEntries() << "\n"
            << "  phi constrained  OS " << os.phiConstr->GetEntries() << "\n"
            << "  D0+aD0           OS " << os.d0->GetEntries() << "  LS " << ls.d0->GetEntries() << "  (mode " << d0_mode
            << "; pairs OS " << os.d0Pairs << " of which both assignments " << os.d0Both << ", LS " << ls.d0Pairs << " / "
            << ls.d0Both << ")\n"
            << "  K0S              OS " << os.k0->GetEntries() << "  LS " << ls.k0->GetEntries() << "\n"
            << "  Lam+aLam         OS " << os.lam->GetEntries() << "  LS " << ls.lam->GetEntries() << "\n"
            << "  Lc pK sub-mass   OS " << os.lc->GetEntries() << "  LS " << ls.lc->GetEntries() << "\n";
}