// StudyPromptHadronTopologyQA.C
//
// One-pass topology QA for phi, D0/anti-D0, and Lambda_c candidates.
// Mirrors the structure of StudyV0TopologyQA.C but is tuned to prompt and
// semi-displaced mesons instead of K0S/Lambda V0s.
//
// For each species the macro produces:
//   * mass vs every topology variable (2D),
//   * mass distribution in peak, left-sideband, right-sideband,
//   * the same topology variable distributions in each mass window,
//   * TH3F: pair pT x mass x key topology variable.
//
// Three separate exactCut tiers allow quick before/after comparisons:
//   exactCut1  "loose kinematics only"   – no topology beyond pT / DCA
//   exactCut2  "Si-scale DCA"            – tight DCA_xy (exploiting Si resolution)
//   exactCut3  "full topology"           – Si DCA + d0v variables for D0
//
// Like-sign background: likeSignPairTree is processed in the same run (processLikeSign) into
// likeSign/<species>/<tier>/... with the same histogram names (D0 slot = K1Pi2, track 1 kaon;
// AntiD0 slot = Pi1K2, track 2 kaon).
//
// PID (all tiers): momentum-dependent dE/dx bands as MakePromptHadronHistograms.C / StudyPhiD0.C
//   phiPidMode 1 both K-compatible, 2 + >= 1 identified kaon (default), 3 both identified
//   d0PidMode  1 K- and pi-compatible (default), 2 K identified        (per K pi assignment)
//   lcPidMode  1 p identified + K-compatible (default), 2 p and K identified (per p K assignment)
// phi: the reference selection of MakeK0sPairHistograms promptMesons/primary_dz_0p3_pid (the
// visible peak) in every tier: ntpc >= 10, |dz of the primary PCAs| < 0.3 cm, momenta at the primary
// PCA (free mass; the PV-constrained mass gave the same peak and is kept under constrained/).
// D0: exactCut3 = L_xy window (w.r.t. the D0 vertex refitted without the daughters, the key cut)
// + DIRA + opposite-sign impact parameters.
//
// Usage:
//   root -l -b -q \
//   'StudyPromptHadronTopologyQA.C("/path","hadron_sitpc_*.root","out",
//                                  "topo_prompt.root","pairTree")'
//
// Si+TPC conventions: same as StudyV0TopologyQA / MakeK0sPairHistograms.

#include <TChain.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH1I.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TMath.h>
#include <TNamed.h>
#include <TString.h>
#include <TSystem.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace
{
  // PDG masses [GeV/c^2]
  constexpr double kKaonMass    = 0.493677;
  constexpr double kPionMass    = 0.13957039;
  constexpr double kProtonMass  = 0.938272088;
  constexpr double kTwoPi       = 2.0 * TMath::Pi();

  // Candidate mask bits (must match TpcV0CandidateTree::CandidateMask)
  constexpr UInt_t kMaskPhi    = 1U << 3;
  constexpr UInt_t kMaskD0     = 1U << 4;
  constexpr UInt_t kMaskAntiD0 = 1U << 5;

  // Mass windows [GeV/c^2]
  // phi peak at ~1.0177 GeV (momentum scale ~0.2 % low), resolution + width ~3.5 MeV
  constexpr double kPhiPeakLo  = 1.010, kPhiPeakHi  = 1.026;
  constexpr double kPhiSBW     = 0.016;   // sideband width = peak width
  constexpr double kD0PeakLo   = 1.840, kD0PeakHi   = 1.890;
  constexpr double kD0SBW      = 0.050;
  constexpr double kLcPeakLo   = 2.270, kLcPeakHi   = 2.310;
  constexpr double kLcSBW      = 0.040;

  double kNaN() { return std::numeric_limits<double>::quiet_NaN(); }

  bool branchExists(TChain& c, const char* n) { return c.GetBranch(n) != nullptr; }

  double invMass(double px1,double py1,double pz1,double m1,
                 double px2,double py2,double pz2,double m2)
  {
    const double e1=std::sqrt(px1*px1+py1*py1+pz1*pz1+m1*m1);
    const double e2=std::sqrt(px2*px2+py2*py2+pz2*pz2+m2*m2);
    const double px=px1+px2,py=py1+py2,pz=pz1+pz2;
    return std::sqrt(std::max(0.0,(e1+e2)*(e1+e2)-px*px-py*py-pz*pz));
  }

  enum class MassRegion { Peak, LeftSide, RightSide, Outside };

  MassRegion classifyMass(double m, double peakLo, double peakHi, double sbW)
  {
    if (m >= peakLo && m < peakHi) return MassRegion::Peak;
    if (m >= peakLo-2*sbW && m < peakLo-sbW) return MassRegion::LeftSide;
    if (m >= peakHi+sbW  && m < peakHi+2*sbW) return MassRegion::RightSide;
    return MassRegion::Outside;
  }

  // -----------------------------------------------------------------------
  // dE/dx PID (fit of the pion / proton bands, see the header)
  // -----------------------------------------------------------------------
  struct DedxModel
  {
    double A{232.916}, K{2.138}, C{64.414}, D{-133.213}, sigmaRel{0.19};
    double expected(const double p, const double m) const
    {
      const double bg = std::max(p / m, 0.05);
      const double beta = bg / std::sqrt(1.0 + bg * bg);
      return A / std::pow(beta, K) + C * std::log(bg) + D;
    }
    double nsigma(const double dedx, const double p, const double m) const
    {
      const double e = expected(p, m);
      return (e > 0 && std::isfinite(dedx) && dedx > 0 && std::isfinite(p) && p > 0) ? (dedx - e) / (sigmaRel * e) : 999.0;
    }
  };
  const DedxModel kDedx;
  bool kaonCompatible(double dedx, double p) { return std::fabs(kDedx.nsigma(dedx, p, kKaonMass)) < 2.0; }
  bool kaonIdentified(double dedx, double p)
  {
    return kaonCompatible(dedx, p) && kDedx.nsigma(dedx, p, kPionMass) > 3.0 && kDedx.nsigma(dedx, p, kProtonMass) < -2.0;
  }
  bool pionCompatible(double dedx, double p) { return std::fabs(kDedx.nsigma(dedx, p, kPionMass)) < 3.0; }
  bool protonIdentified(double dedx, double p)
  {
    return std::fabs(kDedx.nsigma(dedx, p, kProtonMass)) < 2.0 && kDedx.nsigma(dedx, p, kKaonMass) > 2.0;
  }
  bool phiPidPass(int mode, double dedx1, double p1, double dedx2, double p2)
  {
    if (mode <= 0) return true;
    if (!kaonCompatible(dedx1, p1) || !kaonCompatible(dedx2, p2)) return false;
    const int nid = (kaonIdentified(dedx1, p1) ? 1 : 0) + (kaonIdentified(dedx2, p2) ? 1 : 0);
    return mode == 1 || (mode == 2 && nid >= 1) || (mode >= 3 && nid == 2);
  }
  bool d0PidPass(int mode, double dedxK, double pK, double dedxPi, double pPi)
  {
    if (mode <= 0) return true;
    if (!pionCompatible(dedxPi, pPi)) return false;
    return mode == 1 ? kaonCompatible(dedxK, pK) : kaonIdentified(dedxK, pK);
  }
  bool lcPidPass(int mode, double dedxP, double pP, double dedxK, double pK)
  {
    if (mode <= 0) return true;
    if (!protonIdentified(dedxP, pP)) return false;
    return mode == 1 ? kaonCompatible(dedxK, pK) : kaonIdentified(dedxK, pK);
  }

  // -----------------------------------------------------------------------
  // Per-species topology histogram set
  // -----------------------------------------------------------------------

  // Common set of mass-vs-topology 2D histograms for one species + one mass window
  struct WindowQA
  {
    TH1F* h_mass           = nullptr;
    TH2F* h_mass_vs_pt     = nullptr;
    TH2F* h_mass_vs_eta    = nullptr;
    TH2F* h_mass_vs_phi    = nullptr;
    TH2F* h_mass_vs_dcaxy  = nullptr;   // max|DCA_xy| per pair
    TH2F* h_mass_vs_dcaz   = nullptr;
    TH2F* h_mass_vs_dz_pca = nullptr;   // |pca1_z - pca2_z|
    TH2F* h_mass_vs_pairdca= nullptr;   // |prompt pair DCA|
    TH2F* h_mass_vs_lxy    = nullptr;   // d0v_lxy_signed (D0 only)
    TH2F* h_mass_vs_dira   = nullptr;
    TH2F* h_mass_vs_dcaproduct = nullptr;
    TH2F* h_mass_vs_npoints= nullptr;
    TH2F* h_mass_vs_dedx1  = nullptr;
    TH2F* h_mass_vs_dedx2  = nullptr;
    TH3F* h3_mass_pt_pairdca = nullptr;
    TH3F* h3_mass_pt_lxy     = nullptr; // D0 only
  };

  // Per-topology distributions in peak/sideband windows
  struct WindowDist
  {
    TH1F* h_dcaxy    = nullptr;
    TH1F* h_dcaz     = nullptr;
    TH1F* h_dz_pca   = nullptr;
    TH1F* h_pairdca  = nullptr;
    TH1F* h_lxy      = nullptr;
    TH1F* h_dira     = nullptr;
    TH1F* h_dcaprod  = nullptr;
    TH1F* h_npoints  = nullptr;
    TH1F* h_pt       = nullptr;
    TH2F* h_dedx_vs_p= nullptr;
    TH2F* h_pt1_vs_pt2=nullptr;
  };

  struct SpeciesQA
  {
    // Overall: all entries
    WindowQA overall;
    // By mass window
    WindowQA peak, leftSB, rightSB;
    WindowDist peakDist, leftSBDist, rightSBDist;
    // Constrained mass counterparts
    TH1F* h_mass_constrained = nullptr;
    TH2F* h_mass_constrained_vs_pt = nullptr;
    TH2F* h_mass_vs_mass_constrained = nullptr;
  };

  void bookWindowQA(WindowQA& q, TDirectory* dir, const std::string& tag,
                    double massLo, double massHi, int nMassBins,
                    bool hasLxy, bool hasPromptDCA)
  {
    dir->cd();
    // Use TString::Format throughout to avoid std::string+TString ambiguity in cling.
    const TString T = tag.c_str();
    const TString t = TString::Format(" [%s]", tag.c_str());
    q.h_mass = new TH1F("h_mass",
      T+" invariant mass"+t+";m [GeV/c^{2}];pairs",
      nMassBins, massLo, massHi);
    q.h_mass_vs_pt = new TH2F("h_mass_vs_pt",
      "mass vs p_{T}"+t+";p_{T}^{pair} [GeV/c];m [GeV/c^{2}]",
      60,0,12, nMassBins,massLo,massHi);
    q.h_mass_vs_eta = new TH2F("h_mass_vs_eta",
      "mass vs #eta"+t+";#eta^{pair};m [GeV/c^{2}]",
      60,-3,3, nMassBins,massLo,massHi);
    q.h_mass_vs_phi = new TH2F("h_mass_vs_phi",
      "mass vs #phi"+t+";#phi^{pair} [rad];m [GeV/c^{2}]",
      64,-TMath::Pi(),TMath::Pi(), nMassBins,massLo,massHi);
    q.h_mass_vs_dcaxy = new TH2F("h_mass_vs_dcaxy",
      "mass vs max|DCA_{xy}|"+t+";max|DCA_{xy}| [cm];m [GeV/c^{2}]",
      200,0,0.5, nMassBins,massLo,massHi);
    q.h_mass_vs_dcaz = new TH2F("h_mass_vs_dcaz",
      "mass vs max|DCA_{z}|"+t+";max|DCA_{z}| [cm];m [GeV/c^{2}]",
      200,0,1.0, nMassBins,massLo,massHi);
    q.h_mass_vs_dz_pca = new TH2F("h_mass_vs_dz_pca",
      "mass vs |#Deltaz_{PCA}|"+t+";|pca1_z-pca2_z| [cm];m [GeV/c^{2}]",
      200,0,2.0, nMassBins,massLo,massHi);
    q.h_mass_vs_npoints = new TH2F("h_mass_vs_npoints",
      "mass vs min(npoints)"+t+";min npoints;m [GeV/c^{2}]",
      80,0,80, nMassBins,massLo,massHi);
    q.h_mass_vs_dedx1 = new TH2F("h_mass_vs_dedx1",
      "mass vs dE/dx (daughter 1)"+t+";dE/dx [arb.];m [GeV/c^{2}]",
      200,0,2000, nMassBins,massLo,massHi);
    q.h_mass_vs_dedx2 = new TH2F("h_mass_vs_dedx2",
      "mass vs dE/dx (daughter 2)"+t+";dE/dx [arb.];m [GeV/c^{2}]",
      200,0,2000, nMassBins,massLo,massHi);
    if (hasPromptDCA)
    {
      q.h_mass_vs_pairdca = new TH2F("h_mass_vs_pairdca",
        "mass vs |prompt pair DCA|"+t+";|prompt pair DCA| [cm];m [GeV/c^{2}]",
        200,0,0.3, nMassBins,massLo,massHi);
      q.h3_mass_pt_pairdca = new TH3F("h3_mass_pt_pairdca",
        "mass vs p_{T} vs |pair DCA|"+t+";p_{T} [GeV/c];m [GeV/c^{2}];|pair DCA| [cm]",
        30,0,6, nMassBins/4,massLo,massHi, 20,0,0.3);
    }
    if (hasLxy)
    {
      q.h_mass_vs_lxy = new TH2F("h_mass_vs_lxy",
        "mass vs L_{xy}^{signed}"+t+";L_{xy}^{signed} [cm];m [GeV/c^{2}]",
        150,-0.1,0.2, nMassBins,massLo,massHi);  // 20 um bins
      q.h_mass_vs_dira = new TH2F("h_mass_vs_dira",
        "mass vs DIRA"+t+";DIRA;m [GeV/c^{2}]",
        200,0.9,1.0, nMassBins,massLo,massHi);
      q.h_mass_vs_dcaproduct = new TH2F("h_mass_vs_dcaproduct",
        "mass vs DCA product"+t+";d_{1}#timesd_{2} [cm^{2}];m [GeV/c^{2}]",
        200,-2e-4,2e-4, nMassBins,massLo,massHi);
      q.h3_mass_pt_lxy = new TH3F("h3_mass_pt_lxy",
        "mass vs p_{T} vs L_{xy}"+t+";p_{T} [GeV/c];m [GeV/c^{2}];L_{xy} [cm]",
        30,0,6, nMassBins/4,massLo,massHi, 40,-0.05,0.15);  // 50 um L_xy bins
    }
  }

  void bookWindowDist(WindowDist& d, TDirectory* dir, const std::string& tag,
                      bool hasLxy)
  {
    dir->cd();
    const TString t = TString::Format(" [%s]", tag.c_str());
    d.h_dcaxy    = new TH1F("h_dcaxy",     "max|DCA_{xy}|"+t+";max|DCA_{xy}| [cm];pairs",    200,0,0.5);
    d.h_dcaz     = new TH1F("h_dcaz",      "max|DCA_{z}|"+t+";max|DCA_{z}| [cm];pairs",      200,0,1.0);
    d.h_dz_pca   = new TH1F("h_dz_pca",   "|#Deltaz_{PCA}|"+t+";|pca1_z-pca2_z| [cm];pairs",200,0,2.0);
    d.h_pairdca  = new TH1F("h_pairdca",   "|prompt pair DCA|"+t+";|pair DCA| [cm];pairs",    200,0,0.3);
    d.h_npoints  = new TH1F("h_npoints",   "min npoints"+t+";min npoints;pairs",              80,0,80);
    d.h_pt       = new TH1F("h_pt",        "pair p_{T}"+t+";p_{T} [GeV/c];pairs",             60,0,12);
    d.h_dedx_vs_p= new TH2F("h_dedx_vs_p", "both daughters: dE/dx vs p"+t+";p [GeV/c];dE/dx [arb.]",60,0,6,200,0,2000);
    d.h_pt1_vs_pt2=new TH2F("h_pt1_vs_pt2","p_{T,1} vs p_{T,2}"+t+";p_{T,1} [GeV/c];p_{T,2} [GeV/c]",60,0,6,60,0,6);
    if (hasLxy)
    {
      d.h_lxy    = new TH1F("h_lxy",    "L_{xy}^{signed}"+t+";L_{xy}^{signed} [cm];pairs",150,-0.1,0.2);
      d.h_dira   = new TH1F("h_dira",   "DIRA"+t+";DIRA;pairs",                            200,0.9,1.0);
      d.h_dcaprod= new TH1F("h_dcaprod","DCA product"+t+";d_{1}#timesd_{2} [cm^{2}];pairs",200,-2e-4,2e-4);
    }
  }

  void bookSpeciesQA(SpeciesQA& sq, TDirectory* topDir, const std::string& specName,
                     double massLo, double massHi, int nMassBins,
                     bool hasLxy, bool hasPromptDCA)
  {
    const TString sp = specName.c_str();
    auto mkDir = [&](const char* sub) {
      return topDir->mkdir(sub);
    };

    bookWindowQA(sq.overall,  mkDir("overall"),     specName+" overall",  massLo,massHi,nMassBins,hasLxy,hasPromptDCA);
    bookWindowQA(sq.peak,     mkDir("peak"),         specName+" peak",     massLo,massHi,nMassBins,hasLxy,hasPromptDCA);
    bookWindowQA(sq.leftSB,   mkDir("leftSideband"), specName+" left SB",  massLo,massHi,nMassBins,hasLxy,hasPromptDCA);
    bookWindowQA(sq.rightSB,  mkDir("rightSideband"),specName+" right SB", massLo,massHi,nMassBins,hasLxy,hasPromptDCA);

    bookWindowDist(sq.peakDist,    mkDir("peak_dist"),    specName+" peak dist",    hasLxy);
    bookWindowDist(sq.leftSBDist,  mkDir("leftSB_dist"),  specName+" left SB dist", hasLxy);
    bookWindowDist(sq.rightSBDist, mkDir("rightSB_dist"), specName+" right SB dist",hasLxy);

    TDirectory* cd = topDir->mkdir("constrained");
    cd->cd();
    sq.h_mass_constrained = new TH1F("h_mass_constrained",
      sp+" PV-constrained mass;m^{constr} [GeV/c^{2}];pairs",
      nMassBins,massLo,massHi);
    sq.h_mass_constrained_vs_pt = new TH2F("h_mass_constrained_vs_pt",
      sp+" constrained mass vs p_{T};p_{T} [GeV/c];m^{constr} [GeV/c^{2}]",
      60,0,12, nMassBins,massLo,massHi);
    sq.h_mass_vs_mass_constrained = new TH2F("h_mass_vs_mass_constrained",
      sp+" free mass vs constrained;m^{constr} [GeV/c^{2}];m [GeV/c^{2}]",
      nMassBins,massLo,massHi, nMassBins,massLo,massHi);
  }

  // Fill one WindowQA given a mass, topology values, and flags for optional branches
  void fillWindowQA(WindowQA& q,
                    double mass, double pairPt, double pairEta, double pairPhiV,
                    double maxDcaXY, double maxDcaZ, double absDeltaPcaZ,
                    double absPromptPairDCA, int npointsMin,
                    double dedx1, double dedx2,
                    double lxy, double dira, double dcaProd,
                    bool hasLxy, bool hasPromptDCA)
  {
    if (!std::isfinite(mass)) return;
    q.h_mass->Fill(mass);
    q.h_mass_vs_pt->Fill(pairPt, mass);
    q.h_mass_vs_eta->Fill(pairEta, mass);
    q.h_mass_vs_phi->Fill(pairPhiV, mass);
    q.h_mass_vs_dcaxy->Fill(maxDcaXY, mass);
    q.h_mass_vs_dcaz->Fill(maxDcaZ, mass);
    q.h_mass_vs_dz_pca->Fill(absDeltaPcaZ, mass);
    q.h_mass_vs_npoints->Fill(npointsMin, mass);
    q.h_mass_vs_dedx1->Fill(dedx1, mass);
    q.h_mass_vs_dedx2->Fill(dedx2, mass);
    if (hasPromptDCA && q.h_mass_vs_pairdca && std::isfinite(absPromptPairDCA))
    {
      q.h_mass_vs_pairdca->Fill(absPromptPairDCA, mass);
      if (q.h3_mass_pt_pairdca)
        q.h3_mass_pt_pairdca->Fill(pairPt, mass, absPromptPairDCA);
    }
    if (hasLxy && q.h_mass_vs_lxy && std::isfinite(lxy))
    {
      q.h_mass_vs_lxy->Fill(lxy, mass);
      if (std::isfinite(dira)) q.h_mass_vs_dira->Fill(dira, mass);
      if (std::isfinite(dcaProd)) q.h_mass_vs_dcaproduct->Fill(dcaProd, mass);
      if (q.h3_mass_pt_lxy) q.h3_mass_pt_lxy->Fill(pairPt, mass, lxy);
    }
  }

  void fillWindowDist(WindowDist& d,
                      double pairPt, double pt1, double pt2,
                      double maxDcaXY, double maxDcaZ, double absDeltaPcaZ,
                      double absPromptPairDCA,
                      double dedx1, double dedx2, double p1, double p2,
                      int npointsMin,
                      double lxy, double dira, double dcaProd,
                      bool hasLxy)
  {
    d.h_dcaxy->Fill(maxDcaXY);
    d.h_dcaz->Fill(maxDcaZ);
    d.h_dz_pca->Fill(absDeltaPcaZ);
    d.h_npoints->Fill(npointsMin);
    d.h_pt->Fill(pairPt);
    d.h_pt1_vs_pt2->Fill(pt1, pt2);
    d.h_dedx_vs_p->Fill(p1, dedx1);
    d.h_dedx_vs_p->Fill(p2, dedx2);
    if (std::isfinite(absPromptPairDCA)) d.h_pairdca->Fill(absPromptPairDCA);
    if (hasLxy && d.h_lxy && std::isfinite(lxy))
    {
      d.h_lxy->Fill(lxy);
      if (d.h_dira  && std::isfinite(dira))    d.h_dira->Fill(dira);
      if (d.h_dcaprod && std::isfinite(dcaProd)) d.h_dcaprod->Fill(dcaProd);
    }
  }

  void fillSpecies(SpeciesQA& sq,
                   double mass, double massConstr,
                   double pairPt, double pairEta, double pairPhiV,
                   double pt1, double pt2,
                   double p1, double p2,
                   double maxDcaXY, double maxDcaZ, double absDeltaPcaZ,
                   double absPromptPairDCA, int npointsMin,
                   double dedx1, double dedx2,
                   double lxy, double dira, double dcaProd,
                   bool hasLxy, bool hasPromptDCA, bool constrainedValid,
                   double peakLo, double peakHi, double sbW)
  {
    if (!std::isfinite(mass)) return;

    auto fill2D = [&](WindowQA& q) {
      fillWindowQA(q, mass, pairPt, pairEta, pairPhiV,
                   maxDcaXY, maxDcaZ, absDeltaPcaZ,
                   absPromptPairDCA, npointsMin, dedx1, dedx2,
                   lxy, dira, dcaProd, hasLxy, hasPromptDCA);
    };
    auto fillDist = [&](WindowDist& d) {
      fillWindowDist(d, pairPt, pt1, pt2,
                     maxDcaXY, maxDcaZ, absDeltaPcaZ, absPromptPairDCA,
                     dedx1, dedx2, p1, p2, npointsMin,
                     lxy, dira, dcaProd, hasLxy);
    };

    fill2D(sq.overall);

    const MassRegion region = classifyMass(mass, peakLo, peakHi, sbW);
    if (region == MassRegion::Peak)     { fill2D(sq.peak);    fillDist(sq.peakDist); }
    if (region == MassRegion::LeftSide) { fill2D(sq.leftSB);  fillDist(sq.leftSBDist); }
    if (region == MassRegion::RightSide){ fill2D(sq.rightSB); fillDist(sq.rightSBDist); }

    if (constrainedValid && std::isfinite(massConstr))
    {
      sq.h_mass_constrained->Fill(massConstr);
      sq.h_mass_constrained_vs_pt->Fill(pairPt, massConstr);
      sq.h_mass_vs_mass_constrained->Fill(massConstr, mass);
    }
  }

}  // namespace

// ==========================================================================
// Main macro
// ==========================================================================

void StudyPromptHadronTopologyQA(
  const char* inputDir    = "/sphenix/tg/tg01/hf/mitrankova/Hadrons/",
  const char* filePattern = "hadron_sitpc_HITS_pp_mvtxS300_79513_*.root",
  const char* outputDir   = "output",
  const char* outputName  = "prompt_hadron_topology_qa_mvtxS300.root",
  const char* treeName    = "pairTree",
  const Long64_t maxEntries    = -1,
  const bool usePairVertex     = true,
  const double beamX = 0.0,
  const double beamY = 0.0,
  const double beamZ = 0.0,
  const bool  includePrimaryConstrainedPrompt = true,
  const bool  ignoreCrossingZero = false,
  const char* trackClass   = "all",
  const int   minSiHits    = 4,
  // Loose pre-selection thresholds applied before filling any histogram
  const double looseMinDaughterPt = 0.15,
  const double looseMaxDcaXY      = 0.50,   // cm; negative = off
  const double looseMaxPcaZ       = 20.0,   // |pca_z| of the pair
  // exactCut1: loose (+ PID); exactCut2: Si DCA; exactCut3: Si DCA + D0 L_xy window; exactCut4: note
  const double exact2MaxDcaXY = 0.05,  // cm, Si-scale
  const double exact3MaxDcaXY = 0.05,
  const double exact3MinLxy   = 0.010, // cm, D0: 100 um (the key D0 cut; scan it with h3_mass_pt_lxy)
  const double exact3MaxLxy   = 0.070, // cm, D0: 700 um
  const double exact3MinDira  = 0.985,
  // Note-level (Table 8) parameters:
  //   D0 daughters |IPxy|<0.02 cm, pair DCA at SV<0.002 cm (20 um),
  //   Lxy 0.008-0.07 cm, DIRA>0.985; phi pair DCA<0.04 cm.
  const double exact4MaxDcaXY    = 0.020, // cm, daughter DCA to PV (note |IPxy|<0.02)
  const double exact4MinLxy      = 0.008, // cm, 80 um (note Lxy>=80 um)
  const double exact4MaxLxy      = 0.070, // cm, 700 um (note Lxy<=700 um)
  const double exact4MinDira     = 0.985, // note DIRA>0.985
  const double exact4MaxD0vPairDca = 0.002, // cm, 20 um (note DCAxy<0.002 cm)
  const double exact4PhiMaxPairDca = 0.040, // cm (note phi DCA<=0.04 cm)
  const bool   ScaleQualityBy10 = false,
  // PID (see the header) and the phi reference selection
  const int    phiPidMode = 2,
  const int    d0PidMode  = 1,
  const int    lcPidMode  = 1,
  const unsigned int phiMinNtpc   = 10,
  const double phiMaxPrimaryPcaDz = 0.30,  // cm, |z(PCA_1) - z(PCA_2)| of the primary PCAs
  const bool   processLikeSign    = true,
  const char*  likeSignTreeName   = "likeSignPairTree")
{
  TH1::AddDirectory(kTRUE);
  (void) ScaleQualityBy10;
  (void) exact3MaxDcaXY;

  const std::string trackClassName = trackClass ? trackClass : "all";
  const TString pattern = TString::Format("%s/%s", inputDir, filePattern);

  // ---- Output file -------------------------------------------------------
  gSystem->mkdir(outputDir, kTRUE);
  const TString outputPath = TString::Format("%s/%s",outputDir,outputName);
  std::unique_ptr<TFile> output(TFile::Open(outputPath,"RECREATE"));
  if (!output || output->IsZombie())
  { std::cerr << "ERROR: cannot create " << outputPath << std::endl; return; }
  output->cd();

  TNamed infoNode("config",
    TString::Format("trackClass=%s minSiHits=%d loosePt=%.2f looseDcaXY=%.3f exact2DcaXY=%.3f exact3Lxy=%.4f-%.4f "
                    "exact3Dira=%.4f phiPid=%d d0Pid=%d lcPid=%d phiNtpc=%u phiPcaDz=%.2f likeSign=%d",
                    trackClassName.c_str(), minSiHits, looseMinDaughterPt, looseMaxDcaXY,
                    exact2MaxDcaXY, exact3MinLxy, exact3MaxLxy, exact3MinDira, phiPidMode, d0PidMode, lcPidMode,
                    phiMinNtpc, phiMaxPrimaryPcaDz, processLikeSign ? 1 : 0).Data());
  infoNode.Write();

  const std::vector<std::string> tiers = {
    "exactCut1_loose", "exactCut2_siDCA",
    "exactCut3_topology", "exactCut4_note"};

  struct SpeciesSet
  {
    std::map<std::string, SpeciesQA> phi, d0, ad0, lc;
    TH1I* cutflowPhi = nullptr;
    TH1I* cutflowD0 = nullptr;
    TH1I* cutflowLc = nullptr;
  };

  // ---- Event loop over one tree (pairTree = unlike sign, likeSignPairTree = like sign) -----------
  const auto runTree = [&](const char* tname, const bool like, TDirectory* top) -> bool
  {
    TChain chain(tname);
    if (chain.Add(pattern) <= 0 || chain.GetEntries() == 0)
    {
      std::cerr << (like ? "WARNING" : "ERROR") << ": no " << tname << " in " << pattern << std::endl;
      return false;
    }
    const Long64_t nEntries = chain.GetEntries();

    // ---- Branch detection --------------------------------------------------
    const bool haveSiBranches =
      branchExists(chain,"has_si1") && branchExists(chain,"has_si2") &&
      branchExists(chain,"nmvtx1")  && branchExists(chain,"nmvtx2");
    const bool havePairVertex =
      usePairVertex &&
      branchExists(chain,"pv_x") && branchExists(chain,"pv_y") && branchExists(chain,"pv_z");
    const bool havePrimaryBranches =
      branchExists(chain,"primary_px1") && branchExists(chain,"primary_pca_valid") &&
      branchExists(chain,"prompt_pairDCA");
    const bool haveNtpc = branchExists(chain,"ntpc_clusters1") && branchExists(chain,"ntpc_clusters2");
    const bool havePcaDz = branchExists(chain,"primary_pca_dz");
    const bool haveConstrainedBranches =
      includePrimaryConstrainedPrompt &&
      branchExists(chain,"mass_Phi_primary_constrained") &&
      branchExists(chain,"mass_D0_primary_constrained") &&
      branchExists(chain,"mass_K1Pi2_primary_constrained");
    const bool haveD0vBranches =
      branchExists(chain,"d0v_status")    &&
      branchExists(chain,"d0v_lxy_signed") &&
      branchExists(chain,"d0v_dira")      &&
      branchExists(chain,"d0v_dca_product") &&
      branchExists(chain,"d0v_pair_dca");

    std::cout << tname << ": " << nEntries << " entries; branches: Si=" << haveSiBranches
              << " pairVtx=" << havePairVertex << " primary=" << havePrimaryBranches
              << " ntpc=" << haveNtpc << " pcaDz=" << havePcaDz
              << " constrained=" << haveConstrainedBranches << " d0v=" << haveD0vBranches << std::endl;

    // ---- Branch variables --------------------------------------------------
    Float_t px1=0,py1=0,pz1=0, px2=0,py2=0,pz2=0;
    Float_t charge1=0,charge2=0;
    Float_t dca_xy1=0,dca_z1=0, dca_xy2=0,dca_z2=0;
    Float_t dedx_1=0,dedx_2=0;
    Float_t pca_x=0,pca_y=0,pca_z=0, pca1_z=0,pca2_z=0;
    Float_t v0_px=0,v0_py=0,v0_pz=0, v0_pt=0;
    Float_t quality1=0,quality2=0;
    Short_t npoints1=0,npoints2=0;
    Short_t crossing1=0,crossing2=0;
    UInt_t  candidate_mask=0U;
    Float_t pv_x=0,pv_y=0,pv_z=0;
    Int_t   has_si1=0,has_si2=0;
    UInt_t  nmvtx1=0,nmvtx2=0,nintt1=0,nintt2=0;
    UInt_t  ntpc_clusters1=0, ntpc_clusters2=0;
    Float_t primary_px1=0,primary_py1=0,primary_pz1=0;
    Float_t primary_px2=0,primary_py2=0,primary_pz2=0;
    Float_t primary_pca1_z=0,primary_pca2_z=0,primary_pca_dz=0;
    Int_t   primary_pca_valid=0, prompt_pca_valid=0;
    Float_t prompt_pairDCA=0;
    Int_t   primary_constrained_valid1=0, primary_constrained_valid2=0;
    Float_t mass_Phi_primary_constrained=0;
    Float_t mass_D0_primary_constrained=0, mass_AntiD0_primary_constrained=0;
    Float_t mass_K1Pi2_primary_constrained=0, mass_Pi1K2_primary_constrained=0;
    Int_t   d0v_status=0;
    Float_t d0v_lxy_signed=0,d0v_dira=0,d0v_dca_product=0,d0v_pair_dca=0;

    chain.SetBranchAddress("px1",&px1); chain.SetBranchAddress("py1",&py1); chain.SetBranchAddress("pz1",&pz1);
    chain.SetBranchAddress("px2",&px2); chain.SetBranchAddress("py2",&py2); chain.SetBranchAddress("pz2",&pz2);
    chain.SetBranchAddress("charge1",&charge1); chain.SetBranchAddress("charge2",&charge2);
    chain.SetBranchAddress("dca_xy1",&dca_xy1); chain.SetBranchAddress("dca_z1",&dca_z1);
    chain.SetBranchAddress("dca_xy2",&dca_xy2); chain.SetBranchAddress("dca_z2",&dca_z2);
    chain.SetBranchAddress("dedx_1",&dedx_1); chain.SetBranchAddress("dedx_2",&dedx_2);
    chain.SetBranchAddress("pca_x",&pca_x); chain.SetBranchAddress("pca_y",&pca_y); chain.SetBranchAddress("pca_z",&pca_z);
    chain.SetBranchAddress("pca1_z",&pca1_z); chain.SetBranchAddress("pca2_z",&pca2_z);
    chain.SetBranchAddress("v0_px",&v0_px); chain.SetBranchAddress("v0_py",&v0_py); chain.SetBranchAddress("v0_pz",&v0_pz);
    chain.SetBranchAddress("v0_pt",&v0_pt);
    chain.SetBranchAddress("quality1",&quality1); chain.SetBranchAddress("quality2",&quality2);
    chain.SetBranchAddress("npoints1",&npoints1); chain.SetBranchAddress("npoints2",&npoints2);
    chain.SetBranchAddress("crossing1",&crossing1); chain.SetBranchAddress("crossing2",&crossing2);
    chain.SetBranchAddress("candidate_mask",&candidate_mask);
    if (havePairVertex)
    { chain.SetBranchAddress("pv_x",&pv_x); chain.SetBranchAddress("pv_y",&pv_y); chain.SetBranchAddress("pv_z",&pv_z); }
    if (haveSiBranches)
    {
      chain.SetBranchAddress("has_si1",&has_si1); chain.SetBranchAddress("has_si2",&has_si2);
      chain.SetBranchAddress("nmvtx1",&nmvtx1); chain.SetBranchAddress("nmvtx2",&nmvtx2);
      if (branchExists(chain,"nintt1")) { chain.SetBranchAddress("nintt1",&nintt1); chain.SetBranchAddress("nintt2",&nintt2); }
    }
    if (haveNtpc)
    { chain.SetBranchAddress("ntpc_clusters1",&ntpc_clusters1); chain.SetBranchAddress("ntpc_clusters2",&ntpc_clusters2); }
    if (havePcaDz) chain.SetBranchAddress("primary_pca_dz",&primary_pca_dz);
    if (havePrimaryBranches)
    {
      chain.SetBranchAddress("primary_px1",&primary_px1); chain.SetBranchAddress("primary_py1",&primary_py1);
      chain.SetBranchAddress("primary_pz1",&primary_pz1);
      chain.SetBranchAddress("primary_px2",&primary_px2); chain.SetBranchAddress("primary_py2",&primary_py2);
      chain.SetBranchAddress("primary_pz2",&primary_pz2);
      chain.SetBranchAddress("primary_pca1_z",&primary_pca1_z);
      chain.SetBranchAddress("primary_pca2_z",&primary_pca2_z);
      chain.SetBranchAddress("primary_pca_valid",&primary_pca_valid);
      chain.SetBranchAddress("prompt_pairDCA",&prompt_pairDCA);
      chain.SetBranchAddress("prompt_pca_valid",&prompt_pca_valid);
    }
    if (haveConstrainedBranches)
    {
      chain.SetBranchAddress("primary_constrained_valid1",&primary_constrained_valid1);
      chain.SetBranchAddress("primary_constrained_valid2",&primary_constrained_valid2);
      chain.SetBranchAddress("mass_Phi_primary_constrained",&mass_Phi_primary_constrained);
      chain.SetBranchAddress("mass_D0_primary_constrained",&mass_D0_primary_constrained);
      chain.SetBranchAddress("mass_AntiD0_primary_constrained",&mass_AntiD0_primary_constrained);
      chain.SetBranchAddress("mass_K1Pi2_primary_constrained",&mass_K1Pi2_primary_constrained);
      chain.SetBranchAddress("mass_Pi1K2_primary_constrained",&mass_Pi1K2_primary_constrained);
    }
    if (haveD0vBranches)
    {
      chain.SetBranchAddress("d0v_status",&d0v_status);
      chain.SetBranchAddress("d0v_lxy_signed",&d0v_lxy_signed);
      chain.SetBranchAddress("d0v_dira",&d0v_dira);
      chain.SetBranchAddress("d0v_dca_product",&d0v_dca_product);
      chain.SetBranchAddress("d0v_pair_dca",&d0v_pair_dca);
    }

    // ---- Book --------------------------------------------------------------
    SpeciesSet S;
    top->cd();
    S.cutflowPhi = new TH1I("h_cutflow_phi", "#phi: pair count by cut tier;tier;pairs",4,0,4);
    S.cutflowD0  = new TH1I("h_cutflow_d0",  "D^{0}: candidates by cut tier;tier;candidates",4,0,4);
    S.cutflowLc  = new TH1I("h_cutflow_lc",  "#Lambda_{c} pK: candidates by cut tier;tier;candidates",4,0,4);
    for (auto* hc : {S.cutflowPhi,S.cutflowD0,S.cutflowLc})
      for (int b = 0; b < 4; ++b) hc->GetXaxis()->SetBinLabel(b + 1, tiers[b].c_str());
    TDirectory* phiTop = top->mkdir("phi");
    TDirectory* d0Top  = top->mkdir("D0");
    TDirectory* ad0Top = top->mkdir("AntiD0");
    TDirectory* lcTop  = top->mkdir("LambdaC_pK");
    for (const auto& tier : tiers)
    {
      bookSpeciesQA(S.phi[tier], phiTop->mkdir(tier.c_str()), "#phi", 0.98,1.10,240, false, havePrimaryBranches);
      bookSpeciesQA(S.d0[tier],  d0Top->mkdir(tier.c_str()),  "D^{0}", 1.65,2.10,90, haveD0vBranches, havePrimaryBranches);
      bookSpeciesQA(S.ad0[tier], ad0Top->mkdir(tier.c_str()), "#bar{D}^{0}", 1.65,2.10,90, haveD0vBranches, havePrimaryBranches);
      bookSpeciesQA(S.lc[tier],  lcTop->mkdir(tier.c_str()),  "#Lambda_{c} pK", 1.40,3.20,180, false, havePrimaryBranches);
    }

    // ---- Loop ----------------------------------------------------------------
    const Long64_t nProcess = maxEntries>=0 ? std::min(nEntries,maxEntries) : nEntries;
    for (Long64_t iEntry = 0; iEntry < nProcess; ++iEntry)
    {
      chain.GetEntry(iEntry);
      if (iEntry % 200000==0)
        std::cout << tname << ": " << iEntry << " / " << nProcess << std::endl;

      if (ignoreCrossingZero && (crossing1==0||crossing2==0)) continue;

      if (trackClassName != "all" && haveSiBranches)
      {
        const int nsi1=static_cast<int>(nmvtx1+nintt1);
        const int nsi2=static_cast<int>(nmvtx2+nintt2);
        const bool si1=has_si1!=0&&nsi1>=minSiHits;
        const bool si2=has_si2!=0&&nsi2>=minSiHits;
        const bool tpc1=has_si1==0,tpc2=has_si2==0;
        const bool keep=
          (trackClassName=="si"      && si1 && si2) ||
          (trackClassName=="tpconly" && tpc1 && tpc2) ||
          (trackClassName=="mixed"   && ((si1&&tpc2)||(tpc1&&si2)));
        if (!keep) continue;
      }

      const bool unlikeSign = charge1*charge2 < 0.0;
      if (like == unlikeSign) continue;  // pairTree: unlike sign only, likeSignPairTree: like sign only

      // Kinematics
      const double pt1=std::hypot(px1,py1), pt2=std::hypot(px2,py2);
      const double p1=std::sqrt(px1*px1+py1*py1+pz1*pz1);
      const double p2=std::sqrt(px2*px2+py2*py2+pz2*pz2);
      const double maxDcaXY=std::max(std::abs(dca_xy1),std::abs(dca_xy2));
      const double maxDcaZ =std::max(std::abs(dca_z1), std::abs(dca_z2));
      const double absDeltaPcaZ=std::abs(pca1_z-pca2_z);
      const double pairPt =std::hypot(v0_px,v0_py);
      const double pairEta=(pairPt>0)?std::asinh(v0_pz/pairPt):0.0;
      const double pairPhiV=std::atan2(v0_py,v0_px);
      const int    npointsMin=std::min((int)npoints1,(int)npoints2);

      if (pt1<looseMinDaughterPt || pt2<looseMinDaughterPt) continue;
      if (looseMaxDcaXY>0 && maxDcaXY>looseMaxDcaXY) continue;
      if (std::abs(pca_z) > looseMaxPcaZ) continue;

      // Primary kinematics (momenta at the transverse PCA to the collision vertex)
      const bool primOk =
        havePrimaryBranches && primary_pca_valid!=0 &&
        std::isfinite(primary_px1) && std::isfinite(primary_px2);
      const double ppx1=primOk?primary_px1:px1, ppy1=primOk?primary_py1:py1, ppz1=primOk?primary_pz1:pz1;
      const double ppx2=primOk?primary_px2:px2, ppy2=primOk?primary_py2:py2, ppz2=primOk?primary_pz2:pz2;
      const double pp1 = std::sqrt(ppx1*ppx1+ppy1*ppy1+ppz1*ppz1);
      const double pp2 = std::sqrt(ppx2*ppx2+ppy2*ppy2+ppz2*ppz2);
      const double absPromptPairDCA =
        (primOk && prompt_pca_valid!=0 && std::isfinite(prompt_pairDCA))
          ? std::abs(prompt_pairDCA) : kNaN();
      const double absPrimaryPcaDz =
        (havePcaDz && std::isfinite(primary_pca_dz)) ? std::abs(primary_pca_dz) : std::abs(primary_pca1_z - primary_pca2_z);
      const unsigned int ntpcMin = haveNtpc ? std::min(ntpc_clusters1, ntpc_clusters2) : 999U;

      const bool constrOk =
        haveConstrainedBranches &&
        primary_constrained_valid1!=0 && primary_constrained_valid2!=0;

      // D0 vertex
      const bool d0vOk = haveD0vBranches && d0v_status==1;
      const double lxy   = d0vOk && std::isfinite(d0v_lxy_signed)   ? d0v_lxy_signed   : kNaN();
      const double dira  = d0vOk && std::isfinite(d0v_dira)         ? d0v_dira          : kNaN();
      const double dcaProd = d0vOk && std::isfinite(d0v_dca_product)? d0v_dca_product   : kNaN();

      // ---- phi: momenta at the primary PCA (free mass), PV-constrained mass under constrained/
      const double massPhi = invMass(ppx1,ppy1,ppz1,kKaonMass, ppx2,ppy2,ppz2,kKaonMass);
      const double massPhiC = (constrOk && std::isfinite(mass_Phi_primary_constrained)) ? mass_Phi_primary_constrained : kNaN();

      // ---- D0: "D0" slot = negative track is the kaon (unlike) / track 1 is the kaon (like, K1Pi2);
      //      "AntiD0" slot = the other assignment
      const bool track1KaonInD0 = unlikeSign ? (charge1 < 0) : true;
      const double massD0 = track1KaonInD0
        ? invMass(ppx1,ppy1,ppz1,kKaonMass, ppx2,ppy2,ppz2,kPionMass)
        : invMass(ppx2,ppy2,ppz2,kKaonMass, ppx1,ppy1,ppz1,kPionMass);
      const double massAD0 = track1KaonInD0
        ? invMass(ppx2,ppy2,ppz2,kKaonMass, ppx1,ppy1,ppz1,kPionMass)
        : invMass(ppx1,ppy1,ppz1,kKaonMass, ppx2,ppy2,ppz2,kPionMass);
      double massD0c = kNaN(), massAD0c = kNaN();
      if (constrOk)
      {
        massD0c  = unlikeSign ? mass_D0_primary_constrained : mass_K1Pi2_primary_constrained;
        massAD0c = unlikeSign ? mass_AntiD0_primary_constrained : mass_Pi1K2_primary_constrained;
      }
      const bool pidD0 = d0PidPass(d0PidMode, track1KaonInD0 ? dedx_1 : dedx_2, track1KaonInD0 ? pp1 : pp2,
                                              track1KaonInD0 ? dedx_2 : dedx_1, track1KaonInD0 ? pp2 : pp1);
      const bool pidAD0 = d0PidPass(d0PidMode, track1KaonInD0 ? dedx_2 : dedx_1, track1KaonInD0 ? pp2 : pp1,
                                               track1KaonInD0 ? dedx_1 : dedx_2, track1KaonInD0 ? pp1 : pp2);

      // ---- Lambda_c pK sub-system (both assignments), PID per assignment
      const double pKmass_p1K2 = invMass(ppx1,ppy1,ppz1,kProtonMass, ppx2,ppy2,ppz2,kKaonMass);
      const double pKmass_p2K1 = invMass(ppx1,ppy1,ppz1,kKaonMass,   ppx2,ppy2,ppz2,kProtonMass);
      const bool pidP1K2 = lcPidPass(lcPidMode, dedx_1, pp1, dedx_2, pp2);
      const bool pidP2K1 = lcPidPass(lcPidMode, dedx_2, pp2, dedx_1, pp1);

      // ---- Cut tiers -------------------------------------------------------
      const bool tier1 = true;
      const bool tier2 = (exact2MaxDcaXY<=0 || maxDcaXY <= exact2MaxDcaXY);

      // phi reference (all tiers): ntpc, |dz of the primary PCAs|, PID
      const bool phiRef = ntpcMin >= phiMinNtpc &&
        (phiMaxPrimaryPcaDz <= 0 || absPrimaryPcaDz < phiMaxPrimaryPcaDz) &&
        phiPidPass(phiPidMode, dedx_1, pp1, dedx_2, pp2);
      const bool tier3_phi = tier2 &&
        (!havePrimaryBranches || !std::isfinite(absPromptPairDCA) || absPromptPairDCA < 0.05);
      const bool tier4_phi = (exact4MaxDcaXY<=0 || maxDcaXY <= exact4MaxDcaXY) &&
        (!havePrimaryBranches || !std::isfinite(absPromptPairDCA) || absPromptPairDCA < exact4PhiMaxPairDca);

      // D0 L_xy window (the key D0 cut) + DIRA + opposite-sign impact parameters
      const bool tier3_d0 = tier2 &&
        (!haveD0vBranches ||
         (d0vOk &&
          std::isfinite(lxy) && lxy >= exact3MinLxy && lxy <= exact3MaxLxy &&
          std::isfinite(dira) && dira >= exact3MinDira &&
          std::isfinite(dcaProd) && dcaProd < 0.0));
      const bool tier4_d0 = (exact4MaxDcaXY<=0 || maxDcaXY <= exact4MaxDcaXY) &&
        (!haveD0vBranches ||
         (d0vOk &&
          std::isfinite(lxy)    && lxy    >= exact4MinLxy && lxy <= exact4MaxLxy &&
          std::isfinite(dira)   && dira   >= exact4MinDira &&
          std::isfinite(dcaProd)&& dcaProd < 0.0 &&
          std::isfinite(d0v_pair_dca) && d0v_pair_dca < exact4MaxD0vPairDca));

      // ---- Fill phi -------------------------------------------------------
      if ((candidate_mask & kMaskPhi) && std::isfinite(massPhi) && phiRef)
      {
        const bool pass[4] = {tier1, tier2, tier3_phi, tier4_phi};
        for (int t = 0; t < 4; ++t)
        {
          if (!pass[t]) continue;
          S.cutflowPhi->Fill(t + 0.5);
          fillSpecies(S.phi[tiers[t]],
                      massPhi, massPhiC, pairPt, pairEta, pairPhiV,
                      pt1,pt2,pp1,pp2, maxDcaXY,maxDcaZ,absDeltaPcaZ,
                      absPromptPairDCA, npointsMin, dedx_1,dedx_2,
                      kNaN(),kNaN(),kNaN(),
                      false, havePrimaryBranches, constrOk,
                      kPhiPeakLo,kPhiPeakHi,kPhiSBW);
        }
      }

      // ---- Fill D0 (mask bit 16 = "D0" slot, 32 = "AntiD0" slot, also for like sign) ----
      {
        const bool hasD0  = (candidate_mask & kMaskD0)!=0 && pidD0;
        const bool hasAD0 = (candidate_mask & kMaskAntiD0)!=0 && pidAD0;
        const bool pass[4] = {tier1, tier2, tier3_d0, tier4_d0};
        for (int t = 0; t < 4; ++t)
        {
          if (!pass[t]) continue;
          if (hasD0 && std::isfinite(massD0))
          {
            S.cutflowD0->Fill(t + 0.5);
            fillSpecies(S.d0[tiers[t]], massD0, massD0c, pairPt, pairEta, pairPhiV,
                        pt1,pt2,pp1,pp2, maxDcaXY,maxDcaZ,absDeltaPcaZ,
                        absPromptPairDCA, npointsMin, dedx_1,dedx_2,
                        lxy,dira,dcaProd, haveD0vBranches, havePrimaryBranches, constrOk,
                        kD0PeakLo,kD0PeakHi,kD0SBW);
          }
          if (hasAD0 && std::isfinite(massAD0))
          {
            S.cutflowD0->Fill(t + 0.5);
            fillSpecies(S.ad0[tiers[t]], massAD0, massAD0c, pairPt, pairEta, pairPhiV,
                        pt1,pt2,pp1,pp2, maxDcaXY,maxDcaZ,absDeltaPcaZ,
                        absPromptPairDCA, npointsMin, dedx_1,dedx_2,
                        lxy,dira,dcaProd, haveD0vBranches, havePrimaryBranches, constrOk,
                        kD0PeakLo,kD0PeakHi,kD0SBW);
          }
        }
      }

      // ---- Fill Lambda_c pK sub-system (two-body proxy, see MakePromptHadronHistograms.C) -----
      const double kLcDalitzLo = 1.43, kLcDalitzHi = 2.15;
      const bool inLcDalitz_p1K2 = pidP1K2 && (pKmass_p1K2 >= kLcDalitzLo && pKmass_p1K2 < kLcDalitzHi);
      const bool inLcDalitz_p2K1 = pidP2K1 && (pKmass_p2K1 >= kLcDalitzLo && pKmass_p2K1 < kLcDalitzHi);
      if (pairPt > 1.0 && (inLcDalitz_p1K2 || inLcDalitz_p2K1))
      {
        const bool tier3_lc_dca = tier2 &&
          (!havePrimaryBranches || !std::isfinite(absPromptPairDCA) || absPromptPairDCA < 0.04);
        const bool tier4_lc_dca = (exact4MaxDcaXY<=0 || maxDcaXY <= exact4MaxDcaXY) &&
          (!havePrimaryBranches || !std::isfinite(absPromptPairDCA) || absPromptPairDCA < 0.02);
        const bool pass[4] = {tier1, tier2, tier3_lc_dca, tier4_lc_dca};
        for (int t = 0; t < 4; ++t)
        {
          if (!pass[t]) continue;
          for (int a = 0; a < 2; ++a)
          {
            if (!(a == 0 ? inLcDalitz_p1K2 : inLcDalitz_p2K1)) continue;
            S.cutflowLc->Fill(t + 0.5);
            fillSpecies(S.lc[tiers[t]], a == 0 ? pKmass_p1K2 : pKmass_p2K1, kNaN(), pairPt, pairEta, pairPhiV,
                        pt1,pt2,pp1,pp2, maxDcaXY,maxDcaZ,absDeltaPcaZ,
                        absPromptPairDCA, npointsMin, dedx_1,dedx_2,
                        kNaN(),kNaN(),kNaN(), false, havePrimaryBranches, false,
                        kLcPeakLo,kLcPeakHi,kLcSBW);
          }
        }
      }
    }  // end event loop
    return true;
  };

  // unlike sign at the top level (as before), like sign under likeSign/
  if (!runTree(treeName, false, output.get()))
  {
    return;
  }
  if (processLikeSign)
  {
    runTree(likeSignTreeName, true, output->mkdir("likeSign"));
  }

  output->Write();
  output->Close();
  std::cout << "Wrote: " << outputPath << std::endl;
}