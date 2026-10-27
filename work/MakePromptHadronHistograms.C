// MakePromptHadronHistograms.C
//
// One-pass QA for phi -> K+K-, D0/anti-D0 -> K pi, and Lambda_c -> p K pi
// from the TpcV0CandidateTree pairTree / likeSignPairTree.
//
// Species focus (unlike the K0S/Lambda macro this one omits K0S entirely):
//   phi      : prompt, |dca_xy| < few mm, PV-constrained mass
//   D0/aD0   : semi-displaced, d0v_* topology branches used when present
//   Lambda_c : prompt charm baryon, p K pi (no dedicated branch; recalculated
//              from daughter momenta under three mass hypotheses)
//
// Cut levels are organised in three separate stacks so each species gets
// optimised selections without polluting the others:
//   phiSelections       : prompt track DCA_xy / DCA_z ladder
//   d0Selections        : combinatorial then topology (d0v_*) ladder
//   lambdacSelections   : prompt + invariant-mass window ladder
//
// Like-sign background (phi, D0) is read from likeSignPairTree.
//
// dE/dx PID (pidMode of each selection): momentum-dependent bands, <dE/dx>(beta gamma) fitted to the
// pion (K0S) and proton (Lambda) bands, f = A/beta^K + C ln(beta gamma) + D (A 232.9, K 2.138,
// C 64.41, D -133.2), resolution 19 %; n_sigma(h) = (dE/dx - f(p/m_h)) / (0.19 f).
//   K-compatible  |n_K| < 2              K identified  |n_K| < 2, n_pi > 3, n_p < -2  (p < ~0.6 GeV)
//   pi-compatible |n_pi| < 3             p identified  |n_p| < 2, n_K > 2            (p < ~1.3 GeV)
// Check the bands with h_dedx_vs_p_* (and StudyPhiD0.C, page "kaon band").
//
// Output layout: <species>/<selection>/unlike and <species>/<selection>/like.  (The previous version
// booked both in <species>/<selection>: TDirectory::mkdir("a/b") returns "a", so the unlike and like
// histograms had the same names in one directory and only the like-sign ones were written.)
//
// Branch dependencies:
//   core       : always required (same as MakeK0sPairHistograms)
//   primary    : primary_px1/2, prompt_pairDCA, ...  (prompt momenta)
//   constrained: mass_Phi_primary_constrained, mass_D0_primary_constrained, ...
//   d0v        : d0v_status, d0v_lxy_signed, d0v_dira, d0v_dca_product,
//                d0v_pair_dca, d0v_x/y/z, d0v_ntracks (optional: used when found)
//
// Usage:
//   root -l -b -q \
//   'MakePromptHadronHistograms.C("/path","hadron_sitpc_*.root","out","prompt_hadrons.root")'
//
// Si+TPC setup notes (same conventions as MakeK0sPairHistograms):
//   trackClass "all"/"si"/"tpconly"/"mixed", minSiHits 4
//   usePairVertex=true  uses pv_x/y/z per pair
//   ignoreCrossingZero  skips crossing-zero ambiguous events

#include <TBranch.h>
#include <TChain.h>
#include <TChainElement.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH1I.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TMath.h>
#include <TNamed.h>
#include <TObjArray.h>
#include <TString.h>
#include <TSystem.h>
#include <TTree.h>

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
  constexpr double kPionMass     = 0.13957039;
  constexpr double kKaonMass     = 0.493677;
  constexpr double kProtonMass   = 0.938272088;
  constexpr double kElectronMass = 0.00051099895;
  constexpr double kTwoPi        = 2.0 * TMath::Pi();

  // Candidate mask bits (must match TpcV0CandidateTree::CandidateMask)
  constexpr UInt_t kMaskKShort    = 1U << 0;
  constexpr UInt_t kMaskLambda    = 1U << 1;
  constexpr UInt_t kMaskAntiLambda= 1U << 2;
  constexpr UInt_t kMaskPhi       = 1U << 3;
  constexpr UInt_t kMaskD0        = 1U << 4;
  constexpr UInt_t kMaskAntiD0    = 1U << 5;

  // Mass windows for peak / sideband classification
  constexpr double kPhiPeakLo   = 1.010, kPhiPeakHi   = 1.030;
  constexpr double kD0PeakLo    = 1.840, kD0PeakHi    = 1.890;
  constexpr double kLcPeakLo    = 2.270, kLcPeakHi    = 2.310;

  double kNaN() { return std::numeric_limits<double>::quiet_NaN(); }

  double wrapPhi(double phi)
  {
    while (phi >= TMath::Pi())  phi -= kTwoPi;
    while (phi < -TMath::Pi()) phi += kTwoPi;
    return phi;
  }

  bool branchExists(TChain& c, const char* name)
  { return c.GetBranch(name) != nullptr; }

  bool passesDedx(double v, double lo, double hi)
  {
    return (lo < 0.0 || v > lo) && (hi < 0.0 || v < hi);
  }

  // momentum-dependent dE/dx bands (see the header)
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
  bool kaonCompatible(double dedx, double p) { return std::fabs(kDedx.nsigma(dedx, p, 0.493677)) < 2.0; }
  bool kaonIdentified(double dedx, double p)
  {
    return kaonCompatible(dedx, p) && kDedx.nsigma(dedx, p, 0.13957039) > 3.0 && kDedx.nsigma(dedx, p, 0.938272088) < -2.0;
  }
  bool pionCompatible(double dedx, double p) { return std::fabs(kDedx.nsigma(dedx, p, 0.13957039)) < 3.0; }
  bool protonIdentified(double dedx, double p)
  {
    return std::fabs(kDedx.nsigma(dedx, p, 0.938272088)) < 2.0 && kDedx.nsigma(dedx, p, 0.493677) > 2.0;
  }
  // phi pidMode: 0 none, 1 both K-compatible, 2 + >= 1 identified, 3 both identified
  bool phiPidPass(int mode, double dedx1, double p1, double dedx2, double p2)
  {
    if (mode <= 0) return true;
    if (!kaonCompatible(dedx1, p1) || !kaonCompatible(dedx2, p2)) return false;
    const int nid = (kaonIdentified(dedx1, p1) ? 1 : 0) + (kaonIdentified(dedx2, p2) ? 1 : 0);
    return mode == 1 || (mode == 2 && nid >= 1) || (mode >= 3 && nid == 2);
  }
  // D0 pidMode for one K pi assignment: 0 none, 1 K-compatible + pi-compatible, 2 K identified + pi-compatible
  bool d0PidPass(int mode, double dedxK, double pK, double dedxPi, double pPi)
  {
    if (mode <= 0) return true;
    if (!pionCompatible(dedxPi, pPi)) return false;
    return mode == 1 ? kaonCompatible(dedxK, pK) : kaonIdentified(dedxK, pK);
  }
  // Lambda_c pidMode for one p K assignment: 0 none, 1 p identified + K-compatible, 2 p and K identified
  bool lcPidPass(int mode, double dedxP, double pP, double dedxK, double pK)
  {
    if (mode <= 0) return true;
    if (!protonIdentified(dedxP, pP)) return false;
    return mode == 1 ? kaonCompatible(dedxK, pK) : kaonIdentified(dedxK, pK);
  }

  double invMass(double px1, double py1, double pz1, double m1,
                 double px2, double py2, double pz2, double m2)
  {
    const double e1 = std::sqrt(px1*px1+py1*py1+pz1*pz1 + m1*m1);
    const double e2 = std::sqrt(px2*px2+py2*py2+pz2*pz2 + m2*m2);
    const double px = px1+px2, py = py1+py2, pz = pz1+pz2;
    return std::sqrt(std::max(0.0, (e1+e2)*(e1+e2)-px*px-py*py-pz*pz));
  }

  // Lambda_c: three-body from the pair daughters plus a third (missing) pion
  // For a pair tree this is approximate: we sum the two-body system and label
  // hypotheses by which daughter is which.
  // Full Lc->pKpi needs a triplet finder; here we assess the pK sub-system mass
  // and flag pairs whose pK mass is in the Lc Dalitz band (useful for efficiency
  // studies before the triplet reco is available).
  double lambdaCSubMass(double px1, double py1, double pz1,
                        double px2, double py2, double pz2,
                        bool track1IsProton)
  {
    // p K sub-system (the pi is the other leg of the full decay, not yet available)
    if (track1IsProton)
      return invMass(px1,py1,pz1,kProtonMass, px2,py2,pz2,kKaonMass);
    else
      return invMass(px1,py1,pz1,kKaonMass,  px2,py2,pz2,kProtonMass);
  }

  // -----------------------------------------------------------------------
  // Selection structs
  // -----------------------------------------------------------------------

  struct PhiSelection
  {
    std::string name;
    std::string description;
    double minDaughterPt;
    double maxAbsDcaXY;     // per-track |DCA_xy| to the PV
    double maxAbsDcaZ;      // per-track |DCA_z|
    double maxAbsDeltaPcaZ; // |pca1_z - pca2_z| (prompt: should be near 0)
    double maxAbsPromptPairDCA;
    double maxAbsPrimaryPcaZ;
    double kaonDedxMin;     // negative = off
    double kaonDedxMax;
    int pidMode{0};                  // phiPidPass: 0 none, 1 both K-compatible, 2 + >= 1 identified, 3 both identified
    unsigned int minNtpcClusters{0};  // both daughters
    double maxAbsPrimaryPcaDz{-1.0};  // |z(PCA_1) - z(PCA_2)| of the primary PCAs (primary_pca_dz) [cm]
  };

  struct D0Selection
  {
    std::string name;
    std::string description;
    double minDaughterPt;
    double maxAbsDcaXY;
    double maxAbsDcaZ;
    double maxAbsPromptPairDCA;
    // D0-vertex topology (d0v_* branches); -1 = off
    double minLxySigned;    // d0v_lxy_signed lower bound [cm]
    double maxLxySigned;    // d0v_lxy_signed upper bound [cm]; -1 = off
    double minDira;         // d0v_dira
    double maxDcaProduct;   // d0v_dca_product (negative = require < 0 i.e. OS DCA)
    double maxD0vPairDca;   // d0v_pair_dca [cm]
    double kaonDedxMin;
    double kaonDedxMax;
    double pionDedxMin;
    double pionDedxMax;
    int pidMode{0};              // d0PidPass per K pi assignment: 0 none, 1 K/pi-compatible, 2 K identified
    double minAbsD0vDcaXY{-1.0};  // both daughters |d0v_dca_xy| (to the D0 vertex without them) [cm]
  };

  struct LcSelection
  {
    std::string name;
    std::string description;
    double minDaughterPt;
    double maxAbsDcaXY;
    double maxAbsDcaZ;
    double maxAbsPromptPairDCA;
    double pKSubMassMin;   // p-K sub-system mass window for Dalitz band
    double pKSubMassMax;
    double protonDedxMin;
    double protonDedxMax;
    double kaonDedxMin;
    double kaonDedxMax;
    int pidMode{0};  // lcPidPass per p K assignment: 0 none, 1 p identified + K-compatible, 2 p and K identified
  };

  // -----------------------------------------------------------------------
  // Histogram set (one per species per selection level per charge)
  // -----------------------------------------------------------------------

  struct PhiHistSet
  {
    TH1F* h_mass          = nullptr;
    TH2F* h_mass_vs_pt    = nullptr;
    TH2F* h_mass_vs_eta   = nullptr;
    TH2F* h_mass_vs_phi   = nullptr;
    TH2F* h_dca_xy_vs_pt  = nullptr;
    TH2F* h_dca_z_vs_pt   = nullptr;
    TH2F* h_dedx_vs_p_k   = nullptr;  // both daughters under kaon hypothesis
    TH2F* h_pt1_vs_pt2    = nullptr;
    TH2F* h_pair_dca_vs_mass = nullptr;
    TH2F* h_delta_pca_z_vs_mass = nullptr;
    TH1F* h_mass_constrained = nullptr;   // PV-constrained mass
    TH2F* h_mass_constrained_vs_pt = nullptr;
    TH3F* h3_mass_vs_pt_vs_pairdca = nullptr;
  };

  struct D0HistSet
  {
    TH1F* h_mass_d0       = nullptr;
    TH1F* h_mass_ad0      = nullptr;
    TH2F* h_mass_d0_vs_pt = nullptr;
    TH2F* h_mass_ad0_vs_pt= nullptr;
    TH1F* h_mass_d0_constrained  = nullptr;
    TH1F* h_mass_ad0_constrained = nullptr;
    TH2F* h_mass_d0_constrained_vs_pt  = nullptr;
    TH2F* h_mass_ad0_constrained_vs_pt = nullptr;
    // D0-vertex topology
    TH1F* h_lxy_signed    = nullptr;
    TH1F* h_dira          = nullptr;
    TH1F* h_dca_product   = nullptr;
    TH1F* h_d0v_pair_dca  = nullptr;
    TH2F* h_lxy_vs_mass   = nullptr;
    TH2F* h_dira_vs_mass  = nullptr;
    TH2F* h_dca_xy1_vs_pt = nullptr;
    TH2F* h_dca_xy2_vs_pt = nullptr;
    TH2F* h_dedx_vs_p_k   = nullptr;
    TH2F* h_dedx_vs_p_pi  = nullptr;
    TH2F* h_pt1_vs_pt2    = nullptr;
    TH3F* h3_mass_d0_vs_pt_vs_lxy  = nullptr;
    TH3F* h3_mass_ad0_vs_pt_vs_lxy = nullptr;
  };

  struct LcHistSet
  {
    // Two-body (pK sub-system) mass used as a proxy / pre-selection
    TH1F* h_pK_mass_p1K2   = nullptr;  // track1=proton, track2=kaon
    TH1F* h_pK_mass_p2K1   = nullptr;  // track2=proton, track1=kaon
    TH2F* h_pK_mass_vs_pt  = nullptr;
    TH2F* h_dca_xy_vs_pt   = nullptr;
    TH2F* h_dca_z_vs_pt    = nullptr;
    TH2F* h_dedx_vs_p_p    = nullptr;  // proton hypothesis daughter
    TH2F* h_dedx_vs_p_k    = nullptr;  // kaon hypothesis daughter
    TH2F* h_pt1_vs_pt2     = nullptr;
    TH2F* h_pair_dca_vs_mass = nullptr;
  };

  // -----------------------------------------------------------------------
  // Booking helpers
  // -----------------------------------------------------------------------

  PhiHistSet bookPhi(TDirectory* dir, const std::string& tag)
  {
    dir->cd();
    const TString t = (" [" + tag + "]").c_str();
    PhiHistSet h;
    h.h_mass = new TH1F("h_mass_Phi",
      "#phi #rightarrow K^{+}K^{-} mass"+t+" (momenta at the primary PCA);m_{KK} [GeV/c^{2}];pairs",
      300, 0.98, 1.10);
    h.h_mass_vs_pt = new TH2F("h_mass_Phi_vs_pt",
      "#phi mass vs pair p_{T}"+t+";p_{T}^{pair} [GeV/c];m_{KK} [GeV/c^{2}]",
      60,0,12, 300,0.98,1.10);
    h.h_mass_vs_eta = new TH2F("h_mass_Phi_vs_eta",
      "#phi mass vs pair #eta"+t+";#eta^{pair};m_{KK} [GeV/c^{2}]",
      60,-3,3, 300,0.98,1.10);
    h.h_mass_vs_phi = new TH2F("h_mass_Phi_vs_phi",
      "#phi mass vs pair #phi"+t+";#phi^{pair} [rad];m_{KK} [GeV/c^{2}]",
      64,-TMath::Pi(),TMath::Pi(), 300,0.98,1.10);
    h.h_dca_xy_vs_pt = new TH2F("h_dca_xy_vs_pt",
      "#phi daughters: max|DCA_{xy}| vs p_{T}"+t+";p_{T} [GeV/c];max|DCA_{xy}| [cm]",
      40,0,4, 400,0,0.5);
    h.h_dca_z_vs_pt = new TH2F("h_dca_z_vs_pt",
      "#phi daughters: max|DCA_{z}| vs p_{T}"+t+";p_{T} [GeV/c];max|DCA_{z}| [cm]",
      40,0,4, 400,0,1.0);
    h.h_dedx_vs_p_k = new TH2F("h_dedx_vs_p_kaon",
      "#phi daughters: dE/dx vs p"+t+";p [GeV/c];dE/dx [arb.]",
      60,0,6, 200,0,2000);
    h.h_pt1_vs_pt2 = new TH2F("h_pt1_vs_pt2",
      "#phi daughters: p_{T,1} vs p_{T,2}"+t+";p_{T,1} [GeV/c];p_{T,2} [GeV/c]",
      60,0,6, 60,0,6);
    h.h_pair_dca_vs_mass = new TH2F("h_pair_dca_vs_mass",
      "#phi: |prompt pair DCA| vs m_{KK}"+t+";m_{KK} [GeV/c^{2}];|prompt pair DCA| [cm]",
      300,0.98,1.10, 200,0,0.3);
    h.h_delta_pca_z_vs_mass = new TH2F("h_delta_pca_z_vs_mass",
      "#phi: |#Deltaz_{PCA}| vs m_{KK}"+t+";m_{KK} [GeV/c^{2}];|#Delta z_{PCA}| [cm]",
      300,0.98,1.10, 200,0,2.0);
    h.h_mass_constrained = new TH1F("h_mass_Phi_constrained",
      "#phi PV-constrained mass"+t+";m_{KK}^{constr} [GeV/c^{2}];pairs",
      300,0.98,1.10);
    h.h_mass_constrained_vs_pt = new TH2F("h_mass_Phi_constrained_vs_pt",
      "#phi PV-constrained mass vs p_{T}"+t+";p_{T}^{pair} [GeV/c];m_{KK}^{constr} [GeV/c^{2}]",
      60,0,12, 300,0.98,1.10);
    h.h3_mass_vs_pt_vs_pairdca = new TH3F("h3_Phi_mass_pt_pairdca",
      "#phi: m_{KK} vs p_{T} vs |pair DCA|"+t+";p_{T} [GeV/c];m_{KK} [GeV/c^{2}];|pair DCA| [cm]",
      30,0,6, 60,0.98,1.10, 20,0,0.3);
    return h;
  }

  D0HistSet bookD0(TDirectory* dir, const std::string& tag)
  {
    dir->cd();
    const TString t = (" [" + tag + "]").c_str();
    D0HistSet h;
    h.h_mass_d0 = new TH1F("h_mass_D0",
      "D^{0} #rightarrow K^{-}#pi^{+} mass"+t+";m_{K#pi} [GeV/c^{2}];pairs",
      350,1.65,2.10);
    h.h_mass_ad0 = new TH1F("h_mass_AntiD0",
      "#bar{D}^{0} #rightarrow K^{+}#pi^{-} mass"+t+";m_{K#pi} [GeV/c^{2}];pairs",
      350,1.65,2.10);
    h.h_mass_d0_vs_pt = new TH2F("h_mass_D0_vs_pt",
      "D^{0} mass vs pair p_{T}"+t+";p_{T}^{pair} [GeV/c];m_{K#pi} [GeV/c^{2}]",
      60,0,12, 350,1.65,2.10);
    h.h_mass_ad0_vs_pt = new TH2F("h_mass_AntiD0_vs_pt",
      "#bar{D}^{0} mass vs pair p_{T}"+t+";p_{T}^{pair} [GeV/c];m_{K#pi} [GeV/c^{2}]",
      60,0,12, 350,1.65,2.10);
    h.h_mass_d0_constrained = new TH1F("h_mass_D0_constrained",
      "D^{0} PV-constrained mass"+t+";m_{K#pi}^{constr} [GeV/c^{2}];pairs",
      350,1.65,2.10);
    h.h_mass_ad0_constrained = new TH1F("h_mass_AntiD0_constrained",
      "#bar{D}^{0} PV-constrained mass"+t+";m_{K#pi}^{constr} [GeV/c^{2}];pairs",
      350,1.65,2.10);
    h.h_mass_d0_constrained_vs_pt = new TH2F("h_mass_D0_constrained_vs_pt",
      "D^{0} PV-constrained mass vs p_{T}"+t+";p_{T}^{pair} [GeV/c];m_{K#pi}^{constr} [GeV/c^{2}]",
      60,0,12, 350,1.65,2.10);
    h.h_mass_ad0_constrained_vs_pt = new TH2F("h_mass_AntiD0_constrained_vs_pt",
      "#bar{D}^{0} PV-constrained mass vs p_{T}"+t+";p_{T}^{pair} [GeV/c];m_{K#pi}^{constr} [GeV/c^{2}]",
      60,0,12, 350,1.65,2.10);
    h.h_lxy_signed = new TH1F("h_d0v_lxy_signed",
      "D^{0} vertex: signed L_{xy}"+t+";L_{xy}^{signed} [cm];pairs",
      400,-0.5,1.5);
    h.h_dira = new TH1F("h_d0v_dira",
      "D^{0} vertex: DIRA"+t+";DIRA;pairs",
      200,0.9,1.0);
    h.h_dca_product = new TH1F("h_d0v_dca_product",
      "D^{0} vertex: DCA product (d_{1}#timesd_{2})"+t+";DCA_{1}#timesDCA_{2} [cm^{2}];pairs",
      400,-0.05,0.05);
    h.h_d0v_pair_dca = new TH1F("h_d0v_pair_dca",
      "D^{0} vertex: pair DCA"+t+";pair DCA [cm];pairs",
      200,0,0.5);
    h.h_lxy_vs_mass = new TH2F("h_d0v_lxy_vs_mass",
      "D^{0} vertex: L_{xy}^{signed} vs m_{K#pi}"+t+";m_{K#pi} [GeV/c^{2}];L_{xy}^{signed} [cm]",
      350,1.65,2.10, 200,-0.5,1.5);
    h.h_dira_vs_mass = new TH2F("h_d0v_dira_vs_mass",
      "D^{0} vertex: DIRA vs m_{K#pi}"+t+";m_{K#pi} [GeV/c^{2}];DIRA",
      350,1.65,2.10, 100,0.9,1.0);
    h.h_dca_xy1_vs_pt = new TH2F("h_dca_xy1_vs_pt",
      "D^{0}: daughter 1 DCA_{xy} to D^{0} vtx vs p_{T}"+t+";p_{T} [GeV/c];DCA_{xy} [cm]",
      40,0,4, 400,-0.5,0.5);
    h.h_dca_xy2_vs_pt = new TH2F("h_dca_xy2_vs_pt",
      "D^{0}: daughter 2 DCA_{xy} to D^{0} vtx vs p_{T}"+t+";p_{T} [GeV/c];DCA_{xy} [cm]",
      40,0,4, 400,-0.5,0.5);
    h.h_dedx_vs_p_k = new TH2F("h_dedx_vs_p_kaon",
      "D^{0} kaon daughter: dE/dx vs p"+t+";p [GeV/c];dE/dx [arb.]",
      60,0,6, 200,0,2000);
    h.h_dedx_vs_p_pi = new TH2F("h_dedx_vs_p_pion",
      "D^{0} pion daughter: dE/dx vs p"+t+";p [GeV/c];dE/dx [arb.]",
      60,0,6, 200,0,2000);
    h.h_pt1_vs_pt2 = new TH2F("h_pt1_vs_pt2",
      "D^{0} daughters: p_{T,1} vs p_{T,2}"+t+";p_{T,1} [GeV/c];p_{T,2} [GeV/c]",
      60,0,6, 60,0,6);
    h.h3_mass_d0_vs_pt_vs_lxy = new TH3F("h3_D0_mass_pt_lxy",
      "D^{0}: m_{K#pi} vs p_{T} vs L_{xy}^{signed}"+t+";p_{T} [GeV/c];m_{K#pi} [GeV/c^{2}];L_{xy} [cm]",
      30,0,6, 70,1.65,2.10, 20,-0.1,0.5);
    h.h3_mass_ad0_vs_pt_vs_lxy = new TH3F("h3_AntiD0_mass_pt_lxy",
      "#bar{D}^{0}: m_{K#pi} vs p_{T} vs L_{xy}^{signed}"+t+";p_{T} [GeV/c];m_{K#pi} [GeV/c^{2}];L_{xy} [cm]",
      30,0,6, 70,1.65,2.10, 20,-0.1,0.5);
    return h;
  }

  LcHistSet bookLc(TDirectory* dir, const std::string& tag)
  {
    dir->cd();
    const TString t = (" [" + tag + "]").c_str();
    LcHistSet h;
    h.h_pK_mass_p1K2 = new TH1F("h_pK_mass_p1K2",
      "#Lambda_{c}: p(tr1)K(tr2) sub-system mass"+t+";m_{pK} [GeV/c^{2}];pairs",
      300,1.4,3.2);
    h.h_pK_mass_p2K1 = new TH1F("h_pK_mass_p2K1",
      "#Lambda_{c}: p(tr2)K(tr1) sub-system mass"+t+";m_{pK} [GeV/c^{2}];pairs",
      300,1.4,3.2);
    h.h_pK_mass_vs_pt = new TH2F("h_pK_mass_vs_pt",
      "#Lambda_{c}: pK sub-system mass vs pair p_{T}"+t+";p_{T}^{pair} [GeV/c];m_{pK} [GeV/c^{2}]",
      60,0,12, 150,1.4,3.2);
    h.h_dca_xy_vs_pt = new TH2F("h_dca_xy_vs_pt",
      "#Lambda_{c} daughters: max|DCA_{xy}| vs p_{T}"+t+";p_{T} [GeV/c];max|DCA_{xy}| [cm]",
      40,0,4, 400,0,0.5);
    h.h_dca_z_vs_pt = new TH2F("h_dca_z_vs_pt",
      "#Lambda_{c} daughters: max|DCA_{z}| vs p_{T}"+t+";p_{T} [GeV/c];max|DCA_{z}| [cm]",
      40,0,4, 400,0,1.0);
    h.h_dedx_vs_p_p = new TH2F("h_dedx_vs_p_proton",
      "#Lambda_{c} proton daughter: dE/dx vs p"+t+";p [GeV/c];dE/dx [arb.]",
      60,0,6, 200,0,2000);
    h.h_dedx_vs_p_k = new TH2F("h_dedx_vs_p_kaon",
      "#Lambda_{c} kaon daughter: dE/dx vs p"+t+";p [GeV/c];dE/dx [arb.]",
      60,0,6, 200,0,2000);
    h.h_pt1_vs_pt2 = new TH2F("h_pt1_vs_pt2",
      "#Lambda_{c} daughters: p_{T,1} vs p_{T,2}"+t+";p_{T,1} [GeV/c];p_{T,2} [GeV/c]",
      60,0,6, 60,0,6);
    h.h_pair_dca_vs_mass = new TH2F("h_pair_dca_vs_mass",
      "#Lambda_{c}: |prompt pair DCA| vs m_{pK}"+t+";m_{pK} [GeV/c^{2}];|prompt pair DCA| [cm]",
      150,1.4,3.2, 200,0,0.3);
    return h;
  }

  // -----------------------------------------------------------------------
  // Fill helpers
  // -----------------------------------------------------------------------

  void fillPhi(PhiHistSet& h,
               double mass, double massConstr,
               double pairPt, double pairEta, double pairPhiV,
               double pt1, double pt2,
               double dcaXYmax, double dcaZmax,
               double dedxK1, double dedxK2,
               double p1, double p2,
               double promptPairDca, double deltaPcaZ,
               bool   constrainedValid)
  {
    h.h_mass->Fill(mass);
    h.h_mass_vs_pt->Fill(pairPt, mass);
    h.h_mass_vs_eta->Fill(pairEta, mass);
    h.h_mass_vs_phi->Fill(pairPhiV, mass);
    h.h_dca_xy_vs_pt->Fill(std::min(pt1,pt2), dcaXYmax);
    h.h_dca_z_vs_pt->Fill(std::min(pt1,pt2), dcaZmax);
    h.h_dedx_vs_p_k->Fill(p1, dedxK1);
    h.h_dedx_vs_p_k->Fill(p2, dedxK2);
    h.h_pt1_vs_pt2->Fill(pt1, pt2);
    if (std::isfinite(promptPairDca))
    {
      h.h_pair_dca_vs_mass->Fill(mass, std::abs(promptPairDca));
      h.h3_mass_vs_pt_vs_pairdca->Fill(pairPt, mass, std::abs(promptPairDca));
    }
    h.h_delta_pca_z_vs_mass->Fill(mass, deltaPcaZ);
    if (constrainedValid && std::isfinite(massConstr))
    {
      h.h_mass_constrained->Fill(massConstr);
      h.h_mass_constrained_vs_pt->Fill(pairPt, massConstr);
    }
  }

  void fillD0(D0HistSet& h,
              double massD0, double massAD0,
              double massD0c, double massAD0c,
              double pairPt,
              double pt1, double pt2,
              double dedxK, double dedxPi, double pK, double pPi,
              double dcaXY1, double dcaXY2,
              // d0v quantities (NaN if not available)
              double lxySigned, double dira, double dcaProduct, double d0vPairDca,
              bool d0vValid,
              bool isD0, bool isAntiD0, bool constrainedValid)
  {
    if (isD0 && std::isfinite(massD0))
    {
      h.h_mass_d0->Fill(massD0);
      h.h_mass_d0_vs_pt->Fill(pairPt, massD0);
      if (d0vValid && std::isfinite(lxySigned))
      {
        h.h_lxy_vs_mass->Fill(massD0, lxySigned);
        h.h_dira_vs_mass->Fill(massD0, dira);
        h.h3_mass_d0_vs_pt_vs_lxy->Fill(pairPt, massD0, lxySigned);
      }
      if (constrainedValid && std::isfinite(massD0c))
      {
        h.h_mass_d0_constrained->Fill(massD0c);
        h.h_mass_d0_constrained_vs_pt->Fill(pairPt, massD0c);
      }
    }
    if (isAntiD0 && std::isfinite(massAD0))
    {
      h.h_mass_ad0->Fill(massAD0);
      h.h_mass_ad0_vs_pt->Fill(pairPt, massAD0);
      if (d0vValid && std::isfinite(lxySigned))
      {
        h.h_lxy_vs_mass->Fill(massAD0, lxySigned);
        h.h_dira_vs_mass->Fill(massAD0, dira);
        h.h3_mass_ad0_vs_pt_vs_lxy->Fill(pairPt, massAD0, lxySigned);
      }
      if (constrainedValid && std::isfinite(massAD0c))
      {
        h.h_mass_ad0_constrained->Fill(massAD0c);
        h.h_mass_ad0_constrained_vs_pt->Fill(pairPt, massAD0c);
      }
    }
    if (d0vValid)
    {
      if (std::isfinite(lxySigned)) { h.h_lxy_signed->Fill(lxySigned); h.h_dira->Fill(dira); }
      if (std::isfinite(dcaProduct)) h.h_dca_product->Fill(dcaProduct);
      if (std::isfinite(d0vPairDca)) h.h_d0v_pair_dca->Fill(d0vPairDca);
    }
    h.h_dca_xy1_vs_pt->Fill(pt1, dcaXY1);
    h.h_dca_xy2_vs_pt->Fill(pt2, dcaXY2);
    h.h_dedx_vs_p_k->Fill(pK, dedxK);
    h.h_dedx_vs_p_pi->Fill(pPi, dedxPi);
    h.h_pt1_vs_pt2->Fill(pt1, pt2);
  }

}  // namespace

// ==========================================================================
// Main macro
// ==========================================================================

void MakePromptHadronHistograms(
  const char* inputDir      = "/sphenix/tg/tg01/hf/mitrankova/Hadrons/",
  const char* filePattern   = "hadron_sitpc_HITS_pp_mvtxS300_79513_*.root",
  const char* outputDir     = "output",
  const char* outputName    = "prompt_hadrons_mvtxS300.root",
  const char* treeName      = "pairTree",
  const Long64_t maxEntries = -1,
  const bool usePairVertex  = true,
  const double beamX = 0.0,
  const double beamY = 0.0,
  const double beamZ = 0.0,
  const char* likeSignTreeName     = "likeSignPairTree",
  const bool  includeLikeSignTree  = true,
  const bool  includePrimaryConstrainedPrompt = true,
  const bool  ignoreCrossingZero   = false,
  const char* trackClass           = "all",
  const int   minSiHits            = 4,
  const bool  ScaleQualityBy10     = false)
{
  TH1::AddDirectory(kTRUE);

  const std::string trackClassName = trackClass ? trackClass : "all";
  if (trackClassName != "all" && trackClassName != "tpconly" &&
      trackClassName != "si" && trackClassName != "mixed")
  {
    std::cerr << "ERROR: unknown trackClass " << trackClassName << std::endl;
    return;
  }

  // ---- Build chain -------------------------------------------------------
  const TString chainPattern = TString::Format("%s/%s", inputDir, filePattern);
  TChain chain(treeName);
  const int nUnlikeFiles = chain.Add(chainPattern);
  if (nUnlikeFiles <= 0)
  {
    std::cerr << "ERROR: no files matched " << chainPattern << std::endl;
    return;
  }

  std::vector<std::string> inputFiles;
  if (TObjArray* fl = chain.GetListOfFiles())
    for (int i = 0; i < fl->GetEntries(); ++i)
      if (auto* el = dynamic_cast<TChainElement*>(fl->At(i)))
        inputFiles.emplace_back(el->GetTitle());

  const Long64_t nUnlikeEntries = chain.GetEntries();
  int nLikeSignFiles = 0;
  if (includeLikeSignTree)
    for (const auto& fn : inputFiles)
    {
      std::unique_ptr<TFile> f(TFile::Open(fn.c_str(), "READ"));
      if (!f || f->IsZombie()) continue;
      if (!f->Get(likeSignTreeName)) continue;
      chain.AddFile(fn.c_str(), TTree::kMaxEntries, likeSignTreeName);
      ++nLikeSignFiles;
    }

  std::cout << "Input: pairTree files=" << nUnlikeFiles
            << ", likeSignPairTree files=" << nLikeSignFiles << std::endl;

  // ---- Branch detection --------------------------------------------------
  const bool haveSiBranches =
    branchExists(chain,"has_si1") && branchExists(chain,"has_si2") &&
    branchExists(chain,"nmvtx1")  && branchExists(chain,"nmvtx2") &&
    branchExists(chain,"nintt1")  && branchExists(chain,"nintt2");
  const bool havePairVertex =
    usePairVertex &&
    branchExists(chain,"pv_x") && branchExists(chain,"pv_y") && branchExists(chain,"pv_z");
  const bool havePrimaryBranches =
    branchExists(chain,"primary_px1") && branchExists(chain,"primary_px2") &&
    branchExists(chain,"primary_pca_valid") && branchExists(chain,"prompt_pairDCA") &&
    branchExists(chain,"ntpc_clusters1");
  const bool haveConstrainedBranches =
    includePrimaryConstrainedPrompt &&
    branchExists(chain,"mass_Phi_primary_constrained") &&
    branchExists(chain,"mass_D0_primary_constrained");
  // D0 vertex topology branches (optional; written by set_d0_collision_vertex)
  const bool haveD0vBranches =
    branchExists(chain,"d0v_status") &&
    branchExists(chain,"d0v_lxy_signed") &&
    branchExists(chain,"d0v_dira") &&
    branchExists(chain,"d0v_dca_product") &&
    branchExists(chain,"d0v_pair_dca") &&
    branchExists(chain,"d0v_dca_xy1") &&
    branchExists(chain,"d0v_dca_xy2");

  std::cout << "Si branches " << (haveSiBranches ? "found":"absent")
            << "; pair vertex " << (havePairVertex ? "pv_x/y/z":"beam")
            << "; primary branches " << (havePrimaryBranches ? "found":"absent")
            << "; constrained " << (haveConstrainedBranches ? "found":"absent")
            << "; d0v topology " << (haveD0vBranches ? "found":"absent") << std::endl;

  // ---- Branch variables --------------------------------------------------
  Float_t px1=0,py1=0,pz1=0, px2=0,py2=0,pz2=0;
  Float_t charge1=0, charge2=0;
  Float_t dca_xy1=0,dca_z1=0, dca_xy2=0,dca_z2=0;
  Float_t dedx_1=0, dedx_2=0;
  Float_t pca_x=0,pca_y=0,pca_z=0, pca1_z=0,pca2_z=0;
  Float_t v0_px=0,v0_py=0,v0_pz=0, v0_pt=0;
  Float_t pairDCA=0, alpha=0, qT=0, Lproj=0;
  Short_t npoints1=0, npoints2=0;
  Float_t quality1=0, quality2=0;
  Short_t crossing1=0, crossing2=0;
  UInt_t  candidate_mask=0U;
  Float_t mass_Kshort=0,mass_Lambda=0,mass_AntiLambda=0;
  Float_t mass_K1Pi2=0, mass_Pi1K2=0;

  Float_t pv_x=0,pv_y=0,pv_z=0;
  Int_t   has_si1=0,has_si2=0;
  UInt_t  nmvtx1=0,nmvtx2=0,nintt1=0,nintt2=0;

  Float_t primary_px1=0,primary_py1=0,primary_pz1=0;
  Float_t primary_px2=0,primary_py2=0,primary_pz2=0;
  Float_t primary_pca1_z=0,primary_pca2_z=0,primary_pca_dz=0;
  Int_t   primary_pca_valid=0, prompt_pca_valid=0;
  Float_t prompt_pairDCA=0;
  UInt_t  ntpc_clusters1=0, ntpc_clusters2=0;

  Int_t   primary_constrained_valid1=0, primary_constrained_valid2=0;
  Float_t mass_Phi_primary_constrained=0;
  Float_t mass_D0_primary_constrained=0, mass_AntiD0_primary_constrained=0;
  Float_t mass_K1Pi2_primary_constrained=0, mass_Pi1K2_primary_constrained=0;
  Float_t primary_constrained_pair_pt=0;

  // d0v topology
  Int_t   d0v_status=0;
  Float_t d0v_lxy_signed=0, d0v_dira=0, d0v_dca_product=0, d0v_pair_dca=0;
  Float_t d0v_dca_xy1=0, d0v_dca_xy2=0;

  // Mandatory branches
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
  chain.SetBranchAddress("pairDCA",&pairDCA);
  chain.SetBranchAddress("alpha",&alpha); chain.SetBranchAddress("qT",&qT); chain.SetBranchAddress("Lproj",&Lproj);
  chain.SetBranchAddress("npoints1",&npoints1); chain.SetBranchAddress("npoints2",&npoints2);
  chain.SetBranchAddress("quality1",&quality1); chain.SetBranchAddress("quality2",&quality2);
  chain.SetBranchAddress("crossing1",&crossing1); chain.SetBranchAddress("crossing2",&crossing2);
  chain.SetBranchAddress("candidate_mask",&candidate_mask);
  chain.SetBranchAddress("mass_Kshort",&mass_Kshort);
  chain.SetBranchAddress("mass_Lambda",&mass_Lambda); chain.SetBranchAddress("mass_AntiLambda",&mass_AntiLambda);
  chain.SetBranchAddress("mass_K1Pi2",&mass_K1Pi2);  chain.SetBranchAddress("mass_Pi1K2",&mass_Pi1K2);

  if (havePairVertex)
  { chain.SetBranchAddress("pv_x",&pv_x); chain.SetBranchAddress("pv_y",&pv_y); chain.SetBranchAddress("pv_z",&pv_z); }
  if (haveSiBranches)
  {
    chain.SetBranchAddress("has_si1",&has_si1); chain.SetBranchAddress("has_si2",&has_si2);
    chain.SetBranchAddress("nmvtx1",&nmvtx1); chain.SetBranchAddress("nmvtx2",&nmvtx2);
    chain.SetBranchAddress("nintt1",&nintt1); chain.SetBranchAddress("nintt2",&nintt2);
  }
  if (havePrimaryBranches)
  {
    chain.SetBranchAddress("primary_px1",&primary_px1); chain.SetBranchAddress("primary_py1",&primary_py1);
    chain.SetBranchAddress("primary_pz1",&primary_pz1);
    chain.SetBranchAddress("primary_px2",&primary_px2); chain.SetBranchAddress("primary_py2",&primary_py2);
    chain.SetBranchAddress("primary_pz2",&primary_pz2);
    chain.SetBranchAddress("primary_pca1_z",&primary_pca1_z);
    chain.SetBranchAddress("primary_pca2_z",&primary_pca2_z);
    chain.SetBranchAddress("primary_pca_dz",&primary_pca_dz);
    chain.SetBranchAddress("primary_pca_valid",&primary_pca_valid);
    chain.SetBranchAddress("prompt_pairDCA",&prompt_pairDCA);
    chain.SetBranchAddress("prompt_pca_valid",&prompt_pca_valid);
    chain.SetBranchAddress("ntpc_clusters1",&ntpc_clusters1);
    chain.SetBranchAddress("ntpc_clusters2",&ntpc_clusters2);
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
    chain.SetBranchAddress("primary_constrained_pair_pt",&primary_constrained_pair_pt);
  }
  if (haveD0vBranches)
  {
    chain.SetBranchAddress("d0v_status",&d0v_status);
    chain.SetBranchAddress("d0v_lxy_signed",&d0v_lxy_signed);
    chain.SetBranchAddress("d0v_dira",&d0v_dira);
    chain.SetBranchAddress("d0v_dca_product",&d0v_dca_product);
    chain.SetBranchAddress("d0v_pair_dca",&d0v_pair_dca);
    chain.SetBranchAddress("d0v_dca_xy1",&d0v_dca_xy1);
    chain.SetBranchAddress("d0v_dca_xy2",&d0v_dca_xy2);
  }

  // ---- Selection ladders -------------------------------------------------

  // PHI: prompt track ladder (DCA_xy is the key handle)
  // Fields: name, description, minPt, |dca_xy|max, |dca_z|max, |delta_pca_z|max,
  //         |prompt_pair_dca|max, |primary_pca_z|max, kaonDedxMin, kaonDedxMax
  const std::vector<PhiSelection> phiSelections = {
    { "phi_loose",
      "pT>0.15, no DCA cut; baseline",
      0.15, -1.0, -1.0, 1.0e9, -1.0, 20.0, -1.0, -1.0 },
    { "phi_dcaxy_1mm",
      "pT>0.15, max|dca_xy|<0.10 cm",
      0.15, 0.10, -1.0, 1.0, -1.0, 20.0, -1.0, -1.0 },
    { "phi_dcaxy_0p5mm",
      "pT>0.20, max|dca_xy|<0.05 cm (Si scale)",
      0.20, 0.05, -1.0, 0.5, -1.0, 20.0, -1.0, -1.0 },
    { "phi_dcaxy_0p3mm",
      "pT>0.20, max|dca_xy|<0.03 cm, max|dca_z|<0.3 cm",
      0.20, 0.03, 0.30, 0.5, -1.0, 20.0, -1.0, -1.0 },
    { "phi_pairdca_1mm",
      "phi_dcaxy_0p5mm + |prompt_pair_dca|<0.1 cm",
      0.20, 0.05, -1.0, 0.5, 0.10, 20.0, -1.0, -1.0 },
    { "phi_pairdca_0p5mm",
      "phi_dcaxy_0p5mm + |prompt_pair_dca|<0.05 cm",
      0.20, 0.05, -1.0, 0.5, 0.05, 20.0, -1.0, -1.0 },
    { "phi_pt0p5",
      "phi_dcaxy_0p5mm + pT>0.5 GeV",
      0.50, 0.05, -1.0, 0.5, -1.0, 20.0, -1.0, -1.0 },
    { "phi_pid_loose",
      "phi_dcaxy_0p5mm + kaon dE/dx>200 (loose PID hint)",
      0.20, 0.05, -1.0, 0.5, -1.0, 20.0, 200.0, -1.0 },
    // ── Note-informed cuts (sPH-TRK-2025-002 Table 3) ─────────────────────
    // Note uses DCA<=0.04 cm (400 um) between daughters and decay length<=0.05 cm
    // as a prompt VETO (phi is produced at the PV, not displaced).
    // maxAbsPromptPairDCA = 0.04 cm matches note "Track-to-track DCA<=0.04 cm".
    // maxAbsDeltaPcaZ = 0.2 cm: daughters PCAs in z should agree at PV level.
    { "phi_note_dca",
      "note: pT>0.1, |dca_xy|<0.10 cm, |prompt_pair_dca|<0.04 cm (note DCA<=0.04)",
      0.10, 0.10, -1.0, 0.2, 0.04, 20.0, -1.0, -1.0 },
    { "phi_note_pid",
      "phi_note_dca + kaon dE/dx gate: 200<dE/dx<2000 (note: |dE/dx-pred|/pred<=0.2)",
      0.10, 0.10, -1.0, 0.2, 0.04, 20.0, 200.0, 2000.0 },
    { "phi_note_pid_pcap",
      "phi_note_pid + note track p<=0.7 GeV/c enforced via minPt only (offline: cut p>0.7)",
      0.10, 0.10, -1.0, 0.2, 0.04, 20.0, 200.0, 2000.0 },
    // ── reference = MakeK0sPairHistograms promptMesons/primary_dz_0p3_pid (the visible phi peak):
    //    pT>0.15, ntpc>=10, |dca_xy|,|dca_z|<0.5 cm, |dz of the primary PCAs|<0.3 cm, dE/dx>300 both
    //    (fields after kaonDedxMax: pidMode, minNtpcClusters, maxAbsPrimaryPcaDz)
    { "phi_ref_dedx300",
      "reference: pT>0.15, ntpc>=10, |dca|<0.5, |dz_primaryPCA|<0.3, dE/dx>300 both",
      0.15, 0.50, 0.50, 1.0e9, -1.0, 20.0, 300.0, -1.0,  0, 10, 0.3 },
    { "phi_ref_pidK",
      "reference cuts, momentum-dependent PID: both K-compatible (|n_K|<2)",
      0.15, 0.50, 0.50, 1.0e9, -1.0, 20.0, -1.0, -1.0,  1, 10, 0.3 },
    { "phi_ref_pidK_1id",
      "reference cuts, both K-compatible, >= 1 identified kaon (n_pi>3)",
      0.15, 0.50, 0.50, 1.0e9, -1.0, 20.0, -1.0, -1.0,  2, 10, 0.3 },
    { "phi_ref_pidK_2id",
      "reference cuts, both kaons identified",
      0.15, 0.50, 0.50, 1.0e9, -1.0, 20.0, -1.0, -1.0,  3, 10, 0.3 },
    { "phi_pidK_1id_dcaxy0p5mm",
      "phi_ref_pidK_1id + |dca_xy|<0.05 cm",
      0.15, 0.05, 0.50, 1.0e9, -1.0, 20.0, -1.0, -1.0,  2, 10, 0.3 },
    { "phi_pidK_1id_pairdca0p5mm",
      "phi_ref_pidK_1id + |dca_xy|<0.05 cm + |prompt pair DCA|<0.05 cm",
      0.15, 0.05, 0.50, 1.0e9, 0.05, 20.0, -1.0, -1.0,  2, 10, 0.3 }
    // Note: the track p<=0.7 GeV/c cap from Table 3 must be applied offline
    // on the tree using the stored daughter momenta since there is no setter for it here.
  };

  // D0: combinatorial suppression via DCA_xy, then d0v topology ladder
  // Fields: name, description, minPt, |dca_xy|max, |dca_z|max,
  //         |prompt_pair_dca|max, lxy_signed_min, lxy_signed_max,
  //         dira_min, dca_product_max (negative=off), d0v_pair_dca_max,
  //         kaonDedxMin, kaonDedxMax, pionDedxMin, pionDedxMax
  //
  // lxy_signed_max: upper Lxy bound is essential. The D0 exponential has
  // mean <Lxy> ~ cτ*βγ ~ 123 um * (pT/m_D0). At pT=1.5 GeV the mean is
  // ~185 um; signal is negligible beyond ~500 um (3-4 mean free paths).
  // Without an upper bound the signal window extends to 700+ um collecting
  // only combinatorial pairs smeared by the vertex resolution, diluting S/B
  // without adding signal. Use 0.07 cm (700 um) as a generous upper bound
  // matching the tracking note. For pT-differential studies, tune lower.
  const std::vector<D0Selection> d0Selections = {
    //                    minPt  dcaXY  dcaZ  pDCA  minLxy  maxLxy  DIRA   dcaProd pDCA    K_lo  K_hi  pi_lo pi_hi
    { "d0_loose",
      "pT>0.3, |dca_xy|<0.5 cm; combinatorial baseline (no Lxy cut)",
      0.30, 0.50, -1.0, -1.0,  -1.0,  -1.0,  -1.0, -1.0,   -1.0, -1.0,-1.0,-1.0,-1.0 },
    { "d0_dcaxy_1mm",
      "pT>0.3, |dca_xy|<0.10 cm per daughter",
      0.30, 0.10, -1.0, -1.0,  -1.0,  -1.0,  -1.0, -1.0,   -1.0, -1.0,-1.0,-1.0,-1.0 },
    { "d0_dcaxy_0p5mm",
      "pT>0.3, |dca_xy|<0.05 cm (Si scale)",
      0.30, 0.05, -1.0, -1.0,  -1.0,  -1.0,  -1.0, -1.0,   -1.0, -1.0,-1.0,-1.0,-1.0 },
    { "d0_topo_lxy100um",
      "d0_dcaxy_0p5mm + d0v: Lxy 100-700 um, DIRA>0.990, OS-DCA",
      0.30, 0.05, -1.0, -1.0,  0.010, 0.070, 0.990, 0.0,   -1.0, -1.0,-1.0,-1.0,-1.0 },
    { "d0_topo_lxy200um",
      "d0_dcaxy_0p5mm + d0v: Lxy 200-700 um, DIRA>0.995, OS-DCA, pairDCA<0.02 cm",
      0.30, 0.05, -1.0, -1.0,  0.020, 0.070, 0.995, 0.0,   0.020, -1.0,-1.0,-1.0,-1.0 },
    { "d0_topo_lxy300um",
      "d0_dcaxy_0p5mm + d0v: Lxy 300-700 um, DIRA>0.998, OS-DCA, pairDCA<0.02 cm",
      0.30, 0.05, -1.0, -1.0,  0.030, 0.070, 0.998, 0.0,   0.020, -1.0,-1.0,-1.0,-1.0 },
    { "d0_pt1_topo",
      "d0_topo_lxy200um + pT_daughter>0.5 GeV",
      0.50, 0.05, -1.0, -1.0,  0.020, 0.070, 0.995, 0.0,   0.020, -1.0,-1.0,-1.0,-1.0 },
    { "d0_pid_loose",
      "d0_dcaxy_0p5mm + kaon dE/dx>300, pion dE/dx<300 (no Lxy cut)",
      0.30, 0.05, -1.0, -1.0,  -1.0,  -1.0,  -1.0, -1.0,  -1.0,  300.0,-1.0,-1.0,300.0 },
    // ── Note-informed cuts (sPH-TRK-2025-002 Table 8) ─────────────────────
    // Key numbers from note fit: mu=1862.6 MeV, sigma=10.5 MeV, Yield=248+-44
    // Note uses: Track |IPxy|<0.02 cm, DCAxy<0.002 cm daughters to SV,
    //   Lxy 0.008-0.07 cm, DIRA>0.985.
    // maxLxySigned = 0.07 cm (700 um): upper bound now enforced in the event
    //   loop via the new maxLxySigned field.  Without it the Lxy distribution
    //   showed a flat tail to 700 um that is pure combinatorial (no exponential
    //   falloff visible), diluting S/B without adding D0 signal.
    { "d0_note_loose",
      "note baseline: Lxy 80-700 um, DIRA>0.985, OS-DCA, pairDCA<0.05 cm",
      0.20, 0.02, -1.0, -1.0,  0.008, 0.070, 0.985, 0.0,   0.050, -1.0,-1.0,-1.0,-1.0 },
    { "d0_note_tight",
      "note tight: Lxy 80-700 um, DIRA>0.985, OS-DCA, pairDCA<0.002 cm (20 um)",
      0.20, 0.02, -1.0, -1.0,  0.008, 0.070, 0.985, 0.0,   0.002, -1.0,-1.0,-1.0,-1.0 },
    { "d0_note_pid",
      "d0_note_tight + kaon dE/dx>300",
      0.20, 0.02, -1.0, -1.0,  0.008, 0.070, 0.985, 0.0,   0.002, 300.0,-1.0,-1.0,-1.0 },
    // ── momentum-dependent PID per K pi assignment (pidMode 1: K/pi-compatible, 2: K identified)
    //    + L_xy scan (L_xy w.r.t. the D0 vertex refitted without the daughters, the key D0 cut);
    //    fields after pionDedxMax: pidMode, minAbsD0vDcaXY
    //                    minPt  dcaXY  dcaZ  pDCA  minLxy  maxLxy  DIRA   dcaProd pDCA    K_lo  K_hi  pi_lo pi_hi  pid  |d0|min
    { "d0_pid",
      "d0_dcaxy_0p5mm + K/pi dE/dx compatible (no topology)",
      0.30, 0.05, -1.0, -1.0,  -1.0,  -1.0,  -1.0, -1.0,  -1.0,  -1.0,-1.0,-1.0,-1.0,  1, -1.0 },
    { "d0_pid_lxy0",
      "d0_pid + L_xy > 0",
      0.30, 0.05, -1.0, -1.0,  0.000, 0.070, -1.0, -1.0,  -1.0,  -1.0,-1.0,-1.0,-1.0,  1, -1.0 },
    { "d0_pid_lxy50um",
      "d0_pid + L_xy 50-700 um, DIRA>0.95",
      0.30, 0.05, -1.0, -1.0,  0.005, 0.070, 0.950, -1.0,  -1.0,  -1.0,-1.0,-1.0,-1.0,  1, -1.0 },
    { "d0_pid_lxy100um",
      "d0_pid + L_xy 100-700 um, DIRA>0.985, OS-DCA",
      0.30, 0.05, -1.0, -1.0,  0.010, 0.070, 0.985, 0.0,   -1.0,  -1.0,-1.0,-1.0,-1.0,  1, -1.0 },
    { "d0_pid_lxy150um",
      "d0_pid + L_xy 150-700 um, DIRA>0.985, OS-DCA",
      0.30, 0.05, -1.0, -1.0,  0.015, 0.070, 0.985, 0.0,   -1.0,  -1.0,-1.0,-1.0,-1.0,  1, -1.0 },
    { "d0_pid_lxy200um",
      "d0_pid + L_xy 200-700 um, DIRA>0.985, OS-DCA, pairDCA<0.02 cm",
      0.30, 0.05, -1.0, -1.0,  0.020, 0.070, 0.985, 0.0,   0.020, -1.0,-1.0,-1.0,-1.0,  1, -1.0 },
    { "d0_pid_lxy100um_d0min",
      "d0_pid_lxy100um + both |DCA_xy to the D0 vertex| > 30 um",
      0.30, 0.05, -1.0, -1.0,  0.010, 0.070, 0.985, 0.0,   -1.0,  -1.0,-1.0,-1.0,-1.0,  1, 0.003 },
    { "d0_pidKid_lxy100um",
      "K identified (p<~0.6 GeV) + pi compatible, L_xy 100-700 um, DIRA>0.985, OS-DCA",
      0.30, 0.05, -1.0, -1.0,  0.010, 0.070, 0.985, 0.0,   -1.0,  -1.0,-1.0,-1.0,-1.0,  2, -1.0 }
  };

  // Lambda_c: prompt charm baryon; full p-K-pi triplet not yet available,
  // so we study the p-K sub-system from the pair, which defines the Dalitz band.
  // Fields: name, description, minPt, |dca_xy|max, |dca_z|max,
  //         |prompt_pair_dca|max, pKSubMassMin, pKSubMassMax,
  //         protonDedxMin, protonDedxMax, kaonDedxMin, kaonDedxMax
  const std::vector<LcSelection> lcSelections = {
    { "lc_loose",
      "pT>0.5, |dca_xy|<0.5 cm; all pK sub-system masses",
      0.50, 0.50, -1.0, -1.0,  1.4, 3.2, -1.0,-1.0,-1.0,-1.0 },
    { "lc_dcaxy_1mm",
      "pT>0.5, |dca_xy|<0.10 cm",
      0.50, 0.10, -1.0, -1.0,  1.4, 3.2, -1.0,-1.0,-1.0,-1.0 },
    { "lc_dcaxy_0p5mm",
      "pT>0.5, |dca_xy|<0.05 cm (Si scale)",
      0.50, 0.05, -1.0, -1.0,  1.4, 3.2, -1.0,-1.0,-1.0,-1.0 },
    { "lc_pairdca_1mm",
      "lc_dcaxy_0p5mm + |prompt_pair_dca|<0.10 cm",
      0.50, 0.05, -1.0, 0.10,  1.4, 3.2, -1.0,-1.0,-1.0,-1.0 },
    // ── pK mass window: CORRECTED upper edge ─────────────────────────────
    // For a Lc->pKpi Dalitz, the pK sub-system ranges from m_p+m_K~1.43 GeV
    // to m_Lc - m_pi = 2.286 - 0.140 = 2.146 GeV.
    // Previous version had upper edge 2.00 GeV which cut signal — fixed.
    { "lc_pKmass_window",
      "lc_dcaxy_0p5mm + pK sub-mass 1.43-2.15 GeV (correct Dalitz range)",
      0.50, 0.05, -1.0, -1.0,  1.43, 2.15, -1.0,-1.0,-1.0,-1.0 },
    // ── Preselection to control background shape ──────────────────────────
    // The pK combinatorial background without any PID is enormous (~600k pairs
    // at lc_loose) and completely featureless, making it impossible to identify
    // a signal or assess S/B.  Three orthogonal handles reduce it to a tractable
    // level where the background is smooth enough for sideband subtraction:
    //
    // 1. Pair pT > 1 GeV/c: Lc from pp@200 GeV peaks at pT~1.5-3 GeV;
    //    background falls steeply with pT; this alone reduces background ~5x.
    // 2. Tight pair DCA < 0.04 cm: prompt pairs (Lc daughters meet at PV)
    //    should have small pair DCA; combinatorial pairs have flat DCA distribution.
    // 3. pK mass window 1.43-2.15 GeV: removes everything outside the Dalitz band
    //    (K0S, Ks-like pairs, random pion pairs re-interpreted as pK).
    //
    // The WRONG approach (lc_note_lxy removed): using d0v_lxy_signed for Lc.
    // That branch is computed for D0 candidates only and is undefined for pK pairs
    // which do not have a kMaskD0 bit; it selects a completely different subset.
    { "lc_pt1_pairdca",
      "lc_pKmass_window + pairPt>1 GeV + |prompt_pair_dca|<0.04 cm: background control preselection",
      0.20, 0.05, -1.0, 0.04,  1.43, 2.15, -1.0,-1.0,-1.0,-1.0 },
    // ── PID cuts on top of the preselection ──────────────────────────────
    // proton dE/dx > 400: at p < 1 GeV/c protons are clearly separated from
    //   pions and kaons (proton dE/dx rises steeply below beta~0.5).
    //   Use 400 rather than 800 (previous) because 800 only works for p<0.5 GeV/c;
    //   at 400-800 the proton band is visible up to p~0.8 GeV/c.
    // kaon dE/dx > 200: kaon band visible at p < 0.6 GeV/c.
    // Both thresholds should be calibrated from Lambda->p+pi (proton) and
    // phi->K+K- (kaon) peak daughters before applying here.
    { "lc_pid_loose",
      "lc_pt1_pairdca + proton dE/dx>400, kaon dE/dx>200 (calibrate from Lambda+phi)",
      0.20, 0.05, -1.0, 0.04,  1.43, 2.15, 400.0,-1.0,200.0,-1.0 },
    { "lc_pid_tight",
      "lc_pt1_pairdca + proton dE/dx>600, kaon dE/dx>300, pT>1.5 GeV",
      0.30, 0.05, -1.0, 0.04,  1.43, 2.15, 600.0,-1.0,300.0,-1.0 },
    { "lc_pt1p5",
      "lc_pKmass_window + pT>1.5 GeV (background shape diagnostic)",
      1.50, 0.05, -1.0, -1.0,  1.43, 2.15, -1.0,-1.0,-1.0,-1.0 },
    // ── momentum-dependent PID per p K assignment (field after kaonDedxMax: pidMode)
    { "lc_pt1_pairdca_pidP",
      "lc_pt1_pairdca + proton identified (|n_p|<2, n_K>2), kaon compatible",
      0.20, 0.05, -1.0, 0.04,  1.43, 2.15, -1.0,-1.0,-1.0,-1.0,  1 },
    { "lc_pt1_pairdca_pidPK",
      "lc_pt1_pairdca + proton and kaon identified",
      0.20, 0.05, -1.0, 0.04,  1.43, 2.15, -1.0,-1.0,-1.0,-1.0,  2 }
    // Note (Table 10) cuts not yet applied here because they require the full
    // pKpi triplet (Lc->pKpi): Lxy 50-200 um, DIRA>0.6, SV chi2/nDoF<25.
    // These will only make sense once the triplet finder is available.
    // For now the pK two-body proxy is used as a phase-space pre-filter only.
  };

  // ---- Output file -------------------------------------------------------
  gSystem->mkdir(outputDir, kTRUE);
  const TString outputPath = TString::Format("%s/%s", outputDir, outputName);
  std::unique_ptr<TFile> output(TFile::Open(outputPath, "RECREATE"));
  if (!output || output->IsZombie())
  { std::cerr << "ERROR: cannot create " << outputPath << std::endl; return; }
  output->cd();

  TNamed info_track("track_class",
    TString::Format("trackClass=%s minSiHits=%d pairVertex=%d d0vBranches=%d constrainedBranches=%d",
                    trackClassName.c_str(), minSiHits,
                    havePairVertex?1:0, haveD0vBranches?1:0,
                    haveConstrainedBranches?1:0).Data());
  info_track.Write();

  // ---- Book histograms ---------------------------------------------------
  // Structure: species/selectionName/unlike or like
  using PhiMap = std::map<std::string, std::map<std::string, PhiHistSet>>;
  using D0Map  = std::map<std::string, std::map<std::string, D0HistSet>>;
  using LcMap  = std::map<std::string, std::map<std::string, LcHistSet>>;

  PhiMap phiHists;
  D0Map  d0Hists;
  LcMap  lcHists;

  TDirectory* phiDir = output->mkdir("phi");
  TDirectory* d0Dir  = output->mkdir("D0");
  TDirectory* lcDir  = output->mkdir("LambdaC");

  for (const auto& sel : phiSelections)
    for (const auto& chName : std::vector<std::string>{"unlike","like"})
    {
      // mkdir("a/b") returns "a": create the selection directory once, then the charge one inside it
      TDirectory* selDir = phiDir->GetDirectory(sel.name.c_str());
      if (!selDir) selDir = phiDir->mkdir(sel.name.c_str());
      TDirectory* d = selDir->mkdir(chName.c_str());
      phiHists[sel.name][chName] = bookPhi(d, sel.name+" "+chName);
    }

  for (const auto& sel : d0Selections)
    for (const auto& chName : std::vector<std::string>{"unlike","like"})
    {
      // mkdir("a/b") returns "a": create the selection directory once, then the charge one inside it
      TDirectory* selDir = d0Dir->GetDirectory(sel.name.c_str());
      if (!selDir) selDir = d0Dir->mkdir(sel.name.c_str());
      TDirectory* d = selDir->mkdir(chName.c_str());
      d0Hists[sel.name][chName] = bookD0(d, sel.name+" "+chName);
    }

  for (const auto& sel : lcSelections)
    for (const auto& chName : std::vector<std::string>{"unlike","like"})
    {
      // mkdir("a/b") returns "a": create the selection directory once, then the charge one inside it
      TDirectory* selDir = lcDir->GetDirectory(sel.name.c_str());
      if (!selDir) selDir = lcDir->mkdir(sel.name.c_str());
      TDirectory* d = selDir->mkdir(chName.c_str());
      lcHists[sel.name][chName] = bookLc(d, sel.name+" "+chName);
    }

  // ---- Event loop --------------------------------------------------------
  const Long64_t totalEntries    = chain.GetEntries();
  const Long64_t nLikeSignEntries = std::max<Long64_t>(0, totalEntries - nUnlikeEntries);
  const Long64_t nUnlikeProcess  = maxEntries>=0 ? std::min(nUnlikeEntries,maxEntries) : nUnlikeEntries;
  const Long64_t nLikeProcess    = maxEntries>=0 ? std::min(nLikeSignEntries,maxEntries) : nLikeSignEntries;
  const Long64_t nProcess        = nUnlikeProcess + nLikeProcess;

  std::cout << "Processing unlike=" << nUnlikeProcess
            << " likeSign=" << nLikeProcess << std::endl;

  const double qualScale = ScaleQualityBy10 ? 0.0001 : 1.0;

  for (Long64_t proc = 0; proc < nProcess; ++proc)
  {
    const Long64_t entry = proc < nUnlikeProcess
                         ? proc
                         : nUnlikeEntries + (proc - nUnlikeProcess);
    chain.GetEntry(entry);

    if (proc % 100000 == 0)
      std::cout << "Processing " << proc << " / " << nProcess << std::endl;

    if (ignoreCrossingZero && (crossing1==0 || crossing2==0)) continue;

    // Track class filter
    if (trackClassName != "all" && haveSiBranches)
    {
      const int nsi1 = static_cast<int>(nmvtx1+nintt1);
      const int nsi2 = static_cast<int>(nmvtx2+nintt2);
      const bool si1 = has_si1!=0 && nsi1>=minSiHits;
      const bool si2 = has_si2!=0 && nsi2>=minSiHits;
      const bool tpc1 = has_si1==0, tpc2 = has_si2==0;
      const bool keep =
        (trackClassName=="si"      && si1  && si2)  ||
        (trackClassName=="tpconly" && tpc1 && tpc2) ||
        (trackClassName=="mixed"   && ((si1&&tpc2)||(tpc1&&si2)));
      if (!keep) continue;
    }

    const std::string sourceTree = chain.GetTree() ? chain.GetTree()->GetName() : "";
    const bool fromLike  = (sourceTree == likeSignTreeName);
    const bool unlikeSign = charge1*charge2 < 0.0;
    const bool likeSign   = charge1*charge2 > 0.0;
    if ((fromLike && !likeSign) || (!fromLike && !unlikeSign)) continue;

    const std::string chName = fromLike ? "like" : "unlike";

    // Vertex
    const bool pvValid = havePairVertex && std::isfinite(pv_x) && std::isfinite(pv_y) && std::isfinite(pv_z);
    const double vtxX = pvValid ? pv_x : beamX;
    const double vtxY = pvValid ? pv_y : beamY;
    const double vtxZ = pvValid ? pv_z : beamZ;

    // Kinematics
    const double pt1 = std::hypot(px1,py1);
    const double pt2 = std::hypot(px2,py2);
    const double p1  = std::sqrt(px1*px1+py1*py1+pz1*pz1);
    const double p2  = std::sqrt(px2*px2+py2*py2+pz2*pz2);
    const double maxDcaXY = std::max(std::abs(dca_xy1),std::abs(dca_xy2));
    const double maxDcaZ  = std::max(std::abs(dca_z1), std::abs(dca_z2));
    const double absDeltaPcaZ = std::abs(pca1_z - pca2_z);
    const double pairPt  = std::hypot(v0_px,v0_py);
    const double pairEta = (pairPt>0) ? std::asinh(v0_pz/pairPt) : 0.0;
    const double pairPhiV= std::atan2(v0_py,v0_px);
    const double qualMax = qualScale * std::max(quality1,quality2);

    // Primary (prompt) kinematics
    const bool primaryValid =
      havePrimaryBranches && primary_pca_valid!=0 &&
      std::isfinite(primary_px1) && std::isfinite(primary_px2);
    const double ppx1 = primaryValid ? primary_px1 : px1;
    const double ppy1 = primaryValid ? primary_py1 : py1;
    const double ppz1 = primaryValid ? primary_pz1 : pz1;
    const double ppx2 = primaryValid ? primary_px2 : px2;
    const double ppy2 = primaryValid ? primary_py2 : py2;
    const double ppz2 = primaryValid ? primary_pz2 : pz2;
    const double pp1 = std::sqrt(ppx1*ppx1 + ppy1*ppy1 + ppz1*ppz1);
    const double pp2 = std::sqrt(ppx2*ppx2 + ppy2*ppy2 + ppz2*ppz2);
    const unsigned int ntpcMin = havePrimaryBranches ? std::min(ntpc_clusters1, ntpc_clusters2) : 0U;
    const double absPrimaryPcaDz =
      (primaryValid && std::isfinite(primary_pca_dz)) ? std::abs(primary_pca_dz) : std::abs(primary_pca1_z - primary_pca2_z);
    const double absPromptPairDCA =
      (primaryValid && prompt_pca_valid!=0 && std::isfinite(prompt_pairDCA))
        ? std::abs(prompt_pairDCA)
        : kNaN();

    const double primaryPcaZMax =
      primaryValid
        ? std::max(std::abs(primary_pca1_z-vtxZ), std::abs(primary_pca2_z-vtxZ))
        : 1e9;

    // Constrained (vertex-pinned) masses
    const bool constrainedValid =
      haveConstrainedBranches && primary_constrained_valid1!=0 && primary_constrained_valid2!=0;

    // D0 vertex
    const bool d0vOk = haveD0vBranches && d0v_status==1;

    // Charge-ordered kaon/pion for D0 (unlike only)
    // D0  = K- pi+  (positive daughter is pion = track with charge>0)
    // aD0 = K+ pi-  (positive daughter is kaon)
    double massD0=kNaN(), massAD0=kNaN(), massD0c=kNaN(), massAD0c=kNaN();
    double dedxK=0,dedxPi=0, pKdau=0,pPidau=0, dxy1dau=0,dxy2dau=0;
    bool isD0=(candidate_mask&kMaskD0)!=0U;
    bool isAD0=(candidate_mask&kMaskAntiD0)!=0U;

    if (unlikeSign)
    {
      if (charge1>0)
      {
        // track1=pi+, track2=K-
        massD0  = invMass(ppx2,ppy2,ppz2,kKaonMass, ppx1,ppy1,ppz1,kPionMass);
        massAD0 = invMass(ppx1,ppy1,ppz1,kKaonMass, ppx2,ppy2,ppz2,kPionMass);
        dedxK=dedx_2; dedxPi=dedx_1; pKdau=p2; pPidau=p1;
        dxy1dau=dca_xy1; dxy2dau=dca_xy2;
        if (haveConstrainedBranches)
        {
          massD0c  = std::isfinite(mass_D0_primary_constrained)    ? mass_D0_primary_constrained    : kNaN();
          massAD0c = std::isfinite(mass_AntiD0_primary_constrained) ? mass_AntiD0_primary_constrained : kNaN();
        }
      }
      else
      {
        // track1=K-, track2=pi+
        massD0  = invMass(ppx1,ppy1,ppz1,kKaonMass, ppx2,ppy2,ppz2,kPionMass);
        massAD0 = invMass(ppx2,ppy2,ppz2,kKaonMass, ppx1,ppy1,ppz1,kPionMass);
        dedxK=dedx_1; dedxPi=dedx_2; pKdau=p1; pPidau=p2;
        dxy1dau=dca_xy1; dxy2dau=dca_xy2;
        if (haveConstrainedBranches)
        {
          massD0c  = std::isfinite(mass_D0_primary_constrained)    ? mass_D0_primary_constrained    : kNaN();
          massAD0c = std::isfinite(mass_AntiD0_primary_constrained) ? mass_AntiD0_primary_constrained : kNaN();
        }
      }
    }
    else
    {
      // Like-sign: keep both assignments from the tree branches
      massD0  = std::isfinite(mass_K1Pi2) ? mass_K1Pi2 : kNaN();
      massAD0 = std::isfinite(mass_Pi1K2) ? mass_Pi1K2 : kNaN();
      isD0 = (candidate_mask&kMaskD0)!=0U || (candidate_mask&kMaskAntiD0)!=0U;
      isAD0 = isD0;
      dedxK=dedx_1; dedxPi=dedx_2; pKdau=p1; pPidau=p2;
      dxy1dau=dca_xy1; dxy2dau=dca_xy2;
    }

    // Phi: use the PV-constrained mass as the primary analysis mass.
    // For a prompt phi both daughters originate at the PV; constraining their
    // momenta to point back to the measured PV improves the mass resolution
    // (reduces the smearing from the primary-vertex position uncertainty) and
    // gives a mass that is independent of the curvature bias to first order,
    // because the constraint pulls the momenta to satisfy conservation at the
    // vertex rather than relying on the absolute curvature measurement.
    // The free invariant mass (massPhi) is kept as a secondary diagnostic
    // histogram via massPhiC / massPhi role-swap compared to the old version:
    //   h_mass_Phi           -> now filled with constrained mass (primary plot)
    //   h_mass_Phi_constrained (second branch in fillPhi) -> filled with free mass
    // If constrained branches are absent, fall back to the free mass for both.
    const double massFree = invMass(ppx1,ppy1,ppz1,kKaonMass, ppx2,ppy2,ppz2,kKaonMass);
    const double massConstr = (constrainedValid && std::isfinite(mass_Phi_primary_constrained))
                               ? mass_Phi_primary_constrained : kNaN();
    // h_mass_Phi: momenta at the primary PCA (as the reference MakeK0sPairHistograms phi);
    // h_mass_Phi_constrained: PV-constrained (the two gave the same peak on the 2.2M sample)
    const double massPhi  = massFree;
    const double massPhiC = massConstr;
    const bool isPhi = (candidate_mask & kMaskPhi) != 0U;

    // Lambda_c p-K sub-system (both assignments)
    const double pKmass_p1K2 = lambdaCSubMass(ppx1,ppy1,ppz1, ppx2,ppy2,ppz2, true);
    const double pKmass_p2K1 = lambdaCSubMass(ppx1,ppy1,ppz1, ppx2,ppy2,ppz2, false);

    // ---- Fill phi histograms -------------------------------------------
    if (isPhi)
    {
      for (const auto& sel : phiSelections)
      {
        if (pt1 < sel.minDaughterPt || pt2 < sel.minDaughterPt) continue;
        if (sel.maxAbsDcaXY > 0.0 && maxDcaXY > sel.maxAbsDcaXY) continue;
        if (sel.maxAbsDcaZ  > 0.0 && maxDcaZ  > sel.maxAbsDcaZ)  continue;
        if (absDeltaPcaZ > sel.maxAbsDeltaPcaZ) continue;
        if (sel.maxAbsPrimaryPcaZ > 0.0 && primaryPcaZMax > sel.maxAbsPrimaryPcaZ) continue;
        if (sel.maxAbsPromptPairDCA > 0.0 &&
            (!std::isfinite(absPromptPairDCA) || absPromptPairDCA > sel.maxAbsPromptPairDCA)) continue;
        if (!passesDedx(dedx_1, sel.kaonDedxMin, sel.kaonDedxMax)) continue;
        if (!passesDedx(dedx_2, sel.kaonDedxMin, sel.kaonDedxMax)) continue;
        if (ntpcMin < sel.minNtpcClusters) continue;
        if (sel.maxAbsPrimaryPcaDz > 0.0 && !(absPrimaryPcaDz < sel.maxAbsPrimaryPcaDz)) continue;
        if (!phiPidPass(sel.pidMode, dedx_1, pp1, dedx_2, pp2)) continue;

        fillPhi(phiHists[sel.name][chName],
                massPhi, massPhiC, pairPt, pairEta, pairPhiV,
                pt1, pt2, maxDcaXY, maxDcaZ,
                dedx_1, dedx_2, p1, p2,
                absPromptPairDCA, absDeltaPcaZ,
                constrainedValid);
      }
    }

    // ---- Fill D0 histograms --------------------------------------------
    if (isD0 || isAD0)
    {
      const double d0vLxy  = d0vOk ? d0v_lxy_signed  : kNaN();
      const double d0vDira = d0vOk ? d0v_dira         : kNaN();
      const double d0vDcaP = d0vOk ? d0v_dca_product  : kNaN();
      const double d0vPDca = d0vOk ? d0v_pair_dca     : kNaN();

      for (const auto& sel : d0Selections)
      {
        if (pt1 < sel.minDaughterPt || pt2 < sel.minDaughterPt) continue;
        if (sel.maxAbsDcaXY > 0.0 && maxDcaXY > sel.maxAbsDcaXY) continue;
        if (sel.maxAbsDcaZ  > 0.0 && maxDcaZ  > sel.maxAbsDcaZ)  continue;
        if (sel.maxAbsPromptPairDCA > 0.0 &&
            (!std::isfinite(absPromptPairDCA) || absPromptPairDCA > sel.maxAbsPromptPairDCA)) continue;

        // d0v topology gates (only applied when d0v is available and cut is set)
        if (sel.minLxySigned > -0.5)
        {
          if (!d0vOk || !std::isfinite(d0vLxy) || d0vLxy < sel.minLxySigned) continue;
        }
        if (sel.maxLxySigned > -0.5)
        {
          if (!d0vOk || !std::isfinite(d0vLxy) || d0vLxy > sel.maxLxySigned) continue;
        }
        if (sel.minDira > -0.5)
        {
          if (!d0vOk || !std::isfinite(d0vDira) || d0vDira < sel.minDira) continue;
        }
        // dcaProduct < 0 means opposite-sign DCA (OS requirement)
        if (sel.maxDcaProduct >= 0.0)
        {
          if (!d0vOk || !std::isfinite(d0vDcaP) || d0vDcaP >= 0.0) continue;
        }
        if (sel.maxD0vPairDca > 0.0)
        {
          if (!d0vOk || !std::isfinite(d0vPDca) || d0vPDca > sel.maxD0vPairDca) continue;
        }
        if (!passesDedx(dedxK,  sel.kaonDedxMin,  sel.kaonDedxMax))  continue;
        if (!passesDedx(dedxPi, sel.pionDedxMin,  sel.pionDedxMax))  continue;
        if (sel.minAbsD0vDcaXY > 0.0 &&
            !(d0vOk && std::abs(d0v_dca_xy1) > sel.minAbsD0vDcaXY && std::abs(d0v_dca_xy2) > sel.minAbsD0vDcaXY)) continue;

        // PID per K pi assignment: unlike sign D0 = negative track is the kaon, anti-D0 = positive;
        // like sign "D0" slot = K1Pi2 (track 1 kaon), "anti-D0" slot = Pi1K2 (track 2 kaon)
        const bool track1KaonInD0 = unlikeSign ? (charge1 < 0) : true;
        const bool selD0 = isD0 &&
          d0PidPass(sel.pidMode, track1KaonInD0 ? dedx_1 : dedx_2, track1KaonInD0 ? pp1 : pp2,
                                 track1KaonInD0 ? dedx_2 : dedx_1, track1KaonInD0 ? pp2 : pp1);
        const bool selAD0 = isAD0 &&
          d0PidPass(sel.pidMode, track1KaonInD0 ? dedx_2 : dedx_1, track1KaonInD0 ? pp2 : pp1,
                                 track1KaonInD0 ? dedx_1 : dedx_2, track1KaonInD0 ? pp1 : pp2);
        if (!selD0 && !selAD0) continue;

        fillD0(d0Hists[sel.name][chName],
               massD0, massAD0, massD0c, massAD0c,
               pairPt, pt1, pt2,
               dedxK, dedxPi, pKdau, pPidau,
               dxy1dau, dxy2dau,
               d0vLxy, d0vDira, d0vDcaP, d0vPDca,
               d0vOk,
               selD0, selAD0,
               constrainedValid);
      }
    }

    // ---- Fill Lambda_c histograms ----------------------------------------
    // Global Lc pre-gate: pairPt > 0.5 GeV reduces the combinatorial by >5x
    // before any selection is applied, keeping the event loop fast.
    // Per-cut pT cuts (sel.minDaughterPt, sel.maxAbsPromptPairDCA) refine further.
    // Note: Lc is a prompt charm baryon — do NOT apply d0v_lxy_signed here;
    // that branch is only valid for D0 candidates and is undefined for pK pairs.
    if ((unlikeSign || likeSign) && pairPt > 0.5)
    {
      for (const auto& sel : lcSelections)
      {
        if (pt1 < sel.minDaughterPt || pt2 < sel.minDaughterPt) continue;
        if (sel.maxAbsDcaXY > 0.0 && maxDcaXY > sel.maxAbsDcaXY) continue;
        if (sel.maxAbsDcaZ  > 0.0 && maxDcaZ  > sel.maxAbsDcaZ)  continue;
        if (sel.maxAbsPromptPairDCA > 0.0 &&
            (!std::isfinite(absPromptPairDCA) || absPromptPairDCA > sel.maxAbsPromptPairDCA)) continue;

        // pK sub-mass window (applied to both assignments independently)
        const bool pass_p1K2 =
          pKmass_p1K2 >= sel.pKSubMassMin && pKmass_p1K2 < sel.pKSubMassMax &&
          passesDedx(dedx_1, sel.protonDedxMin, sel.protonDedxMax) &&
          passesDedx(dedx_2, sel.kaonDedxMin,   sel.kaonDedxMax) &&
          lcPidPass(sel.pidMode, dedx_1, pp1, dedx_2, pp2);

        const bool pass_p2K1 =
          pKmass_p2K1 >= sel.pKSubMassMin && pKmass_p2K1 < sel.pKSubMassMax &&
          passesDedx(dedx_2, sel.protonDedxMin, sel.protonDedxMax) &&
          passesDedx(dedx_1, sel.kaonDedxMin,   sel.kaonDedxMax) &&
          lcPidPass(sel.pidMode, dedx_2, pp2, dedx_1, pp1);

        if (!pass_p1K2 && !pass_p2K1) continue;

        LcHistSet& h = lcHists[sel.name][chName];
        if (pass_p1K2)
        {
          h.h_pK_mass_p1K2->Fill(pKmass_p1K2);
          h.h_pK_mass_vs_pt->Fill(pairPt, pKmass_p1K2);
          h.h_dedx_vs_p_p->Fill(p1, dedx_1);
          h.h_dedx_vs_p_k->Fill(p2, dedx_2);
        }
        if (pass_p2K1)
        {
          h.h_pK_mass_p2K1->Fill(pKmass_p2K1);
          h.h_pK_mass_vs_pt->Fill(pairPt, pKmass_p2K1);
          h.h_dedx_vs_p_p->Fill(p2, dedx_2);
          h.h_dedx_vs_p_k->Fill(p1, dedx_1);
        }
        h.h_dca_xy_vs_pt->Fill(std::min(pt1,pt2), maxDcaXY);
        h.h_dca_z_vs_pt->Fill(std::min(pt1,pt2), maxDcaZ);
        h.h_pt1_vs_pt2->Fill(pt1, pt2);
        if (std::isfinite(absPromptPairDCA))
          h.h_pair_dca_vs_mass->Fill(pass_p1K2 ? pKmass_p1K2 : pKmass_p2K1, absPromptPairDCA);
      }
    }
  }  // end event loop

  output->Write();
  output->Close();
  std::cout << "Wrote: " << outputPath << std::endl;
}