// MakeMineV0PairHistograms.C
//
// One-pass V0 QA for TpcV0CandidateTree pairTree and likeSignPairTree.
// Produces 10 cumulative cut levels plus exactCut1/2/3, with:
//   * K0S mass vs V0 pT
//   * Lambda mass vs V0 pT
//   * anti-Lambda mass vs V0 pT
//   * phi -> K+K- mass QA with prompt-track cuts
//   * D0/anti-D0 -> K pi mass QA with prompt-track cuts
//   * ordinary primary-PCA and forced-primary-vertex-constrained prompt sets
//   * like-sign Lambda, phi, and D0 backgrounds read from likeSignPairTree
//   * Armenteros-Podolanski qT vs alpha
//   * compact TH3F: V0 pT vs mass vs |pairDCA|
//
//
// Output layout:
//   * cutXX/.../unlike is filled only from pairTree
//   * cutXX/.../like, plusplus, minusminus are filled only from likeSignPairTree
//   * promptMesons/... uses ordinary primary-PCA momenta
//   * promptMesonsPrimaryConstrained/... uses forced-primary-vertex momenta
//
// Lambda convention from TpcDstV0Finder:
//   mass_Lambda:     positive daughter = proton, negative daughter = pion
//   mass_AntiLambda: positive daughter = pion,   negative daughter = antiproton
//
// For the Lambda plots, require the proton-hypothesis daughter to have the
// larger transverse momentum. This is a useful background rejection, but it
// is not a fundamental kinematic requirement and can be disabled.
//
// Example:
// root -l -b -q \
// 'MakeMineV0PairHistograms.C("/path/to/V0/files","HITS*_V0.root","output","v0_qa.root","pairTree",true)'
//
// Si+TPC setup (TpcV0CandidateTree with use_sitpc_tracks):
//   * trackClass selects the daughters, run once per class and compare the outputs:
//       "all"     : no requirement
//       "tpconly" : both daughters without Si points (has_si == 0; needs
//                   set_sitpc_add_unmatched_tpc_tracks(true) or a TPC-only production)
//       "si"      : both daughters with >= minSiHits Si points (nmvtx + nintt)
//       "mixed"   : one daughter of each kind
//   * the per-pair collision vertex (pv_x, pv_y, pv_z) is used for the V0 flight/DIRA and the
//     primary-PCA z when the branches exist (usePairVertex), otherwise beamX/Y/Z
//   * applySignedDeltaPhiCut = false: no sailor/cowboy (signed Delta-phi) cut
//   * loose cut levels: decay radius from 0 (Si-matched V0 daughters only exist for decays inside
//     the MVTX), pair DCA down to Si scale (0.05 cm)
//   * added QA (all sets): dE/dx vs p of the daughters in the K0S / Lambda / phi peak and
//     sidebands (pion, proton, kaon bands), fine |pair DCA| vs pT vs mass, per-track DCA_xy/DCA_z
//     vs pT, and the pT shift of the vertex constraint (pT_constrained - pT)/pT vs pT
//   root -l -b -q 'MakeK0sPairHistograms.C("/path","hadron_sitpc_*.root","out","qa_si.root",
//                   "pairTree",true,false,-1,true,0,0,0,"likeSignPairTree",true,true,false,
//                   "si",4,false,true)'

#include <TBranch.h>
#include <TChain.h>
#include <TChainElement.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TH1I.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TString.h>
#include <TSystem.h>
#include <TTree.h>
#include <TMath.h>
#include <TNamed.h>
#include <TObjArray.h>

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
  enum class ChargeCategory
  {
    Unlike,
    Like
  };

  struct Selection
  {
    std::string name;
    std::string description;

    double maxAbsPcaZ;
    double maxAbsDeltaPcaZ;
    double minDaughterPt;
    double minDecayRadius;
    double maxAbsAlpha;
    double maxAbsPairDCA;
    double minDIRA;
    double maxQuality;
    int minnpoints;

    // Negative bound disables that side of the dE/dx selection.
    double pionDedxMin{-1.0};
    double pionDedxMax{400.0};
    double protonDedxMin{-1.0};
    double protonDedxMax{-1.0};
  };

  struct PromptSelection
  {
    std::string name;
    std::string description;

    double minDaughterPt;
    unsigned int minNtpcClusters;
    double maxQuality;
    double maxAbsTrackDcaXY;
    double maxAbsTrackDcaZ;
    double maxAbsPrimaryPcaZ;
    double maxAbsDeltaPrimaryPcaZ;
    double maxAbsPromptPairDCA;

    // Negative bounds disable the corresponding dE/dx side.
    double pionDedxMin{-1.0};
    double pionDedxMax{-1.0};
    double kaonDedxMin{-1.0};
    double kaonDedxMax{-1.0};
  };

  struct HistSet
  {
    TH1F* h_k0s_mass = nullptr;
    TH1F* h_lambda_mass = nullptr;
    TH1F* h_antilambda_mass = nullptr;
    TH1F* h_phi_mass = nullptr;
    TH1F* h_d0_mass = nullptr;
    TH1F* h_antid0_mass = nullptr;
    TH1F* h_jpsi_mass = nullptr;

    TH2F* h_k0s_mass_vs_v0pt = nullptr;
    TH2F* h_lambda_mass_vs_v0pt = nullptr;
    TH2F* h_antilambda_mass_vs_v0pt = nullptr;
    TH2F* h_phi_mass_vs_v0pt = nullptr;
    TH2F* h_d0_mass_vs_v0pt = nullptr;
    TH2F* h_antid0_mass_vs_v0pt = nullptr;
    TH2F* h_jpsi_mass_vs_v0pt = nullptr;
    TH2F* h_armenteros_podolanski = nullptr;
    TH2F* h_pair_dca_vs_delta_pca_z = nullptr;

    // Prompt-primary QA. Filled only in the promptMesons directories.
    TH1F* h_primary_pca_dz = nullptr;
    TH2F* h_primary_pca_z1_vs_z2 = nullptr;
    TH2F* h_max_dca_xy_vs_primary_pca_dz = nullptr;

    // Compact 3D histograms.
    TH3F* h3_k0s = nullptr;
    TH3F* h3_lambda = nullptr;
    TH3F* h3_antilambda = nullptr;
    TH3F* h3_phi = nullptr;
    TH3F* h3_d0 = nullptr;
    TH3F* h3_antid0 = nullptr;
    TH3F* h3_jpsi = nullptr;

    TH3F* h3_k0s_crossing_ptv0 = nullptr;
    TH3F* h3_lambda_crossing_ptv0 = nullptr;
    TH3F* h3_antilambda_crossing_ptv0 = nullptr;

    // Daughter pT correlations versus K0S invariant mass.
    TH3F* h3_k0s_pt1_vs_pt2_vs_mass = nullptr;
    TH3F* h3_k0s_ptplus_vs_ptminus_vs_mass = nullptr;
    // momentum-scale study (energy loss + charge-odd curvature bias per TPC side / eta / phi):
    // pT(pi+) vs pT(pi-) vs mass, variable pT bins, unlike sign only
    //   side_eta[k]: both daughters in the same TPC side (z at r = 50 cm) and |eta| class,
    //                k = 3 * side + etaClass (side 0: z < 0, 1: z >= 0; |eta| < 0.4, 0.4-0.8, >= 0.8),
    //                k = 6: daughters in different classes
    //   side_phi[k]: both daughters in the same TPC side, K0S phi quadrant q (0: -pi..-pi/2, ...,
    //                3: pi/2..pi), k = 4 * side + q; k = 8: daughters in different sides
    TH3F* h3_k0s_ptplus_vs_ptminus_vs_mass_side_eta[7] = {};
    TH3F* h3_k0s_ptplus_vs_ptminus_vs_mass_side_phi[9] = {};

    // Si-scale pair DCA: pT vs mass vs |pair DCA| (0-0.6 cm)
    TH3F* h3_k0s_fine_pairDCA = nullptr;
    TH3F* h3_lambda_fine_pairDCA = nullptr;
    TH3F* h3_phi_fine_pairDCA = nullptr;

    // dE/dx vs total momentum of the daughters, peak and sidebands (sideband subtraction gives
    // clean bands: pions from K0S, protons + pions from Lambda, kaons from phi)
    TH2F* h_dedx_vs_p_k0s_peak = nullptr;
    TH2F* h_dedx_vs_p_k0s_side = nullptr;
    TH2F* h_dedx_vs_p_lambda_proton_peak = nullptr;
    TH2F* h_dedx_vs_p_lambda_proton_side = nullptr;
    TH2F* h_dedx_vs_p_lambda_pion_peak = nullptr;
    TH2F* h_dedx_vs_p_lambda_pion_side = nullptr;
    TH2F* h_dedx_vs_p_phi_peak = nullptr;
    TH2F* h_dedx_vs_p_phi_side = nullptr;

    // Prompt-track resolution QA (each daughter of a prompt pair; a track enters once per pair)
    TH2F* h_dca_xy_vs_pt = nullptr;
    TH2F* h_dca_z_vs_pt = nullptr;
    TH2F* h_dpt_constrained_vs_pt = nullptr;
  };
  constexpr double kTwoPi = 2.0 * TMath::Pi();
  constexpr double kPionMass = 0.13957039;
  constexpr double kKaonMass = 0.493677;
  constexpr double kElectronMass = 0.00051099895;

  // Must match TpcV0CandidateTree::CandidateMask.
  constexpr UInt_t kCandidateKShort = 1U << 0;
  constexpr UInt_t kCandidateLambda = 1U << 1;
  constexpr UInt_t kCandidateAntiLambda = 1U << 2;
  constexpr UInt_t kCandidatePhi = 1U << 3;
  constexpr UInt_t kCandidateD0 = 1U << 4;
  constexpr UInt_t kCandidateAntiD0 = 1U << 5;

  // dE/dx axis (TPC_POLYTRACKS get_dedx units) and the PID-QA mass windows [GeV]
  constexpr double kDedxAxisMax = 2000.0;
  constexpr double kK0sPeakMin = 0.485, kK0sPeakMax = 0.510;
  constexpr double kLambdaPeakMin = 1.108, kLambdaPeakMax = 1.124;
  constexpr double kPhiPeakMin = 1.010, kPhiPeakMax = 1.030;

  enum class MassRegion
  {
    Peak,
    Sideband,
    Outside
  };

  MassRegion massRegion(const double mass, const double peakMin, const double peakMax)
  {
    const double width = peakMax - peakMin;
    if (mass >= peakMin && mass < peakMax)
    {
      return MassRegion::Peak;
    }
    // two sidebands of the peak width, one peak width away from it
    if ((mass >= peakMin - 2.0 * width && mass < peakMin - width) ||
        (mass >= peakMax + width && mass < peakMax + 2.0 * width))
    {
      return MassRegion::Sideband;
    }
    return MassRegion::Outside;
  }

  double invariantMass(const double px1,
                       const double py1,
                       const double pz1,
                       const double mass1,
                       const double px2,
                       const double py2,
                       const double pz2,
                       const double mass2)
  {
    const double p1Squared = px1 * px1 + py1 * py1 + pz1 * pz1;
    const double p2Squared = px2 * px2 + py2 * py2 + pz2 * pz2;

    const double energy1 = std::sqrt(p1Squared + mass1 * mass1);
    const double energy2 = std::sqrt(p2Squared + mass2 * mass2);

    const double totalPx = px1 + px2;
    const double totalPy = py1 + py2;
    const double totalPz = pz1 + pz2;
    const double totalEnergy = energy1 + energy2;

    const double massSquared =
      totalEnergy * totalEnergy -
      totalPx * totalPx -
      totalPy * totalPy -
      totalPz * totalPz;

    return std::sqrt(std::max(0.0, massSquared));
  }

  double wrapPhi(double phi)
  {
    while (phi >= TMath::Pi()) phi -= kTwoPi;
    while (phi < -TMath::Pi()) phi += kTwoPi;
    return phi;
  }

  bool branchExists(TChain& chain, const char* name)
  {
    return chain.GetBranch(name) != nullptr;
  }

  bool passesDedx(const double value,
                  const double minValue,
                  const double maxValue)
  {
    return
      (minValue < 0.0 || value > minValue) &&
      (maxValue < 0.0 || value < maxValue);
  }

  std::string chargeName(const ChargeCategory category)
  {
    switch (category)
    {
      case ChargeCategory::Unlike: return "unlike";
      case ChargeCategory::Like: return "like";
    }
    return "unknown";
  }

  std::vector<ChargeCategory> chargeCategories(const double charge1,
                                               const double charge2)
  {
    std::vector<ChargeCategory> result;

    if (charge1 * charge2 < 0.)
    {
      result.push_back(ChargeCategory::Unlike);
    }
    else if (charge1 * charge2 > 0.)
    {
      result.push_back(ChargeCategory::Like);

      //if (charge1 > 0. && charge2 > 0.)
      //{
      //  result.push_back(ChargeCategory::PlusPlus);
      //}
      //else if (charge1 < 0. && charge2 < 0.)
      //{
      //  result.push_back(ChargeCategory::MinusMinus);
      //}
    }

    return result;
  }

  HistSet bookHistograms(TDirectory* directory,
                         const Selection& selection,
                         const std::string& charge)
  {
    directory->cd();

    const TString tag = TString::Format(
      " [%s, %s]", selection.name.c_str(), charge.c_str());

    HistSet h;

    h.h_k0s_mass = new TH1F(
      "h_mass_Kshort",
      "K^{0}_{S} invariant mass" + tag +
        ";m_{#pi^{+}#pi^{-}} [GeV/c^{2}];pairs",
      500, 0.0, 1.0);

    h.h_lambda_mass = new TH1F(
      "h_mass_Lambda",
      "#Lambda invariant mass, p_{T}(p)>p_{T}(#pi)" + tag +
        ";m_{p#pi^{-}} [GeV/c^{2}];pairs",
      300, 1.0, 1.3);

    h.h_antilambda_mass = new TH1F(
      "h_mass_AntiLambda",
      "#bar{#Lambda} invariant mass, p_{T}(#bar{p})>p_{T}(#pi)" + tag +
        ";m_{#bar{p}#pi^{+}} [GeV/c^{2}];pairs",
      300, 1.0, 1.3);

    h.h_phi_mass = new TH1F(
      "h_mass_Phi",
      "#phi invariant mass" + tag +
        ";m_{K^{+}K^{-}} [GeV/c^{2}];pairs",
      300, 0.95, 1.10);

    h.h_d0_mass = new TH1F(
      "h_mass_D0",
      "D^{0} invariant mass" + tag +
        ";m_{K^{-}#pi^{+}} [GeV/c^{2}];pairs",
      350, 1.65, 2.10);

    h.h_antid0_mass = new TH1F(
      "h_mass_AntiD0",
      "#bar{D}^{0} invariant mass" + tag +
        ";m_{K^{+}#pi^{-}} [GeV/c^{2}];pairs",
      350, 1.65, 2.10);

    h.h_jpsi_mass = new TH1F(
      "h_mass_Jpsi",
      "J/#psi #rightarrow e^{+}e^{-} invariant mass" + tag +
        ";m_{e^{+}e^{-}} [GeV/c^{2}];pairs",
      600, 2.0, 4.0);

    h.h_k0s_mass_vs_v0pt = new TH2F(
      "h_mass_Kshort_vs_v0_pt",
      "K^{0}_{S} mass vs V0 p_{T}" + tag +
        ";p_{T}^{V0} [GeV/c];m_{#pi^{+}#pi^{-}} [GeV/c^{2}]",
      100,0.0,10.0,
      500, 0.0, 1.0);

    h.h_lambda_mass_vs_v0pt = new TH2F(
      "h_mass_Lambda_vs_v0_pt",
      "#Lambda mass vs V0 p_{T}, p_{T}(p)>p_{T}(#pi)" + tag +
        ";p_{T}^{V0} [GeV/c];m_{p#pi^{-}} [GeV/c^{2}]",
      100,0.0,10.0,
      300, 1.0, 1.3);

    h.h_antilambda_mass_vs_v0pt = new TH2F(
      "h_mass_AntiLambda_vs_v0_pt",
      "#bar{#Lambda} mass vs V0 p_{T}, p_{T}(#bar{p})>p_{T}(#pi)" + tag +
        ";p_{T}^{V0} [GeV/c];m_{#bar{p}#pi^{+}} [GeV/c^{2}]",
      100,0.0,10.0,
      300, 1.0, 1.3);

    h.h_phi_mass_vs_v0pt = new TH2F(
      "h_mass_Phi_vs_v0_pt",
      "#phi mass vs pair p_{T}" + tag +
        ";p_{T}^{pair} [GeV/c];m_{K^{+}K^{-}} [GeV/c^{2}]",
      100,0.0,10.0,
      300, 0.95, 1.10);

    h.h_d0_mass_vs_v0pt = new TH2F(
      "h_mass_D0_vs_v0_pt",
      "D^{0} mass vs pair p_{T}" + tag +
        ";p_{T}^{pair} [GeV/c];m_{K^{-}#pi^{+}} [GeV/c^{2}]",
      100,0.0,10.0,
      350, 1.65, 2.10);

    h.h_antid0_mass_vs_v0pt = new TH2F(
      "h_mass_AntiD0_vs_v0_pt",
      "#bar{D}^{0} mass vs pair p_{T}" + tag +
        ";p_{T}^{pair} [GeV/c];m_{K^{+}#pi^{-}} [GeV/c^{2}]",
      100,0.0,10.0,
      350, 1.65, 2.10);

    h.h_jpsi_mass_vs_v0pt = new TH2F(
      "h_mass_Jpsi_vs_pair_pt",
      "J/#psi #rightarrow e^{+}e^{-} mass vs pair p_{T}" + tag +
        ";p_{T}^{pair} [GeV/c];m_{e^{+}e^{-}} [GeV/c^{2}]",
      60, 0., 12.,
      600, 2.0, 4.0);

    h.h_armenteros_podolanski = new TH2F(
      "h_armenteros_podolanski",
      "Armenteros-Podolanski" + tag +
        ";#alpha;q_{T} [GeV/c]",
      240, -1.2, 1.2,
      240, 0.0, 0.40);

    h.h_pair_dca_vs_delta_pca_z = new TH2F(
      "h_pair_dca_vs_abs_delta_pca_z",
      "Pair DCA vs |PCA_{z,1}-PCA_{z,2}|" + tag +
        ";|PCA_{z,1}-PCA_{z,2}| [cm];|pair DCA| [cm]",
      240, 0.0, 2.4,
      240, 0.0, 2.4);

    h.h_primary_pca_dz = new TH1F(
      "h_primary_pca_dz",
      "Primary beam-axis PCA z difference" + tag +
        ";|z_{PCA,1}^{primary}-z_{PCA,2}^{primary}| [cm];pairs",
      240, 0.0, 2.4);

    h.h_primary_pca_z1_vs_z2 = new TH2F(
      "h_primary_pca_z1_vs_z2",
      "Primary beam-axis PCA z correlation" + tag +
        ";z_{PCA,1}^{primary} [cm];z_{PCA,2}^{primary} [cm]",
      160, -40.0, 40.0,
      160, -40.0, 40.0);

    h.h_max_dca_xy_vs_primary_pca_dz = new TH2F(
      "h_max_dca_xy_vs_primary_pca_dz",
      "Track DCA_{xy} vs primary PCA z difference" + tag +
        ";|z_{PCA,1}^{primary}-z_{PCA,2}^{primary}| [cm];max |DCA_{xy}| [cm]",
      240, 0.0, 2.4,
      200, 0.0, 5.0);

    // Compact 3D binning to control output size:
    //   30 bins in V0 pT, 80 bins in mass, 20 bins in |pairDCA|.
    h.h3_k0s = new TH3F(
      "h3_mass_Kshort_vs_v0_pt_vs_pairDCA",
      "K^{0}_{S}: p_{T} vs mass vs |pair DCA|" + tag +
        ";p_{T}^{V0} [GeV/c];m_{#pi^{+}#pi^{-}} [GeV/c^{2}];|pair DCA| [cm]",
      30, 0.0, 5.0,
      80, 0.30, 0.70,
      20, 0.0, 4.0);

    h.h3_k0s_pt1_vs_pt2_vs_mass = new TH3F(
      "h3_Kshort_pt1_vs_pt2_vs_mass",
      "K^{0}_{S}: daughter p_{T,1} vs p_{T,2} vs mass" + tag +
        ";p_{T,1}^{#pi} [GeV/c];p_{T,2}^{#pi} [GeV/c];"
        "m_{#pi^{+}#pi^{-}} [GeV/c^{2}]",
      100,0.0,10.00,
      100,0.0,10.00,
      40, 0.40, 0.60);

    h.h3_k0s_ptplus_vs_ptminus_vs_mass = new TH3F(
      "h3_Kshort_ptplus_vs_ptminus_vs_mass",
      "K^{0}_{S}: p_{T}^{#pi^{+}} vs p_{T}^{#pi^{-}} vs mass" + tag +
        ";p_{T}^{#pi^{+}} [GeV/c];p_{T}^{#pi^{-}} [GeV/c];"
        "m_{#pi^{+}#pi^{-}} [GeV/c^{2}]",
      100,0.0,10.00,
      100,0.0,10.00,
      40, 0.40, 0.60);

    {
      // momentum-scale study, see HistSet
      static const double kPtEdges[] = {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75,
                                        0.90, 1.10, 1.40, 2.00, 3.00};
      const int nPt = static_cast<int>(sizeof(kPtEdges) / sizeof(kPtEdges[0])) - 1;
      const int nMass = 80;
      double massEdges[nMass + 1];
      for (int i = 0; i <= nMass; ++i)
      {
        massEdges[i] = 0.40 + 0.20 * i / nMass;
      }
      const char* sideName[2] = {"zneg", "zpos"};
      const char* sideTitle[2] = {"z<0", "z#geq0"};
      const char* etaName[3] = {"eta0", "eta1", "eta2"};
      const char* etaTitle[3] = {"|#eta|<0.4", "0.4#leq|#eta|<0.8", "|#eta|#geq0.8"};
      const char* axes = ";p_{T}^{#pi^{+}} [GeV/c];p_{T}^{#pi^{-}} [GeV/c];m_{#pi^{+}#pi^{-}} [GeV/c^{2}]";
      for (int k = 0; k < 7; ++k)
      {
        const TString name = k < 6 ? TString::Format("h3_Kshort_ptplus_vs_ptminus_vs_mass_%s_%s", sideName[k / 3], etaName[k % 3])
                                   : TString("h3_Kshort_ptplus_vs_ptminus_vs_mass_sideeta_mixed");
        const TString title = k < 6 ? TString::Format("K^{0}_{S} p_{T}^{+} vs p_{T}^{-} vs mass, both daughters %s, %s", sideTitle[k / 3], etaTitle[k % 3])
                                    : TString("K^{0}_{S} p_{T}^{+} vs p_{T}^{-} vs mass, daughters in different side/#eta classes");
        h.h3_k0s_ptplus_vs_ptminus_vs_mass_side_eta[k] =
          new TH3F(name, title + tag + axes, nPt, kPtEdges, nPt, kPtEdges, nMass, massEdges);
      }
      for (int k = 0; k < 9; ++k)
      {
        const TString name = k < 8 ? TString::Format("h3_Kshort_ptplus_vs_ptminus_vs_mass_%s_phi%d", sideName[k / 4], k % 4)
                                   : TString("h3_Kshort_ptplus_vs_ptminus_vs_mass_sidephi_mixed");
        const TString title = k < 8 ? TString::Format("K^{0}_{S} p_{T}^{+} vs p_{T}^{-} vs mass, both daughters %s, #phi_{V0} in [%.2f, %.2f)", sideTitle[k / 4], -M_PI + (k % 4) * M_PI / 2, -M_PI + (k % 4 + 1) * M_PI / 2)
                                    : TString("K^{0}_{S} p_{T}^{+} vs p_{T}^{-} vs mass, daughters in different TPC sides");
        h.h3_k0s_ptplus_vs_ptminus_vs_mass_side_phi[k] =
          new TH3F(name, title + tag + axes, nPt, kPtEdges, nPt, kPtEdges, nMass, massEdges);
      }
    }

    h.h3_lambda = new TH3F(
      "h3_mass_Lambda_vs_v0_pt_vs_pairDCA",
      "#Lambda: p_{T} vs mass vs |pair DCA|" + tag +
        ";p_{T}^{V0} [GeV/c];m_{p#pi^{-}} [GeV/c^{2}];|pair DCA| [cm]",
      30, 0.0, 5.0,
      80, 1.05, 1.25,
      20, 0.0, 4.0);

    h.h3_antilambda = new TH3F(
      "h3_mass_AntiLambda_vs_v0_pt_vs_pairDCA",
      "#bar{#Lambda}: p_{T} vs mass vs |pair DCA|" + tag +
        ";p_{T}^{V0} [GeV/c];m_{#bar{p}#pi^{+}} [GeV/c^{2}];|pair DCA| [cm]",
      30, 0.0, 5.0,
      80, 1.05, 1.25,
      20, 0.0, 4.0);

    h.h3_phi = new TH3F(
      "h3_mass_Phi_vs_v0_pt_vs_pairDCA",
      "#phi: p_{T} vs mass vs |pair DCA|" + tag +
        ";p_{T}^{pair} [GeV/c];m_{K^{+}K^{-}} [GeV/c^{2}];|pair DCA| [cm]",
      30, 0.0, 5.0,
      80, 0.98, 1.06,
      20, 0.0, 2.0);

    h.h3_d0 = new TH3F(
      "h3_mass_D0_vs_v0_pt_vs_pairDCA",
      "D^{0}: p_{T} vs mass vs |pair DCA|" + tag +
        ";p_{T}^{pair} [GeV/c];m_{K^{-}#pi^{+}} [GeV/c^{2}];|pair DCA| [cm]",
      30, 0.0, 5.0,
      90, 1.70, 2.05,
      20, 0.0, 2.0);

    h.h3_antid0 = new TH3F(
      "h3_mass_AntiD0_vs_v0_pt_vs_pairDCA",
      "#bar{D}^{0}: p_{T} vs mass vs |pair DCA|" + tag +
        ";p_{T}^{pair} [GeV/c];m_{K^{+}#pi^{-}} [GeV/c^{2}];|pair DCA| [cm]",
      30, 0.0, 5.0,
      90, 1.70, 2.05,
      20, 0.0, 2.0);

    h.h3_jpsi = new TH3F(
      "h3_mass_Jpsi_vs_pair_pt_vs_pairDCA",
      "J/#psi #rightarrow e^{+}e^{-}: p_{T} vs mass vs |pair DCA|" + tag +
        ";p_{T}^{pair} [GeV/c];m_{e^{+}e^{-}} [GeV/c^{2}];|pair DCA| [cm]",
      60, 0.0, 12.0,
      100, 2.0, 4.0,
      20, 0.0, 2.0);

    h.h3_k0s_crossing_ptv0 = new TH3F(
      "h3_Kshort_crossing_pt_vs_v0_pt_vs_mass",
      "K^{0}_{S}: crossing p_{T} vs V0 p_{T} vs mass" + tag +
        ";crossing;p_{T}^{V0} [GeV/c];m_{#pi^{+}#pi^{-}} [GeV/c^{2}]",
      600, -120, 480,
      10, 0.5, 5.5,
      100, 0.4, 0.6);

    h.h3_lambda_crossing_ptv0 = new TH3F(
      "h3_Lambda_crossing_pt_vs_v0_pt_vs_mass",
      "#Lambda: crossing p_{T} vs V0 p_{T} vs mass" + tag +
        ";crossing;p_{T}^{V0} [GeV/c];m_{p#pi^{-}} [GeV/c^{2}]",
      600, -120, 480,
      10, 0.5, 5.5,
      100, 1.05, 1.25);
    h.h3_antilambda_crossing_ptv0 = new TH3F(
      "h3_AntiLambda_crossing_pt_vs_v0_pt_vs_mass",
      "#bar{#Lambda}: crossing p_{T} vs V0 p_{T} vs mass" + tag +
        ";crossing;p_{T}^{V0} [GeV/c];m_{#bar{p}#pi^{+}} [GeV/c^{2}]",
      600, -120, 480,
      10, 0.5, 5.5,
      100, 1.05, 1.25);

    h.h3_k0s_fine_pairDCA = new TH3F(
      "h3_mass_Kshort_vs_v0_pt_vs_pairDCA_fine",
      "K^{0}_{S}: p_{T} vs mass vs |pair DCA|, Si scale" + tag +
        ";p_{T}^{V0} [GeV/c];m_{#pi^{+}#pi^{-}} [GeV/c^{2}];|pair DCA| [cm]",
      20, 0.0, 4.0,
      80, 0.40, 0.60,
      60, 0.0, 0.6);

    h.h3_lambda_fine_pairDCA = new TH3F(
      "h3_mass_Lambda_vs_v0_pt_vs_pairDCA_fine",
      "#Lambda + #bar{#Lambda}: p_{T} vs mass vs |pair DCA|, Si scale" + tag +
        ";p_{T}^{V0} [GeV/c];m_{p#pi} [GeV/c^{2}];|pair DCA| [cm]",
      20, 0.0, 4.0,
      80, 1.08, 1.16,
      60, 0.0, 0.6);

    h.h3_phi_fine_pairDCA = new TH3F(
      "h3_mass_Phi_vs_pair_pt_vs_pairDCA_fine",
      "#phi: p_{T} vs mass vs |prompt pair DCA|, Si scale" + tag +
        ";p_{T}^{pair} [GeV/c];m_{K^{+}K^{-}} [GeV/c^{2}];|pair DCA| [cm]",
      20, 0.0, 4.0,
      50, 0.98, 1.08,
      60, 0.0, 0.6);

    auto bookDedx = [&tag](const char* name, const char* what)
    {
      return new TH2F(
        name,
        TString::Format("dE/dx vs p, %s", what) + tag +
          ";p [GeV/c];dE/dx [arb.]",
        100, 0.0, 2.0,
        200, 0.0, kDedxAxisMax);
    };
    h.h_dedx_vs_p_k0s_peak = bookDedx("h_dedx_vs_p_k0s_peak", "K^{0}_{S} peak, both daughters");
    h.h_dedx_vs_p_k0s_side = bookDedx("h_dedx_vs_p_k0s_side", "K^{0}_{S} sidebands, both daughters");
    h.h_dedx_vs_p_lambda_proton_peak = bookDedx("h_dedx_vs_p_lambda_proton_peak", "#Lambda+#bar{#Lambda} peak, (anti)proton");
    h.h_dedx_vs_p_lambda_proton_side = bookDedx("h_dedx_vs_p_lambda_proton_side", "#Lambda+#bar{#Lambda} sidebands, (anti)proton");
    h.h_dedx_vs_p_lambda_pion_peak = bookDedx("h_dedx_vs_p_lambda_pion_peak", "#Lambda+#bar{#Lambda} peak, pion");
    h.h_dedx_vs_p_lambda_pion_side = bookDedx("h_dedx_vs_p_lambda_pion_side", "#Lambda+#bar{#Lambda} sidebands, pion");
    h.h_dedx_vs_p_phi_peak = bookDedx("h_dedx_vs_p_phi_peak", "#phi 1.010-1.030, both daughters");
    h.h_dedx_vs_p_phi_side = bookDedx("h_dedx_vs_p_phi_side", "#phi sidebands, both daughters");

    h.h_dca_xy_vs_pt = new TH2F(
      "h_dca_xy_vs_pt",
      "prompt-pair daughters: DCA_{xy} to the vertex vs p_{T}" + tag +
        ";p_{T} [GeV/c];DCA_{xy} [cm]",
      40, 0.0, 4.0,
      400, -0.5, 0.5);

    h.h_dca_z_vs_pt = new TH2F(
      "h_dca_z_vs_pt",
      "prompt-pair daughters: DCA_{z} to the vertex vs p_{T}" + tag +
        ";p_{T} [GeV/c];DCA_{z} [cm]",
      40, 0.0, 4.0,
      400, -1.0, 1.0);

    h.h_dpt_constrained_vs_pt = new TH2F(
      "h_dpt_constrained_vs_pt",
      "prompt-pair daughters: (p_{T}^{constrained} - p_{T}) / p_{T} vs p_{T}" + tag +
        ";p_{T} [GeV/c];#Deltap_{T}/p_{T}",
      40, 0.0, 4.0,
      200, -0.2, 0.2);

    return h;
  }

  bool passesSelection(const Selection& selection,
                       const double pcaZ,
                       const double absDeltaPcaZ,
                       const double daughterPtMin,
                       const double decayRadius,
                       const double absAlpha,
                       const double absPairDCA,
                       const double dira,
                       const double qualityMax,
                       const int npointsMin)
  {
    return
      std::abs(pcaZ) < selection.maxAbsPcaZ &&
      absDeltaPcaZ < selection.maxAbsDeltaPcaZ &&
      daughterPtMin > selection.minDaughterPt &&
      decayRadius > selection.minDecayRadius &&
      absAlpha < selection.maxAbsAlpha &&
      absPairDCA < selection.maxAbsPairDCA &&
      dira > selection.minDIRA &&
      // maxQuality >= 1e8: quality not used (Kalman chi2/ndf can be NaN for very short fits)
      (selection.maxQuality >= 1.0e8 || qualityMax < selection.maxQuality) &&
      npointsMin > selection.minnpoints;
  }


  bool passesPromptSelection(const PromptSelection& selection,
                             const double daughterPtMin,
                             const unsigned int ntpcClustersMin,
                             const double qualityMax,
                             const double maxAbsTrackDcaXY,
                             const double maxAbsTrackDcaZ,
                             const double maxAbsPrimaryPcaZ,
                             const double absDeltaPrimaryPcaZ,
                             const double absPromptPairDCA)
  {
    return
      daughterPtMin > selection.minDaughterPt &&
      ntpcClustersMin >= selection.minNtpcClusters &&
      (selection.maxQuality < 0.0 ||
       qualityMax < selection.maxQuality) &&
      (selection.maxAbsTrackDcaXY < 0.0 ||
       maxAbsTrackDcaXY < selection.maxAbsTrackDcaXY) &&
      (selection.maxAbsTrackDcaZ < 0.0 ||
       maxAbsTrackDcaZ < selection.maxAbsTrackDcaZ) &&
      (selection.maxAbsPrimaryPcaZ < 0.0 ||
       maxAbsPrimaryPcaZ < selection.maxAbsPrimaryPcaZ) &&
      (selection.maxAbsDeltaPrimaryPcaZ < 0.0 ||
       absDeltaPrimaryPcaZ < selection.maxAbsDeltaPrimaryPcaZ) &&
      (selection.maxAbsPromptPairDCA < 0.0 ||
       (std::isfinite(absPromptPairDCA) &&
        absPromptPairDCA < selection.maxAbsPromptPairDCA));
  }
}

void MakeK0sPairHistograms(
  const char* inputDir = "/sphenix/tg/tg01/hf/mitrankova/Hadrons/",
  const char* filePattern = "/hadron_sitpc_HITS_pp_mvtxS300_79513_*.root",
  const char* outputDir = "output",
  const char* outputName = "v0_pair_histograms_mvtxS300.root",
  const char* treeName = "pairTree",
  const bool requireProtonHigherPt = true,
  const bool ScaleQualityBy10 = false,
  const Long64_t maxEntries = -1,
  const bool usePrimaryVertexKinematicsForPrompt = true,
  // beam in the Si+TPC frame (beam alignment centres it); only used without pv_x/pv_y/pv_z
  const double beamX = 0.0,
  const double beamY = 0.0,
  const double beamZ = 0.0,
  const char* likeSignTreeName = "likeSignPairTree",
  const bool includeLikeSignTree = true,
  const bool includePrimaryConstrainedPrompt = true,
  // streaming Si+TPC: crossing 0 is an ordinary crossing
  const bool ignoreCrossingZero = false,
  // "all", "tpconly", "si" (both >= minSiHits Si points), "mixed"
  const char* trackClass = "all",
  const int minSiHits = 4,
  // signed (sailor/cowboy) Delta-phi cut for K0S/Lambda
  const bool applySignedDeltaPhiCut = false,
  // collision vertex of the pair (pv_x, pv_y, pv_z) instead of beamX/Y/Z when present
  const bool usePairVertex = true)
{
  TH1::AddDirectory(kTRUE);

  std::cout << "MakeK0sPairHistograms: inputDir=" << inputDir
            << ", filePattern=" << filePattern
            << ", outputDir=" << outputDir
            << ", outputName=" << outputName
            << ", treeName=" << treeName
            << ", requireProtonHigherPt=" << requireProtonHigherPt
            << ", ScaleQualityBy10=" << ScaleQualityBy10
            << ", maxEntries=" << maxEntries
            << ", usePrimaryVertexKinematicsForPrompt="
            << usePrimaryVertexKinematicsForPrompt
            << ", likeSignTreeName=" << likeSignTreeName
            << ", includeLikeSignTree=" << includeLikeSignTree
            << ", includePrimaryConstrainedPrompt="
            << includePrimaryConstrainedPrompt
            << ", ignoreCrossingZero=" << ignoreCrossingZero
            << ", trackClass=" << trackClass
            << ", minSiHits=" << minSiHits
            << ", applySignedDeltaPhiCut=" << applySignedDeltaPhiCut
            << ", usePairVertex=" << usePairVertex
            << std::endl;

  const std::string trackClassName = trackClass ? trackClass : "all";
  if (trackClassName != "all" && trackClassName != "tpconly" &&
      trackClassName != "si" && trackClassName != "mixed")
  {
    std::cerr << "ERROR: unknown trackClass " << trackClassName
              << " (all, tpconly, si, mixed)" << std::endl;
    return;
  }

  const TString chainPattern =
    TString::Format("%s/%s", inputDir, filePattern);

  // Build one logical chain containing unlike-sign pairTree entries followed
  // by like-sign entries from the separate likeSignPairTree.  TChain supports
  // a different tree name for each added file through AddFile(..., tname).
  TChain chain(treeName);
  const int nUnlikeFiles = chain.Add(chainPattern);

  if (nUnlikeFiles <= 0)
  {
    std::cerr << "ERROR: no files matched "
              << chainPattern << std::endl;
    return;
  }

  std::vector<std::string> inputFiles;
  if (TObjArray* fileList = chain.GetListOfFiles())
  {
    inputFiles.reserve(fileList->GetEntries());

    for (int index = 0; index < fileList->GetEntries(); ++index)
    {
      auto* element =
        dynamic_cast<TChainElement*>(fileList->At(index));

      if (element != nullptr)
      {
        inputFiles.emplace_back(element->GetTitle());
      }
    }
  }

  const Long64_t nUnlikeEntries =
    chain.GetEntries();

  int nLikeSignFiles = 0;

  if (includeLikeSignTree)
  {
    for (const auto& filename : inputFiles)
    {
      std::unique_ptr<TFile> inputFile(
        TFile::Open(filename.c_str(), "READ"));

      if (!inputFile || inputFile->IsZombie())
      {
        std::cerr
          << "WARNING: could not inspect " << filename
          << " for tree " << likeSignTreeName
          << std::endl;
        continue;
      }

      if (inputFile->Get(likeSignTreeName) == nullptr)
      {
        std::cerr
          << "WARNING: " << filename
          << " does not contain " << likeSignTreeName
          << std::endl;
        continue;
      }

      chain.AddFile(
        filename.c_str(),
        TTree::kMaxEntries,
        likeSignTreeName);

      ++nLikeSignFiles;
    }
  }

  const int nFiles = nUnlikeFiles + nLikeSignFiles;

  std::cout
    << "Input trees: pairTree files=" << nUnlikeFiles
    << ", " << likeSignTreeName << " files=" << nLikeSignFiles
    << std::endl;

  const std::vector<std::string> requiredBranches = {
    "mass_Kshort",
    "mass_Lambda",
    "mass_AntiLambda",
    "v0_pt",
    "pca_x",
    "pca_y",
    "pca_z",
    "pca1_z",
    "pca2_z",
    "px1",
    "py1",
    "pz1",
    "px2",
    "py2",
    "pz2",
    "v0_px",
    "v0_py",
    "v0_pz",
    "alpha",
    "qT",
    "pairDCA",
    "Lproj",
    "dca_xy1",
    "dca_z1",
    "dca_xy2",
    "dca_z2",
    "npoints1",
    "npoints2",
    "charge1",
    "charge2",
    "quality1",
    "quality2",
    "dedx_1",
    "dedx_2",
    "candidate_mask",
    "mass_P1Pi2",
    "mass_Pi1P2",
    "mass_K1Pi2",
    "mass_Pi1K2"
  };

  bool missingBranch = false;

  for (const auto& name : requiredBranches)
  {
    if (!branchExists(chain, name.c_str()))
    {
      std::cerr << "ERROR: missing branch "
                << name << std::endl;
      missingBranch = true;
    }
  }

  if (usePrimaryVertexKinematicsForPrompt || includePrimaryConstrainedPrompt)
  {
    const std::vector<std::string> requiredPrimaryBranches = {
      "primary_px1",
      "primary_py1",
      "primary_pz1",
      "primary_px2",
      "primary_py2",
      "primary_pz2",
      "primary_pca1_z",
      "primary_pca2_z",
      "primary_pca_dz",
      "primary_pca_valid",
      "prompt_pairDCA",
      "prompt_pca_valid",
      "ntpc_clusters1",
      "ntpc_clusters2"
    };

    for (const auto& name : requiredPrimaryBranches)
    {
      if (!branchExists(chain, name.c_str()))
      {
        std::cerr
          << "ERROR: usePrimaryVertexKinematicsForPrompt=true, "
          << "but branch " << name << " is missing" << std::endl;
        missingBranch = true;
      }
    }
  }

  if (includePrimaryConstrainedPrompt)
  {
    const std::vector<std::string> requiredConstrainedBranches = {
      "primary_constrained_valid1",
      "primary_constrained_valid2",
      "primary_constrained_chi2_1",
      "primary_constrained_chi2_2",
      "primary_constrained_px1",
      "primary_constrained_py1",
      "primary_constrained_pz1",
      "primary_constrained_px2",
      "primary_constrained_py2",
      "primary_constrained_pz2",
      "primary_constrained_pair_pt",
      "mass_Phi_primary_constrained",
      "mass_D0_primary_constrained",
      "mass_AntiD0_primary_constrained",
      "mass_K1Pi2_primary_constrained",
      "mass_Pi1K2_primary_constrained"
    };

    for (const auto& name : requiredConstrainedBranches)
    {
      if (!branchExists(chain, name.c_str()))
      {
        std::cerr
          << "ERROR: includePrimaryConstrainedPrompt=true, "
          << "but branch " << name << " is missing"
          << std::endl;
        missingBranch = true;
      }
    }
  }

  // Si+TPC branches (TpcV0CandidateTree with use_sitpc_tracks)
  const bool haveSiBranches =
    branchExists(chain, "has_si1") && branchExists(chain, "has_si2") &&
    branchExists(chain, "nmvtx1") && branchExists(chain, "nmvtx2") &&
    branchExists(chain, "nintt1") && branchExists(chain, "nintt2");
  const bool havePairVertex =
    usePairVertex &&
    branchExists(chain, "pv_x") && branchExists(chain, "pv_y") && branchExists(chain, "pv_z");
  if (trackClassName != "all" && !haveSiBranches)
  {
    std::cerr << "ERROR: trackClass=" << trackClassName
              << " needs has_si1/2, nmvtx1/2, nintt1/2" << std::endl;
    missingBranch = true;
  }
  std::cout << "Si branches " << (haveSiBranches ? "found" : "absent")
            << ", pair vertex " << (havePairVertex ? "pv_x/pv_y/pv_z" : "beamX/Y/Z")
            << std::endl;

  if (missingBranch)
  {
    return;
  }

  Float_t mass_Kshort = 0.f;
  Float_t mass_Lambda = 0.f;
  Float_t mass_AntiLambda = 0.f;
  Float_t v0_pt = 0.f;

  Float_t pca_x = 0.f;
  Float_t pca_y = 0.f;
  Float_t pca_z = 0.f;
  Float_t pca1_z = 0.f;
  Float_t pca2_z = 0.f;

  Float_t px1 = 0.f;
  Float_t py1 = 0.f;
  Float_t pz1 = 0.f;
  Float_t px2 = 0.f;
  Float_t py2 = 0.f;
  Float_t pz2 = 0.f;

  Float_t primary_px1 = 0.f;
  Float_t primary_py1 = 0.f;
  Float_t primary_pz1 = 0.f;
  Float_t primary_px2 = 0.f;
  Float_t primary_py2 = 0.f;
  Float_t primary_pz2 = 0.f;
  Float_t primary_pca1_z = 0.f;
  Float_t primary_pca2_z = 0.f;
  Float_t primary_pca_dz = 0.f;
  Int_t primary_pca_valid = 0;
  Float_t prompt_pairDCA = 0.f;
  Int_t prompt_pca_valid = 0;

  Int_t primary_constrained_valid1 = 0;
  Int_t primary_constrained_valid2 = 0;
  Float_t primary_constrained_chi2_1 = 0.f;
  Float_t primary_constrained_chi2_2 = 0.f;
  Float_t primary_constrained_px1 = 0.f;
  Float_t primary_constrained_py1 = 0.f;
  Float_t primary_constrained_pz1 = 0.f;
  Float_t primary_constrained_px2 = 0.f;
  Float_t primary_constrained_py2 = 0.f;
  Float_t primary_constrained_pz2 = 0.f;
  Float_t primary_constrained_pair_pt = 0.f;
  Float_t mass_Phi_primary_constrained = 0.f;
  Float_t mass_D0_primary_constrained = 0.f;
  Float_t mass_AntiD0_primary_constrained = 0.f;
  Float_t mass_K1Pi2_primary_constrained = 0.f;
  Float_t mass_Pi1K2_primary_constrained = 0.f;

  Float_t mass_P1Pi2 = 0.f;
  Float_t mass_Pi1P2 = 0.f;
  Float_t mass_K1Pi2 = 0.f;
  Float_t mass_Pi1K2 = 0.f;
  UInt_t candidate_mask = 0U;

  Float_t v0_px = 0.f;
  Float_t v0_py = 0.f;
  Float_t v0_pz = 0.f;

  Float_t alpha = 0.f;
  Float_t qT = 0.f;
  Float_t pairDCA = 0.f;
  Float_t Lproj = 0.f;
  Float_t dca_xy1 = 0.f;
  Float_t dca_z1 = 0.f;
  Float_t dca_xy2 = 0.f;
  Float_t dca_z2 = 0.f;
  Float_t quality1 = 0.f;
  Float_t quality2 = 0.f;
  Float_t dedx_1 = 0.f;
  Float_t dedx_2 = 0.f;

  Float_t charge1 = 0;
  Float_t charge2 = 0;
  Short_t npoints1 = 0;
  Short_t npoints2 = 0;
  UInt_t ntpc_clusters1 = 0;
  UInt_t ntpc_clusters2 = 0;

  Short_t crossing1 = 0;
  Short_t crossing2 = 0;

  Int_t has_si1 = 0;
  Int_t has_si2 = 0;
  UInt_t nmvtx1 = 0;
  UInt_t nmvtx2 = 0;
  UInt_t nintt1 = 0;
  UInt_t nintt2 = 0;
  Float_t pv_x = 0.f;
  Float_t pv_y = 0.f;
  Float_t pv_z = 0.f;
  if (haveSiBranches)
  {
    chain.SetBranchAddress("has_si1", &has_si1);
    chain.SetBranchAddress("has_si2", &has_si2);
    chain.SetBranchAddress("nmvtx1", &nmvtx1);
    chain.SetBranchAddress("nmvtx2", &nmvtx2);
    chain.SetBranchAddress("nintt1", &nintt1);
    chain.SetBranchAddress("nintt2", &nintt2);
  }
  if (havePairVertex)
  {
    chain.SetBranchAddress("pv_x", &pv_x);
    chain.SetBranchAddress("pv_y", &pv_y);
    chain.SetBranchAddress("pv_z", &pv_z);
  }


  chain.SetBranchAddress("mass_Kshort", &mass_Kshort);
  chain.SetBranchAddress("mass_Lambda", &mass_Lambda);
  chain.SetBranchAddress("mass_AntiLambda", &mass_AntiLambda);
  chain.SetBranchAddress("v0_pt", &v0_pt);

  chain.SetBranchAddress("pca_x", &pca_x);
  chain.SetBranchAddress("pca_y", &pca_y);
  chain.SetBranchAddress("pca_z", &pca_z);
  chain.SetBranchAddress("pca1_z", &pca1_z);
  chain.SetBranchAddress("pca2_z", &pca2_z);

  chain.SetBranchAddress("px1", &px1);
  chain.SetBranchAddress("py1", &py1);
  chain.SetBranchAddress("pz1", &pz1);
  chain.SetBranchAddress("px2", &px2);
  chain.SetBranchAddress("py2", &py2);
  chain.SetBranchAddress("pz2", &pz2);

  chain.SetBranchAddress("crossing1", &crossing1);
  chain.SetBranchAddress("crossing2", &crossing2);

  if (usePrimaryVertexKinematicsForPrompt || includePrimaryConstrainedPrompt)
  {
    chain.SetBranchAddress("primary_px1", &primary_px1);
    chain.SetBranchAddress("primary_py1", &primary_py1);
    chain.SetBranchAddress("primary_pz1", &primary_pz1);
    chain.SetBranchAddress("primary_px2", &primary_px2);
    chain.SetBranchAddress("primary_py2", &primary_py2);
    chain.SetBranchAddress("primary_pz2", &primary_pz2);
    chain.SetBranchAddress("primary_pca1_z", &primary_pca1_z);
    chain.SetBranchAddress("primary_pca2_z", &primary_pca2_z);
    chain.SetBranchAddress("primary_pca_dz", &primary_pca_dz);
    chain.SetBranchAddress("primary_pca_valid", &primary_pca_valid);
    chain.SetBranchAddress("prompt_pairDCA", &prompt_pairDCA);
    chain.SetBranchAddress("prompt_pca_valid", &prompt_pca_valid);
    chain.SetBranchAddress("ntpc_clusters1", &ntpc_clusters1);
    chain.SetBranchAddress("ntpc_clusters2", &ntpc_clusters2);
  }

  if (includePrimaryConstrainedPrompt)
  {
    chain.SetBranchAddress(
      "primary_constrained_valid1",
      &primary_constrained_valid1);
    chain.SetBranchAddress(
      "primary_constrained_valid2",
      &primary_constrained_valid2);
    chain.SetBranchAddress(
      "primary_constrained_chi2_1",
      &primary_constrained_chi2_1);
    chain.SetBranchAddress(
      "primary_constrained_chi2_2",
      &primary_constrained_chi2_2);
    chain.SetBranchAddress(
      "primary_constrained_px1",
      &primary_constrained_px1);
    chain.SetBranchAddress(
      "primary_constrained_py1",
      &primary_constrained_py1);
    chain.SetBranchAddress(
      "primary_constrained_pz1",
      &primary_constrained_pz1);
    chain.SetBranchAddress(
      "primary_constrained_px2",
      &primary_constrained_px2);
    chain.SetBranchAddress(
      "primary_constrained_py2",
      &primary_constrained_py2);
    chain.SetBranchAddress(
      "primary_constrained_pz2",
      &primary_constrained_pz2);
    chain.SetBranchAddress(
      "primary_constrained_pair_pt",
      &primary_constrained_pair_pt);
    chain.SetBranchAddress(
      "mass_Phi_primary_constrained",
      &mass_Phi_primary_constrained);
    chain.SetBranchAddress(
      "mass_D0_primary_constrained",
      &mass_D0_primary_constrained);
    chain.SetBranchAddress(
      "mass_AntiD0_primary_constrained",
      &mass_AntiD0_primary_constrained);
    chain.SetBranchAddress(
      "mass_K1Pi2_primary_constrained",
      &mass_K1Pi2_primary_constrained);
    chain.SetBranchAddress(
      "mass_Pi1K2_primary_constrained",
      &mass_Pi1K2_primary_constrained);
  }

  chain.SetBranchAddress("mass_P1Pi2", &mass_P1Pi2);
  chain.SetBranchAddress("mass_Pi1P2", &mass_Pi1P2);
  chain.SetBranchAddress("mass_K1Pi2", &mass_K1Pi2);
  chain.SetBranchAddress("mass_Pi1K2", &mass_Pi1K2);
  chain.SetBranchAddress("candidate_mask", &candidate_mask);

  chain.SetBranchAddress("v0_px", &v0_px);
  chain.SetBranchAddress("v0_py", &v0_py);
  chain.SetBranchAddress("v0_pz", &v0_pz);

  chain.SetBranchAddress("alpha", &alpha);
  chain.SetBranchAddress("qT", &qT);
  chain.SetBranchAddress("pairDCA", &pairDCA);
  chain.SetBranchAddress("Lproj", &Lproj);
  chain.SetBranchAddress("dca_xy1", &dca_xy1);
  chain.SetBranchAddress("dca_z1", &dca_z1);
  chain.SetBranchAddress("dca_xy2", &dca_xy2);
  chain.SetBranchAddress("dca_z2", &dca_z2);

  chain.SetBranchAddress("charge1", &charge1);
  chain.SetBranchAddress("charge2", &charge2);
  chain.SetBranchAddress("quality1", &quality1);
  chain.SetBranchAddress("quality2", &quality2);
  chain.SetBranchAddress("dedx_1", &dedx_1);
  chain.SetBranchAddress("dedx_2", &dedx_2);
  chain.SetBranchAddress("npoints1", &npoints1);
  chain.SetBranchAddress("npoints2", &npoints2);

  // Ten cumulative cut levels, loose to Si scale.
  //
  // Fields: |pca_z|, |pca1_z-pca2_z|, min daughter pT, min decay radius (from the pair's collision
  // vertex), |alpha|, |pairDCA|, DIRA, max quality (fit chi2/ndf), min npoints (Si + TPC points).
  // Distances in cm.  Decay radius starts at 0: Si-matched daughters only exist for V0s decaying
  // inside the MVTX; TPC-only daughters (trackClass "tpconly") need a larger radius to see a
  // clean peak (cut06+).  No sailor/cowboy cut unless applySignedDeltaPhiCut.
  const std::vector<Selection> selections = {
    {
      "cut00_no_topology",
      "|pca_z|<20, pt>0.15, no other cut",
      20.0, 1.0e9, 0.15, -1.0, 1.01, 1.0e9, -3.0, 1.0e9, 0
    },
    {
      "cut01_very_loose",
      "|pca_z|<20, pt>0.15, |dz|<2.0, pairDCA<3.0, DIRA>0.0",
      20.0, 2.00, 0.15, -1.0, 1.01, 3.00, 0.00, 1.0e9, 10
    },
    {
      "cut02_loose",
      "|pca_z|<20, pt>0.15, |dz|<1.0, pairDCA<1.0, DIRA>0.5",
      20.0, 1.00, 0.15, -1.0, 1.01, 1.00, 0.50, 1.0e9, 12
    },
    {
      "cut03_radius_0p05",
      "+ decay radius>0.05, pairDCA<0.5, DIRA>0.8",
      20.0, 0.70, 0.15, 0.05, 1.01, 0.50, 0.80, 1.0e9, 12
    },
    {
      "cut04_radius_0p1",
      "+ decay radius>0.1, pairDCA<0.3, DIRA>0.9",
      20.0, 0.50, 0.15, 0.10, 1.01, 0.30, 0.90, 1.0e9, 12
    },
    {
      "cut05_pairDCA_1mm",
      "+ decay radius>0.2, pairDCA<0.1, DIRA>0.95",
      20.0, 0.30, 0.15, 0.20, 1.01, 0.10, 0.95, 1.0e9, 12
    },
    {
      "cut06_pairDCA_0p5mm",
      "+ decay radius>0.3, pairDCA<0.05, DIRA>0.98",
      20.0, 0.20, 0.15, 0.30, 1.01, 0.05, 0.98, 1.0e9, 12
    },
    {
      "cut07_radius_1cm",
      "decay radius>1.0, pairDCA<0.5, DIRA>0.95 (TPC-only scale)",
      20.0, 0.50, 0.15, 1.00, 1.01, 0.50, 0.95, 1.0e9, 12
    },
    {
      "cut08_radius_2cm",
      "decay radius>2.0, pairDCA<1.0, DIRA>0.9 (previous TPC baseline region)",
      20.0, 0.70, 0.15, 2.00, 1.01, 1.00, 0.90, 1.0e9, 12
    },
    {
      "cut09_quality",
      "cut04 + track chi2/ndf<10, npoints>20",
      20.0, 0.50, 0.15, 0.10, 1.01, 0.30, 0.90, 10.0, 20
    }
  };

  gSystem->mkdir(outputDir, kTRUE);

  const TString outputPath =
    TString::Format("%s/%s", outputDir, outputName);

  std::unique_ptr<TFile> output(
    TFile::Open(outputPath, "RECREATE"));

  if (!output || output->IsZombie())
  {
    std::cerr << "ERROR: cannot create "
              << outputPath << std::endl;
    return;
  }

  output->cd();

  TNamed promptKinematicsInfo(
    "prompt_kinematics",
    usePrimaryVertexKinematicsForPrompt
      ? "daughter momenta at independent transverse PCAs to the configured beam axis"
      : "secondary pair-PCA daughter momenta");
  promptKinematicsInfo.Write();

  TNamed likeSignDeltaPhiInfo(
    "likesign_v0_delta_phi",
    "signed V0 Delta-phi cut is applied only to unlike-sign pairs; "
    "like-sign K0S/Lambda backgrounds are not rejected by that charge-ordered cut");
  likeSignDeltaPhiInfo.Write();

  TNamed constrainedPromptInfo(
    "primary_constrained_prompt_kinematics",
    includePrimaryConstrainedPrompt
      ? "enabled: second prompt-meson histogram set uses the forced primary-vertex-constrained daughter momenta"
      : "disabled");
  constrainedPromptInfo.Write();

  TNamed trackClassInfo(
    "track_class",
    TString::Format("trackClass=%s, minSiHits=%d, signedDeltaPhiCut=%d, pairVertex=%d, ignoreCrossingZero=%d",
                    trackClassName.c_str(), minSiHits, applySignedDeltaPhiCut ? 1 : 0,
                    havePairVertex ? 1 : 0, ignoreCrossingZero ? 1 : 0).Data());
  trackClassInfo.Write();

  TNamed likeSignSourceInfo(
    "like_sign_source",
    includeLikeSignTree
      ? likeSignTreeName
      : "disabled");
  likeSignSourceInfo.Write();

  TH1I* h_cutflow_unlike = new TH1I(
    "h_cutflow_unlike",
    "Unlike-sign cumulative cut flow;selection;pair count",
    selections.size(), 0, selections.size());

  TH1I* h_cutflow_like = new TH1I(
    "h_cutflow_like",
    "Like-sign cumulative cut flow;selection;pair count",
    selections.size(), 0, selections.size());

  for (std::size_t i = 0; i < selections.size(); ++i)
  {
    h_cutflow_unlike->GetXaxis()->SetBinLabel(
      i + 1, selections[i].name.c_str());

    h_cutflow_like->GetXaxis()->SetBinLabel(
      i + 1, selections[i].name.c_str());
  }

  std::map<std::string, std::map<std::string, HistSet>> histograms;

  for (const auto& selection : selections)
  {
    TDirectory* selectionDir =
      output->mkdir(selection.name.c_str());

    selectionDir->cd();

    TNamed selectionInfo(
      "selection",
      selection.description.c_str());
    selectionInfo.Write();

    for (const auto category : {
           ChargeCategory::Unlike,
           ChargeCategory::Like})
    {
      const std::string categoryName =
        chargeName(category);

      TDirectory* chargeDir =
        selectionDir->mkdir(categoryName.c_str());

      histograms[selection.name][categoryName] =
        bookHistograms(chargeDir, selection, categoryName);
    }
  }

  // Separate exact user selections. These do not replace the 10-cut scan.
  const std::vector<std::pair<std::string, std::string>> exactSelections = {
    {
      "exactCut1",
      "-15<pca_z<15, |Delta pca_z|<0.5, daughter pT>0.3, "
      "radius>6, |alpha|<0.8, |pairDCA|<1, DIRA>0.8, "
      "npoints>30, quality<20"
    },
    {
      "exactCut2",
      "-15<pca_z<0, |Delta pca_z|<0.5, daughter pT>0.3, "
      "radius>6, |alpha|<0.8, |pairDCA|<4, pz1/2<0, "
      "DIRA>0.8, npoints>30"
    },
    {
      "exactCut3",
      "-15<pca_z<0, |Delta pca_z|<1.0, daughter pT>0.3, "
      "radius>6, |alpha|<0.8, |pairDCA|<4, pz1/2<0, DIRA>0.8"
    }
  };

  std::map<std::string, std::map<std::string, HistSet>> exactHistograms;

  for (const auto& exactSelection : exactSelections)
  {
    TDirectory* selectionDir =
      output->mkdir(exactSelection.first.c_str());

    selectionDir->cd();

    TNamed selectionInfo(
      "selection",
      exactSelection.second.c_str());
    selectionInfo.Write();

    Selection bookingSelection{
      exactSelection.first, exactSelection.second,
      0., 0., 0., 0., 0., 0., 0., 0., 0
    };

    for (const auto category : {
           ChargeCategory::Unlike,
           ChargeCategory::Like})
    {
      const std::string categoryName = chargeName(category);
      TDirectory* chargeDir =
        selectionDir->mkdir(categoryName.c_str());

      exactHistograms[exactSelection.first][categoryName] =
        bookHistograms(chargeDir, bookingSelection, categoryName);
    }
  }

  // Prompt-meson selections are intentionally separate from the V0 cuts.
  // These candidates are treated as primary within the present DCA resolution.
  const std::vector<PromptSelection> promptSelections = {
    // fields: min pT, min nTPC, max quality, |DCAxy|, |DCAz|, |primary PCA z|, |Delta primary PCA z|,
    //         |prompt pair DCA| (negative = off), then optional pion / kaon dE/dx windows
    {
      "primary_loose",
      "pT>0.15, nTPC>=10, |primary PCA z|<20; nothing else",
      0.15, 10, -1.0, -1.0, -1.0, 20.0, -1.0, -1.0
    },
    {
      "primary_dz_1p0",
      "pT>0.15, nTPC>=10, |DCAxy|<1, |Delta primary PCA z|<1.0",
      0.15, 10, -1.0, 1.0, -1.0, 20.0, 1.0, -1.0
    },
    {
      "primary_dz_0p3",
      "pT>0.15, nTPC>=10, |DCAxy|<0.5, |DCAz|<0.5, |Delta primary PCA z|<0.3",
      0.15, 10, -1.0, 0.5, 0.5, 20.0, 0.3, -1.0
    },
    {
      "primary_dz_0p1",
      "pT>0.15, nTPC>=10, |DCAxy|<0.2, |DCAz|<0.2, |Delta primary PCA z|<0.1 (Si scale)",
      0.15, 10, -1.0, 0.2, 0.2, 20.0, 0.1, -1.0
    },
    {
      "primary_pairDCA_1mm",
      "primary_dz_0p3 + |prompt pair DCA|<0.1",
      0.15, 10, -1.0, 0.5, 0.5, 20.0, 0.3, 0.1
    },
    {
      "primary_dz_0p3_pt0p3",
      "primary_dz_0p3 with pT>0.3, nTPC>=20",
      0.30, 20, -1.0, 0.5, 0.5, 20.0, 0.3, -1.0
    },
    {
      "primary_dz_0p3_pid",
      "primary_dz_0p3 + pion dE/dx<300 and kaon dE/dx>300 (tune from the dE/dx-vs-p histograms)",
      0.15, 10, -1.0, 0.5, 0.5, 20.0, 0.3, -1.0,
      -1.0, 300.0, 300.0, -1.0
    }
  };

  std::map<std::string, std::map<std::string, HistSet>> promptHistograms;

  TDirectory* promptTopDir = output->mkdir("promptMesons");
  for (const auto& selection : promptSelections)
  {
    TDirectory* selectionDir = promptTopDir->mkdir(selection.name.c_str());
    selectionDir->cd();

    TNamed selectionInfo("selection", selection.description.c_str());
    selectionInfo.Write();

    Selection bookingSelection{
      selection.name, selection.description,
      0., 0., 0., 0., 0., 0., 0., 0., 0
    };

    for (const auto category : {
           ChargeCategory::Unlike,
           ChargeCategory::Like})
    {
      const std::string categoryName = chargeName(category);
      TDirectory* chargeDir = selectionDir->mkdir(categoryName.c_str());
      promptHistograms[selection.name][categoryName] =
        bookHistograms(chargeDir, bookingSelection, categoryName);
    }
  }

  std::map<std::string, std::map<std::string, HistSet>>
    promptConstrainedHistograms;

  if (includePrimaryConstrainedPrompt)
  {
    TDirectory* constrainedPromptTopDir =
      output->mkdir("promptMesonsPrimaryConstrained");

    for (const auto& selection : promptSelections)
    {
      TDirectory* selectionDir =
        constrainedPromptTopDir->mkdir(selection.name.c_str());

      selectionDir->cd();

      const std::string constrainedDescription =
        std::string("primary-vertex-constrained momenta; ") +
        selection.description;

      TNamed selectionInfo(
        "selection",
        constrainedDescription.c_str());
      selectionInfo.Write();

      Selection bookingSelection{
        selection.name, constrainedDescription,
        0., 0., 0., 0., 0., 0., 0., 0., 0
      };

      for (const auto category : {
             ChargeCategory::Unlike,
             ChargeCategory::Like})
      {
        const std::string categoryName =
          chargeName(category);

        TDirectory* chargeDir =
          selectionDir->mkdir(categoryName.c_str());

        promptConstrainedHistograms
          [selection.name][categoryName] =
            bookHistograms(
              chargeDir,
              bookingSelection,
              categoryName);
      }
    }
  }

  const Long64_t totalEntries =
    chain.GetEntries();

  const Long64_t nLikeSignEntries =
    std::max<Long64_t>(
      0,
      totalEntries - nUnlikeEntries);

  // maxEntries is applied independently to pairTree and likeSignPairTree.
  // This keeps a short debug run from consuming only the unlike-sign block.
  const Long64_t unlikeEntriesToProcess =
    maxEntries >= 0
      ? std::min(nUnlikeEntries, maxEntries)
      : nUnlikeEntries;

  const Long64_t likeSignEntriesToProcess =
    maxEntries >= 0
      ? std::min(nLikeSignEntries, maxEntries)
      : nLikeSignEntries;

  const Long64_t entriesToProcess =
    unlikeEntriesToProcess +
    likeSignEntriesToProcess;

  std::cout
    << "Added " << nFiles
    << " tree/file segments"
    << ", pairTree entries = " << nUnlikeEntries
    << ", likeSignPairTree entries = " << nLikeSignEntries
    << ", processing pairTree = " << unlikeEntriesToProcess
    << ", processing likeSignPairTree = " << likeSignEntriesToProcess
    << ", prompt kinematics = "
    << (usePrimaryVertexKinematicsForPrompt
          ? "primary vertex"
          : "secondary pair PCA")
    << ", forced-vertex prompt histograms = "
    << (includePrimaryConstrainedPrompt ? "ON" : "OFF")
    << ", separate like-sign tree files = "
    << nLikeSignFiles
    << std::endl;

  for (Long64_t processedEntry = 0;
       processedEntry < entriesToProcess;
       ++processedEntry)
  {
    const Long64_t entry =
      processedEntry < unlikeEntriesToProcess
        ? processedEntry
        : nUnlikeEntries +
            (processedEntry - unlikeEntriesToProcess);

    chain.GetEntry(entry);

    if (ignoreCrossingZero && (crossing1 == 0 || crossing2 == 0))
    {
      continue;
    }

    // daughter class: TPC-only / Si (>= minSiHits Si points)
    if (trackClassName != "all")
    {
      const int nSi1 = static_cast<int>(nmvtx1 + nintt1);
      const int nSi2 = static_cast<int>(nmvtx2 + nintt2);
      const bool si1 = has_si1 != 0 && nSi1 >= minSiHits;
      const bool si2 = has_si2 != 0 && nSi2 >= minSiHits;
      const bool tpc1 = has_si1 == 0;
      const bool tpc2 = has_si2 == 0;
      const bool keep =
        (trackClassName == "si" && si1 && si2) ||
        (trackClassName == "tpconly" && tpc1 && tpc2) ||
        (trackClassName == "mixed" && ((si1 && tpc2) || (tpc1 && si2)));
      if (!keep)
      {
        continue;
      }
    }

    // collision vertex of this pair (Si+TPC) or the configured beam position
    const bool pairVertexValid =
      havePairVertex && std::isfinite(pv_x) && std::isfinite(pv_y) && std::isfinite(pv_z);
    const double vertexX = pairVertexValid ? pv_x : beamX;
    const double vertexY = pairVertexValid ? pv_y : beamY;
    const double vertexZ = pairVertexValid ? pv_z : beamZ;

    const double qualityScale =
      ScaleQualityBy10 ? .0001 : 1.0;

    if (processedEntry % 100000 == 0)
    {
      std::cout
        << "Processing " << processedEntry
        << " / " << entriesToProcess
        << " (chain entry " << entry << ")"
        << std::endl;
    }

    const std::string sourceTreeName =
      chain.GetTree() != nullptr
        ? chain.GetTree()->GetName()
        : "";

    const bool fromLikeSignTree =
      sourceTreeName == likeSignTreeName;

    const bool unlikeSign =
      charge1 * charge2 < 0.0;

    const bool likeSign =
      charge1 * charge2 > 0.0;

    // The updated producer writes unlike-sign candidates to pairTree and
    // like-sign candidates to likeSignPairTree. Enforce this separation here
    // so old files cannot accidentally double-count same-sign rows.
    if ((fromLikeSignTree && !likeSign) ||
        (!fromLikeSignTree && !unlikeSign))
    {
      continue;
    }

    auto categories =
      chargeCategories(charge1, charge2);

    categories.erase(
      std::remove_if(
        categories.begin(),
        categories.end(),
        [fromLikeSignTree](const ChargeCategory category)
        {
          if (fromLikeSignTree)
          {
            return category == ChargeCategory::Unlike;
          }

          return category != ChargeCategory::Unlike;
        }),
      categories.end());

    if (categories.empty())
    {
      continue;
    }

    // The signed V0 delta-phi definition is unambiguous for unlike-sign pairs.
    // For like-sign pairs there is no positive/negative daughter ordering, so
    // require the same opening in either track ordering.
    const double deltaPhiThreshold =
      0.8 - 0.4 * (v0_pt < 2.0 ? v0_pt : 2.0);

    const double phi1 =
      std::atan2(py1, px1);

    const double phi2 =
      std::atan2(py2, px2);

    double deltaPhi = 0.0;

    if (unlikeSign)
    {
      const double phiPositive =
        charge1 > 0.0 ? phi1 : phi2;

      const double phiNegative =
        charge1 > 0.0 ? phi2 : phi1;

      deltaPhi =
        wrapPhi(phiPositive - phiNegative);
    }
    else
    {
      ////for k0s it just random phi-phi, for Lambda phi of proton is phiPositive and for anti-Lambda phi of proton is phiNegative
      if (mass_Kshort > 0.4 && mass_Kshort < 0.6)
      {
        deltaPhi = -(wrapPhi(phi1 - phi2));
      }
      else if (mass_Lambda > 1.0 && mass_Lambda < 1.3)
      {
        const double phiProton =
          sqrt(px1*px1 + py1*py1) > sqrt(px2*px2 + py2*py2) ? phi1 : phi2;

        const double phiPion = 
          sqrt(px1*px1 + py1*py1) > sqrt(px2*px2 + py2*py2) ? phi2 : phi1;
        
        deltaPhi =
          wrapPhi(phiProton - phiPion);
      }
      else if (mass_AntiLambda > 1.0 && mass_AntiLambda < 1.3)
      {
        const double phiProton =
          sqrt(px1*px1 + py1*py1) > sqrt(px2*px2 + py2*py2) ? phi2 : phi1;

        const double phiPion = 
          sqrt(px1*px1 + py1*py1) > sqrt(px2*px2 + py2*py2) ? phi1 : phi2;

        deltaPhi =
          wrapPhi(phiProton - phiPion);
      }
      else
      {
        deltaPhi = (wrapPhi(phi1 - phi2));
      }
    }

    const bool passV0DeltaPhi =
      !applySignedDeltaPhiCut || deltaPhi >= deltaPhiThreshold;

    // The signed Delta-phi requirement was designed for unlike-sign V0
    // daughters, where positive and negative tracks have a physical ordering.
    // A like-sign pair has no positive-versus-negative daughter assignment, so
    // applying the same signed requirement can remove essentially all K0S
    // background candidates.  Keep the cut for unlike-sign signal candidates
    // and do not apply it to rows read from likeSignPairTree.
    const bool passV0DeltaPhiForV0 = passV0DeltaPhi;
      //unlikeSign ? passV0DeltaPhi : mass_Kshort > .4 && mass_Kshort < .6 ? true : passV0DeltaPhi;

    // Secondary pair-PCA kinematics: always used for K0S/Lambda.
    const double pt1 =
      std::hypot(px1, py1);

    const double pt2 =
      std::hypot(px2, py2);

    const double daughterPtMin =
      std::min(pt1, pt2);

    const bool track1HasHigherPt =
      pt1 >= pt2;

    // Unique like-sign p-pi assignment: the higher-pT daughter receives the
    // proton hypothesis. This mirrors the default Lambda background rule.
    const double likeSignLambdaMass =
      track1HasHigherPt
        ? mass_P1Pi2
        : mass_Pi1P2;

    const double likeSignProtonDedx =
      track1HasHigherPt
        ? dedx_1
        : dedx_2;

    const double likeSignPionDedx =
      track1HasHigherPt
        ? dedx_2
        : dedx_1;

    // Ordinary prompt kinematics: selectable between the independent primary
    // beam-axis PCA and the secondary pair PCA.
    const double promptPx1 =
      usePrimaryVertexKinematicsForPrompt
        ? primary_px1
        : px1;

    const double promptPy1 =
      usePrimaryVertexKinematicsForPrompt
        ? primary_py1
        : py1;

    const double promptPz1 =
      usePrimaryVertexKinematicsForPrompt
        ? primary_pz1
        : pz1;

    const double promptPx2 =
      usePrimaryVertexKinematicsForPrompt
        ? primary_px2
        : px2;

    const double promptPy2 =
      usePrimaryVertexKinematicsForPrompt
        ? primary_py2
        : py2;

    const double promptPz2 =
      usePrimaryVertexKinematicsForPrompt
        ? primary_pz2
        : pz2;

    const bool promptKinematicsFinite =
      primary_pca_valid != 0 &&
      std::isfinite(primary_pca1_z) &&
      std::isfinite(primary_pca2_z) &&
      std::isfinite(promptPx1) &&
      std::isfinite(promptPy1) &&
      std::isfinite(promptPz1) &&
      std::isfinite(promptPx2) &&
      std::isfinite(promptPy2) &&
      std::isfinite(promptPz2);

    const double promptPt1 =
      std::hypot(promptPx1, promptPy1);

    const double promptPt2 =
      std::hypot(promptPx2, promptPy2);

    const double promptDaughterPtMin =
      std::min(promptPt1, promptPt2);

    const double promptPairPt =
      std::hypot(
        promptPx1 + promptPx2,
        promptPy1 + promptPy2);

    const double promptMassPhi =
      invariantMass(
        promptPx1, promptPy1, promptPz1, kKaonMass,
        promptPx2, promptPy2, promptPz2, kKaonMass);

    const double promptMassJpsi =
      invariantMass(
        promptPx1, promptPy1, promptPz1, kElectronMass,
        promptPx2, promptPy2, promptPz2, kElectronMass);

    const double promptMassK1Pi2 =
      invariantMass(
        promptPx1, promptPy1, promptPz1, kKaonMass,
        promptPx2, promptPy2, promptPz2, kPionMass);

    const double promptMassPi1K2 =
      invariantMass(
        promptPx1, promptPy1, promptPz1, kPionMass,
        promptPx2, promptPy2, promptPz2, kKaonMass);

    double promptMassD0 =
      std::numeric_limits<double>::quiet_NaN();

    double promptMassAntiD0 =
      std::numeric_limits<double>::quiet_NaN();

    if (unlikeSign)
    {
      if (charge1 > 0.0)
      {
        // D0: K- pi+; anti-D0: K+ pi-.
        promptMassD0 =
          promptMassPi1K2;

        promptMassAntiD0 =
          promptMassK1Pi2;
      }
      else
      {
        promptMassD0 =
          promptMassK1Pi2;

        promptMassAntiD0 =
          promptMassPi1K2;
      }
    }

    // Forced-primary-vertex kinematics. These are a second, independent set of
    // prompt variables and never overwrite the ordinary primary-PCA values.
    const bool constrainedPromptKinematicsFinite =
      includePrimaryConstrainedPrompt &&
      primary_constrained_valid1 != 0 &&
      primary_constrained_valid2 != 0 &&
      std::isfinite(primary_constrained_px1) &&
      std::isfinite(primary_constrained_py1) &&
      std::isfinite(primary_constrained_pz1) &&
      std::isfinite(primary_constrained_px2) &&
      std::isfinite(primary_constrained_py2) &&
      std::isfinite(primary_constrained_pz2);

    const double constrainedPromptPt1 =
      std::hypot(
        primary_constrained_px1,
        primary_constrained_py1);

    const double constrainedPromptPt2 =
      std::hypot(
        primary_constrained_px2,
        primary_constrained_py2);

    const double constrainedPromptDaughterPtMin =
      std::min(
        constrainedPromptPt1,
        constrainedPromptPt2);

    const double constrainedPromptPairPt =
      std::isfinite(primary_constrained_pair_pt)
        ? primary_constrained_pair_pt
        : std::hypot(
            primary_constrained_px1 +
              primary_constrained_px2,
            primary_constrained_py1 +
              primary_constrained_py2);

    const double constrainedPromptMassPhi =
      std::isfinite(mass_Phi_primary_constrained)
        ? mass_Phi_primary_constrained
        : invariantMass(
            primary_constrained_px1,
            primary_constrained_py1,
            primary_constrained_pz1,
            kKaonMass,
            primary_constrained_px2,
            primary_constrained_py2,
            primary_constrained_pz2,
            kKaonMass);

    const double constrainedPromptMassJpsi =
      invariantMass(
        primary_constrained_px1,
        primary_constrained_py1,
        primary_constrained_pz1,
        kElectronMass,
        primary_constrained_px2,
        primary_constrained_py2,
        primary_constrained_pz2,
        kElectronMass);

    const double constrainedPromptMassK1Pi2 =
      std::isfinite(mass_K1Pi2_primary_constrained)
        ? mass_K1Pi2_primary_constrained
        : invariantMass(
            primary_constrained_px1,
            primary_constrained_py1,
            primary_constrained_pz1,
            kKaonMass,
            primary_constrained_px2,
            primary_constrained_py2,
            primary_constrained_pz2,
            kPionMass);

    const double constrainedPromptMassPi1K2 =
      std::isfinite(mass_Pi1K2_primary_constrained)
        ? mass_Pi1K2_primary_constrained
        : invariantMass(
            primary_constrained_px1,
            primary_constrained_py1,
            primary_constrained_pz1,
            kPionMass,
            primary_constrained_px2,
            primary_constrained_py2,
            primary_constrained_pz2,
            kKaonMass);

    double constrainedPromptMassD0 =
      std::numeric_limits<double>::quiet_NaN();

    double constrainedPromptMassAntiD0 =
      std::numeric_limits<double>::quiet_NaN();

    if (unlikeSign)
    {
      constrainedPromptMassD0 =
        std::isfinite(mass_D0_primary_constrained)
          ? mass_D0_primary_constrained
          : (charge1 > 0.0
               ? constrainedPromptMassPi1K2
               : constrainedPromptMassK1Pi2);

      constrainedPromptMassAntiD0 =
        std::isfinite(mass_AntiD0_primary_constrained)
          ? mass_AntiD0_primary_constrained
          : (charge1 > 0.0
               ? constrainedPromptMassK1Pi2
               : constrainedPromptMassPi1K2);
    }

    const double absDeltaPcaZ =
      std::abs(pca1_z - pca2_z);

    // Secondary V0 flight is measured from the configured beam position,
    // not from the detector origin.
    const double flightX = pca_x - vertexX;
    const double flightY = pca_y - vertexY;
    const double flightZ = pca_z - vertexZ;

    const double decayRadius =
      std::hypot(flightX, flightY);

    const double absAlpha =
      std::abs(alpha);

    const double absPairDCA =
      std::abs(pairDCA);

    const double maxAbsTrackDcaXY =
      std::max(std::abs(dca_xy1), std::abs(dca_xy2));
    const double maxAbsTrackDcaZ =
      std::max(std::abs(dca_z1), std::abs(dca_z2));

    const double qualityMax =
      qualityScale * std::max(quality1, quality2);

    const int npointsMin =
      std::min<int>(npoints1, npoints2);

    const unsigned int ntpcClustersMin =
      std::min(ntpc_clusters1, ntpc_clusters2);

    const double maxAbsPrimaryPcaZ =
      std::max(
        std::abs(primary_pca1_z - vertexZ),
        std::abs(primary_pca2_z - vertexZ));

    const double absDeltaPrimaryPcaZ =
      std::isfinite(primary_pca_dz)
        ? std::abs(primary_pca_dz)
        : std::abs(primary_pca1_z - primary_pca2_z);

    const double absPromptPairDCA =
      (prompt_pca_valid != 0 && std::isfinite(prompt_pairDCA))
        ? std::abs(prompt_pairDCA)
        : std::numeric_limits<double>::quiet_NaN();

    const double v0Momentum =
      std::sqrt(
        v0_px * v0_px +
        v0_py * v0_py +
        v0_pz * v0_pz);

    const double flightLength =
      std::sqrt(
        flightX * flightX +
        flightY * flightY +
        flightZ * flightZ);

    const double dira =
      (v0Momentum > 0. && flightLength > 0.)
        ? (v0_px * flightX +
           v0_py * flightY +
           v0_pz * flightZ) /
          (v0Momentum * flightLength)
        : -2.;

    const double p1Total = std::sqrt(px1 * px1 + py1 * py1 + pz1 * pz1);
    const double p2Total = std::sqrt(px2 * px2 + py2 * py2 + pz2 * pz2);

    // momentum-scale study: TPC side (z of the daughter at r = 50 cm, straight line from the decay
    // vertex) and |eta| class of each daughter, phi quadrant of the V0
    int sideEtaClass = -1;
    int sidePhiClass = -1;
    {
      const auto sideOf = [&](const double px, const double py, const double pz)
      {
        const double ptd = std::hypot(px, py);
        return (ptd > 0.0 && pca_z + 50.0 * pz / ptd >= 0.0) ? 1 : 0;
      };
      const auto etaClassOf = [](const double px, const double py, const double pz)
      {
        const double ptd = std::hypot(px, py);
        const double aeta = ptd > 0.0 ? std::fabs(std::asinh(pz / ptd)) : 99.0;
        return aeta < 0.4 ? 0 : (aeta < 0.8 ? 1 : 2);
      };
      const int side1 = sideOf(px1, py1, pz1);
      const int side2 = sideOf(px2, py2, pz2);
      const int eta1Class = etaClassOf(px1, py1, pz1);
      const int eta2Class = etaClassOf(px2, py2, pz2);
      sideEtaClass = (side1 == side2 && eta1Class == eta2Class) ? 3 * side1 + eta1Class : 6;
      const double v0Phi = std::atan2(py1 + py2, px1 + px2);
      const int quadrant = std::min(3, std::max(0, static_cast<int>(std::floor((v0Phi + M_PI) / (M_PI / 2)))));
      sidePhiClass = side1 == side2 ? 4 * side1 + quadrant : 8;
    }

    double positivePt = -1.;
    double negativePt = -1.;
    double positiveDedx = -1.;
    double negativeDedx = -1.;

    if (unlikeSign)
    {
      if (charge1 > 0)
      {
        positivePt = pt1;
        negativePt = pt2;
        positiveDedx = dedx_1;
        negativeDedx = dedx_2;
      }
      else
      {
        positivePt = pt2;
        negativePt = pt1;
        positiveDedx = dedx_2;
        negativeDedx = dedx_1;
      }
    }

    const int effectiveNpoints1 = npoints1;

    const int effectiveNpoints2 = npoints2;

    const bool exactCut1 =
      passV0DeltaPhiForV0 &&
      pca_z > -15.f && pca_z < 15.f &&
      absDeltaPcaZ < 0.5 &&
      pt1 > 0.3 && pt2 > 0.3 &&
      decayRadius > 6.0 &&
      absAlpha < 0.8 &&
      absPairDCA < 1.0 &&
      dira > 0.8 &&
      effectiveNpoints1 > 30 && effectiveNpoints2 > 30 &&
      quality1 < 20.0 && quality2 < 20.0;

    const bool exactCut2 =
      passV0DeltaPhiForV0 &&
      pca_z > -15.f && pca_z < 0.f &&
      absDeltaPcaZ < 0.5 &&
      pt1 > 0.3 && pt2 > 0.3 &&
      decayRadius > 6.0 &&
      absAlpha < 0.8 &&
      absPairDCA < 4.0 &&
      pz1 < 0.f && pz2 < 0.f &&
      dira > 0.8 &&
      effectiveNpoints1 > 30 && effectiveNpoints2 > 30;

    const bool exactCut3 =
      passV0DeltaPhiForV0 &&
      pca_z > -15.f && pca_z < 0.f &&
      absDeltaPcaZ < 1.0 &&
      pt1 > 0.3 && pt2 > 0.3 &&
      decayRadius > 6.0 &&
      absAlpha < 0.8 &&
      absPairDCA < 4.0 &&
      pz1 < 0.f && pz2 < 0.f &&
      dira > 0.8;

    // Lambda: positive daughter carries the proton hypothesis.
    const bool lambdaProtonPtPass =
      unlikeSign &&
      (!requireProtonHigherPt ||
       positivePt > negativePt);

    // anti-Lambda: negative daughter carries the antiproton hypothesis.
    const bool antiLambdaProtonPtPass =
      unlikeSign &&
      (!requireProtonHigherPt ||
       negativePt > positivePt);

    for (std::size_t selectionIndex = 0;
         selectionIndex < selections.size();
         ++selectionIndex)
    {
      const auto& selection =
        selections[selectionIndex];

      const bool kshortDedxPass =
        passesDedx(dedx_1, selection.pionDedxMin, selection.pionDedxMax) &&
        passesDedx(dedx_2, selection.pionDedxMin, selection.pionDedxMax);

      const bool lambdaDedxPass =
        unlikeSign &&
        passesDedx(positiveDedx,
                   selection.protonDedxMin,
                   selection.protonDedxMax) &&
        passesDedx(negativeDedx,
                   selection.pionDedxMin,
                   selection.pionDedxMax);

      const bool antiLambdaDedxPass =
        unlikeSign &&
        passesDedx(positiveDedx,
                   selection.pionDedxMin,
                   selection.pionDedxMax) &&
        passesDedx(negativeDedx,
                   selection.protonDedxMin,
                   selection.protonDedxMax);

      const bool likeLambdaDedxPass =
        likeSign &&
        passesDedx(likeSignProtonDedx,
                   selection.protonDedxMin,
                   selection.protonDedxMax) &&
        passesDedx(likeSignPionDedx,
                   selection.pionDedxMin,
                   selection.pionDedxMax);

      const bool treeAllowsKshort =
        !fromLikeSignTree ||
        (candidate_mask & kCandidateKShort) != 0U;

      const bool treeAllowsLambda =
        !fromLikeSignTree ||
        (charge1 + charge2 > 0.0
           ? (candidate_mask & kCandidateLambda) != 0U
           : (candidate_mask & kCandidateAntiLambda) != 0U);

      //if (!passV0DeltaPhi)
      //{
      //  continue;
      //}

      if (!passesSelection(
            selection,
            pca_z,
            absDeltaPcaZ,
            daughterPtMin,
            decayRadius,
            absAlpha,
            absPairDCA,
            dira,
            qualityMax,
            npointsMin))
      {
        continue;
      }

      if (unlikeSign)
      {
        h_cutflow_unlike->Fill(selectionIndex + 0.5);
      }

      if (likeSign)
      {
        h_cutflow_like->Fill(selectionIndex + 0.5);
      }

      for (const auto category : categories)
      {
        HistSet& h =
          histograms
            .at(selection.name)
            .at(chargeName(category));

        if (treeAllowsKshort &&
            kshortDedxPass &&
            passV0DeltaPhiForV0)
        {
          h.h_k0s_mass->Fill(mass_Kshort);
          h.h_k0s_mass_vs_v0pt->Fill(v0_pt, mass_Kshort);
          h.h3_k0s->Fill(v0_pt, mass_Kshort, absPairDCA);
          h.h3_k0s_fine_pairDCA->Fill(v0_pt, mass_Kshort, absPairDCA);
          {
            const MassRegion region = massRegion(mass_Kshort, kK0sPeakMin, kK0sPeakMax);
            TH2F* hd = region == MassRegion::Peak       ? h.h_dedx_vs_p_k0s_peak
                       : region == MassRegion::Sideband ? h.h_dedx_vs_p_k0s_side
                                                        : nullptr;
            if (hd)
            {
              hd->Fill(p1Total, dedx_1);
              hd->Fill(p2Total, dedx_2);
            }
          }
          h.h3_k0s_pt1_vs_pt2_vs_mass->Fill(
            pt1, pt2, mass_Kshort);
            if (crossing1 == crossing2)
            {
              h.h3_k0s_crossing_ptv0->Fill(
                crossing1, v0_pt, mass_Kshort);
            }

          if (category == ChargeCategory::Unlike)
          {
            h.h3_k0s_ptplus_vs_ptminus_vs_mass->Fill(
              positivePt, negativePt, mass_Kshort);
            if (sideEtaClass >= 0)
            {
              h.h3_k0s_ptplus_vs_ptminus_vs_mass_side_eta[sideEtaClass]->Fill(positivePt, negativePt, mass_Kshort);
            }
            if (sidePhiClass >= 0)
            {
              h.h3_k0s_ptplus_vs_ptminus_vs_mass_side_phi[sidePhiClass]->Fill(positivePt, negativePt, mass_Kshort);
            }
          }
        }

        h.h_armenteros_podolanski->Fill(
          alpha, qT);

        h.h_pair_dca_vs_delta_pca_z->Fill(
          absDeltaPcaZ, absPairDCA);

        if (category == ChargeCategory::Unlike)
        {
          if (lambdaProtonPtPass &&
              lambdaDedxPass &&
              passV0DeltaPhi)
          {
            h.h_lambda_mass->Fill(mass_Lambda);
            h.h_lambda_mass_vs_v0pt->Fill(
              v0_pt, mass_Lambda);

            h.h3_lambda->Fill(
              v0_pt, mass_Lambda, absPairDCA);
            h.h3_lambda_fine_pairDCA->Fill(v0_pt, mass_Lambda, absPairDCA);
            {
              // proton = positive daughter
              const MassRegion region = massRegion(mass_Lambda, kLambdaPeakMin, kLambdaPeakMax);
              const bool posIs1 = charge1 > 0.0;
              if (region != MassRegion::Outside)
              {
                const bool peak = region == MassRegion::Peak;
                (peak ? h.h_dedx_vs_p_lambda_proton_peak : h.h_dedx_vs_p_lambda_proton_side)
                  ->Fill(posIs1 ? p1Total : p2Total, posIs1 ? dedx_1 : dedx_2);
                (peak ? h.h_dedx_vs_p_lambda_pion_peak : h.h_dedx_vs_p_lambda_pion_side)
                  ->Fill(posIs1 ? p2Total : p1Total, posIs1 ? dedx_2 : dedx_1);
              }
            }
            if (crossing1 == crossing2)
            {
              h.h3_lambda_crossing_ptv0->Fill(
                crossing1, v0_pt, mass_Lambda);
            }
          }

          if (antiLambdaProtonPtPass &&
              antiLambdaDedxPass &&
              passV0DeltaPhi)
          {
            h.h_antilambda_mass->Fill(mass_AntiLambda);
            h.h_antilambda_mass_vs_v0pt->Fill(
              v0_pt, mass_AntiLambda);

            h.h3_antilambda->Fill(
              v0_pt, mass_AntiLambda, absPairDCA);
            h.h3_lambda_fine_pairDCA->Fill(v0_pt, mass_AntiLambda, absPairDCA);
            {
              // antiproton = negative daughter
              const MassRegion region = massRegion(mass_AntiLambda, kLambdaPeakMin, kLambdaPeakMax);
              const bool negIs1 = charge1 < 0.0;
              if (region != MassRegion::Outside)
              {
                const bool peak = region == MassRegion::Peak;
                (peak ? h.h_dedx_vs_p_lambda_proton_peak : h.h_dedx_vs_p_lambda_proton_side)
                  ->Fill(negIs1 ? p1Total : p2Total, negIs1 ? dedx_1 : dedx_2);
                (peak ? h.h_dedx_vs_p_lambda_pion_peak : h.h_dedx_vs_p_lambda_pion_side)
                  ->Fill(negIs1 ? p2Total : p1Total, negIs1 ? dedx_2 : dedx_1);
              }
            }

            if (crossing1 == crossing2)
            {
              h.h3_antilambda_crossing_ptv0->Fill(
                crossing1, v0_pt, mass_AntiLambda);
            }
          }
        }
        else if (treeAllowsLambda &&
                 likeLambdaDedxPass &&
                 passV0DeltaPhiForV0)
        {
          // The separate like-sign tree stores both p-pi assignments.
          // Use one unique background mass per pair by assigning the proton
          // hypothesis to the higher-pT daughter.
          if (charge1 + charge2 > 0.0)
          {
            h.h_lambda_mass->Fill(
              likeSignLambdaMass);

            h.h_lambda_mass_vs_v0pt->Fill(
              v0_pt,
              likeSignLambdaMass);

            h.h3_lambda->Fill(
              v0_pt,
              likeSignLambdaMass,
              absPairDCA);
          }
          else
          {
            h.h_antilambda_mass->Fill(
              likeSignLambdaMass);

            h.h_antilambda_mass_vs_v0pt->Fill(
              v0_pt,
              likeSignLambdaMass);

            h.h3_antilambda->Fill(
              v0_pt,
              likeSignLambdaMass,
              absPairDCA);
          }
        }
      }
    }

    // Prompt phi and D0/anti-D0 QA. The same filling logic is run for:
    //   1. ordinary primary-PCA momenta;
    //   2. forced-primary-vertex-constrained momenta.
    //
    // Unlike-sign rows come from pairTree. Like-sign rows come exclusively
    // from likeSignPairTree and use the explicit track-order K-pi assignments.
    auto fillPromptHistograms =
      [&](std::map<std::string,
                   std::map<std::string, HistSet>>& targetHistograms,
          const bool kinematicsFinite,
          const double localDaughterPtMin,
          const double localPairPt,
          const double localMassPhi,
          const double localMassJpsi,
          const double localMassD0,
          const double localMassAntiD0,
          const double localMassK1Pi2,
          const double localMassPi1K2,
          const bool fillTrackQA)
      {
        if (!kinematicsFinite)
        {
          return;
        }

        for (const auto& promptSelection : promptSelections)
        {
          const bool phiDedxPass =
            passesDedx(
              dedx_1,
              promptSelection.kaonDedxMin,
              promptSelection.kaonDedxMax) &&
            passesDedx(
              dedx_2,
              promptSelection.kaonDedxMin,
              promptSelection.kaonDedxMax);

          const bool d0DedxPass =
            unlikeSign &&
            passesDedx(
              positiveDedx,
              promptSelection.pionDedxMin,
              promptSelection.pionDedxMax) &&
            passesDedx(
              negativeDedx,
              promptSelection.kaonDedxMin,
              promptSelection.kaonDedxMax);

          const bool antiD0DedxPass =
            unlikeSign &&
            passesDedx(
              positiveDedx,
              promptSelection.kaonDedxMin,
              promptSelection.kaonDedxMax) &&
            passesDedx(
              negativeDedx,
              promptSelection.pionDedxMin,
              promptSelection.pionDedxMax);

          const bool likeK1Pi2DedxPass =
            likeSign &&
            passesDedx(
              dedx_1,
              promptSelection.kaonDedxMin,
              promptSelection.kaonDedxMax) &&
            passesDedx(
              dedx_2,
              promptSelection.pionDedxMin,
              promptSelection.pionDedxMax);

          const bool likePi1K2DedxPass =
            likeSign &&
            passesDedx(
              dedx_1,
              promptSelection.pionDedxMin,
              promptSelection.pionDedxMax) &&
            passesDedx(
              dedx_2,
              promptSelection.kaonDedxMin,
              promptSelection.kaonDedxMax);

          if (!passesPromptSelection(
                promptSelection,
                localDaughterPtMin,
                ntpcClustersMin,
                qualityMax,
                maxAbsTrackDcaXY,
                maxAbsTrackDcaZ,
                maxAbsPrimaryPcaZ,
                absDeltaPrimaryPcaZ,
                absPromptPairDCA))
          {
            continue;
          }

          const bool treeAllowsPhi =
            !fromLikeSignTree ||
            (candidate_mask & kCandidatePhi) != 0U;

          const bool treeAllowsD0 =
            !fromLikeSignTree ||
            (candidate_mask & kCandidateD0) != 0U;

          const bool treeAllowsAntiD0 =
            !fromLikeSignTree ||
            (candidate_mask & kCandidateAntiD0) != 0U;

          for (const auto category : categories)
          {
            HistSet& h =
              targetHistograms
                .at(promptSelection.name)
                .at(chargeName(category));

            // per-track resolution QA (ordinary primary-PCA set only)
            if (fillTrackQA)
            {
              const double ptA = std::hypot(primary_px1, primary_py1);
              const double ptB = std::hypot(primary_px2, primary_py2);
              h.h_dca_xy_vs_pt->Fill(ptA, dca_xy1);
              h.h_dca_xy_vs_pt->Fill(ptB, dca_xy2);
              h.h_dca_z_vs_pt->Fill(ptA, dca_z1);
              h.h_dca_z_vs_pt->Fill(ptB, dca_z2);
              if (constrainedPromptKinematicsFinite && ptA > 0.0 && ptB > 0.0)
              {
                h.h_dpt_constrained_vs_pt->Fill(ptA, (constrainedPromptPt1 - ptA) / ptA);
                h.h_dpt_constrained_vs_pt->Fill(ptB, (constrainedPromptPt2 - ptB) / ptB);
              }
            }

            if (treeAllowsPhi &&
                phiDedxPass &&
                std::isfinite(localMassPhi))
            {
              h.h_phi_mass->Fill(
                localMassPhi);

              if (std::isfinite(absPromptPairDCA))
              {
                h.h3_phi_fine_pairDCA->Fill(localPairPt, localMassPhi, absPromptPairDCA);
              }
              {
                const MassRegion region = massRegion(localMassPhi, kPhiPeakMin, kPhiPeakMax);
                TH2F* hd = region == MassRegion::Peak       ? h.h_dedx_vs_p_phi_peak
                           : region == MassRegion::Sideband ? h.h_dedx_vs_p_phi_side
                                                            : nullptr;
                if (hd)
                {
                  hd->Fill(std::sqrt(primary_px1 * primary_px1 + primary_py1 * primary_py1 + primary_pz1 * primary_pz1), dedx_1);
                  hd->Fill(std::sqrt(primary_px2 * primary_px2 + primary_py2 * primary_py2 + primary_pz2 * primary_pz2), dedx_2);
                }
              }

              h.h_phi_mass_vs_v0pt->Fill(
                localPairPt,
                localMassPhi);

              if (std::isfinite(absPromptPairDCA))
              {
                h.h3_phi->Fill(
                  localPairPt,
                  localMassPhi,
                  absPromptPairDCA);
              }
            }

            // Electron-mass hypothesis. Fill unlike-sign signal candidates and
            // like-sign combinatorial background with identical prompt cuts.
            if (std::isfinite(localMassJpsi))
            {
              h.h_jpsi_mass->Fill(localMassJpsi);

              h.h_jpsi_mass_vs_v0pt->Fill(
                localPairPt,
                localMassJpsi);

              if (std::isfinite(absPromptPairDCA))
              {
                h.h3_jpsi->Fill(
                  localPairPt,
                  localMassJpsi,
                  absPromptPairDCA);
              }
            }

            h.h_primary_pca_dz->Fill(
              absDeltaPrimaryPcaZ);

            h.h_primary_pca_z1_vs_z2->Fill(
              primary_pca1_z,
              primary_pca2_z);

            h.h_max_dca_xy_vs_primary_pca_dz->Fill(
              absDeltaPrimaryPcaZ,
              maxAbsTrackDcaXY);

            if (std::isfinite(absPromptPairDCA))
            {
              h.h_pair_dca_vs_delta_pca_z->Fill(
                absDeltaPrimaryPcaZ,
                absPromptPairDCA);
            }

            if (category == ChargeCategory::Unlike)
            {
              if (d0DedxPass &&
                  std::isfinite(localMassD0))
              {
                h.h_d0_mass->Fill(
                  localMassD0);

                h.h_d0_mass_vs_v0pt->Fill(
                  localPairPt,
                  localMassD0);

                if (std::isfinite(absPromptPairDCA))
                {
                  h.h3_d0->Fill(
                    localPairPt,
                    localMassD0,
                    absPromptPairDCA);
                }
              }

              if (antiD0DedxPass &&
                  std::isfinite(localMassAntiD0))
              {
                h.h_antid0_mass->Fill(
                  localMassAntiD0);

                h.h_antid0_mass_vs_v0pt->Fill(
                  localPairPt,
                  localMassAntiD0);

                if (std::isfinite(absPromptPairDCA))
                {
                  h.h3_antid0->Fill(
                    localPairPt,
                    localMassAntiD0,
                    absPromptPairDCA);
                }
              }
            }
            else
            {
              // Same-sign D0 background has no charge-defined K/pi ordering.
              // Keep both explicit track-order assignments in separate
              // D0-like and anti-D0-like histograms.
              if (treeAllowsD0 &&
                  likeK1Pi2DedxPass &&
                  std::isfinite(localMassK1Pi2))
              {
                h.h_d0_mass->Fill(
                  localMassK1Pi2);

                h.h_d0_mass_vs_v0pt->Fill(
                  localPairPt,
                  localMassK1Pi2);

                if (std::isfinite(absPromptPairDCA))
                {
                  h.h3_d0->Fill(
                    localPairPt,
                    localMassK1Pi2,
                    absPromptPairDCA);
                }
              }

              if (treeAllowsAntiD0 &&
                  likePi1K2DedxPass &&
                  std::isfinite(localMassPi1K2))
              {
                h.h_antid0_mass->Fill(
                  localMassPi1K2);

                h.h_antid0_mass_vs_v0pt->Fill(
                  localPairPt,
                  localMassPi1K2);

                if (std::isfinite(absPromptPairDCA))
                {
                  h.h3_antid0->Fill(
                    localPairPt,
                    localMassPi1K2,
                    absPromptPairDCA);
                }
              }
            }
          }
        }
      };

    fillPromptHistograms(
      promptHistograms,
      promptKinematicsFinite,
      promptDaughterPtMin,
      promptPairPt,
      promptMassPhi,
      promptMassJpsi,
      promptMassD0,
      promptMassAntiD0,
      promptMassK1Pi2,
      promptMassPi1K2,
      true);

    if (includePrimaryConstrainedPrompt)
    {
      fillPromptHistograms(
        promptConstrainedHistograms,
        constrainedPromptKinematicsFinite,
        constrainedPromptDaughterPtMin,
        constrainedPromptPairPt,
        constrainedPromptMassPhi,
        constrainedPromptMassJpsi,
        constrainedPromptMassD0,
        constrainedPromptMassAntiD0,
        constrainedPromptMassK1Pi2,
        constrainedPromptMassPi1K2,
        false);
    }

    const bool exactKshortDedxPass =
      passesDedx(dedx_1, -1.0, 400.0) &&
      passesDedx(dedx_2, -1.0, 400.0);

    const bool exactLambdaDedxPass =
      unlikeSign && passesDedx(negativeDedx, -1.0, 400.0);

    const bool exactAntiLambdaDedxPass =
      unlikeSign && passesDedx(positiveDedx, -1.0, 400.0);

    const bool exactLikeLambdaDedxPass =
      likeSign &&
      passesDedx(likeSignPionDedx, -1.0, 400.0);

    const bool exactTreeAllowsKshort =
      !fromLikeSignTree ||
      (candidate_mask & kCandidateKShort) != 0U;

    const bool exactTreeAllowsLambda =
      !fromLikeSignTree ||
      (charge1 + charge2 > 0.0
         ? (candidate_mask & kCandidateLambda) != 0U
         : (candidate_mask & kCandidateAntiLambda) != 0U);

    const std::vector<std::pair<std::string, bool>> exactPasses = {
      {"exactCut1", exactCut1},
      {"exactCut2", exactCut2},
      {"exactCut3", exactCut3}
    };

    for (const auto& exactPass : exactPasses)
    {
      if (!exactPass.second)
      {
        continue;
      }

      for (const auto category : categories)
      {
        HistSet& h =
          exactHistograms
            .at(exactPass.first)
            .at(chargeName(category));

        if (exactTreeAllowsKshort &&
            exactKshortDedxPass)
        {
          h.h_k0s_mass->Fill(mass_Kshort);
          h.h_k0s_mass_vs_v0pt->Fill(v0_pt, mass_Kshort);
          h.h3_k0s->Fill(v0_pt, mass_Kshort, absPairDCA);
          h.h3_k0s_pt1_vs_pt2_vs_mass->Fill(
            pt1, pt2, mass_Kshort);

          if (category == ChargeCategory::Unlike)
          {
            h.h3_k0s_ptplus_vs_ptminus_vs_mass->Fill(
              positivePt, negativePt, mass_Kshort);
            if (sideEtaClass >= 0)
            {
              h.h3_k0s_ptplus_vs_ptminus_vs_mass_side_eta[sideEtaClass]->Fill(positivePt, negativePt, mass_Kshort);
            }
            if (sidePhiClass >= 0)
            {
              h.h3_k0s_ptplus_vs_ptminus_vs_mass_side_phi[sidePhiClass]->Fill(positivePt, negativePt, mass_Kshort);
            }
          }
        }

        h.h_armenteros_podolanski->Fill(alpha, qT);
        h.h_pair_dca_vs_delta_pca_z->Fill(
          absDeltaPcaZ, absPairDCA);

        if (category == ChargeCategory::Unlike)
        {
          if (lambdaProtonPtPass &&
              exactLambdaDedxPass)
          {
            h.h_lambda_mass->Fill(mass_Lambda);
            h.h_lambda_mass_vs_v0pt->Fill(
              v0_pt,
              mass_Lambda);
            h.h3_lambda->Fill(
              v0_pt,
              mass_Lambda,
              absPairDCA);
          }

          if (antiLambdaProtonPtPass &&
              exactAntiLambdaDedxPass)
          {
            h.h_antilambda_mass->Fill(mass_AntiLambda);
            h.h_antilambda_mass_vs_v0pt->Fill(
              v0_pt,
              mass_AntiLambda);
            h.h3_antilambda->Fill(
              v0_pt,
              mass_AntiLambda,
              absPairDCA);
          }
        }
        else if (exactTreeAllowsLambda &&
                 exactLikeLambdaDedxPass)
        {
          if (charge1 + charge2 > 0.0)
          {
            h.h_lambda_mass->Fill(
              likeSignLambdaMass);

            h.h_lambda_mass_vs_v0pt->Fill(
              v0_pt,
              likeSignLambdaMass);

            h.h3_lambda->Fill(
              v0_pt,
              likeSignLambdaMass,
              absPairDCA);
          }
          else
          {
            h.h_antilambda_mass->Fill(
              likeSignLambdaMass);

            h.h_antilambda_mass_vs_v0pt->Fill(
              v0_pt,
              likeSignLambdaMass);

            h.h3_antilambda->Fill(
              v0_pt,
              likeSignLambdaMass,
              absPairDCA);
          }
        }
      }
    }
  }

  output->Write();
  output->Close();

  std::cout
    << "Wrote output: "
    << outputPath
    << std::endl;
}