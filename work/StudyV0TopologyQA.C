// StudyV0TopologyQA.C
//
// Expanded one-pass V0 topology QA macro.
//
// This keeps the core K0S double-peak QA from StudyK0sDoublePeakQA.C and adds:
//   * Lambda -> p pi- and anti-Lambda -> pi+ pbar mass hypotheses;
//   * more 2D topology-vs-mass histograms;
//   * selected TH3 histograms for pT / decay-radius dependence;
//   * peak / left-sideband / right-sideband / combined-sideband topology QA;
//   * explicit configurable qualityScale (default 0.1);
//   * the old K0S low/high mass-window maps.
//
// The same macro can be run independently on data and simulation so that the
// resulting ROOT files have identical histogram names and directory structure.
//
// Example:
// root -l -b -q \
// 'StudyV0TopologyQA.C("/path/to/files","v0_pp_*.root","output","v0_topology_qa.root","pairTree")'
//
// Existing StudyK0sDoublePeakQA-style calls remain simple because all newly
// added arguments have defaults.
//
// Si+TPC setup (TpcV0CandidateTree with use_sitpc_tracks):
//   * trackClass "all" / "tpconly" / "si" (both daughters >= minSiHits Si points) / "mixed":
//     run once per class and compare the outputs (same histogram names)
//   * decay radius and DIRA are measured from the pair's collision vertex (pv_x, pv_y, pv_z) when
//     present (usePairVertex), otherwise from (0, 0, 0)
//   * exactCut1 is loose (no dE/dx or quality cut, decay radius > minDecayRadius, default 0.05 cm,
//     because Si-matched daughters only exist for decays inside the MVTX)
//   * applySignedDeltaPhiCut = false: no sailor/cowboy cut (the post-cut QA equals the pre-cut QA)
//   * like-sign pairs are in likeSignPairTree: run a second time with treeName = "likeSignPairTree"
//
//   root -l -b -q 'StudyV0TopologyQA.C("/path","hadron_sitpc_*.root","out","topo_si.root","pairTree",
//                   0.470,0.490,0.490,0.510,-1,1.0,0.485,0.510,0.440,0.470,0.525,0.555,
//                   1.108,1.124,1.080,1.100,1.132,1.152,"si",4,false,true,0.05)'
//
// IMPORTANT:
//   * exactCut2 and the pT-dependent signed Delta-phi cut are kept
//     from the supplied K0S double-peak code; the Delta-phi cut is off by default.
//   * qualityScale is applied to local copies; branch values are not modified.
//   * Lambda hypothesis: positive daughter = proton, negative = pion.
//   * anti-Lambda hypothesis: positive daughter = pion, negative = antiproton.
//

#include <TChain.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH1I.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TMath.h>
#include <TNamed.h>
#include <TParameter.h>
#include <TString.h>
#include <TSystem.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{
  constexpr double kPionMass = 0.13957039;
  constexpr double kProtonMass = 0.938272088;
  constexpr double kTwoPi = 2.0 * TMath::Pi();

  enum class Species
  {
    K0s,
    Lambda,
    AntiLambda
  };

  const char* speciesName(const Species species)
  {
    switch (species)
    {
      case Species::K0s:       return "K0s";
      case Species::Lambda:    return "Lambda";
      case Species::AntiLambda:return "AntiLambda";
    }
    return "Unknown";
  }

  const char* speciesTitle(const Species species)
  {
    switch (species)
    {
      case Species::K0s:       return "K^{0}_{S}";
      case Species::Lambda:    return "#Lambda";
      case Species::AntiLambda:return "#bar{#Lambda}";
    }
    return "V0";
  }

  const char* massAxisTitle(const Species species)
  {
    switch (species)
    {
      case Species::K0s:       return "m_{#pi#pi} [GeV/c^{2}]";
      case Species::Lambda:    return "m_{p#pi^{-}} [GeV/c^{2}]";
      case Species::AntiLambda:return "m_{#pi^{+}#bar{p}} [GeV/c^{2}]";
    }
    return "m [GeV/c^{2}]";
  }

  void speciesMassRange(const Species species, double& minMass, double& maxMass)
  {
    if (species == Species::K0s)
    {
      minMass = 0.42;
      maxMass = 0.57;
    }
    else
    {
      minMass = 1.08;
      maxMass = 1.16;
    }
  }

  double wrapPhi(double phi)
  {
    while (phi >= TMath::Pi()) phi -= kTwoPi;
    while (phi < -TMath::Pi()) phi += kTwoPi;
    return phi;
  }

  double safeEta(double px, double py, double pz)
  {
    const double pT = std::hypot(px, py);
    return (pT > 0.) ? std::asinh(pz / pT) : 0.;
  }

  double safeMassHypothesis(double pxPos, double pyPos, double pzPos, double mPos,
                            double pxNeg, double pyNeg, double pzNeg, double mNeg)
  {
    const double pPos2 = pxPos*pxPos + pyPos*pyPos + pzPos*pzPos;
    const double pNeg2 = pxNeg*pxNeg + pyNeg*pyNeg + pzNeg*pzNeg;

    const double ePos = std::sqrt(pPos2 + mPos*mPos);
    const double eNeg = std::sqrt(pNeg2 + mNeg*mNeg);

    const double px = pxPos + pxNeg;
    const double py = pyPos + pyNeg;
    const double pz = pzPos + pzNeg;

    const double m2 =
      (ePos + eNeg)*(ePos + eNeg) -
      px*px - py*py - pz*pz;

    return (m2 > 0.) ? std::sqrt(m2) : 0.;
  }

  double openingAngle(double px1, double py1, double pz1,
                      double px2, double py2, double pz2)
  {
    const double p1 = std::sqrt(px1*px1 + py1*py1 + pz1*pz1);
    const double p2 = std::sqrt(px2*px2 + py2*py2 + pz2*pz2);

    if (p1 <= 0. || p2 <= 0.) return -1.;

    double c = (px1*px2 + py1*py2 + pz1*pz2)/(p1*p2);
    c = std::max(-1.0, std::min(1.0, c));

    return std::acos(c);
  }

  bool branchExists(TChain& chain, const char* name)
  {
    return chain.GetBranch(name) != nullptr;
  }

  double safeRatio(double numerator, double denominator)
  {
    return (denominator != 0.) ? numerator / denominator : 0.;
  }

  double safeLog10OneMinusDira(double dira)
  {
    const double value = std::max(1.e-8, 1.0 - std::min(dira, 1.0));
    return std::log10(value);
  }

  struct EventKinematics
  {
    // charge ordered
    double pxPos = 0.;
    double pyPos = 0.;
    double pzPos = 0.;
    double pxNeg = 0.;
    double pyNeg = 0.;
    double pzNeg = 0.;

    double ptPos = 0.;
    double ptNeg = 0.;
    double pPos = 0.;
    double pNeg = 0.;

    double etaPos = 0.;
    double etaNeg = 0.;
    double phiPos = 0.;
    double phiNeg = 0.;

    double dcaXYPos = 0.;
    double dcaXYNeg = 0.;
    double dcaZPos = 0.;
    double dcaZNeg = 0.;

    double dedxPos = 0.;
    double dedxNeg = 0.;

    int npointsPos = 0;
    int npointsNeg = 0;

    double qualityPos = 0.;
    double qualityNeg = 0.;

    double v0Pt = 0.;
    double v0Phi = 0.;
    double v0Eta = 0.;

    double decayR = 0.;
    double decayPhi = 0.;
    double pcaZ = 0.;
    double deltaPcaX = 0.;
    double deltaPcaY = 0.;
    double deltaPcaZ = 0.;
    double deltaPca3D = 0.;

    double pairDCA = 0.;
    double dira = -2.;
    double alpha = 0.;
    double qT = 0.;

    double opening = 0.;
    double deltaPhi = 0.;
    double absDeltaPhi = 0.;
    double deltaEta = 0.;
    double absDeltaEta = 0.;
    double deltaTheta = 0.;
    double absDeltaTheta = 0.;
    double phiEtaRatio = 0.;

    double ptAsymSigned = 0.;
    double ptAsymAbs = 0.;
    double pAsymSigned = 0.;
    double pAsymAbs = 0.;

    double minAbsDcaXY = 0.;
    double maxAbsDcaXY = 0.;
    double absDcaXYProduct = 0.;
    double maxAbsDcaZ = 0.;

    int npointsMin = 0;
    int npointsMax = 0;
    double qualityMin = 0.;
    double qualityMax = 0.;
  };

  void heavyLightKinematics(const Species species,
                            const EventKinematics& k,
                            double& ptHeavy,
                            double& ptLight,
                            double& pHeavy,
                            double& pLight)
  {
    if (species == Species::Lambda)
    {
      ptHeavy = k.ptPos;
      ptLight = k.ptNeg;
      pHeavy = k.pPos;
      pLight = k.pNeg;
      return;
    }

    if (species == Species::AntiLambda)
    {
      ptHeavy = k.ptNeg;
      ptLight = k.ptPos;
      pHeavy = k.pNeg;
      pLight = k.pPos;
      return;
    }

    // K0S has two identical pion mass hypotheses; keep charge ordering.
    ptHeavy = k.ptPos;
    ptLight = k.ptNeg;
    pHeavy = k.pPos;
    pLight = k.pNeg;
  }

  TH2F* bookMassVs(TDirectory* dir,
                   const Species species,
                   const char* name,
                   const char* xTitle,
                   int nx, double xmin, double xmax,
                   int nmass = 300)
  {
    double massMin = 0.;
    double massMax = 0.;
    speciesMassRange(species, massMin, massMax);

    dir->cd();

    return new TH2F(
      name,
      TString::Format("%s mass vs %s;%s;%s",
                      speciesTitle(species),
                      xTitle,
                      xTitle,
                      massAxisTitle(species)),
      nx, xmin, xmax,
      nmass, massMin, massMax);
  }

  struct MassVsQA
  {
    // Existing histograms retained.
    TH2F* v0Pt = nullptr;
    TH2F* v0Phi = nullptr;
    TH2F* v0Eta = nullptr;
    TH2F* decayR = nullptr;
    TH2F* decayPhi = nullptr;
    TH2F* pcaZ = nullptr;
    TH2F* deltaPcaZ = nullptr;
    TH2F* pairDCA = nullptr;
    TH2F* dira = nullptr;
    TH2F* alpha = nullptr;
    TH2F* absAlpha = nullptr;

    TH2F* ptPos = nullptr;
    TH2F* ptNeg = nullptr;
    TH2F* ptMin = nullptr;
    TH2F* ptMax = nullptr;
    TH2F* ptAsymmetry = nullptr;
    TH2F* etaPos = nullptr;
    TH2F* etaNeg = nullptr;
    TH2F* etaMin = nullptr;
    TH2F* etaMax = nullptr;
    TH2F* phiPos = nullptr;
    TH2F* phiNeg = nullptr;
    TH2F* deltaPhi = nullptr;
    TH2F* opening = nullptr;
    TH2F* qOverPtPos = nullptr;
    TH2F* qOverPtNeg = nullptr;
    TH2F* pzPos = nullptr;
    TH2F* pzNeg = nullptr;
    TH2F* npointsMin = nullptr;
    TH2F* npointsMax = nullptr;
    TH2F* qualityMin = nullptr;
    TH2F* qualityMax = nullptr;
    TH2F* massRecalc = nullptr;
    TH2F* massDifference = nullptr;
    TH3F* deltaPhiVsMassVsPt = nullptr;
    TH3F* decayRVsMassVsPt = nullptr;
    TH2F* massvsPtminusoverPtplus = nullptr;

    // New 2D QA.
    TH2F* qT = nullptr;
    TH2F* absDeltaPcaX = nullptr;
    TH2F* absDeltaPcaY = nullptr;
    TH2F* deltaPca3D = nullptr;

    TH2F* absDeltaPhi = nullptr;
    TH2F* deltaEta = nullptr;
    TH2F* absDeltaEta = nullptr;
    TH2F* deltaTheta = nullptr;
    TH2F* absDeltaTheta = nullptr;
    TH2F* phiEtaRatio = nullptr;

    TH2F* absPtAsymmetry = nullptr;
    TH2F* pAsymmetry = nullptr;
    TH2F* absPAsymmetry = nullptr;

    TH2F* dcaXYPos = nullptr;
    TH2F* dcaXYNeg = nullptr;
    TH2F* minAbsDcaXY = nullptr;
    TH2F* maxAbsDcaXY = nullptr;
    TH2F* absDcaXYProduct = nullptr;
    TH2F* dcaZPos = nullptr;
    TH2F* dcaZNeg = nullptr;
    TH2F* maxAbsDcaZ = nullptr;

    TH2F* log10OneMinusDira = nullptr;
    TH2F* dedxPos = nullptr;
    TH2F* dedxNeg = nullptr;

    TH2F* ptHeavyOverLight = nullptr;
    TH2F* pHeavyOverLight = nullptr;

    // New selected TH3 QA.
    TH3F* openingVsMassVsPt = nullptr;
    TH3F* pairDCAVsMassVsPt = nullptr;
    TH3F* diraVsMassVsPt = nullptr;
    TH3F* alphaVsQTvsPt = nullptr;
    TH3F* deltaPhiVsDeltaEtaVsPt = nullptr;
    TH3F* minNpointsVsRVsPt = nullptr;
    TH3F* pairDCAVsRVsPt = nullptr;
    TH3F* diraVsRVsPt = nullptr;
  };

  MassVsQA bookMassVsQA(TDirectory* dir, const Species species)
  {
    MassVsQA h;

    double massMin = 0.;
    double massMax = 0.;
    speciesMassRange(species, massMin, massMax);

    h.v0Pt = bookMassVs(dir, species,
      "h_mass_vs_v0_pt", "p_{T}^{V0} [GeV/c]", 100, 0., 5.);
    h.v0Phi = bookMassVs(dir, species,
      "h_mass_vs_v0_phi", "#phi_{V0} [rad]", 144, -TMath::Pi(), TMath::Pi());
    h.v0Eta = bookMassVs(dir, species,
      "h_mass_vs_v0_eta", "#eta_{V0}", 120, -1.5, 1.5);
    h.decayR = bookMassVs(dir, species,
      "h_mass_vs_decay_radius", "decay radius [cm]", 120, 0., 80.);
    h.decayPhi = bookMassVs(dir, species,
      "h_mass_vs_decay_phi", "#phi_{decay} [rad]", 144, -TMath::Pi(), TMath::Pi());
    h.pcaZ = bookMassVs(dir, species,
      "h_mass_vs_pca_z", "PCA_{z} [cm]", 120, -15., 15.);
    h.deltaPcaZ = bookMassVs(dir, species,
      "h_mass_vs_abs_delta_pca_z", "|PCA_{z,1}-PCA_{z,2}| [cm]", 100, 0., 0.5);
    h.pairDCA = bookMassVs(dir, species,
      "h_mass_vs_abs_pairDCA", "|pair DCA| [cm]", 100, 0., 1.);
    h.dira = bookMassVs(dir, species,
      "h_mass_vs_DIRA", "DIRA", 100, 0.8, 1.0);
    h.alpha = bookMassVs(dir, species,
      "h_mass_vs_alpha", "#alpha", 120, -0.8, 0.8);
    h.absAlpha = bookMassVs(dir, species,
      "h_mass_vs_abs_alpha", "|#alpha|", 80, 0., 0.8);

    h.ptPos = bookMassVs(dir, species,
      "h_mass_vs_pt_positive", "p_{T}^{+} [GeV/c]", 100, 0., 5.);
    h.ptNeg = bookMassVs(dir, species,
      "h_mass_vs_pt_negative", "p_{T}^{-} [GeV/c]", 100, 0., 5.);
    h.ptMin = bookMassVs(dir, species,
      "h_mass_vs_pt_min", "min(p_{T}^{+},p_{T}^{-}) [GeV/c]", 100, 0., 3.);
    h.ptMax = bookMassVs(dir, species,
      "h_mass_vs_pt_max", "max(p_{T}^{+},p_{T}^{-}) [GeV/c]", 100, 0., 5.);
    h.ptAsymmetry = bookMassVs(dir, species,
      "h_mass_vs_pt_asymmetry",
      "(p_{T}^{+}-p_{T}^{-})/(p_{T}^{+}+p_{T}^{-})",
      120, -1., 1.);

    h.etaPos = bookMassVs(dir, species,
      "h_mass_vs_eta_positive", "#eta^{+}", 120, -1.5, 1.5);
    h.etaNeg = bookMassVs(dir, species,
      "h_mass_vs_eta_negative", "#eta^{-}", 120, -1.5, 1.5);
    h.etaMin = bookMassVs(dir, species,
      "h_mass_vs_eta_min", "min(#eta^{+},#eta^{-})", 120, -1.5, 1.5);
    h.etaMax = bookMassVs(dir, species,
      "h_mass_vs_eta_max", "max(#eta^{+},#eta^{-})", 120, -1.5, 1.5);

    h.phiPos = bookMassVs(dir, species,
      "h_mass_vs_phi_positive", "#phi^{+} [rad]", 144, -TMath::Pi(), TMath::Pi());
    h.phiNeg = bookMassVs(dir, species,
      "h_mass_vs_phi_negative", "#phi^{-} [rad]", 144, -TMath::Pi(), TMath::Pi());
    h.deltaPhi = bookMassVs(dir, species,
      "h_mass_vs_delta_phi", "#Delta#phi_{+-} [rad]", 144, -TMath::Pi(), TMath::Pi());

    h.opening = bookMassVs(dir, species,
      "h_mass_vs_opening_angle", "opening angle [rad]", 120, 0., TMath::Pi());

    h.qOverPtPos = bookMassVs(dir, species,
      "h_mass_vs_qOverPt_positive",
      "q/p_{T}^{+} [(GeV/c)^{-1}]",
      120, 0., 4.);
    h.qOverPtNeg = bookMassVs(dir, species,
      "h_mass_vs_qOverPt_negative",
      "q/p_{T}^{-} [(GeV/c)^{-1}]",
      120, -4., 0.);

    h.pzPos = bookMassVs(dir, species,
      "h_mass_vs_pz_positive", "p_{z}^{+} [GeV/c]", 120, -5., 5.);
    h.pzNeg = bookMassVs(dir, species,
      "h_mass_vs_pz_negative", "p_{z}^{-} [GeV/c]", 120, -5., 5.);

    h.npointsMin = bookMassVs(dir, species,
      "h_mass_vs_npoints_min",
      "min(N_{points}^{+},N_{points}^{-})",
      40, 30., 70.);
    h.npointsMax = bookMassVs(dir, species,
      "h_mass_vs_npoints_max",
      "max(N_{points}^{+},N_{points}^{-})",
      40, 30., 70.);

    h.qualityMin = bookMassVs(dir, species,
      "h_mass_vs_quality_min",
      "min(quality^{+},quality^{-})",
      100, 0., 2.);
    h.qualityMax = bookMassVs(dir, species,
      "h_mass_vs_quality_max",
      "max(quality^{+},quality^{-})",
      100, 0., 2.);

    h.massRecalc = new TH2F(
      "h_stored_mass_vs_recalculated_mass",
      TString::Format("%s stored vs recalculated;%s recalculated;%s stored",
                      speciesTitle(species),
                      massAxisTitle(species),
                      massAxisTitle(species)),
      300, massMin, massMax,
      300, massMin, massMax);

    h.massDifference = bookMassVs(dir, species,
      "h_mass_vs_stored_minus_recalculated",
      "stored mass - recalculated mass [GeV/c^{2}]",
      200, -0.02, 0.02);

    h.massvsPtminusoverPtplus = bookMassVs(dir, species,
      "h_mass_vs_pt_minus_over_pt_plus",
      "p_{T}^{-}/p_{T}^{+}",
      100, 0., 5.0);

    // Existing TH3s retained.
    h.deltaPhiVsMassVsPt = new TH3F(
      "h_delta_phi_vs_mass_vs_pt",
      TString::Format("%s #Delta#phi_{+-} vs mass vs p_{T};%s;p_{T}^{V0} [GeV/c];#Delta#phi_{+-} [rad]",
                      speciesTitle(species), massAxisTitle(species)),
      150, massMin, massMax,
      40, 0., 5.,
      96, -TMath::Pi(), TMath::Pi());

    h.decayRVsMassVsPt = new TH3F(
      "h_decay_radius_vs_mass_vs_pt",
      TString::Format("%s decay radius vs mass vs p_{T};%s;p_{T}^{V0} [GeV/c];decay radius [cm]",
                      speciesTitle(species), massAxisTitle(species)),
      150, massMin, massMax,
      40, 0., 5.,
      80, 0., 80.);

    // New QA.
    h.qT = bookMassVs(dir, species,
      "h_mass_vs_qT", "q_{T} [GeV/c]", 120, 0., 0.4);

    h.absDeltaPcaX = bookMassVs(dir, species,
      "h_mass_vs_abs_delta_pca_x",
      "|PCA_{x,1}-PCA_{x,2}| [cm]",
      100, 0., 1.);
    h.absDeltaPcaY = bookMassVs(dir, species,
      "h_mass_vs_abs_delta_pca_y",
      "|PCA_{y,1}-PCA_{y,2}| [cm]",
      100, 0., 1.);
    h.deltaPca3D = bookMassVs(dir, species,
      "h_mass_vs_delta_pca_3d",
      "|#DeltaPCA|_{3D} [cm]",
      120, 0., 2.);

    h.absDeltaPhi = bookMassVs(dir, species,
      "h_mass_vs_abs_delta_phi",
      "|#Delta#phi_{+-}| [rad]",
      144, 0., TMath::Pi());
    h.deltaEta = bookMassVs(dir, species,
      "h_mass_vs_delta_eta",
      "#Delta#eta_{+-}",
      160, -2., 2.);
    h.absDeltaEta = bookMassVs(dir, species,
      "h_mass_vs_abs_delta_eta",
      "|#Delta#eta_{+-}|",
      120, 0., 2.);
    h.deltaTheta = bookMassVs(dir, species,
      "h_mass_vs_delta_theta",
      "#Delta#theta_{+-} [rad]",
      160, -1.5, 1.5);
    h.absDeltaTheta = bookMassVs(dir, species,
      "h_mass_vs_abs_delta_theta",
      "|#Delta#theta_{+-}| [rad]",
      120, 0., 1.5);
    h.phiEtaRatio = bookMassVs(dir, species,
      "h_mass_vs_abs_dphi_over_abs_deta",
      "|#Delta#phi|/(|#Delta#eta|+10^{-3})",
      120, 0., 6.);

    h.absPtAsymmetry = bookMassVs(dir, species,
      "h_mass_vs_abs_pt_asymmetry",
      "|p_{T}^{+}-p_{T}^{-}|/(p_{T}^{+}+p_{T}^{-})",
      100, 0., 1.);
    h.pAsymmetry = bookMassVs(dir, species,
      "h_mass_vs_p_asymmetry",
      "(p^{+}-p^{-})/(p^{+}+p^{-})",
      120, -1., 1.);
    h.absPAsymmetry = bookMassVs(dir, species,
      "h_mass_vs_abs_p_asymmetry",
      "|p^{+}-p^{-}|/(p^{+}+p^{-})",
      100, 0., 1.);

    h.dcaXYPos = bookMassVs(dir, species,
      "h_mass_vs_dca_xy_positive",
      "DCA_{xy}^{+} [cm]",
      160, -4., 4.);
    h.dcaXYNeg = bookMassVs(dir, species,
      "h_mass_vs_dca_xy_negative",
      "DCA_{xy}^{-} [cm]",
      160, -4., 4.);
    h.minAbsDcaXY = bookMassVs(dir, species,
      "h_mass_vs_min_abs_dca_xy",
      "min(|DCA_{xy}^{+}|,|DCA_{xy}^{-}|) [cm]",
      120, 0., 3.);
    h.maxAbsDcaXY = bookMassVs(dir, species,
      "h_mass_vs_max_abs_dca_xy",
      "max(|DCA_{xy}^{+}|,|DCA_{xy}^{-}|) [cm]",
      120, 0., 5.);
    h.absDcaXYProduct = bookMassVs(dir, species,
      "h_mass_vs_abs_dca_xy_product",
      "|DCA_{xy}^{+}DCA_{xy}^{-}| [cm^{2}]",
      120, 0., 4.);

    h.dcaZPos = bookMassVs(dir, species,
      "h_mass_vs_dca_z_positive",
      "DCA_{z}^{+} [cm]",
      160, -8., 8.);
    h.dcaZNeg = bookMassVs(dir, species,
      "h_mass_vs_dca_z_negative",
      "DCA_{z}^{-} [cm]",
      160, -8., 8.);
    h.maxAbsDcaZ = bookMassVs(dir, species,
      "h_mass_vs_max_abs_dca_z",
      "max(|DCA_{z}^{+}|,|DCA_{z}^{-}|) [cm]",
      120, 0., 8.);

    h.log10OneMinusDira = bookMassVs(dir, species,
      "h_mass_vs_log10_one_minus_DIRA",
      "log_{10}(1-DIRA)",
      120, -8., 0.);

    h.dedxPos = bookMassVs(dir, species,
      "h_mass_vs_dedx_positive",
      "dE/dx^{+}",
      120, 0., 600.);
    h.dedxNeg = bookMassVs(dir, species,
      "h_mass_vs_dedx_negative",
      "dE/dx^{-}",
      120, 0., 600.);

    h.ptHeavyOverLight = bookMassVs(dir, species,
      "h_mass_vs_pt_heavy_over_light",
      (species == Species::K0s)
        ? "p_{T}^{+}/p_{T}^{-}"
        : "p_{T}^{p}/p_{T}^{#pi}",
      120, 0., 6.);

    h.pHeavyOverLight = bookMassVs(dir, species,
      "h_mass_vs_p_heavy_over_light",
      (species == Species::K0s)
        ? "p^{+}/p^{-}"
        : "p^{p}/p^{#pi}",
      120, 0., 6.);

    // Selected TH3s. Binning is intentionally moderate.
    h.openingVsMassVsPt = new TH3F(
      "h_opening_vs_mass_vs_pt",
      TString::Format("%s opening vs mass vs p_{T};%s;p_{T}^{V0} [GeV/c];opening [rad]",
                      speciesTitle(species), massAxisTitle(species)),
      120, massMin, massMax,
      30, 0., 5.,
      60, 0., TMath::Pi());

    h.pairDCAVsMassVsPt = new TH3F(
      "h_pairDCA_vs_mass_vs_pt",
      TString::Format("%s pair DCA vs mass vs p_{T};%s;p_{T}^{V0} [GeV/c];|pair DCA| [cm]",
                      speciesTitle(species), massAxisTitle(species)),
      120, massMin, massMax,
      30, 0., 5.,
      60, 0., 1.2);

    h.diraVsMassVsPt = new TH3F(
      "h_DIRA_vs_mass_vs_pt",
      TString::Format("%s DIRA vs mass vs p_{T};%s;p_{T}^{V0} [GeV/c];DIRA",
                      speciesTitle(species), massAxisTitle(species)),
      120, massMin, massMax,
      30, 0., 5.,
      60, 0.8, 1.0);

    h.alphaVsQTvsPt = new TH3F(
      "h_alpha_vs_qT_vs_pt",
      TString::Format("%s Armenteros vs p_{T};#alpha;q_{T} [GeV/c];p_{T}^{V0} [GeV/c]",
                      speciesTitle(species)),
      80, -1., 1.,
      60, 0., 0.3,
      30, 0., 5.);

    h.deltaPhiVsDeltaEtaVsPt = new TH3F(
      "h_abs_dphi_vs_abs_deta_vs_pt",
      TString::Format("%s daughter opening coordinates;|#Delta#eta|;|#Delta#phi| [rad];p_{T}^{V0} [GeV/c]",
                      speciesTitle(species)),
      60, 0., 1.5,
      60, 0., 1.5,
      30, 0., 5.);

    h.minNpointsVsRVsPt = new TH3F(
      "h_npoints_min_vs_decayR_vs_pt",
      TString::Format("%s min Npoints vs R vs p_{T};decay radius [cm];p_{T}^{V0} [GeV/c];min N_{points}",
                      speciesTitle(species)),
      60, 0., 60.,
      30, 0., 5.,
      50, 20., 70.);

    h.pairDCAVsRVsPt = new TH3F(
      "h_pairDCA_vs_decayR_vs_pt",
      TString::Format("%s pair DCA vs R vs p_{T};decay radius [cm];p_{T}^{V0} [GeV/c];|pair DCA| [cm]",
                      speciesTitle(species)),
      60, 0., 60.,
      30, 0., 5.,
      60, 0., 1.2);

    h.diraVsRVsPt = new TH3F(
      "h_DIRA_vs_decayR_vs_pt",
      TString::Format("%s DIRA vs R vs p_{T};decay radius [cm];p_{T}^{V0} [GeV/c];DIRA",
                      speciesTitle(species)),
      60, 0., 60.,
      30, 0., 5.,
      60, 0.8, 1.0);

    return h;
  }

  void fillMassVsQA(MassVsQA& h,
                    const Species species,
                    double mass,
                    const EventKinematics& k,
                    double recalculatedMass,
                    bool fillDeltaPhiHistograms = true)
  {
    h.v0Pt->Fill(k.v0Pt, mass);
    h.v0Phi->Fill(k.v0Phi, mass);
    h.v0Eta->Fill(k.v0Eta, mass);
    h.decayR->Fill(k.decayR, mass);
    h.decayPhi->Fill(k.decayPhi, mass);
    h.pcaZ->Fill(k.pcaZ, mass);
    h.deltaPcaZ->Fill(std::abs(k.deltaPcaZ), mass);
    h.pairDCA->Fill(k.pairDCA, mass);
    h.dira->Fill(k.dira, mass);
    h.alpha->Fill(k.alpha, mass);
    h.absAlpha->Fill(std::abs(k.alpha), mass);

    h.ptPos->Fill(k.ptPos, mass);
    h.ptNeg->Fill(k.ptNeg, mass);
    h.ptMin->Fill(std::min(k.ptPos, k.ptNeg), mass);
    h.ptMax->Fill(std::max(k.ptPos, k.ptNeg), mass);
    h.ptAsymmetry->Fill(k.ptAsymSigned, mass);

    h.etaPos->Fill(k.etaPos, mass);
    h.etaNeg->Fill(k.etaNeg, mass);
    h.etaMin->Fill(std::min(k.etaPos, k.etaNeg), mass);
    h.etaMax->Fill(std::max(k.etaPos, k.etaNeg), mass);

    h.phiPos->Fill(k.phiPos, mass);
    h.phiNeg->Fill(k.phiNeg, mass);
    if (fillDeltaPhiHistograms)
      h.deltaPhi->Fill(k.deltaPhi, mass);
    h.opening->Fill(k.opening, mass);

    if (k.ptPos > 0.) h.qOverPtPos->Fill(1.0/k.ptPos, mass);
    if (k.ptNeg > 0.) h.qOverPtNeg->Fill(-1.0/k.ptNeg, mass);

    h.pzPos->Fill(k.pzPos, mass);
    h.pzNeg->Fill(k.pzNeg, mass);

    h.npointsMin->Fill(k.npointsMin, mass);
    h.npointsMax->Fill(k.npointsMax, mass);

    h.qualityMin->Fill(k.qualityMin, mass);
    h.qualityMax->Fill(k.qualityMax, mass);

    h.massRecalc->Fill(recalculatedMass, mass);
    h.massDifference->Fill(mass - recalculatedMass, mass);

    if (k.ptPos > 0.)
      h.massvsPtminusoverPtplus->Fill(k.ptNeg/k.ptPos, mass);

    // Existing TH3s.
    if (fillDeltaPhiHistograms)
      h.deltaPhiVsMassVsPt->Fill(mass, k.v0Pt, k.deltaPhi);
    h.decayRVsMassVsPt->Fill(mass, k.v0Pt, k.decayR);

    // New QA.
    h.qT->Fill(k.qT, mass);
    h.absDeltaPcaX->Fill(std::abs(k.deltaPcaX), mass);
    h.absDeltaPcaY->Fill(std::abs(k.deltaPcaY), mass);
    h.deltaPca3D->Fill(k.deltaPca3D, mass);

    h.absDeltaPhi->Fill(k.absDeltaPhi, mass);
    h.deltaEta->Fill(k.deltaEta, mass);
    h.absDeltaEta->Fill(k.absDeltaEta, mass);
    h.deltaTheta->Fill(k.deltaTheta, mass);
    h.absDeltaTheta->Fill(k.absDeltaTheta, mass);
    h.phiEtaRatio->Fill(k.phiEtaRatio, mass);

    h.absPtAsymmetry->Fill(k.ptAsymAbs, mass);
    h.pAsymmetry->Fill(k.pAsymSigned, mass);
    h.absPAsymmetry->Fill(k.pAsymAbs, mass);

    h.dcaXYPos->Fill(k.dcaXYPos, mass);
    h.dcaXYNeg->Fill(k.dcaXYNeg, mass);
    h.minAbsDcaXY->Fill(k.minAbsDcaXY, mass);
    h.maxAbsDcaXY->Fill(k.maxAbsDcaXY, mass);
    h.absDcaXYProduct->Fill(k.absDcaXYProduct, mass);

    h.dcaZPos->Fill(k.dcaZPos, mass);
    h.dcaZNeg->Fill(k.dcaZNeg, mass);
    h.maxAbsDcaZ->Fill(k.maxAbsDcaZ, mass);

    h.log10OneMinusDira->Fill(safeLog10OneMinusDira(k.dira), mass);

    h.dedxPos->Fill(k.dedxPos, mass);
    h.dedxNeg->Fill(k.dedxNeg, mass);

    double ptHeavy = 0.;
    double ptLight = 0.;
    double pHeavy = 0.;
    double pLight = 0.;

    heavyLightKinematics(species, k, ptHeavy, ptLight, pHeavy, pLight);

    if (ptLight > 0.) h.ptHeavyOverLight->Fill(ptHeavy/ptLight, mass);
    if (pLight > 0.) h.pHeavyOverLight->Fill(pHeavy/pLight, mass);

    h.openingVsMassVsPt->Fill(mass, k.v0Pt, k.opening);
    h.pairDCAVsMassVsPt->Fill(mass, k.v0Pt, k.pairDCA);
    h.diraVsMassVsPt->Fill(mass, k.v0Pt, k.dira);

    h.alphaVsQTvsPt->Fill(k.alpha, k.qT, k.v0Pt);
    h.deltaPhiVsDeltaEtaVsPt->Fill(
      k.absDeltaEta, k.absDeltaPhi, k.v0Pt);

    h.minNpointsVsRVsPt->Fill(
      k.decayR, k.v0Pt, k.npointsMin);

    h.pairDCAVsRVsPt->Fill(
      k.decayR, k.v0Pt, k.pairDCA);

    h.diraVsRVsPt->Fill(
      k.decayR, k.v0Pt, k.dira);
  }

  // --------------------------------------------------------------------------
  // Existing K0S low/high window maps retained.
  // --------------------------------------------------------------------------

  struct WindowMaps
  {
    TH2F* phiPosVsPhiNeg = nullptr;
    TH2F* ptPosVsPtNeg = nullptr;
    TH2F* etaPosVsEtaNeg = nullptr;
    TH3F* decayRVsMassVsPt = nullptr;
    TH2F* decayPhiVsR = nullptr;
    TH2F* v0PhiVsPt = nullptr;
    TH2F* phiPosVsPtPos = nullptr;
    TH2F* phiNegVsPtNeg = nullptr;
  };

  WindowMaps bookWindowMaps(TDirectory* dir, const char* label)
  {
    dir->cd();

    WindowMaps h;

    h.phiPosVsPhiNeg = new TH2F(
      "h_phi_positive_vs_phi_negative",
      TString::Format("%s;#phi^{-} [rad];#phi^{+} [rad]", label),
      144, -TMath::Pi(), TMath::Pi(),
      144, -TMath::Pi(), TMath::Pi());

    h.ptPosVsPtNeg = new TH2F(
      "h_pt_positive_vs_pt_negative",
      TString::Format("%s;p_{T}^{-} [GeV/c];p_{T}^{+} [GeV/c]", label),
      100, 0., 4.,
      100, 0., 4.);

    h.etaPosVsEtaNeg = new TH2F(
      "h_eta_positive_vs_eta_negative",
      TString::Format("%s;#eta^{-};#eta^{+}", label),
      120, -1.5, 1.5,
      120, -1.5, 1.5);

    h.decayRVsMassVsPt = new TH3F(
      "h_decay_radius_vs_mass_vs_pt",
      TString::Format("%s;m_{#pi#pi} [GeV/c^{2}];p_{T}^{V0} [GeV/c];decay radius [cm]",
                      label),
      150, 0.42, 0.57,
      40, 0., 5.,
      80, 0., 80.);

    h.decayPhiVsR = new TH2F(
      "h_decay_phi_vs_radius",
      TString::Format("%s;decay radius [cm];#phi_{decay} [rad]", label),
      120, 0., 80.,
      144, -TMath::Pi(), TMath::Pi());

    h.v0PhiVsPt = new TH2F(
      "h_v0_phi_vs_v0_pt",
      TString::Format("%s;p_{T}^{V0} [GeV/c];#phi_{V0} [rad]", label),
      100, 0., 5.,
      144, -TMath::Pi(), TMath::Pi());

    h.phiPosVsPtPos = new TH2F(
      "h_phi_positive_vs_pt_positive",
      TString::Format("%s;p_{T}^{+} [GeV/c];#phi^{+} [rad]", label),
      100, 0., 4.,
      144, -TMath::Pi(), TMath::Pi());

    h.phiNegVsPtNeg = new TH2F(
      "h_phi_negative_vs_pt_negative",
      TString::Format("%s;p_{T}^{-} [GeV/c];#phi^{-} [rad]", label),
      100, 0., 4.,
      144, -TMath::Pi(), TMath::Pi());

    return h;
  }

  void fillWindowMaps(WindowMaps& h,
                      double mass,
                      const EventKinematics& k)
  {
    h.phiPosVsPhiNeg->Fill(k.phiNeg, k.phiPos);
    h.ptPosVsPtNeg->Fill(k.ptNeg, k.ptPos);
    h.etaPosVsEtaNeg->Fill(k.etaNeg, k.etaPos);
    h.decayRVsMassVsPt->Fill(mass, k.v0Pt, k.decayR);
    h.decayPhiVsR->Fill(k.decayR, k.decayPhi);
    h.v0PhiVsPt->Fill(k.v0Pt, k.v0Phi);
    h.phiPosVsPtPos->Fill(k.ptPos, k.phiPos);
    h.phiNegVsPtNeg->Fill(k.ptNeg, k.phiNeg);
  }

  // --------------------------------------------------------------------------
  // Peak/sideband topology distributions.
  // --------------------------------------------------------------------------

  struct TopologyWindowQA
  {
    TH1F* mass = nullptr;
    TH1F* v0Pt = nullptr;
    TH1F* v0Eta = nullptr;
    TH1F* decayR = nullptr;
    TH1F* decayPhi = nullptr;
    TH1F* pcaZ = nullptr;
    TH1F* deltaPcaZ = nullptr;
    TH1F* deltaPca3D = nullptr;

    TH1F* pairDCA = nullptr;
    TH1F* dira = nullptr;
    TH1F* log10OneMinusDira = nullptr;

    TH1F* alpha = nullptr;
    TH1F* qT = nullptr;

    TH1F* opening = nullptr;
    TH1F* deltaPhi = nullptr;
    TH1F* absDeltaPhi = nullptr;
    TH1F* deltaEta = nullptr;
    TH1F* absDeltaEta = nullptr;
    TH1F* deltaTheta = nullptr;
    TH1F* phiEtaRatio = nullptr;

    TH1F* ptAsymSigned = nullptr;
    TH1F* ptAsymAbs = nullptr;
    TH1F* pAsymSigned = nullptr;
    TH1F* pAsymAbs = nullptr;

    TH1F* dcaXYPos = nullptr;
    TH1F* dcaXYNeg = nullptr;
    TH1F* minAbsDcaXY = nullptr;
    TH1F* absDcaXYProduct = nullptr;
    TH1F* dcaZPos = nullptr;
    TH1F* dcaZNeg = nullptr;

    TH1F* npointsMin = nullptr;
    TH1F* npointsMax = nullptr;
    TH1F* qualityMin = nullptr;
    TH1F* qualityMax = nullptr;

    TH1F* ptHeavyOverLight = nullptr;
    TH1F* pHeavyOverLight = nullptr;

    TH2F* ptPosVsPtNeg = nullptr;
    TH2F* alphaVsQT = nullptr;
    TH2F* deltaPhiVsDeltaEta = nullptr;
    TH2F* pairDCAVsDecayR = nullptr;
    TH2F* diraVsDecayR = nullptr;
    TH2F* npointsMinVsDecayR = nullptr;
    TH2F* openingVsV0Pt = nullptr;
    TH2F* decayPhiVsR = nullptr;

    TH3F* openingVsRVsPt = nullptr;
    TH3F* pairDCAVsRVsPt = nullptr;
    TH3F* diraVsRVsPt = nullptr;
  };

  TopologyWindowQA bookTopologyWindowQA(TDirectory* dir,
                                        const Species species,
                                        const char* label)
  {
    dir->cd();

    double massMin = 0.;
    double massMax = 0.;
    speciesMassRange(species, massMin, massMax);

    TopologyWindowQA h;

    h.mass = new TH1F(
      "h_mass",
      TString::Format("%s %s;%s;candidates",
                      speciesTitle(species), label, massAxisTitle(species)),
      300, massMin, massMax);

    h.v0Pt = new TH1F(
      "h_v0_pt", TString::Format("%s %s;p_{T}^{V0} [GeV/c];candidates",
                                 speciesTitle(species), label),
      100, 0., 5.);

    h.v0Eta = new TH1F(
      "h_v0_eta", TString::Format("%s %s;#eta_{V0};candidates",
                                  speciesTitle(species), label),
      120, -1.5, 1.5);

    h.decayR = new TH1F(
      "h_decay_radius", TString::Format("%s %s;decay radius [cm];candidates",
                                        speciesTitle(species), label),
      120, 0., 80.);

    h.decayPhi = new TH1F(
      "h_decay_phi", TString::Format("%s %s;#phi_{decay};candidates",
                                     speciesTitle(species), label),
      144, -TMath::Pi(), TMath::Pi());

    h.pcaZ = new TH1F(
      "h_pca_z", TString::Format("%s %s;PCA_{z} [cm];candidates",
                                 speciesTitle(species), label),
      120, -15., 15.);

    h.deltaPcaZ = new TH1F(
      "h_abs_delta_pca_z", TString::Format("%s %s;|#DeltaPCA_{z}| [cm];candidates",
                                           speciesTitle(species), label),
      100, 0., 0.5);

    h.deltaPca3D = new TH1F(
      "h_delta_pca_3d", TString::Format("%s %s;|#DeltaPCA|_{3D} [cm];candidates",
                                        speciesTitle(species), label),
      120, 0., 2.);

    h.pairDCA = new TH1F(
      "h_pairDCA", TString::Format("%s %s;|pair DCA| [cm];candidates",
                                   speciesTitle(species), label),
      120, 0., 1.2);

    h.dira = new TH1F(
      "h_DIRA", TString::Format("%s %s;DIRA;candidates",
                                speciesTitle(species), label),
      120, 0.8, 1.0);

    h.log10OneMinusDira = new TH1F(
      "h_log10_one_minus_DIRA",
      TString::Format("%s %s;log_{10}(1-DIRA);candidates",
                      speciesTitle(species), label),
      120, -8., 0.);

    h.alpha = new TH1F(
      "h_alpha", TString::Format("%s %s;#alpha;candidates",
                                 speciesTitle(species), label),
      160, -1., 1.);

    h.qT = new TH1F(
      "h_qT", TString::Format("%s %s;q_{T} [GeV/c];candidates",
                              speciesTitle(species), label),
      120, 0., 0.4);

    h.opening = new TH1F(
      "h_opening_angle", TString::Format("%s %s;opening angle [rad];candidates",
                                         speciesTitle(species), label),
      120, 0., TMath::Pi());

    h.deltaPhi = new TH1F(
      "h_delta_phi", TString::Format("%s %s;#Delta#phi_{+-} [rad];candidates",
                                     speciesTitle(species), label),
      144, -TMath::Pi(), TMath::Pi());

    h.absDeltaPhi = new TH1F(
      "h_abs_delta_phi", TString::Format("%s %s;|#Delta#phi_{+-}| [rad];candidates",
                                         speciesTitle(species), label),
      120, 0., TMath::Pi());

    h.deltaEta = new TH1F(
      "h_delta_eta", TString::Format("%s %s;#Delta#eta_{+-};candidates",
                                     speciesTitle(species), label),
      160, -2., 2.);

    h.absDeltaEta = new TH1F(
      "h_abs_delta_eta", TString::Format("%s %s;|#Delta#eta_{+-}|;candidates",
                                         speciesTitle(species), label),
      120, 0., 2.);

    h.deltaTheta = new TH1F(
      "h_delta_theta", TString::Format("%s %s;#Delta#theta_{+-} [rad];candidates",
                                       speciesTitle(species), label),
      160, -1.5, 1.5);

    h.phiEtaRatio = new TH1F(
      "h_abs_dphi_over_abs_deta",
      TString::Format("%s %s;|#Delta#phi|/(|#Delta#eta|+10^{-3});candidates",
                      speciesTitle(species), label),
      120, 0., 6.);

    h.ptAsymSigned = new TH1F(
      "h_pt_asymmetry_signed",
      TString::Format("%s %s;(p_{T}^{+}-p_{T}^{-})/(p_{T}^{+}+p_{T}^{-});candidates",
                      speciesTitle(species), label),
      120, -1., 1.);

    h.ptAsymAbs = new TH1F(
      "h_pt_asymmetry_abs",
      TString::Format("%s %s;|p_{T}^{+}-p_{T}^{-}|/(p_{T}^{+}+p_{T}^{-});candidates",
                      speciesTitle(species), label),
      100, 0., 1.);

    h.pAsymSigned = new TH1F(
      "h_p_asymmetry_signed",
      TString::Format("%s %s;(p^{+}-p^{-})/(p^{+}+p^{-});candidates",
                      speciesTitle(species), label),
      120, -1., 1.);

    h.pAsymAbs = new TH1F(
      "h_p_asymmetry_abs",
      TString::Format("%s %s;|p^{+}-p^{-}|/(p^{+}+p^{-});candidates",
                      speciesTitle(species), label),
      100, 0., 1.);

    h.dcaXYPos = new TH1F(
      "h_dca_xy_positive",
      TString::Format("%s %s;DCA_{xy}^{+} [cm];candidates",
                      speciesTitle(species), label),
      160, -4., 4.);

    h.dcaXYNeg = new TH1F(
      "h_dca_xy_negative",
      TString::Format("%s %s;DCA_{xy}^{-} [cm];candidates",
                      speciesTitle(species), label),
      160, -4., 4.);

    h.minAbsDcaXY = new TH1F(
      "h_min_abs_dca_xy",
      TString::Format("%s %s;min |DCA_{xy}| [cm];candidates",
                      speciesTitle(species), label),
      120, 0., 3.);

    h.absDcaXYProduct = new TH1F(
      "h_abs_dca_xy_product",
      TString::Format("%s %s;|DCA_{xy}^{+}DCA_{xy}^{-}| [cm^{2}];candidates",
                      speciesTitle(species), label),
      120, 0., 4.);

    h.dcaZPos = new TH1F(
      "h_dca_z_positive",
      TString::Format("%s %s;DCA_{z}^{+} [cm];candidates",
                      speciesTitle(species), label),
      160, -8., 8.);

    h.dcaZNeg = new TH1F(
      "h_dca_z_negative",
      TString::Format("%s %s;DCA_{z}^{-} [cm];candidates",
                      speciesTitle(species), label),
      160, -8., 8.);

    h.npointsMin = new TH1F(
      "h_npoints_min",
      TString::Format("%s %s;min N_{points};candidates",
                      speciesTitle(species), label),
      50, 20., 70.);

    h.npointsMax = new TH1F(
      "h_npoints_max",
      TString::Format("%s %s;max N_{points};candidates",
                      speciesTitle(species), label),
      50, 20., 70.);

    h.qualityMin = new TH1F(
      "h_quality_min",
      TString::Format("%s %s;min quality;candidates",
                      speciesTitle(species), label),
      100, 0., 2.);

    h.qualityMax = new TH1F(
      "h_quality_max",
      TString::Format("%s %s;max quality;candidates",
                      speciesTitle(species), label),
      100, 0., 2.);

    h.ptHeavyOverLight = new TH1F(
      "h_pt_heavy_over_light",
      TString::Format("%s %s;%s;candidates",
                      speciesTitle(species), label,
                      (species == Species::K0s)
                        ? "p_{T}^{+}/p_{T}^{-}"
                        : "p_{T}^{p}/p_{T}^{#pi}"),
      120, 0., 6.);

    h.pHeavyOverLight = new TH1F(
      "h_p_heavy_over_light",
      TString::Format("%s %s;%s;candidates",
                      speciesTitle(species), label,
                      (species == Species::K0s)
                        ? "p^{+}/p^{-}"
                        : "p^{p}/p^{#pi}"),
      120, 0., 6.);

    h.ptPosVsPtNeg = new TH2F(
      "h_pt_positive_vs_pt_negative",
      TString::Format("%s %s;p_{T}^{-} [GeV/c];p_{T}^{+} [GeV/c]",
                      speciesTitle(species), label),
      100, 0., 4.,
      100, 0., 4.);

    h.alphaVsQT = new TH2F(
      "h_alpha_vs_qT",
      TString::Format("%s %s;#alpha;q_{T} [GeV/c]",
                      speciesTitle(species), label),
      160, -1., 1.,
      120, 0., 0.4);

    h.deltaPhiVsDeltaEta = new TH2F(
      "h_abs_delta_phi_vs_abs_delta_eta",
      TString::Format("%s %s;|#Delta#eta|;|#Delta#phi| [rad]",
                      speciesTitle(species), label),
      120, 0., 2.,
      120, 0., 2.);

    h.pairDCAVsDecayR = new TH2F(
      "h_pairDCA_vs_decayR",
      TString::Format("%s %s;decay radius [cm];|pair DCA| [cm]",
                      speciesTitle(species), label),
      120, 0., 80.,
      100, 0., 1.);

    h.diraVsDecayR = new TH2F(
      "h_DIRA_vs_decayR",
      TString::Format("%s %s;decay radius [cm];DIRA",
                      speciesTitle(species), label),
      120, 0., 80.,
      100, 0.8, 1.0);

    h.npointsMinVsDecayR = new TH2F(
      "h_npoints_min_vs_decayR",
      TString::Format("%s %s;decay radius [cm];min N_{points}",
                      speciesTitle(species), label),
      120, 0., 80.,
      50, 20., 70.);

    h.openingVsV0Pt = new TH2F(
      "h_opening_vs_v0_pt",
      TString::Format("%s %s;p_{T}^{V0} [GeV/c];opening angle [rad]",
                      speciesTitle(species), label),
      100, 0., 5.,
      100, 0., TMath::Pi());

    h.decayPhiVsR = new TH2F(
      "h_decay_phi_vs_radius",
      TString::Format("%s %s;decay radius [cm];#phi_{decay}",
                      speciesTitle(species), label),
      120, 0., 80.,
      144, -TMath::Pi(), TMath::Pi());

    h.openingVsRVsPt = new TH3F(
      "h_opening_vs_decayR_vs_pt",
      TString::Format("%s %s;decay radius [cm];p_{T}^{V0} [GeV/c];opening [rad]",
                      speciesTitle(species), label),
      60, 0., 60.,
      30, 0., 5.,
      60, 0., TMath::Pi());

    h.pairDCAVsRVsPt = new TH3F(
      "h_pairDCA_vs_decayR_vs_pt",
      TString::Format("%s %s;decay radius [cm];p_{T}^{V0} [GeV/c];|pair DCA| [cm]",
                      speciesTitle(species), label),
      60, 0., 60.,
      30, 0., 5.,
      60, 0., 1.2);

    h.diraVsRVsPt = new TH3F(
      "h_DIRA_vs_decayR_vs_pt",
      TString::Format("%s %s;decay radius [cm];p_{T}^{V0} [GeV/c];DIRA",
                      speciesTitle(species), label),
      60, 0., 60.,
      30, 0., 5.,
      60, 0.8, 1.0);

    return h;
  }

  void fillTopologyWindowQA(TopologyWindowQA& h,
                            const Species species,
                            double mass,
                            const EventKinematics& k)
  {
    h.mass->Fill(mass);
    h.v0Pt->Fill(k.v0Pt);
    h.v0Eta->Fill(k.v0Eta);
    h.decayR->Fill(k.decayR);
    h.decayPhi->Fill(k.decayPhi);
    h.pcaZ->Fill(k.pcaZ);
    h.deltaPcaZ->Fill(std::abs(k.deltaPcaZ));
    h.deltaPca3D->Fill(k.deltaPca3D);

    h.pairDCA->Fill(k.pairDCA);
    h.dira->Fill(k.dira);
    h.log10OneMinusDira->Fill(safeLog10OneMinusDira(k.dira));

    h.alpha->Fill(k.alpha);
    h.qT->Fill(k.qT);

    h.opening->Fill(k.opening);
    h.deltaPhi->Fill(k.deltaPhi);
    h.absDeltaPhi->Fill(k.absDeltaPhi);
    h.deltaEta->Fill(k.deltaEta);
    h.absDeltaEta->Fill(k.absDeltaEta);
    h.deltaTheta->Fill(k.deltaTheta);
    h.phiEtaRatio->Fill(k.phiEtaRatio);

    h.ptAsymSigned->Fill(k.ptAsymSigned);
    h.ptAsymAbs->Fill(k.ptAsymAbs);
    h.pAsymSigned->Fill(k.pAsymSigned);
    h.pAsymAbs->Fill(k.pAsymAbs);

    h.dcaXYPos->Fill(k.dcaXYPos);
    h.dcaXYNeg->Fill(k.dcaXYNeg);
    h.minAbsDcaXY->Fill(k.minAbsDcaXY);
    h.absDcaXYProduct->Fill(k.absDcaXYProduct);
    h.dcaZPos->Fill(k.dcaZPos);
    h.dcaZNeg->Fill(k.dcaZNeg);

    h.npointsMin->Fill(k.npointsMin);
    h.npointsMax->Fill(k.npointsMax);
    h.qualityMin->Fill(k.qualityMin);
    h.qualityMax->Fill(k.qualityMax);

    double ptHeavy = 0.;
    double ptLight = 0.;
    double pHeavy = 0.;
    double pLight = 0.;

    heavyLightKinematics(species, k, ptHeavy, ptLight, pHeavy, pLight);

    if (ptLight > 0.) h.ptHeavyOverLight->Fill(ptHeavy/ptLight);
    if (pLight > 0.) h.pHeavyOverLight->Fill(pHeavy/pLight);

    h.ptPosVsPtNeg->Fill(k.ptNeg, k.ptPos);
    h.alphaVsQT->Fill(k.alpha, k.qT);
    h.deltaPhiVsDeltaEta->Fill(k.absDeltaEta, k.absDeltaPhi);
    h.pairDCAVsDecayR->Fill(k.decayR, k.pairDCA);
    h.diraVsDecayR->Fill(k.decayR, k.dira);
    h.npointsMinVsDecayR->Fill(k.decayR, k.npointsMin);
    h.openingVsV0Pt->Fill(k.v0Pt, k.opening);
    h.decayPhiVsR->Fill(k.decayR, k.decayPhi);

    h.openingVsRVsPt->Fill(k.decayR, k.v0Pt, k.opening);
    h.pairDCAVsRVsPt->Fill(k.decayR, k.v0Pt, k.pairDCA);
    h.diraVsRVsPt->Fill(k.decayR, k.v0Pt, k.dira);
  }

  struct SpeciesWindows
  {
    double peakMin = 0.;
    double peakMax = 0.;
    double leftMin = 0.;
    double leftMax = 0.;
    double rightMin = 0.;
    double rightMax = 0.;
  };

  struct SpeciesOutput
  {
    MassVsQA massQA;

    TH2F* massVsDeltaPhiPreCut = nullptr;
    TH2F* massVsDeltaPhiMargin = nullptr;

    TopologyWindowQA peak;
    TopologyWindowQA sidebandLeft;
    TopologyWindowQA sidebandRight;
    TopologyWindowQA sidebandsCombined;

    // Same QA as above, filled after exactCut1 + unlike but BEFORE the
    // signed (sailor/cowboy) Delta-phi cut.
    MassVsQA massQAPreCut;

    TopologyWindowQA peakPreCut;
    TopologyWindowQA sidebandLeftPreCut;
    TopologyWindowQA sidebandRightPreCut;
    TopologyWindowQA sidebandsCombinedPreCut;

    SpeciesWindows windows;
  };

  bool inside(double x, double lo, double hi)
  {
    return x >= lo && x < hi;
  }

  double massForSpecies(const Species species,
                        double massK0s,
                        double massLambda,
                        double massAntiLambda)
  {
    if (species == Species::K0s) return massK0s;
    if (species == Species::Lambda) return massLambda;
    return massAntiLambda;
  }

  double recalcMassForSpecies(const Species species,
                              const EventKinematics& k)
  {
    if (species == Species::K0s)
    {
      return safeMassHypothesis(
        k.pxPos, k.pyPos, k.pzPos, kPionMass,
        k.pxNeg, k.pyNeg, k.pzNeg, kPionMass);
    }

    if (species == Species::Lambda)
    {
      return safeMassHypothesis(
        k.pxPos, k.pyPos, k.pzPos, kProtonMass,
        k.pxNeg, k.pyNeg, k.pzNeg, kPionMass);
    }

    return safeMassHypothesis(
      k.pxPos, k.pyPos, k.pzPos, kPionMass,
      k.pxNeg, k.pyNeg, k.pzNeg, kProtonMass);
  }

  SpeciesOutput bookSpeciesOutput(TDirectory* speciesDir,
                                  const Species species,
                                  const SpeciesWindows& windows)
  {
    SpeciesOutput out;
    out.windows = windows;

    TDirectory* massDir = speciesDir->mkdir("mass_vs_phase_space");
    out.massQA = bookMassVsQA(massDir, species);

    double massMin = 0.;
    double massMax = 0.;
    speciesMassRange(species, massMin, massMax);

    TDirectory* preCutDir = speciesDir->mkdir("pre_deltaPhi_cut");
    preCutDir->cd();

    out.massVsDeltaPhiPreCut = new TH2F(
      "h_mass_vs_delta_phi_pre_cut",
      TString::Format("%s after exactCut1, before signed #Delta#phi cut;#Delta#phi_{+-} [rad];%s",
                      speciesTitle(species), massAxisTitle(species)),
      144, -TMath::Pi(), TMath::Pi(),
      300, massMin, massMax);

    out.massVsDeltaPhiMargin = new TH2F(
      "h_mass_vs_delta_phi_cut_margin",
      TString::Format("%s signed #Delta#phi cut margin;#Delta#phi - threshold [rad];%s",
                      speciesTitle(species), massAxisTitle(species)),
      160, -1.0, 1.0,
      300, massMin, massMax);

    // Full QA before the signed Delta-phi cut, mirroring the post-cut layout.
    TDirectory* preMassDir = preCutDir->mkdir("mass_vs_phase_space");
    out.massQAPreCut = bookMassVsQA(preMassDir, species);

    TDirectory* prePeakDir = preCutDir->mkdir("peak");
    out.peakPreCut = bookTopologyWindowQA(
      prePeakDir, species,
      TString::Format("pre #Delta#phi cut, peak %.4f-%.4f", windows.peakMin, windows.peakMax));

    TDirectory* preLeftDir = preCutDir->mkdir("sideband_left");
    out.sidebandLeftPreCut = bookTopologyWindowQA(
      preLeftDir, species,
      TString::Format("pre #Delta#phi cut, left sideband %.4f-%.4f", windows.leftMin, windows.leftMax));

    TDirectory* preRightDir = preCutDir->mkdir("sideband_right");
    out.sidebandRightPreCut = bookTopologyWindowQA(
      preRightDir, species,
      TString::Format("pre #Delta#phi cut, right sideband %.4f-%.4f", windows.rightMin, windows.rightMax));

    TDirectory* preCombinedDir = preCutDir->mkdir("sidebands_combined");
    out.sidebandsCombinedPreCut = bookTopologyWindowQA(
      preCombinedDir, species,
      "pre #Delta#phi cut, left + right sidebands");

    TDirectory* peakDir = speciesDir->mkdir("peak");
    out.peak = bookTopologyWindowQA(
      peakDir, species,
      TString::Format("peak %.4f-%.4f", windows.peakMin, windows.peakMax));

    TDirectory* leftDir = speciesDir->mkdir("sideband_left");
    out.sidebandLeft = bookTopologyWindowQA(
      leftDir, species,
      TString::Format("left sideband %.4f-%.4f", windows.leftMin, windows.leftMax));

    TDirectory* rightDir = speciesDir->mkdir("sideband_right");
    out.sidebandRight = bookTopologyWindowQA(
      rightDir, species,
      TString::Format("right sideband %.4f-%.4f", windows.rightMin, windows.rightMax));

    TDirectory* combinedDir = speciesDir->mkdir("sidebands_combined");
    out.sidebandsCombined = bookTopologyWindowQA(
      combinedDir, species,
      "left + right sidebands");

    return out;
  }

  void fillSpeciesWindows(SpeciesOutput& out,
                          const Species species,
                          double mass,
                          const EventKinematics& k,
                          bool preCut = false)
  {
    const auto& w = out.windows;

    TopologyWindowQA& peak = preCut ? out.peakPreCut : out.peak;
    TopologyWindowQA& left = preCut ? out.sidebandLeftPreCut : out.sidebandLeft;
    TopologyWindowQA& right = preCut ? out.sidebandRightPreCut : out.sidebandRight;
    TopologyWindowQA& combined = preCut ? out.sidebandsCombinedPreCut : out.sidebandsCombined;

    if (inside(mass, w.peakMin, w.peakMax))
    {
      fillTopologyWindowQA(peak, species, mass, k);
    }

    if (inside(mass, w.leftMin, w.leftMax))
    {
      fillTopologyWindowQA(left, species, mass, k);
      fillTopologyWindowQA(combined, species, mass, k);
    }

    if (inside(mass, w.rightMin, w.rightMax))
    {
      fillTopologyWindowQA(right, species, mass, k);
      fillTopologyWindowQA(combined, species, mass, k);
    }
  }
}

// ============================================================================

void StudyV0TopologyQA(
  const char* inputDir = "/sphenix/tg/tg01/hf/mitrankova/Hadrons/",
  const char* filePattern = "hadron_sitpc_HITS_pp_mvtxS300_79513_*.root",
  const char* outputDir = "output",
  const char* outputName = "v0_topology_qa_mvtxS300.root",
  const char* treeName = "pairTree",

  // Existing K0S double-peak windows.
  double lowMassMin = 0.470,
  double lowMassMax = 0.490,
  double highMassMin = 0.490,
  double highMassMax = 0.510,

  Long64_t maxEntries = -1,

  // Explicit replacement for the old in-place quality *= 0.1.
  double qualityScale = 0.1,

  // K0S peak and sidebands.
  double k0sPeakMin = 0.485,
  double k0sPeakMax = 0.510,
  double k0sLeftSidebandMin = 0.440,
  double k0sLeftSidebandMax = 0.470,
  double k0sRightSidebandMin = 0.525,
  double k0sRightSidebandMax = 0.555,

  // Lambda peak and sidebands.
  double lambdaPeakMin = 1.108,
  double lambdaPeakMax = 1.124,
  double lambdaLeftSidebandMin = 1.080,
  double lambdaLeftSidebandMax = 1.100,
  double lambdaRightSidebandMin = 1.132,
  double lambdaRightSidebandMax = 1.152,

  // Si+TPC setup
  const char* trackClass = "all",   // "all", "tpconly", "si", "mixed"
  int minSiHits = 4,                // "si": both daughters with >= minSiHits Si points
  bool applySignedDeltaPhiCut = false,
  bool usePairVertex = true,
  double minDecayRadius = 0.05)     // exactCut1 decay radius [cm]
{
  TH1::AddDirectory(kTRUE);

  const TString chainPattern =
    TString::Format("%s/%s", inputDir, filePattern);

  TChain chain(treeName);
  const int nFiles = chain.Add(chainPattern);

  if (nFiles <= 0)
  {
    std::cerr
      << "ERROR: no files matched "
      << chainPattern
      << std::endl;
    return;
  }

  const std::vector<std::string> requiredBranches = {
    "mass_Kshort",
    "mass_Lambda",
    "mass_AntiLambda",
    "v0_pt",
    "pca_x", "pca_y", "pca_z",
    "pca1_x", "pca1_y", "pca1_z",
    "pca2_x", "pca2_y", "pca2_z",
    "px1", "py1", "pz1",
    "px2", "py2", "pz2",
    "v0_px", "v0_py", "v0_pz",
    "alpha", "qT", "pairDCA",
    "dca_xy1", "dca_z1",
    "dca_xy2", "dca_z2",
    "npoints1", "npoints2",
    "charge1", "charge2",
    "quality1", "quality2",
    "dedx_1", "dedx_2"
  };

  bool missing = false;

  for (const auto& name : requiredBranches)
  {
    if (!branchExists(chain, name.c_str()))
    {
      std::cerr
        << "ERROR: missing branch "
        << name
        << std::endl;
      missing = true;
    }
  }

  const std::string trackClassName = trackClass ? trackClass : "all";
  if (trackClassName != "all" && trackClassName != "tpconly" &&
      trackClassName != "si" && trackClassName != "mixed")
  {
    std::cerr << "ERROR: unknown trackClass " << trackClassName << std::endl;
    return;
  }
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
    missing = true;
  }

  if (missing) return;

  Float_t mass_Kshort = 0.f;
  Float_t mass_Lambda = 0.f;
  Float_t mass_AntiLambda = 0.f;
  Float_t v0_pt = 0.f;

  Float_t pca_x = 0.f;
  Float_t pca_y = 0.f;
  Float_t pca_z = 0.f;

  Float_t pca1_x = 0.f;
  Float_t pca1_y = 0.f;
  Float_t pca1_z = 0.f;

  Float_t pca2_x = 0.f;
  Float_t pca2_y = 0.f;
  Float_t pca2_z = 0.f;

  Float_t px1 = 0.f;
  Float_t py1 = 0.f;
  Float_t pz1 = 0.f;

  Float_t px2 = 0.f;
  Float_t py2 = 0.f;
  Float_t pz2 = 0.f;

  Float_t v0_px = 0.f;
  Float_t v0_py = 0.f;
  Float_t v0_pz = 0.f;

  Float_t alpha = 0.f;
  Float_t qT = 0.f;
  Float_t pairDCA = 0.f;

  Float_t dca_xy1 = 0.f;
  Float_t dca_z1 = 0.f;
  Float_t dca_xy2 = 0.f;
  Float_t dca_z2 = 0.f;

  Float_t charge1 = 0.f;
  Float_t charge2 = 0.f;

  Float_t quality1 = 0.f;
  Float_t quality2 = 0.f;

  Short_t npoints1 = 0;
  Short_t npoints2 = 0;

  Float_t dedx_1 = 0.f;
  Float_t dedx_2 = 0.f;

  chain.SetBranchAddress("mass_Kshort", &mass_Kshort);
  chain.SetBranchAddress("mass_Lambda", &mass_Lambda);
  chain.SetBranchAddress("mass_AntiLambda", &mass_AntiLambda);
  chain.SetBranchAddress("v0_pt", &v0_pt);

  chain.SetBranchAddress("pca_x", &pca_x);
  chain.SetBranchAddress("pca_y", &pca_y);
  chain.SetBranchAddress("pca_z", &pca_z);

  chain.SetBranchAddress("pca1_x", &pca1_x);
  chain.SetBranchAddress("pca1_y", &pca1_y);
  chain.SetBranchAddress("pca1_z", &pca1_z);

  chain.SetBranchAddress("pca2_x", &pca2_x);
  chain.SetBranchAddress("pca2_y", &pca2_y);
  chain.SetBranchAddress("pca2_z", &pca2_z);

  chain.SetBranchAddress("px1", &px1);
  chain.SetBranchAddress("py1", &py1);
  chain.SetBranchAddress("pz1", &pz1);

  chain.SetBranchAddress("px2", &px2);
  chain.SetBranchAddress("py2", &py2);
  chain.SetBranchAddress("pz2", &pz2);

  chain.SetBranchAddress("v0_px", &v0_px);
  chain.SetBranchAddress("v0_py", &v0_py);
  chain.SetBranchAddress("v0_pz", &v0_pz);

  chain.SetBranchAddress("alpha", &alpha);
  chain.SetBranchAddress("qT", &qT);
  chain.SetBranchAddress("pairDCA", &pairDCA);

  chain.SetBranchAddress("dca_xy1", &dca_xy1);
  chain.SetBranchAddress("dca_z1", &dca_z1);
  chain.SetBranchAddress("dca_xy2", &dca_xy2);
  chain.SetBranchAddress("dca_z2", &dca_z2);

  chain.SetBranchAddress("charge1", &charge1);
  chain.SetBranchAddress("charge2", &charge2);

  chain.SetBranchAddress("quality1", &quality1);
  chain.SetBranchAddress("quality2", &quality2);

  chain.SetBranchAddress("npoints1", &npoints1);
  chain.SetBranchAddress("npoints2", &npoints2);

  chain.SetBranchAddress("dedx_1", &dedx_1);
  chain.SetBranchAddress("dedx_2", &dedx_2);

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

  gSystem->mkdir(outputDir, kTRUE);

  const TString outputPath =
    TString::Format("%s/%s", outputDir, outputName);

  std::unique_ptr<TFile> output(
    TFile::Open(outputPath, "RECREATE"));

  if (!output || output->IsZombie())
  {
    std::cerr
      << "ERROR: cannot create "
      << outputPath
      << std::endl;
    return;
  }

  // --------------------------------------------------------------------------
  // Metadata.
  // --------------------------------------------------------------------------

  output->cd();

  TNamed(
    "input_pattern",
    chainPattern.Data()).Write();

  TNamed(
    "selection_note",
    "exactCut1/exactCut2 and the signed pT-dependent Delta-phi cut are retained "
    "from the supplied StudyK0sDoublePeakQA.C. Peak/sideband topology QA is "
    "filled after exactCut1 and after the signed Delta-phi cut. The same full QA "
    "is also filled before the signed Delta-phi cut under pre_deltaPhi_cut/ "
    "(top level for legacy K0S, and <Species>/pre_deltaPhi_cut/).").Write();

  TParameter<double>("qualityScale", qualityScale).Write();
  TNamed("track_class",
         TString::Format("trackClass=%s, minSiHits=%d, signedDeltaPhiCut=%d, pairVertex=%d, minDecayRadius=%.3f",
                         trackClassName.c_str(), minSiHits, applySignedDeltaPhiCut ? 1 : 0,
                         havePairVertex ? 1 : 0, minDecayRadius).Data()).Write();

  TParameter<double>("k0sPeakMin", k0sPeakMin).Write();
  TParameter<double>("k0sPeakMax", k0sPeakMax).Write();
  TParameter<double>("k0sLeftSidebandMin", k0sLeftSidebandMin).Write();
  TParameter<double>("k0sLeftSidebandMax", k0sLeftSidebandMax).Write();
  TParameter<double>("k0sRightSidebandMin", k0sRightSidebandMin).Write();
  TParameter<double>("k0sRightSidebandMax", k0sRightSidebandMax).Write();

  TParameter<double>("lambdaPeakMin", lambdaPeakMin).Write();
  TParameter<double>("lambdaPeakMax", lambdaPeakMax).Write();
  TParameter<double>("lambdaLeftSidebandMin", lambdaLeftSidebandMin).Write();
  TParameter<double>("lambdaLeftSidebandMax", lambdaLeftSidebandMax).Write();
  TParameter<double>("lambdaRightSidebandMin", lambdaRightSidebandMin).Write();
  TParameter<double>("lambdaRightSidebandMax", lambdaRightSidebandMax).Write();

  // --------------------------------------------------------------------------
  // Legacy K0S objects: keep the old names and directories.
  // --------------------------------------------------------------------------

  TDirectory* legacyMassDir =
    output->mkdir("mass_vs_phase_space_exactCut1_unlike");

  MassVsQA legacyK0sMassQA =
    bookMassVsQA(legacyMassDir, Species::K0s);

  output->cd();

  TH1F* hMassUnlike = new TH1F(
    "h_mass_exactCut1_unlike",
    "exactCut1 unlike sign;m_{#pi#pi} [GeV/c^{2}];candidates",
    600, 0.42, 0.57);

  TH1F* hMassLike = new TH1F(
    "h_mass_exactCut1_like",
    "exactCut1 like sign;m_{#pi#pi} [GeV/c^{2}];candidates",
    600, 0.42, 0.57);

  TH2F* hMassVsV0PtLike = new TH2F(
    "h_mass_vs_v0_pt_exactCut1_like",
    "exactCut1 like sign;p_{T}^{V0} [GeV/c];m_{#pi#pi} [GeV/c^{2}]",
    100, 0., 5.,
    300, 0.42, 0.57);

  TDirectory* lowDir =
    output->mkdir("low_mass_window");

  WindowMaps lowMaps =
    bookWindowMaps(
      lowDir,
      TString::Format(
        "low mass %.3f-%.3f GeV/c^{2}",
        lowMassMin, lowMassMax));

  TDirectory* highDir =
    output->mkdir("high_mass_window");

  WindowMaps highMaps =
    bookWindowMaps(
      highDir,
      TString::Format(
        "high mass %.3f-%.3f GeV/c^{2}",
        highMassMin, highMassMax));

  // --------------------------------------------------------------------------
  // Legacy K0S objects booked again, filled BEFORE the signed Delta-phi cut.
  // Same names as above, stored under pre_deltaPhi_cut/.
  // --------------------------------------------------------------------------

  TDirectory* preCutTopDir =
    output->mkdir("pre_deltaPhi_cut");

  TDirectory* preLegacyMassDir =
    preCutTopDir->mkdir("mass_vs_phase_space_exactCut1_unlike");

  MassVsQA preLegacyK0sMassQA =
    bookMassVsQA(preLegacyMassDir, Species::K0s);

  preCutTopDir->cd();

  TH1F* hMassUnlikePreCut = new TH1F(
    "h_mass_exactCut1_unlike",
    "exactCut1 unlike sign, before #Delta#phi cut;m_{#pi#pi} [GeV/c^{2}];candidates",
    600, 0.42, 0.57);

  TH1F* hMassLikePreCut = new TH1F(
    "h_mass_exactCut1_like",
    "exactCut1 like sign, before #Delta#phi cut;m_{#pi#pi} [GeV/c^{2}];candidates",
    600, 0.42, 0.57);

  TH2F* hMassVsV0PtLikePreCut = new TH2F(
    "h_mass_vs_v0_pt_exactCut1_like",
    "exactCut1 like sign, before #Delta#phi cut;p_{T}^{V0} [GeV/c];m_{#pi#pi} [GeV/c^{2}]",
    100, 0., 5.,
    300, 0.42, 0.57);

  TDirectory* preLowDir =
    preCutTopDir->mkdir("low_mass_window");

  WindowMaps preLowMaps =
    bookWindowMaps(
      preLowDir,
      TString::Format(
        "pre #Delta#phi cut, low mass %.3f-%.3f GeV/c^{2}",
        lowMassMin, lowMassMax));

  TDirectory* preHighDir =
    preCutTopDir->mkdir("high_mass_window");

  WindowMaps preHighMaps =
    bookWindowMaps(
      preHighDir,
      TString::Format(
        "pre #Delta#phi cut, high mass %.3f-%.3f GeV/c^{2}",
        highMassMin, highMassMax));

  output->cd();

  // --------------------------------------------------------------------------
  // New species-organized QA.
  // --------------------------------------------------------------------------

  SpeciesWindows k0sWindows;
  k0sWindows.peakMin = k0sPeakMin;
  k0sWindows.peakMax = k0sPeakMax;
  k0sWindows.leftMin = k0sLeftSidebandMin;
  k0sWindows.leftMax = k0sLeftSidebandMax;
  k0sWindows.rightMin = k0sRightSidebandMin;
  k0sWindows.rightMax = k0sRightSidebandMax;

  SpeciesWindows lambdaWindows;
  lambdaWindows.peakMin = lambdaPeakMin;
  lambdaWindows.peakMax = lambdaPeakMax;
  lambdaWindows.leftMin = lambdaLeftSidebandMin;
  lambdaWindows.leftMax = lambdaLeftSidebandMax;
  lambdaWindows.rightMin = lambdaRightSidebandMin;
  lambdaWindows.rightMax = lambdaRightSidebandMax;

  TDirectory* k0sDir = output->mkdir("K0s");
  SpeciesOutput k0sOutput =
    bookSpeciesOutput(k0sDir, Species::K0s, k0sWindows);

  TDirectory* lambdaDir = output->mkdir("Lambda");
  SpeciesOutput lambdaOutput =
    bookSpeciesOutput(lambdaDir, Species::Lambda, lambdaWindows);

  TDirectory* antiLambdaDir = output->mkdir("AntiLambda");
  SpeciesOutput antiLambdaOutput =
    bookSpeciesOutput(antiLambdaDir, Species::AntiLambda, lambdaWindows);

  output->cd();

  TH1I* hCutflow = new TH1I(
    "h_cutflow",
    "common V0 cut flow;stage;pairs",
    6, 0., 6.);

  hCutflow->GetXaxis()->SetBinLabel(1, "all entries");
  hCutflow->GetXaxis()->SetBinLabel(2, "exactCut1");
  hCutflow->GetXaxis()->SetBinLabel(3, "unlike");
  hCutflow->GetXaxis()->SetBinLabel(4, "like");
  hCutflow->GetXaxis()->SetBinLabel(5, "signed dphi pass");
  hCutflow->GetXaxis()->SetBinLabel(6, "exactCut2 + unlike");

  const Long64_t totalEntries =
    chain.GetEntries();

  const Long64_t nEntries =
    (maxEntries >= 0)
      ? std::min(totalEntries, maxEntries)
      : totalEntries;

  std::cout
    << "Added " << nFiles
    << " files, processing "
    << nEntries
    << " / "
    << totalEntries
    << " entries"
    << std::endl;

  std::cout
    << "qualityScale = "
    << qualityScale
    << std::endl;

  std::cout
    << "K0S low window:  ["
    << lowMassMin << ", "
    << lowMassMax << "]\n"
    << "K0S high window: ["
    << highMassMin << ", "
    << highMassMax << "]"
    << std::endl;

  Long64_t nExactUnlike = 0;
  Long64_t nExactLike = 0;
  Long64_t nFinalUnlike = 0;

  // --------------------------------------------------------------------------
  // Event loop.
  // --------------------------------------------------------------------------

  for (Long64_t entry = 0; entry < nEntries; ++entry)
  {
    chain.GetEntry(entry);

    if (entry % 1000000 == 0)
    {
      std::cout
        << "Processing "
        << entry
        << " / "
        << nEntries
        << std::endl;
    }

    hCutflow->Fill(0.5);

    // daughter class: TPC-only / Si (>= minSiHits Si points)
    if (trackClassName != "all")
    {
      const bool si1 = has_si1 != 0 && static_cast<int>(nmvtx1 + nintt1) >= minSiHits;
      const bool si2 = has_si2 != 0 && static_cast<int>(nmvtx2 + nintt2) >= minSiHits;
      const bool tpc1 = has_si1 == 0;
      const bool tpc2 = has_si2 == 0;
      const bool keep =
        (trackClassName == "si" && si1 && si2) ||
        (trackClassName == "tpconly" && tpc1 && tpc2) ||
        (trackClassName == "mixed" && ((si1 && tpc2) || (tpc1 && si2)));
      if (!keep)
        continue;
    }

    // flight from the pair's collision vertex (Si+TPC) or from the origin
    const bool pairVertexValid =
      havePairVertex && std::isfinite(pv_x) && std::isfinite(pv_y) && std::isfinite(pv_z);
    const double flightX = pairVertexValid ? pca_x - pv_x : pca_x;
    const double flightY = pairVertexValid ? pca_y - pv_y : pca_y;
    const double flightZ = pairVertexValid ? pca_z - pv_z : pca_z;

    // Preserve the old 0.1 behavior, but do it explicitly and locally.
    const double scaledQuality1 =
      quality1 * qualityScale;

    const double scaledQuality2 =
      quality2 * qualityScale;

    const double pt1 =
      std::hypot(px1, py1);

    const double pt2 =
      std::hypot(px2, py2);

    const double absDeltaPcaZ =
      std::abs(pca1_z - pca2_z);

    const double decayRadius =
      std::hypot(flightX, flightY);

    const double absAlpha =
      std::abs(alpha);

    const double absPairDCA =
      std::abs(pairDCA);

    const double pMag =
      std::sqrt(
        v0_px*v0_px +
        v0_py*v0_py +
        v0_pz*v0_pz);

    const double rMag =
      std::sqrt(
        flightX*flightX +
        flightY*flightY +
        flightZ*flightZ);

    const double dira =
      (pMag > 0. && rMag > 0.)
        ? (v0_px*flightX +
           v0_py*flightY +
           v0_pz*flightZ)/(pMag*rMag)
        : -2.;

    // exactCut1: loose V0 preselection for the Si+TPC setup (no dE/dx, no quality cut; the
    // previous version required decay radius > 2 cm, dE/dx < 400, npoints > 30, quality < 1.5).
    (void)scaledQuality1;
    const bool exactCut1 =
      pca_z > -20.f &&
      pca_z < 20.f &&
      absDeltaPcaZ < 1.0 &&
      pt1 > 0.15 &&
      pt2 > 0.15 &&
      decayRadius > minDecayRadius &&
      absAlpha < 0.9999 &&
      absPairDCA < 1.0 &&
      dira > 0.5 &&
      npoints1 > 12 &&
      npoints2 > 12;

    const bool exactCut2 =
      pca_z > -10.f &&
      pca_z < 10.f &&
      absDeltaPcaZ < 0.2 &&
      pt1 > 0.2 &&
      pt2 > 0.2 &&
      decayRadius > 2.0 &&
      absAlpha < 0.99 &&
      absPairDCA < 0.5 &&
      dira > 0.95 &&
      npoints1 > 30 &&
      npoints2 > 30 &&
      dedx_1 < 400 &&
      dedx_2 < 400 &&
      scaledQuality1 < 0.5 &&
      scaledQuality2 < 0.5;

    if (!exactCut1)
      continue;

    hCutflow->Fill(1.5);

    const bool unlike =
      charge1 * charge2 < 0.f;

    const bool like =
      charge1 * charge2 > 0.f;

    if (!unlike && !like)
      continue;

    if (unlike)
    {
      hCutflow->Fill(2.5);
      ++nExactUnlike;
    }

    if (like)
    {
      hCutflow->Fill(3.5);
      ++nExactLike;
    }

    // Charge ordering.
    const bool track1Positive =
      charge1 > 0.f;

    EventKinematics k;

    k.pxPos = track1Positive ? px1 : px2;
    k.pyPos = track1Positive ? py1 : py2;
    k.pzPos = track1Positive ? pz1 : pz2;

    k.pxNeg = track1Positive ? px2 : px1;
    k.pyNeg = track1Positive ? py2 : py1;
    k.pzNeg = track1Positive ? pz2 : pz1;

    k.ptPos = std::hypot(k.pxPos, k.pyPos);
    k.ptNeg = std::hypot(k.pxNeg, k.pyNeg);

    k.pPos = std::sqrt(
      k.pxPos*k.pxPos +
      k.pyPos*k.pyPos +
      k.pzPos*k.pzPos);

    k.pNeg = std::sqrt(
      k.pxNeg*k.pxNeg +
      k.pyNeg*k.pyNeg +
      k.pzNeg*k.pzNeg);

    k.phiPos = std::atan2(k.pyPos, k.pxPos);
    k.phiNeg = std::atan2(k.pyNeg, k.pxNeg);

    k.etaPos = safeEta(k.pxPos, k.pyPos, k.pzPos);
    k.etaNeg = safeEta(k.pxNeg, k.pyNeg, k.pzNeg);

    k.dcaXYPos = track1Positive ? dca_xy1 : dca_xy2;
    k.dcaXYNeg = track1Positive ? dca_xy2 : dca_xy1;

    k.dcaZPos = track1Positive ? dca_z1 : dca_z2;
    k.dcaZNeg = track1Positive ? dca_z2 : dca_z1;

    k.dedxPos = track1Positive ? dedx_1 : dedx_2;
    k.dedxNeg = track1Positive ? dedx_2 : dedx_1;

    k.npointsPos = track1Positive ? npoints1 : npoints2;
    k.npointsNeg = track1Positive ? npoints2 : npoints1;

    k.qualityPos =
      track1Positive ? scaledQuality1 : scaledQuality2;

    k.qualityNeg =
      track1Positive ? scaledQuality2 : scaledQuality1;

    k.v0Pt = v0_pt;
    k.v0Phi = std::atan2(v0_py, v0_px);
    k.v0Eta = safeEta(v0_px, v0_py, v0_pz);

    k.decayR = decayRadius;
    k.decayPhi = std::atan2(pca_y, pca_x);
    k.pcaZ = pca_z;

    k.deltaPcaX = pca1_x - pca2_x;
    k.deltaPcaY = pca1_y - pca2_y;
    k.deltaPcaZ = pca1_z - pca2_z;

    k.deltaPca3D =
      std::sqrt(
        k.deltaPcaX*k.deltaPcaX +
        k.deltaPcaY*k.deltaPcaY +
        k.deltaPcaZ*k.deltaPcaZ);

    k.pairDCA = absPairDCA;
    k.dira = dira;
    k.alpha = alpha;
    k.qT = qT;

    k.opening =
      openingAngle(
        k.pxPos, k.pyPos, k.pzPos,
        k.pxNeg, k.pyNeg, k.pzNeg);

    k.deltaPhi =
      wrapPhi(k.phiPos - k.phiNeg);

    k.absDeltaPhi =
      std::abs(k.deltaPhi);

    k.deltaEta =
      k.etaPos - k.etaNeg;

    k.absDeltaEta =
      std::abs(k.deltaEta);

    const double thetaPos =
      std::atan2(k.ptPos, k.pzPos);

    const double thetaNeg =
      std::atan2(k.ptNeg, k.pzNeg);

    k.deltaTheta =
      thetaPos - thetaNeg;

    k.absDeltaTheta =
      std::abs(k.deltaTheta);

    k.phiEtaRatio =
      k.absDeltaPhi/(k.absDeltaEta + 1.e-3);

    const double ptSum =
      k.ptPos + k.ptNeg;

    k.ptAsymSigned =
      (ptSum > 0.)
        ? (k.ptPos - k.ptNeg)/ptSum
        : 0.;

    k.ptAsymAbs =
      std::abs(k.ptAsymSigned);

    const double pSum =
      k.pPos + k.pNeg;

    k.pAsymSigned =
      (pSum > 0.)
        ? (k.pPos - k.pNeg)/pSum
        : 0.;

    k.pAsymAbs =
      std::abs(k.pAsymSigned);

    k.minAbsDcaXY =
      std::min(
        std::abs(k.dcaXYPos),
        std::abs(k.dcaXYNeg));

    k.maxAbsDcaXY =
      std::max(
        std::abs(k.dcaXYPos),
        std::abs(k.dcaXYNeg));

    k.absDcaXYProduct =
      std::abs(
        k.dcaXYPos *
        k.dcaXYNeg);

    k.maxAbsDcaZ =
      std::max(
        std::abs(k.dcaZPos),
        std::abs(k.dcaZNeg));

    k.npointsMin =
      std::min(
        k.npointsPos,
        k.npointsNeg);

    k.npointsMax =
      std::max(
        k.npointsPos,
        k.npointsNeg);

    k.qualityMin =
      std::min(
        k.qualityPos,
        k.qualityNeg);

    k.qualityMax =
      std::max(
        k.qualityPos,
        k.qualityNeg);

    const double deltaPhiThreshold =
      0.8 -
      0.4 *
      ((v0_pt < 2.0) ? v0_pt : 2.0);

    // New pre-cut QA for all three mass hypotheses.
    if (unlike)
    {
      for (const Species species :
           {Species::K0s, Species::Lambda, Species::AntiLambda})
      {
        const double mass =
          massForSpecies(
            species,
            mass_Kshort,
            mass_Lambda,
            mass_AntiLambda);

        SpeciesOutput* out = nullptr;

        if (species == Species::K0s)
          out = &k0sOutput;
        else if (species == Species::Lambda)
          out = &lambdaOutput;
        else
          out = &antiLambdaOutput;

        out->massVsDeltaPhiPreCut->Fill(
          k.deltaPhi, mass);

        out->massVsDeltaPhiMargin->Fill(
          k.deltaPhi - deltaPhiThreshold,
          mass);
      }
    }

    // Preserve the old exactCut2 K0S pre-cut diagnostic.
    if (unlike && exactCut2)
    {
      hCutflow->Fill(5.5);

      legacyK0sMassQA.deltaPhi->Fill(
        k.deltaPhi,
        mass_Kshort);

      legacyK0sMassQA.deltaPhiVsMassVsPt->Fill(
        mass_Kshort,
        v0_pt,
        k.deltaPhi);
    }

    // ------------------------------------------------------------------------
    // Full QA BEFORE the signed Delta-phi cut: same content as the post-cut
    // block below, written to the pre_deltaPhi_cut directories.
    // ------------------------------------------------------------------------

    if (like)
    {
      hMassLikePreCut->Fill(
        mass_Kshort);

      hMassVsV0PtLikePreCut->Fill(
        v0_pt,
        mass_Kshort);
    }

    if (unlike)
    {
      hMassUnlikePreCut->Fill(
        mass_Kshort);

      fillMassVsQA(
        preLegacyK0sMassQA,
        Species::K0s,
        mass_Kshort,
        k,
        recalcMassForSpecies(Species::K0s, k),
        true);  // no Delta-phi cut applied yet, so fill Delta-phi histograms too

      if (mass_Kshort >= lowMassMin &&
          mass_Kshort < lowMassMax)
      {
        fillWindowMaps(
          preLowMaps,
          mass_Kshort,
          k);
      }

      if (mass_Kshort >= highMassMin &&
          mass_Kshort < highMassMax)
      {
        fillWindowMaps(
          preHighMaps,
          mass_Kshort,
          k);
      }

      for (const Species species :
           {Species::K0s, Species::Lambda, Species::AntiLambda})
      {
        const double mass =
          massForSpecies(
            species,
            mass_Kshort,
            mass_Lambda,
            mass_AntiLambda);

        const double recalcMass =
          recalcMassForSpecies(
            species,
            k);

        SpeciesOutput* out = nullptr;

        if (species == Species::K0s)
          out = &k0sOutput;
        else if (species == Species::Lambda)
          out = &lambdaOutput;
        else
          out = &antiLambdaOutput;

        fillMassVsQA(
          out->massQAPreCut,
          species,
          mass,
          k,
          recalcMass);

        fillSpeciesWindows(
          *out,
          species,
          mass,
          k,
          true);
      }
    }

    // Signed Delta-phi (sailor/cowboy) selection, off by default.
    if (applySignedDeltaPhiCut && k.deltaPhi < deltaPhiThreshold)
    {
      continue;
    }

    hCutflow->Fill(4.5);

    if (like)
    {
      hMassLike->Fill(
        mass_Kshort);

      hMassVsV0PtLike->Fill(
        v0_pt,
        mass_Kshort);

      continue;
    }

    ++nFinalUnlike;

    // ------------------------------------------------------------------------
    // Legacy K0S QA retained after the signed Delta-phi cut.
    // ------------------------------------------------------------------------

    hMassUnlike->Fill(
      mass_Kshort);

    const double k0sMassRecalc =
      recalcMassForSpecies(
        Species::K0s,
        k);

    fillMassVsQA(
      legacyK0sMassQA,
      Species::K0s,
      mass_Kshort,
      k,
      k0sMassRecalc,
      false);  // preserve legacy deltaPhi histograms as exactCut2 pre-cut QA

    if (mass_Kshort >= lowMassMin &&
        mass_Kshort < lowMassMax)
    {
      fillWindowMaps(
        lowMaps,
        mass_Kshort,
        k);
    }

    if (mass_Kshort >= highMassMin &&
        mass_Kshort < highMassMax)
    {
      fillWindowMaps(
        highMaps,
        mass_Kshort,
        k);
    }

    // ------------------------------------------------------------------------
    // Species-organized QA.
    // ------------------------------------------------------------------------

    for (const Species species :
         {Species::K0s, Species::Lambda, Species::AntiLambda})
    {
      const double mass =
        massForSpecies(
          species,
          mass_Kshort,
          mass_Lambda,
          mass_AntiLambda);

      const double recalcMass =
        recalcMassForSpecies(
          species,
          k);

      SpeciesOutput* out = nullptr;

      if (species == Species::K0s)
        out = &k0sOutput;
      else if (species == Species::Lambda)
        out = &lambdaOutput;
      else
        out = &antiLambdaOutput;

      fillMassVsQA(
        out->massQA,
        species,
        mass,
        k,
        recalcMass);

      fillSpeciesWindows(
        *out,
        species,
        mass,
        k);
    }
  }

  // --------------------------------------------------------------------------
  // Finish.
  // --------------------------------------------------------------------------

  output->Write();
  output->Close();

  std::cout
    << "exactCut1 unlike candidates: "
    << nExactUnlike
    << std::endl;

  std::cout
    << "exactCut1 like candidates:   "
    << nExactLike
    << std::endl;

  std::cout
    << "final unlike after signed Delta-phi cut: "
    << nFinalUnlike
    << std::endl;

  std::cout
    << "Wrote output: "
    << outputPath
    << std::endl;
}