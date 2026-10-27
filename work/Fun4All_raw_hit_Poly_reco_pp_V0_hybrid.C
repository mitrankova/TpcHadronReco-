/*
 * Full reconstruction -> hadron reconstruction with the HYBRID Kalman fit.
 *
 * Goal: recover the Kalman S/B advantage while keeping the TPC curvature (1/pT)
 * unbiased by the Si-TPC transverse kink.
 *
 * Diagnosis from the sitpc vs Kalman EL100 comparison:
 *   - sitpc  (Taubin + hybrid XY): K0S shift ~+0.7 MeV, phi shift ~+0.6 MeV
 *            S/B cut06 = 3.5 (Taubin pair-DCA is wide -> loose background)
 *   - Kalman (field map, MVTX sigma 50 um): K0S shift -2.4 .. -7.7 MeV (Si pulls curvature)
 *            S/B cut06 = 7.9 (Kalman 3-D pair-DCA is tight -> good background)
 *
 * The fix: Kalman refit with field map, but inflate the MVTX rphi sigma from
 * 50 um to 150 um.  Si still anchors the position and z0 (z weights unchanged);
 * at 150 um the MVTX weight on curvature drops by 9x and can no longer resolve
 * the Si-TPC kink against ~50 TPC clusters, so 1/pT stays at the TPC value.
 * Expected outcome: K0S mass within ~1 MeV of PDG, S/B intermediate between
 * sitpc and Kalman EL100, pair-DCA resolution from the full Kalman track state.
 *
 * Parameters to tune (all marked TUNE below):
 *   mvtxRPhiSigmaCm   - MVTX rphi Kalman sigma [cm]; scan 0.010, 0.015, 0.020, 0.030
 *                       0.005 = current Kalman (biased), 0.150 = near-sitpc (unbiased)
 *   kalmanFitIter     - Kalman fit iterations; 2 iterations re-linearise after outlier removal
 *   kalmanMsSigma     - multiple-scattering process noise; 0 disables, 1.0 = default
 *
 * Run command (scan three sigma values on the same 500 events):
 *   root -l -b -q 'Fun4All_raw_hit_Poly_reco_pp_V0_hybrid.C(500,79513,0,".",0,"run3pp","ana561_2025p013_v001","HITS_pp","physics",true,false,0.015)'
 *   root -l -b -q 'Fun4All_raw_hit_Poly_reco_pp_V0_hybrid.C(500,79513,0,".",0,"run3pp","ana561_2025p013_v001","HITS_pp","physics",true,false,0.020)'
 *   root -l -b -q 'Fun4All_raw_hit_Poly_reco_pp_V0_hybrid.C(500,79513,0,".",0,"run3pp","ana561_2025p013_v001","HITS_pp","physics",true,false,0.030)'
 *
 * Output: hadron_sitpc_<outfilename>_mvtxS<tag>_<run>_<seg>.root
 *
 * Everything from the TPC frame, Si alignment, seeding, resolver, matcher and
 * vertex finder is IDENTICAL to Fun4All_raw_hit_Poly_reco_pp_V0_ELscan.C
 * (tpcFrameMode 5, hybridXY, dE/dx EL100 = 1e-4 GeV/cm fixed).
 * Only the Kalman measurement sigma for MVTX rphi is changed.
 */

#include <GlobalVariables.C>

#include <G4_ActsGeom.C>
#include <G4_Global.C>
#include <G4_Magnet.C>
#include <G4_Mbd.C>
#include <QA.C>
#include <Trkr_Clustering.C>
#include <Trkr_LaserClustering.C>
#include <Trkr_Reco.C>
#include <Trkr_RecoInit.C>
#include <Trkr_TpcReadoutInit.C>

#include <phgarfield/PHGarfield.h>
#include <tpcconditions/TpcConditionsReco.h>
#include <tpctrackreco/TpcCrossingFinder.h>
#include <tpctrackreco/Tpc_ModuleTrackReco.h>
#include <tpctrackreco/Tpc_AssembledTrackReco.h>
#include <tpctrackreco/Tpc_PolyTrackReco.h>
#include <tpctrackreco/Tpc_PolyTrackVertexer.h>
#include <tpctrackreco/Tpc_PolyClusterizer.h>
#include <tpctrackreco/Tpc_PolyTripletReco.h>

#include <sitrackreco/SiHitSeedReco.h>
#include <sitrackreco/SiTpcBeamAlignment.h>
#include <sitrackreco/SiTpcTrackMatcher.h>
#include <sitrackreco/SiTrajectoryFitter.h>
#include <sitrackreco/TpcCrossingResolver.h>
#include <sitrackreco/SiTpc_VertexFinder.h>

#include <trackingdiagnostics/SiTpc_EventDisplay.h>
#include <trackingdiagnostics/SiTpc_TrackResiduals.h>
#include <trackingdiagnostics/SiTpc_VertexQA.h>

#include <tpchadronreco/TpcV0CandidateTree.h>

#include <cdbobjects/CDBTTree.h>
#include <ffamodules/CDBInterface.h>
#include <ffamodules/FlagHandler.h>

#include <fun4all/Fun4AllDstInputManager.h>
#include <fun4all/Fun4AllDstOutputManager.h>
#include <fun4all/Fun4AllInputManager.h>
#include <fun4all/Fun4AllOutputManager.h>
#include <fun4all/Fun4AllRunNodeInputManager.h>
#include <fun4all/Fun4AllServer.h>
#include <fun4all/Fun4AllUtils.h>
#include <fun4all/SubsysReco.h>
#include <fun4all/Fun4AllReturnCodes.h>

#include <phool/recoConsts.h>
#include <phool/getClass.h>

#include <string>
#include <vector>
#include <format>
#include <cmath>

R__LOAD_LIBRARY(libfun4all.so)
R__LOAD_LIBRARY(libffamodules.so)
R__LOAD_LIBRARY(libphool.so)
R__LOAD_LIBRARY(libcdbobjects.so)
R__LOAD_LIBRARY(libmvtx.so)
R__LOAD_LIBRARY(libintt.so)
R__LOAD_LIBRARY(libtpc.so)
R__LOAD_LIBRARY(libmicromegas.so)
R__LOAD_LIBRARY(libPHGarfield.so)
R__LOAD_LIBRARY(libtpctrackreco.so)
R__LOAD_LIBRARY(libsitrackreco.so)
R__LOAD_LIBRARY(libTrackingDiagnostics.so)
R__LOAD_LIBRARY(libTpcHadronReco.so)


class SkipFirstN : public SubsysReco {
 public:
  explicit SkipFirstN(int n) : SubsysReco("SkipFirstN"), target_(n) {}
  int process_event(PHCompositeNode* /*unused*/) override {
    if (count_ < target_) { ++count_; return Fun4AllReturnCodes::ABORTEVENT; }
    return Fun4AllReturnCodes::EVENT_OK;
  }
 private:
  int target_ = 0;
  int count_  = 0;
};


void Fun4All_raw_hit_Poly_reco_pp_V0_hybrid(
    const int nEvents              = 500,
    const int runnumber            = 79513,
    const int segment              = 0,
    const std::string& outdir      = ".",
    const int nSkip                = 0,
    const std::string& collision   = "run3pp",
    const std::string& production  = "ana561_2025p013_v001",
    const std::string& outfilename = "HITS_pp",
    const std::string& datatype    = "physics",
    const bool runHadronReco       = true,
    const bool runQA               = false,
    // ── TUNE: MVTX rphi Kalman sigma [cm] ────────────────────────────────
    // 0.005 = original Kalman (Si pulls curvature, K0S shift -2.4 .. -7.7 MeV)
    // 0.010 = mild deweighting; try first
    // 0.015 = moderate; expected ~halfway between Kalman and sitpc
    // 0.020 = recommended starting point based on 7 MeV shift / sqrt(9) lever arm
    // 0.030 = strong deweighting; K0S mass should approach sitpc +0.7 MeV
    // 0.150 = near-sitpc (TPC dominates curvature completely)
    const double mvtxRPhiSigmaCm  = 0.030)
{
  int MyVerbosity = 0;

  // ── Frame / alignment: identical to ELscan macro (tpcFrameMode 5) ─────
  const int    tpcFrameMode          = 5;
  const bool   useSurveyFieldGeometry= false;
  const bool   useOfficialSiAlignment= true;
  const std::string siAlignParamsFile= useOfficialSiAlignment ? "si_align_params.txt"
                                                              : "si_align_params_zero.txt";
  const double tpcLineXcm = 0.0499;   // mode 5
  const double tpcLineYcm = 0.2725;   // mode 5
  const bool   hardwareFromActs      = true;
  const double inttPhysOffsetXcm     = -0.019;
  const double inttPhysOffsetYcm     =  0.034;
  const bool   hybridXY              = true;

  // ── dE/dx: EL0 (no correction) ───────────────────────────────────────
  // The EL0 vs EL100 comparison showed that EL100 shifts the K0S mass
  // further BELOW PDG at tight topology cuts (−2.2 MeV at cut06), opposite
  // to the expected direction for a physical energy-loss correction.
  // The sign reversal arises because the Kalman energy-loss term modifies
  // how the filter weights Si vs TPC hits, amplifying the Si-TPC kink pull
  // on curvature at tight pairDCA cuts.  Use EL0 until the Si-TPC alignment
  // is corrected; the hybrid sigma scan already targets the kink directly.
  const double kalmanDedxGeVPerCm    = 0.0;

  // ── Output tag encodes the MVTX sigma so runs cannot overwrite each other
  const int mvtxSigmaUm = static_cast<int>(std::lround(mvtxRPhiSigmaCm * 1.0e4));
  const std::string outname = outfilename + std::format("_mvtxS{}", mvtxSigmaUm);

  // ─────────────────────────────────────────────────────────────────────
  const bool convertSeeds = true;
  auto* se = Fun4AllServer::instance();
  se->Verbosity(1);
  auto* rc = recoConsts::instance();
  se->registerSubsystem(new SkipFirstN(nSkip));

  const std::string dsttype        = "STREAMING_EVENT";
  const std::string dsttype_to_save= "TPC";

  // ── Input streams: identical to Fun4All_raw_hit_Poly_reco_pp.C ──────────
  std::vector<std::string> streams;

  // TPC EBDC streams
  for (int ebdc = 0; ebdc < 24; ++ebdc)
  {
    for (int endpoint = 0; endpoint < 2; ++endpoint)
    {
      std::ostringstream s;
      s << "ebdc" << std::setw(2) << std::setfill('0') << ebdc << "_" << endpoint;
      streams.push_back(s.str());
    }
  }
  if (collision != "run3line_laser" && collision != "run3cosmics")
  {
    // TPOT
    streams.emplace_back("ebdc39");
    // INTT streams
    for (int server = 0; server < 8; ++server)
    {
      std::ostringstream s;
      s << "intt" << server;
      streams.push_back(s.str());
    }
    // MVTX streams
    for (int felix = 0; felix < 6; ++felix)
    {
      std::ostringstream s;
      s << "mvtx" << felix;
      streams.push_back(s.str());
    }
  }

  int i = 0;

  int rounded_up   = 100 * (int)(std::ceil( (float)runnumber / 100));
  int rounded_down = 100 * (int)(std::floor((float)runnumber / 100));
  std::stringstream nice_rounded_up;
  nice_rounded_up   << std::setw(8) << std::setfill('0') << rounded_up;
  std::stringstream nice_rounded_down;
  nice_rounded_down << std::setw(8) << std::setfill('0') << rounded_down;

  for (auto& stream : streams)
  {
    std::string filename = std::format("DST_{}_{}_{}_{}-{:08d}-{:05d}.root",
                                       dsttype, stream, collision, production,
                                       runnumber, segment);
    std::string filepath = std::format(
        "/sphenix/lustre01/sphnxpro/production/{}/{}/{}/DST_{}_{}/run_{}_{}/{}",
        collision, datatype, production, dsttype, stream,
        nice_rounded_down.str(), nice_rounded_up.str(), filename);
    std::cout << "Adding DST: " << filepath << std::endl;
    if (i == 0)
    {
      rc->set_IntFlag("RUNNUMBER", runnumber);
      rc->set_uint64Flag("TIMESTAMP", runnumber);
    }
    std::string inputname = "InputManager" + std::to_string(i);
    auto* hitsin = new Fun4AllDstInputManager(inputname);
    hitsin->fileopen(filename);   // bare filename, same as Fun4All_raw_hit_Poly_reco_pp.C
    se->registerInputManager(hitsin);
    ++i;
  }

  rc->set_IntFlag("RUNNUMBER", runnumber);
  rc->set_IntFlag("RUNSEGMENT", segment);
  Enable::QA  = false;
  Enable::CDB = true;
  rc->set_StringFlag("CDB_GLOBALTAG", "newcdbtag");
  rc->set_uint64Flag("TIMESTAMP", runnumber);

  G4TRACKING::convert_seeds_to_svtxtracks = convertSeeds;
  TRACKING::streaming_mode = true;

  std::cout << " run: "     << runnumber
            << " samples: " << TRACKING::reco_tpc_maxtime_sample
            << " pre: "     << TRACKING::reco_tpc_time_presample
            << " vdrift: "  << G4TPC::tpc_drift_velocity_reco
            << std::endl;

  se->registerSubsystem(new FlagHandler());

  const std::string geofile = CDBInterface::instance()->getUrl("Tracking_Geometry");
  auto* ingeo = new Fun4AllRunNodeInputManager("GeoIn");
  ingeo->AddFile(geofile);
  se->registerInputManager(ingeo);

  TpcReadoutInit(runnumber);
  G4TPC::REJECT_LASER_EVENTS = true;
  TRACKING::tpc_zero_supp    = true;
  TrackingInit();

  // ── Hit unpacking ─────────────────────────────────────────────────────
  for (int f = 0; f < 6; ++f) Mvtx_HitUnpacking(std::to_string(f));
  for (int s = 0; s < 8; ++s) Intt_HitUnpacking(std::to_string(s));
  std::ostringstream ebdcname;
  for (int ebdc = 0; ebdc < 24; ++ebdc)
    for (int ep = 0; ep < 2; ++ep) {
      ebdcname.str("");
      if (ebdc < 10) ebdcname << "0";
      ebdcname << ebdc << "_" << ep;
      Tpc_HitUnpacking(ebdcname.str());
    }
  Micromegas_HitUnpacking();

  // ── Clustering ────────────────────────────────────────────────────────
  Mvtx_Clustering();
  Intt_Clustering();
  Micromegas_Clustering();
  Tpc_LaserEventIdentifying();
  Reject_Laser_Events();

  // ── TPC track finding ─────────────────────────────────────────────────
  auto* garfield = new PHGarfield();
  garfield->SetUseSurveyGeometry(useSurveyFieldGeometry);
  se->registerSubsystem(garfield);

  se->registerSubsystem(new Tpc_ModuleTrackReco());
  se->registerSubsystem(new Tpc_AssembledTrackReco());

  // Triplet seeder (streaming mode)
  const bool useTripletSeeds = true;
  const bool triggeredMode   = false;

  auto* triplet = new Tpc_PolyTripletReco();
  triplet->setTriggeredMode(triggeredMode);
  triplet->setEdgePadMargin(1);
  triplet->setMinLayerSpan(10);
  se->registerSubsystem(triplet);

  // ── Si frame constants ────────────────────────────────────────────────
  const double siDispXmm  =  6.2955, siDispYmm = -1.06874, siDispZmm = -6.66289;
  const bool   correctSiBeamSpot = true;
  const double siBeamX0cm = -0.070755, siBeamY0cm = 0.139325;
  const double siBeamDxDz = -8.932876e-04, siBeamDyDz = 9.177683e-05;
  const double siUphiRotation   = 0.1481;
  const double siCenterXmm = siDispXmm,  siCenterYmm = siDispYmm, siCenterZmm = 0.0;
  const double inttCenterXmm = 0.4027, inttCenterYmm = -2.8866,  inttCenterZmm = 0.0;

  SiTpcBeamAlignment beamAlign;
  beamAlign.setEnabled(correctSiBeamSpot);
  beamAlign.setCenterBeam(true);
  beamAlign.setSiPerHalf(false);
  beamAlign.setBeamLineCm(SiTpcBeamAlignment::SiFull, siBeamX0cm, siBeamY0cm, siBeamDxDz, siBeamDyDz);
  beamAlign.setBeamLineCm(SiTpcBeamAlignment::Tpc,    tpcLineXcm, tpcLineYcm, 0.0, 0.0);
  beamAlign.print();

  // ── Si seeding ────────────────────────────────────────────────────────
  auto* siHitSeed = new SiHitSeedReco("SiHitSeedReco");
  siHitSeed->Verbosity(MyVerbosity);
  siHitSeed->setUphiRotation(siUphiRotation);
  siHitSeed->setDetectorCenterMm(siCenterXmm, siCenterYmm, siCenterZmm);
  siHitSeed->setInttCenterMm(inttCenterXmm, inttCenterYmm, inttCenterZmm);
  siHitSeed->setBeamAlignment(beamAlign);
  siHitSeed->setApplySiAlignment(true);
  siHitSeed->setSiAlignmentParamsFile(siAlignParamsFile);
  siHitSeed->setUseNewSiliconRotationOrder(false);
  siHitSeed->setUseFlatSensorGeometry(true);
  siHitSeed->setCompareWithActsGeometry(true);
  siHitSeed->setUseActsGeometry(true);
  siHitSeed->setActsZShiftCm(-siDispZmm / 10.0);
  siHitSeed->setHardwareFromActs(hardwareFromActs);
  siHitSeed->setInttPhysicsOffsetCm(inttPhysOffsetXcm, inttPhysOffsetYcm);
  const double siLayerZOffsetCm[7] = {0, 0, 0, 0, 0, 0, 0};
  for (int l = 0; l < 7; ++l) siHitSeed->setSiLayerZOffsetCm(l, siLayerZOffsetCm[l]);
  siHitSeed->setBlobsFromPixelAdjacency(true);
  siHitSeed->setInttBlobsPerTimeBucket(true);
  siHitSeed->setAnchorWindow(1.0, 1.0);
  siHitSeed->setSeedZHalfWindowCm(0.5);
  siHitSeed->setMvtxPropagationZFromTpc(true);
  siHitSeed->setPropagationZHalfWindowBins(2.0);
  siHitSeed->setInttCentre(SiHitSeedReco::InttCentre::TpcOffset);
  siHitSeed->setInttTpcPhiHalfWindowBins(4.0);
  siHitSeed->setInttPropagationZHalfWindowBins(5.0);
  siHitSeed->setTpcAgreementZModel(SiHitSeedReco::TpcZModel::OffsetPrior);
  siHitSeed->setTpcAgreementZOffsetSigma(0.35);
  siHitSeed->setTpcAgreementZWeight(1.0);
  siHitSeed->setMinLayers(2);
  siHitSeed->setMinMvtxLayers(2);
  siHitSeed->setTpcAgreementSigmaLayer(0, 0.37, 0.16);
  siHitSeed->setTpcAgreementSigmaLayer(1, 0.45, 0.10);
  siHitSeed->setTpcAgreementSigmaLayer(2, 0.54, 0.16);
  siHitSeed->setTpcAgreementSigmaLayer(3, 0.36, 0.90);
  siHitSeed->setTpcAgreementSigmaLayer(4, 0.46, 0.74);
  siHitSeed->setTpcAgreementSigmaLayer(5, 0.43, 0.62);
  siHitSeed->setTpcAgreementSigmaLayer(6, 0.47, 0.66);
  if (useTripletSeeds) {
    siHitSeed->setTpcTrajectoryNodeName("TPC_CROSSINGTRAJECTORIES");
    siHitSeed->setMinTpcClusters(3);
    if (!triggeredMode) siHitSeed->setInttCrossingWindow(1);
  }
  se->registerSubsystem(siHitSeed);

  // ── Crossing resolver ─────────────────────────────────────────────────
  auto* resolver = new TpcCrossingResolver();
  resolver->setTriggeredMode(triggeredMode);
  resolver->setDzSigma(0.4);
  resolver->setNoInttFactor(0.5);
  resolver->setMinLoneCredit(0.08);
  resolver->setAmbiguityRatio(1.5);
  resolver->setSiZSigma(0.22);
  resolver->setClusterZWindow(0.7);
  resolver->setLoneIgnoreSameSi(true);
  resolver->setRequireInttForCollision(true);
  resolver->setMergeSplitCollisions(true);
  resolver->setMergeMaxCrossings(2);
  resolver->setMergeZNSigma(3.0);
  resolver->setMergeZWindow(1.0);
  resolver->setMergeMaxDeltaChi2(3.0);
  resolver->setMergeTier0MinTracks(3);
  resolver->setKeepUntimedLone(true);
  resolver->setMinUntimedLoneCredit(-1);
  resolver->setVertexChainReselect(true);
  resolver->Verbosity(MyVerbosity);
  se->registerSubsystem(resolver);

  // ── Si trajectory fitter ──────────────────────────────────────────────
  auto* siTraj = new SiTrajectoryFitter("SiTrajectoryFitter");
  siTraj->Verbosity(MyVerbosity);
  siTraj->setInputNodeName("SI_HIT_SEED_EVENT");
  siTraj->setOutputNodeName("SI_TRAJECTORY");
  siTraj->setFitSelectedChainsOnly(true);
  siTraj->setUphiRotation(siUphiRotation);
  siTraj->setDetectorCenterMm(siCenterXmm, siCenterYmm, siCenterZmm);
  siTraj->setInttCenterMm(inttCenterXmm, inttCenterYmm, inttCenterZmm);
  siTraj->setBeamAlignment(beamAlign);
  siTraj->setBeamPositionCm(0.0, 0.0);
  siTraj->setUseIntt(true);
  siTraj->setMinPoints(3);
  siTraj->setZWeights(1.0, 1.0e-4);
  siTraj->setMinCurvatureLeverArm(3.0);
  se->registerSubsystem(siTraj);

  // ── Full TPC clustering (after resolver) ──────────────────────────────
  auto* cluster = new Tpc_PolyClusterizer();
  cluster->setReverseDriftSubsteps(1);
  se->registerSubsystem(cluster);
  se->registerSubsystem(new Tpc_PolyTrackReco());
  se->registerSubsystem(new Tpc_PolyTrackVertexer());

  // ── Si+TPC matcher: hybrid XY ─────────────────────────────────────────
  auto* siTpcMatch = new SiTpcTrackMatcher("SiTpcTrackMatcher");
  siTpcMatch->Verbosity(MyVerbosity);
  siTpcMatch->setAssociation(SiTpcTrackMatcher::Association::FromSiChains);
  siTpcMatch->setBeamAlignment(beamAlign);
  siTpcMatch->setBeamPositionCm(0.0, 0.0);
  siTpcMatch->setMatchWindow(0.2, 0.1);
  siTpcMatch->setMatchWindowDz0(0.0);
  siTpcMatch->setRequireSameCharge(false);
  siTpcMatch->setMinTpcClusters(10);
  siTpcMatch->setXYWeights(0, 0, 1.0);    // curvature from TPC only
  siTpcMatch->setHybridXY(hybridXY, 1.0, 0.04, 2); // circle centre from Si
  siTpcMatch->setZWeights(1.0, 1.0e-4, 1.0e-4);
  siTpcMatch->setFitTpcZOffset(true);
  se->registerSubsystem(siTpcMatch);

  // ── Vertex finder: same weights as ELscan macro ───────────────────────
  const double siTpcVtxSigmaRPhi = 0.002, siTpcVtxSigmaZ = 0.04; // [cm]
  const double siTpcVtxMsTerm    = 0.006;                         // [cm GeV]
  const double siTpcVtxMaxChi2   = 16.0;
  const double siTpcBeamSigmaX   = 0.013, siTpcBeamSigmaY = 0.013;
  const unsigned int siTpcVtxMinMvtx    = 3;
  const double       siTpcVtxMaxMatchDphi = 0.05;

  if (runQA)
  {
    auto* vtxSi = new SiTpc_VertexFinder("SiVertexFinder", SiTpc_VertexFinder::Source::Si);
    vtxSi->setMinSiPoints(3);
    vtxSi->setMinMvtxPoints(2);
    vtxSi->setLineTrackResolution(0.1, 0.1);
    se->registerSubsystem(vtxSi);

    auto* vtxTpc = new SiTpc_VertexFinder("TpcVertexFinder", SiTpc_VertexFinder::Source::Tpc);
    vtxTpc->setBeamSpot(tpcLineXcm, tpcLineYcm);
    vtxTpc->setVertexShiftCm(-tpcLineXcm, -tpcLineYcm, -siDispZmm / 10.0);
    vtxTpc->setMinTpcClusters(10);
    se->registerSubsystem(vtxTpc);
  }

  auto* vtxSiTpc = new SiTpc_VertexFinder("SiTpcVertexFinder", SiTpc_VertexFinder::Source::SiTpc);
  vtxSiTpc->setMinTpcClusters(10);
  vtxSiTpc->setSiTpcPositionFromSi(false);
  vtxSiTpc->setTrackResolution(siTpcVtxSigmaRPhi, siTpcVtxSigmaZ);
  vtxSiTpc->setMultipleScatteringTerm(siTpcVtxMsTerm);
  vtxSiTpc->setMaxTrackChi2(siTpcVtxMaxChi2);
  vtxSiTpc->setUseBeamSpotConstraint(true);
  vtxSiTpc->setBeamSpot(0.0, 0.0);
  vtxSiTpc->setBeamSpotSigma(siTpcBeamSigmaX, siTpcBeamSigmaY);
  vtxSiTpc->setMinMvtxPoints(siTpcVtxMinMvtx);
  vtxSiTpc->setTrackPtCut(0.3);
  vtxSiTpc->setMaxMatchDphi(siTpcVtxMaxMatchDphi);
  vtxSiTpc->setMinTracksPerVertex(3);
  se->registerSubsystem(vtxSiTpc);

  // ── Hadron reconstruction ─────────────────────────────────────────────
  if (runHadronReco)
  {
    const std::string hadronFile = std::format("{}/hadron_sitpc_{}_mvtxS{}_{}_{}.root",
                                               outdir, outfilename, mvtxSigmaUm,
                                               runnumber, segment);
    std::cout << "[Hybrid Kalman] MVTX rphi sigma = " << mvtxRPhiSigmaCm * 1e4
              << " um  dE/dx = " << kalmanDedxGeVPerCm << " GeV/cm"
              << "  output = " << hadronFile << std::endl;

    auto* hadron = new tpchadronreco::TpcV0CandidateTree("TpcV0CandidateTree", hadronFile);
    hadron->Verbosity(0);
    hadron->use_sitpc_tracks(true);
    hadron->set_sitpc_track_node("SITPC_TRACKS");
    hadron->set_tpc_sa_cluster_node("TPC_POLYCLUSTERS");
    hadron->set_tpc_sa_track_node("TPC_POLYTRACKS");
    hadron->set_tpc_crossing_decision_node("TPC_CROSSING_DECISIONS");
    hadron->set_require_same_crossing(true);
    hadron->set_sitpc_add_unmatched_tpc_tracks(false);

    // Track quality gate: same as ELscan
    //                              MVTX INTT Si  TPC
    hadron->set_sitpc_track_selection(2,   0,   2,  10, -1.0, -1.0);
    hadron->set_sitpc_max_crossing_tier(1);
    hadron->set_sitpc_require_timed_collision(false);
    hadron->set_sitpc_apply_tpc_z_offset(true);

    // Vertex: same as ELscan
    hadron->set_sitpc_vertex_mode("map");
    hadron->set_sitpc_vertex_map_node("SiTpcPolyVertexMap");
    hadron->set_sitpc_vertex_attach_max_dz(0.5);
    hadron->set_sitpc_vertex_track_selection(0.1, 0.5, 2);
    hadron->set_sitpc_beam_line(0.0, 0.0, 0.0, 0.0);
    hadron->set_primary_vertex(0.0, 0.0, 0.0);

    // ── Track fit: Kalman with field map ─────────────────────────────────
    // Method: "kalman" uses the full Kalman/RKN refit with the CDB field map.
    // The seed is the hybrid matcher helix (Si position, TPC curvature).
    hadron->set_track_fit_method("kalman");
    hadron->set_bfield(1.4);             // nominal field [T]; actual field from map
    hadron->set_min_points(12);
    hadron->set_point_order("radius-descending");
    hadron->set_sitpc_helix_search(10.0, 5.0);
    hadron->use_kalman_field_map(true);  // FIELDMAP_TRACKING from CDB
    hadron->set_kalman_analytic_uniform_propagation(false);
    hadron->set_kalman_rkn4(10.0, 0, 1, 2000);
    hadron->set_kalman_fast_field_jacobian(true);
    hadron->set_kalman_fast_field_pca(true);
    hadron->set_kalman_field_pca_refine_iterations(1);
    hadron->set_kalman_initial_sigmas(0.05, 0.010, 0.020);
    hadron->set_kalman_relative_qop_seed(true, 0.05, 5.0e-4);
    hadron->set_kalman_absolute_qop_seed_sigma(0.20);
    hadron->set_kalman_robust_measurement_updates(true, 16.0, 100.0);
    // ── TUNE: 2 iterations re-linearise the fit after outlier removal ────
    // 1 = ELscan default (faster); 2 = more stable at inflated sigma
    hadron->set_kalman_fit_iterations(2);
    hadron->set_kalman_process_sigmas(0.0, 0.0, 0.0, 0.0);
    //                          X0/cm   MS scale  dE/dx              dE/dx sigma
    hadron->set_kalman_material(2.5e-4, 1.0,      kalmanDedxGeVPerCm, 0.0);

    // ── Measurement sigmas: THE KEY CHANGE ───────────────────────────────
    // MVTX rphi is inflated from 50 um to mvtxRPhiSigmaCm.
    // All other sigmas are identical to the ELscan macro.
    //
    // Why only rphi?
    //   rphi is the coordinate that updates curvature (1/pT) in the Kalman.
    //   r and z do not couple to curvature at leading order.
    //   Inflating rphi only therefore decouples the Si position (which is
    //   precise) from the curvature measurement (which should come from TPC).
    //
    // Why NOT inflate INTT rphi?
    //   INTT already has sigma 250 um (5x MVTX at 50 um) -> weight 25x lower.
    //   Its contribution to the curvature bias is already negligible.
    //   Inflating it further would only hurt z resolution (INTT strips are long).
    //
    //                                 MVTX                    INTT
    //                           rphi       r       z   |  rphi     r      z
    hadron->set_si_measurement_sigmas(
        mvtxRPhiSigmaCm,   // MVTX rphi: TUNE (inflated to protect curvature)
        0.0050,            // MVTX r:    unchanged (5 um)
        0.0050,            // MVTX z:    unchanged (5 um pixel)
        0.0250,            // INTT rphi: unchanged (250 um)
        0.0250,            // INTT r:    unchanged
        0.50);             // INTT z:    unchanged (strip length)

    // TPC sigmas: identical to ELscan macro
    //                                         R1 rphi   r   | R2 rphi   r   | R3 rphi   r   |   z
    hadron->set_tpc_region_measurement_sigmas(0.040, 0.200, 0.025, 0.050, 0.025, 0.200, 0.100);
    hadron->set_tpc_transition_measurement_sigmas(0.200, 0.500, 0.200);
    hadron->set_exclude_tpc_transition_layers(true);

    // Primary vertex constraint for prompt phi / D0 constrained mass
    hadron->set_primary_vertex_constraint(true);
    hadron->set_primary_vertex_constraint_mode("xy");
    hadron->set_primary_vertex_constraint_sigmas(0.005, 0.02);
    hadron->set_compute_constrained_phi(true);
    hadron->set_compute_constrained_d0(true);

    // Loose preselection
    hadron->set_pre_track_pt_min(0.15);
    hadron->set_pre_track_dca_xy_min(-1.0);
    hadron->set_pre_pair_dca_max(-1.0);
    hadron->set_pre_lproj_min(-1.0);
    hadron->set_pre_track_npoints_min(12);

    // D0 displaced vertex: same as ELscan
    const unsigned int d0MinVertexTracks = 4;
    const unsigned int d0MinOtherTracks  = 2;
    hadron->set_d0_collision_vertex(true, d0MinVertexTracks, true, d0MinOtherTracks, -1.0);
    hadron->set_d0_vertex_refit(siTpcVtxSigmaRPhi, siTpcVtxSigmaZ, siTpcVtxMsTerm,
                                siTpcVtxMaxChi2, true, 0.0, 0.0,
                                siTpcBeamSigmaX, siTpcBeamSigmaY);
    hadron->set_d0_daughter_si_selection(siTpcVtxMinMvtx, siTpcVtxMaxMatchDphi);
    // DCA and topology from hybrid helix (unchanged by sigma inflation):
    hadron->set_d0_use_sitpc_helix(true);

    // ── Species selections: same as ELscan macro ─────────────────────────
    //                     clu  pt    mass           |dca_xy| |dca_z| pairDCA flight |pcaz| pcadz
    hadron->set_phi_selection(true, 10, 0.15, 0.98, 1.08,
                              0.10,    0.20,   -1.0,   -1.0,  20.0,  0.20);
    hadron->set_d0_selection(true, 10, 0.20, 1.70, 2.05,
                             0.50,  0.30,  -1.0,  -1.0,  20.0, 0.30);
    hadron->set_antid0_selection(true, 10, 0.20, 1.70, 2.05,
                                 0.50,  0.30,  -1.0,  -1.0,  20.0, 0.30);
    //                                          Lxy   L_min L_max  DIRA   |d0|  pDCA  ptDCA  d0*d0
    hadron->set_d0_displaced_selection(false, 0.0, -1.0, 0.5, 0.90, 0.005, 0.03, 0.01, 2.0);
    hadron->set_kshort_selection(true, 10, 0.0, 0.40, 0.60,
                                 20.0, 0.3, 0.05, 1.0, 0.5, 0.9);
    hadron->set_lambda_selection(true, 10, 0.0, 1.08, 1.16,
                                 20.0, 0.3, 0.05, 1.0, 0.5, 0.9);
    hadron->set_antilambda_selection(true, 10, 0.0, 1.08, 1.16,
                                     20.0, 0.3, 0.05, 1.0, 0.5, 0.9);
    hadron->set_jpsi_selection(false, 10, 0.5, 2.0, 4.0,
                               0.1, 0.2, -1.0, -1.0, 20.0, 0.2);

    hadron->set_write_likesign_tree(true);
    hadron->set_write_likesign_phi(true);
    hadron->set_write_likesign_d0(true);
    hadron->set_write_track_tree(true);
    se->registerSubsystem(hadron);
  }

  // ── Optional QA modules ───────────────────────────────────────────────
  if (runQA)
  {
    auto* vtxQA = new SiTpc_VertexQA("SiTpc_VertexQA",
        outdir + "/sitpc_vertex_qa_" + outname + ".root");
    se->registerSubsystem(vtxQA);

    auto* siTpcResid = new SiTpc_TrackResiduals("SiTpc_TrackResiduals",
        outdir + "/sitpc_track_residuals_" + outname + ".root");
    siTpcResid->setMinPt(0.1);
    siTpcResid->setMinTpcClusters(10);
    siTpcResid->setWriteMatchingTree(true);
    siTpcResid->setSiTpcVertexMapName("SiTpcPolyVertexMap");
    siTpcResid->setSiVertexMapName("SiPolyVertexMap");
    se->registerSubsystem(siTpcResid);
  }

  se->run(nEvents + nSkip);
  se->Print("NODETREE");
  se->End();
  se->PrintTimer();
  CDBInterface::instance()->Print();
  delete se;
  std::cout << "Finished" << std::endl;
  gSystem->Exit(0);
}