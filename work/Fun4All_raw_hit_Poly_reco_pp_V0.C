/*
 * This macro shows a minimum working example of running the 
 * TPC track reconstruction starting with hit unpackers
 */

// leave the GlobalVariables.C at the beginning, an empty line afterwards
// protects its position against reshuffling by clang-format
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

#include <trackingdiagnostics/Tpc_ModuleTrackDisplay.h>
#include <trackingdiagnostics/Tpc_AssembledTrackDisplay.h>
#include <trackingdiagnostics/Tpc_PolyClusterDisplay.h>
#include <trackingdiagnostics/Tpc_PolyClusterResiduals.h>
#include <trackingdiagnostics/SiTpc_EventDisplay.h>
#include <trackingdiagnostics/SiTpc_TrackResiduals.h>

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
#include <phool/recoConsts.h>

#include <string>
#include <vector>

#include <format>

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

class TpcAssembledTrackCountCut : public SubsysReco
{
 public:
  explicit TpcAssembledTrackCountCut(unsigned int maxTracks)
    : SubsysReco("TpcAssembledTrackCountCut")
    , maxTracks_(maxTracks)
  {
  }
  int process_event(PHCompositeNode* topNode) override
  {
    auto* tracks = findNode::getClass<Tpc_AssembledTrackContainer>(topNode, "TPC_ASSEMBLEDTRACKS");
    if (tracks && tracks->size() > maxTracks_)
    {
      std::cout << "Abort event - number of tpc assembled tracks is " << tracks->size() << " > " << maxTracks_ << std::endl;
      return Fun4AllReturnCodes::ABORTEVENT;
    }
    return Fun4AllReturnCodes::EVENT_OK;
  }

 private:
  unsigned int maxTracks_;
};

// 'Golden' store of p+p
//.x Fun4All_raw_hit_TPC_reco.C(2, 79513, 0, ".", 0, "run3pp", "ana532_nocdbtag_v001","HITS_ppFieldOn")
//.x Fun4All_raw_hit_TPC_reco.C(2, 79516, 0, ".", 0, "run3pp", "ana532_nocdbtag_v001","HITS_ppFieldOn")

// 6x6       75570-75573 - 1mrad; 75574 - 0mrad
//.x Fun4All_raw_hit_TPC_reco.C(10, 75570, 0, ".", 0, "run3auau", "ana514_nocdbtag_v001","HITS_AuAu_ZeroField_1mrad")
//.x Fun4All_raw_hit_TPC_reco.C(10, 75574, 0, ".", 0, "run3auau", "ana514_nocdbtag_v001","HITS_AuAu_ZeroField_0mrad")

//111x111 O+O
//.x Fun4All_raw_hit_TPC_reco.C(2, 82626, 0, ".", 0, "run3oo", "ana537_nocdbtag_v001","HITS_OO")

void Fun4All_raw_hit_Poly_reco_pp_V0(
    const int nEvents = 10,
    const int runnumber = 79513,
    const int segment = 0,
    const std::string& outdir = ".",
    const int nSkip = 0,
    const std::string& collision = "run3pp",
    const std::string& production = "ana561_2025p013_v001",
    const std::string& outfilename = "ppFieldOn_Test_3pts_resolv_si_align2",
    const std::string& datatype = "physics",
    const bool runHadronReco = true,   // phi / D0 / K0S / Lambda from the Si+TPC tracks (this job)
    const bool writeDst = false)       // DST with the Si+TPC nodes for a later TpcHadronReco pass
{
  const bool convertSeeds = true;
  auto *se = Fun4AllServer::instance();
  se->Verbosity(1);
  auto *rc = recoConsts::instance();
  se->registerSubsystem(new SkipFirstN(nSkip));

  const std::string dsttype = "STREAMING_EVENT";
  const std::string dsttype_to_save = "TPC";


  std::ostringstream runstr;
  runstr << std::setw(8) << std::setfill('0') << runnumber;

  std::ostringstream segstr;
  segstr << std::setw(5) << std::setfill('0') << segment;

  const int runbase = (runnumber / 100) * 100;
  const int runnext = runbase + 100;

  std::ostringstream rundir;
  rundir << "run_" << std::setw(8) << std::setfill('0') << runbase << "_" << std::setw(8) << std::setfill('0') << runnext;

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
if(collision!="run3line_laser"&&collision!="run3cosmics")
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
    std::stringstream nice_runnumber;
  nice_runnumber << std::setw(8) << std::setfill('0') << std::to_string(runnumber);

  int rounded_up = 100 * (std::ceil((float) runnumber / 100));
  std::stringstream nice_rounded_up;
  nice_rounded_up << std::setw(8) << std::setfill('0') << std::to_string(rounded_up);

  int rounded_down = 100 * (std::floor((float) runnumber / 100));
  std::stringstream nice_rounded_down;
  nice_rounded_down << std::setw(8) << std::setfill('0') << std::to_string(rounded_down);

  for (auto &stream : streams)
  {
    std::string filename = std::format("DST_{}_{}_{}_{}-{:08d}-{:05d}.root", dsttype, stream, collision, production,  runnumber, segment);
    std::string filepath = std::format("/sphenix/lustre01/sphnxpro/production/{}/{}/{}/DST_{}_{}/run_{}_{}/{}",collision, datatype, production, dsttype, stream,  nice_rounded_down.str(), nice_rounded_up.str(),filename);
    std::cout << "Adding DST: " << filepath << std::endl;
    if (i == 0)
    {
      rc->set_IntFlag("RUNNUMBER", runnumber);
      rc->set_uint64Flag("TIMESTAMP", runnumber);
    }

    std::string inputname = "InputManager" + std::to_string(i);
    auto *hitsin = new Fun4AllDstInputManager(inputname);
    hitsin->fileopen(filename);
    se->registerInputManager(hitsin);

    ++i;
  }

  rc->set_IntFlag("RUNNUMBER", runnumber);
  rc->set_IntFlag("RUNSEGMENT", segment);

  Enable::QA = false;
  Enable::CDB = true;
  rc->set_StringFlag("CDB_GLOBALTAG", "newcdbtag");
  rc->set_uint64Flag("TIMESTAMP", runnumber);

 

  G4TRACKING::convert_seeds_to_svtxtracks = convertSeeds;


  std::cout << " run: " << runnumber
            << " samples: " << TRACKING::reco_tpc_maxtime_sample
            << " pre: " << TRACKING::reco_tpc_time_presample
            << " vdrift: " << G4TPC::tpc_drift_velocity_reco
            << std::endl;

  TRACKING::streaming_mode = true;

  FlagHandler *flag = new FlagHandler();
  se->registerSubsystem(flag);

  std::string geofile = CDBInterface::instance()->getUrl("Tracking_Geometry");

  Fun4AllRunNodeInputManager *ingeo = new Fun4AllRunNodeInputManager("GeoIn");
  ingeo->AddFile(geofile);
  se->registerInputManager(ingeo);



  TpcReadoutInit(runnumber);
  G4TPC::REJECT_LASER_EVENTS = true;
  // Flag for running the tpc hit unpacker with zero suppression on
  TRACKING::tpc_zero_supp = true;

  TrackingInit();

  for (int felix = 0; felix < 6; felix++)
  {
    Mvtx_HitUnpacking(std::to_string(felix));
  }
  for (int server = 0; server < 8; server++)
  {
    Intt_HitUnpacking(std::to_string(server));
  }
  std::ostringstream ebdcname;
  for (int ebdc = 0; ebdc < 24; ebdc++)
  {
    for (int endpoint = 0; endpoint < 2; endpoint++)
      {
        ebdcname.str("");
        if (ebdc < 10)
	  {
	    ebdcname << "0";
	  }
        ebdcname << ebdc << "_" << endpoint;
        Tpc_HitUnpacking(ebdcname.str());
      }
  }


  Micromegas_HitUnpacking();
 
  Mvtx_Clustering();
  Intt_Clustering();
  Micromegas_Clustering();

  Tpc_LaserEventIdentifying();

  Reject_Laser_Events();

  //==============================================================
/*
  Tracking_Reco_SiliconSeed_run2pp();
  auto *converter = new TrackSeedTrackMapConverter("SiliconSeedToSvtxTrackMap");
  converter->setTrackSeedName("SiliconTrackSeedContainer");
  converter->setTrackMapName("SiliconSvtxTrackMap");
  converter->setClusterMapName("TRKR_CLUSTER");
  se->registerSubsystem(converter);

  auto *finder_svx = new PHSimpleVertexFinder("SiliconVertexFinder");
  finder_svx->Verbosity(0);
  finder_svx->setDcaCut(0.1);
  finder_svx->setTrackPtCut(0.2);
  finder_svx->setBeamLineCut(1);
  finder_svx->setTrackQualityCut(500);
  finder_svx->setNmvtxRequired(3);
  finder_svx->setOutlierPairCut(0.1);
  finder_svx->setTrackMapName("SiliconSvtxTrackMap");
  finder_svx->setVertexMapName("SiliconSvtxVertexMap");
  se->registerSubsystem(finder_svx);

*/
 // se->registerSubsystem(new TpcConditionsReco());

  // Configure the shared drift model before the crossing finder and clusterizer.
  auto *garfield = new PHGarfield();
  // The TPC beam line below (x0 = -1.068 mm, y0 = 2.675 mm) was measured WITH the survey
  // field geometry.  Either use the same setting here or re-measure the TPC beam line.
  garfield->SetUseSurveyGeometry(false);
  se->registerSubsystem(garfield);

  se->registerSubsystem(new Tpc_ModuleTrackReco()); // makes TPC_MODULETRACKS
  se->registerSubsystem(new Tpc_AssembledTrackReco()); // makes TPC_ASSEMBLEDTRACKS
 
  //se->registerSubsystem(new TpcAssembledTrackCountCut(700));

  // true: Si seeding from 3-point crossing trajectories, crossing resolved after the Si seeding
  // false: TpcCrossingFinder + full TPC clusters before the Si seeding
  const bool useTripletSeeds = true;
  

  // true: triggered data (crossing 0); false: streaming (every INTT crossing, resolved below)
  const bool triggeredMode = false;

  if (!useTripletSeeds)
  {
    auto *crossingFinder = new TpcCrossingFinder();
    crossingFinder->Verbosity(0);
    crossingFinder->setTriggeredMode(triggeredMode);
    crossingFinder->setInputNodeName("TPC_ASSEMBLEDTRACKS");
    crossingFinder->setOutputNodeName("TPC_CROSSING_DECISIONS");
    crossingFinder->setVertexMapNodeName("SiliconSvtxVertexMap");  // optional, configurable
    se->registerSubsystem(crossingFinder);
  }


  //------------------------------------------------------------

  if (useTripletSeeds)
  {
    auto *triplet = new Tpc_PolyTripletReco();  // makes TPC_CROSSINGTRAJECTORIES
    triplet->setTriggeredMode(triggeredMode);  // true: crossing 0 only; false: every INTT crossing with |z0| < 20 cm
    triplet->setEdgePadMargin(1);  // skip layers with hits on the sector-edge pads
    triplet->setMinLayerSpan(10);  // beginning-to-end distance in layers
    se->registerSubsystem(triplet);
  }
  else
  {
    se->registerSubsystem(new Tpc_PolyClusterizer());
    se->registerSubsystem(new Tpc_PolyTrackReco());      // makes TPC_POLYTRACKS
    se->registerSubsystem(new Tpc_PolyTrackVertexer());  // makes TPC_POLYTRACKVERTICES
  }

  //------------------------------------------------------------
  SiTpcBeamAlignment beamAlign;
  beamAlign.setEnabled(true);
  beamAlign.setCenterBeam(true);  
  beamAlign.setClamshellBoundary(0.0);  // halves: A = Uphi in [0, pi)
  beamAlign.setSiPerHalf(true);   
 /*
  beamAlign.setBeamLineCm(SiTpcBeamAlignment::Tpc, -0.106835, 0.267482, 0.0, 0.0);
  beamAlign.setBeamLineMm( SiTpcBeamAlignment::SiHalfA,-0.533300, 1.772693,0.0, 0.0);
  beamAlign.setBeamLineMm(SiTpcBeamAlignment::SiHalfB,0.595174, 1.608226, 0.0, 0.0);

  //                       part                           x0         y0         dx/dz          dy/dz
  beamAlign.setBeamLineCm(SiTpcBeamAlignment::Tpc,     -0.106835,  0.267482,  8.478616e-04, -8.041254e-04);  // [cm]
  beamAlign.setBeamLineMm(SiTpcBeamAlignment::SiHalfA, -0.533300,  1.772693,  7.094553e-04, -2.743455e-04);  // [mm]
  beamAlign.setBeamLineMm(SiTpcBeamAlignment::SiHalfB,  0.595174,  1.608226,  2.343532e-03,  6.926381e-04);  // [mm]
  const double siUphiRotation = 0.1481;                // global phi = Uphi + 0.1481 rad
  const double siCenterXmm = 6.0, siCenterYmm = -1.0;  // Si detector shift (detector -> global frame)
 */

beamAlign.setBeamLineCm(SiTpcBeamAlignment::Tpc, -0.106835, 0.267482, 8.478616e-04, -8.041254e-04);
 
// Si: the measured fits are UNALIGNED, global frame with centre (+6, -1, 0) mm.
// PROVISIONAL bridge to this configuration (REPLACE with fits from a run with exactly these
// settings):
//   1. alignment file ON  -> beam moves by +t (mean MVTX sensor translation of that half)
//        half A (upper): t = (-0.185, -0.207) mm   half B (lower): t = (-0.248, -0.035) mm
//        whole MVTX    : t = (-0.216, -0.121) mm
//   2. centre (6, -1) -> D -> beam moves by D - (6, -1) = (+0.2955, -0.06874) mm
const double siDispXmm = 6.2955, siDispYmm = -1.06874, siDispZmm = -6.66289;
const double dCx = siDispXmm - 6.0, dCy = siDispYmm - (-1.0);
 
beamAlign.setBeamLineMm(SiTpcBeamAlignment::SiHalfA, -0.533300, 1.772693, 7.094553e-04, -2.743455e-04);
beamAlign.shiftBeamLineMm(SiTpcBeamAlignment::SiHalfA, -0.185, -0.207);   // alignment file
beamAlign.shiftBeamLineMm(SiTpcBeamAlignment::SiHalfA, dCx, dCy);         // -> (-0.423, 1.497)
 
beamAlign.setBeamLineMm(SiTpcBeamAlignment::SiHalfB, 0.595174, 1.608226, 2.343532e-03, 6.926381e-04);
beamAlign.shiftBeamLineMm(SiTpcBeamAlignment::SiHalfB, -0.248, -0.035);   // alignment file
beamAlign.shiftBeamLineMm(SiTpcBeamAlignment::SiHalfB, dCx, dCy);         // -> ( 0.643, 1.505)
 
// whole Si (used only with setSiPerHalf(false)): mean of the two halves + whole-MVTX t
beamAlign.setBeamLineMm(SiTpcBeamAlignment::SiFull, 0.030937, 1.690460, 0.0, 0.0);
beamAlign.shiftBeamLineMm(SiTpcBeamAlignment::SiFull, -0.216, -0.121);    // alignment file
beamAlign.shiftBeamLineMm(SiTpcBeamAlignment::SiFull, dCx, dCy);          // -> ( 0.110, 1.501)
 
beamAlign.print();        
// Silicon intrinsic -> ideal detector frame.
const double siUphiRotation = 0.1481;

// Ideal detector -> current silicon global convention.
//const double siCenterXmm =  6.0;
//const double siCenterYmm = -1.0;

const double siCenterXmm =  siDispXmm;
const double siCenterYmm =  siDispYmm;
// z stays 0: SiHitSeedReco seeds with the detector-frame z but builds the TPC residuals with
// the global z, so a non-zero centre z would shift every chain's dz by D_z = -0.666 cm against
// the OffsetPrior (sigma 0.35 cm) and the resolver dz sigma.  Apply D_z only in the z-alignment
// step, together with setTpcZOffset() and the seeding windows.
const double siCenterZmm =  0.0;
 
  //------------------------------------------------------------

  auto* siHitSeed = new SiHitSeedReco("SiHitSeedReco");
  siHitSeed->Verbosity(0);  // 1: run summary + residual widths at End; 2: one line per TPC track (slow); 3: + every chain
  siHitSeed->setUphiRotation(siUphiRotation);
  siHitSeed->setDetectorCenterMm(siCenterXmm, siCenterYmm, siCenterZmm);
  //siHitSeed->setDetectorCenterMm(siCenterXmm, siCenterYmm,0.0);
  siHitSeed->setBeamAlignment(beamAlign);
  siHitSeed->setApplySiAlignment(true);                     // default off
  // REQUIRED with setApplySiAlignment(true): no CDB fallback, a missing file aborts the run.
  // Use an absolute path on batch workers.
  siHitSeed->setSiAlignmentParamsFile("si_align_params_inttshift.txt");
  siHitSeed->setUseNewSiliconRotationOrder(false); 
  //siHitSeed->setDebugEvent(1);  // full dump of one event: very slow, only for debugging

  // ---- blobs = hardware clusters (touching MVTX pixels / neighbouring INTT strips, one time bucket)
  siHitSeed->setBlobsFromPixelAdjacency(true);   // default
  siHitSeed->setInttBlobsPerTimeBucket(true);    // default (used with the grid blobs only)

  // ---- z guided by the TPC crossing trajectory: offset from the TPC z at this crossing,
  // inclination (theta) from the TPC tanl at every step
  siHitSeed->setAnchorWindow(1.0, 1.0);           // layer 2/1 anchor: r*phi 1 cm, z +-1 cm around the TPC z at this crossing
  siHitSeed->setSeedZHalfWindowCm(0.5);           // seed pair: dz = tanl_TPC * ds +- 0.5 cm
  siHitSeed->setMvtxPropagationZFromTpc(true);    // 3rd MVTX layer: z from the TPC inclination, not the 2-hit line
  siHitSeed->setPropagationZHalfWindowBins(2.0);  // MVTX: +- 2 z bins (~0.7 cm)
  siHitSeed->setInttCentre(SiHitSeedReco::InttCentre::TpcOffset);  // INTT: TPC prediction + mean MVTX offset
  siHitSeed->setInttTpcPhiHalfWindowBins(4.0);        // INTT: +- 0.14 rad (was 2 bins: cut real hits at ~1.5-2 sigma)
  siHitSeed->setInttPropagationZHalfWindowBins(5.0);  // INTT: +- ~1.7 cm (strip half-length + prediction error)

  // ---- agreement: common z offset with a prior, per-layer z sigmas = widths AROUND the offset
  siHitSeed->setTpcAgreementZModel(SiHitSeedReco::TpcZModel::OffsetPrior);
  siHitSeed->setTpcAgreementZOffsetSigma(0.35);   // resolver End: robust sigma of the tier-0 dz
  siHitSeed->setTpcAgreementZWeight(1.0);
  siHitSeed->setMinLayers(2);       // was 3: 2 MVTX hits alone are a chain
  siHitSeed->setMinMvtxLayers(2);   // unchanged



  siHitSeed->setTpcAgreementSigmaLayer(0, 0.37, 0.16);
  siHitSeed->setTpcAgreementSigmaLayer(1, 0.45, 0.10);
  siHitSeed->setTpcAgreementSigmaLayer(2, 0.54, 0.16);
  siHitSeed->setTpcAgreementSigmaLayer(3, 0.36, 0.90);
  siHitSeed->setTpcAgreementSigmaLayer(4, 0.46, 0.74);
  siHitSeed->setTpcAgreementSigmaLayer(5, 0.43, 0.62);
  siHitSeed->setTpcAgreementSigmaLayer(6, 0.47, 0.66);
  if (useTripletSeeds)
  {
    siHitSeed->setTpcTrajectoryNodeName("TPC_CROSSINGTRAJECTORIES");
    siHitSeed->setMinTpcClusters(3);
    if (!triggeredMode)
    {
      // the chain of a crossing hypothesis takes only INTT hits of that crossing (+-1 bucket);
      // the resolver uses these INTT hits as the time of the chain (a collision needs one)
      siHitSeed->setInttCrossingWindow(1);
    }
  }
  se->registerSubsystem(siHitSeed);

  if (useTripletSeeds)
  {
    // one crossing + one Si chain per assembled track -> TPC_CROSSING_DECISIONS (replaces
    // TpcCrossingFinder). Must run BEFORE SiTrajectoryFitter: it selects the chains to fit.
    // best chain per crossing trajectory -> credit -> vertex vote -> one collision per crossing
    auto *resolver = new TpcCrossingResolver();
    resolver->setTriggeredMode(triggeredMode);
    resolver->setDzSigma(0.4);         // Si - TPC z width of the 3-point trajectories at the silicon (measured 0.33 cm)
    resolver->setNoInttFactor(0.5);    // chains without INTT hit carry no time information
    resolver->setMinLoneCredit(0.08);  // the only track of its crossing: reachable by a good 3-layer MVTX chain
    resolver->setAmbiguityRatio(1.5);  // lone: best >= 1.5x its other hypotheses (also the tier-0 ratio)
    resolver->setSiZSigma(0.22);       // single-track Si z at the beam [cm]; support uses 0.22 * sqrt(1 + 1/n_other)
    resolver->setClusterZWindow(0.7);  // tracks of one collision within 0.7 cm of the vertex mean (pair sigma 0.31 cm; 0.5 lost ~10% of real pairs)
    resolver->setLoneIgnoreSameSi(true);  // default: a lone track ambiguous only between crossings of the SAME Si hits is kept (tier 1)
    resolver->setRequireInttForCollision(true);  // default: a collision needs a chain with INTT hits
    // split collisions (merged before pass 2 drops any track)
    resolver->setMergeSplitCollisions(true);   // default on
    resolver->setMergeMaxCrossings(2);
    resolver->setMergeZNSigma(3.0);            // Si vertices agree within 3 sigma_z sqrt(1/n1 + 1/n2)
    resolver->setMergeZWindow(1.0);            // upper limit of that z test [cm]
    resolver->setMergeMaxDeltaChi2(3.0);
    resolver->setMergeTier0MinTracks(3);       // merged collisions with >= 3 tracks can be tier 0
    resolver->Verbosity(0);  // 1: summary per event + End; 2: every track with its hypotheses (reasons for the grey tracks)
    resolver->setKeepUntimedLone(true);
    resolver->setMinUntimedLoneCredit(-1);
    resolver->setVertexChainReselect(true);
    se->registerSubsystem(resolver);
  }


  auto* siTraj = new SiTrajectoryFitter("SiTrajectoryFitter");
  siTraj->Verbosity(0);
  siTraj->setInputNodeName("SI_HIT_SEED_EVENT");
  siTraj->setOutputNodeName("SI_TRAJECTORY");
  siTraj->setFitSelectedChainsOnly(true);
  siTraj->setUphiRotation(siUphiRotation);
  siTraj->setDetectorCenterMm(siCenterXmm, siCenterYmm, siCenterZmm);  // same frame as SiHitSeedReco
  siTraj->setBeamAlignment(beamAlign);
  siTraj->setBeamPositionCm(0.0, 0.0);   // the beam is at (0,0) after the alignment
  siTraj->setUseIntt(true);
  siTraj->setMinPoints(3);
  siTraj->setZWeights(1.0, 0.01);        // MVTX pixel vs INTT strip length
  siTraj->setMinCurvatureLeverArm(3.0);  // [cm] shorter radial span -> straight line, pt = NaN
  se->registerSubsystem(siTraj);


  if (useTripletSeeds)
  {
    // full TPC clusters and tracks for the Si+TPC fit, with the resolved crossing
    
    auto *cluster = new Tpc_PolyClusterizer();  // makes TPC_POLYCLUSTERS
   // cluster->Verbosity(10);
    cluster->setReverseDriftSubsteps(1);
  // cluster->setMaxAcceptedTier(2); 
    se->registerSubsystem(cluster);
    //se->registerSubsystem(new Tpc_PolyClusterizer());
    se->registerSubsystem(new Tpc_PolyTrackReco());      // makes TPC_POLYTRACKS
    se->registerSubsystem(new Tpc_PolyTrackVertexer());  // makes TPC_POLYTRACKVERTICES
  }

  auto* siTpcMatch = new SiTpcTrackMatcher("SiTpcTrackMatcher");
  siTpcMatch->Verbosity(0);
  siTpcMatch->setAssociation(SiTpcTrackMatcher::Association::FromSiChains);
  siTpcMatch->setBeamAlignment(beamAlign);
  siTpcMatch->setBeamPositionCm(0.0, 0.0);
  siTpcMatch->setMatchWindow(0.2, 0.1);  // |dphi| [rad], |deta|: matching variables (not a cut for chains)
  siTpcMatch->setMatchWindowDz0(0.0);    // z0 not used (TPC z depends on the crossing)
  siTpcMatch->setRequireSameCharge(false);
  siTpcMatch->setMinTpcClusters(10);
  siTpcMatch->setXYWeights(0, 0, 1.0);   // MVTX, INTT, TPC weights in the circle fit
  siTpcMatch->setZWeights(1, 1, 1.0);    // MVTX, INTT, TPC weights in z(s)
  siTpcMatch->setFitTpcZOffset(true);    // free TPC z offset (crossing / t0) in the refit
  se->registerSubsystem(siTpcMatch);

  //------------------------------------------------------------
  // Hadron reconstruction from the Si-associated tracks (SITPC_TRACKS).
  // Same frame as the matcher: beam at (0, 0) after beamAlign, collision vertex z
  // = Si vertex z of TpcCrossingResolver; pairs only inside one collision.
  if (runHadronReco)
  {
    auto *hadron = new tpchadronreco::TpcV0CandidateTree(
        "TpcV0CandidateTree", outdir + "/hadron_sitpc_" + outfilename + ".root");
    hadron->Verbosity(0);
    hadron->use_sitpc_tracks(true);
    hadron->set_sitpc_track_node("SITPC_TRACKS");
    hadron->set_tpc_sa_cluster_node("TPC_POLYCLUSTERS");
    hadron->set_tpc_sa_track_node("TPC_POLYTRACKS");  // dE/dx (+ unmatched TPC tracks if enabled)
    hadron->set_tpc_crossing_decision_node("TPC_CROSSING_DECISIONS");
    hadron->set_require_same_crossing(true);
    hadron->set_sitpc_add_unmatched_tpc_tracks(false);  // true: TPC-only tracks too (K0S/Lambda outside the MVTX)

    // Si+TPC track quality and crossing
    //                             MVTX INTT  Si  TPC  circle rms  z rms [cm]
    hadron->set_sitpc_track_selection(2,   0,   2,  10,   -1.0,      -1.0);
    hadron->set_sitpc_max_crossing_tier(1);   // tier 2 tracks are not clustered by Tpc_PolyClusterizer anyway
    hadron->set_sitpc_require_timed_collision(false);
    hadron->set_sitpc_apply_tpc_z_offset(true);

    // collision vertex: Si vertex z of the resolver (fallback: robust mean of the prompt Si+TPC tracks)
    hadron->set_sitpc_vertex_mode("resolver");
    hadron->set_sitpc_vertex_track_selection(0.1, 0.5, 2);  // |dca_xy| [cm], z window [cm], min tracks
    hadron->set_sitpc_beam_line(0.0, 0.0, 0.0, 0.0);        // TPC frame, beam centred by beamAlign
    hadron->set_primary_vertex(0.0, 0.0, 0.0);              // fallback only

    // track model: "sitpc" = SiTpcTrackMatcher combined helix (no refit), or "kalman"
    hadron->set_track_fit_method("sitpc");
    hadron->set_bfield(1.4);                                // same as SiTpcTrackMatcher::setBz
    hadron->set_min_points(12);
    hadron->set_point_order("radius-descending");
    hadron->set_sitpc_helix_search(10.0, 5.0);
    //                               MVTX rphi    r     z  | INTT rphi    r     z  [cm]
    hadron->set_si_measurement_sigmas(0.0010, 0.0010, 0.0010, 0.0030, 0.0030, 0.50);

    // loose pair preselection
    hadron->set_pre_track_pt_min(0.15);
    hadron->set_pre_track_dca_xy_min(-1.0);
    hadron->set_pre_pair_dca_max(-1.0);
    hadron->set_pre_lproj_min(-1.0);
    hadron->set_pre_track_npoints_min(12);

    // phi -> K+ K-  (prompt):    clu  pt   mass        |dca_xy| |dca_z| pairDCA flight  |pca z| pca dz
    hadron->set_phi_selection(true, 10, 0.15, 0.98, 1.08, 0.10,    0.20,   -1.0,   -1.0,   20.0,   0.20);
    // D0 -> K- pi+, anti-D0 -> K+ pi-
    hadron->set_d0_selection(true, 10, 0.20, 1.70, 2.05, 0.50, 0.30, -1.0, -1.0, 20.0, 0.30);
    hadron->set_antid0_selection(true, 10, 0.20, 1.70, 2.05, 0.50, 0.30, -1.0, -1.0, 20.0, 0.30);
    // displaced D0 vertex (on top of the D0 selection; starting values, tune on data):
    //                                      Lxy  L_min L_max  DIRA  |d0|_min pairDCA pointDCA d0*d0
    hadron->set_d0_displaced_selection(false, 0.0, -1.0, 0.5,  0.90, 0.005,   0.03,   0.01,    2.0);

    // V0s (only the decays inside the MVTX are Si-matched; see set_sitpc_add_unmatched_tpc_tracks)
    hadron->set_kshort_selection(true, 10, 0.0, 0.40, 0.60, 20.0, 0.3, 0.05, 1.0, 0.5, 0.9);
    hadron->set_lambda_selection(true, 10, 0.0, 1.08, 1.16, 20.0, 0.3, 0.05, 1.0, 0.5, 0.9);
    hadron->set_antilambda_selection(true, 10, 0.0, 1.08, 1.16, 20.0, 0.3, 0.05, 1.0, 0.5, 0.9);
    hadron->set_jpsi_selection(false, 10, 0.5, 2.0, 4.0, 0.1, 0.2, -1.0, -1.0, 20.0, 0.2);

    hadron->set_write_likesign_tree(true);
    hadron->set_write_likesign_phi(true);
    hadron->set_write_likesign_d0(true);
    hadron->set_write_track_tree(true);
    se->registerSubsystem(hadron);
  }
  




  auto* siTpcResid = new SiTpc_TrackResiduals("SiTpc_TrackResiduals", outdir + "/sitpc_track_residuals_" + outfilename + ".root");
  siTpcResid->Verbosity(0);
  siTpcResid->setMinPt(0.1);
  siTpcResid->setMinTpcClusters(10);
  siTpcResid->setWriteMatchingTree(true);
  se->registerSubsystem(siTpcResid);

  auto* display = new SiTpc_EventDisplay("SiTpc_EventDisplay", outdir + "/sitpc_event_display_" + outfilename + ".root", 10);  // max events saved
  display->Verbosity(0);
  display->setTpcClusterNodeName("TPC_POLYCLUSTERS");
  display->setTpcTrackNodeName("TPC_POLYTRACKS");
  display->setTpcVertexNodeName("TPC_POLYTRACKVERTICES");
  display->setTrackVertexZRange(-20.0, 20.0);
  display->setUseStraightLineTracks(false);
  display->setSiSeedNodeName("SI_HIT_SEED_EVENT");
  display->setSiTrajectoryNodeName("SI_TRAJECTORY");
  display->setUphiRotation(siUphiRotation);
  display->setDetectorCenterMm(siCenterXmm, siCenterYmm);  // display: x, y only
  display->setBeamAlignment(beamAlign);
  display->setBeamPositionCm(0.0, 0.0);
  display->setSiMinPt(0.2);  // circle fits below this are drawn dotted
  display->setDrawLowPtSi(true);
  display->setTrajectoryExtension(SiTpc_EventDisplay::ExtendBoth);
  display->setSiExtrapolationRadius(80.0);
  display->setTpcExtrapolationRadius(15.0);
  display->setRequireSiTrajectory(true);  // only save events with Si trajectories
  display->setDrawSiTpcTracks(true);
  display->setSiTpcTrackColor(kBlack);
  display->setColorTpcByMatch(true);        // matched TPC tracks in the Si chain colour, unmatched grey
  display->setDrawCandidateChains(false);   // true: also the non-selected candidate chains
  //se->registerSubsystem(display);

  Fun4AllOutputManager *out = new Fun4AllDstOutputManager("out", std::format("{}/DST_{}_{}_{}-{:08d}-{:05d}.root",outdir, dsttype_to_save, collision, production, runnumber, segment));

  out->AddNode("Sync");
  out->AddNode("EventHeader");
  out->AddRunNode("TPCGEOMCONTAINER");
  out->AddNode("TRKR_HITSET");
  //out->AddNode("TPC_MODULETRACKS");
  //out->AddNode("TPC_ASSEMBLEDTRACKS");
  out->AddNode("TPC_POLYCLUSTERS");
  out->AddNode("TPC_POLYTRACKS");
  out->AddNode("TPC_POLYTRACKVERTICES");
  out->AddNode("TRKR_CLUSTER");
  // Si-associated reconstruction, input of TpcHadronReco in a separate pass
  // (Fun4All_TpcV0CandidateTree.C with V0_TRACK_SOURCE=sitpc)
  out->AddNode("TPC_CROSSING_DECISIONS");
  out->AddNode("SI_TRAJECTORY");
  out->AddNode("SITPC_TRACKS");

  if (writeDst)
  {
    se->registerOutputManager(out);
  }
  
  se->run(nEvents+nSkip);
  se->Print("NODETREE");
  se->End();
  se->PrintTimer();


  CDBInterface::instance()->Print();
  delete se;
  std::cout << "Finished" << std::endl;
  gSystem->Exit(0);
}