/*
 * Full reconstruction (TPC + MVTX/INTT hit unpackers -> Si+TPC tracks -> collision vertices)
 * followed by the hadron reconstruction (TpcV0CandidateTree: K0S, Lambda, phi, D0) on the
 * Si-associated tracks with the Kalman/RKN refit of the Si + TPC points.
 *
 * Same Si / TPC frame, alignment, seeding, resolver, matcher and vertex settings as
 * Fun4All_raw_hit_Poly_reco_pp.C (tpcFrameMode 5, hybrid xy, good-seed Si+TPC vertices with >= 3 tracks).
 * D0: only collisions whose Si+TPC vertex has >= 4 tracks; the vertex is refitted without the two
 * daughters (unbiased by the decay length) and the daughter DCA_xy / secondary vertex come from the
 * hybrid matcher helix (the DCA_xy of plot_dca_xy.C).  Output:
 *   <outdir>/hadron_sitpc_<outfilename>_<run>_<segment>.root   (pairTree, likeSignPairTree, trackTree)
 *   QA (runQA = true): sitpc_vertex_qa_<outname>.root, sitpc_track_residuals_<outname>.root
 *
 *   root -l -b -q 'Fun4All_raw_hit_Poly_reco_pp_V0_ELscan.C(500, 79513, 0, ".", 0, "run3pp",
 *                  "ana561_2025p013_v001", "HITS_pp", "physics", true, false, 1.0e-4)'
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
#include <sitrackreco/SiTpc_VertexFinder.h>

#include <trackingdiagnostics/Tpc_ModuleTrackDisplay.h>
#include <trackingdiagnostics/Tpc_AssembledTrackDisplay.h>
#include <trackingdiagnostics/Tpc_PolyClusterDisplay.h>
#include <trackingdiagnostics/Tpc_PolyClusterResiduals.h>
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

void Fun4All_raw_hit_Poly_reco_pp_V0_ELscan(
    const int nEvents = 500,
    const int runnumber = 79513,
    const int segment = 0,
    const std::string& outdir = ".",
    const int nSkip = 0,
    const std::string& collision = "run3pp",
    const std::string& production = "ana561_2025p013_v001",
    const std::string& outfilename = "ppFieldOn",
    const std::string& datatype = "physics",
    const bool runHadronReco = true,   // K0S / Lambda / phi / D0 from the Si+TPC tracks
    const bool runQA = false,          // Si-only / TPC-only vertices, SiTpc_VertexQA, SiTpc_TrackResiduals
    const double kalmanDedxGeVPerCm = 1.0e-4)  // uniform effective dE/dx scan; default EL100
{
  int MyVerbosity=0;

  //==============================================================
  // ---- Si - TPC alignment test matrix (run the SAME events for every configuration and compare the
  // L0 residual fits: A + B sin(phi) + C cos(phi) per side / charge, the q/pT slope, mean match_dphi)
  //   tpcFrameMode 0: TPC points unchanged (reference)
  //                1: TPC shifted by the Si - TPC offset measured from the MVTX L0 residuals of the
  //                   TPC-only circle (B = dx, C = -dy, average of the side and charge fits):
  //                   dx = +0.028 cm, dy = +0.164 cm  (+-0.006 cm)
  //                2: old TPC-only beam spot (-0.1106, +0.2330) cm (former centerTpcBeamToo)
  //                3: 2nd iteration of mode 1: mode 1 + remaining offset of the tpc1 run
  //                   (fit_sitpc_offset.C: dx = +0.0146, dy = +0.0750) = (0.0426, 0.2390) cm
  //                4: linear extrapolation of modes 1 and 3 to zero remaining offset.  The residual method
  //                   under-corrects (the Si chains are searched around the TPC prediction): applying a
  //                   shift removed only ~40-80 % of it (remaining dy 0.075 -> 0.029 cm, dx 0.0146 -> 0.0031 cm
  //                   for lines 0.164 -> 0.239, 0.028 -> 0.0426) -> (0.0465, 0.287) cm
  //                5: mode 4 moved with the silicon: the 2nd Si beam-spot iteration shifts every Si point by
  //                   -(0.00336, -0.01448) cm, the TPC line gets the same shift to keep the TPC - Si alignment
  //                   of mode 4 -> (0.0499, 0.2725) cm
  //   A translation of the TPC moves every TPC circle rigidly: TPC pt is NOT changed.  Expected with
  //   mode 1: B, C -> ~0; A (charge-even constant), the q/pT slope and match_dphi unchanged.
  //   useSurveyFieldGeometry: PHGarfield drift model with the survey field geometry.  It moves the TPC
  //   clusters (r, r*phi), so A, the q/pT slope, match_dphi AND the TPC pt can change: check K0S mass.
  //   The TPC offset has to be re-measured (mode 0) after changing it.
  const int tpcFrameMode = 5;   // 4 = converged TPC line for the first Si beam spot; 5 = the same after the 2nd Si beam-spot iteration
  const bool useSurveyFieldGeometry = false;
  // Si alignment file; the Si beam-spot constants below were measured with the official file
  const bool useOfficialSiAlignment = true;
  const std::string siAlignParamsFile = useOfficialSiAlignment ? "si_align_params.txt" : "si_align_params_zero.txt";
  const double tpcLineXcm = tpcFrameMode == 1 ? 0.028 : (tpcFrameMode == 2 ? -0.110619 : (tpcFrameMode == 3 ? 0.0426 : (tpcFrameMode == 4 ? 0.0465 : (tpcFrameMode == 5 ? 0.0499 : 0.0))));
  const double tpcLineYcm = tpcFrameMode == 1 ? 0.164 : (tpcFrameMode == 2 ? 0.232978 : (tpcFrameMode == 3 ? 0.2390 : (tpcFrameMode == 4 ? 0.287 : (tpcFrameMode == 5 ? 0.2725 : 0.0))));
  const bool hardwareFromActs = true;  // SiHitSeedReco::setHardwareFromActs (chaining coordinates from the official positions)
  // Si-level DCA_xy / vertices for D0:
  //   inttPhysOffset: INTT physics points relative to the MVTX (plot_phys_steps.C: new = old - (b_INTT - b_MVTX)).
  //     The first try (+0.019, -0.034) had the wrong sign: b_INTT - b_MVTX went from (+0.019, -0.034) to
  //     (+0.038, -0.068) cm, i.e. the offset adds to b_INTT one to one -> (-0.019, +0.034).  After changing it
  //     re-measure the Si beam spot (plot_vertices.C, Si only) and ADD it to siBeamX0cm / siBeamY0cm.
  //   hybridXY: Si+TPC track = TPC curvature + circle centre from the Si points (SiTpcTrackMatcher::setHybridXY)
  const double inttPhysOffsetXcm = -0.019, inttPhysOffsetYcm = 0.034;
  const bool hybridXY = true;
  const std::string configTag = std::format("_tpc{}_survey{}_si{}_hw{}{}", tpcFrameMode, useSurveyFieldGeometry ? 1 : 0,
                                            useOfficialSiAlignment ? "official" : "zero", hardwareFromActs ? "acts" : "analytic",
                                            hybridXY ? "_hybrid" : "");
  const std::string outname = outfilename + configTag;
  //==============================================================
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
  garfield->SetUseSurveyGeometry(useSurveyFieldGeometry);
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
  // ---- Si frame
  // The silicon is placed by the official tracking geometry (setUseActsGeometry) incl. the MVTX global
  // displacement D, then shifted so that its beam spot is at (0, 0) (correctSiBeamSpot below).
  // Convention (sPHENIX coresoftware):
  //   MakeActsGeometry::makeMvtxMapPairs: MVTX surfaces of the ACTS geometry sit at nominal + D,
  //     D = MVTX_ALIGNMENT global displacement = (6.2955, -1.06874, -6.66289) mm (log line
  //     "Global displacement"), the INTT gets NO displacement;
  //   PHActsSiliconSeeding: m_mvtx_x0, m_mvtx_y0 = (+0.6, -0.1) cm = MVTX layer circles centre
  //     in the global frame, INTT circles centred at (0, 0);
  //   TRACKINGALIGNMENT translations are small corrections on top of that geometry.
  // So: global = detector + D for the MVTX (SiDetectorFrame::toGlobal adds the centre: sign OK),
  // and global = detector for the INTT.  SiDetectorFrame now has a separate INTT centre
  // (setInttCenterMm), so the OFFICIAL si_align_params.txt is used unchanged:
  //   MVTX centre = D (x, y), INTT centre = (0, 0, 0).
  // check_si_alignment.py si_align_params.txt: MVTX and INTT means both small -> this setup.
  const double siDispXmm = 6.2955, siDispYmm = -1.06874, siDispZmm = -6.66289;  // D [mm]

  // ---- Si beam spot correction: the WHOLE silicon (MVTX + INTT, both halves together) is shifted in
  // x, y so that its beam is at (0, 0).  The TPC is NOT moved (TPC beam line 0: tpcToTpcFrame is the
  // identity, the TPC points stay in the raw TPC frame).
  //   Si output = Si global - (x0, y0)     (SiTpcBeamAlignment::siHalfToTpcAxis, SiFull, x, y only)
  // (x0, y0) = beam spot of the Si-only vertices, measured with EXACTLY this silicon setup (official
  // ActsGeometry positions, z shift +D_z, MVTX centre D, INTT survey centre) and the correction OFF:
  //   plot_vertices.C, Si only: x0 = -0.07411 +- 0.00079 cm, y0 = +0.15380 +- 0.00074 cm
  //   (Si+TPC: -0.07236, +0.14684).  The slopes are stored but not applied (no tilt correction).
  // To iterate: run with the correction ON and ADD the new Si-only x0, y0 to the values below.
  // Re-measure if the silicon geometry / alignment changes.
  const bool correctSiBeamSpot = true;
  // 2nd iteration (after the INTT physics offset (-0.019, +0.034) cm): the Si-only beam line came out at
  // x0 = +0.00336, y0 = -0.01448 cm with the previous (-0.074114, 0.153802) -> added
  const double siBeamX0cm = -0.070755, siBeamY0cm = 0.139325;
  const double siBeamDxDz = -8.932876e-04, siBeamDyDz = 9.177683e-05;  // metadata only
  // TPC frame: TPC output = TPC point - (tpcLineXcm, tpcLineYcm) (a frame shift, not a change of the
  // TPC internal alignment), see tpcFrameMode at the top.

  SiTpcBeamAlignment beamAlign;
  beamAlign.setEnabled(correctSiBeamSpot);  // false: TPC and Si points are used as they are
  beamAlign.setCenterBeam(true);            // every part moved by its own beam line
  beamAlign.setSiPerHalf(false);            // one offset for the whole silicon (SiFull)
  beamAlign.setBeamLineCm(SiTpcBeamAlignment::SiFull, siBeamX0cm, siBeamY0cm, siBeamDxDz, siBeamDyDz);
  beamAlign.setBeamLineCm(SiTpcBeamAlignment::Tpc, tpcLineXcm, tpcLineYcm, 0.0, 0.0);  // slopes not applied
  beamAlign.print();

  // Silicon intrinsic -> ideal detector frame.
  const double siUphiRotation = 0.1481;
  // Ideal detector -> global: MVTX + D (D is what the ACTS geometry uses), INTT + 0
  const double siCenterXmm = siDispXmm;
  const double siCenterYmm = siDispYmm;
  // z stays 0: SiHitSeedReco seeds with the detector-frame z but builds the TPC residuals with
  // the global z, so a non-zero centre z would shift every chain's dz by D_z = -0.666 cm against
  // the OffsetPrior (sigma 0.35 cm) and the resolver dz sigma.  Apply D_z only in the z-alignment
  // step, together with setTpcZOffset() and the seeding windows.
  const double siCenterZmm =  0.0;
  // INTT: the tracking geometry uses the INTT SURVEY geometry (log: "Use Intt survey geometry? m_inttSurvey =1"):
  // barrel centre (0.4027, -2.8866) mm, sensor radii 7.453 / 8.046 / 9.934 / 10.569 cm instead of the
  // design 7.188 / 7.732 / 9.680 / 10.262 cm.  The own geometry (setUseActsGeometry(false)) has the design
  // radii, so only setUseActsGeometry(true) places the INTT hits right.
  const double inttCenterXmm = 0.4027, inttCenterYmm = -2.8866, inttCenterZmm = 0.0;
 
  //------------------------------------------------------------

  auto* siHitSeed = new SiHitSeedReco("SiHitSeedReco");
  siHitSeed->Verbosity(MyVerbosity);  // 1: run summary + residual widths at End; 2: one line per TPC track (slow); 3: + every chain
  siHitSeed->setUphiRotation(siUphiRotation);
  siHitSeed->setDetectorCenterMm(siCenterXmm, siCenterYmm, siCenterZmm);   // MVTX
  siHitSeed->setInttCenterMm(inttCenterXmm, inttCenterYmm, inttCenterZmm);  // INTT
  siHitSeed->setBeamAlignment(beamAlign);
  siHitSeed->setApplySiAlignment(true);                     // default off
  // REQUIRED with setApplySiAlignment(true): no CDB fallback, a missing file aborts the run.
  // Use an absolute path on batch workers.
  // official TRACKINGALIGNMENT constants, unchanged (MVTX centre D + INTT centre 0)
  siHitSeed->setSiAlignmentParamsFile(siAlignParamsFile);
  siHitSeed->setUseNewSiliconRotationOrder(false); 
  // ---- silicon PHYSICS points (track fits, vertices); chaining always uses the hardware Uphi, Vz
  siHitSeed->setUseFlatSensorGeometry(true);     // own geometry: flat tilted MVTX staves (false: old cylinders)
  siHitSeed->setCompareWithActsGeometry(true);   // End: per-layer table own - official (ActsGeometry) positions [mm]
  // official geometry = the TRKR_CLUSTER positions (MVTX displacement D, INTT SURVEY geometry,
  // TRACKINGALIGNMENT from the CDB).  With it the INTT centre for the chaining predictions should
  // be the INTT survey barrel centre (MakeActsGeometry: 0.4027, -2.8866 mm), see inttCenterXmm above.
  // own - official table (run of 2 events): MVTX dx, dy, dr, r*dphi < 0.04 mm (own MVTX geometry is
  // right), dz = +6.65 mm = -D_z; INTT dr = -2 ... -3.4 mm, rms 1.3 - 3.2 mm, dz = +5.1 mm (own INTT
  // geometry is the design one, the data geometry is the survey one) -> use the official positions
  const bool useActsSiGeometry = true;
  siHitSeed->setUseActsGeometry(useActsSiGeometry);
  siHitSeed->setActsZShiftCm(-siDispZmm / 10.0);  // +0.666 cm: same Si z frame as before (MVTX at nominal z), see SiHitSeedReco.h
  // hardware (Uphi, Vz) for the blobs and the chaining from the official hit positions (default true with
  // setUseActsGeometry): fixes the analytic INTT model (design radii, per-ladder offsets up to 0.03 rad,
  // sensors with reversed strip direction, ~15 mrad MVTX - INTT rotation).  End prints analytic - official.
  siHitSeed->setHardwareFromActs(hardwareFromActs);
  siHitSeed->setInttPhysicsOffsetCm(inttPhysOffsetXcm, inttPhysOffsetYcm);  // INTT relative to the MVTX (physics points)
  // ---- z alignment iteration (plot_z_alignment.C page 4: "add ... to setSiLayerZOffsetCm"); 0 = none
  const double siLayerZOffsetCm[7] = {0, 0, 0, 0, 0, 0, 0};  // MVTX L0-L2, INTT L3-L6 [cm]
  for (int l = 0; l < 7; ++l)
  {
    siHitSeed->setSiLayerZOffsetCm(l, siLayerZOffsetCm[l]);
  }
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
    resolver->Verbosity(MyVerbosity);  // 1: summary per event + End; 2: every track with its hypotheses (reasons for the grey tracks)
    resolver->setKeepUntimedLone(true);
    resolver->setMinUntimedLoneCredit(-1);
    resolver->setVertexChainReselect(true);
    se->registerSubsystem(resolver);
  }


  auto* siTraj = new SiTrajectoryFitter("SiTrajectoryFitter");
  siTraj->Verbosity(MyVerbosity);
  siTraj->setInputNodeName("SI_HIT_SEED_EVENT");
  siTraj->setOutputNodeName("SI_TRAJECTORY");
  siTraj->setFitSelectedChainsOnly(true);
  siTraj->setUphiRotation(siUphiRotation);
  siTraj->setDetectorCenterMm(siCenterXmm, siCenterYmm, siCenterZmm);  // same frame as SiHitSeedReco
  siTraj->setInttCenterMm(inttCenterXmm, inttCenterYmm, inttCenterZmm);
  siTraj->setBeamAlignment(beamAlign);
  siTraj->setBeamPositionCm(0.0, 0.0);   // Si beam at (0,0) after the Si beam-spot correction
  siTraj->setUseIntt(true);
  siTraj->setMinPoints(3);
  siTraj->setZWeights(1.0, 1.0e-4);      // MVTX pixel vs INTT strip (sigma_z 0.003 vs 0.46 cm); 0.01 moved z0 by ~0.5 mm
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
  siTpcMatch->Verbosity(MyVerbosity);
  siTpcMatch->setAssociation(SiTpcTrackMatcher::Association::FromSiChains);
  siTpcMatch->setBeamAlignment(beamAlign);
  siTpcMatch->setBeamPositionCm(0.0, 0.0);
  siTpcMatch->setMatchWindow(0.2, 0.1);  // |dphi| [rad], |deta|: matching variables (not a cut for chains)
  siTpcMatch->setMatchWindowDz0(0.0);    // z0 not used (TPC z depends on the crossing)
  siTpcMatch->setRequireSameCharge(false);
  siTpcMatch->setMinTpcClusters(10);
  siTpcMatch->setXYWeights(0, 0, 1.0);   // MVTX, INTT, TPC weights in the circle fit (curvature from the TPC)
  // hybrid xy: circle centre refitted (radius fixed) to the Si points, weights ~1/sigma_rphi^2 (MVTX 5 um,
  // INTT 25 um -> 0.04); dca / pca / phi / z0 of the Si+TPC track at silicon precision, TPC pt unchanged
  siTpcMatch->setHybridXY(hybridXY, 1.0, 0.04, 2);
  // z(s) weights ~ 1/sigma^2 relative to the MVTX: MVTX ~0.003 cm (pixel + residual alignment), INTT strip
  // ~0.46 cm (-> 1e-4), TPC 0.1 cm but with multiple scattering between the MVTX and the TPC (a few
  // mrad at 0.3-1 GeV) its slope must not steer z0 (-> 1e-4; it still fixes the slope of 2-MVTX-layer
  // tracks).  Equal weights (1, 1, 1) let the INTT strips decide z0: ~1 mm prompt |DCA_z| (toy).
  siTpcMatch->setZWeights(1.0, 1.0e-4, 1.0e-4);  // MVTX, INTT, TPC weights in z(s)
  siTpcMatch->setFitTpcZOffset(true);    // free TPC z offset (crossing / t0) in the refit
  se->registerSubsystem(siTpcMatch);

  //------------------------------------------------------------
  // collision vertices from three track sources, same pair finder (PairVertexFinder), one set of
  // vertices per decided bunch crossing (TPC_CROSSING_DECISIONS, tier <= 1)
  //   Si only : SI_TRAJECTORY  (SiTrajectoryFitter)  -> SiPolyVertexMap
  //   TPC only: TPC_POLYTRACKS (Tpc_PolyTrackReco)    -> TpcPolyVertexMap
  //   Si+TPC  : SITPC_TRACKS   (SiTpcTrackMatcher)    -> SiTpcPolyVertexMap
  // Si-only and TPC-only vertices are only used by the QA modules (runQA); the hadron
  // reconstruction reads the Si+TPC vertices (SiTpcPolyVertexMap)
  if (runQA)
  {
  auto* vtxSi = new SiTpc_VertexFinder("SiVertexFinder", SiTpc_VertexFinder::Source::Si);
  vtxSi->Verbosity(0);
  vtxSi->setMinSiPoints(3);          // MVTX + INTT points of the chain
  vtxSi->setMinMvtxPoints(2);
  vtxSi->setLineTrackResolution(0.1, 0.1);  // MVTX-only chains are straight-line fits (lever arm < 3 cm): ~1 mm at the beam
  se->registerSubsystem(vtxSi);

  auto* vtxTpc = new SiTpc_VertexFinder("TpcVertexFinder", SiTpc_VertexFinder::Source::Tpc);
  vtxTpc->Verbosity(0);
  // beam-spot constraint in the RAW TPC frame of the poly tracks = the TPC line (the vertices are shifted by -line
  // below, so the constraint lands at (0, 0) like the Si beam).  It was fixed at (-0.106835, 0.267482): the TPC-only
  // vertices then followed that constant instead of the TPC line (mode 2: spot ~ (0, 0.01), mode 3: ~ (-0.13, 0.00)),
  // so the TPC-only beam spot was not an alignment measurement.  With the TPC line 0 the raw constraint is kept.
  if (tpcFrameMode == 0)
  {
    vtxTpc->setBeamSpot(-0.106835, 0.267482);
  }
  else
  {
    vtxTpc->setBeamSpot(tpcLineXcm, tpcLineYcm);
  }
  // TPC vertices in the same x, y frame as the TPC points (shifted by -TPC line, see tpcFrameMode),
  // so Si - TPC vertex differences show the remaining TPC offset; z: shifted by -D_z, the same z
  // frame as the Si (setActsZShiftCm)
  vtxTpc->setVertexShiftCm(-tpcLineXcm, -tpcLineYcm, -siDispZmm / 10.0);
  vtxTpc->setMinTpcClusters(10);
  se->registerSubsystem(vtxTpc);
  }

  // Si+TPC vertex fit settings: shared by SiTpc_VertexFinder and the D0 vertex refit of the hadron
  // reconstruction (TpcV0CandidateTree::set_d0_vertex_refit), which must use the same weights
  const double siTpcVtxSigmaRPhi = 0.002, siTpcVtxSigmaZ = 0.04;  // [cm]
  const double siTpcVtxMsTerm = 0.006;                             // [cm GeV]
  const double siTpcVtxMaxChi2 = 16.0;
  const double siTpcBeamSigmaX = 0.013, siTpcBeamSigmaY = 0.013;   // [cm]
  // vertex tracks = good Si seeds only (also the D0 daughter selection below)
  const unsigned int siTpcVtxMinMvtx = 3;
  const double siTpcVtxMaxMatchDphi = 0.05;  // [rad]

  auto* vtxSiTpc = new SiTpc_VertexFinder("SiTpcVertexFinder", SiTpc_VertexFinder::Source::SiTpc);
  vtxSiTpc->Verbosity(0);
  vtxSiTpc->setMinTpcClusters(10);
  if (hybridXY)
  {
    // the Si+TPC perigee is the hybrid track (Si position, TPC curvature): use it directly (the "position
    // from Si" mode puts the circle through the innermost and OUTERMOST Si point, i.e. an INTT point)
    vtxSiTpc->setSiTpcPositionFromSi(false);
    // weights: hybrid track resolution at the vertex, r*phi 20 um (+) 60 um GeV / p (plot_dca_xy.C: 20 um at
    // pT > 1 GeV, ~70 um GeV incl. the vertex error), z 0.4 mm (Si+TPC z0 - vertex: sigma 0.5 mm); track chi2 cut
    // 16 (the first try, 0.002 / 0.03 / 0.004 / 12, dropped too many tracks: 2.6 instead of 3.2 per vertex)
    vtxSiTpc->setTrackResolution(siTpcVtxSigmaRPhi, siTpcVtxSigmaZ);
    vtxSiTpc->setMultipleScatteringTerm(siTpcVtxMsTerm);
    vtxSiTpc->setMaxTrackChi2(siTpcVtxMaxChi2);
    // weak beam-spot constraint (beam at (0, 0) after the Si beam-spot correction): stabilises the 2-3 track
    // collisions.  Measure the beam spot with the Si-only vertices (no constraint), not with these.
    vtxSiTpc->setUseBeamSpotConstraint(true);
    vtxSiTpc->setBeamSpot(0.0, 0.0);
    // beam size: intrinsic width of the beam-line fit of the good-seed Si+TPC vertices (0.0134 / 0.0131 cm; the
    // earlier 0.021 / 0.019 included vertex errors of the unselected tracks)
    vtxSiTpc->setBeamSpotSigma(siTpcBeamSigmaX, siTpcBeamSigmaY);
    // vertex tracks = good Si seeds only: >= 3 MVTX points (plot_dca_xy.C: 2-MVTX tracks have a DCA core of
    // 1-5 mm, >= 3 MVTX 85-130 um incl. the vertex error), pT >= 0.3 GeV (multiple scattering), |phi_Si - phi_TPC|
    // < 0.05 rad (core fraction 73-89 % below 0.05 rad, 44 % above; 0.02 rejected good tracks).  Tracks failing
    // these are not in the vertex but keep their own (hybrid) DCA for the analysis.
    //   minTracks 3: 658 vertices (3.8 tracks) in 500 events, ~100 um for 4 tracks (split collisions)
    //   minTracks 2: ~4x more collisions get a vertex; for 2 tracks the beam constraint (0.13 mm) dominates
    // (K0S / Lambda / phi use these vertices too; the D0 asks for >= 4 tracks itself, see the hadron reco)
    const unsigned int siTpcVtxMinTracks = 3;
    vtxSiTpc->setMinMvtxPoints(siTpcVtxMinMvtx);
    vtxSiTpc->setTrackPtCut(0.3);
    vtxSiTpc->setMaxMatchDphi(siTpcVtxMaxMatchDphi);
    vtxSiTpc->setMinTracksPerVertex(siTpcVtxMinTracks);
  }
  else
  {
    vtxSiTpc->setSiTpcPositionFromSi(true);  // position from the Si points, curvature from the Si+TPC fit (its circle uses TPC points only: setXYWeights(0, 0, 1))
  }
  se->registerSubsystem(vtxSiTpc);


  //------------------------------------------------------------
  // Hadron reconstruction from the Si-associated tracks (SITPC_TRACKS), Kalman/RKN refit.
  // Frame: the SITPC_TRACKS points are already in the matcher frame (beamAlign: Si beam spot at
  // (0, 0), TPC moved by its line of tpcFrameMode), so the beam line is (0, 0) here.  The TPC z
  // of every track is moved into the Si z frame by its fitted TPC z offset.  Collision vertex =
  // SiTpcPolyVertexMap (vtxSiTpc above); pairs only inside one collision.
  if (runHadronReco)
  {
    // The EL tag is in keV/cm; EL0, EL50, EL100, EL150 for the scan.
    // Tag each hadron output separately so comparison runs cannot overwrite one another.
    if (!std::isfinite(kalmanDedxGeVPerCm) || kalmanDedxGeVPerCm < 0.0 || kalmanDedxGeVPerCm > 0.01)
    {
      std::cerr << "Invalid Kalman dE/dx (GeV/cm): " << kalmanDedxGeVPerCm << std::endl;
      return;
    }
    const int energyLossTag = static_cast<int>(std::lround(kalmanDedxGeVPerCm * 1.0e6));
    const std::string hadronFile = outdir + "/hadron_sitpc_" + outfilename +
        "_EL" + std::to_string(energyLossTag) + "_" + std::to_string(runnumber) +
        "_" + std::to_string(segment) + ".root";
    std::cout << "[V0 energy-loss scan] dE/dx = " << kalmanDedxGeVPerCm
              << " GeV/cm; output = " << hadronFile << std::endl;
    auto *hadron = new tpchadronreco::TpcV0CandidateTree("TpcV0CandidateTree", hadronFile);
    hadron->Verbosity(0);
    hadron->use_sitpc_tracks(true);
    hadron->set_sitpc_track_node("SITPC_TRACKS");
    hadron->set_tpc_sa_cluster_node("TPC_POLYCLUSTERS");
    hadron->set_tpc_sa_track_node("TPC_POLYTRACKS");  // dE/dx (+ unmatched TPC tracks if enabled)
    hadron->set_tpc_crossing_decision_node("TPC_CROSSING_DECISIONS");
    hadron->set_require_same_crossing(true);
    // true: TPC-only tracks too (V0s decaying outside the MVTX; needed for the TPC-only vs Si
    // comparison of MakeK0sPairHistograms.C trackClass "tpconly" / "si")
    hadron->set_sitpc_add_unmatched_tpc_tracks(false);

    // Si+TPC track quality and crossing
    //                             MVTX INTT  Si  TPC  circle rms  z rms [cm]
    hadron->set_sitpc_track_selection(2,   0,   2,  10,   -1.0,      -1.0);
    hadron->set_sitpc_max_crossing_tier(1);
    hadron->set_sitpc_require_timed_collision(false);
    hadron->set_sitpc_apply_tpc_z_offset(true);  // the matcher fits a free TPC z offset (setFitTpcZOffset)

    // collision vertex: the SiTpcPolyVertexMap vertex containing the track, else the vertex of its
    // crossing nearest in z (|dz| < 0.5 cm); fallback: resolver Si z, then the robust mean of the
    // prompt tracks.  pv_source in the trees: 3 map, 1 resolver, 2 tracks, 0 fixed
    hadron->set_sitpc_vertex_mode("map");
    hadron->set_sitpc_vertex_map_node("SiTpcPolyVertexMap");
    hadron->set_sitpc_vertex_attach_max_dz(0.5);
    hadron->set_sitpc_vertex_track_selection(0.1, 0.5, 2);  // fallback: |dca_xy| [cm], z window [cm], min tracks
    hadron->set_sitpc_beam_line(0.0, 0.0, 0.0, 0.0);       // beam at (0, 0) after beamAlign
    hadron->set_primary_vertex(0.0, 0.0, 0.0);              // fallback only

    // ---- track model: Kalman/RKN refit of the Si + TPC points of SITPC_TRACKS with the field map.
    // Seed = the matcher helix (hybrid xy: Si circle centre, TPC curvature).
    hadron->set_track_fit_method("kalman");
    hadron->set_bfield(1.4);
    hadron->set_min_points(12);
    hadron->set_point_order("radius-descending");
    hadron->set_sitpc_helix_search(10.0, 5.0);
    hadron->use_kalman_field_map(true);  // FIELDMAP_TRACKING from the CDB
    hadron->set_kalman_analytic_uniform_propagation(false);
    hadron->set_kalman_rkn4(10.0, 0, 1, 2000);  // max step [cm], tolerance, trials, max steps
    hadron->set_kalman_fast_field_jacobian(true);
    hadron->set_kalman_fast_field_pca(true);
    hadron->set_kalman_field_pca_refine_iterations(1);
    hadron->set_kalman_initial_sigmas(0.05, 0.010, 0.020);      // position [cm], phi, tanl around the seed
    hadron->set_kalman_relative_qop_seed(true, 0.05, 5.0e-4);   // q/pT prior relative to the seed
    hadron->set_kalman_absolute_qop_seed_sigma(0.20);
    hadron->set_kalman_robust_measurement_updates(true, 16.0, 100.0);  // down-weight / drop outliers
    hadron->set_kalman_fit_iterations(1);
    hadron->set_kalman_process_sigmas(0.0, 0.0, 0.0, 0.0);
    // multiple scattering (uniform material, Kalman process noise): without it the Si points are
    // trusted over the whole Si -> TPC gap.  ~2% X0 over ~80 cm.
    //                          X0/cm   MS scale  dE/dx  dE/dx sigma
    // Effective UNIFORM dE/dx only (not yet Bethe-Bloch or geometry aware).
    // Note: current V0 Kalman fit uses the pion mass for all tracks.
    hadron->set_kalman_material(2.5e-4, 1.0, kalmanDedxGeVPerCm, 0.0);

    // ---- measurement sigmas [cm] = weights 1/sigma^2
    // Si incl. residual misalignment (hybrid fit weights: MVTX 1, INTT 0.04 -> 5:1 in sigma;
    // matcher z weights MVTX 1, INTT 1e-4: INTT z = strip length)
    //                                MVTX rphi    r     z  | INTT rphi    r     z
    hadron->set_si_measurement_sigmas(0.0050, 0.0050, 0.0050, 0.0250, 0.0250, 0.50);
    // TPC by region (the four module-boundary layers are not used)
    //                                       R1 rphi   r   | R2 rphi   r   | R3 rphi   r   |  z
    hadron->set_tpc_region_measurement_sigmas(0.040, 0.200, 0.025, 0.050, 0.025, 0.200, 0.100);
    hadron->set_tpc_transition_measurement_sigmas(0.200, 0.500, 0.200);
    hadron->set_exclude_tpc_transition_layers(true);

    // ---- prompt phi / D0: momenta after a Gaussian update with the collision vertex (xy); the
    // sigmas are a floor, the fitted vertex error is added.  *_primary_constrained masses.
    hadron->set_primary_vertex_constraint(true);
    hadron->set_primary_vertex_constraint_mode("xy");
    hadron->set_primary_vertex_constraint_sigmas(0.005, 0.02);
    hadron->set_compute_constrained_phi(true);
    hadron->set_compute_constrained_d0(true);

    // ---- loose pair preselection
    hadron->set_pre_track_pt_min(0.15);
    hadron->set_pre_track_dca_xy_min(-1.0);
    hadron->set_pre_pair_dca_max(-1.0);
    hadron->set_pre_lproj_min(-1.0);
    hadron->set_pre_track_npoints_min(12);

    // ---- D0 collision vertex: a precise, unbiased vertex for the D0 (applies to pairTree and likeSignPairTree)
    //   * both daughters belong to one SiTpcPolyVertexMap vertex with >= d0MinVertexTracks tracks (incl. the
    //     daughters when they are in it), otherwise no D0 hypothesis for the pair (d0v_status 2/3)
    //   * daughters in that vertex are removed and the vertex is refitted with its other tracks (same model and
    //     weights as vtxSiTpc, >= 2 others + the beam constraint): a D0 daughter pulls a 4-track vertex by ~1/3
    //     of its impact parameter towards the decay (toy: 51 um for 150 um), i.e. Lxy and the daughter DCAs
    //     would be biased low
    //   * daughters: good Si seeds as the vertex tracks (>= 3 MVTX, |match dphi| < 0.05 rad); 2-MVTX tracks have
    //     a 1-5 mm DCA core
    //   * DCA_xy / DCA_z of the daughters, secondary vertex, Lxy, DIRA, pointing: hybrid matcher helix (Si
    //     position, TPC curvature: ~23 um at pT > 1 GeV); masses and pT: the Kalman momenta
    //   * max vertex xy error: -1 = off; the d0v_ex / d0v_ey branches hold it for an offline cut
    //   tree: d0v_status (1 = ok), d0v_x/y/z, d0v_ex/ey/ez, d0v_ntracks, d0v_nused, d0v_nremoved, d0v_dca_xy1/2,
    //   d0v_dca_z1/2, d0v_dca_product, d0v_pair_dca, d0v_sv_x/y/z, d0v_lxy(_signed), d0v_decay_length, d0v_dira,
    //   d0v_pointing_dca_xy/3d, d0v_mass_D0_sv, d0v_mass_AntiD0_sv, d0v_match_dphi1/2
    const unsigned int d0MinVertexTracks = 4;  // tracks of the SiTpc_VertexFinder vertex
    const unsigned int d0MinOtherTracks = 2;   // vertex tracks left after removing the daughters
    hadron->set_d0_collision_vertex(true, d0MinVertexTracks, true, d0MinOtherTracks, -1.0);
    hadron->set_d0_vertex_refit(siTpcVtxSigmaRPhi, siTpcVtxSigmaZ, siTpcVtxMsTerm, siTpcVtxMaxChi2,
                                true, 0.0, 0.0, siTpcBeamSigmaX, siTpcBeamSigmaY);
    hadron->set_d0_daughter_si_selection(siTpcVtxMinMvtx, siTpcVtxMaxMatchDphi);
    hadron->set_d0_use_sitpc_helix(true);

    // ---- selections (Si in the xy fit: Si-level DCA_xy cuts)
    // phi -> K+ K-  (prompt):    clu  pt   mass        |dca_xy| |dca_z| pairDCA flight  |pca z| pca dz
    hadron->set_phi_selection(true, 10, 0.15, 0.98, 1.08, 0.10,    0.20,   -1.0,   -1.0,   20.0,   0.20);
    // D0 -> K- pi+, anti-D0 -> K+ pi-; |dca_xy| / |dca_z| of the daughters to the D0 vertex (loose: the decay
    // topology is in the d0v_* branches)
    hadron->set_d0_selection(true, 10, 0.20, 1.70, 2.05, 0.50, 0.30, -1.0, -1.0, 20.0, 0.30);
    hadron->set_antid0_selection(true, 10, 0.20, 1.70, 2.05, 0.50, 0.30, -1.0, -1.0, 20.0, 0.30);
    // displaced D0 vertex (on top of the D0 selection, w.r.t. the D0 vertex; off, tune on data with d0v_*):
    //                                      Lxy  L_min L_max  DIRA  |d0|_min pairDCA pointDCA d0*d0
    hadron->set_d0_displaced_selection(false, 0.0, -1.0, 0.5,  0.90, 0.005,   0.03,   0.01,    2.0);
    // V0s: Si-matched daughters only for decays inside the MVTX (r < ~4 cm)
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

  if (runQA)
  {
  // one entry per crossing: resolver z, Si, TPC and Si+TPC vertices; End: z differences and pulls
  auto* vtxQA = new SiTpc_VertexQA("SiTpc_VertexQA", outdir + "/sitpc_vertex_qa_" + outname + ".root");
  se->registerSubsystem(vtxQA);
  




  auto* siTpcResid = new SiTpc_TrackResiduals("SiTpc_TrackResiduals", outdir + "/sitpc_track_residuals_" + outname + ".root");
  siTpcResid->Verbosity(0);
  siTpcResid->setMinPt(0.1);
  siTpcResid->setMinTpcClusters(10);
  siTpcResid->setWriteMatchingTree(true);
  // pv_* branches for plot_z_alignment.C: pair-finder vertices of the same crossing (registered above)
  siTpcResid->setSiTpcVertexMapName("SiTpcPolyVertexMap");
  siTpcResid->setSiVertexMapName("SiPolyVertexMap");
  se->registerSubsystem(siTpcResid);
  }

  auto* display = new SiTpc_EventDisplay("SiTpc_EventDisplay", outdir + "/sitpc_event_display_" + outname + ".root", 10);  // max events saved
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
  display->setInttCenterMm(inttCenterXmm, inttCenterYmm);
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

  //se->registerOutputManager(out);
  
  se->run(nEvents+nSkip);
  se->Print("NODETREE");
  se->End();
  se->PrintTimer();


  CDBInterface::instance()->Print();
  delete se;
  std::cout << "Finished" << std::endl;
  gSystem->Exit(0);
}