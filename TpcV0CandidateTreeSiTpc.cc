// Si-associated (MVTX/INTT + TPC) track input for TpcV0CandidateTree.
//
// Reads the combined Si+TPC tracks of SiTpcTrackMatcher (SITPC_TRACKS) and the
// resolved crossings of TpcCrossingResolver (TPC_CROSSING_DECISIONS) from the
// crossing-vertex-si-assoc reconstruction chain, builds the Tracklets used by
// the pair / hyperon stages, assigns one primary vertex per collision, and
// provides the displaced two-track quantities used by the D0 selection.

#include "TpcV0CandidateTree.h"

#include "TpcTrackHelixFitter.h"

#include <sitrackreco/SiTpc_Track.h>
#include <sitrackreco/SiTpc_TrackContainer.h>

#include <tpctrackreco/TpcCrossingDecision.h>
#include <tpctrackreco/TpcCrossingDecisionContainer.h>
#include <tpctrackreco/Tpc_PolyTrack.h>
#include <tpctrackreco/Tpc_PolyTrackContainer.h>

#include <globalvertex/SvtxVertex.h>
#include <globalvertex/SvtxVertexMap.h>

#include <phool/phool.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tpchadronreco
{
  namespace
  {
    constexpr double kTwoPi = 6.28318530717958647692;
    constexpr int kSiTpcFitOk = 1;          // SiTpcHelixFit::Status / Si_Trajectory::Ok
    constexpr int kLastMvtxLayer = 2;       // MVTX layers 0-2
    constexpr int kLastSiLayer = 6;         // INTT layers 3-6

    double median_of(std::vector<double> values)
    {
      if (values.empty())
      {
        return std::numeric_limits<double>::quiet_NaN();
      }
      const std::size_t mid = values.size() / 2;
      std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
      double result = values[mid];
      if (values.size() % 2 == 0)
      {
        const double lower = *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
        result = 0.5 * (result + lower);
      }
      return result;
    }

    // Robust 1-D location: median, then the mean of the values inside +-window
    // around the current estimate (two passes).  Returns the number of values used.
    unsigned int robust_location(const std::vector<double> &values, const double window, double &location)
    {
      location = median_of(values);
      if (!std::isfinite(location))
      {
        return 0;
      }
      if (!(window > 0.0))
      {
        return static_cast<unsigned int>(values.size());
      }
      unsigned int used = 0;
      for (int pass = 0; pass < 2; ++pass)
      {
        double sum = 0.0;
        used = 0;
        for (const double value : values)
        {
          if (std::abs(value - location) <= window)
          {
            sum += value;
            ++used;
          }
        }
        if (used == 0)
        {
          return 0;
        }
        location = sum / static_cast<double>(used);
      }
      return used;
    }

    std::string lowered(const std::string &value)
    {
      std::string out = value;
      std::transform(out.begin(), out.end(), out.begin(),
                     [](const unsigned char ch)
                     { return static_cast<char>(std::tolower(ch)); });
      return out;
    }
  }  // namespace

  // ------------------------------------------------------------------------
  // configuration
  // ------------------------------------------------------------------------
  bool TpcV0CandidateTree::set_sitpc_vertex_mode(const std::string &mode)
  {
    const std::string value = lowered(mode);
    if (value == "map" || value == "vertexmap" || value == "fit")
    {
      m_sitpc_vertex_mode = SiTpcVertexMode::Map;
      return true;
    }
    if (value == "resolver" || value == "si" || value == "silicon")
    {
      m_sitpc_vertex_mode = SiTpcVertexMode::Resolver;
      return true;
    }
    if (value == "tracks" || value == "track")
    {
      m_sitpc_vertex_mode = SiTpcVertexMode::Tracks;
      return true;
    }
    if (value == "fixed" || value == "beam")
    {
      m_sitpc_vertex_mode = SiTpcVertexMode::Fixed;
      return true;
    }
    std::cerr << PHWHERE << Name() << ": unknown Si+TPC vertex mode '" << mode
              << "'. Valid modes are map, resolver, tracks, and fixed." << std::endl;
    return false;
  }

  void TpcV0CandidateTree::set_d0_displaced_selection(
      const bool enabled,
      const double lxy_min,
      const double decay_length_min,
      const double decay_length_max,
      const double dira_min,
      const double track_dca_xy_min,
      const double pair_dca_max,
      const double pointing_dca_xy_max,
      const double dca_product_max)
  {
    m_d0_displaced_cuts = {enabled, lxy_min, decay_length_min, decay_length_max,
                           dira_min, track_dca_xy_min, pair_dca_max,
                           pointing_dca_xy_max, dca_product_max};
  }

  // ------------------------------------------------------------------------
  // tracklets from SITPC_TRACKS
  // ------------------------------------------------------------------------
  std::map<int, TpcV0CandidateTree::Tracklet> TpcV0CandidateTree::build_sitpc_tracklets(
      const SiTpc_TrackContainer *sitpc_tracks,
      Tpc_PolyTrackContainer *tpc_tracks,
      const TpcCrossingDecisionContainer *crossing_decisions) const
  {
    std::map<int, Tracklet> tracklets;
    if (!sitpc_tracks)
    {
      return tracklets;
    }

    // TpcCrossingDecisionContainer::get_decision() takes the ASSEMBLED TRACK ID
    // (the container is a map keyed by it), not an index.
    const auto find_decision = [crossing_decisions](const unsigned int assembled_track_id)
        -> const TpcCrossingDecision *
    {
      if (!crossing_decisions)
      {
        return nullptr;
      }
      const TpcCrossingDecision *decision = crossing_decisions->get_decision(assembled_track_id);
      return (decision && decision->isValid() &&
              decision->get_assembled_track_id() == assembled_track_id)
                 ? decision
                 : nullptr;
    };

    // TPC poly tracks by id (dE/dx and cluster count); the stored index is tried first.
    std::map<unsigned int, const Tpc_PolyTrack *> tpc_by_id;
    if (tpc_tracks)
    {
      for (unsigned int itrack = 0; itrack < tpc_tracks->size(); ++itrack)
      {
        const Tpc_PolyTrack *track = tpc_tracks->get_track(itrack);
        if (track)
        {
          tpc_by_id[track->get_track_id()] = track;
        }
      }
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();

    for (unsigned int itrack = 0; itrack < sitpc_tracks->size(); ++itrack)
    {
      const SiTpc_Track *track = sitpc_tracks->get_track(itrack);
      if (!track)
      {
        continue;
      }
      ++m_counter_sitpc_seen;

      // ---- combined fit
      const int charge = track->get_charge();
      const double track_pt = track->get_pt();
      if (track->get_fit_status() != kSiTpcFitOk || charge == 0 ||
          !std::isfinite(track_pt) || track_pt <= 0.0 ||
          !std::isfinite(track->get_radius()) || track->get_radius() <= 0.0 ||
          !std::isfinite(track->get_tanl()) || !std::isfinite(track->get_z0()) ||
          !std::isfinite(track->get_phi()) || std::abs(track->get_helicity()) != 1)
      {
        ++m_counter_sitpc_reject_fit;
        continue;
      }

      // ---- point content
      unsigned int n_mvtx = 0;
      unsigned int n_intt = 0;
      unsigned int n_tpc = 0;
      for (unsigned int ipoint = 0; ipoint < track->size_points(); ++ipoint)
      {
        const int layer = track->get_point_layer(ipoint);
        if (track->get_point_type(ipoint) == SiTpc_Track::SiPoint)
        {
          if (layer >= 0 && layer <= kLastMvtxLayer)
          {
            ++n_mvtx;
          }
          else if (layer > kLastMvtxLayer && layer <= kLastSiLayer)
          {
            ++n_intt;
          }
        }
        else if (track->get_point_type(ipoint) == SiTpc_Track::TpcPoint)
        {
          ++n_tpc;
        }
      }
      const unsigned int n_si = n_mvtx + n_intt;
      const bool quality_ok =
          n_mvtx >= m_sitpc_min_mvtx && n_intt >= m_sitpc_min_intt &&
          n_si >= m_sitpc_min_si_points && n_tpc >= m_sitpc_min_tpc_points &&
          (m_sitpc_max_circle_rms_cm < 0.0 ||
           (std::isfinite(track->get_circle_rms()) && track->get_circle_rms() <= m_sitpc_max_circle_rms_cm)) &&
          (m_sitpc_max_z_rms_cm < 0.0 ||
           (std::isfinite(track->get_z_rms()) && track->get_z_rms() <= m_sitpc_max_z_rms_cm));
      if (!quality_ok)
      {
        ++m_counter_sitpc_reject_quality;
        continue;
      }

      // ---- resolved crossing
      const unsigned int source_id = track->get_tpc_assembled_track_id();
      const TpcCrossingDecision *decision = find_decision(source_id);
      if (!decision && m_sitpc_require_crossing_decision)
      {
        ++m_counter_sitpc_reject_crossing;
        continue;
      }
      const unsigned char tier = decision ? decision->get_selected_tier()
                                          : std::numeric_limits<unsigned char>::max();
      if (decision && m_sitpc_max_crossing_tier >= 0 &&
          static_cast<int>(tier) > m_sitpc_max_crossing_tier)
      {
        ++m_counter_sitpc_reject_crossing;
        continue;
      }
      const double si_vertex_z = decision ? static_cast<double>(decision->get_silicon_vertex_z()) : nan;
      if (m_sitpc_require_timed_collision && !std::isfinite(si_vertex_z))
      {
        ++m_counter_sitpc_reject_crossing;
        continue;
      }

      Tracklet tracklet;
      tracklet.track_id = static_cast<int>(track->get_tpc_track_id());
      tracklet.source_assembled_track_id = source_id;
      tracklet.shower_id = static_cast<int>(source_id);
      if (decision)
      {
        tracklet.crossing_valid = true;
        tracklet.crossing = decision->get_selected_crossing();
        tracklet.crossing_status = decision->get_status();
      }
      tracklet.crossing_tier = tier;
      tracklet.si_vertex_z = si_vertex_z;
      tracklet.has_si_vertex_z = std::isfinite(si_vertex_z);
      tracklet.charge = charge > 0 ? 1 : -1;

      tracklet.has_si = true;
      tracklet.sitpc_index = itrack;
      tracklet.sitpc_id = track->get_id();
      tracklet.si_chain_id = track->get_si_chain_id();
      tracklet.n_mvtx = n_mvtx;
      tracklet.n_intt = n_intt;
      tracklet.n_si_points = n_si;
      tracklet.sitpc_fit_status = track->get_fit_status();
      tracklet.sitpc_dca = track->get_dca();
      tracklet.sitpc_z0 = track->get_z0();
      tracklet.sitpc_circle_rms = track->get_circle_rms();
      tracklet.sitpc_z_rms = track->get_z_rms();
      tracklet.sitpc_tpc_z_offset = std::isfinite(track->get_tpc_z_offset()) ? track->get_tpc_z_offset() : 0.0;
      tracklet.sitpc_match_chi2 = track->get_match_chi2();

      // ---- dE/dx from the TPC poly track
      const Tpc_PolyTrack *tpc_track = nullptr;
      if (tpc_tracks && track->get_tpc_track_index() < tpc_tracks->size())
      {
        const Tpc_PolyTrack *candidate = tpc_tracks->get_track(track->get_tpc_track_index());
        if (candidate && candidate->get_track_id() == track->get_tpc_track_id())
        {
          tpc_track = candidate;
        }
      }
      if (!tpc_track)
      {
        const auto tpc_iter = tpc_by_id.find(track->get_tpc_track_id());
        if (tpc_iter != tpc_by_id.end())
        {
          tpc_track = tpc_iter->second;
        }
      }
      if (tpc_track)
      {
        tracklet.dedx = tpc_track->get_dedx();
        tracklet.has_dedx = std::isfinite(tracklet.dedx) && tracklet.dedx > 0.0;
      }

      // ---- state at the transverse PCA to the beam (Si+TPC frame)
      const double phi = track->get_phi();
      const double tanl = track->get_tanl();
      tracklet.position = {track->get_pca_x(), track->get_pca_y(), track->get_z0()};
      tracklet.momentum = {track_pt * std::cos(phi), track_pt * std::sin(phi), track_pt * tanl};
      tracklet.truth_momentum = tracklet.momentum;

      // ---- measured points: Si layer points + TPC clusters (TPC z moved into the Si z frame)
      const double tpc_z_shift = m_sitpc_apply_tpc_z_offset ? tracklet.sitpc_tpc_z_offset : 0.0;
      std::map<int, unsigned int> side_counts;
      tracklet.points.reserve(track->size_points());
      for (unsigned int ipoint = 0; ipoint < track->size_points(); ++ipoint)
      {
        const int type = track->get_point_type(ipoint);
        if (type != SiTpc_Track::SiPoint && type != SiTpc_Track::TpcPoint)
        {
          continue;
        }
        TruthPoint point;
        point.track_id = tracklet.track_id;
        point.shower_id = tracklet.shower_id;
        point.layer = track->get_point_layer(ipoint);
        const double z_shift = type == SiTpc_Track::TpcPoint ? tpc_z_shift : 0.0;
        point.position = {track->get_point_x(ipoint),
                          track->get_point_y(ipoint),
                          track->get_point_z(ipoint) - z_shift};
        point.momentum = tracklet.momentum;
        point.t = 0.0;
        point.path = point.layer >= 0 ? static_cast<double>(point.layer)
                                      : static_cast<double>(ipoint);
        assign_point_measurement_metadata(point, tracklet.points.size());
        if (!finite(point.position))
        {
          continue;
        }
        tracklet.points.push_back(point);
        if (type == SiTpc_Track::TpcPoint)
        {
          ++side_counts[point.position.z < 0.0 ? 0 : 1];
        }
      }
      if (!side_counts.empty())
      {
        tracklet.side = std::max_element(
                            side_counts.begin(), side_counts.end(),
                            [](const auto &lhs, const auto &rhs)
                            { return lhs.second < rhs.second; })
                            ->first;
      }
      tracklet.ntpc_clusters = n_tpc;

      bool ok = false;
      if (m_use_sitpc_native_helix)
      {
        // Helix of the combined fit: circle (cx, cy, R, helicity) + z(s) = z0 + tanl s,
        // s = transverse path from the pca.  TpcTrackHelix: z = z0_h + pitch * theta.
        HelixFit &helix = tracklet.helix;
        const double direction = static_cast<double>(track->get_helicity());
        helix.cx = track->get_circle_x();
        helix.cy = track->get_circle_y();
        helix.radius = track->get_radius();
        helix.direction = direction;
        helix.bfield_t = m_bfield_t;
        const double theta0 = std::atan2(track->get_pca_y() - helix.cy,
                                         track->get_pca_x() - helix.cx);
        helix.pitch = tanl * helix.radius * direction;
        helix.z0 = track->get_z0() - helix.pitch * theta0;
        helix.theta_first = theta0;
        double theta_last = theta0;
        for (const auto &point : tracklet.points)
        {
          const double raw = std::atan2(point.position.y - helix.cy, point.position.x - helix.cx);
          double phase = std::remainder(direction * (raw - theta0), kTwoPi);
          if (phase < 0.0 && point.layer > kLastSiLayer)
          {
            phase += kTwoPi;  // outgoing TPC points lie downstream of the pca
          }
          const double theta = theta0 + direction * phase;
          if (direction * (theta - theta_last) > 0.0)
          {
            theta_last = theta;
          }
        }
        helix.theta_last = theta_last;
        helix.theta_min = std::min(theta0, theta_last);
        helix.theta_max = std::max(theta0, theta_last);
        tracklet.has_helix = TpcTrackHelixFitter::finite(helix_point(helix, theta0));
        ok = tracklet.has_helix && finalize_sitpc_native_helix(tracklet);
      }
      else
      {
        ok = finalize_pattern_tracklet(tracklet, true);
      }
      if (!ok)
      {
        ++m_counter_sitpc_reject_fit;
        continue;
      }

      tracklet.truth_momentum = tracklet.momentum;
      ++m_counter_sitpc_accepted;
      tracklets[tracklet.track_id] = std::move(tracklet);
    }
    return tracklets;
  }

  bool TpcV0CandidateTree::finalize_sitpc_native_helix(Tracklet &tracklet) const
  {
    // No refit: the points are kept for the search range and the QA trees only.
    tracklet.spatial_correction_points = 0;
    order_track_points(tracklet.points, m_point_order);
    tracklet.npoints = static_cast<int>(tracklet.points.size());
    if (tracklet.npoints < m_min_points || tracklet.charge == 0)
    {
      return false;
    }

    tracklet.has_helix_search_range =
        TpcTrackHelixFitter::measurement_anchored_search_range(
            tracklet.helix, tracklet.points,
            m_sitpc_helix_max_upstream_cm,
            m_sitpc_helix_downstream_margin_cm,
            tracklet.helix_search_range);
    if (!tracklet.has_helix_search_range)
    {
      ++m_counter_reject_helix_anchor;
      return false;
    }

    tracklet.position = helix_point(tracklet.helix, tracklet.helix.theta_first);
    tracklet.momentum = helix_momentum(tracklet.helix, tracklet.helix.theta_first);
    assign_fit_quality(tracklet);
    return finite(tracklet.position) && finite(tracklet.momentum) &&
           norm(tracklet.momentum) > 0.0;
  }

  // ------------------------------------------------------------------------
  // collision (primary) vertices: one per resolved crossing
  // ------------------------------------------------------------------------
  TpcV0CandidateTree::Vec3 TpcV0CandidateTree::beam_at_z(const double z) const
  {
    return {m_sitpc_beam_x0_cm + m_sitpc_beam_dxdz * z,
            m_sitpc_beam_y0_cm + m_sitpc_beam_dydz * z,
            z};
  }

  bool TpcV0CandidateTree::tracklet_z_at_beam(const Tracklet &tracklet, double &z, double &dca_xy) const
  {
    // z and signed DCA_xy at the transverse PCA to the beam line, from the track model used for
    // the pairs (Kalman refit, helix, or the Si+TPC matcher helix): the vertex and the DCAs use
    // the same tracks.
    Vec3 pca;
    double dca = std::numeric_limits<double>::quiet_NaN();
    const Vec3 beam = beam_at_z(std::isfinite(tracklet.sitpc_z0) ? tracklet.sitpc_z0 : 0.0);
    if (!track_pca_to_xy(tracklet, beam, pca, dca) || !std::isfinite(pca.z) || !std::isfinite(dca))
    {
      return false;
    }
    z = pca.z;
    dca_xy = dca;
    return true;
  }

  void TpcV0CandidateTree::assign_sitpc_collision_vertices(std::map<int, Tracklet> &tracklets,
                                                           const SvtxVertexMap *vertex_map) const
  {
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // z at the beam of every track with the current fit (used by "tracks" and by the
    // nearest-vertex attachment of "map")
    std::map<int, double> z_at_beam;
    std::map<int, double> dca_at_beam;
    for (const auto &entry : tracklets)
    {
      double z = nan;
      double dca = nan;
      if (tracklet_z_at_beam(entry.second, z, dca))
      {
        z_at_beam[entry.first] = z;
        dca_at_beam[entry.first] = dca;
      }
    }

    // ---- per-crossing fallback (resolver Si z, robust mean of the tracks, fixed)
    struct Accumulator
    {
      std::vector<double> si_z;
      std::vector<double> track_z;
    };
    std::map<short, Accumulator> by_crossing;
    for (const auto &entry : tracklets)
    {
      const Tracklet &tracklet = entry.second;
      if (!tracklet.crossing_valid)
      {
        continue;
      }
      Accumulator &acc = by_crossing[tracklet.crossing];
      if (tracklet.has_si_vertex_z)
      {
        acc.si_z.push_back(tracklet.si_vertex_z);
      }
      const auto iz = z_at_beam.find(entry.first);
      if (tracklet.has_si && iz != z_at_beam.end() &&
          (m_sitpc_vertex_track_dca_xy_max_cm < 0.0 ||
           std::abs(dca_at_beam[entry.first]) <= m_sitpc_vertex_track_dca_xy_max_cm))
      {
        acc.track_z.push_back(iz->second);
      }
    }

    struct Collision
    {
      Vec3 vertex;
      Vec3 sigma{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::quiet_NaN()};
      int source{0};
      unsigned int ntracks{0};
    };
    std::map<short, Collision> fallback;
    for (const auto &entry : by_crossing)
    {
      Collision collision;
      collision.vertex = m_fixed_primary_vertex;
      const Accumulator &acc = entry.second;
      const bool try_resolver = m_sitpc_vertex_mode == SiTpcVertexMode::Resolver ||
                                m_sitpc_vertex_mode == SiTpcVertexMode::Map;

      bool done = false;
      if (try_resolver && !acc.si_z.empty())
      {
        const double z = median_of(acc.si_z);
        if (std::isfinite(z))
        {
          collision.vertex = beam_at_z(z);
          collision.source = 1;
          collision.ntracks = static_cast<unsigned int>(acc.si_z.size());
          done = true;
        }
      }
      if (!done && m_sitpc_vertex_mode != SiTpcVertexMode::Fixed &&
          acc.track_z.size() >= std::max(1U, m_sitpc_vertex_min_tracks))
      {
        double z = nan;
        const unsigned int used = robust_location(acc.track_z, m_sitpc_vertex_track_z_window_cm, z);
        if (used >= std::max(1U, m_sitpc_vertex_min_tracks) && std::isfinite(z))
        {
          collision.vertex = beam_at_z(z);
          collision.source = 2;
          collision.ntracks = used;
          done = true;
        }
      }
      fallback[entry.first] = collision;
    }

    // ---- fitted vertices (SiTpc_VertexFinder): per crossing, and track id -> vertex
    struct FittedVertex
    {
      unsigned int id{0};
      Vec3 position;
      Vec3 sigma;
      unsigned int ntracks{0};
    };
    std::map<short, std::vector<FittedVertex>> fitted_by_crossing;
    std::map<unsigned int, std::pair<short, std::size_t>> vertex_of_track;  // SiTpc id -> (crossing, index)
    const bool use_map = m_sitpc_vertex_mode == SiTpcVertexMode::Map && vertex_map;
    if (use_map)
    {
      for (auto iter = vertex_map->begin(); iter != vertex_map->end(); ++iter)
      {
        const SvtxVertex *vertex = iter->second;
        if (!vertex)
        {
          continue;
        }
        FittedVertex fv;
        fv.id = iter->first;
        fv.position = {vertex->get_x(), vertex->get_y(), vertex->get_z()};
        if (!finite(fv.position))
        {
          continue;
        }
        fv.sigma = {std::sqrt(std::max(0.0F, vertex->get_error(0, 0))),
                    std::sqrt(std::max(0.0F, vertex->get_error(1, 1))),
                    std::sqrt(std::max(0.0F, vertex->get_error(2, 2)))};
        fv.ntracks = static_cast<unsigned int>(vertex->size_tracks());
        const short crossing = static_cast<short>(vertex->get_beam_crossing());
        auto &list = fitted_by_crossing[crossing];
        for (auto it = vertex->begin_tracks(); it != vertex->end_tracks(); ++it)
        {
          vertex_of_track[*it] = {crossing, list.size()};
        }
        list.push_back(fv);
      }
    }

    for (auto &entry : tracklets)
    {
      Tracklet &tracklet = entry.second;
      tracklet.has_collision_vertex = false;
      tracklet.collision_vertex = m_fixed_primary_vertex;
      tracklet.collision_vertex_sigma = {nan, nan, nan};
      tracklet.collision_vertex_source = 0;
      tracklet.collision_vertex_ntracks = 0;
      tracklet.collision_id = -1;
      tracklet.in_collision_vertex = false;
      if (!tracklet.crossing_valid)
      {
        continue;
      }

      if (use_map)
      {
        const FittedVertex *chosen = nullptr;
        // 1. the vertex that contains this Si+TPC track (ids of the same event's SITPC_TRACKS)
        if (tracklet.has_si)
        {
          const auto it = vertex_of_track.find(tracklet.sitpc_id);
          if (it != vertex_of_track.end() && it->second.first == tracklet.crossing)
          {
            chosen = &fitted_by_crossing[it->second.first][it->second.second];
            tracklet.in_collision_vertex = true;
          }
        }
        // 2. otherwise the vertex of its crossing nearest in z at the beam
        if (!chosen)
        {
          const auto ic = fitted_by_crossing.find(tracklet.crossing);
          const auto iz = z_at_beam.find(entry.first);
          if (ic != fitted_by_crossing.end() && iz != z_at_beam.end())
          {
            double best = std::numeric_limits<double>::max();
            for (const auto &fv : ic->second)
            {
              const double dz = std::abs(iz->second - fv.position.z);
              if (dz < best)
              {
                best = dz;
                chosen = &fv;
              }
            }
            if (chosen && m_sitpc_vertex_attach_max_dz >= 0.0 && best > m_sitpc_vertex_attach_max_dz)
            {
              chosen = nullptr;
            }
            if (chosen)
            {
              ++m_counter_sitpc_vertex_map_attached;
            }
          }
        }
        if (chosen)
        {
          tracklet.has_collision_vertex = true;
          tracklet.collision_vertex = chosen->position;
          tracklet.collision_vertex_sigma = chosen->sigma;
          tracklet.collision_vertex_source = 3;
          tracklet.collision_vertex_ntracks = chosen->ntracks;
          tracklet.collision_id = static_cast<int>(chosen->id);
          ++m_counter_sitpc_vertex_map;
          continue;
        }
      }

      const auto iter = fallback.find(tracklet.crossing);
      if (iter == fallback.end())
      {
        continue;
      }
      tracklet.has_collision_vertex = true;
      tracklet.collision_vertex = iter->second.vertex;
      tracklet.collision_vertex_sigma = iter->second.sigma;
      tracklet.collision_vertex_source = iter->second.source;
      tracklet.collision_vertex_ntracks = iter->second.ntracks;
      tracklet.collision_id = -1;  // one fallback collision per crossing
      if (iter->second.source == 1)
      {
        ++m_counter_sitpc_vertex_resolver;
      }
      else if (iter->second.source == 2)
      {
        ++m_counter_sitpc_vertex_tracks;
      }
      else
      {
        ++m_counter_sitpc_vertex_fixed;
      }
    }
  }

  const TpcV0CandidateTree::Vec3 &TpcV0CandidateTree::tracklet_primary_vertex(
      const Tracklet &tracklet, const Vec3 &fallback) const
  {
    return (m_use_sitpc_tracks && tracklet.has_collision_vertex) ? tracklet.collision_vertex
                                                                 : fallback;
  }

  TpcV0CandidateTree::Vec3 TpcV0CandidateTree::pair_primary_vertex(
      const Tracklet &track1, const Tracklet &track2, const Vec3 &fallback) const
  {
    if (!m_use_sitpc_tracks)
    {
      return fallback;
    }
    if (track1.has_collision_vertex && track2.has_collision_vertex)
    {
      // Different collisions only reach here without set_require_same_crossing:
      // use the vertex of the better-timed (Si-matched) track.
      if ((track1.crossing == track2.crossing && track1.collision_id == track2.collision_id) ||
          track1.has_si || !track2.has_si)
      {
        return track1.collision_vertex;
      }
      return track2.collision_vertex;
    }
    if (track1.has_collision_vertex)
    {
      return track1.collision_vertex;
    }
    if (track2.has_collision_vertex)
    {
      return track2.collision_vertex;
    }
    return fallback;
  }

  // ------------------------------------------------------------------------
  // displaced two-track vertex (D0)
  // ------------------------------------------------------------------------
  TpcV0CandidateTree::DisplacedPair TpcV0CandidateTree::displaced_pair(
      const Vec3 &pair_vertex,
      const Vec3 &total_momentum,
      const Vec3 &primary_vertex,
      const Vec3 &primary_pca1, const Vec3 &primary_mom1,
      const Vec3 &primary_pca2, const Vec3 &primary_mom2,
      const bool have_primary_pair) const
  {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    DisplacedPair out;
    out.lxy = out.lxy_signed = out.decay_length = out.dira = nan;
    out.pointing_dca_xy = out.pointing_dca_3d = nan;
    out.signed_dca_xy1 = out.signed_dca_xy2 = out.dca_product = nan;

    const auto signed_dca = [&primary_vertex](const Vec3 &pca, const Vec3 &mom)
    {
      const double dx = pca.x - primary_vertex.x;
      const double dy = pca.y - primary_vertex.y;
      const double cross_z = dx * mom.y - dy * mom.x;
      const double magnitude = std::hypot(dx, dy);
      return cross_z >= 0.0 ? magnitude : -magnitude;
    };
    if (have_primary_pair)
    {
      out.signed_dca_xy1 = signed_dca(primary_pca1, primary_mom1);
      out.signed_dca_xy2 = signed_dca(primary_pca2, primary_mom2);
      out.dca_product = out.signed_dca_xy1 * out.signed_dca_xy2;
    }

    if (!finite(pair_vertex) || !finite(total_momentum) || !finite(primary_vertex))
    {
      return out;
    }
    const Vec3 flight = subtract(pair_vertex, primary_vertex);
    const double ptot = norm(total_momentum);
    const double ptrans = pt(total_momentum);
    if (!(ptot > 0.0) || !(ptrans > 0.0))
    {
      return out;
    }

    out.lxy = std::hypot(flight.x, flight.y);
    out.lxy_signed = (flight.x * total_momentum.x + flight.y * total_momentum.y) / ptrans;
    out.decay_length = norm(flight);
    out.dira = out.decay_length > 0.0 ? vector_cosine(flight, total_momentum) : nan;
    out.pointing_dca_xy = std::abs(flight.x * total_momentum.y - flight.y * total_momentum.x) / ptrans;
    out.pointing_dca_3d = norm(cross(flight, total_momentum)) / ptot;
    out.valid = std::isfinite(out.lxy_signed) && std::isfinite(out.decay_length);
    return out;
  }

  bool TpcV0CandidateTree::passes_d0_displaced_selection(
      const DisplacedPair &displaced,
      const std::pair<double, double> & /*dca1*/,
      const std::pair<double, double> & /*dca2*/,
      const double pair_dca) const
  {
    const DisplacedCuts &cuts = m_d0_displaced_cuts;
    if (!cuts.enabled)
    {
      return true;
    }
    if (!displaced.valid)
    {
      return false;
    }
    if (cuts.lxy_min >= 0.0 && !(displaced.lxy_signed >= cuts.lxy_min))
    {
      return false;
    }
    if (cuts.decay_length_min >= 0.0 && !(displaced.decay_length >= cuts.decay_length_min))
    {
      return false;
    }
    if (cuts.decay_length_max >= 0.0 && !(displaced.decay_length <= cuts.decay_length_max))
    {
      return false;
    }
    if (cuts.dira_min >= -1.0 && !(displaced.dira >= cuts.dira_min))
    {
      return false;
    }
    if (cuts.pair_dca_max >= 0.0 && !(pair_dca <= cuts.pair_dca_max))
    {
      return false;
    }
    if (cuts.pointing_dca_xy_max >= 0.0 && !(displaced.pointing_dca_xy <= cuts.pointing_dca_xy_max))
    {
      return false;
    }
    if (cuts.track_dca_xy_min >= 0.0 &&
        !(std::abs(displaced.signed_dca_xy1) >= cuts.track_dca_xy_min &&
          std::abs(displaced.signed_dca_xy2) >= cuts.track_dca_xy_min))
    {
      return false;
    }
    if (cuts.dca_product_max <= 1.0 && !(displaced.dca_product <= cuts.dca_product_max))
    {
      return false;
    }
    return true;
  }
}  // namespace tpchadronreco
