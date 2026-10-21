// Tell emacs that this is a C++ source
// -*- C++ -*-.
#ifndef TPCHADRONRECO_TPCTRACKKALMANFITTER_H
#define TPCHADRONRECO_TPCTRACKKALMANFITTER_H

#include "TpcTrackFit.h"

#include <array>
#include <utility>
#include <vector>

namespace tpchadronreco
{
  // Custom Kalman / RKN track fit and propagation (implementation in
  // TpcTrackKalmanFitter.cc).
  //
  // State vector (6): position x, y, z [cm], direction phi [rad], q/pt [1/GeV],
  // tan(lambda).  Covariances are 6 x 6, stored row-major in std::array<double, 36>
  // (TpcKalmanResult::covs_*, TpcKalmanVertexConstraintResult::covariance).
  class TpcTrackKalmanFitter
  {
   public:
    enum StateIndex : int
    {
      X = 0,
      Y = 1,
      Z = 2,
      Phi = 3,
      QOverPt = 4,
      TanLambda = 5
    };
    static constexpr int StateDim = 6;

    // Fit the measured points with the given charge; points are used in the
    // order given by config.point_order (TpcTrackPointOrder::Input keeps them).
    static bool fit(const std::vector<TpcTrackPoint> &input_points,
                    int charge,
                    const TpcKalmanConfig &config,
                    TpcKalmanResult &result,
                    double mass_gev);

    static TpcTrackVec3 state_position(const std::array<double, StateDim> &state);
    static TpcTrackVec3 state_momentum(const std::array<double, StateDim> &state);
    static TpcTrackVec3 state_tangent(const std::array<double, StateDim> &state);

    // Smoothed state used as the start of every propagation of this fit.
    static std::array<double, StateDim> propagation_state(const TpcKalmanResult &fit,
                                                          const TpcTrackVec3 &reference_vertex);

    // Propagate a state by a signed transverse path ds [cm] (RKN with the field
    // map when config.magnetic_field is set, analytic uniform Bz otherwise).
    static std::array<double, StateDim> propagate_state(const std::array<double, StateDim> &state,
                                                        double ds_cm,
                                                        const TpcKalmanConfig &config,
                                                        double mass_gev);

    // Gaussian primary-vertex update of the fitted track at its PCA to the vertex.
    static bool constrain_to_vertex(const TpcKalmanResult &fit,
                                    const TpcTrackVec3 &vertex,
                                    const TpcKalmanConfig &input_config,
                                    double sigma_xy_cm,
                                    double sigma_z_cm,
                                    bool constrain_z,
                                    TpcKalmanVertexConstraintResult &result);

    // (DCA_xy, DCA_z) of the fitted track to a vertex.
    static std::pair<double, double> dca_to_vertex(const TpcKalmanResult &fit,
                                                   const TpcTrackVec3 &vertex,
                                                   const TpcKalmanConfig *input_config = nullptr);
  };
}  // namespace tpchadronreco

#endif