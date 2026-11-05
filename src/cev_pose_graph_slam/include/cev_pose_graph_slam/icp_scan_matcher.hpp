#pragma once

#include "cev_pose_graph_slam/sliding_window_pgo.hpp"

#include <memory>
#include <string>

namespace icp {
    class ICPDriver;
}

namespace cev_slam {

// Scan matcher built on the CEV icp library, plus an independent check of the result
// (inlier fraction + inlier RMSE) so a bad ICP convergence never becomes a graph edge.
class IcpScanMatcher {
public:
    struct Config {
        std::string method = "trimmed";
        double overlap_rate = 0.7;
        int max_iterations = 50;
        double inlier_dist = 0.15;  // m
        double min_fitness = 0.6;
        double max_rmse = 0.06;  // m
        size_t min_points = 40;
    };

    explicit IcpScanMatcher(Config cfg);
    ~IcpScanMatcher();

    std::optional<MatchResult> match(const PointList& a, const PointList& b, const Pose2& guess);

    // Adapter for SlidingWindowPGO.
    static ScanMatcher asFunction(std::shared_ptr<IcpScanMatcher> m) {
        return [m](const PointList& a, const PointList& b, const Pose2& g) { return m->match(a, b, g); };
    }

private:
    Config cfg_;
    std::unique_ptr<icp::ICPDriver> driver_;
};

// Range filter + voxel-grid downsample (keeps ICP and the fitness check cheap: both are
// brute-force nearest neighbour).
PointList filterScan(const PointList& pts, double min_range, double max_range, double voxel);

}  // namespace cev_slam
