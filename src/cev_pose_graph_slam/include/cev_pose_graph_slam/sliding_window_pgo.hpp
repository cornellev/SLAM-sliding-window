#pragma once

// Sliding-window 2D pose graph optimization (ROS-free so it can be unit tested).
//
// - Keyframes live in a fixed-size window. When the window is full, the oldest keyframe is
//   *marginalized* with the Schur complement: its information is folded into a dense linear
//   prior on the keyframes it was connected to, instead of being thrown away.
// - Keyframes that leave the window are archived with their final estimate and scan, so a new
//   keyframe can still close a loop against them (a unary "map" edge to a fixed pose).
// - Measurements z are always "pose of j expressed in frame of i": z = x_i^-1 * x_j.

#include <Eigen/Core>

#include <deque>
#include <functional>
#include <optional>
#include <vector>

namespace cev_slam {

using Point = Eigen::Vector2d;
using PointList = std::vector<Point>;

double wrapAngle(double a);

struct Pose2 {
    double x = 0, y = 0, th = 0;
};

Pose2 compose(const Pose2& a, const Pose2& b);  // a * b
Pose2 inverse(const Pose2& a);
Pose2 between(const Pose2& a, const Pose2& b);  // a^-1 * b: pose of b in frame of a
Point transformPoint(const Pose2& T, const Point& p);
Eigen::Vector3d toVec(const Pose2& p);

// Error e = between(xi, xj) - z (angle wrapped) and its Jacobians wrt xi and xj.
Eigen::Vector3d computeError(const Pose2& xi, const Pose2& xj, const Eigen::Vector3d& z);
void computeJacobians(const Pose2& xi, const Pose2& xj, Eigen::Matrix3d& Ji, Eigen::Matrix3d& Jj);

struct MatchResult {
    Pose2 rel;       // pose of scan b's frame expressed in scan a's frame
    double fitness;  // fraction of b points with a neighbour in a after alignment
    double rmse;     // RMS distance of those inliers [m]
};

// Aligns scan b to scan a, starting from `guess` (same convention as MatchResult::rel).
// Returns nullopt when the alignment is not trustworthy.
using ScanMatcher =
    std::function<std::optional<MatchResult>(const PointList& a, const PointList& b, const Pose2& guess)>;

enum class WindowMode {
    Marginalize,  // Schur-complement marginalization (correct)
    Truncate,     // drop oldest pose + hard-pin the new oldest (old behaviour, kept for comparison)
};

struct Params {
    int window_size = 20;
    WindowMode mode = WindowMode::Marginalize;

    // Keyframe selection (measured on raw odometry)
    double kf_trans = 0.3;  // m
    double kf_rot = 0.26;   // rad

    // Measurement noise (1 sigma: x, y, theta)
    Eigen::Vector3d odom_sigma{0.045, 0.045, 0.018};
    Eigen::Vector3d scan_sigma{0.03, 0.03, 0.01};
    Eigen::Vector3d loop_sigma{0.05, 0.05, 0.02};
    double prior_sigma = 1e-3;  // first keyframe anchors the map frame

    double huber_delta = 1.5;  // robust kernel threshold on sqrt(chi2) for scan/loop edges

    // Local scan-matching constraints to non-adjacent keyframes inside the window
    double local_radius = 3.0;
    int local_min_gap = 2;
    int max_local_matches = 3;

    // Loop closure against archived (marginalized) keyframes
    double loop_radius = 3.0;
    int loop_min_gap = 30;  // in keyframes, so we do not "close a loop" with the recent past

    // Reject scan matches that move the pose further than this from the prior guess
    double max_match_correction_trans = 0.5;
    double max_match_correction_rot = 0.35;
    double max_loop_correction_trans = 2.0;
    double max_loop_correction_rot = 0.6;

    int max_iterations = 15;
    double convergence_eps = 1e-6;
};

struct Keyframe {
    int id = 0;
    double stamp = 0;
    Pose2 odom;  // raw odometry pose (odom frame) when the keyframe was taken
    Pose2 est;   // optimized pose (map frame)
    PointList scan;  // body frame, already filtered/downsampled
};

enum class EdgeType { Odom, Scan, Loop };

struct BetweenEdge {
    int i, j;  // keyframe ids, both in the window
    Eigen::Vector3d z;
    Eigen::Matrix3d info;
    EdgeType type;
};

// Constraint from a fixed, archived keyframe pose to a keyframe in the window.
struct MapEdge {
    int archived_id;
    Pose2 anchor;
    int j;
    Eigen::Vector3d z;
    Eigen::Matrix3d info;
};

// Linearized prior left behind by marginalization:
// cost(x) = 2 g^T dx + dx^T H dx, dx = x - x0 (angles wrapped).
struct LinearPrior {
    std::vector<int> ids;
    Eigen::VectorXd x0;
    Eigen::MatrixXd H;
    Eigen::VectorXd g;
};

class SlidingWindowPGO {
public:
    struct Stats {
        int keyframes = 0;
        int odom_edges = 0;
        int scan_edges = 0;
        int loop_edges = 0;
        int rejected_matches = 0;
        double last_chi2 = 0;
        int last_iterations = 0;
    };

    explicit SlidingWindowPGO(Params params, ScanMatcher matcher = nullptr);

    // Feed the latest odometry pose. Inserts and optimizes a keyframe if the robot moved far
    // enough since the last one. Returns true if a keyframe was inserted.
    bool addOdometry(const Pose2& odom, double stamp, const PointList& scan_body = {});

    // Extra relative constraint between two keyframes currently in the window.
    bool addBetweenEdge(int i, int j, const Eigen::Vector3d& z, const Eigen::Matrix3d& info,
        EdgeType type = EdgeType::Scan);

    void optimize();

    // T_map_odom: map pose = mapToOdom() * odom pose.
    const Pose2& mapToOdom() const { return map_to_odom_; }
    Pose2 correctedPose(const Pose2& odom) const { return compose(map_to_odom_, odom); }

    const std::deque<Keyframe>& window() const { return window_; }
    const std::vector<Keyframe>& archive() const { return archive_; }
    const Stats& stats() const { return stats_; }

private:
    int localIndex(int id) const;  // -1 if not in window
    bool inWindow(int id) const { return localIndex(id) >= 0; }

    // Gauss-Newton system over the window. Returns total (robust) chi2.
    double linearize(Eigen::MatrixXd& H, Eigen::VectorXd& b) const;
    double evaluate() const;

    void addLocalConstraints(const Keyframe& kf);
    void tryLoopClosure(const Keyframe& kf);
    void slideWindow();
    void marginalizeOldest();
    void truncateOldest();
    void updateMapToOdom();

    Params p_;
    ScanMatcher matcher_;

    std::deque<Keyframe> window_;
    std::vector<Keyframe> archive_;
    std::vector<BetweenEdge> edges_;
    std::vector<MapEdge> map_edges_;
    LinearPrior prior_;

    Pose2 map_to_odom_;
    int next_id_ = 0;
    Stats stats_;
};

}  // namespace cev_slam
