// Offline test for the sliding-window backend (no ROS needed).
//
// 1. Jacobians match finite differences.
// 2. map->odom: the new formula is exact, the old "subtract the poses" formula is not.
// 3. Marginalization keeps the window solution equal to the full batch solution;
//    truncation (the old approach) does not.
// 4. End-to-end: a car drives 2 laps of a corridor with drifting odometry and a simulated
//    2D lidar; compare raw odometry, the old backend, and the new backend against ground truth.
//
// Usage: pgo_sim_test [output.svg]

#include "cev_pose_graph_slam/icp_scan_matcher.hpp"
#include "cev_pose_graph_slam/sliding_window_pgo.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace cev_slam;

static int failures = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            std::printf("  FAIL: " __VA_ARGS__);          \
            std::printf("\n");                            \
            failures++;                                   \
        }                                                 \
    } while (0)

// ---------------------------------------------------------------------------------------------
// 1. Jacobians
static void testJacobians() {
    std::printf("[1] Jacobians vs finite differences\n");
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> U(-3, 3);
    double worst = 0;
    for (int t = 0; t < 100; ++t) {
        Pose2 xi{U(rng), U(rng), U(rng)}, xj{U(rng), U(rng), U(rng)};
        Eigen::Vector3d z(U(rng), U(rng), U(rng) / 3);
        Eigen::Matrix3d Ji, Jj, Ni, Nj;
        computeJacobians(xi, xj, Ji, Jj);
        const double h = 1e-6;
        for (int k = 0; k < 3; ++k) {
            Pose2 a = xi, b = xj;
            (&a.x)[k] += h;
            (&b.x)[k] += h;
            Eigen::Vector3d di = computeError(a, xj, z) - computeError(xi, xj, z);
            Eigen::Vector3d dj = computeError(xi, b, z) - computeError(xi, xj, z);
            di[2] = wrapAngle(di[2]);
            dj[2] = wrapAngle(dj[2]);
            Ni.col(k) = di / h;
            Nj.col(k) = dj / h;
        }
        worst = std::max({worst, (Ji - Ni).cwiseAbs().maxCoeff(), (Jj - Nj).cwiseAbs().maxCoeff()});
    }
    std::printf("  max |analytic - numeric| = %.2e\n", worst);
    CHECK(worst < 1e-4, "Jacobian mismatch");
}

// ---------------------------------------------------------------------------------------------
// 2. map->odom
static Pose2 oldCorrection(const Pose2& opt, const Pose2& odom) {
    return {opt.x - odom.x, opt.y - odom.y, wrapAngle(opt.th - odom.th)};
}

static void testMapToOdom() {
    std::printf("[2] map->odom transform\n");
    // Robot believes (odom) it is at (5, 2) facing 90deg; the optimizer says it is really at
    // (5.3, 1.8) facing 80deg. TF then reports map pose = T_map_odom * odom.
    Pose2 odom{5, 2, M_PI / 2}, opt{5.3, 1.8, 80 * M_PI / 180};
    Pose2 good = compose(compose(opt, inverse(odom)), odom);
    Pose2 bad = compose(oldCorrection(opt, odom), odom);
    std::printf("  wanted map pose     (%.3f, %.3f, %.1f deg)\n", opt.x, opt.y, opt.th * 180 / M_PI);
    std::printf("  new formula gives   (%.3f, %.3f, %.1f deg)\n", good.x, good.y, good.th * 180 / M_PI);
    std::printf("  old formula gives   (%.3f, %.3f, %.1f deg)  <- off by %.2f m\n", bad.x, bad.y,
        bad.th * 180 / M_PI, std::hypot(bad.x - opt.x, bad.y - opt.y));
    CHECK(std::hypot(good.x - opt.x, good.y - opt.y) < 1e-9, "new map->odom wrong");
}

// ---------------------------------------------------------------------------------------------
// 3. Marginalization vs truncation vs batch
static void testMarginalization() {
    std::printf("[3] Sliding window vs full batch (window = 6, extra edges i-3 -> i)\n");
    auto run = [&](int window, WindowMode mode, std::vector<Pose2>& newest) {
        std::mt19937 r(42);
        std::normal_distribution<double> n(0, 1);
        Params p;
        p.window_size = window;
        p.mode = mode;
        p.kf_trans = 0.01;
        p.max_iterations = 30;
        p.convergence_eps = 1e-10;
        SlidingWindowPGO pgo(p);
        std::vector<Pose2> gt{{0, 0, 0}};
        Pose2 odom{0, 0, 0};
        pgo.addOdometry(odom, 0);
        for (int k = 1; k < 60; ++k) {
            Pose2 step{0.3, 0.0, 0.08 * std::sin(k * 0.2)};
            gt.push_back(compose(gt.back(), step));
            Pose2 noisy{step.x + 0.02 * n(r), 0.02 * n(r), step.th + 0.01 * n(r)};
            odom = compose(odom, noisy);
            pgo.addOdometry(odom, k);
            if (k >= 3) {
                Pose2 z = between(gt[k - 3], gt[k]);
                Eigen::Vector3d zv(z.x + 0.005 * n(r), z.y + 0.005 * n(r), z.th + 0.002 * n(r));
                Eigen::Matrix3d info = Eigen::Vector3d(4e4, 4e4, 2.5e5).asDiagonal();
                pgo.addBetweenEdge(k - 3, k, zv, info);
                pgo.optimize();
            }
            newest.push_back(pgo.window().back().est);
        }
    };

    std::vector<Pose2> batch, marg, trunc;
    run(1000, WindowMode::Marginalize, batch);
    run(6, WindowMode::Marginalize, marg);
    run(6, WindowMode::Truncate, trunc);
    double dm = 0, dt = 0;
    for (size_t k = 0; k < batch.size(); ++k) {
        dm = std::max(dm, std::hypot(marg[k].x - batch[k].x, marg[k].y - batch[k].y));
        dt = std::max(dt, std::hypot(trunc[k].x - batch[k].x, trunc[k].y - batch[k].y));
    }
    std::printf("  max distance of newest pose from batch solution:\n");
    std::printf("    marginalization: %.6f m\n    truncation:      %.6f m\n", dm, dt);
    CHECK(dm < 0.01, "marginalized window diverged from batch");
    CHECK(dm < dt, "marginalization should beat truncation");
}

// ---------------------------------------------------------------------------------------------
// 4. End-to-end simulation
struct Seg {
    Point a, b;
};

static std::vector<Seg> makeWorld() {
    std::vector<Seg> w;
    auto box = [&](double x0, double y0, double x1, double y1) {
        w.push_back({{x0, y0}, {x1, y0}});
        w.push_back({{x1, y0}, {x1, y1}});
        w.push_back({{x1, y1}, {x0, y1}});
        w.push_back({{x0, y1}, {x0, y0}});
    };
    box(-12, -8, 12, 8);  // outer walls
    box(-7, -3, 7, 3);    // inner block -> 5 m wide corridor loop
    // Pillars / clutter so the corridor is not featureless.
    double pillars[][2] = {{-4, -7}, {3, -6.8}, {10.5, -4}, {11, 2}, {6, 7}, {-1, 6.7},
        {-8, 7.2}, {-11, 1}, {-10.6, -5}, {0.5, -3.6}, {8, -1}, {-2, 3.6}, {-8, -0.5}};
    for (auto& p: pillars) box(p[0] - 0.25, p[1] - 0.25, p[0] + 0.25, p[1] + 0.25);
    return w;
}

static PointList simulateScan(const std::vector<Seg>& world, const Pose2& pose, std::mt19937& rng) {
    std::normal_distribution<double> noise(0, 0.01);
    PointList pts;
    const int beams = 360;
    for (int k = 0; k < beams; ++k) {
        double a = -M_PI + 2 * M_PI * k / beams;
        Point o(pose.x, pose.y), d(std::cos(pose.th + a), std::sin(pose.th + a));
        double best = 12.0;
        for (const auto& s: world) {
            Point e = s.b - s.a;
            double den = d.x() * e.y() - d.y() * e.x();
            if (std::abs(den) < 1e-12) continue;
            Point ao = s.a - o;
            double t = (ao.x() * e.y() - ao.y() * e.x()) / den;
            double u = (ao.x() * d.y() - ao.y() * d.x()) / den;
            if (t > 0 && u >= 0 && u <= 1) best = std::min(best, t);
        }
        if (best < 12.0) {
            double r = best + noise(rng);
            pts.emplace_back(r * std::cos(a), r * std::sin(a));
        }
    }
    return pts;
}

// Rounded-rectangle loop through the corridor centre line.
static Pose2 trackAt(double s) {
    const double a = 9.5, b = 5.5, r = 1.5;
    const double sx = 2 * (a - r), sy = 2 * (b - r), q = M_PI * r / 2;
    const double L = 2 * sx + 2 * sy + 4 * q;
    s = std::fmod(s, L);
    struct Corner {
        double cx, cy, a0;
    };
    // straight segments: start point, heading, length; each followed by an arc
    double sxs[4][3] = {{-a + r, -b, 0}, {a, -b + r, M_PI / 2}, {a - r, b, M_PI}, {-a, b - r, -M_PI / 2}};
    double lens[4] = {sx, sy, sx, sy};
    Corner corners[4] = {{a - r, -b + r, -M_PI / 2}, {a - r, b - r, 0}, {-a + r, b - r, M_PI / 2},
        {-a + r, -b + r, M_PI}};
    for (int i = 0; i < 4; ++i) {
        if (s < lens[i]) {
            double h = sxs[i][2];
            return {sxs[i][0] + s * std::cos(h), sxs[i][1] + s * std::sin(h), h};
        }
        s -= lens[i];
        if (s < q) {
            double ang = corners[i].a0 + s / r;
            return {corners[i].cx + r * std::cos(ang), corners[i].cy + r * std::sin(ang),
                wrapAngle(ang + M_PI / 2)};
        }
        s -= q;
    }
    return trackAt(0);
}

struct Run {
    std::string name;
    std::vector<Pose2> traj;  // online map-frame pose at every odometry step
    double ate = 0;
    SlidingWindowPGO::Stats stats;
    double ms_early = 0, ms_late = 0, ms_max = 0;
};

static double ate(const std::vector<Pose2>& est, const std::vector<Pose2>& gt) {
    double s = 0;
    for (size_t i = 0; i < gt.size(); ++i) s += std::pow(est[i].x - gt[i].x, 2) + std::pow(est[i].y - gt[i].y, 2);
    return std::sqrt(s / gt.size());
}

static void writeSvg(const std::string& path, const std::vector<Seg>& world, const std::vector<Pose2>& gt,
    const std::vector<Run>& runs) {
    std::ofstream f(path);
    auto X = [](double x) { return (x + 13) * 30; };
    auto Y = [](double y) { return (9 - y) * 30; };
    f << "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 780 600' style='background:#fff'>\n";
    for (const auto& s: world)
        f << "<line x1='" << X(s.a.x()) << "' y1='" << Y(s.a.y()) << "' x2='" << X(s.b.x()) << "' y2='"
          << Y(s.b.y()) << "' stroke='#999' stroke-width='2'/>\n";
    auto poly = [&](const std::vector<Pose2>& t, const char* color, double width) {
        f << "<polyline fill='none' stroke='" << color << "' stroke-width='" << width << "' points='";
        for (size_t i = 0; i < t.size(); i += 4) f << X(t[i].x) << "," << Y(t[i].y) << " ";
        f << "'/>\n";
    };
    const char* colors[] = {"#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#2ca02c"};
    poly(gt, "#000", 3);
    for (size_t i = 0; i < runs.size(); ++i) poly(runs[i].traj, colors[i % 5], 1.5);
    int y = 20;
    f << "<text x='10' y='" << y << "' font-family='sans-serif' font-size='13'>black: ground truth</text>\n";
    for (size_t i = 0; i < runs.size(); ++i) {
        y += 16;
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s: ATE %.2f m", runs[i].name.c_str(), runs[i].ate);
        f << "<text x='10' y='" << y << "' font-family='sans-serif' font-size='13' fill='" << colors[i % 5]
          << "'>" << buf << "</text>\n";
    }
    f << "</svg>\n";
}

static void testEndToEnd(const char* svg_path) {
    std::printf("[4] End-to-end: 2 laps (~115 m) of a 5 m corridor, drifting odometry, simulated lidar\n");
    const auto world = makeWorld();
    const double lap = 2 * 16 + 2 * 8 + 2 * M_PI * 1.5;  // matches trackAt()
    const double step = 0.05, total = 2 * lap;
    const int n = static_cast<int>(total / step);

    // Ground truth, odometry (scale error + heading bias + noise) and scans.
    std::mt19937 rng(3);
    std::normal_distribution<double> N(0, 1);
    std::vector<Pose2> gt, odom;
    std::vector<PointList> scans;
    gt.push_back(trackAt(0));
    odom.push_back(gt[0]);  // map and odom frames coincide at start
    for (int i = 1; i < n; ++i) {
        gt.push_back(trackAt(i * step));
        Pose2 d = between(gt[i - 1], gt[i]);
        double dist = std::hypot(d.x, d.y);
        Pose2 noisy{d.x * 1.02 + 0.002 * N(rng), d.y + 0.002 * N(rng),
            d.th + dist * 0.25 * M_PI / 180 + 0.001 * N(rng)};  // +0.25 deg/m heading bias
        odom.push_back(compose(odom.back(), noisy));
    }
    std::mt19937 scan_rng(5);
    for (int i = 0; i < n; ++i) scans.push_back(filterScan(simulateScan(world, gt[i], scan_rng), 0.15, 10.0, 0.1));

    std::vector<Run> runs;

    {  // raw odometry
        Run r{"raw odometry", odom};
        r.ate = ate(r.traj, gt);
        runs.push_back(r);
    }
    {  // old backend: odometry-only chain, so keyframe poses == odometry; old TF formula
        Run r{"old backend"};
        Pose2 kf = odom[0];
        for (int i = 0; i < n; ++i) {
            Pose2 d = between(kf, odom[i]);
            if (std::hypot(d.x, d.y) > 0.3 || std::abs(d.th) > 0.26) kf = odom[i];
            r.traj.push_back(compose(oldCorrection(kf, odom[i]), odom[i]));
        }
        r.ate = ate(r.traj, gt);
        runs.push_back(r);
    }

    auto runPgo = [&](const std::string& name, WindowMode mode, bool loops) {
        Params p;
        p.mode = mode;
        if (!loops) p.loop_min_gap = 1 << 30;
        IcpScanMatcher::Config mc;
        auto matcher = IcpScanMatcher::asFunction(std::make_shared<IcpScanMatcher>(mc));
        SlidingWindowPGO pgo(p, matcher);
        Run r{name};
        std::vector<double> times;
        for (int i = 0; i < n; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            bool kf = pgo.addOdometry(odom[i], i * 0.05, scans[i]);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (kf) times.push_back(ms);
            r.traj.push_back(pgo.correctedPose(odom[i]));
        }
        r.ate = ate(r.traj, gt);
        r.stats = pgo.stats();
        size_t k = std::min<size_t>(50, times.size() / 2);
        for (size_t i = 0; i < k; ++i) {
            r.ms_early += times[i] / k;
            r.ms_late += times[times.size() - 1 - i] / k;
        }
        for (double t: times) r.ms_max = std::max(r.ms_max, t);
        runs.push_back(r);
    };
    runPgo("new: window, no loop closure", WindowMode::Marginalize, false);
    runPgo("new: truncation + loop closure", WindowMode::Truncate, true);
    runPgo("new: marginalization + loop closure", WindowMode::Marginalize, true);

    std::printf("  %-38s %8s %6s %6s %6s %6s %10s %10s\n", "run", "ATE[m]", "KFs", "scan", "loop", "rej",
        "ms/KF 1st", "ms/KF last");
    for (const auto& r: runs) {
        std::printf("  %-38s %8.3f", r.name.c_str(), r.ate);
        if (r.stats.keyframes)
            std::printf(" %6d %6d %6d %6d %10.1f %10.1f", r.stats.keyframes, r.stats.scan_edges,
                r.stats.loop_edges, r.stats.rejected_matches, r.ms_early, r.ms_late);
        std::printf("\n");
    }
    const Run& best = runs.back();
    CHECK(best.ate < 0.5 * runs[0].ate, "SLAM should at least halve the odometry error");
    CHECK(best.stats.loop_edges > 0, "expected loop closures on lap 2");
    CHECK(best.ms_late < 3 * best.ms_early + 5, "per-keyframe cost should stay bounded");
    if (svg_path) {
        writeSvg(svg_path, world, gt, runs);
        std::printf("  wrote %s\n", svg_path);
    }
}

int main(int argc, char** argv) {
    testJacobians();
    testMapToOdom();
    testMarginalization();
    testEndToEnd(argc > 1 ? argv[1] : nullptr);
    std::printf(failures ? "\n%d CHECK(S) FAILED\n" : "\nALL CHECKS PASSED\n", failures);
    return failures ? 1 : 0;
}
