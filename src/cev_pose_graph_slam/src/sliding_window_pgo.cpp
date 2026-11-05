#include "cev_pose_graph_slam/sliding_window_pgo.hpp"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cev_slam {

// --- Math helpers ---

double wrapAngle(double a) {
    a = std::fmod(a + M_PI, 2.0 * M_PI);
    if (a < 0) a += 2.0 * M_PI;
    return a - M_PI;
}

static inline Eigen::Matrix2d R(double th) {
    double c = std::cos(th), s = std::sin(th);
    Eigen::Matrix2d M;
    M << c, -s, s, c;
    return M;
}

Pose2 compose(const Pose2& a, const Pose2& b) {
    double c = std::cos(a.th), s = std::sin(a.th);
    return {a.x + c * b.x - s * b.y, a.y + s * b.x + c * b.y, wrapAngle(a.th + b.th)};
}

Pose2 inverse(const Pose2& a) {
    double c = std::cos(a.th), s = std::sin(a.th);
    return {-c * a.x - s * a.y, s * a.x - c * a.y, wrapAngle(-a.th)};
}

Pose2 between(const Pose2& a, const Pose2& b) { return compose(inverse(a), b); }

Point transformPoint(const Pose2& T, const Point& p) { return R(T.th) * p + Point(T.x, T.y); }

Eigen::Vector3d toVec(const Pose2& p) { return {p.x, p.y, p.th}; }

Eigen::Vector3d computeError(const Pose2& xi, const Pose2& xj, const Eigen::Vector3d& z) {
    Eigen::Vector2d dt(xj.x - xi.x, xj.y - xi.y);
    Eigen::Vector3d e;
    e.head<2>() = R(xi.th).transpose() * dt - z.head<2>();
    e[2] = wrapAngle((xj.th - xi.th) - z[2]);
    return e;
}

void computeJacobians(const Pose2& xi, const Pose2& xj, Eigen::Matrix3d& Ji, Eigen::Matrix3d& Jj) {
    Ji.setZero();
    Jj.setZero();
    Eigen::Matrix2d RiT = R(xi.th).transpose();
    Ji.block<2, 2>(0, 0) = -RiT;
    Jj.block<2, 2>(0, 0) = RiT;
    double c = std::cos(xi.th), s = std::sin(xi.th);
    Eigen::Matrix2d dRiT;  // d(R^T)/dth
    dRiT << -s, c, -c, -s;
    Eigen::Vector2d d = dRiT * Eigen::Vector2d(xj.x - xi.x, xj.y - xi.y);
    Ji(0, 2) = d[0];
    Ji(1, 2) = d[1];
    Ji(2, 2) = -1.0;
    Jj(2, 2) = 1.0;
}

static Eigen::Matrix3d infoFromSigma(const Eigen::Vector3d& sigma) {
    return sigma.cwiseInverse().cwiseAbs2().asDiagonal();
}

// Huber kernel on sqrt(chi2). Returns the robust cost and sets the IRLS weight.
static double robustCost(double chi2, bool robust, double delta, double& w) {
    if (!robust || chi2 <= delta * delta) {
        w = 1.0;
        return chi2;
    }
    double s = std::sqrt(chi2);
    w = delta / s;
    return 2.0 * delta * s - delta * delta;
}

// --- SlidingWindowPGO ---

SlidingWindowPGO::SlidingWindowPGO(Params params, ScanMatcher matcher)
    : p_(params), matcher_(std::move(matcher)) {
    if (p_.window_size < 2) throw std::invalid_argument("window_size must be >= 2");
}

int SlidingWindowPGO::localIndex(int id) const {
    if (window_.empty()) return -1;
    int idx = id - window_.front().id;  // ids in the window are contiguous
    if (idx < 0 || idx >= static_cast<int>(window_.size())) return -1;
    return idx;
}

bool SlidingWindowPGO::addOdometry(const Pose2& odom, double stamp, const PointList& scan_body) {
    if (window_.empty()) {
        // First keyframe defines the map frame: map == odom at start.
        window_.push_back({next_id_++, stamp, odom, odom, scan_body});
        if (p_.mode == WindowMode::Marginalize) {
            double info = 1.0 / (p_.prior_sigma * p_.prior_sigma);
            prior_ = {{window_.front().id}, toVec(odom), Eigen::Matrix3d::Identity() * info,
                Eigen::Vector3d::Zero()};
        }
        stats_.keyframes++;
        updateMapToOdom();
        return true;
    }

    const Keyframe& last = window_.back();
    // Keyframe test and measurement both use RAW odometry. The optimized estimate lives in the
    // map frame and must never be mixed with odom-frame poses.
    Pose2 delta = between(last.odom, odom);
    if (std::hypot(delta.x, delta.y) < p_.kf_trans && std::abs(delta.th) < p_.kf_rot) return false;

    int last_id = last.id;
    // Initial guess in the map frame: last optimized pose + odometry increment.
    window_.push_back({next_id_++, stamp, odom, compose(last.est, delta), scan_body});
    edges_.push_back({last_id, window_.back().id, toVec(delta), infoFromSigma(p_.odom_sigma),
        EdgeType::Odom});
    stats_.keyframes++;
    stats_.odom_edges++;

    if (matcher_ && !scan_body.empty()) {
        addLocalConstraints(window_.back());
        tryLoopClosure(window_.back());
    }

    optimize();
    slideWindow();
    updateMapToOdom();
    return true;
}

bool SlidingWindowPGO::addBetweenEdge(int i, int j, const Eigen::Vector3d& z,
    const Eigen::Matrix3d& info, EdgeType type) {
    if (!inWindow(i) || !inWindow(j) || i == j) return false;
    edges_.push_back({i, j, z, info, type});
    return true;
}

double SlidingWindowPGO::linearize(Eigen::MatrixXd& H, Eigen::VectorXd& b) const {
    const int D = 3 * static_cast<int>(window_.size());
    H.setZero(D, D);
    b.setZero(D);
    double chi2 = 0.0;
    double w;

    for (const auto& e: edges_) {
        int li = localIndex(e.i), lj = localIndex(e.j);
        const Pose2& xi = window_[li].est;
        const Pose2& xj = window_[lj].est;
        Eigen::Vector3d err = computeError(xi, xj, e.z);
        chi2 += robustCost(err.dot(e.info * err), e.type != EdgeType::Odom, p_.huber_delta, w);
        Eigen::Matrix3d Ji, Jj;
        computeJacobians(xi, xj, Ji, Jj);
        Eigen::Matrix3d Om = w * e.info;
        H.block<3, 3>(3 * li, 3 * li) += Ji.transpose() * Om * Ji;
        H.block<3, 3>(3 * li, 3 * lj) += Ji.transpose() * Om * Jj;
        H.block<3, 3>(3 * lj, 3 * li) += Jj.transpose() * Om * Ji;
        H.block<3, 3>(3 * lj, 3 * lj) += Jj.transpose() * Om * Jj;
        b.segment<3>(3 * li) += Ji.transpose() * Om * err;
        b.segment<3>(3 * lj) += Jj.transpose() * Om * err;
    }

    for (const auto& e: map_edges_) {
        int lj = localIndex(e.j);
        const Pose2& xj = window_[lj].est;
        Eigen::Vector3d err = computeError(e.anchor, xj, e.z);
        chi2 += robustCost(err.dot(e.info * err), true, p_.huber_delta, w);
        Eigen::Matrix3d Ja, Jj;
        computeJacobians(e.anchor, xj, Ja, Jj);
        Eigen::Matrix3d Om = w * e.info;
        H.block<3, 3>(3 * lj, 3 * lj) += Jj.transpose() * Om * Jj;
        b.segment<3>(3 * lj) += Jj.transpose() * Om * err;
    }

    if (p_.mode == WindowMode::Marginalize && !prior_.ids.empty()) {
        const int K = static_cast<int>(prior_.ids.size());
        Eigen::VectorXd dx(3 * K);
        for (int k = 0; k < K; ++k) {
            dx.segment<3>(3 * k) = toVec(window_[localIndex(prior_.ids[k])].est) - prior_.x0.segment<3>(3 * k);
            dx[3 * k + 2] = wrapAngle(dx[3 * k + 2]);
        }
        Eigen::VectorXd grad = prior_.g + prior_.H * dx;
        chi2 += 2.0 * prior_.g.dot(dx) + dx.dot(prior_.H * dx);
        for (int r = 0; r < K; ++r) {
            int lr = localIndex(prior_.ids[r]);
            b.segment<3>(3 * lr) += grad.segment<3>(3 * r);
            for (int c = 0; c < K; ++c) {
                int lc = localIndex(prior_.ids[c]);
                H.block<3, 3>(3 * lr, 3 * lc) += prior_.H.block<3, 3>(3 * r, 3 * c);
            }
        }
    }
    return chi2;
}

double SlidingWindowPGO::evaluate() const {
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    return linearize(H, b);
}

void SlidingWindowPGO::optimize() {
    if (window_.size() < 2) return;

    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    double chi2 = linearize(H, b);
    double lambda = 1e-4;
    int it = 0;

    for (; it < p_.max_iterations; ++it) {
        Eigen::MatrixXd A = H;
        A.diagonal().array() += lambda * (H.diagonal().array() + 1e-6);
        Eigen::VectorXd rhs = -b;
        if (p_.mode == WindowMode::Truncate) {
            // Old behaviour: hard-pin the oldest pose in the window.
            A.topRows<3>().setZero();
            A.leftCols<3>().setZero();
            A.topLeftCorner<3, 3>().setIdentity();
            rhs.head<3>().setZero();
        }
        Eigen::VectorXd dx = A.ldlt().solve(rhs);
        if (!dx.allFinite()) break;

        std::vector<Pose2> backup;
        backup.reserve(window_.size());
        for (auto& kf: window_) backup.push_back(kf.est);
        for (size_t i = 0; i < window_.size(); ++i) {
            Pose2& x = window_[i].est;
            x.x += dx[3 * i];
            x.y += dx[3 * i + 1];
            x.th = wrapAngle(x.th + dx[3 * i + 2]);
        }

        double new_chi2 = evaluate();
        if (new_chi2 <= chi2) {
            lambda = std::max(lambda * 0.1, 1e-9);
            chi2 = linearize(H, b);
            if (dx.norm() < p_.convergence_eps) {
                ++it;
                break;
            }
        } else {
            for (size_t i = 0; i < window_.size(); ++i) window_[i].est = backup[i];
            lambda *= 10.0;
            if (lambda > 1e8) break;
        }
    }
    stats_.last_chi2 = chi2;
    stats_.last_iterations = it;
}

void SlidingWindowPGO::addLocalConstraints(const Keyframe& kf) {
    std::vector<std::pair<double, const Keyframe*>> candidates;
    for (const auto& other: window_) {
        if (kf.id - other.id < p_.local_min_gap || other.scan.empty()) continue;
        double d = std::hypot(other.est.x - kf.est.x, other.est.y - kf.est.y);
        if (d < p_.local_radius) candidates.emplace_back(d, &other);
    }
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    if (static_cast<int>(candidates.size()) > p_.max_local_matches) candidates.resize(p_.max_local_matches);

    for (const auto& [d, other]: candidates) {
        Pose2 guess = between(other->est, kf.est);
        auto res = matcher_(other->scan, kf.scan, guess);
        Pose2 corr = res ? between(guess, res->rel) : Pose2{};
        if (!res || std::hypot(corr.x, corr.y) > p_.max_match_correction_trans
            || std::abs(corr.th) > p_.max_match_correction_rot) {
            stats_.rejected_matches++;
            continue;
        }
        edges_.push_back({other->id, kf.id, toVec(res->rel), infoFromSigma(p_.scan_sigma), EdgeType::Scan});
        stats_.scan_edges++;
    }
}

void SlidingWindowPGO::tryLoopClosure(const Keyframe& kf) {
    std::vector<std::pair<double, const Keyframe*>> candidates;
    for (const auto& old: archive_) {
        if (kf.id - old.id < p_.loop_min_gap || old.scan.empty()) continue;
        double d = std::hypot(old.est.x - kf.est.x, old.est.y - kf.est.y);
        if (d < p_.loop_radius) candidates.emplace_back(d, &old);
    }
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    if (candidates.size() > 3) candidates.resize(3);

    for (const auto& [d, old]: candidates) {
        Pose2 guess = between(old->est, kf.est);
        auto res = matcher_(old->scan, kf.scan, guess);
        Pose2 corr = res ? between(guess, res->rel) : Pose2{};
        if (!res || std::hypot(corr.x, corr.y) > p_.max_loop_correction_trans
            || std::abs(corr.th) > p_.max_loop_correction_rot) {
            stats_.rejected_matches++;
            continue;
        }
        map_edges_.push_back({old->id, old->est, kf.id, toVec(res->rel), infoFromSigma(p_.loop_sigma)});
        stats_.loop_edges++;
        return;  // one verified loop closure per keyframe is enough
    }
}

void SlidingWindowPGO::slideWindow() {
    while (static_cast<int>(window_.size()) > p_.window_size) {
        if (p_.mode == WindowMode::Marginalize)
            marginalizeOldest();
        else
            truncateOldest();
    }
}

void SlidingWindowPGO::truncateOldest() {
    int m = window_.front().id;
    edges_.erase(std::remove_if(edges_.begin(), edges_.end(),
                     [m](const BetweenEdge& e) { return e.i == m || e.j == m; }),
        edges_.end());
    map_edges_.erase(std::remove_if(map_edges_.begin(), map_edges_.end(),
                         [m](const MapEdge& e) { return e.j == m; }),
        map_edges_.end());
    archive_.push_back(std::move(window_.front()));
    window_.pop_front();
}

void SlidingWindowPGO::marginalizeOldest() {
    const int m = window_.front().id;

    // Markov blanket of m: everything sharing a factor with it. The current prior is always
    // folded in so that there is only ever one prior factor.
    std::vector<int> ids{m};
    auto addId = [&](int id) {
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
    };
    for (const auto& e: edges_) {
        if (e.i == m) addId(e.j);
        if (e.j == m) addId(e.i);
    }
    for (int id: prior_.ids) addId(id);

    auto pos = [&](int id) {
        return static_cast<int>(std::find(ids.begin(), ids.end(), id) - ids.begin());
    };
    auto est = [&](int id) -> const Pose2& { return window_[localIndex(id)].est; };

    const int K = static_cast<int>(ids.size());
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3 * K, 3 * K);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * K);
    double w;

    for (const auto& e: edges_) {
        if (e.i != m && e.j != m) continue;
        int pi = pos(e.i), pj = pos(e.j);
        Eigen::Vector3d err = computeError(est(e.i), est(e.j), e.z);
        robustCost(err.dot(e.info * err), e.type != EdgeType::Odom, p_.huber_delta, w);
        Eigen::Matrix3d Ji, Jj;
        computeJacobians(est(e.i), est(e.j), Ji, Jj);
        Eigen::Matrix3d Om = w * e.info;
        H.block<3, 3>(3 * pi, 3 * pi) += Ji.transpose() * Om * Ji;
        H.block<3, 3>(3 * pi, 3 * pj) += Ji.transpose() * Om * Jj;
        H.block<3, 3>(3 * pj, 3 * pi) += Jj.transpose() * Om * Ji;
        H.block<3, 3>(3 * pj, 3 * pj) += Jj.transpose() * Om * Jj;
        b.segment<3>(3 * pi) += Ji.transpose() * Om * err;
        b.segment<3>(3 * pj) += Jj.transpose() * Om * err;
    }
    for (const auto& e: map_edges_) {
        if (e.j != m) continue;
        Eigen::Vector3d err = computeError(e.anchor, est(m), e.z);
        robustCost(err.dot(e.info * err), true, p_.huber_delta, w);
        Eigen::Matrix3d Ja, Jj;
        computeJacobians(e.anchor, est(m), Ja, Jj);
        H.block<3, 3>(0, 0) += Jj.transpose() * (w * e.info) * Jj;
        b.segment<3>(0) += Jj.transpose() * (w * e.info) * err;
    }
    if (!prior_.ids.empty()) {
        const int P = static_cast<int>(prior_.ids.size());
        Eigen::VectorXd dx(3 * P);
        for (int k = 0; k < P; ++k) {
            dx.segment<3>(3 * k) = toVec(est(prior_.ids[k])) - prior_.x0.segment<3>(3 * k);
            dx[3 * k + 2] = wrapAngle(dx[3 * k + 2]);
        }
        Eigen::VectorXd grad = prior_.g + prior_.H * dx;
        for (int r = 0; r < P; ++r) {
            int pr = pos(prior_.ids[r]);
            b.segment<3>(3 * pr) += grad.segment<3>(3 * r);
            for (int c = 0; c < P; ++c)
                H.block<3, 3>(3 * pr, 3 * pos(prior_.ids[c])) += prior_.H.block<3, 3>(3 * r, 3 * c);
        }
    }

    // Schur complement: eliminate m, keep the information it carried about its neighbours.
    LinearPrior next;
    if (K > 1) {
        const int n = 3 * (K - 1);
        Eigen::Matrix3d Hmm = H.topLeftCorner<3, 3>();
        Eigen::MatrixXd Hmr = H.topRightCorner(3, n);
        Eigen::LDLT<Eigen::Matrix3d> Hmm_ldlt(Hmm);
        next.ids.assign(ids.begin() + 1, ids.end());
        next.H = H.bottomRightCorner(n, n) - Hmr.transpose() * Hmm_ldlt.solve(Hmr);
        next.H = 0.5 * (next.H + next.H.transpose());
        next.g = b.tail(n) - Hmr.transpose() * Hmm_ldlt.solve(Eigen::Vector3d(b.head<3>()));
        next.x0.resize(n);
        for (int k = 0; k < K - 1; ++k) next.x0.segment<3>(3 * k) = toVec(est(next.ids[k]));
    }
    prior_ = std::move(next);

    edges_.erase(std::remove_if(edges_.begin(), edges_.end(),
                     [m](const BetweenEdge& e) { return e.i == m || e.j == m; }),
        edges_.end());
    map_edges_.erase(std::remove_if(map_edges_.begin(), map_edges_.end(),
                         [m](const MapEdge& e) { return e.j == m; }),
        map_edges_.end());
    archive_.push_back(std::move(window_.front()));
    window_.pop_front();
}

void SlidingWindowPGO::updateMapToOdom() {
    // T_map_odom = T_map_base * T_odom_base^-1, evaluated at the newest keyframe.
    const Keyframe& kf = window_.back();
    map_to_odom_ = compose(kf.est, inverse(kf.odom));
}

}  // namespace cev_slam
