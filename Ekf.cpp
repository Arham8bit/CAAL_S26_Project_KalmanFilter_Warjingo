#include <iomanip>


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cfenv>
#include <stdexcept>

static const int    NUM_JOINTS      = 23;
static const int    STATE_PER_JOINT = 12;
static const int    MEAS_PER_JOINT  = 3;
static const int    FULL_STATE      = NUM_JOINTS * STATE_PER_JOINT; // 276
static const int    FULL_MEAS       = NUM_JOINTS * MEAS_PER_JOINT;  // 69
static const double DT              = 0.01;
static const double SIGMA_J         = 1.0;
static const double SIGMA_R         = 0.5;



static const double MY_PI = 3.14159265358979323846;

double manual_atan(double z) {
    double absz = z < 0 ? -z : z;
    return (MY_PI / 4.0) * z - z * (absz - 1.0) * (0.2447 + 0.0663 * absz);
}

double manual_atan2(double y, double x) {
    if (x == 0.0 && y == 0.0) return 0.0;
    if (x == 0.0) return (y > 0) ? MY_PI / 2.0 : -MY_PI / 2.0;

    double r, angle;
    if (std::abs(x) >= std::abs(y)) {
        r     = y / x;
        angle = manual_atan(r);
        if (x < 0.0) angle += (y >= 0.0) ? MY_PI : -MY_PI;
    } else {
        r     = x / y;
        angle = MY_PI / 2.0 - manual_atan(r);
        if (y < 0.0) angle = -angle;
        if (x < 0.0) angle += (y >= 0.0) ? MY_PI : -MY_PI;
    }
    return angle;
}

//  Matrix class (heap-allocated, row-major)

class Matrix {
public:
    int rows, cols;
    std::vector<double> data;

    Matrix(int r, int c, double val = 0.0)
        : rows(r), cols(c), data(r * c, val) {}

    double& at(int r, int c)       { return data[r * cols + c]; }
    double  at(int r, int c) const { return data[r * cols + c]; }

    Matrix operator+(const Matrix& o) const {
        Matrix res(rows, cols);
        for (int i = 0; i < rows * cols; ++i) res.data[i] = data[i] + o.data[i];
        return res;
    }
    Matrix operator-(const Matrix& o) const {
        Matrix res(rows, cols);
        for (int i = 0; i < rows * cols; ++i) res.data[i] = data[i] - o.data[i];
        return res;
    }
    Matrix operator*(const Matrix& o) const {
        Matrix res(rows, o.cols);
        for (int i = 0; i < rows; ++i)
            for (int k = 0; k < cols; ++k) {
                if (data[i * cols + k] == 0.0) continue;
                for (int j = 0; j < o.cols; ++j)
                    res.data[i * o.cols + j] = fma(data[i * cols + k], o.data[k * o.cols + j], res.data[i * o.cols + j]);
            }
        return res;
    }
    Matrix operator*(double s) const {
        Matrix res(rows, cols);
        for (int i = 0; i < rows * cols; ++i) res.data[i] = data[i] * s;
        return res;
    }
    Matrix transpose() const {
        Matrix res(cols, rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                res.at(j, i) = at(i, j);
        return res;
    }
    static Matrix identity(int n) {
        Matrix I(n, n);
        for (int i = 0; i < n; ++i) I.at(i, i) = 1.0;
        return I;
    }

    // LU decomposition with partial pivoting — avoids explicit inversion
    static Matrix solve(const Matrix& A, const Matrix& B) {
        int n = A.rows, m = B.cols;
        Matrix LU = A;
        std::vector<int> piv(n);
        for (int i = 0; i < n; ++i) piv[i] = i;

        for (int k = 0; k < n; ++k) {
            int maxRow = k;
            double maxVal = std::abs(LU.at(k, k));
            for (int i = k+1; i < n; ++i)
                if (std::abs(LU.at(i, k)) > maxVal) {
                    maxVal = std::abs(LU.at(i, k));
                    maxRow = i;
                }
            if (maxRow != k) {
                std::swap(piv[k], piv[maxRow]);
                for (int j = 0; j < n; ++j)
                    std::swap(LU.at(k, j), LU.at(maxRow, j));
            }
            if (std::abs(LU.at(k, k)) < 1e-14)
                throw std::runtime_error("Singular matrix in LU solve");
            for (int i = k+1; i < n; ++i) {
                LU.at(i, k) /= LU.at(k, k);
                for (int j = k+1; j < n; ++j)
                    LU.at(i, j) = fma(-LU.at(i, k), LU.at(k, j), LU.at(i, j));
            }
        }
        Matrix Bp(n, m);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                Bp.at(i, j) = B.at(piv[i], j);

        Matrix Y(n, m);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j) {
                double s = Bp.at(i, j);
                for (int k = 0; k < i; ++k) s = fma(-LU.at(i, k), Y.at(k, j), s);
                Y.at(i, j) = s;
            }
        Matrix X(n, m);
        for (int i = n-1; i >= 0; --i)
            for (int j = 0; j < m; ++j) {
                double s = Y.at(i, j);
                for (int k = i+1; k < n; ++k) s = fma(-LU.at(i, k), X.at(k, j), s);
                X.at(i, j) = s / LU.at(i, i);
            }
        return X;
    }
};


//  Build system matrices

Matrix buildFullF(double dt) {
    Matrix F(FULL_STATE, FULL_STATE);
    // 4x4 block per axis
    double fb[4][4] = {
        {1, dt, 0.5*dt*dt, (1.0/6.0)*dt*dt*dt},
        {0,  1,        dt,          0.5*dt*dt},
        {0,  0,         1,                 dt},
        {0,  0,         0,                  1}
    };
    for (int jt = 0; jt < NUM_JOINTS; ++jt)
        for (int axis = 0; axis < 3; ++axis)
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    F.at(jt*12 + axis*4+i, jt*12 + axis*4+j) = fb[i][j];
    return F;
}

Matrix buildFullQ(double dt, double sigma_j) {
    double g[4] = {(1.0/6.0)*dt*dt*dt, 0.5*dt*dt, dt, 1.0};
    Matrix Q(FULL_STATE, FULL_STATE);
    for (int jt = 0; jt < NUM_JOINTS; ++jt)
        for (int axis = 0; axis < 3; ++axis)
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    Q.at(jt*12+axis*4+i, jt*12+axis*4+j) =
                        sigma_j * sigma_j * g[i] * g[j];
    return Q;
}


//  Nonlinear h(x): Cartesian -> spherical

void h_joint(double px, double py, double pz,
             double& r, double& theta, double& phi) {
    double rho = std::sqrt(px*px + py*py);
    r     = std::sqrt(px*px + py*py + pz*pz);
    theta = manual_atan2(py, px);
    phi   = manual_atan2(pz, rho);
}

Matrix computeHx(const Matrix& x) {
    Matrix hx(FULL_MEAS, 1);
    for (int j = 0; j < NUM_JOINTS; ++j) {
        double px = x.at(j*12+0, 0);
        double py = x.at(j*12+4, 0);
        double pz = x.at(j*12+8, 0);
        double r, theta, phi;
        h_joint(px, py, pz, r, theta, phi);
        hx.at(j*3+0, 0) = r;
        hx.at(j*3+1, 0) = theta;
        hx.at(j*3+2, 0) = phi;
    }
    return hx;
}

//  Jacobian (69x276) — computed at x_pred

Matrix buildFullJacobian(const Matrix& x) {
    Matrix J(FULL_MEAS, FULL_STATE);
    for (int jt = 0; jt < NUM_JOINTS; ++jt) {
        double px   = x.at(jt*12+0, 0);
        double py   = x.at(jt*12+4, 0);
        double pz   = x.at(jt*12+8, 0);
        double r2   = px*px + py*py + pz*pz;
        double r    = std::sqrt(r2);
        double rho2 = px*px + py*py;
        double rho  = std::sqrt(rho2);

        if (r < 1e-9 || rho < 1e-9) continue;

        int row0 = jt*3;
        int col_px = jt*12+0, col_py = jt*12+4, col_pz = jt*12+8;

        // dr/d(pos)
        J.at(row0+0, col_px) =  px / r;
        J.at(row0+0, col_py) =  py / r;
        J.at(row0+0, col_pz) =  pz / r;

        // dtheta/d(pos)
        J.at(row0+1, col_px) = -py / rho2;
        J.at(row0+1, col_py) =  px / rho2;
        J.at(row0+1, col_pz) =  0.0;

        // dphi/d(pos)
        J.at(row0+2, col_px) = -(px*pz) / (r2*rho);
        J.at(row0+2, col_py) = -(py*pz) / (r2*rho);
        J.at(row0+2, col_pz) =  rho / r2;
    }
    return J;
}


//  Convert noisy Cartesian row -> spherical z

Matrix cartesianToSpherical(const std::vector<double>& row) {
    Matrix z(FULL_MEAS, 1);
    for (int j = 0; j < NUM_JOINTS; ++j) {
        double px = row[j*3+0], py = row[j*3+1], pz = row[j*3+2];
        double r, theta, phi;
        h_joint(px, py, pz, r, theta, phi);
        z.at(j*3+0, 0) = r;
        z.at(j*3+1, 0) = theta;
        z.at(j*3+2, 0) = phi;
    }
    return z;
}


//  CSV helpers

std::vector<std::vector<double>> readCSV(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::vector<std::vector<double>> data;
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ','))
            row.push_back(std::stod(tok));
        if (!row.empty()) data.push_back(row);
    }
    return data;
}

void writeCSV(const std::string& path,
              const std::vector<std::vector<double>>& rows,
              const std::vector<std::string>& headers) {
    std::ofstream f(path);
    for (int i = 0; i < (int)headers.size(); ++i)
        f << headers[i] << (i+1==(int)headers.size() ? "\n" : ",");
    for (auto& row : rows) {
        for (int i = 0; i < (int)row.size(); ++i)
            f << std::setprecision(15) << row[i] << (i+1==(int)row.size() ? "\n" : ",");
    }
}


int main(int argc, char* argv[]) {

    std::string noisyPath = "NoisyValues.csv";
    if (argc > 1) noisyPath = argv[1];

    std::cout << "[EKF] Reading dataset ...\n";
    auto noisyData = readCSV(noisyPath);
    int T = (int)noisyData.size();
    std::cout << "[EKF] Timesteps: " << T << "\n";

    std::cout << "[EKF] Building matrices ...\n";
    Matrix F = buildFullF(DT);
    Matrix Q = buildFullQ(DT, SIGMA_J);

    // Measurement noise R (69x69) diagonal
    Matrix R(FULL_MEAS, FULL_MEAS);
    for (int j = 0; j < NUM_JOINTS; ++j) {
        R.at(j*3+0, j*3+0) = SIGMA_R * SIGMA_R; // range variance
        R.at(j*3+1, j*3+1) = 0.01;              // azimuth variance (rad^2)
        R.at(j*3+2, j*3+2) = 0.01;              // elevation variance (rad^2)
    }

    // Initialise state from first noisy measurement
    Matrix x(FULL_STATE, 1, 0.0);
    for (int j = 0; j < NUM_JOINTS; ++j) {
        x.at(j*12+0, 0) = noisyData[0][j*3+0]; // px
        x.at(j*12+4, 0) = noisyData[0][j*3+1]; // py
        x.at(j*12+8, 0) = noisyData[0][j*3+2]; // pz
    }

    // Initial covariance
    Matrix P(FULL_STATE, FULL_STATE, 0.0);
    for (int i = 0; i < FULL_STATE; ++i) P.at(i,i) = 1.0;

    std::vector<std::vector<double>> outputRows;

    std::cout << "[EKF] Running filter ...\n";
    for (int t = 0; t < T; ++t) {

        // ── PREDICTION 
        Matrix x_pred = F * x;
        Matrix P_pred = F * P * F.transpose() + Q;

        // ── MEASUREMENT (convert to spherical) ─
        Matrix z = cartesianToSpherical(noisyData[t]);

        // ── JACOBIAN at predicted state 
        Matrix Jac  = buildFullJacobian(x_pred);
        Matrix JacT = Jac.transpose();

        // ── UPDATE 
        // S = J*P_pred*J^T + R  (69x69)
        Matrix S = Jac * P_pred * JacT + R;

        // K = P_pred*J^T*S^{-1}  solved via LU
        Matrix PJt = P_pred * JacT;
        Matrix K   = Matrix::solve(S, PJt.transpose()).transpose();

        // Innovation y = z - h(x_pred)
        Matrix y = z - computeHx(x_pred);

        // State update
        x = x_pred + K * y;

        // Covariance update (Joseph form for stability)
        Matrix IKJ = Matrix::identity(FULL_STATE) - K * Jac;
        P = IKJ * P_pred * IKJ.transpose() + K * R * K.transpose();

        // Store output
        std::vector<double> row(FULL_STATE);
        for (int i = 0; i < FULL_STATE; ++i) row[i] = x.at(i, 0);
        outputRows.push_back(row);

        if (t % 500 == 0)
            std::cout << "[EKF] t=" << t << "/" << T << "\n";
    }

    std::cout << "[EKF] Writing output ...\n";
    std::string jointNames[] = {
        "pelvis","L5","L3","T12","T8","neck","head",
        "shoulderRight","upperArmRight","forearmRight","handRight",
        "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
        "upperLegRight","lowerLegRight","footRight","toeRight",
        "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
    };
    std::string stateNames[] = {
        "px","vx","ax","jx","py","vy","ay","jy","pz","vz","az","jz"
    };
    std::vector<std::string> headers;
    for (int j = 0; j < NUM_JOINTS; ++j)
        for (int s = 0; s < STATE_PER_JOINT; ++s)
            headers.push_back(jointNames[j] + "_" + stateNames[s]);

    writeCSV("EKF_output.csv", outputRows, headers);
    std::cout << "[EKF] Done. Output saved to EKF_output.csv\n";
    return 0;
}