#include <iomanip>


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cfenv>
#include <stdexcept>


static const int    NUM_JOINTS       = 23;
static const int    STATE_PER_JOINT  = 12;   // [px vx ax jx py vy ay jy pz vz az jz]
static const int    MEAS_PER_JOINT   = 3;    // [px py pz]
static const int    FULL_STATE       = NUM_JOINTS * STATE_PER_JOINT;  // 276
static const int    FULL_MEAS        = NUM_JOINTS * MEAS_PER_JOINT;   // 69
static const double DT               = 0.01; // sampling interval (seconds)
static const double SIGMA_J          = 1.0;  // process noise intensity
static const double SIGMA_R          = 0.5;  // measurement noise std dev


class Matrix {
public:
    int rows, cols;
    std::vector<double> data; // heap-allocated storage

    Matrix(int r, int c, double val = 0.0)
        : rows(r), cols(c), data(r * c, val) {}

    double& at(int r, int c)       { return data[r * cols + c]; }
    double  at(int r, int c) const { return data[r * cols + c]; }

    // ── basic operations 
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
                if (data[i * cols + k] == 0.0) continue; // skip sparse zeros
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

    // ── identity 
    static Matrix identity(int n) {
        Matrix I(n, n);
        for (int i = 0; i < n; ++i) I.at(i, i) = 1.0;
        return I;
    }

    
    static Matrix solve(const Matrix& A, const Matrix& B) {
        int n = A.rows;
        int m = B.cols;

        // LU decomposition with partial pivoting
        Matrix LU = A;
        std::vector<int> piv(n);
        for (int i = 0; i < n; ++i) piv[i] = i;

        for (int k = 0; k < n; ++k) {
            // find pivot
            int maxRow = k;
            double maxVal = std::abs(LU.at(k, k));
            for (int i = k + 1; i < n; ++i) {
                if (std::abs(LU.at(i, k)) > maxVal) {
                    maxVal = std::abs(LU.at(i, k));
                    maxRow = i;
                }
            }
            // swap rows in LU and pivot vector
            if (maxRow != k) {
                std::swap(piv[k], piv[maxRow]);
                for (int j = 0; j < n; ++j)
                    std::swap(LU.at(k, j), LU.at(maxRow, j));
            }
            if (std::abs(LU.at(k, k)) < 1e-14)
                throw std::runtime_error("Singular matrix in LU solve");

            for (int i = k + 1; i < n; ++i) {
                LU.at(i, k) /= LU.at(k, k);
                for (int j = k + 1; j < n; ++j)
                    LU.at(i, j) = fma(-LU.at(i, k), LU.at(k, j), LU.at(i, j));
            }
        }

        // apply same row permutation to B
        Matrix Bp(n, m);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                Bp.at(i, j) = B.at(piv[i], j);

        // forward substitution  L*Y = Bp
        Matrix Y(n, m);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                double s = Bp.at(i, j);
                for (int k = 0; k < i; ++k) s = fma(-LU.at(i, k), Y.at(k, j), s);
                Y.at(i, j) = s;
            }
        }
        // back substitution  U*X = Y
        Matrix X(n, m);
        for (int i = n - 1; i >= 0; --i) {
            for (int j = 0; j < m; ++j) {
                double s = Y.at(i, j);
                for (int k = i + 1; k < n; ++k) s -= LU.at(i, k) * X.at(k, j);
                X.at(i, j) = s / LU.at(i, i);
            }
        }
        return X;
    }
};

//  Build single-joint 4×4 F block

Matrix buildFblock(double dt) {
    Matrix F(4, 4);
    F.at(0,0)=1;  F.at(0,1)=dt; F.at(0,2)=0.5*dt*dt; F.at(0,3)=(1.0/6.0)*dt*dt*dt;
    F.at(1,0)=0;  F.at(1,1)=1;  F.at(1,2)=dt;         F.at(1,3)=0.5*dt*dt;
    F.at(2,0)=0;  F.at(2,1)=0;  F.at(2,2)=1;          F.at(2,3)=dt;
    F.at(3,0)=0;  F.at(3,1)=0;  F.at(3,2)=0;          F.at(3,3)=1;
    return F;
}


//  Build single-joint 12×12 F (block-diagonal)

Matrix buildFjoint(double dt) {
    Matrix F(12, 12);
    Matrix Fb = buildFblock(dt);
    for (int axis = 0; axis < 3; ++axis)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                F.at(axis*4+i, axis*4+j) = Fb.at(i,j);
    return F;
}


//  Build single-joint 4×4 Q block
//  Q_axis = sigma_j^2 * Phi(dt)*G*G^T*Phi^T(dt)
//  evaluated at tau = dt  (no integration — correct answer-key formula)

Matrix buildQblock(double dt, double sigma_j) {
    // g = [dt^3/6, dt^2/2, dt, 1]^T  (= Phi(dt)*G)
    double g0 = (1.0/6.0)*dt*dt*dt;
    double g1 = 0.5*dt*dt;
    double g2 = dt;
    double g3 = 1.0;
    double g[4] = {g0, g1, g2, g3};

    Matrix Q(4, 4);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            Q.at(i,j) = sigma_j * sigma_j * g[i] * g[j];
    return Q;
}


//  Build single-joint 12×12 Q (block-diagonal)

Matrix buildQjoint(double dt, double sigma_j) {
    Matrix Q(12, 12);
    Matrix Qb = buildQblock(dt, sigma_j);
    for (int axis = 0; axis < 3; ++axis)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                Q.at(axis*4+i, axis*4+j) = Qb.at(i,j);
    return Q;
}

//  Build single-joint 3×12 H

Matrix buildHjoint() {
    Matrix H(3, 12);
    H.at(0, 0) = 1.0;  // px
    H.at(1, 4) = 1.0;  // py
    H.at(2, 8) = 1.0;  // pz
    return H;
}


//  Build full-body (276×276) block-diagonal F, Q
//  and full-body (69×276) block-diagonal H

Matrix buildFullF(double dt) {
    Matrix F(FULL_STATE, FULL_STATE);
    Matrix Fj = buildFjoint(dt);
    for (int j = 0; j < NUM_JOINTS; ++j)
        for (int i = 0; i < STATE_PER_JOINT; ++i)
            for (int k = 0; k < STATE_PER_JOINT; ++k)
                F.at(j*STATE_PER_JOINT+i, j*STATE_PER_JOINT+k) = Fj.at(i,k);
    return F;
}

Matrix buildFullQ(double dt, double sigma_j) {
    Matrix Q(FULL_STATE, FULL_STATE);
    Matrix Qj = buildQjoint(dt, sigma_j);
    for (int j = 0; j < NUM_JOINTS; ++j)
        for (int i = 0; i < STATE_PER_JOINT; ++i)
            for (int k = 0; k < STATE_PER_JOINT; ++k)
                Q.at(j*STATE_PER_JOINT+i, j*STATE_PER_JOINT+k) = Qj.at(i,k);
    return Q;
}

Matrix buildFullH() {
    Matrix H(FULL_MEAS, FULL_STATE);
    Matrix Hj = buildHjoint();
    for (int j = 0; j < NUM_JOINTS; ++j)
        for (int i = 0; i < MEAS_PER_JOINT; ++i)
            for (int k = 0; k < STATE_PER_JOINT; ++k)
                H.at(j*MEAS_PER_JOINT+i, j*STATE_PER_JOINT+k) = Hj.at(i,k);
    return H;
}


//  CSV helpers
std::vector<std::vector<double>> readCSV(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::vector<std::vector<double>> data;
    std::string line;
    std::getline(f, line); // skip header
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

    std::cout << "[LKF] Reading dataset ...\n";
    auto noisyData = readCSV(noisyPath);
    int T = (int)noisyData.size();
    std::cout << "[LKF] Timesteps: " << T << "\n";

    // ── Build system matrices 
    std::cout << "[LKF] Building matrices ...\n";
    Matrix F = buildFullF(DT);
    Matrix Q = buildFullQ(DT, SIGMA_J);
    Matrix H = buildFullH();
    Matrix Ht = H.transpose();

    // Measurement noise covariance R (69×69) diagonal
    Matrix R(FULL_MEAS, FULL_MEAS);
    for (int i = 0; i < FULL_MEAS; ++i)
        R.at(i,i) = SIGMA_R * SIGMA_R;

    // ── Initialise state and covariance
    // State x (276×1): initialise positions from first measurement,
    // velocities / accelerations / jerks set to zero.
    Matrix x(FULL_STATE, 1, 0.0);
    for (int j = 0; j < NUM_JOINTS; ++j) {
        x.at(j*STATE_PER_JOINT + 0, 0) = noisyData[0][j*3 + 0]; // px
        x.at(j*STATE_PER_JOINT + 4, 0) = noisyData[0][j*3 + 1]; // py
        x.at(j*STATE_PER_JOINT + 8, 0) = noisyData[0][j*3 + 2]; // pz
    }

    // P (276×276): large initial uncertainty on positions, small elsewhere
    Matrix P(FULL_STATE, FULL_STATE, 0.0);
    for (int i = 0; i < FULL_STATE; ++i) P.at(i,i) = 1.0;

    // ── Output container
    std::vector<std::vector<double>> outputRows;

    std::cout << "[LKF] Running filter ...\n";
    for (int t = 0; t < T; ++t) {

        // ── PREDICTION 
        // x_pred = F * x
        Matrix x_pred = F * x;
        // P_pred = F * P * F^T + Q
        Matrix P_pred = F * P * F.transpose() + Q;

        // ── MEASUREMENT 
        Matrix z(FULL_MEAS, 1);
        for (int j = 0; j < NUM_JOINTS; ++j) {
            z.at(j*3+0, 0) = noisyData[t][j*3+0];
            z.at(j*3+1, 0) = noisyData[t][j*3+1];
            z.at(j*3+2, 0) = noisyData[t][j*3+2];
        }

        // ── UPDATE 
        // Innovation covariance S = H*P_pred*H^T + R  (69×69)
        Matrix S = H * P_pred * Ht + R;

        // Kalman gain K = P_pred * H^T * S^{-1}
        // Computed as: solve S^T * K^T = (P_pred * H^T)^T
        // i.e. K = (S \ (H * P_pred^T))^T  — avoids explicit 276×276 inversion
        Matrix PHt = P_pred * Ht;          // 276×69
        // We need K = PHt * S^{-1}
        // Equivalent: K^T = S^{-T} * PHt^T  =>  S * K^T = PHt^T
        Matrix K = Matrix::solve(S, PHt.transpose()).transpose(); // 276×69

        // Innovation  y = z - H*x_pred
        Matrix y = z - H * x_pred;

        // State update
        x = x_pred + K * y;

        // Covariance update (Joseph form for numerical stability)
        Matrix IKH = Matrix::identity(FULL_STATE) - K * H;
        P = IKH * P_pred * IKH.transpose() + K * R * K.transpose();

        // ── Store full state row
        std::vector<double> row(FULL_STATE);
        for (int i = 0; i < FULL_STATE; ++i) row[i] = x.at(i, 0);
        outputRows.push_back(row);

        if (t % 500 == 0)
            std::cout << "[LKF] t=" << t << "/" << T << "\n";
    }

    // ── Write output CSV
    std::cout << "[LKF] Writing output ...\n";
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

    writeCSV("LKF_output.csv", outputRows, headers);
    std::cout << "[LKF] Done. Output saved to LKF_output.csv\n";
    return 0;
}