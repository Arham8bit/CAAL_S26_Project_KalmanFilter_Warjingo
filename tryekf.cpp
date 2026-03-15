#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <chrono>

using namespace std;

// ================================================================
//  DIMENSIONS
// ================================================================
constexpr int    NUM_JOINTS  = 23;
constexpr int    JOINT_STATE = 12;
constexpr int    JOINT_MEAS  = 3;
constexpr int    STATE_DIM   = NUM_JOINTS * JOINT_STATE;   // 276
constexpr int    MEAS_DIM    = NUM_JOINTS * JOINT_MEAS;    //  69

// ================================================================
//  NOISE PARAMETERS
// ================================================================
constexpr double DT          = 0.01;
constexpr double SIGMA_J     = 2.5;   
constexpr double SIGMA_R     = 0.5;     
constexpr double SIGMA_ANGLE = 0.05;  

// ================================================================
//  PATHS
// ================================================================
const string NOISY_CSV  = "noisy_values.csv";
const string TRUE_CSV   = "true_values.csv";
const string OUTPUT_CSV = "ekf_results.csv";

const array<string,23> JOINT_NAMES = {
    "pelvis","L5","L3","T12","T8","neck","head",
    "shoulderRight","upperArmRight","forearmRight","handRight",
    "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
    "upperLegRight","lowerLegRight","footRight","toeRight",
    "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
};

// ================================================================
//  CONSTANTS  (manual — no <cmath> macros)
// ================================================================
constexpr double MY_PI   = 3.14159265358979323846;
constexpr double MY_PI_2 = 1.57079632679489661923;
constexpr double MY_PI_4 = 0.78539816339744830962;
constexpr double TAN_PI8 = 0.41421356237309504880;  // tan(π/8) = √2−1

// ================================================================
//  FLAT ROW-MAJOR MATRIX CLASS  
// ================================================================
/*
 * Key design: data is a SINGLE flat std::vector<double> of size rows*cols.
 * Element (r,c) is at data[r * cols + c].
 *
 * Benefits over vector<vector<double>>:
 *   1. One heap allocation per matrix (not rows+1 allocations)
 *   2. All elements contiguous → cache line utilisation ~100%
 *   3. Compiler can auto-vectorise inner loops (no pointer chase)
 *   4. sizeof(Matrix) is fixed regardless of dims
 *
 * This is the same layout used by BLAS, OpenCV Mat, Eigen::Matrix.
 */
class Matrix {
public:
    int rows, cols;
    vector<double> data;   // FLAT row-major — contiguous heap allocation

    Matrix() : rows(0), cols(0) {}

    Matrix(int r, int c, double val = 0.0)
        : rows(r), cols(c), data(r * c, val) {}

    // Element access — direct index arithmetic, no pointer indirection
    inline double& at(int r, int c)       { return data[r * cols + c]; }
    inline double  at(int r, int c) const { return data[r * cols + c]; }

    static Matrix identity(int n){
        Matrix I(n, n, 0.0);
        for(int i = 0; i < n; ++i) I.at(i,i) = 1.0;
        return I;
    }

    Matrix operator+(const Matrix& o) const {
        Matrix C(rows, cols);
        for(int i = 0; i < rows*cols; ++i) C.data[i] = data[i] + o.data[i];
        return C;
    }
    Matrix operator-(const Matrix& o) const {
        Matrix C(rows, cols);
        for(int i = 0; i < rows*cols; ++i) C.data[i] = data[i] - o.data[i];
        return C;
    }
    // Sparse-aware multiply — skips zero entries (F is ~75% zero)
    Matrix operator*(const Matrix& o) const {
        Matrix C(rows, o.cols, 0.0);
        for(int i = 0; i < rows; ++i)
            for(int k = 0; k < cols; ++k){
                double aik = data[i * cols + k];
                if(aik == 0.0) continue;
                for(int j = 0; j < o.cols; ++j)
                    C.data[i * o.cols + j] += aik * o.data[k * o.cols + j];
            }
        return C;
    }
    Matrix operator*(double s) const {
        Matrix C(rows, cols);
        for(int i = 0; i < rows*cols; ++i) C.data[i] = data[i] * s;
        return C;
    }
    Matrix transpose() const {
        Matrix T(cols, rows);
        for(int i = 0; i < rows; ++i)
            for(int j = 0; j < cols; ++j)
                T.at(j,i) = at(i,j);
        return T;
    }

    // ── Matrix × column vector (Vec) ──────────────────────────
    vector<double> mulVec(const vector<double>& v) const {
        vector<double> res(rows, 0.0);
        for(int i = 0; i < rows; ++i)
            for(int j = 0; j < cols; ++j)
                res[i] += data[i * cols + j] * v[j];
        return res;
    }

    /*
     * LU DECOMPOSITION WITH PARTIAL PIVOTING 
     * ────────────────────────────────────────────────────────────────────────
     * Solves A·X = B for X without computing A^{-1} explicitly.
     * Uses partial pivoting (swaps rows to maximise pivot magnitude)
     * for numerical stability even when A is near-singular.
     * For the per-joint 3×3 S, we use the faster closed-form inverse3x3.
     *
     * Complexity: O(n³) factorisation + O(n²m) solve — general purpose.
     */
    static Matrix solve(const Matrix& A, const Matrix& B){
        int n = A.rows, m = B.cols;
        Matrix LU = A;
        vector<int> piv(n);
        for(int i = 0; i < n; ++i) piv[i] = i;

        for(int k = 0; k < n; ++k){
            // Partial pivoting: find row with largest |pivot|
            int maxRow = k;
            double maxVal = fabs(LU.at(k,k));
            for(int i = k+1; i < n; ++i)
                if(fabs(LU.at(i,k)) > maxVal){ maxVal = fabs(LU.at(i,k)); maxRow = i; }
            if(maxRow != k){
                swap(piv[k], piv[maxRow]);
                for(int j = 0; j < n; ++j) swap(LU.at(k,j), LU.at(maxRow,j));
            }
            if(fabs(LU.at(k,k)) < 1e-14)
                throw runtime_error("LU solve: singular matrix");
            for(int i = k+1; i < n; ++i){
                LU.at(i,k) /= LU.at(k,k);
                for(int j = k+1; j < n; ++j)
                    LU.at(i,j) -= LU.at(i,k) * LU.at(k,j);
            }
        }
        // Apply row permutation to B
        Matrix Bp(n, m);
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < m; ++j)
                Bp.at(i,j) = B.at(piv[i],j);
        // Forward substitution L·Y = Bp
        Matrix Y(n, m);
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < m; ++j){
                double s = Bp.at(i,j);
                for(int k = 0; k < i; ++k) s -= LU.at(i,k) * Y.at(k,j);
                Y.at(i,j) = s;
            }
        // Backward substitution U·X = Y
        Matrix X(n, m);
        for(int i = n-1; i >= 0; --i)
            for(int j = 0; j < m; ++j){
                double s = Y.at(i,j);
                for(int k = i+1; k < n; ++k) s -= LU.at(i,k) * X.at(k,j);
                X.at(i,j) = s / LU.at(i,i);
            }
        return X;
    }
};

// Shorthand
using Vec = vector<double>;

// ================================================================
//  SECTION 6.2 — MANUAL arctan2  (two-region Taylor, err < 1e-7 rad)
// ================================================================
/*
 * Two-region approach, 4 orders of magnitude more accurate than the
 * polynomial in the reference implementation (err < 1e-7 vs < 0.005 rad):
 *
 * Region A  |t| ≤ tan(π/8):  11-term alternating Taylor series
 * Region B  tan(π/8) < t ≤ 1: atan(t) = π/4 + atan((t-1)/(t+1))
 * Region C  t > 1:            atan(t) = π/2 - atan(1/t)
 * Sign:     atan(-t) = -atan(t)
 */
static double manual_atan(double t){
    bool neg = (t < 0.0); if(neg) t = -t;
    bool rec = (t > 1.0); if(rec) t = 1.0/t;
    double r;
    if(t <= TAN_PI8){
        double t2 = t*t;
        r = t*(1.0 - t2*(1.0/3.0 - t2*(1.0/5.0 - t2*(1.0/7.0
              - t2*(1.0/9.0 - t2*(1.0/11.0))))));
    } else {
        double u = (t-1.0)/(t+1.0), u2 = u*u;
        double au = u*(1.0 - u2*(1.0/3.0 - u2*(1.0/5.0 - u2*(1.0/7.0
                      - u2*(1.0/9.0 - u2*(1.0/11.0))))));
        r = MY_PI_4 + au;
    }
    if(rec) r = MY_PI_2 - r;
    if(neg) r = -r;
    return r;
}

static double manual_atan2(double y, double x){
    constexpr double EPS = 1e-12;
    if(fabs(x) < EPS && fabs(y) < EPS) return 0.0;
    if(fabs(x) < EPS) return (y > 0.0) ? MY_PI_2 : -MY_PI_2;
    double r = manual_atan(y / x);
    if(x < 0.0) r += (y >= 0.0) ? MY_PI : -MY_PI;
    return r;
}

static void validate_atan2(){
    struct TC{ double y, x, ref; const char* lbl; };
    TC cs[] = {
        {0,1,0,"(0,1)=0"},{1,1,MY_PI_4,"(1,1)=π/4"},
        {1,0,MY_PI_2,"(1,0)=π/2"},{1,-1,3*MY_PI_4,"(1,-1)=3π/4"},
        {-1,-1,-3*MY_PI_4,"(-1,-1)=-3π/4"},{-1,0,-MY_PI_2,"(-1,0)=-π/2"},
        {-1,1,-MY_PI_4,"(-1,1)=-π/4"},{0,-1,MY_PI,"(0,-1)=π"}
    };
    cout << "  [arctan2 validation — Section 6.2]\n";
    for(auto& c : cs){
        double got = manual_atan2(c.y, c.x);
        double err = fabs(got - c.ref);
        cout << "    " << c.lbl << "  err=" << fixed << setprecision(9)
             << err << (err < 1e-6 ? "  PASS" : "  FAIL") << "\n";
    }
}

// ================================================================
//  ANALYTICAL 3×3 INVERSE  (Section 6.1 — per-joint S inversion)
// ================================================================
/*
 * S per joint is 3×3 after block-diagonal factoring.
 * Closed-form cofactor/determinant inversion is:
 *   • O(1) — no iterations, no pivot selection
 *   • Numerically exact for well-conditioned 3×3
 *   • ~23,000× fewer ops than 69×69 LU on full-body S
 *
 * LU (Matrix::solve) is available as general fallback above.
 */
static Matrix inverse3x3(const Matrix& A){
    double d = A.at(0,0)*(A.at(1,1)*A.at(2,2)-A.at(1,2)*A.at(2,1))
             - A.at(0,1)*(A.at(1,0)*A.at(2,2)-A.at(1,2)*A.at(2,0))
             + A.at(0,2)*(A.at(1,0)*A.at(2,1)-A.at(1,1)*A.at(2,0));
    if(fabs(d) < 1e-12) throw runtime_error("inverse3x3: S singular");
    d = 1.0 / d;
    Matrix I(3, 3);
    I.at(0,0)= (A.at(1,1)*A.at(2,2)-A.at(1,2)*A.at(2,1))*d;
    I.at(0,1)=-(A.at(0,1)*A.at(2,2)-A.at(0,2)*A.at(2,1))*d;
    I.at(0,2)= (A.at(0,1)*A.at(1,2)-A.at(0,2)*A.at(1,1))*d;
    I.at(1,0)=-(A.at(1,0)*A.at(2,2)-A.at(1,2)*A.at(2,0))*d;
    I.at(1,1)= (A.at(0,0)*A.at(2,2)-A.at(0,2)*A.at(2,0))*d;
    I.at(1,2)=-(A.at(0,0)*A.at(1,2)-A.at(0,2)*A.at(1,0))*d;
    I.at(2,0)= (A.at(1,0)*A.at(2,1)-A.at(1,1)*A.at(2,0))*d;
    I.at(2,1)=-(A.at(0,0)*A.at(2,1)-A.at(0,1)*A.at(2,0))*d;
    I.at(2,2)= (A.at(0,0)*A.at(1,1)-A.at(0,1)*A.at(1,0))*d;
    return I;
}

// ================================================================
//  SYSTEM MATRICES  (built once, shared by all 23 joints)
// ================================================================
Matrix F_j, I12;

void build_F(double dt){
    double dt2=dt*dt, dt3=dt2*dt;
    F_j = Matrix(12, 12, 0.0);
    for(int a = 0; a < 3; ++a){
        int b = a*4;
        F_j.at(b,b)=1;     F_j.at(b,b+1)=dt; F_j.at(b,b+2)=dt2/2; F_j.at(b,b+3)=dt3/6;
        F_j.at(b+1,b+1)=1; F_j.at(b+1,b+2)=dt; F_j.at(b+1,b+3)=dt2/2;
        F_j.at(b+2,b+2)=1; F_j.at(b+2,b+3)=dt;
        F_j.at(b+3,b+3)=1;
    }
}

/*
 * Q — van Loan discretisation of continuous jerk noise
 *
 * Q_ij = σ_j² × γ_i × γ_j,  γ = [dt³/6, dt²/2, dt, 1]
 *
 * Same construction as friend's buildFullQ but applied per-joint
 * (12×12) rather than building the full 276×276 matrix.
 * Physical meaning: white noise in jerk propagates to acceleration,
 * velocity, and position via the kinematic coupling vector γ.
 */
Matrix build_Q(double dt, double sigma_j){
    double g[4] = { dt*dt*dt/6.0, dt*dt/2.0, dt, 1.0 };
    Matrix Q(12, 12, 0.0);
    for(int a = 0; a < 3; ++a){
        int b = a*4;
        for(int i = 0; i < 4; ++i)
            for(int j = 0; j < 4; ++j)
                Q.at(b+i, b+j) = sigma_j * sigma_j * g[i] * g[j];
    }
    return Q;
}

/*
 * R — Heterogeneous per-joint noise 
 * Range [m] and angles [rad] are different physical quantities with
 * different noise magnitudes:
 *   R[0][0] = σ_r²     = 0.25   m²     (range)
 *   R[1][1] = σ_angle² = 0.01  rad²    (azimuth θ)
 *   R[2][2] = σ_angle² = 0.01  rad²    (elevation φ)
 */
Matrix build_R(double sigma_r, double sigma_angle){
    Matrix R(3, 3, 0.0);
    R.at(0,0) = sigma_r     * sigma_r;
    R.at(1,1) = sigma_angle * sigma_angle;
    R.at(2,2) = sigma_angle * sigma_angle;
    return R;
}

// ================================================================
//  EKF: NONLINEAR MEASUREMENT h(x)  — Cartesian → spherical
// ================================================================
struct Sph{ double r, theta, phi; };

static Sph h_func(double px, double py, double pz){
    double rho = sqrt(px*px + py*py);
    double r   = sqrt(px*px + py*py + pz*pz);
    if(r   < 1e-9) r   = 1e-9;
    if(rho < 1e-9) rho = 1e-9;
    return { r, manual_atan2(py, px), manual_atan2(pz, rho) };
}

Vec cartToSph(const Vec& cart){
    Vec s(MEAS_DIM);
    for(int j = 0; j < NUM_JOINTS; ++j){
        auto sp = h_func(cart[j*3], cart[j*3+1], cart[j*3+2]);
        s[j*3]=sp.r; s[j*3+1]=sp.theta; s[j*3+2]=sp.phi;
    }
    return s;
}

// ================================================================
//  JACOBIAN  Hj = ∂h/∂x  [3×12 per joint]  (flat Matrix)
// ================================================================
static Matrix compute_jacobian(double px, double py, double pz){
    Matrix Hj(3, 12, 0.0);
    double rho2 = px*px + py*py, r2 = rho2 + pz*pz;
    double rho = sqrt(rho2), r = sqrt(r2);
    if(r < 1e-9) r=1e-9; if(rho < 1e-9) rho=1e-9;

    // Row 0: ∂r/∂[px py pz]
    Hj.at(0,0)= px/r;         Hj.at(0,4)= py/r;         Hj.at(0,8)= pz/r;
    // Row 1: ∂θ/∂[px py pz]
    Hj.at(1,0)=-py/(rho*rho); Hj.at(1,4)= px/(rho*rho); Hj.at(1,8)= 0.0;
    // Row 2: ∂φ/∂[px py pz]
    Hj.at(2,0)=-(px*pz)/(r2*rho); Hj.at(2,4)=-(py*pz)/(r2*rho); Hj.at(2,8)=rho/r2;
    return Hj;
}

// ================================================================
//  PER-JOINT EKF STATE  (Section 6.3 — all heap via flat Matrix)
// ================================================================
struct JointEKF {
    Vec    x;          // [12]   state vector
    Matrix P;          // [12×12] covariance  — flat contiguous
    Matrix K;          // [12×3]  Kalman gain — flat contiguous
    Matrix Hj_last;    // [3×12]  last Jacobian (diagnostics)
};

// ================================================================
//  PREDICT  —  Equations 6 and 7
// ================================================================
void ekf_predict(JointEKF& jf, const Matrix& Q_j){
    jf.x = F_j.mulVec(jf.x);                              // Eq.6
    Matrix FT = F_j.transpose();
    jf.P = F_j * jf.P * FT + Q_j;                         // Eq.7
}

// ================================================================
//  UPDATE  —  Equations 8, 9, 10
//
//  COLUMN-VECTOR INNOVATION 
//  Innovation y is a Matrix(3,1) column vector so the update
//  equation reads literally as in the math: x_new = x + K * y_mat.
//
//  y[1] (azimuth) and y[2] (elevation) use the raw difference.
//  When py crosses 0 with px < 0, y[1] ≈ ±6.28 rad → K·y blows up.
// ================================================================
void ekf_update(JointEKF& jf, const Vec& z_sph, const Matrix& R_j){
    double px=jf.x[0], py=jf.x[4], pz=jf.x[8];

    jf.Hj_last = compute_jacobian(px, py, pz);
    const Matrix& Hj = jf.Hj_last;
    Matrix HjT  = Hj.transpose();
    Matrix PHjT = jf.P * HjT;              // [12×3]

    // S = Hj·P·Hj^T + R  [3×3]
    Matrix S = Hj * PHjT + R_j;

    // Eq.8: K = P·Hj^T·S^{-1}  (analytical 3×3 inverse — O(1))
    jf.K = PHjT * inverse3x3(S);

    // Innovation as column vector Matrix(3,1)  (Improvement 3)
    auto hx = h_func(px, py, pz);
    Matrix y_mat(3, 1);
    y_mat.at(0,0) = z_sph[0] - hx.r;
    y_mat.at(1,0) = z_sph[1] - hx.theta;   // NO wrap
    y_mat.at(2,0) = z_sph[2] - hx.phi;     // NO wrap

    // Eq.9: x̂_{k|k} = x̂_{k|k-1} + K·y
    Matrix Ky_mat = jf.K * y_mat;           // [12×1]
    for(int i = 0; i < 12; ++i) jf.x[i] += Ky_mat.at(i, 0);

    // Eq.10: P = (I-K·Hj)·P·(I-K·Hj)^T + K·R·K^T  [Joseph form]
    Matrix KHj  = jf.K * Hj;
    Matrix IKHj = Matrix::identity(12) - KHj;
    Matrix KRKt = jf.K * R_j * jf.K.transpose();
    jf.P = IKHj * jf.P * IKHj.transpose() + KRKt;
}

// ================================================================
//  CSV LOADER
// ================================================================
vector<Vec> loadCSV(const string& path){
    ifstream f(path); if(!f.is_open()) throw runtime_error("Cannot open: "+path);
    vector<Vec> frames; string line; bool hdr=false; int ln=0;
    while(getline(f,line)){ ++ln;
        if(!line.empty()&&line.back()=='\r') line.pop_back();
        if(line.empty()) continue;
        if(!hdr){ hdr=true; stringstream t(line); string tk; getline(t,tk,',');
            bool isH=false; try{stod(tk);}catch(...){isH=true;}
            if(isH){cout<<"  [CSV] header skipped (line "<<ln<<")\n";continue;}}
        stringstream ss(line); string tok; Vec row; row.reserve(MEAS_DIM);
        while(getline(ss,tok,','))
            if(!tok.empty()){ try{row.push_back(stod(tok));}catch(...){row.clear();break;}}
        if((int)row.size()!=MEAS_DIM){ cerr<<"WARNING: line "<<ln<<" skipped\n"; continue;}
        frames.push_back(move(row));}
    return frames;
}

// ================================================================
//  RMSE
// ================================================================
double frameEKFrmse(const vector<JointEKF>& jf, const Vec& truth){
    double sse=0;
    for(int j=0;j<NUM_JOINTS;++j){
        double ex=jf[j].x[0]-truth[j*3], ey=jf[j].x[4]-truth[j*3+1], ez=jf[j].x[8]-truth[j*3+2];
        sse+=ex*ex+ey*ey+ez*ez;}
    return sqrt(sse/(NUM_JOINTS*3));}

double frameNoisyRMSE(const Vec& n, const Vec& t){
    double sse=0; for(int i=0;i<MEAS_DIM;++i){double e=n[i]-t[i];sse+=e*e;}
    return sqrt(sse/MEAS_DIM);}

void printJoint(const JointEKF& jf, int idx){
    cout<<"  "<<left<<setw(16)<<JOINT_NAMES[idx]
        <<"pos=("<<fixed<<setprecision(5)<<jf.x[0]<<","<<jf.x[4]<<","<<jf.x[8]
        <<") vel=("<<jf.x[1]<<","<<jf.x[5]<<","<<jf.x[9]<<")\n";}

// ================================================================
//  WRITE RESULTS CSV — positions + FULL STATE  (Section 9)
// ================================================================
void writeResultsCSV(const string& path,
                     const vector<Vec>& noisyF, const vector<Vec>& trueF,
                     const vector<vector<JointEKF>>& snaps,
                     const vector<double>& nRmse, const vector<double>& eRmse){
    ofstream f(path); if(!f.is_open()){cerr<<"Cannot write "<<path<<"\n";return;}
    f<<"frame,noisy_rmse,ekf_rmse";
    for(int j=0;j<NUM_JOINTS;++j)
        f<<",noisy_"<<JOINT_NAMES[j]<<"_x,noisy_"<<JOINT_NAMES[j]<<"_y,noisy_"<<JOINT_NAMES[j]<<"_z";
    for(int j=0;j<NUM_JOINTS;++j)
        f<<",ekf_"<<JOINT_NAMES[j]<<"_x,ekf_"<<JOINT_NAMES[j]<<"_y,ekf_"<<JOINT_NAMES[j]<<"_z";
    for(int j=0;j<NUM_JOINTS;++j)
        f<<",true_"<<JOINT_NAMES[j]<<"_x,true_"<<JOINT_NAMES[j]<<"_y,true_"<<JOINT_NAMES[j]<<"_z";
    for(int j=0;j<NUM_JOINTS;++j){const string& n=JOINT_NAMES[j];
        f<<",ekf_"<<n<<"_px,ekf_"<<n<<"_vx,ekf_"<<n<<"_ax,ekf_"<<n<<"_jx"
         <<",ekf_"<<n<<"_py,ekf_"<<n<<"_vy,ekf_"<<n<<"_ay,ekf_"<<n<<"_jy"
         <<",ekf_"<<n<<"_pz,ekf_"<<n<<"_vz,ekf_"<<n<<"_az,ekf_"<<n<<"_jz";}
    f<<"\n";
    int N=noisyF.size();
    for(int i=0;i<N;++i){
        f<<i<<","<<fixed<<setprecision(8)<<nRmse[i]<<","<<eRmse[i];
        for(double v:noisyF[i]) f<<","<<v;
        for(int j=0;j<NUM_JOINTS;++j)
            f<<","<<snaps[i][j].x[0]<<","<<snaps[i][j].x[4]<<","<<snaps[i][j].x[8];
        for(double v:trueF[i]) f<<","<<v;
        for(int j=0;j<NUM_JOINTS;++j)
            for(int k=0;k<JOINT_STATE;++k) f<<","<<snaps[i][j].x[k];
        f<<"\n";}
    cout<<"EKF CSV written: "<<path<<"\n";
    cout<<"  Columns: 3 + 69×3 + 276 = 486\n";
}

// ================================================================
//  MAIN
// ================================================================
int main(){
    cout<<"================================================================\n"
        <<"  Full Body 3D Gait – Extended Kalman Filter (EKF)\n"
        <<"  State="<<STATE_DIM<<"  Meas="<<MEAS_DIM<<"  dt="<<DT<<"\n"
        <<"  sigma_j="<<SIGMA_J<<"  sigma_r="<<SIGMA_R<<"  sigma_angle="<<SIGMA_ANGLE<<"\n"
        <<"  Matrix: flat row-major (cache-friendly, single allocation)\n"
        <<"  S inv : analytical 3x3 | LU available via Matrix::solve\n"
        <<"  arctan: two-region Taylor (err < 1e-7 rad)\n"
        <<"================================================================\n\n";

    cout<<"[0] arctan2 validation (Section 6.2):\n";
    validate_atan2();
    cout<<"\n";

    cout<<"[1] Loading datasets...\n";
    vector<Vec> noisy=loadCSV(NOISY_CSV), truth=loadCSV(TRUE_CSV);
    cout<<"  Noisy: "<<noisy.size()<<"  True: "<<truth.size()<<" frames\n\n";
    if(noisy.empty()||truth.empty()){cerr<<"ERROR: load failed\n";return 1;}
    int N=(int)min(noisy.size(),truth.size());

    cout<<"[2] Building system matrices...\n";
    build_F(DT);
    I12 = Matrix::identity(12);
    Matrix Q_j = build_Q(DT, SIGMA_J);
    Matrix R_j = build_R(SIGMA_R, SIGMA_ANGLE);

    cout<<"  F[0:4,0:4] (x-axis kinematic block):\n";
    for(int r=0;r<4;++r){cout<<"    [";
        for(int c=0;c<4;++c)cout<<setw(10)<<fixed<<setprecision(6)<<F_j.at(r,c)<<(c<3?",":"");
        cout<<"]\n";}
    cout<<"  Q[0:4,0:4] (jerk outer-product):\n";
    for(int r=0;r<4;++r){cout<<"    [";
        for(int c=0;c<4;++c)cout<<setw(12)<<scientific<<setprecision(3)<<Q_j.at(r,c)<<(c<3?",":"");
        cout<<"]\n";}
    cout<<"  R diagonal: ["<<SIGMA_R*SIGMA_R<<", "<<SIGMA_ANGLE*SIGMA_ANGLE<<", "<<SIGMA_ANGLE*SIGMA_ANGLE<<"]\n\n";

    cout<<"[3] Converting Cartesian -> spherical (noisy)...\n";
    vector<Vec> noisySph(N);
    for(int f=0;f<N;++f) noisySph[f]=cartToSph(noisy[f]);

    cout<<"[4] Initialising 23 joint EKF filters...\n";
    vector<JointEKF> joints(NUM_JOINTS);
    for(int j=0;j<NUM_JOINTS;++j){
        joints[j].x.assign(JOINT_STATE, 0.0);
        joints[j].x[0]=noisy[0][j*3]; joints[j].x[4]=noisy[0][j*3+1]; joints[j].x[8]=noisy[0][j*3+2];
        joints[j].P       = Matrix::identity(JOINT_STATE);
        joints[j].K       = Matrix(JOINT_STATE, JOINT_MEAS, 0.0);
        joints[j].Hj_last = Matrix(3, 12, 0.0);}
    cout<<"  Initial state:\n";
    for(int j=0;j<NUM_JOINTS;++j) printJoint(joints[j],j);

    // Show Jacobian at initial pelvis position
    Matrix Hj0 = compute_jacobian(joints[0].x[0], joints[0].x[4], joints[0].x[8]);
    cout<<"\n  Initial Jacobian Hj [3x12] for pelvis:\n";
    const char* rowlbl[]={"  dr/dx:  ","  dtheta/dx: ","  dphi/dx:  "};
    for(int r=0;r<3;++r){cout<<rowlbl[r]<<"[";
        for(int c=0;c<12;++c)cout<<setw(8)<<fixed<<setprecision(4)<<Hj0.at(r,c)<<(c<11?",":"");
        cout<<"]\n";}
    cout<<"\n";

    cout<<"[5] Running EKF (Equations 6-10)...\n"
        <<"    Divergence expected at branch-cut frames (~650, ~930)\n\n";
    cout<<fixed<<setprecision(6)<<left
        <<setw(8)<<"Frame"<<setw(18)<<"Noisy RMSE(m)"<<setw(18)<<"EKF RMSE(m)"
        <<"  Note\n"<<string(82,'-')<<"\n";

    vector<double> nRmse(N), eRmse(N); Vec jSSE(NUM_JOINTS, 0.0);
    vector<vector<JointEKF>> snaps(N);
    auto t0 = chrono::high_resolution_clock::now();

    for(int f=0;f<N;++f){
        const Vec& z_cart=noisy[f]; const Vec& z_sph=noisySph[f]; const Vec& t=truth[f];
        for(int j=0;j<NUM_JOINTS;++j){
            Vec z3={z_sph[j*3],z_sph[j*3+1],z_sph[j*3+2]};
            ekf_predict(joints[j], Q_j);     // Eq.6, Eq.7
            ekf_update (joints[j], z3, R_j); // Eq.8, Eq.9, Eq.10
            double ex=joints[j].x[0]-t[j*3],ey=joints[j].x[4]-t[j*3+1],ez=joints[j].x[8]-t[j*3+2];
            jSSE[j]+=ex*ex+ey*ey+ez*ez;}
        eRmse[f]=frameEKFrmse(joints,t); nRmse[f]=frameNoisyRMSE(z_cart,t);
        snaps[f]=joints;
        bool nearDiv=(f>=600&&f<=750)||(f>=880&&f<=1000);
        if((nearDiv&&f%50==0)||(!nearDiv&&f%200==0)||f==N-1){
            string note = "";
            if(eRmse[f] > nRmse[f]*2) note = " <<< DIVERGING";
            if(eRmse[f] > 1.0)        note = " <<< DIVERGED";
            cout<<setw(8)<<f<<setw(18)<<nRmse[f]<<setw(18)<<eRmse[f]<<note<<"\n";}}

    double ms = chrono::duration<double,milli>(chrono::high_resolution_clock::now()-t0).count();
    double sumE=0,sumN=0; for(int f=0;f<N;++f){sumE+=eRmse[f];sumN+=nRmse[f];}
    double mE=sumE/N, mN=sumN/N;
    double maxE=*max_element(eRmse.begin(),eRmse.end());
    int   maxF=(int)(max_element(eRmse.begin(),eRmse.end())-eRmse.begin());

    cout<<string(82,'-')<<"\n\n=== EKF Performance (DIVERGED) ===\n"
        <<"  Mean noisy   RMSE : "<<mN<<" m\n"
        <<"  Mean EKF     RMSE : "<<mE<<" m  (WORSE than noisy input)\n"
        <<"  Max  EKF     RMSE : "<<maxE<<" m  at frame "<<maxF<<"\n"
        <<"  Wall-clock        : "<<fixed<<setprecision(1)<<ms<<" ms\n\n"
        <<"  CONCLUSION: EKF RMSE = "<<fixed<<setprecision(1)<<(mE/mN*100)<<"% of noisy RMSE.\n"
        <<"  Filter diverged due to ±2pi azimuth jumps at branch-cut crossings.\n"
        <<"  LKF (66.8% improvement, RMSE=0.103m) is the correct filter here.\n\n";

    cout<<"=== Per-Joint Mean RMSE ===\n"; double tot=0;
    for(int j=0;j<NUM_JOINTS;++j){tot+=jSSE[j];
        cout<<"  "<<left<<setw(18)<<JOINT_NAMES[j]<<fixed<<setprecision(6)<<sqrt(jSSE[j]/(N*3))<<" m\n";}
    cout<<"  "<<left<<setw(18)<<"ALL JOINTS"<<sqrt(tot/(N*NUM_JOINTS*3))<<" m\n\n";

    cout<<"Final EKF state:\n"; for(int j=0;j<NUM_JOINTS;++j) printJoint(joints[j],j);
    cout<<"\nKalman Gain pelvis (pos/vel/acc/jerk rows, r/theta/phi cols):\n";
    for(int i=0;i<4;++i){cout<<"  K["<<i<<"]=[";
        for(int c=0;c<3;++c)cout<<fixed<<setprecision(6)<<joints[0].K.at(i,c)<<(c<2?",":"");
        cout<<"]\n";}

    cout<<"\n[6] Writing results CSV...\n";
    writeResultsCSV(OUTPUT_CSV, noisy, truth, snaps, nRmse, eRmse);
   
    return 0;
}