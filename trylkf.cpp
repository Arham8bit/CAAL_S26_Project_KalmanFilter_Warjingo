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
using Matrix = vector<vector<double>>;
using Vec    = vector<double>;

constexpr int    NUM_JOINTS  = 23;
constexpr int    JOINT_STATE = 12;
constexpr int    JOINT_MEAS  = 3;
constexpr int    STATE_DIM   = NUM_JOINTS * JOINT_STATE;   // 276
constexpr int    MEAS_DIM    = NUM_JOINTS * JOINT_MEAS;    //  69
constexpr double DT          = 0.01;
constexpr double Q_NOISE     = 1e-4;
constexpr double R_NOISE     = 1e-2;

const string NOISY_CSV  = "noisy_values.csv";
const string TRUE_CSV   = "true_values.csv";
const string OUTPUT_CSV = "lkf_results.csv";

const array<string,23> JOINT_NAMES = {
    "pelvis","L5","L3","T12","T8","neck","head",
    "shoulderRight","upperArmRight","forearmRight","handRight",
    "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
    "upperLegRight","lowerLegRight","footRight","toeRight",
    "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
};

// ================================================================
//  MATRIX UTILITIES  (heap-allocated via std::vector)
// ================================================================
Matrix makeMatrix(int r, int c, double v=0.0){ return Matrix(r,Vec(c,v)); }
Matrix identity(int n){ Matrix I=makeMatrix(n,n); for(int i=0;i<n;++i) I[i][i]=1.0; return I; }

Matrix add(const Matrix& A, const Matrix& B){
    int r=A.size(),c=A[0].size(); Matrix C=makeMatrix(r,c);
    for(int i=0;i<r;++i) for(int j=0;j<c;++j) C[i][j]=A[i][j]+B[i][j]; return C;
}
Matrix subtract(const Matrix& A, const Matrix& B){
    int r=A.size(),c=A[0].size(); Matrix C=makeMatrix(r,c);
    for(int i=0;i<r;++i) for(int j=0;j<c;++j) C[i][j]=A[i][j]-B[i][j]; return C;
}
Matrix multiply(const Matrix& A, const Matrix& B){
    int rA=A.size(),cA=A[0].size(),cB=B[0].size(); Matrix C=makeMatrix(rA,cB);
    for(int i=0;i<rA;++i) for(int k=0;k<cA;++k){ if(A[i][k]==0.0)continue;
        for(int j=0;j<cB;++j) C[i][j]+=A[i][k]*B[k][j]; } return C;
}
Vec multiplyVec(const Matrix& A, const Vec& v){
    int r=A.size(),c=A[0].size(); Vec res(r,0.0);
    for(int i=0;i<r;++i) for(int j=0;j<c;++j) res[i]+=A[i][j]*v[j]; return res;
}
Matrix transpose(const Matrix& A){
    int r=A.size(),c=A[0].size(); Matrix T=makeMatrix(c,r);
    for(int i=0;i<r;++i) for(int j=0;j<c;++j) T[j][i]=A[i][j]; return T;
}

// Analytical 3x3 inverse — Section 6.1: avoids Gauss-Jordan on 69x69 S
Matrix inverse3x3(const Matrix& A){
    double d=A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
            -A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
            +A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    if(fabs(d)<1e-12) throw runtime_error("inverse3x3: singular");
    d=1.0/d; Matrix I=makeMatrix(3,3);
    I[0][0]= (A[1][1]*A[2][2]-A[1][2]*A[2][1])*d; I[0][1]=-(A[0][1]*A[2][2]-A[0][2]*A[2][1])*d; I[0][2]= (A[0][1]*A[1][2]-A[0][2]*A[1][1])*d;
    I[1][0]=-(A[1][0]*A[2][2]-A[1][2]*A[2][0])*d; I[1][1]= (A[0][0]*A[2][2]-A[0][2]*A[2][0])*d; I[1][2]=-(A[0][0]*A[1][2]-A[0][2]*A[1][0])*d;
    I[2][0]= (A[1][0]*A[2][1]-A[1][1]*A[2][0])*d; I[2][1]=-(A[0][0]*A[2][1]-A[0][1]*A[2][0])*d; I[2][2]= (A[0][0]*A[1][1]-A[0][1]*A[1][0])*d;
    return I;
}

// ================================================================
//  SYSTEM MATRICES  (built once, shared by all joints)
// ================================================================
Matrix F_j, H_j, Q_j, R_j, I12;

// F: constant-jerk kinematic model — 3 independent 4x4 axis blocks
void build_F(double dt){
    double dt2=dt*dt, dt3=dt2*dt; F_j=makeMatrix(12,12);
    for(int a=0;a<3;++a){ int b=a*4;
        F_j[b][b]=1; F_j[b][b+1]=dt; F_j[b][b+2]=dt2/2; F_j[b][b+3]=dt3/6;
        F_j[b+1][b+1]=1; F_j[b+1][b+2]=dt; F_j[b+1][b+3]=dt2/2;
        F_j[b+2][b+2]=1; F_j[b+2][b+3]=dt; F_j[b+3][b+3]=1; }
}
// H: z=H*x — picks position only (vel/acc/jerk cols = 0)
void build_H(){ H_j=makeMatrix(3,12); H_j[0][0]=1.0; H_j[1][4]=1.0; H_j[2][8]=1.0; }
void build_QR(double q, double r){ Q_j=makeMatrix(12,12); for(int i=0;i<12;++i) Q_j[i][i]=q;
    R_j=makeMatrix(3,3); for(int i=0;i<3;++i) R_j[i][i]=r; }

// ================================================================
//  PER-JOINT STATE  (heap-allocated — Section 6.3)
// ================================================================
struct JointFilter {
    Vec    x;  // [12] state: [px vx ax jx | py vy ay jy | pz vz az jz]
    Matrix P;  // [12x12] error covariance
    Matrix K;  // [12x3]  Kalman gain
};

// PREDICT — Equations 1 and 2
void predict(JointFilter& jf){
    jf.x = multiplyVec(F_j, jf.x);                              // Eq.1
    Matrix FT=transpose(F_j);
    jf.P = add(multiply(multiply(F_j,jf.P),FT), Q_j);          // Eq.2
}

// UPDATE — Equations 3, 4, 5
void update(JointFilter& jf, const Vec& z3){
    Matrix HT=transpose(H_j);
    Matrix PHt=multiply(jf.P,HT);
    Matrix S=add(multiply(H_j,PHt),R_j);                       // S=H*P*H^T+R
    jf.K=multiply(PHt,inverse3x3(S));                          // Eq.3 K=P*H^T*S^-1
    Vec Hx=multiplyVec(H_j,jf.x);
    Vec y={z3[0]-Hx[0],z3[1]-Hx[1],z3[2]-Hx[2]};             // innovation
    Vec Ky=multiplyVec(jf.K,y);
    for(int i=0;i<12;++i) jf.x[i]+=Ky[i];                    // Eq.4
    Matrix KH=multiply(jf.K,H_j); Matrix IKH=subtract(I12,KH);
    Matrix KRKt=multiply(multiply(jf.K,R_j),transpose(jf.K));
    jf.P=add(multiply(multiply(IKH,jf.P),transpose(IKH)),KRKt); // Eq.5 Joseph form
}

// ================================================================
//  CSV LOADER
// ================================================================
vector<Vec> loadCSV(const string& path){
    ifstream f(path); if(!f.is_open()) throw runtime_error("Cannot open: "+path);
    vector<Vec> frames; string line; bool hdr=false; int ln=0;
    while(getline(f,line)){ ++ln; if(!line.empty()&&line.back()=='\r') line.pop_back();
        if(line.empty()) continue;
        if(!hdr){ hdr=true; stringstream t(line); string tk; getline(t,tk,',');
            bool isH=false; try{stod(tk);}catch(...){isH=true;}
            if(isH){ cout<<"  [CSV] header skipped (line "<<ln<<")\n"; continue; } }
        stringstream ss(line); string tok; Vec row; row.reserve(MEAS_DIM);
        while(getline(ss,tok,',')) if(!tok.empty()){ try{row.push_back(stod(tok));}catch(...){row.clear();break;} }
        if((int)row.size()!=MEAS_DIM){ cerr<<"WARNING: line "<<ln<<" skipped\n"; continue; }
        frames.push_back(move(row)); }
    return frames;
}

// ================================================================
//  RMSE
// ================================================================
double frameFilteredRMSE(const vector<JointFilter>& jf, const Vec& truth){
    double sse=0; for(int j=0;j<NUM_JOINTS;++j){
        double ex=jf[j].x[0]-truth[j*3]; double ey=jf[j].x[4]-truth[j*3+1]; double ez=jf[j].x[8]-truth[j*3+2];
        sse+=ex*ex+ey*ey+ez*ez; } return sqrt(sse/(NUM_JOINTS*3));
}
double frameNoisyRMSE(const Vec& n, const Vec& t){
    double sse=0; for(int i=0;i<MEAS_DIM;++i){ double e=n[i]-t[i]; sse+=e*e; } return sqrt(sse/MEAS_DIM);
}

void printJoint(const JointFilter& jf, int idx){
    cout<<"  "<<left<<setw(16)<<JOINT_NAMES[idx]<<"pos=("<<fixed<<setprecision(5)
        <<setw(10)<<jf.x[0]<<", "<<setw(10)<<jf.x[4]<<", "<<setw(10)<<jf.x[8]
        <<")  vel=("<<setw(8)<<jf.x[1]<<", "<<setw(8)<<jf.x[5]<<", "<<setw(8)<<jf.x[9]<<")\n";
}

// ================================================================
//  WRITE RESULTS CSV — positions + FULL STATE VECTOR (Section 9)
// ================================================================
void writeResultsCSV(const string& path,
                     const vector<Vec>& noisyF, const vector<Vec>& trueF,
                     const vector<Vec>& filtPos,
                     const vector<vector<JointFilter>>& snaps,
                     const vector<double>& nRmse, const vector<double>& fRmse){
    ofstream f(path); if(!f.is_open()){ cerr<<"Cannot write "<<path<<"\n"; return; }
    f<<"frame,noisy_rmse,filtered_rmse";
    for(int j=0;j<NUM_JOINTS;++j) f<<",noisy_"<<JOINT_NAMES[j]<<"_x,noisy_"<<JOINT_NAMES[j]<<"_y,noisy_"<<JOINT_NAMES[j]<<"_z";
    for(int j=0;j<NUM_JOINTS;++j) f<<",filt_"<<JOINT_NAMES[j]<<"_x,filt_"<<JOINT_NAMES[j]<<"_y,filt_"<<JOINT_NAMES[j]<<"_z";
    for(int j=0;j<NUM_JOINTS;++j) f<<",true_"<<JOINT_NAMES[j]<<"_x,true_"<<JOINT_NAMES[j]<<"_y,true_"<<JOINT_NAMES[j]<<"_z";
    for(int j=0;j<NUM_JOINTS;++j){ const string& n=JOINT_NAMES[j];
        f<<",lkf_"<<n<<"_px,lkf_"<<n<<"_vx,lkf_"<<n<<"_ax,lkf_"<<n<<"_jx"
         <<",lkf_"<<n<<"_py,lkf_"<<n<<"_vy,lkf_"<<n<<"_ay,lkf_"<<n<<"_jy"
         <<",lkf_"<<n<<"_pz,lkf_"<<n<<"_vz,lkf_"<<n<<"_az,lkf_"<<n<<"_jz"; }
    f<<"\n";
    int N=noisyF.size();
    for(int i=0;i<N;++i){
        f<<i<<","<<fixed<<setprecision(8)<<nRmse[i]<<","<<fRmse[i];
        for(double v:noisyF[i]) f<<","<<v;
        for(double v:filtPos[i]) f<<","<<v;
        for(double v:trueF[i]) f<<","<<v;
        for(int j=0;j<NUM_JOINTS;++j) for(int k=0;k<JOINT_STATE;++k) f<<","<<snaps[i][j].x[k];
        f<<"\n"; }
    cout<<"LKF CSV written: "<<path<<" (486 cols = 3+69*3+276)\n";
}

// ================================================================
//  MAIN
// ================================================================
int main(){
    cout<<"================================================================\n"
        <<"  Full Body 3D Gait – Linear Kalman Filter (LKF)\n"
        <<"  State="<<STATE_DIM<<"  Meas="<<MEAS_DIM
        <<"  dt="<<DT<<"  Q="<<Q_NOISE<<"  R="<<R_NOISE<<"\n"
        <<"  Memory: HEAP (std::vector)  |  S-inv: analytical 3x3\n"
        <<"================================================================\n\n";

    cout<<"[1/5] Loading datasets...\n";
    vector<Vec> noisy=loadCSV(NOISY_CSV), truth=loadCSV(TRUE_CSV);
    cout<<"  Noisy: "<<noisy.size()<<"  True: "<<truth.size()<<" frames\n\n";
    if(noisy.empty()||truth.empty()){ cerr<<"ERROR: load failed\n"; return 1; }
    int N=(int)min(noisy.size(),truth.size());

    cout<<"[2/5] Building matrices...\n";
    build_F(DT); build_H(); build_QR(Q_NOISE,R_NOISE); I12=identity(12);
    cout<<"  F[0:4,0:4] x-axis block:\n";
    for(int r=0;r<4;++r){ cout<<"    ["; for(int c=0;c<4;++c) cout<<setw(10)<<fixed<<setprecision(6)<<F_j[r][c]<<(c<3?",":""); cout<<"]\n"; }
    cout<<"  H: px->col0  py->col4  pz->col8\n\n";

    cout<<"[3/5] Initialising 23 joint filters...\n";
    vector<JointFilter> joints(NUM_JOINTS);
    for(int j=0;j<NUM_JOINTS;++j){
        joints[j].x.assign(JOINT_STATE,0.0);
        joints[j].x[0]=noisy[0][j*3]; joints[j].x[4]=noisy[0][j*3+1]; joints[j].x[8]=noisy[0][j*3+2];
        joints[j].P=identity(JOINT_STATE); joints[j].K=makeMatrix(JOINT_STATE,JOINT_MEAS,0.0); }
    cout<<"  Initial state (seeded from first noisy frame):\n";
    for(int j=0;j<NUM_JOINTS;++j) printJoint(joints[j],j);
    cout<<"\n";

    cout<<"[4/5] Running LKF ("<<N<<" frames)...\n\n";
    cout<<fixed<<setprecision(6)<<left
        <<setw(8)<<"Frame"<<setw(18)<<"Noisy RMSE(m)"<<setw(18)<<"Filtered RMSE(m)"<<"  Pelvis: noisy->filtered\n"<<string(82,'-')<<"\n";

    vector<double> nRmse(N),fRmse(N); Vec jSSE(NUM_JOINTS,0.0);
    vector<Vec> filtPos(N,Vec(MEAS_DIM));
    vector<vector<JointFilter>> snaps(N);
    auto t0=chrono::high_resolution_clock::now();

    for(int f=0;f<N;++f){
        const Vec& z=noisy[f]; const Vec& t=truth[f];
        for(int j=0;j<NUM_JOINTS;++j){
            Vec z3={z[j*3],z[j*3+1],z[j*3+2]};
            predict(joints[j]); update(joints[j],z3);
            filtPos[f][j*3]=joints[j].x[0]; filtPos[f][j*3+1]=joints[j].x[4]; filtPos[f][j*3+2]=joints[j].x[8];
            double ex=joints[j].x[0]-t[j*3],ey=joints[j].x[4]-t[j*3+1],ez=joints[j].x[8]-t[j*3+2];
            jSSE[j]+=ex*ex+ey*ey+ez*ez; }
        fRmse[f]=frameFilteredRMSE(joints,t); nRmse[f]=frameNoisyRMSE(z,t);
        snaps[f]=joints;
        if(f%200==0||f==N-1) cout<<setw(8)<<f<<setw(18)<<nRmse[f]<<setw(18)<<fRmse[f]
            <<"  noisy=("<<z[0]<<","<<z[1]<<","<<z[2]<<")"
            <<"  filt=("<<joints[0].x[0]<<","<<joints[0].x[4]<<","<<joints[0].x[8]<<")\n"; }

    double ms=chrono::duration<double,milli>(chrono::high_resolution_clock::now()-t0).count();
    double sumF=0,sumN=0; for(int f=0;f<N;++f){sumF+=fRmse[f];sumN+=nRmse[f];}
    double mF=sumF/N,mN=sumN/N;
    cout<<string(82,'-')<<"\n\n=== LKF Summary ===\n"
        <<"  Mean noisy RMSE  : "<<mN<<" m\n  Mean filt RMSE   : "<<mF<<" m\n"
        <<"  Improvement      : "<<fixed<<setprecision(2)<<(mN>0?100*(mN-mF)/mN:0)<<"%\n"
        <<"  Wall-clock       : "<<fixed<<setprecision(1)<<ms<<" ms\n\n";

    cout<<"=== Per-Joint Mean RMSE ===\n"; double tot=0;
    for(int j=0;j<NUM_JOINTS;++j){ tot+=jSSE[j]; cout<<"  "<<left<<setw(18)<<JOINT_NAMES[j]<<fixed<<setprecision(6)<<sqrt(jSSE[j]/(N*3))<<" m\n"; }
    cout<<"  "<<left<<setw(18)<<"ALL JOINTS"<<sqrt(tot/(N*NUM_JOINTS*3))<<" m\n\n";

    cout<<"Final Pose:\n"; for(int j=0;j<NUM_JOINTS;++j) printJoint(joints[j],j);
    cout<<"\nKalman Gain pelvis (pos/vel/acc/jerk, px/py/pz):\n";
    for(int i=0;i<4;++i){ cout<<"  K["<<i<<"]=["; for(int c=0;c<3;++c) cout<<fixed<<setprecision(6)<<joints[0].K[i][c]<<(c<2?",":""); cout<<"]\n"; }

    cout<<"\n[5/5] Writing LKF CSV...\n";
    writeResultsCSV(OUTPUT_CSV,noisy,truth,filtPos,snaps,nRmse,fRmse);
    cout<<"\nDone. Run: python3 ekf_plot.py\n";
    return 0;
}