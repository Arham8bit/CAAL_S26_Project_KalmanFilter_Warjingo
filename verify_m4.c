/*
 * verify_m4.c — Milestone 4 Numerical Verification
 * Compares vectorised (M4) output against scalar (M3) reference.
 * Generates all 4 tables required by the M4 specification:
 *   Table 1: Average absolute error per joint
 *   Table 2: Average absolute error per state component
 *   Table 3: Global max/min absolute error
 *   Table 4: M3 scalar error vs M4 vector error comparison
 *
 * Compile:  gcc -O2 -o verify_m4 verify_m4.c -lm
 * Run:      ./verify_m4
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define N_STATES  276
#define N_JOINTS   23
#define SPJ        12   /* states per joint */
#define MAX_ROWS 3100

static const char *joint_names[N_JOINTS] = {
    "pelvis","L5","L3","T12","T8","neck","head",
    "shoulderRight","upperArmRight","forearmRight","handRight",
    "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
    "upperLegRight","lowerLegRight","footRight","toeRight",
    "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
};
static const char *state_names[4] = {"position","velocity","acceleration","jerk"};
/* Within each joint's 12 states: px vx ax jx py vy ay jy pz vz az jz
   Indices 0,4,8 = position; 1,5,9 = velocity; 2,6,10 = acceleration; 3,7,11 = jerk */
static int state_group(int s) {
    return s % 4;  /* 0=pos, 1=vel, 2=acc, 3=jerk */
}

/* Read a CSV of doubles (skip header). Returns row count. */
static int read_csv(const char *path, double data[][N_STATES]) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    char line[500000];
    fgets(line, sizeof(line), f); /* skip header */
    int rows = 0;
    while (fgets(line, sizeof(line), f) && rows < MAX_ROWS) {
        int col = 0;
        char *tok = strtok(line, ",\n");
        while (tok && col < N_STATES) {
            data[rows][col++] = atof(tok);
            tok = strtok(NULL, ",\n");
        }
        rows++;
    }
    fclose(f);
    return rows;
}

static double (*vec_data)[N_STATES];
static double (*scl_data)[N_STATES];

static void run_comparison(const char *vec_file, const char *scl_file, const char *filter_name) {

    vec_data = malloc(MAX_ROWS * sizeof(*vec_data));
    scl_data = malloc(MAX_ROWS * sizeof(*scl_data));
    if (!vec_data || !scl_data) { fprintf(stderr, "malloc failed\n"); return; }

    int rv = read_csv(vec_file, vec_data);
    int rs = read_csv(scl_file, scl_data);
    if (rv < 0 || rs < 0) { free(vec_data); free(scl_data); return; }
    int rows = rv < rs ? rv : rs;

    printf("\n========================================\n");
    printf("  %s: M4 (vector) vs M3 (scalar)\n", filter_name);
    printf("  Vector file : %s\n", vec_file);
    printf("  Scalar file : %s\n", scl_file);
    printf("  Rows compared: %d\n", rows);
    printf("========================================\n");

    /* --- Compute errors --- */
    double global_max = 0.0, global_min = DBL_MAX;
    double joint_sum[N_JOINTS] = {0};
    int    joint_cnt[N_JOINTS] = {0};
    double comp_sum[4] = {0};  /* pos, vel, acc, jerk */
    int    comp_cnt[4] = {0};

    for (int t = 0; t < rows; t++) {
        for (int i = 0; i < N_STATES; i++) {
            double err = fabs(vec_data[t][i] - scl_data[t][i]);
            if (err > global_max) global_max = err;
            if (err < global_min) global_min = err;

            int j = i / SPJ;           /* joint index */
            int s = i % SPJ;           /* state within joint */
            joint_sum[j] += err;
            joint_cnt[j]++;
            int g = state_group(s);
            comp_sum[g] += err;
            comp_cnt[g]++;
        }
    }

    /* --- Table 1: Average error per joint --- */
    printf("\n--- Table 1: Average Absolute Error per Joint ---\n");
    printf("%-20s  %s Avg Error\n", "Joint", filter_name);
    printf("%-20s  ---------------\n", "--------------------");
    for (int j = 0; j < N_JOINTS; j++) {
        double avg = joint_cnt[j] > 0 ? joint_sum[j] / joint_cnt[j] : 0;
        printf("%-20s  %.6e\n", joint_names[j], avg);
    }

    /* --- Table 2: Average error per state component --- */
    printf("\n--- Table 2: Average Absolute Error per State Component ---\n");
    printf("%-15s  %s Avg Error\n", "Component", filter_name);
    printf("%-15s  ---------------\n", "---------------");
    for (int g = 0; g < 4; g++) {
        double avg = comp_cnt[g] > 0 ? comp_sum[g] / comp_cnt[g] : 0;
        printf("%-15s  %.6e\n", state_names[g], avg);
    }

    /* --- Table 3: Global max/min error --- */
    printf("\n--- Table 3: Global Max/Min Absolute Error ---\n");
    printf("%-15s  %.6e\n", "Max error:", global_max);
    printf("%-15s  %.6e\n", "Min error:", global_min);
    printf("%-15s  %s\n", "Status:", global_max <= 1e-9 ? "PASS (<=1e-9)" : "CHECK (>1e-9)");

    /* --- Table 4: Per-joint per-component sample comparison --- */
    printf("\n--- Table 4: M3 Scalar vs M4 Vector Error (sample joints) ---\n");
    printf("%-20s  %-15s  M3-vs-M4 Error\n", "Joint", "Component");
    printf("%-20s  %-15s  ---------------\n", "--------------------", "---------------");
    int sample_joints[] = {0, 6, 10, 14, 18, 22};  /* pelvis,head,handR,handL,toeR,toeL */
    for (int sj = 0; sj < 6; sj++) {
        int j = sample_joints[sj];
        for (int g = 0; g < 4; g++) {
            /* average error for this joint/component across all frames */
            double sum = 0; int cnt = 0;
            for (int t = 0; t < rows; t++) {
                /* 3 axes per component group */
                for (int ax = 0; ax < 3; ax++) {
                    int idx = j * SPJ + ax * 4 + g;
                    double err = fabs(vec_data[t][idx] - scl_data[t][idx]);
                    sum += err; cnt++;
                }
            }
            printf("%-20s  %-15s  %.6e\n", joint_names[j], state_names[g],
                   cnt > 0 ? sum / cnt : 0);
        }
    }

    free(vec_data);
    free(scl_data);
}

int main() {
    printf("=== Milestone 4 Numerical Verification ===\n");
    printf("Tolerance: 1e-9\n");

    run_comparison("LKF_vector_output.csv", "LKF_asm_output.csv", "LKF");
    run_comparison("EKF_vector_output.csv", "EKF_asm_output.csv", "EKF");

    printf("\n=== Verification Complete ===\n");
    return 0;
}
