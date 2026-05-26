#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double* read_csv(const char *path, int *out_rows, int *out_cols) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    char line[65536];
    fgets(line, sizeof(line), f);
    int rows = 0, cols = 0;
    long data_start = ftell(f);
    while (fgets(line, sizeof(line), f)) {
        if (rows == 0) {
            char *tmp = strdup(line);
            char *tok = strtok(tmp, ",\n");
            while (tok) { cols++; tok = strtok(NULL, ",\n"); }
            free(tmp);
        }
        rows++;
    }
    double *data = (double*)malloc(rows * cols * sizeof(double));
    fseek(f, data_start, SEEK_SET);
    int idx = 0;
    while (fgets(line, sizeof(line), f)) {
        char *tok = strtok(line, ",\n");
        while (tok) { data[idx++] = atof(tok); tok = strtok(NULL, ",\n"); }
    }
    fclose(f);
    *out_rows = rows;
    *out_cols = cols;
    return data;
}

FILE* open_output(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot create %s\n", path); exit(1); }
    const char *joints[] = {
        "pelvis","L5","L3","T12","T8","neck","head",
        "shoulderRight","upperArmRight","forearmRight","handRight",
        "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
        "upperLegRight","lowerLegRight","footRight","toeRight",
        "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
    };
    const char *states[] = {
        "px","vx","ax","jx","py","vy","ay","jy","pz","vz","az","jz"
    };
    for (int j = 0; j < 23; j++)
        for (int s = 0; s < 12; s++)
            fprintf(f, "%s_%s%s", joints[j], states[s],
                    (j==22 && s==11) ? "\n" : ",");
    return f;
}

void write_row(FILE *f, const double *x, int n) {
    for (int i = 0; i < n; i++)
        fprintf(f, "%.15g%s", x[i], (i == n-1) ? "\n" : ",");
}

void close_output(FILE *f) { fclose(f); }
