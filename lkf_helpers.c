#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * I/O-only helper functions called FROM assembly.
 * No computation here — just file read/write.
 */

/* Reads CSV, skips header, returns flat array of doubles.
 * out_rows and out_cols are set.
 * Returns pointer to flat row-major double array [rows * cols]. */
double* read_csv(const char *path, int *out_rows, int *out_cols) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    char line[65536];
    fgets(line, sizeof(line), f);  /* skip header */

    /* First pass: count rows and cols */
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

    /* Allocate flat array */
    double *data = (double*)malloc(rows * cols * sizeof(double));

    /* Second pass: read values */
    fseek(f, data_start, SEEK_SET);
    int idx = 0;
    while (fgets(line, sizeof(line), f)) {
        char *tok = strtok(line, ",\n");
        while (tok) {
            data[idx++] = atof(tok);
            tok = strtok(NULL, ",\n");
        }
    }

    fclose(f);
    *out_rows = rows;
    *out_cols = cols;
    return data;
}

/* Opens output CSV and writes header. Returns FILE pointer. */
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

/* Writes one row (276 doubles) to the output file. */
void write_row(FILE *f, const double *x, int n) {
    for (int i = 0; i < n; i++)
        fprintf(f, "%.15g%s", x[i], (i == n-1) ? "\n" : ",");
}

/* Closes the output file. */
void close_output(FILE *f) {
    fclose(f);
}
