#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CSV file reader - returns flat double array */
double* load_csv(const char *filename, int *nrows, int *ncols) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", filename); exit(1); }
    char buf[65536];
    fgets(buf, sizeof(buf), fp);
    int r = 0, c = 0;
    long start = ftell(fp);
    while (fgets(buf, sizeof(buf), fp)) {
        if (r == 0) {
            char *cp = strdup(buf);
            char *p = strtok(cp, ",\n");
            while (p) { c++; p = strtok(NULL, ",\n"); }
            free(cp);
        }
        r++;
    }
    double *arr = (double*)malloc(r * c * sizeof(double));
    fseek(fp, start, SEEK_SET);
    int pos = 0;
    while (fgets(buf, sizeof(buf), fp)) {
        char *p = strtok(buf, ",\n");
        while (p) { arr[pos++] = atof(p); p = strtok(NULL, ",\n"); }
    }
    fclose(fp);
    *nrows = r;
    *ncols = c;
    return arr;
}

/* Open CSV for writing, print header row */
FILE* csv_open(const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Error: cannot create %s\n", filename); exit(1); }
    const char *jnames[] = {
        "pelvis","L5","L3","T12","T8","neck","head",
        "shoulderRight","upperArmRight","forearmRight","handRight",
        "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
        "upperLegRight","lowerLegRight","footRight","toeRight",
        "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
    };
    const char *snames[] = {"px","vx","ax","jx","py","vy","ay","jy","pz","vz","az","jz"};
    for (int j = 0; j < 23; j++)
        for (int s = 0; s < 12; s++)
            fprintf(fp, "%s_%s%s", jnames[j], snames[s], (j==22&&s==11)?"\n":",");
    return fp;
}

/* Write a single state row to CSV */
void csv_write_state(FILE *fp, const double *state, int dim) {
    for (int i = 0; i < dim; i++)
        fprintf(fp, "%.15g%s", state[i], (i==dim-1)?"\n":",");
}

/* Close CSV file */
void csv_close(FILE *fp) { fclose(fp); }
