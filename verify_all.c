#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
void compare(const char *f1, const char *f2, const char *name) {
    FILE *a = fopen(f1, "r");
    FILE *b = fopen(f2, "r");
    if(!a||!b){printf("Cannot open %s or %s\n",f1,f2);return;}
    char l1[500000], l2[500000];
    fgets(l1,sizeof(l1),a); fgets(l2,sizeof(l2),b);
    double maxerr=0; int row=0;
    while(fgets(l1,sizeof(l1),a)&&fgets(l2,sizeof(l2),b)){
        char *s1=l1,*s2=l2;
        char *t1=strtok_r(s1,",\n",&s1),*t2=strtok_r(s2,",\n",&s2);
        while(t1&&t2){
            double e=fabs(atof(t1)-atof(t2));
            if(e>maxerr)maxerr=e;
            t1=strtok_r(NULL,",\n",&s1);t2=strtok_r(NULL,",\n",&s2);
        }
        row++;
    }
    printf("[%s] Rows: %d, Max error: %.3e, %s\n",name,row,maxerr,maxerr<1e-9?"PASS":"FAIL");
    fclose(a);fclose(b);
}
int main(){
    compare("LKF_output.csv","LKF_asm_output.csv","LKF");
    compare("EKF_output.csv","EKF_asm_output.csv","EKF");
    return 0;
}
