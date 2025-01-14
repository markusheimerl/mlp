#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { double **X, **y; int n, fx, fy; } Data;

double complex_function(double x1, double x2, double x3, double x4, int output) {
    switch(output) {
        case 0: return sin(x1*2)*cos(x2*1.5) + pow(x3,2)*x4 + exp(-pow(x1-x2,2)) + 0.5*sin(x1*x2*3.14159);
        case 1: return tanh(x1+x2)*sin(x3*2) + log(fabs(x4)+1)*cos(x1) + 0.3*pow(x2-x3,3);
        case 2: return exp(-pow(x1-0.5,2))*sin(x2*3) + pow(cos(x3),2)*x4 + 0.2*sinh(x1*x2);
        default: return 0.0;
    }
}

Data* synth(int n, int fx, int fy, double noise) {
    Data* d = malloc(sizeof(Data));
    d->n = n; d->fx = fx; d->fy = fy;
    d->X = malloc(n * sizeof(double*));
    d->y = malloc(n * sizeof(double*));
    
    for(int i = 0; i < n; i++) {
        d->X[i] = malloc(fx * sizeof(double));
        d->y[i] = malloc(fy * sizeof(double));
        for(int j = 0; j < fx; j++) d->X[i][j] = (double)rand()/RAND_MAX * 4 - 2;
        for(int j = 0; j < fy; j++) {
            d->y[i][j] = complex_function(d->X[i][0], d->X[i][1], d->X[i][2], d->X[i][3], j);
            d->y[i][j] += 0.2 * sin(d->X[i][0] * 10.0) + ((double)rand()/RAND_MAX - 0.5) * noise;
        }
    }
    return d;
}

void save_csv(const char* filename, Data* data) {
    FILE* fp = fopen(filename, "w");
    if(!fp) return;
    for(int i = 0; i < data->fx; i++) fprintf(fp, "x%d,", i+1);
    for(int i = 0; i < data->fy; i++) fprintf(fp, "y%d%c", i+1, i==data->fy-1?'\n':',');
    for(int i = 0; i < data->n; i++) {
        for(int j = 0; j < data->fx; j++) fprintf(fp, "%.6f,", data->X[i][j]);
        for(int j = 0; j < data->fy; j++) fprintf(fp, "%.6f%c", data->y[i][j], j==data->fy-1?'\n':',');
    }
    fclose(fp);
}

Data* load_csv(const char* f) {
    FILE* fp = fopen(f, "r");
    if(!fp) return NULL;
    
    Data* d = malloc(sizeof(Data));
    char line[4096], *token;
    
    fgets(line, sizeof(line), fp);
    d->fx = 0; d->fy = 0;
    token = strtok(line, ",");
    while(token) {
        if(token[0] == 'x') d->fx++;
        if(token[0] == 'y') d->fy++;
        token = strtok(NULL, ",");
    }
    
    d->n = 0;
    while(fgets(line, sizeof(line), fp)) d->n++;
    rewind(fp);
    fgets(line, sizeof(line), fp);
    
    d->X = malloc(d->n * sizeof(double*));
    d->y = malloc(d->n * sizeof(double*));
    for(int i = 0; i < d->n; i++) {
        d->X[i] = malloc(d->fx * sizeof(double));
        d->y[i] = malloc(d->fy * sizeof(double));
        fgets(line, sizeof(line), fp);
        token = strtok(line, ",");
        for(int j = 0; j < d->fx; j++) {
            d->X[i][j] = atof(token);
            token = strtok(NULL, ",");
        }
        for(int j = 0; j < d->fy; j++) {
            d->y[i][j] = atof(token);
            token = strtok(NULL, ",");
        }
    }
    fclose(fp);
    return d;
}

void free_data(Data* d) {
    for(int i = 0; i < d->n; i++) { free(d->X[i]); free(d->y[i]); }
    free(d->X); free(d->y); free(d);
}

#endif