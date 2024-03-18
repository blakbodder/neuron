//
//  matrix.hpp
//  neuron

#ifndef matrix_hpp
#define matrix_hpp

typedef struct  {
    unsigned int rows;
    unsigned int cols;
    unsigned int base;  // offset in data buffer
} matdescriptor_t;

typedef struct {
    matdescriptor_t mda;
    matdescriptor_t mdb;
    matdescriptor_t mdr;
    float scalar;
} metadata_t;

#ifdef __cplusplus
#include <stdio.h>
#include <stdlib.h>

class Matrix {
    matdescriptor_t md;
    friend Matrix operator * (Matrix a, Matrix b);
    friend Matrix operator * (float scalar, Matrix a);
    friend Matrix operator + (Matrix a, Matrix b);
    friend Matrix relu(Matrix a);
    friend Matrix sigmoid(Matrix a);
    friend float bigl(void);
    friend void train(void);
    friend void validate(void);
    friend void push_mat_buff_chunk(Matrix* mat);
    friend void update_theta(void);
    friend void matrixtest(void);
    
public:
    Matrix(unsigned int rows, unsigned int cols);
    void operator = (Matrix other);
    void init_data(float* data);
    void fill_random(float mean, float spread);
    void y_th1(void);
    void z_th1(void);
    void g_th1(void);
    void g_th2(void);
    void g_theta(void);
    void fhat_theta(void);
    void L_fhat(void);
    void L_fhat_binary_cross_entropy(void);
    void L_theta(void);
    void print(void);
    unsigned int syze(void);
};

#endif

#endif /* matrix_hpp */
