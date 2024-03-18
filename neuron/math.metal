//
//  File.metal
//  neuron

#include <metal_stdlib>
using namespace metal;

typedef struct  {
    unsigned int rows;
    unsigned int cols;
    unsigned int base;
} matdescriptor_t;

typedef struct {
    matdescriptor_t mda;
    matdescriptor_t mdb;
    matdescriptor_t mdr;
    float scalar;
} metadata_t;

// notte benne: mr must not be same as ma or mb
// matrices stored in row-major order
kernel void matrix_multiply(device float* data,
                            device metadata_t* meta,
                            vector_uint2 index [[thread_position_in_grid]] )    // row,col of r[]
{
    uint  row, col, j, bstride;
    float sigma;
    device float *a, *b, *r;
    
    row = index.x;  col = index.y;
                                                              // rows and cols indexed 0,1,2,...
    a = data + meta->mda.base + row * meta->mda.cols;         // addr of ma[row][0]
    b = data + meta->mdb.base + col;                          // addr of mb[0][col]
    r = data + meta->mdr.base + row * meta->mdr.cols + col;   // addr of mr[row][col]
    bstride = meta->mdb.cols;
    sigma = 0.0;
    for (j=0; j<meta->mda.cols; j++) {
        sigma += (*a) * (*b);
        a++;  b += bstride;
    }
    *r = sigma;
}

/*
kernel void matrix_add(device float* data,
                       device metadata_t* meta,
                       vector_uint2 index [[thread_position_in_grid]] )    // row,col of r[]
{   // assumes a, b and r structured same way
    uint  row, col, offset;
    device float *a, *b, *r;
    
    row = index.x;  col = index.y;
    offset = row * mdr->cols + col;
    
    a = data + meta->mda.base + offset;
    b = data + meta->mdb.base + offset;
    r = data + meta->mdr.base + offset;
    *r = *a + *b;
} */

// vector_add can also add matrices which look like vectors when flat
kernel void vector_add(device float* data,
                       device metadata_t* meta,
                       uint index [[thread_position_in_grid]] )
{
    device float *a, *b, *r;
    a = data + meta->mda.base + index;
    b = data + meta->mdb.base + index;
    r = data + meta->mdr.base + index;
    *r = *a + *b;
}

kernel void vector_plusequ(device float* data,   // r += a
                           device metadata_t* meta,
                           uint index [[thread_position_in_grid]] )
{
    device float *a, *r;
    a = data + meta->mda.base + index;
    r = data + meta->mdr.base + index;
    *r += *a;
}

kernel void vector_scale(device float* data,
                         device metadata_t* meta,
                         uint index [[thread_position_in_grid]] )
{
    device float *a, *r;
    a = data + meta->mda.base + index;
    r = data + meta->mdr.base + index;
    *r = *a * meta->scalar;
}

