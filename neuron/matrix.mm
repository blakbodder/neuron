//
//  matrix.cpp
//  neuron

#include "matrix.hpp"
#include <Metal/Metal.h>
#include "make_data.h"
#include <CoreFoundation/CoreFoundation.h>
#include <Cocoa/Cocoa.h>
#import "proto.h"

#define ONE_YEAR 31536000.0

extern id <MTLCommandQueue> commandQueue;
extern id <MTLComputePipelineState> mulFunctionPSO;
extern id <MTLComputePipelineState> addFunctionPSO;
extern id <MTLComputePipelineState> plusequFunctionPSO;
extern id <MTLComputePipelineState> scaleFunctionPSO;

extern id <MTLBuffer> databuff;
extern id <MTLBuffer> metabuff;

extern sample_t data[NSAMPLES];

void multiply(matdescriptor_t* mr, matdescriptor_t* ma, matdescriptor_t* mb);
void mat_add(matdescriptor_t* mr, matdescriptor_t* ma, matdescriptor_t* mb);
void mat_scale(matdescriptor_t* mr, float scalar, matdescriptor_t* ma);

float* data_buffer; // == metal buffer contents
metadata_t* meta_buffer;

NSViewController <graph>* madre;
int numpoints=0;
int crrct[33];
float mse[33];

static CFRunLoopRef mainrunloop;
static CFRunLoopTimerContext timer_context = { 0, 0, 0, 0, 0 };

static int itrain=0;
void train(CFRunLoopTimerRef timer, void* info);
void validate(void);

static CFRunLoopTimerCallBack timer_callback = train;

// need to do a bit of memory management
// becuse the meat of the matrix is stored in metal buffer
// where the GPU can crunch the numbers
// so : keep track of allocated memory blocks and recycle
// when the corresponding matrices are done with.
// returning blocks to the pool in reverse order of allocation
// avoids need for garbage collection.
// need to be careful not to use matrices after their data has been binned

#define STAKLEN 128
#define BUFFLEN 2048        // the data buffer is 2048 floats
static unsigned int pool=0;

static unsigned int allocation_stak[STAKLEN]; // stack of chunk sizes
static int sp=0;    // stak pointer
int state = 0;

unsigned int get_chunk_offset(unsigned int syze) // grab chunk of buffer (syze * sizeof(float))
{                                               // return addr
    unsigned int offset, poolsize=BUFFLEN-pool;
    if (syze > poolsize) {
        printf("FATAL ERROR. data buffer not big enough.\n" );
        return 0;
    }
    // bite chunk off pool
    offset = pool;  pool += syze;
    return offset;
}

void push_mat_buff_chunk(Matrix* mat)
{
    if (sp > STAKLEN-1) {
        printf("TROUBLE: allocation_stak overflow.\n"
               "need bigger stak or more frequent recycling\n" );
        return;
    }
    unsigned int syze = mat->md.rows * mat->md.cols;
    allocation_stak[sp++] = syze;
}

void recycle_buffer_space(void)
{
    unsigned int syze;
    while (sp > 0) { syze = allocation_stak[--sp]; pool -= syze; }
    //printf("pool=%u\n", pool);
}

Matrix::Matrix(unsigned int rows, unsigned int cols)
{
    unsigned int syze;
    md.rows = rows;
    md.cols = cols;
    syze = rows*cols;
    md.base = get_chunk_offset(syze);
    //printf("base = %d\n", md.base);
}

unsigned int Matrix::syze(void)
{
    return md.rows * md.cols;
}


void Matrix::init_data(float* data)     // note data in row-major order
{                                       // some matrix libraries such as simd use column-major order
    float* dest = data_buffer + md.base;
    
    memcpy(dest, data, md.rows * md.cols * sizeof(float));
}

void Matrix::fill_random(float mean, float spread)
{
    unsigned int i, n;
    float* f = data_buffer + md.base;
    float m = (float) RAND_MAX;
    
    n = md.rows*md.cols;
    for (i=0; i<n; i++) {
        *f = spread*((float)random()/m -0.5f) + mean;
        f++;
    }
}

Matrix operator * (Matrix a, Matrix b)
{
    Matrix r(a.md.rows, b.md.cols);
    push_mat_buff_chunk(&r);        // track buffer use of temporary matrices
    multiply(&r.md, &a.md, &b.md);
    return r;
}

Matrix operator * (float scalar, Matrix a)
{
    Matrix r(a.md.rows, a.md.cols);
    push_mat_buff_chunk(&r);
    mat_scale(&r.md, scalar, &a.md);
    return r;
}

Matrix operator + (Matrix a, Matrix b)
{
    Matrix r(a.md.rows, a.md.cols);
    push_mat_buff_chunk(&r);
    mat_add(&r.md, &a.md, &b.md);
    return r;
}

// BEWARE. strange things will happen when assigning matrix of different size
void Matrix::operator = (Matrix other)
{
    unsigned int n = md.rows * md.cols;
    memcpy(data_buffer + md.base, data_buffer + other.md.base, n * sizeof(float));
    // why do a memcpy instead of just copy matrix_descriptor? because the data
    // of temporary matrices regularly gets trashed to make more buffer space available
}

Matrix relu(Matrix a)   //rectified linear unit
{
    unsigned int i, n;
    float *f, *fa;
    
    Matrix r(a.md.rows, a.md.cols);
    push_mat_buff_chunk(&r);
    f = data_buffer + r.md.base;  fa = data_buffer + a.md.base;
    n = a.syze();
    for (i=0; i<n; i++) {
        if (*fa > 0.0)  *f++ = *fa++;
        else { *f++ = 0.0;  fa++; }
    }
    return r;
}

Matrix sigmoid(Matrix a)  // r[i] = 1 / (1 + exp(-a[i]))
{
    unsigned int i, n;
    float *f, *fa;
       
    Matrix r(a.md.rows, a.md.cols);
    push_mat_buff_chunk(&r);
    f = data_buffer + r.md.base;  fa = data_buffer + a.md.base;
    n = a.syze();
    for (i=0; i<n; i++) {
        if (*fa < -16.0)  *f = 0.0;
        else  *f = 1.0/(1.0 + expf(-*fa));
        f++;  fa++;
    }
    return r;
}

void Matrix::print(void)    // maybe pass in print format
{
    int i, j;
    float *data;
    data = data_buffer + md.base;
    printf("base=%d\n", md.base );
    
    for (i=0; i<md.rows; i++) {
        for (j=0; j<md.cols; j++)  printf("%5.3f ", *data++);
        printf("\n");
    }
    printf("\n");
}

void multiply (matdescriptor_t* mr, matdescriptor_t* ma, matdescriptor_t* mb)
{
   // CFAbsoluteTime t0, t1;
   // t0 = CFAbsoluteTimeGetCurrent();
    
    meta_buffer->mdr = *mr;  meta_buffer->mda = *ma;  meta_buffer->mdb = *mb;
    
    id <MTLCommandBuffer> commandBuffer = [ commandQueue commandBuffer];

    id <MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState: mulFunctionPSO];
    [computeEncoder setBuffer: databuff offset:0 atIndex:0];    // metal code inputs from and outputs to
    [computeEncoder setBuffer: metabuff offset:0 atIndex:1];    // databuff

    MTLSize gridSize = MTLSizeMake(mr->rows, mr->cols, 1);
    NSUInteger w = mulFunctionPSO.threadExecutionWidth;
    NSUInteger h = mulFunctionPSO.maxTotalThreadsPerThreadgroup / w;
   // printf("w = %lu  h = %lu\n", w, h);
    MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);

    [computeEncoder dispatchThreads: gridSize threadsPerThreadgroup: threadsPerThreadgroup];

    [computeEncoder endEncoding];
    
    [commandBuffer commit];
          
    [commandBuffer waitUntilCompleted];
    
   // t1 = CFAbsoluteTimeGetCurrent();
   // printf("multiply time = %lf\n", t1-t0);
}

void mat_add (matdescriptor_t* mr, matdescriptor_t* ma, matdescriptor_t* mb)
{
    //CFAbsoluteTime t0, t1;
    //t0 = CFAbsoluteTimeGetCurrent();
     meta_buffer->mdr = *mr;  meta_buffer->mda = *ma;  meta_buffer->mdb = *mb;
     
    id <MTLCommandBuffer> commandBuffer = [ commandQueue commandBuffer];

    id <MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState: addFunctionPSO];
    [computeEncoder setBuffer: databuff offset:0 atIndex:0];
    [computeEncoder setBuffer: metabuff offset:0 atIndex:1];
    
    NSUInteger g = mr->rows * mr->cols;
    MTLSize gridSize = MTLSizeMake(g, 1, 1);
    NSUInteger threadspergroup = addFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadspergroup > g)  threadspergroup = g;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadspergroup, 1, 1);

    [computeEncoder dispatchThreads: gridSize threadsPerThreadgroup: threadsPerThreadgroup];

    [computeEncoder endEncoding];

    [commandBuffer commit];
       
    [commandBuffer waitUntilCompleted];

    //t1 = CFAbsoluteTimeGetCurrent();
    //printf("mar_add time = %lf\n", t1-t0);
}

void mat_plusequ (matdescriptor_t* mr, matdescriptor_t* ma)
{
    meta_buffer->mdr = *mr;  meta_buffer->mda = *ma;

    id <MTLCommandBuffer> commandBuffer = [ commandQueue commandBuffer];

    id <MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState: plusequFunctionPSO];
    [computeEncoder setBuffer: databuff offset:0 atIndex:0];
    [computeEncoder setBuffer: metabuff offset:0 atIndex:1];
        
    NSUInteger g = mr->rows * mr->cols;
    MTLSize gridSize = MTLSizeMake(g, 1, 1);
    NSUInteger threadspergroup = plusequFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadspergroup > g)  threadspergroup = g;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadspergroup, 1, 1);

    [computeEncoder dispatchThreads: gridSize threadsPerThreadgroup: threadsPerThreadgroup];

    [computeEncoder endEncoding];
    
    [commandBuffer commit];
       
    [commandBuffer waitUntilCompleted];
    
}

void mat_scale(matdescriptor_t* mr, float s, matdescriptor_t* ma)
{
    meta_buffer->mdr = *mr;  meta_buffer->scalar = s;  meta_buffer->mda = *ma;
      
    id <MTLCommandBuffer> commandBuffer = [ commandQueue commandBuffer];

    id <MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState: scaleFunctionPSO];
    [computeEncoder setBuffer: databuff offset:0 atIndex:0];
    [computeEncoder setBuffer: metabuff offset:0 atIndex:1];

    NSUInteger g = mr->rows * mr->cols;
    MTLSize gridSize = MTLSizeMake(g, 1, 1);
    NSUInteger threadspergroup = scaleFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadspergroup > g)  threadspergroup = g;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadspergroup, 1, 1);

    [computeEncoder dispatchThreads: gridSize threadsPerThreadgroup: threadsPerThreadgroup];

    [computeEncoder endEncoding];

    [commandBuffer commit];
       
    [commandBuffer waitUntilCompleted];
    
}

Matrix u(3, 1);          // input triple of floats
Matrix f(3, 1);          // 0 if u[i] not median   1 if u[i] is median
                         // for each u, f is an example of a correct
                         // answer that the neural net can "learn" from

Matrix W(8, 3);          // 1st layer matrix W  } t
Matrix c(8, 1);          // 1st layer bias c    } h
Matrix M(3, 8);          // 2nd layer matrix M  } e
Matrix b(3, 1);          // 2nd layer bias b    } t
                         //                       a
// the elements of W c M b lie in contiguous blocks inside the buffer.
// togther they can be thought of as a single vector theta
// comprising all the parameters that define the current state
// of the neural net.  training the model is a process of iteratively
// adjusting theta (thereby tweaking W c M b) so that fhat better
// approximates the unknown function we are after

Matrix y(8, 1);          // y = W * u + c
Matrix z(8, 1);          // z = relu(y)
Matrix g(3, 1);          // g = M * z + b
                         // fhat = sigmoid(g)
Matrix fhat(3, 1);          // output fhat (neural net estimate of f)

Matrix Jy_th1(8, 32);       // jacobian: [i][j] = dyi_dth1j
Matrix Jz_th1(8, 32);       // jacobian: [i][j] = dzi_dth1j
Matrix Jg_th1(3, 32);       // jacobian: [i][j] = dgi_dth1j
Matrix Jg_th2(3, 27);       // jacobian: [i][j] = dgi_dth2j
Matrix Jg_theta(3, 59);     // jacobian: [i][j] = dgi_dthetaj
Matrix Jfhat_theta(3, 59);  // jacobian: [i][j] = dfhati_dthetaj
Matrix dL_dfhat(1, 3);      // grad L_fhat.  L is loss function
Matrix dL_dtheta(1, 59);    // grad L_theta = dL_dfhat * Jfhat_theta
Matrix delta_theta(1, 59);  // delta_theta = -epsilon * dL_dtheta
                            // the adjustments to apply to the elements of W c M b (the model params)


// th1 = ( W00 W01 W02 W10 W11 W12 ... W70 W71 W72 c0 c1 c2 )
// jacobian dyi_dth1j
void Matrix::y_th1()  // y = W * u + c
{
    unsigned int wr, wc, wrxc, ir, ic, wi, wj, ci;
    
    float *jac = data_buffer + md.base;
    float *uu = data_buffer + u.md.base;
    
    wr = W.md.rows;  wc = W.md.cols;  wrxc = wr * wc;

    for (ir=0; ir<md.rows; ir++) {  // for y[ir]
        
        for (ic=0; ic<wrxc; ic++) {     // for W[wi][wj]
            wi = ic / wc;
            // y[i] depends only on W[i][...] and c[i]
            if (ir == wi) {  wj = ic % wc;  *jac++ = uu[wj]; }
            else *jac++ = 0.0;
        }
        ci=0;
        for (ic=wrxc; ic<md.cols; ic++) {    // for c[ci]
            *jac++ = (float) (ir == ci);
            ci++;
        }
    }
}

// jacobian dzi_dth1j
void Matrix::z_th1()    // z  = relu(y)    y is a function of th1
{
    unsigned int i, k, c = md.cols;
    float *jac = data_buffer + md.base;
    float *jacy = data_buffer + Jy_th1.md.base;
    float *yy = data_buffer + y.md.base;
    
    // zi depends on yj only when i==j
    // when i==j  dzi_dth1k = dyi_dth1k (for all k) when yi > 0
    //            dzi_dth1k = 0                     when yi <= 0
    
    for (i=0; i<md.rows; i++) {
        if (*yy > 0.0) {
            for (k=0; k<c; k++)  *jac++ = *jacy++;
        }
        else {
            for(k=0; k<c; k++)  *jac++ = 0.0;
            jacy += c;
        }
        yy++;
    }
}

// jacobian dgi_dth1j
void Matrix::g_th1()        // g = M * z + b       z is a function of th1
{
    multiply(&md, &M.md, &Jz_th1.md);
}

// th2 = (M00 M01 .. M07 M10 M11 .. M17 M20 M21 .. M27 b0 b1 b2)
// jacobian dgi_dth2j
void Matrix::g_th2()        // g is a function of th2 and z, but z only depends on
{                           // elements of th1 so treat z as constant vector
    unsigned int mr, mc, mrxc, ir, ic, mi, mj, bi;
     
    float *jac = data_buffer + md.base;
    float *zz = data_buffer + z.md.base;
     
    mr = M.md.rows;  mc = M.md.cols;  mrxc = mr * mc;

    for (ir=0; ir<md.rows; ir++) {  // for g[ir]
         
        for (ic=0; ic<mrxc; ic++) {     // for M[mi][mj]
            mi = ic / mc;
                     // g[i] depends only on M[i][...], z and b[i]
            if (ir == mi) {  mj = ic % mc;  *jac++ = zz[mj]; }
            else *jac++ = 0.0;
        }
        bi=0;
        for (ic=mrxc; ic<md.cols; ic++) {    // for b[bi]
            *jac++ = (float) (ir == bi);
            bi++;
        }
    }
}

// jacobian:  row-wise splice of Jg_th1 and Jg_th2
void Matrix::g_theta()
{
    unsigned int i, c1 = Jg_th1.md.cols, c2 = Jg_th2.md.cols;
    float *jac = data_buffer + md.base;
    float *j1 = data_buffer + Jg_th1.md.base;
    float *j2 = data_buffer + Jg_th2.md.base;
    
    for (i=0; i<md.rows; i++) {
        memcpy(jac, j1, c1*sizeof(float));
        jac += c1;  j1 += c1;
        memcpy(jac, j2, c2*sizeof(float));
        jac += c2;  j2 += c2;
    }
}

// jacobian  dfhati_dthetaj
void Matrix::fhat_theta()   // fhat = sigmoid(g)
{
    unsigned int i, j, c = md.cols;
    float *jac = data_buffer + md.base;
    float *jacg = data_buffer + Jg_theta.md.base;
    float *gg = data_buffer + g.md.base;
    float w, sigprime;
    
    // fhat[i] = sigmoid(g[i](theta))
    // dfhat[i]_dtheta[j]  = g[i]' * sigmoid'(g[i](theta))
    //                     = Jg_theta[i][j] * u/(1+u)^2  where u = exp(-g[i])
    for (i=0; i<md.rows; i++) {
        if (gg[i] < -16.0)  sigprime = 0.0;
        else {
            w = expf(-gg[i]);
            sigprime = w / ((1.0 + w)*(1.0 + w));
        }
        for (j=0; j<c; j++) {
            *jac++ = *jacg * sigprime;  jacg++;
        }
    }
}

// grad of L with respect to fhat.  L = sum((fhat[i] - f[i])^2) / 3
void Matrix::L_fhat()
{
    unsigned int i;
    float* gl = data_buffer + md.base;
    float* fh = data_buffer + fhat.md.base;
    float* ff = data_buffer + f.md.base;
    float n = (float) md.cols;
    
    for (i=0; i<md.cols; i++) {
        *gl++ = 2.0 * (*fh - *ff) / n;  fh++;  ff++;
    }
}

// grad of L with respect to fhat when loss function is binary cross-entropy loss
// L = -1/n sum(f[i]*log(fhat[i]) + (1-f[i])*log(1-fhat[i]))
// try epsilon=0.01 if using binary cross-entropy loss
void Matrix::L_fhat_binary_cross_entropy()
{
    unsigned int i;
    float* gl = data_buffer + md.base;
    float* fh = data_buffer + fhat.md.base;
    float* ff = data_buffer + f.md.base;
    float n = (float) md.cols;
    float fhati;
    
    for (i=0; i<md.cols; i++) {
        fhati = *fh;
        if (fhati < 0.0001)  fhati=0.0001;
        if (fhati > 0.9999)  fhati=0.9999;
        *gl++ = ((1.0-*ff)/(1.0-fhati) - (*ff)/fhati) / n;  fh++;  ff++;
    }
}

// grad L with respect to theta
void Matrix::L_theta()
{
    multiply(&md, &dL_dfhat.md, &Jfhat_theta.md);
}

float bigl(void)    // loss function ( sum of (fhat[i] - f[i])^2) / 3
{                   // TODO generalize for fhat[0..n]
    float df0, df1, df2, bl;
    float *fdat = data_buffer + f.md.base;
    float *fhatdat = data_buffer + fhat.md.base;
    
    df0 = *fhatdat - *fdat;  fdat++;  fhatdat++;
    df1 = *fhatdat - *fdat;  fdat++;  fhatdat++;
    df2 = *fhatdat - *fdat;
    
    bl = (df0*df0 + df1*df1 + df2*df2) / 3.0f;
    
    return bl;
}

void update_theta()
{
    matdescriptor_t theta_md;
    // faking a vector theta that is the concatenation of W c M b
    // already living in the buffer.  this is a bit dirty because
    // the "control" thet a Matrix object has of itself is being bypassed
    theta_md.rows = 1;
    theta_md.cols = W.syze() + c.syze() + M.syze() + b.syze();
    theta_md.base = W.md.base;
    mat_plusequ(&theta_md, &delta_theta.md);    // theta += delta_theta
    // now have modified neural net with W c M b hopefully closer to where they should be
}

void setup(void)
{  // fill W, c, M, b with random values.  then add train() to runloop
    W.fill_random(0.0, 1.0);
    c.fill_random(0.5, 1.0);
    M.fill_random(0.0, 2.0);
    b.fill_random(0.5, 1.0);
    
    itrain = numpoints = 0;
    validate();

    // set up timer to fire in 5 millisecs, calling train()
    mainrunloop = CFRunLoopGetMain();
    CFAbsoluteTime firetime = CFAbsoluteTimeGetCurrent() + 0.005;

    CFRunLoopTimerRef timer = CFRunLoopTimerCreate(kCFAllocatorDefault, firetime, ONE_YEAR, 0, 0, timer_callback, &timer_context);

    CFRunLoopAddTimer(mainrunloop, timer, kCFRunLoopDefaultMode);

}

// train() is called by a runloop-timer.
// after processing the sample it resets the timer
// so train() is called again and again in a loop

void train(CFRunLoopTimerRef timer, void* info)
{
    // get sampledata[itrain]  feed into neural net
    // compute derivatives of loss with respact to weights (dL_dtheta)
    // adjust weights by gradient descent:  theta = theta - epsilon * dL_dtheta
    int i;
    
    float ff[3], epsilon= 0.1;
    sample_t* samp = data + itrain;
    CFAbsoluteTime firetime;
    
    u.init_data(samp->dat);
    for (i=0; i<3; i++)  ff[i] = (float) samp->med[i];
    f.init_data(ff);
    
    y = W * u + c;          // nice notation in c++
    z = relu(y);
    g = M * z + b;
    fhat = sigmoid(g);      // neural net output: estimate of f()

    Jy_th1.y_th1();         // see notes. calculate gradient of loss function
    Jz_th1.z_th1();
    Jg_th1.g_th1();         // = M * Jz_th1
    Jg_th2.g_th2();
    Jg_theta.g_theta();
    Jfhat_theta.fhat_theta();
    dL_dfhat.L_fhat();      // alternatively dL_dfhat.L_fhat_binary_cross_entropy()
    dL_dtheta.L_theta();    // = dL_dfhat * Jfhat_theta;
                            // doing multiply inside function saves a memcpy
    delta_theta = -epsilon * dL_dtheta;

    update_theta();  // adjust W c M b by adding corresponding element of delta_theta
    recycle_buffer_space();
    
    itrain++;
    
    if (itrain % 500 == 0) {
        printf("itrain = %d\n", itrain);
        //cprint();
        validate();
    }
    if (itrain > 15000) {
        CFRunLoopTimerInvalidate(timer);
        CFRunLoopRemoveTimer(mainrunloop, timer, kCFRunLoopDefaultMode);
        //validate();
        W.print();
        c.print();
        M.print();
        //z.print();
        b.print();
        [ madre finish ];
    }
    else {  // schedule next train in 3ms to allow runloop do other stuff
        firetime = CFAbsoluteTimeGetCurrent() + 0.003;
        CFRunLoopTimerSetNextFireDate(timer, firetime);
    }
}

void validate(void)     // check how well model performs on untrained samples
{
    int i, j, vfirst = NSAMPLES - 100, correct=0;
    float ff[3], *fh, loss=0.0f;
    bool fhatb[3], wrong;
    sample_t* samp;
    
    for (i=vfirst; i<NSAMPLES; i++) {
        samp = data + i;
        u.init_data(samp->dat);
        for (j=0; j<3; j++)  ff[j] = (float) samp->med[j];
        f.init_data(ff);
        y = W * u + c;
        z = relu(y);
        g = M * z + b;
        fhat = sigmoid(g);
        
        fh = data_buffer + fhat.md.base;        // convert fhat to bool
        fhatb[0] = (*fh > 0.5f);  fh++;
        fhatb[1] = (*fh > 0.5f);  fh++;
        fhatb[2] = (*fh > 0.5f);
        recycle_buffer_space();     // clean up buffer after matrix math fragmented it
        //printf("f^ %d %d %d    f %d %d %d\n",
        //       fhatb[0], fhatb[1], fhatb[2],
        //       samp->med[0], samp->med[1], samp->med[2]);
        // wrong if any output disagrees with f
        wrong = (samp->med[0] ^ fhatb[0]) | (samp->med[1] ^ fhatb[1]) | (samp->med[2] ^ fhatb[2]);
        correct += !wrong;
        loss += bigl();
    }
    printf("validate:  correct = %d  meanloss = %f\n", correct, loss/100.0f);
    mse[numpoints] = loss/100.0f;
    crrct[numpoints++] = correct;
    [ madre update_graph ];
}

float uu[] = { 5, 3, 2 };

float ww[] = { 2,  3,  5,
               7, 11, 13,
              17, 19, 23 };

float cc[] = { 100, 200, 400 };


void matrixtest(void)
{
    Matrix Q(3,3), T(3,8);
    push_mat_buff_chunk(&Q);
    push_mat_buff_chunk(&T);
    
    u.init_data(uu);
    Q.init_data(ww);
    
    c.init_data(cc);
    
    y = Q * u + c;
    y.print();
    
    M.fill_random(2.0, 0.5);
    M.print();
    T = 3 * M;
    T.print();
    
    recycle_buffer_space();
    mat_plusequ(&u.md, &c.md);
    u.print();
}

// this is a wrapper that allows c++ code to be called from OBJC
void (*funcptr)(void) = setup;  // substitute setup with matrixtest
                                // to test metal code
