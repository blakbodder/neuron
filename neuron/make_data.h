//
//  make_data.h
//  neuron

#ifndef make_data_h
#define make_data_h
#include <stdbool.h>

#define NSAMPLES 16000

struct sample {
    float dat[3];
    bool med[3];    //  if dat[i] is median of dat[0..3]  then med[i] is true
};

typedef struct sample sample_t;

void make_data(void);
void dump_vdat(void);

#endif /* make_data_h */
