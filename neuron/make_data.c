//
//  make_data.c
//  neuron

#include <stdio.h>
#include <stdlib.h>
#include "make_data.h"

sample_t data[NSAMPLES];

void make_data(void)
{
    int i, d[3], dsort[3], imedian, k;
    sample_t* samp;
    
    for (i=0; i<NSAMPLES; i++) {
        d[0] = random() % 10L;  d[1] = random() % 10L;  d[2] = random() % 10L;
        samp = data+i;
        samp->dat[0] = (float) d[0];  samp->dat[1] = (float) d[1];  samp->dat[2] = (float) d[2];
        
        if (d[1] < d[0])  {
            if (d[2] < d[1]) {
                dsort[0] = d[2];  dsort[1]= d[1];  dsort[2] = d[0];     // 2 1 0
                imedian = 1;
            }
            else {          // 1 2 0 or 1 0 2
                if (d[2] < d[0]) {
                    dsort[0] = d[1];  dsort[1] = d[2];  dsort[2] = d[0];    // 1 2 0
                    imedian = 2;
                }
                else {
                    dsort[0] = d[1];  dsort[1] = d[0];  dsort[2] = d[2];    // 1 0 2
                    imedian = 0;
                }
            }
        }
        else {
            if (d[2] < d[0]) {
                dsort[0] = d[2];  dsort[1] = d[0];  dsort[2] = d[1];    // 2 0 1
                imedian = 0;
            }
            else {  // 0 1 2 or 0 2 1
                if (d[1] < d[2])  {
                    dsort[0] = d[0];  dsort[1] = d[1];  dsort[2] = d[2];    // 0 1 2
                    imedian = 1;
                }
                else  {
                    dsort[0] = d[0];  dsort[1] = d[2] ;  dsort[2] = d[1];   // 0 2 1
                    imedian = 2;
                }
            }
        }
        samp->med[imedian] = true;
        k = (imedian + 1) % 3;          // if tied values then both or all 3 can be median
        samp->med[k] = (d[imedian] == d[k]);
        k = (k+1) % 3;
        samp->med[k] = (d[imedian] == d[k]);
       // printf("%2d %2d %2d\n", dsort[0], dsort[1], dsort[2] );
       // printf("%2d %2d %2d     %d %d %d\n", d[0], d[1], d[2], samp->med[0], samp->med[1], samp->med[2]);
    }
}
