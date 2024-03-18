//
//  ViewController.m
//  neuron
//
// supervised machine learning demonstration.
// the task is to identify the median of three numbers in the range
// 0 to 9.  it is a silly task to do by machine learning but
// the code/methodology here is adaptable to other problems with
// m inputs and n outputs for reasonable m and n.
// because the training is driven by the runloop it is possible
// to do other tasks in "parallel" such as monitoring the progress
// of the model's evolution.
// the model does not do so well when the range of input numbers is larger.
// the reason for this is that the median-of-three function is choppy in
// regions where two or three of the inputs are close to each other.
// when the input range is large there are many such bumpy regions so a more
// complex neural net and/or more training would be required.
// also a big range of inputs results in less overlap between the
// training data and the validation set.

#import "ViewController.h"
@import Metal;

#include "matrix.hpp"
#include "make_data.h"

extern float* data_buffer;
extern metadata_t* meta_buffer;
extern NSViewController <graph> * madre;

extern void (*funcptr)(void);

id <MTLDevice> device;
id <MTLCommandQueue> commandQueue;
id <MTLLibrary> defaultLibrary;
id <MTLFunction> mulFunction;
id <MTLComputePipelineState> mulFunctionPSO;
id <MTLFunction> addFunction;
id <MTLComputePipelineState> addFunctionPSO;
id <MTLFunction> plusequFunction;
id <MTLComputePipelineState> plusequFunctionPSO;
id <MTLFunction> scaleFunction;
id <MTLComputePipelineState> scaleFunctionPSO;

id <MTLBuffer> databuff;    // where most of the matrix maths happens
id <MTLBuffer> metabuff;

@implementation ViewController

- (void)viewDidLoad {
    NSError* error = nil;
    [super viewDidLoad];

    // Do any additional setup after loading the view.
    srandom(CFAbsoluteTimeGetCurrent());
    madre = self;
    
    [ zview create_font ];
    
    device = MTLCreateSystemDefaultDevice();
    commandQueue = [ device newCommandQueue ];
    
    defaultLibrary = [ device newDefaultLibrary ];
    if (defaultLibrary == nil)  NSLog(@"Failed to find the default library.");
    
    mulFunction = [defaultLibrary newFunctionWithName:@"matrix_multiply"];
    if (mulFunction == nil)  NSLog(@"Failed to find matrix_multiply function.");

    mulFunctionPSO = [device newComputePipelineStateWithFunction: mulFunction error: &error];
    
    addFunction = [defaultLibrary newFunctionWithName:@"vector_add"];
    if (addFunction == nil)  NSLog(@"Failed to find vector_add function.");

    addFunctionPSO = [device newComputePipelineStateWithFunction: addFunction error: &error];
    
    plusequFunction = [defaultLibrary newFunctionWithName:@"vector_plusequ"];
    if (plusequFunction == nil)  NSLog(@"Failed to find vector_plusequ function.");

    plusequFunctionPSO = [device newComputePipelineStateWithFunction: plusequFunction error: &error];
    
    scaleFunction = [defaultLibrary newFunctionWithName:@"vector_scale"];
    if (scaleFunction == nil)  NSLog(@"Failed to find vector_scale function.");

    scaleFunctionPSO = [device newComputePipelineStateWithFunction: scaleFunction error: &error];
    
    commandQueue = [device newCommandQueue];
    
    // might need bigger buffer.  if so, fix BUFFLEN in matrix.mm
    databuff = [ device newBufferWithLength: 2048 * sizeof(float) options: MTLResourceStorageModeShared ];
    data_buffer = databuff.contents;
    metabuff = [ device newBufferWithLength: sizeof(metadata_t) options: MTLResourceStorageModeShared ];
    meta_buffer = metabuff.contents;

    make_data();
}


- (IBAction)go:(id)sender
{
    self.go_butt.hidden = true;
    funcptr();      //  call setup(); in matrix.mm
}

-(void) update_graph
{
    self.graphics_view.needsDisplay = true;
}

-(void) finish
{
    make_data();
    self.go_butt.hidden = false;
}

- (void)setRepresentedObject:(id)representedObject {
    [super setRepresentedObject:representedObject];

    // Update the view, if already loaded.
}

@end
