//
//  ViewController.h
//  neuron

#import <Cocoa/Cocoa.h>
#import "zview.h"
#import "proto.h"

@interface ViewController : NSViewController <graph>

@property (strong) IBOutlet zview *graphics_view;

@property (weak) IBOutlet NSButton *go_butt;
- (IBAction)go:(id)sender;

-(void) update_graph;
-(void) finish;
@end

