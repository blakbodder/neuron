//
//  zview.h
//  neuron

#ifndef zview_h
#define zview_h
#import <Cocoa/Cocoa.h>

@interface zview : NSView
+(void) create_font;
-(void) drawRect: (NSRect) dirtyrect;
@end

#endif /* zview_h */
