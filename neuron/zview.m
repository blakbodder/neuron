//
//  zview.m
//  neuron

#include "zview.h"
#import <Foundation/Foundation.h>

extern int numpoints;   // defined in matrix.mm
extern int crrct[33];
extern float mse[33];

CGColorRef cgwhite, cgblue, cggreen;
CTFontRef bigfont, smallfont;
CFDictionaryRef attribs1, attribs2, attribs3, attribs4;

@implementation zview

+(void) create_font
{
    cgwhite = CGColorCreateSRGB(1.0, 1.0, 1.0, 1.0);
    cgblue = CGColorCreateSRGB(0.4, 0.4, 1.0, 1.0);
    cggreen = CGColorCreateSRGB(0.0, 1.0, 0.0, 1.0);
    CFStringRef fontname = CFSTR("System");
    CGAffineTransform fmat = { 1.0, 0.0, 0.0, 1.0, 0.0, 0.0 };

    bigfont = CTFontCreateWithName(fontname, 18.0, &fmat);
    smallfont = CTFontCreateWithName(fontname, 12.0, &fmat);
    CFRelease(fontname);
    
    CFStringRef keys[] = { kCTFontAttributeName, kCTForegroundColorAttributeName };
    CFTypeRef values[] = { bigfont, cgblue };
    
    attribs1 = CFDictionaryCreate(kCFAllocatorDefault, (const void**)&keys,
                                  (const void**)&values, 2,
                                  &kCFTypeDictionaryKeyCallBacks,
                                  &kCFTypeDictionaryValueCallBacks);
    
    CFTypeRef values2[] = { bigfont, cggreen };
    attribs2 = CFDictionaryCreate(kCFAllocatorDefault, (const void**)&keys,
                                  (const void**)&values2, 2,
                                  &kCFTypeDictionaryKeyCallBacks,
                                  &kCFTypeDictionaryValueCallBacks);
    
    CFTypeRef values3[] = { smallfont, cgblue };
    attribs3 = CFDictionaryCreate(kCFAllocatorDefault, (const void**)&keys,
                                  (const void**)&values3, 2,
                                  &kCFTypeDictionaryKeyCallBacks,
                                  &kCFTypeDictionaryValueCallBacks);
    
    CFTypeRef values4[] = { smallfont, cggreen };
    attribs4 = CFDictionaryCreate(kCFAllocatorDefault, (const void**)&keys,
                                  (const void**)&values4, 2,
                                  &kCFTypeDictionaryKeyCallBacks,
                                  &kCFTypeDictionaryValueCallBacks);
}

-(void) clear: (CGContextRef) context : (NSRect) box
{
    float x, y, w, h;
    
    CGContextSetRGBFillColor(context, 0.0, 0.0, 0.0, 1.0);
    x = box.origin.x;  y = box.origin.y;
    w = box.size.width;  h = box.size.height;
    CGContextMoveToPoint(context, x, y);
    x+=w;  CGContextAddLineToPoint(context, x, y);
    y+=h;  CGContextAddLineToPoint(context, x, y);
    x-=w;  CGContextAddLineToPoint(context, x, y);
    CGContextClosePath(context);
    CGContextFillPath(context);
}


-(void) axes : (CGContextRef) context
{
    CFStringRef msestr, accstr, numstr;
    CFAttributedStringRef attrstr, attrstr2, attrstr3, attrstr4;
    CTLineRef line, line2;
    //float x, y;
    char mse[] = "mean squared error";
    char accuracy[] = "acccuracy";
    CGAffineTransform fmat = { 0.0, 1.0, -1.0, 0.0, 0.0, 0.0 }; // rotate pi/2
    int i;
    char num[5];
    float e;
    
    CGContextSaveGState(context);
    msestr = CFStringCreateWithCString(kCFAllocatorDefault, mse, kCFStringEncodingASCII);
    attrstr = CFAttributedStringCreate(kCFAllocatorDefault, msestr, attribs1);
    line = CTLineCreateWithAttributedString(attrstr);
    accstr = CFStringCreateWithCString(kCFAllocatorDefault, accuracy, kCFStringEncodingASCII);
    attrstr2 = CFAttributedStringCreate(kCFAllocatorDefault, accstr, attribs2);
    line2 = CTLineCreateWithAttributedString(attrstr2);
    
    CGContextSaveGState(context);
    CGContextConcatCTM(context, fmat);              // y axis points left. x axis points up
    CGContextSetTextPosition(context, 120, -20);    // coords inside view look weird
    CTLineDraw(line, context);
    CGContextSetTextPosition(context, 160, -620);
    CTLineDraw(line2, context);
    
    CFRelease(line2);
    CFRelease(attrstr2);
    CFRelease(accstr);
    CFRelease(line);
    CFRelease(attrstr);
    CFRelease(msestr);
    CGContextRestoreGState(context);
    for (i=0; i<105; i+=10) {
        sprintf(num, "%d", i);
        numstr = CFStringCreateWithCString(kCFAllocatorDefault, num, kCFStringEncodingASCII);
        attrstr4 = CFAttributedStringCreate(kCFAllocatorDefault, numstr, attribs4);
        line = CTLineCreateWithAttributedString(attrstr4);
        CGContextSetTextPosition(context, 580.0, 4.0*i + 14.0);
        CTLineDraw(line, context);
        CFRelease(line);
        CFRelease(attrstr4);
        CFRelease(numstr);
    }
    for (e=0.0; e<0.401; e+= 0.05) {
        sprintf(num, "%4.2f", e);
        numstr = CFStringCreateWithCString(kCFAllocatorDefault, num, kCFStringEncodingASCII);
        attrstr3 = CFAttributedStringCreate(kCFAllocatorDefault, numstr, attribs3);
        line = CTLineCreateWithAttributedString(attrstr3);
        CGContextSetTextPosition(context, 30.0, 1000.0*e + 14.0);
        CTLineDraw(line, context);
        CFRelease(line);
        CFRelease(attrstr3);
        CFRelease(numstr);
    }
    CGContextRestoreGState(context);
}

-(void) drawRect: (NSRect) dirtyrect
{
    CGContextRef context = [[ NSGraphicsContext currentContext ] CGContext ];
    int i;
    float x;
    NSRect box = dirtyrect;
    box.size.width = 640;
    
    [ self clear: context : box];
    [ self axes: context ];
    if (numpoints>1) {
        CGContextSetRGBStrokeColor(context, 0.0, 1.0, 0.0, 1.0);
        CGContextSetLineWidth(context, 2.0f);
        x = 60.0;
        CGContextMoveToPoint(context, x, 4.0*crrct[0]+20.0);
        x+= 17.0;
        for (i=1; i<numpoints; i++)  {
            CGContextAddLineToPoint(context, x, 4.0*crrct[i]+20.0);
            x += 17;
        }
        CGContextStrokePath(context);
        CGContextSetRGBStrokeColor(context, 0.4, 0.4, 1.0, 1.0);
        x = 60.0;
        CGContextMoveToPoint(context, x, 1000.0*mse[0]+20.0);
        x+= 17.0;
        for (i=1; i<numpoints; i++)  {
            CGContextAddLineToPoint(context, x, 1000.0*mse[i]+20.0);
            x += 17;
        }
        CGContextStrokePath(context);
    }
}

@end
