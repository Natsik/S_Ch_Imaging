#ifndef _IMAGE_DECODING_H_
#define _IMAGE_DECODING_H_

void GetGrayImage(const unsigned char * srcImageRGBA, unsigned char * dstImage, int w, int h, int depth, int x = 0, int y = 0, int x1 = 0, int y1 = 0);
void BuildRGBAFromGrayImage(const unsigned char * srcImage, char unsigned * dstImageRGBA, int w, int h, int depth, int x = 0, int y = 0, int x1 = 0, int y1 = 0);

#endif//_IMAGE_DECODING_H_