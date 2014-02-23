#include "imagedecoding.h"

void GetGrayImage( const unsigned char * srcImageRGBA, unsigned char * dstImage, int w, int h, int depth, int x, int y, int x1, int y1)
{
	unsigned char * src = (unsigned char*)srcImageRGBA;
	unsigned char * dst = dstImage;

  if (x1 == 0)
    x1 = w;
  if (y1 == 0)
    y1 = h;

  int partRow = (x1 - x);

	int pixelNum = w * h;

	int i, j;
  for (j = 0; j < y1 - y; j++)
  {
    src = (unsigned char*)srcImageRGBA + ((y + j) * w + x) * (depth >> 3);
	  for (i = 0; i < partRow; i++)
	  {
	  	*dst = (*src + *(src + 1) + *(src + 2)) / 3;
	  	src += (depth >> 3);
	  	dst++;
	  }
  }
}

void BuildRGBAFromGrayImage( const unsigned char * srcImage, char unsigned * dstImageRGBA, int w, int h, int depth, int x, int y, int x1, int y1)
{
	const unsigned char * src = srcImage;
	unsigned char * dst = dstImageRGBA;

  if (x1 == 0)
    x1 = w;
  if (y1 == 0)
    y1 = h;

  int partRow = (x1 - x);

	int pixelNum = w * h;

	int i, j, k;

  for (k = 0; k < (y1- y); k++)
  {
    dst = dstImageRGBA + ((y + k) * w + x) * (depth >> 3);
	  for (i = 0; i < partRow; i++)
	  {
      for (j = 0; j < (depth >> 3); j++)
		    *(dst++) = *(src);
		  src++;
	  }
  }
}

