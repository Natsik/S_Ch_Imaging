#include <stdio.h>

void c_inversion(const unsigned char * src, unsigned char * dst, int w, int h)
{
  const unsigned char MAX_CHANNEL_VALUE = -1;

  unsigned char * srcIterator = (unsigned char *)src;
  unsigned char * dstIterator = (unsigned char *)dst;

  long pixelCount = w * h;

  const unsigned char * SRC_ITERATOR_END = srcIterator + pixelCount;

  for (;srcIterator != SRC_ITERATOR_END; ++srcIterator, ++dstIterator)
  {
    *dstIterator = MAX_CHANNEL_VALUE - *srcIterator;
  }
}


/*Takes RGB packed images (3byte per pixel)*/
void c_dilatation(const unsigned char * src, unsigned char * dst, int channelStride, int h)
{
  const unsigned char MAX_CHANNEL_VALUE = -1;
  const long CHANNEL_COUNT = 3;

  unsigned char * srcIterator = (unsigned char *)src;
  unsigned char * dstIterator = (unsigned char *)dst;

  long pixelCount = channelStride * h;

  int i, j = 0;

  const unsigned char * SRC_ITERATOR_END = srcIterator + pixelCount;

  for (; srcIterator != SRC_ITERATOR_END; ++srcIterator, ++dstIterator)
  {
    long offside = (srcIterator - src) % (channelStride);

    if (offside <=2 || offside >= (channelStride - 3) || (srcIterator - src) < channelStride || (SRC_ITERATOR_END - srcIterator) < channelStride)//If we on border nothing to do
    {
      *dstIterator = *srcIterator;
      continue;
    }

    int max = 0;
    for (i = -1; i <= 1; ++i)
      for (j = -1; j <= 1; ++j)
      {
        if ((*(srcIterator + i * channelStride + j * CHANNEL_COUNT)) > max)
        {
          max = (*(srcIterator + i * channelStride + j * CHANNEL_COUNT));
        }
      }
    *dstIterator = max;
  }
}

/*Takes RGB packed images (3byte per pixel)*/
void c_errosion(const unsigned char * src, unsigned char * dst, int channelStride, int h)
{
  const unsigned char MAX_CHANNEL_VALUE = -1;
  const long CHANNEL_COUNT = 3;

  unsigned char * srcIterator = (unsigned char *)src;
  unsigned char * dstIterator = (unsigned char *)dst;

  long pixelCount = channelStride * h;

  int i, j = 0;

  const unsigned char * SRC_ITERATOR_END = srcIterator + pixelCount;

  for (; srcIterator != SRC_ITERATOR_END; ++srcIterator, ++dstIterator)
  {
    long offside = (srcIterator - src) % (channelStride);

    if (offside <=2 || offside >= (channelStride - 3) || (srcIterator - src) < channelStride || (SRC_ITERATOR_END - srcIterator) < channelStride)//If we on border nothing to do
    {
      *dstIterator = *srcIterator;
      continue;
    }

    int min = MAX_CHANNEL_VALUE;
    for (i = -1; i <= 1; ++i)
      for (j = -1; j <= 1; ++j)
      {
        if ((*(srcIterator + i * channelStride + j * CHANNEL_COUNT)) < min)
        {
          min = (*(srcIterator + i * channelStride + j * CHANNEL_COUNT));
        }
      }
      *dstIterator = min;
  }
}