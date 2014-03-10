#include "median.h"
#define CHANNEL_COUNT 3
#define MAX_RADIUS 7

void sort(unsigned char * arr, int length)
{
  int i, j;
  int iPad = length / 2 + 1;

  for (i = 0; i < iPad; ++i)
    for(j = 0; j < length - i - 1; ++j)
    {
      if (arr[j] > arr[j + 1])
      {
        unsigned char tmp;
        tmp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = tmp;
      }
    }
}

void c_median_filter(const unsigned char * src, unsigned char * dst, int channelStride, int h, int matrix_radius)
{
  const unsigned char MAX_CHANNEL_VALUE = -1;

  unsigned char * srcIterator = (unsigned char *)src;
  unsigned char * dstIterator = (unsigned char *)dst;

  unsigned char matrix[MAX_RADIUS * MAX_RADIUS];

  long pixelCount = channelStride * h;
  int half_matrix_radius = matrix_radius / 2;

  int i, j = 0;

  const unsigned char * SRC_ITERATOR_END = srcIterator + pixelCount;
  int lenght = matrix_radius * matrix_radius;

  if (matrix_radius > MAX_RADIUS)
    matrix_radius = MAX_RADIUS;

  for (; srcIterator != SRC_ITERATOR_END; ++srcIterator, ++dstIterator)
  {
    long offside = (srcIterator - src) % (channelStride);

    if (offside < half_matrix_radius * CHANNEL_COUNT || offside >= (channelStride - half_matrix_radius * CHANNEL_COUNT) || (srcIterator - src) < half_matrix_radius * channelStride || (SRC_ITERATOR_END - srcIterator) < half_matrix_radius * channelStride)//If we on border nothing to do
    {
      *dstIterator = *srcIterator;
      continue;
    }


    for (i = -half_matrix_radius; i <= half_matrix_radius; ++i)
      for (j = -half_matrix_radius; j <= half_matrix_radius; ++j)
      {
        matrix[(i + half_matrix_radius) *matrix_radius + j + half_matrix_radius] = *(srcIterator + i * channelStride + j * CHANNEL_COUNT);
      }

    
    sort(matrix, lenght);

    *dstIterator = matrix[lenght / 2];
  }
}