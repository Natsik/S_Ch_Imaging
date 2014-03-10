#define CHANNEL_COUNT 3
#define MAX_MEDIAN_RADIUS 7

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*linear filter*/
void c_linear_filter(const unsigned char * src, unsigned char * dst, int channelStride, int h, int * matrix, int matrix_dimension, int divisor)
{
  const unsigned char MAX_CHANNEL_VALUE = -1;

  unsigned char * srcIterator = (unsigned char *)src;
  unsigned char * dstIterator = (unsigned char *)dst;

  long pixelCount = channelStride * h;
  int half_matrix_dimension = matrix_dimension / 2;

  int i, j = 0;

  const unsigned char * SRC_ITERATOR_END = srcIterator + pixelCount;

  for (; srcIterator != SRC_ITERATOR_END; ++srcIterator, ++dstIterator)
  {
    long offside = (srcIterator - src) % (channelStride);

    if (offside < half_matrix_dimension * CHANNEL_COUNT || offside >= (channelStride - half_matrix_dimension * CHANNEL_COUNT) || (srcIterator - src) < half_matrix_dimension * channelStride || (SRC_ITERATOR_END - srcIterator) < half_matrix_dimension * channelStride)//If we on border nothing to do
    {
      *dstIterator = *srcIterator;
      continue;
    }

    int accumulator = 0;
    int * matrix_start = matrix;

    for (i = -half_matrix_dimension; i <= half_matrix_dimension; ++i)
      for (j = -half_matrix_dimension; j <= half_matrix_dimension; ++j)
      {
        accumulator += (*(srcIterator + i * channelStride + j * CHANNEL_COUNT)) * (*(matrix_start++));
      }

    *dstIterator = MIN(MAX((accumulator / divisor), 0), MAX_CHANNEL_VALUE);  
  }
}

/*noize white/black*/
void c_white_noise(const unsigned char * src, unsigned char * dst, int channelStride, int h, int p, int d)
{
  const unsigned char MAX_CHANNEL_VALUE = -1;

  unsigned char * srcIterator = (unsigned char *)src;
  unsigned char * dstIterator = (unsigned char *)dst;

  long pixelCount = channelStride * h;

  const unsigned char * SRC_ITERATOR_END = srcIterator + pixelCount;

  for (;srcIterator != SRC_ITERATOR_END; srcIterator += 3, dstIterator += 3)
  {
    int genered_d;
    if (rand() % 100 <= p)
    {
      genered_d = (rand() % (2 * d)) - d;
    }
    else
    {
      genered_d = 0;
    }

    *(dstIterator + 0) = *(srcIterator + 0) + genered_d;
    *(dstIterator + 1) = *(srcIterator + 1) + genered_d;
    *(dstIterator + 2) = *(srcIterator + 2) + genered_d;
  }
}

void c_bil(const unsigned char * src, unsigned char * dst, int channelStride, int h, int p, int min)
{
  const unsigned char MAX_CHANNEL_VALUE = -1;

  unsigned char * srcIterator = (unsigned char *)src;
  unsigned char * dstIterator = (unsigned char *)dst;

  long pixelCount = channelStride * h;

  const unsigned char * SRC_ITERATOR_END = srcIterator + pixelCount;

  int i = 0;

  for (;srcIterator != SRC_ITERATOR_END; srcIterator += 3, dstIterator += 3)
  {
    int genered_d[3];
    if (rand() % 100 <= p)
    {
      genered_d[0] = (rand() % (MAX_CHANNEL_VALUE - min)) + min;
      genered_d[1] = genered_d[0];
      genered_d[2] = genered_d[0];
    }
    else
    {
      genered_d[0] = *(srcIterator + 0);
      genered_d[1] = *(srcIterator + 1);
      genered_d[2] = *(srcIterator + 2);
    }

    for (i = 0; i < CHANNEL_COUNT; ++i)
    {
      *(dstIterator + i) = genered_d[i];
    }
  }
}

void c_mesh(const unsigned char * src, unsigned char * dst, int channelStride, int h, int mesh_w, int mesh_h, unsigned char tone)
{
  const unsigned char MAX_CHANNEL_VALUE = -1;

  unsigned char * srcIterator = (unsigned char *)src;
  unsigned char * dstIterator = (unsigned char *)dst;

  long pixelCount = channelStride * h;

  const unsigned char * SRC_ITERATOR_END = srcIterator + pixelCount;

  int i = 0;
  int w_accum = 0, h_accum = 0;

  for (;srcIterator != SRC_ITERATOR_END; srcIterator += 3, dstIterator += 3)
  {
    w_accum++;
    if (w_accum >= channelStride / CHANNEL_COUNT)
    {
      w_accum = 0;
      h_accum++;
    }


    if (!(w_accum % mesh_w) || !(h_accum % mesh_h))
    {
      *(dstIterator + 0) = tone;
      *(dstIterator + 1) = tone;
      *(dstIterator + 2) = tone;
    }
    else
    {
      *(dstIterator + 0) = *(srcIterator + 0);
      *(dstIterator + 1) = *(srcIterator + 1);
      *(dstIterator + 2) = *(srcIterator + 2);
    }
  }
}

double c_diff_images(const unsigned char * src1, unsigned char * src2, int channelStride, int h)
{
  const unsigned char MAX_CHANNEL_VALUE = -1;

  unsigned char * srcIterator1 = (unsigned char *)src1;
  unsigned char * srcIterator2 = (unsigned char *)src2;

  long pixelCount = channelStride * h;

  long long result = 0;

  const unsigned char * SRC_ITERATOR_END = srcIterator1 + pixelCount;

  for (;srcIterator1 != SRC_ITERATOR_END; ++srcIterator1, ++srcIterator2)
  {
    result += (*srcIterator1 - *srcIterator2);
  }

  return (double)result / (double)(pixelCount * MAX_CHANNEL_VALUE);
}