#define CHANNEL_COUNT 3
#define MAX_MEDIAN_RADIUS 7

#define ABS(a) ((a) > 0 ? (a) : -(a))
double c_diff_images(const unsigned char * src1, const unsigned char * src2, unsigned char * diff, int channelStride, int h)
{
  const unsigned char MAX_CHANNEL_VALUE = -1;

  unsigned char * srcIterator1 = (unsigned char *)src1;
  unsigned char * srcIterator2 = (unsigned char *)src2;
  unsigned char * dstIterator = (unsigned char *)diff;

  long pixelCount = channelStride * h;

  long long result = 0;
  unsigned char diff_value = 0;

  const unsigned char * SRC_ITERATOR_END = srcIterator1 + pixelCount;

  for (;srcIterator1 != SRC_ITERATOR_END; ++srcIterator1, ++srcIterator2, ++dstIterator)
  {
    diff_value = ABS(*srcIterator1 - *srcIterator2);
    *dstIterator = diff_value;
    result += diff_value;
  }

  return (double)result / (double)(pixelCount * MAX_CHANNEL_VALUE);
}