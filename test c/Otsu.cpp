#include "Otsu.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "../core/bacmathdefinitions.h"

#define HISTOGRAM_RANGE 256

#define INDEX(i, j, w) (j) * (w) + (i)

void CountDownHistogram(int * histogram, const unsigned char * imageSrc, int w, int h, int left, int up, int right, int down)
{
	int i, j;

	assert (right > left);
	assert (down > up);

	assert (left >= 0);
	assert (up >= 0);

	assert (right <= w);
	assert (down <= h);

	for (i = 0; i < HISTOGRAM_RANGE; i++) 
		histogram[i] = 0;

	for (i = left; i < right; i++)
	{
		for (j = up; j < down; j++)
		{
			histogram[imageSrc[INDEX(i, j, w)]]++;
		}
	} 
}

void UpdateHistogram(int * histogram, const unsigned char * imageSrc, int w, int h
							, int left, int up, int right, int down
							, int nleft, int nup, int nright, int ndown)
{
	int i, j;
  if (right == nright && left == nleft && down == ndown && up == nup)
    return;

	assert (nright > nleft);
	assert (ndown > nup);
	assert (right > left);
	assert (down > up);

	assert (nleft >= left);
	assert (ndown >= down);
	assert ((right - left) == (nright - nleft));
	assert ((down - up) == (ndown - nup));

	assert (nright < w);
	assert (ndown < h);
	assert (left >= 0);
	assert (up >= 0);

	if (nleft > (right - (nright- nleft) / 2) || //if we need to subtract more then add
		 nup > (down - (ndown- nup) / 2 ))
	{
		CountDownHistogram(histogram, imageSrc, w, h, nleft, nup, nright, ndown);
		return ;
	}

  /*****************************/
  /*
    _ _ _ _ _ _ _ _ _ _ _ 
  |%%%%% &&&&&&&&&&&&&&&&|
   %%%%% ________________________
  |%%%%%|                |*******|
   %%%%%|                 *******|
  |%%%%%|                |*******|
   %%%%%|                 *******|
  |%%%%%|                |*******|
   %%%%%|                 *******|
  |_ _ _|_ _ _ _ _ _ _ _ |*******|
        |#################*******|
        |#################*******|
        |________________________|
	/*****************************/



	// Add that marked by *****
	for (i = right; i < nright; i++)
	{
		for (j = nup; j < ndown; j++)
		{
			histogram[imageSrc[INDEX(i, j, w)]]++;
		}
	} 

	// Add that marked by #####
	for (i = nleft; i < right; i++)
	{
		for (j = down; j < ndown; j++)
		{
			histogram[imageSrc[INDEX(i, j, w)]]++;
		}
	}

	// Delete that marked by &&&
	for (i = nleft; i < right; i++)
	{
		for (j = up; j < nup; j++)
		{
			histogram[imageSrc[INDEX(i, j, w)]]--;
		}
	} 

	// Delete that marked by %%%%
	for (i = left; i < nleft; i++)
	{
		for (j = up; j < down; j++)
		{
			histogram[imageSrc[INDEX(i, j, w)]]--;
		}
	}
}

int GetOtsuTreshold(const int * histogram, int numPixels)
{
	int hStart, hEnd;

	static float probability[HISTOGRAM_RANGE];
	static float ch[HISTOGRAM_RANGE];
	static float m[HISTOGRAM_RANGE];

	float mean, max, bcv;
	int threshold, i;

	for (hStart = 0; histogram[hStart] == 0; hStart++) 
	{
	}

	for (hEnd   = HISTOGRAM_RANGE - 1; histogram[hEnd] == 0; hEnd--)
	{
	}

	/*Calculate probability for each color*/

	for (i = 0; i < HISTOGRAM_RANGE; i++)
	{
		probability[i] = histogram[i] / (float)numPixels;
	}

	ch[0] = probability[0];
	m[0]  = 0.0f;

	for (i = 1; i < HISTOGRAM_RANGE; i++)
	{
		ch[i] = ch[i-1] + probability[i];
		m[i] = m[i-1] + i * probability[i];
	}

	mean      = m[HISTOGRAM_RANGE - 1];

	max       = 0;
	threshold = 0;

	for (i = hStart; i <= hEnd; i++)
	{
		bcv = mean * ch[i] - m[i];
		bcv = bcv * bcv / (ch[i] * (1.0f - ch[i]));

		if (max < bcv)
		{
			max = bcv;
			threshold = i;
		}
	}

	return threshold;
}

void AcceptTreshold(const unsigned char * imageSrc, unsigned char * imageDst, int numPixels, int threshold)
{
	int i;

	const unsigned char * src = imageSrc;
	unsigned char * dst = imageDst;

	for (i = 0; i < numPixels; i++)
	{
		unsigned char val;

		val = *src;

		if (val < threshold)
			val = 0;
		else
			val = 255;
		*dst = val;

		src++;
		dst++;
	}
}

void StoneOtsu( const unsigned char * imageSrc, unsigned char * imageDst, int w, int h)
{
	int histogram[HISTOGRAM_RANGE];
	int threshold;

	int numPixels = w * h;

	CountDownHistogram(histogram, imageSrc, w, h, 0, 0, w, h);
	threshold = GetOtsuTreshold(histogram, numPixels);

	AcceptTreshold(imageSrc, imageDst, numPixels, threshold);
}

void AdaptiveOtsu( const unsigned char * imageSrc, unsigned char * imageDst, int w, int h, int frame_size)
{
  int histogram[HISTOGRAM_RANGE];
  int histogramBase[HISTOGRAM_RANGE];
  
  int i, j;
  int threshold;
  if (frame_size % 2 == 1)
    frame_size --;
  int numPixels = frame_size * frame_size;
  int half_frame_size = frame_size / 2;


  CountDownHistogram(histogramBase, imageSrc, w, h, 0, 0, frame_size, frame_size);

  for (i = 0; i < w; i++)
  {
    int last_i = i - 1, new_i = i;

    last_i = PIN(half_frame_size, last_i, w - half_frame_size - 1);
    new_i  = PIN(half_frame_size, new_i , w - half_frame_size - 1);

    UpdateHistogram(histogramBase, imageSrc, w, h, last_i - half_frame_size, 0, last_i + half_frame_size, frame_size, new_i - half_frame_size, 0, half_frame_size + new_i, frame_size);
    memcpy(histogram, histogramBase, sizeof(histogram));

    for (j = 0; j < h; j++)
    {
      int last_j = j - 1, new_j = j;

      last_j = PIN(half_frame_size, last_j, h - half_frame_size - 1);
      new_j  = PIN(half_frame_size, new_j , h - half_frame_size - 1);

      UpdateHistogram(histogram, imageSrc, w, h, new_i - half_frame_size, last_j - half_frame_size, new_i + half_frame_size, last_j + half_frame_size, new_i - half_frame_size, new_j - half_frame_size, new_i + half_frame_size, new_j + half_frame_size);

      threshold = GetOtsuTreshold(histogram, numPixels);

      imageDst[INDEX(i, j, w)] = imageSrc[INDEX(i, j, w)] < threshold ? 0 : 255;
    }
  }
}

struct s_thresholds{
  int x, y;
  int threshold;
};

void AdaptiveHoleOtsu( const unsigned char * imageSrc, unsigned char * imageDst, int w, int h, int frame_size)
{
  int histogram[HISTOGRAM_RANGE];

  int i, j;
  if (frame_size % 2 == 1)
    frame_size --;

  frame_size*=4;

  if (frame_size < 20)
    frame_size = 20;
  int numPixels = frame_size * frame_size;
  int half_frame_size = frame_size / 2;

  int threshold = 0;

  int count_frames = (w * h) / (frame_size * frame_size) + 1;
  s_thresholds * thresholds = (s_thresholds *)malloc(((w * h) / (frame_size * frame_size) + w + h) * sizeof(s_thresholds));
  int current_treshold = 0;

  for (i = half_frame_size; i < w; i += frame_size)
  {
    for (j = half_frame_size; j < h; j += frame_size)
    {
      int lx, uy, rx, dy;

      lx = PIN(0, i - half_frame_size, w - 1);
      rx = PIN(frame_size, i + half_frame_size, w - 1);
      uy = PIN(0, j - half_frame_size, h - 1);
      dy = PIN(frame_size, j + half_frame_size, h - 1);

      numPixels = (rx - lx) * (dy - uy);

      CountDownHistogram(histogram, imageSrc, w, h, lx, uy, rx, dy);

      thresholds[current_treshold].threshold = GetOtsuTreshold(histogram, numPixels);
      thresholds[current_treshold].x = i;
      thresholds[current_treshold++].y = j;
    }
  }

  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++)
    {
      int count = 0;
      //int threshold = 0;
      for (int k = 0; k < current_treshold; k++)
      {
        int x = (i - thresholds[k].x);
        int y = (j - thresholds[k].y);
        int ds = ABS(x) + ABS(y);
        if (ds < 4 * frame_size)
        {
          threshold += thresholds[k].threshold;
          count++;
        }
      }

      if (count != 0)
        threshold /= count;
      else
        threshold = 127;

      imageDst[INDEX(i, j, w)] = imageSrc[INDEX(i, j, w)] < threshold ? 0 : 255;
    }

    free(thresholds);
}
