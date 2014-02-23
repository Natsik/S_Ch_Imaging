#ifndef _OTSU_H_
#define _OTSU_H_

/**
@brief build the otsu image treshold
@param histogram	[IN] source image histogram
@param numPixels	[IN] pixels counted in histogram
@return int - treshold value [0..255]
*/
int GetOtsuTreshold(const int * histogram, int numPixels);


/**
@brief make the binarizated image by otsu method
@param imageSrc	[IN] source image
@param imageDst	[IN] destination image
@param w				[IN] width
@param h				[IN] height
@return none
*/
void  StoneOtsu(const unsigned char * imageSrc, char unsigned * imageDst, int w, int h);

/**
@brief make the binarizated image by adaptive otsu method
@param imageSrc	[IN] source image
@param imageDst	[IN] destination image
@param w				[IN] width
@param h				[IN] height
@param frame_size	[IN] frame size
@return none
*/
void AdaptiveOtsu(const unsigned char * imageSrc, unsigned char * imageDst, int w, int h, int frame_size);
void AdaptiveHoleOtsu( const unsigned char * imageSrc, unsigned char * imageDst, int w, int h, int frame_size);

#endif//_OTSU_H_H_
