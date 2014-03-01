void c_inversion(const unsigned char * src, unsigned char * dst, int w, int h);
void c_dilatation(const unsigned char * src, unsigned char * dst, int channelStride, int h);
void c_errosion(const unsigned char * src, unsigned char * dst, int channelStride, int h);
//example if image depth is 24 (R, G, B) and width is 100 then channelStride == 300