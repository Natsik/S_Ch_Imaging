void c_linear_filter(const unsigned char * src, unsigned char * dst, int channelStride, int h, int * matrix, int matrix_dimension, int divisor);
void c_white_noise(const unsigned char * src, unsigned char * dst, int channelStride, int h, int p, int d);
void c_bil(const unsigned char * src, unsigned char * dst, int channelStride, int h, int p, int min);
void c_mesh(const unsigned char * src, unsigned char * dst, int channelStride, int h, int mesh_w, int mesh_h, unsigned char tone);
double c_diff_images(const unsigned char * src1, unsigned char * src2, int channelStride, int h);