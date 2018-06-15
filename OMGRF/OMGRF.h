/*
An implementation of OMGRF descriptor

For more information, refer to:

Z. Sun, F. Zhou and Q. Liao, "A robust feature descriptor based on multiple
gradient-related features," 2017 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), New Orleans, LA, 2017, pp. 1408-1412.

Copyright (C) 2017 Z. Sun <szm15@mails.tsinghua.edu.cn>
All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
at your option any later version.
See the GNU General Public License for more details.

*/


#ifndef OMGRF_H
#define OMGRF_H

#include <highgui.h>
#include <cxcore.h>
#include <cv.h>
#include <vector>
using namespace std;


struct OxKey 
{
	float x;
	float y;
	float a;
	float b;
	float c;
	float trans[4];
	float square;
};

struct Pixel 
{
	float gray;
	double *SamplingValue;
	double *SamplingValueDiff;
	double *Sampling_gradient_dir;
	double *SamplingDirDiff;

	bool operator < (const Pixel &m1) const
	{
		return gray < m1.gray;
	};
};


OxKey* ReadKeyFile(const char* filename, int& keynum);
void CalcuTrans(OxKey* pKeys,int n);
int* Extract_MOMGRF(const OxKey &key, IplImage *im, int nOrder, int quan_level, int nSampling, int nRegion);
int* Extract_OMGRF(const OxKey &key, IplImage *imSrc, int nOrder, int quan_level, int nSampling, double scale, int patch_width);
void Norm_desc(float *desc, double illuThresh, int dim);
float get_image_value(IplImage *pImg, float x, float y);
Pixel* Normalize_Patch(const OxKey &key, IplImage* in, int nSampling, float scale, int patch_width, int &nPixels);

#endif
