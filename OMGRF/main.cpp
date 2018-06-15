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


#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "OMGRF.h"
#include <ctime>


int main(int argc, char** argv)
{
	char *im_file = 0;
	char *feat_file = 0;
	char *out_file= 0;
	int nOrder = 6,  nMultiRegion = 4;
	int m_Dim = 0, m_Dim1 = 0, m_Dim2 = 0, m_Dim3 = 0, m_Dim4 = 0;
	int nSampling = 8, quan_level = 4;

	int counter = 1;
	while( counter < argc )
	{
		if( !strcmp("-i", argv[counter] ))
		{
			im_file = argv[++counter];
			counter++;
			continue;
		}
		if( !strcmp("-f", argv[counter] ))
		{
			feat_file = argv[++counter];
			counter++;
			continue;
		}
		if( !strcmp("-o", argv[counter] ))
		{
			out_file = argv[++counter];
			counter++;
			continue;
		}
		if( !strcmp("-Order", argv[counter] ) )
		{
			nOrder = atoi(argv[++counter]);
			counter++;
			continue;
		}
		if (!strcmp("-quan_level", argv[counter]))
		{
			quan_level = atoi(argv[++counter]);
			counter++;
			continue;
		}
		if (!strcmp("-nSampling", argv[counter]))
		{
			nSampling = atoi(argv[++counter]);
			counter++;
			continue;
		}
		if( !strcmp("-R", argv[counter] ) )
		{
			nMultiRegion = atoi(argv[++counter]);
			counter++;
			continue;
		}
		exit(1);
	}

	/* do the job */

	m_Dim1 = nOrder * quan_level * quan_level * quan_level * nSampling / 3 * nMultiRegion; //MultiOGOP
	m_Dim2 = nOrder * quan_level * quan_level * quan_level * nSampling / 3 * nMultiRegion; //MultiOGDOP
	m_Dim3 = nOrder * quan_level * quan_level * quan_level * nSampling / 3 * nMultiRegion; //MultiOIDOP
	m_Dim = m_Dim1 + m_Dim2 + m_Dim3;

	clock_t start,final;
	start = clock();

	int m_nKeys = 0;
	OxKey *m_pKeys = ReadKeyFile(feat_file,m_nKeys);

	CalcuTrans(m_pKeys,m_nKeys);

	IplImage* m_pImg = cvLoadImage(im_file,CV_LOAD_IMAGE_GRAYSCALE);
	cvSmooth(m_pImg,m_pImg,CV_GAUSSIAN,5,5,1);

	FILE *fid = fopen(out_file,"wt");
	fprintf(fid,"%d\n%d\n",m_Dim,m_nKeys);
	int i;
	for (i = 0;i < m_nKeys;i++)
	{
		int *desc = 0;
		desc = Extract_MOMGRF(m_pKeys[i], m_pImg, nOrder, quan_level, nSampling, nMultiRegion);
		if ( !desc )	continue;
		fprintf(fid,"%f %f %f %f %f",m_pKeys[i].x,m_pKeys[i].y,m_pKeys[i].a,m_pKeys[i].b,m_pKeys[i].c);
		for (int j = 0;j < m_Dim;j++)
		{
			fprintf(fid," %d",desc[j]);
		}
		fprintf(fid,"\n");
		delete [] desc;
	}
	fclose(fid);

	final = clock();
	printf("\nTotal time used is %lf seconds ", (double)(final - start) / CLOCKS_PER_SEC);
	printf("and each keypoint uses %lf ms\n", (double)(final - start) / CLOCKS_PER_SEC / m_nKeys * 1000);

	cvReleaseImage(&m_pImg);

	delete [] m_pKeys;

	return 0;
}
