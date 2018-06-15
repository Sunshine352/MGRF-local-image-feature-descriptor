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


#include "OMGRF.h"
#include <stdio.h>
#include <vector>
#include <functional>
#include <algorithm>
using namespace std;


OxKey* ReadKeyFile(const char* filename, int& keynum)
{
	FILE* f = fopen(filename,"rt");
	if ( f == NULL )
	{
		printf("file does not exist, %s\n",filename);
		return NULL;
	}
	float temp;
	fscanf(f,"%f\n",&temp);
	int n_keys;
	fscanf(f,"%d\n",&n_keys);
	keynum = n_keys;
	OxKey* pKeys = new OxKey[n_keys];
	int i;
	for (i = 0;i < n_keys;i++)
	{
		fscanf(f,"%f %f %f %f %f",&pKeys[i].x,&pKeys[i].y,&pKeys[i].a,&pKeys[i].b,&pKeys[i].c);
		if (temp > 1.0)
		{
			int drop = 0;
			for(int j = 0;j < temp;j++)
				fscanf(f," %d",&drop);
		}
		fscanf(f,"\n");
	}
	fclose(f);
	return pKeys;
}

void CalcuTrans(OxKey* pKeys,int n)
{
	CvMat *A = cvCreateMat(2,2,CV_32FC1);
	CvMat *EigenVals = cvCreateMat(2,1,CV_32FC1);
	CvMat *EigenVects = cvCreateMat(2,2,CV_32FC1);
	CvMat *EigenVals_sqrt_inv = cvCreateMat(2,2,CV_32FC1);
	float *A_data = A->data.fl;
	for (int i = 0;i < n;i++)
	{
		A_data[0] = pKeys[i].a;
		A_data[1] = pKeys[i].b;
		A_data[2] = pKeys[i].b;
		A_data[3] = pKeys[i].c;
		// A = U' * D * U;
		cvEigenVV(A, EigenVects, EigenVals);
		// D = D^-0.5
		EigenVals_sqrt_inv->data.fl[0] = 1.0f / (float)sqrt(EigenVals->data.fl[0]);
		EigenVals_sqrt_inv->data.fl[1] = 0;
		EigenVals_sqrt_inv->data.fl[2] = 0;
		EigenVals_sqrt_inv->data.fl[3] = 1.0f / (float)sqrt(EigenVals->data.fl[1]);
		pKeys[i].square = EigenVals_sqrt_inv->data.fl[0] * EigenVals_sqrt_inv->data.fl[3];
		// U = D * U
		cvMatMul(EigenVals_sqrt_inv,EigenVects,EigenVals_sqrt_inv);
		cvTranspose(EigenVects,EigenVects);
		// A = U' * (D * U)
		cvMatMul(EigenVects,EigenVals_sqrt_inv,A);
		
		pKeys[i].trans[0] = A_data[0];
		pKeys[i].trans[1] = A_data[1];
		pKeys[i].trans[2] = A_data[2];
		pKeys[i].trans[3] = A_data[3];
	}
	cvReleaseMat(&A);
	cvReleaseMat(&EigenVals);
	cvReleaseMat(&EigenVects);
	cvReleaseMat(&EigenVals_sqrt_inv);
}


int* Extract_MOMGRF(const OxKey &key, IplImage *im, int nOrder, int quan_level, int nSampling, int nRegion)
{
	int i;
	int dim1 = nOrder * quan_level * quan_level * quan_level * nSampling / 3 * nRegion; //MultiOGOP
	int dim2 = nOrder * quan_level * quan_level * quan_level * nSampling / 3 * nRegion; //MultiOGDOP
	int dim3 = nOrder * quan_level * quan_level * quan_level * nSampling / 3 * nRegion; //MultiOIDOP
	int dim = dim1 + dim2 + dim3;
	int *desc = new int[dim];

	for (i = 0;i < nRegion;i++)
	{
		int *tmp_desc = 0;
		if (tmp_desc = Extract_OMGRF(key, im, nOrder, quan_level, nSampling, 1.5*i + 3, 41))
		{
			for (int j = 0;j < dim/nRegion;j++)
			{
				desc[i*dim/nRegion+j] = tmp_desc[j];
			}
			delete [] tmp_desc;
		}
		else
		{
			delete [] desc;
			return NULL;
		}
	}
	return desc;
}


int* Extract_OMGRF(const OxKey &key, IplImage *imSrc, int nOrder, int quan_level, int nSampling, double scale, int patch_width)
{
	int nPixels = 0;
	Pixel *pPixel_Array = Normalize_Patch(key, imSrc, nSampling, scale, patch_width, nPixels);
	if( pPixel_Array == NULL ) return NULL;
	std::sort(pPixel_Array,pPixel_Array+nPixels);

	int dim1 = nOrder * quan_level * quan_level * quan_level * nSampling / 3;  //MultiOGOP
	float *desc1 = new float[dim1];
	memset(desc1, 0, sizeof(float)*dim1);

	int dim2 = nOrder * quan_level * quan_level * quan_level * nSampling / 3;  //MultiOGDOP
	float *desc2 = new float[dim2];
	memset(desc2, 0, sizeof(float)*dim2);

	int dim3 = nOrder * quan_level * quan_level * quan_level * nSampling / 3;  //MultiOIDOP
	float *desc3 = new float[dim3];
	memset(desc3, 0, sizeof(float)*dim3);


	//determine quantization thresholds: ogop/ogdop/oidop_quan_thres
	//ogop_quan_thres
	double *AllSamDir = new double[nPixels*nSampling];
	memset(AllSamDir, 0, sizeof(double)*nPixels*nSampling);
	for (int m = 0; m < nPixels; m++)
	{
		for (int n = 0; n < nSampling; n++)
		{
			AllSamDir[m*nSampling + n] = pPixel_Array[m].Sampling_gradient_dir[n];
		}
	}
	std::sort(AllSamDir, AllSamDir + nPixels*nSampling);
	double *ogop_quan_thres = new double[quan_level + 1];
	for (int i = 0; i < (quan_level + 1); i++)
	{
		int quan_gap = int(nPixels * nSampling / double(quan_level) + 0.5);
		int idx_thresh = quan_gap*i;
		if (idx_thresh > nPixels*nSampling - 1) idx_thresh = nPixels*nSampling - 1;
		ogop_quan_thres[i] = AllSamDir[idx_thresh];
	}
	delete[] AllSamDir;

	//ogdop_quan_thres
	double *AllSamDirDiff = new double[nPixels*nSampling];
	memset(AllSamDirDiff, 0, sizeof(double)*nPixels*nSampling);
	for (int m = 0; m < nPixels; m++)
	{
		for (int n = 0; n < nSampling; n++)
		{
			AllSamDirDiff[m*nSampling + n] = pPixel_Array[m].SamplingDirDiff[n];
		}
	}
	std::sort(AllSamDirDiff, AllSamDirDiff + nPixels*nSampling);
	double *ogdop_quan_thres = new double[quan_level + 1];
	for (int i = 0; i < (quan_level + 1); i++)
	{
		int quan_gap = int(nPixels * nSampling / double(quan_level) + 0.5);
		int idx_thresh = quan_gap*i;
		if (idx_thresh > nPixels*nSampling - 1) idx_thresh = nPixels*nSampling - 1;
		ogdop_quan_thres[i] = AllSamDirDiff[idx_thresh];
	}
	delete[] AllSamDirDiff;

	//oidop_quan_thres
	double *AllSamValueDiff = new double[nPixels*nSampling];
	memset(AllSamValueDiff, 0, sizeof(double)*nPixels*nSampling);
	for (int m = 0; m < nPixels; m++)
	{
		for (int n = 0; n < nSampling; n++)
		{
			AllSamValueDiff[m*nSampling + n] = pPixel_Array[m].SamplingValueDiff[n];
		}
	}
	std::sort(AllSamValueDiff, AllSamValueDiff + nPixels*nSampling);
	double *oidop_quan_thres = new double[quan_level + 1];
	for (int i = 0; i < (quan_level + 1); i++)
	{
		int quan_gap = int(nPixels * nSampling / double(quan_level) + 0.5);
		int idx_thresh = quan_gap*i;
		if (idx_thresh > nPixels*nSampling - 1) idx_thresh = nPixels*nSampling - 1;
		oidop_quan_thres[i] = AllSamValueDiff[idx_thresh];
	}
	delete[] AllSamValueDiff;


	for (int i = 0;i < nOrder;i++)
	{
		int gap = int(nPixels / double(nOrder) + 0.5);
		int idxGrayOrder = 0; 
		int idxOrderVec[2];
		double idxOrderWeight[2];

		for (int j = 0;j < nPixels;j++)
		{
			int idx_thresh_low = gap*i;
			int idx_thresh_high = gap*(i+1);
			if (idx_thresh_high > nPixels-1) idx_thresh_high = nPixels-1;
			if (pPixel_Array[j].gray < pPixel_Array[idx_thresh_low].gray) continue;
			if (pPixel_Array[j].gray > pPixel_Array[idx_thresh_high].gray) break;
			
			idxGrayOrder = i;
			idxOrderWeight[0] = ((pPixel_Array[idx_thresh_high].gray - pPixel_Array[idx_thresh_low].gray) - (pPixel_Array[j].gray - pPixel_Array[idx_thresh_low].gray)) / (pPixel_Array[idx_thresh_high].gray - pPixel_Array[idx_thresh_low].gray);
			idxOrderWeight[1] = ((pPixel_Array[idx_thresh_high].gray - pPixel_Array[idx_thresh_low].gray) - (pPixel_Array[idx_thresh_high].gray - pPixel_Array[j].gray)) / (pPixel_Array[idx_thresh_high].gray - pPixel_Array[idx_thresh_low].gray);

			idxOrderVec[0] = idxGrayOrder;
			idxOrderVec[1] = (idxOrderVec[0] + 1) % nOrder;


			/**************************global order*********************************/
			for (int k = 0; k < nSampling / 3; k++)
			{

				/**********************************MultiOGOP***************************************/
				double tempDir[3];
				tempDir[0] = pPixel_Array[j].Sampling_gradient_dir[k];
				tempDir[1] = pPixel_Array[j].Sampling_gradient_dir[nSampling / 3 + k];
				tempDir[2] = pPixel_Array[j].Sampling_gradient_dir[nSampling / 3 * 2 + k];
				
				int ogop_code[3];
				for (int m = 0; m < 3; m++)
				{
					for (int n = 0; n < quan_level; n++)
					{
						if ((tempDir[m] >= ogop_quan_thres[n]) && (tempDir[m] < ogop_quan_thres[n + 1]))
						{
							ogop_code[m] = n;   //0.1.2.3
						}
						if (tempDir[m] == ogop_quan_thres[quan_level]) ogop_code[m] = quan_level - 1;
					}
				}

				int ogop_pattern = ogop_code[0] + ogop_code[1] * quan_level + ogop_code[2] * quan_level * quan_level;


				//according to ls_radius
				desc1[k * nOrder * quan_level * quan_level * quan_level + idxOrderVec[0] * quan_level * quan_level * quan_level + ogop_pattern] += idxOrderWeight[0];// *ogopWeight;
				desc1[k * nOrder * quan_level * quan_level * quan_level + idxOrderVec[1] * quan_level * quan_level * quan_level + ogop_pattern] += idxOrderWeight[1];// *ogopWeight;



				/**********************************MultiOGDOP***************************************/
				double tempDirDiff[3];   //0-1,1-2,2-0
				tempDirDiff[0] = pPixel_Array[j].SamplingDirDiff[k * 3 + 0];
				tempDirDiff[1] = pPixel_Array[j].SamplingDirDiff[k * 3 + 1];
				tempDirDiff[2] = pPixel_Array[j].SamplingDirDiff[k * 3 + 2];

				int ogdop_code[3];
				for (int m = 0; m < 3; m++)
				{
					for (int n = 0; n < quan_level; n++)
					{
						if ((tempDirDiff[m] >= ogdop_quan_thres[n]) && (tempDirDiff[m] < ogdop_quan_thres[n + 1]))
						{
							ogdop_code[m] = n;   //0.1.2.3
						}
						if (tempDirDiff[m] == ogdop_quan_thres[quan_level]) ogdop_code[m] = quan_level - 1;
					}
				}

				int ogdop_pattern = ogdop_code[0] + ogdop_code[1] * quan_level + ogdop_code[2] * quan_level * quan_level;


				//according to ls_radius
				desc2[k * nOrder * quan_level * quan_level * quan_level + idxOrderVec[0] * quan_level * quan_level * quan_level + ogdop_pattern] += idxOrderWeight[0];// *ogdopWeight;
				desc2[k * nOrder * quan_level * quan_level * quan_level + idxOrderVec[1] * quan_level * quan_level * quan_level + ogdop_pattern] += idxOrderWeight[1];// *ogdopWeight;



				/**********************************MultiOIDOP***************************************/
				double tempValueDiff[3]; //0-1,1-2,2-0
				tempValueDiff[0] = pPixel_Array[j].SamplingValueDiff[k * 3 + 0];
				tempValueDiff[1] = pPixel_Array[j].SamplingValueDiff[k * 3 + 1];
				tempValueDiff[2] = pPixel_Array[j].SamplingValueDiff[k * 3 + 2];

				int oidop_code[3];
				for (int m = 0; m < 3; m++)
				{
					for (int n = 0; n < quan_level; n++)
					{
						if ((tempValueDiff[m] >= oidop_quan_thres[n]) && (tempValueDiff[m] < oidop_quan_thres[n + 1]))
						{
							oidop_code[m] = n;   //0.1.2.3
						}
						if (tempValueDiff[m] == oidop_quan_thres[quan_level]) oidop_code[m] = quan_level - 1;
					}
				}

				int oidop_pattern = oidop_code[0] + oidop_code[1] * quan_level + oidop_code[2] * quan_level * quan_level;

				//according to ls_radius
				desc3[k * nOrder * quan_level * quan_level * quan_level + idxOrderVec[0] * quan_level * quan_level * quan_level + oidop_pattern] += idxOrderWeight[0];// *oidopWeight;
				desc3[k * nOrder * quan_level * quan_level * quan_level + idxOrderVec[1] * quan_level * quan_level * quan_level + oidop_pattern] += idxOrderWeight[1];// *oidopWeight;

			}
		}
	}

	Norm_desc(desc1, 0.2, dim1);
	Norm_desc(desc2, 0.2, dim2);
	Norm_desc(desc3, 0.2, dim3);

	delete[] pPixel_Array;

	int *desc1_temp = new int[dim1];
	for (int i = 0; i < dim1; i++)
	{
		desc1_temp[i] = (int)(desc1[i] * 255 + 0.5);
	}
	delete[] desc1;

	int *desc2_temp = new int[dim2];
	for (int i = 0; i < dim2; i++)
	{
		desc2_temp[i] = (int)(desc2[i] * 255 + 0.5);
	}
	delete[] desc2;

	int *desc3_temp = new int[dim3];
	for (int i = 0; i < dim3; i++)
	{
		desc3_temp[i] = (int)(desc3[i] * 255 + 0.5);
	}
	delete[] desc3;


	int dim = dim1 + dim2 + dim3;
	int *desc = new int[dim];
	for (int m = 0; m < dim; m++)
	{
		if (m < dim1) desc[m] = desc1_temp[m];
		else if (m < dim1 + dim2) desc[m] = desc2_temp[m - dim1];
		else desc[m] = desc3_temp[m - dim1 - dim2];
	}

	return desc;
}


void Norm_desc(float *desc, double illuThresh, int dim)
{
	// Normalize the descriptor, and threshold 
	// value of each element to 'illuThresh'.
	int i;
	double norm = 0.0;
	
	for (i=0; i<dim; ++i)
	{
		norm += desc[i] * desc[i];
	}
	
	norm = sqrt(norm);
	
	for (i=0; i<dim; ++i)
	{
		desc[i] /= norm;
		
		if (desc[i] > illuThresh)
		{
			desc[i] = illuThresh;
		}
	}
	
	// Normalize again.
	
	norm = 0.0;
	
	for (i=0; i<dim; ++i)
	{
		norm += desc[i] * desc[i];
	}
	
	norm = sqrt(norm);
	
	for (i=0; i<dim; ++i)
	{
		desc[i] /= norm;
	}
}

Pixel* Normalize_Patch(const OxKey &key, IplImage* in, int nSampling, float scale, int patch_width, int &nPixels)
{
	float trans[4];
	trans[0] = key.trans[0] * (2.0 * scale / patch_width);
	trans[1] = key.trans[1] * (2.0 * scale / patch_width);
	trans[2] = key.trans[2] * (2.0 * scale / patch_width);
	trans[3] = key.trans[3] * (2.0 * scale / patch_width);
	int minX = in->width;
	int maxX = 0;
	int minY = in->height;
	int maxY = 0;
	double theta_interval = 5 * CV_PI / 180;
	for (int i = 0;i < 72;i++)
	{
		double xS = (1.414 * (patch_width / 2.0) + 8) * cos(theta_interval * i);
		double yS = (1.414 * (patch_width / 2.0) + 8) * sin(theta_interval * i);
		double x_trans = trans[0] * xS + trans[1] * yS + key.x;
		double y_trans = trans[2] * xS + trans[3] * yS + key.y;
		if (int(x_trans) < minX) minX = int(x_trans);
		if ((int(x_trans)+1) > maxX) maxX = int(x_trans) + 1;
		if (int(y_trans) < minY) minY = int(y_trans);
		if ((int(y_trans)+1) > maxY) maxY = int(y_trans) + 1;
	}
	minX = minX < 0 ? 0 : minX;
	minY = minY < 0 ? 0 : minY;
	maxX = maxX > (in->width - 1) ? (in->width - 1) : maxX;
	maxY = maxY > (in->height - 1) ? (in->height - 1) : maxY;
	int regionW = maxX - minX + 1;
	int regionH = maxY - minY + 1;
	CvRect rc = cvRect(minX,minY,regionW,regionH);
	cvSetImageROI(in,rc);
	IplImage *in_smooth = cvCreateImage(cvSize(regionW,regionH),IPL_DEPTH_8U,1);
 	if ( key.square * scale * scale > (patch_width * patch_width / 4.0) )
 	{
 		double sigma = key.square * scale * scale / ((patch_width * patch_width / 4.0));
 		sigma = sqrt(sigma);
 		cvSmooth(in,in_smooth,CV_GAUSSIAN,5,5,sigma);
 	}
	else
	{
		cvCopy(in,in_smooth);
	}
	cvResetImageROI(in);

	int patch_radius = patch_width / 2;
	int x,y;
	IplImage* outPatch = cvCreateImage(cvSize(patch_radius*2+1+16,patch_radius*2+1+16),IPL_DEPTH_32F,1);
	float *out_data = (float*)outPatch->imageData;
	for (y = -patch_radius-8;y <= patch_radius+8;y++)
	{
		for (x = -patch_radius-8;x <= patch_radius+8;x++)
		{
			float x1 = trans[0] * x + trans[1] * y + key.x;
			float y1 = trans[2] * x + trans[3] * y + key.y;
			x1 -= minX;
			y1 -= minY;
			if (x1 < 0 || x1 > (in_smooth->width - 1) || y1 < 0 || y1 > (in_smooth->height - 1))
			{
				out_data[(y + patch_radius + 8) * outPatch->width + (x + patch_radius + 8)] = 0;
			}
			else
			{
				out_data[(y + patch_radius + 8) * outPatch->width + (x + patch_radius + 8)] = 
					get_image_value(in_smooth,x1,y1);
			}
		}
	}
	
	cvSmooth(outPatch,outPatch,CV_GAUSSIAN,5,5,1.6);

	Pixel *pPixel_Array = new Pixel[patch_width * patch_width - 1];
	int nCount = 0;

	for (y = -patch_radius;y <= patch_radius;y++)
	{
		for (x = -patch_radius;x <= patch_radius;x++)
		{
			if( 0 == y && 0 == x) continue;
			double dis = x * x + y * y;
			dis = sqrt(dis);
			if (dis > patch_radius) continue;
			float x_ori = trans[0] * x + trans[1] * y + key.x;
			float y_ori = trans[2] * x + trans[3] * y + key.y;
			x_ori -= minX;
			y_ori -= minY;


			float *ls_Radius = new float[nSampling];
			int ngroup = nSampling / 3;
			for (int i = 0; i < nSampling; i++)
			{
				ls_Radius[i] = 2 * (i % ngroup) + 6;  //{6,8,10,12}
			}

			double *SamplingValue = new double[nSampling];
			double *SamplingGradientDir = new double[nSampling];

			//get the coordinate of sampling points
			double theta = 2 * CV_PI / nSampling;
			double nOri = atan2((double)y, (double)x); 

			int nSamValidcount = 0; 
			for (int k = 0; k < nSampling; k++)
			{
				float x1 = x + ls_Radius[k] * cos(nOri + k * theta);
				float y1 = y + ls_Radius[k] * sin(nOri + k * theta);

				//First judge whether out of bound -> image coordinate system
				float trans_x = trans[0] * x1 + trans[1] * y1 + key.x;
				float trans_y = trans[2] * x1 + trans[3] * y1 + key.y;
				trans_x -= minX;
				trans_y -= minY;
				if (trans_x < 0 || trans_x >(in_smooth->width - 1) || trans_y < 0 || trans_y >(in_smooth->height - 1))  break;
				SamplingValue[k] = get_image_value(in_smooth, trans_x, trans_y);
				nSamValidcount = nSamValidcount + 1;
		
				//each sampling point again sample
				double SamValue[4];
				double Samtheta = 2 * CV_PI / 4;
				double SamOri = atan2((double)(y1-y), (double)(x1-x));
				for (int m = 0; m < 4; m++)
				{
					float SamX = x1 + 6 * cos(SamOri + m * Samtheta);
					float SamY = y1 + 6 * sin(SamOri + m * Samtheta);

					float trans_Samx = trans[0] * SamX + trans[1] * SamY + key.x;
					float trans_Samy = trans[2] * SamX + trans[3] * SamY + key.y;
					trans_Samx -= minX;
					trans_Samy -= minY;

					//Second judge whether out of bound  -> image coordinate system
					if (trans_Samx < 0 || trans_Samx >(in_smooth->width - 1) || trans_Samy < 0 || trans_Samy >(in_smooth->height - 1))
					{
						SamValue[m] = 0;
						continue;
					}

					SamValue[m] = get_image_value(in_smooth, trans_Samx, trans_Samy);
				}

				double tempdX = SamValue[0] - SamValue[2];
				double tempdY = SamValue[1] - SamValue[3];
				SamplingGradientDir[k] = atan2(tempdY, tempdX);
		
			}

			if (nSamValidcount != nSampling) continue; 


			double *SamplingValueDiff = new double[nSampling];
			double *SamplingDirDiff = new double[nSampling]; //0-1,1-2,2-0
			for (int n = 0; n < nSampling / 3; n++)
			{
				SamplingValueDiff[n * 3 + 0] = SamplingValue[n] - SamplingValue[nSampling / 3 + n];
				SamplingValueDiff[n * 3 + 1] = SamplingValue[nSampling / 3 + n] - SamplingValue[nSampling / 3 * 2 + n];
				SamplingValueDiff[n * 3 + 2] = SamplingValue[nSampling / 3 * 2 + n] - SamplingValue[n];
				SamplingDirDiff[n * 3 + 0] = SamplingGradientDir[n] - SamplingGradientDir[nSampling / 3 + n];
				SamplingDirDiff[n * 3 + 1] = SamplingGradientDir[nSampling / 3 + n] - SamplingGradientDir[nSampling / 3 * 2 + n];
				SamplingDirDiff[n * 3 + 2] = SamplingGradientDir[nSampling / 3 * 2 + n] - SamplingGradientDir[n];
			}


			pPixel_Array[nCount].SamplingValue = SamplingValue;
			pPixel_Array[nCount].SamplingValueDiff = SamplingValueDiff;
			pPixel_Array[nCount].Sampling_gradient_dir = SamplingGradientDir;
			pPixel_Array[nCount].SamplingDirDiff = SamplingDirDiff;
			pPixel_Array[nCount].gray = out_data[(y + patch_radius + 8) * outPatch->width + (x + patch_radius + 8)];
			delete [] ls_Radius;
			nCount++;
		}
	}

	nPixels = nCount;
	cvReleaseImage(&outPatch);
	cvReleaseImage(&in_smooth);
	return pPixel_Array;
}


float get_image_value(IplImage *pImg, float x, float y)
{
	int widthstep = pImg->widthStep;

	int x1 = (int)x;
	int y1 = (int)y;
	int x2 = x1 + 1;
	int y2 = y1 + 1;
	float gray = 0;

	//gray = (uchar)pImg->imageData[y1*widthstep + x1] / 255.0f;
	if ((x2 - x) * (y2 - y) != 0) gray += (x2 - x) * (y2 - y) * (uchar)pImg->imageData[y1*widthstep + x1] / 255.0f;
	if ((x - x1) * (y2 - y) != 0) gray += (x - x1) * (y2 - y) * (uchar)pImg->imageData[y1*widthstep + x2] / 255.0f;
	if ((x2 - x) * (y - y1) != 0) gray += (x2 - x) * (y - y1) * (uchar)pImg->imageData[y2*widthstep + x1] / 255.0f;
	if ((x - x1) * (y - y1) != 0) gray += (x - x1) * (y - y1) * (uchar)pImg->imageData[y2*widthstep + x2] / 255.0f;

	return gray;
}

