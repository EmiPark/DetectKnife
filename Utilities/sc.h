#pragma once

#include "cv.h"    
#include "highgui.h"
#include "tchar.h"
#include "windows.h"
#include <math.h>
#include <iostream>
#include <iomanip> 
#include <stack>
#include <ctime>
#include <imgproc.hpp>
#include <core\core.hpp>
#include <highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>
using namespace cv;
using namespace std;

double ** theta_array;
double ** r_array;			//��������ľ�������
double ** total_r_array;	//����֮ǰ�ľ�������
double ** SC;

#define BIG 1e+10
#define PI  3.1415926
#ifndef		eps			//a value used to avoid dividing zero
#define	eps 2.2204e-016
#endif	//eps


#ifndef _MYCODE_H_
#define _MYCODE_H_
#ifdef DLLDEMO1_EXPORTS
#define EXPORTS_DEMO _declspec(dllexport )
#else
#define EXPORTS_DEMO _declspec(dllimport)
#endif
extern "C" EXPORTS_DEMO IplImage * _stdcall DetectKnife(IplImage *image);
#endif

typedef struct MyPoint
{
	int x;
	int y;
	int mask;
}
MyPoint;
double random(double, double);
double ** calSC(CvPoint *points, CvMemStorage *storage, CvSeq *contour)
{
	/*
	������Jitendra�����㷨�����ȫ����Եľ��룬ÿ��ȥ��������С�ĵ���е�һ���㣬ֱ��ʣ�µĵ�������ﵽҪȡ���ĵ������nPointNeed�����ٶ�����(10s+)
	*/
	//�������points�����е�����
	int onetourlength = contour->total;
	MyPoint *mypoints = (MyPoint *)malloc(sizeof(MyPoint) * onetourlength);
	if (mypoints == NULL) exit(1);			//�������ڴ�ռ�󣬱������Ƿ����ɹ���
	int index, tempx = 0, tempy = 0;		//���ڴ�������˳��
	int i, minPoint = 0, mini = 0, minj = 0;	//���ڼ�¼��С�������
	double minDist;
	int nPointNeed = 30;

	//ȡ������ע�Ϳ��Իָ�Jitendra�����㷨
	/*
	srand(time(NULL));
	for (i = 0; i < onetourlength; i++)
	{
	mypoints[i].mask = 1;
	mypoints[i].x = points[i].x;
	mypoints[i].y = points[i].y;
	}
	for (i = 0; i < onetourlength; i++)
	{
	index = rand() % (onetourlength - i) + i;
	if (index != i)
	{
	tempx = mypoints[i].x;
	mypoints[i].x = mypoints[index].x;
	mypoints[index].x = tempx;

	tempy = mypoints[i].y;
	mypoints[i].y = mypoints[index].y;
	mypoints[index].y = tempy;
	}
	}
	//�����������������total_r_array��
	total_r_array = new double*[onetourlength];
	for (size_t i = 0; i < onetourlength; i++)
	{
	total_r_array[i] = new double[onetourlength];
	}
	for (size_t i = 0; i < onetourlength; i++)
	{
	for (size_t j = i + 1; j < onetourlength; j++)
	{
	double dx = mypoints[i].x - mypoints[j].x;
	double dy = mypoints[i].y - mypoints[j].y;
	total_r_array[i][j] = sqrt(dx*dx + dy*dy);
	//cout << total_r_array[i][j] << endl;
	}
	}
	for (size_t k = onetourlength; k > nPointNeed; k--)
	{
	minDist = 500;
	for (size_t i = 0; i < onetourlength; i++)
	{
	for (size_t j = i + 1; j < onetourlength; j++)
	{
	if (mypoints[i].mask == 0 || mypoints[j].mask == 0) continue;
	else
	{
	if (0 < total_r_array[i][j] < minDist)
	{
	minDist = total_r_array[i][j];
	minPoint = j;
	mini = i;
	minj = j;
	}
	}
	}
	}
	mypoints[minPoint].mask = 0;
	total_r_array[mini][minj] = -1;
	}
	for (size_t i = 0; i < onetourlength; i++)
	{
	delete[] total_r_array[i];
	}
	delete[] total_r_array;
	cout << "------ִ�е㼯���ҽ���------" << endl;
	*/

	double sum_dist = 0, mean_dist = 0;	//����֮���ƽ������
										//ͨ��ɸѡ�����ÿ�������еĵ����
	int nbins_theta = 12;		//ÿ���Ƕȿ��еĵ�ĸ���	
	int nbins_r = 5;		//ÿ��������е�ĸ���
	double r_bins_edges[10];	//���ڴ洢���������
	double r_inner = 0.125;
	double r_outer = 2;
	double	nDist;
	CvPoint center = cvPoint(0, 0);
	CvSeq* allpointsSeq = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage);
	CvPoint *treatPoints = (CvPoint *)malloc(sizeof(CvPoint) * onetourlength);
	//MyPoint *treatPoints = (MyPoint *)malloc(sizeof(MyPoint) * onetourlength);

	//��������������Ĳ��ִ��룬�ָ��������ʱ��ȡ������ע��
	/*
	for (size_t i = 0, j = 0; i < onetourlength; i++)
	{
	cout << i << "���������" << mypoints[i].x << "," << mypoints[i].y << endl;
	if (mypoints[i].mask == 1)
	{
	treatPoints[j].x = mypoints[i].x;
	treatPoints[j].y = mypoints[i].y;
	j++;
	}
	}
	*/

	//��CvPoint�����ݽṹ�����Զ����MyPoint
	for (size_t i = 0; i < onetourlength; i++)
	{
		mypoints[i].x = points[i].x;
		mypoints[i].y = points[i].y;
		mypoints[i].mask = 0;
	}

	srand(unsigned(time(0)));
	for (size_t i = 0; i < nPointNeed; )
	{
		//index = rand() % (onetourlength - i) + i;
		//mask==0ΪδΪtreatPoints�������ݣ�==1Ϊ�ѷ�������
		index = int(random(1, onetourlength));
		if (mypoints[index].mask == 0)
		{
			treatPoints[i].x = mypoints[index].x;
			treatPoints[i].y = mypoints[index].y;
			mypoints[index].mask = 1;
			i++;
		}
		else continue;
	}

	for (size_t i = 0; i < nPointNeed; i++)
	{
		center.x = treatPoints[i].x;
		center.y = treatPoints[i].y;
		cvSeqPush(allpointsSeq, &center);
		//cout << i <<"���������" << treatPoints[i].x << "," << treatPoints[i].y << endl;
		//cvCircle(dst, center, 1, Scalar(255, 255, 255), 1, 8, 0);
	}

	r_array = new double*[nPointNeed];
	for (size_t i = 0; i < nPointNeed; i++)
	{
		r_array[i] = new double[nPointNeed];
	}
	//�������е��ƽ������,�������������������r_array��
	for (size_t i = 0; i < nPointNeed; i++)
	{
		for (size_t j = 0; j < nPointNeed; j++)
		{
			//double dx = mypoints[i].x - mypoints[j].x;
			//double dy = mypoints[i].y - mypoints[j].y;
			double dx = treatPoints[i].x - treatPoints[j].x;
			double dy = treatPoints[i].y - treatPoints[j].y;
			r_array[i][j] = sqrt(dx*dx + dy*dy);
			sum_dist += sqrt(dx*dx + dy*dy);
			//cout << "��" << i +1<< "�������" << j+1 << "����ľ�����" << sqrt(dx*dx + dy*dy) << endl;
		}
	}
	mean_dist = sum_dist / (nPointNeed*nPointNeed);
	//CalPointDist()
	//������������ҳ��󣬾Ϳ�������Щ�㻭�������   
	//theta_array�����ʼ��
	theta_array = new double*[nPointNeed];
	for (size_t i = 0; i < nPointNeed; i++)
	{
		theta_array[i] = new double[nPointNeed];
	}

	//���㲢���������ĽǶ�(ͨ������)
	for (size_t i = 0; i < nPointNeed; i++)
	{
		for (size_t j = 0; j < nPointNeed; j++)
		{
			if (i == j)
				theta_array[i][j] = 0;
			else
			{
				theta_array[i][j] = atan2(treatPoints[j].y - treatPoints[i].y, treatPoints[j].x - treatPoints[i].x);			//atan2(1,sqrt(3))*180/Pi ����30 ����������Y��������ļнǣ������ﷵ�ص�������������X��������н�
																																//��thetaת����[0, 2*pi)
				theta_array[i][j] = fmod(fmod(theta_array[i][j] + 1e-5, 2 * PI) + 2 * PI, 2 * PI);		//fmodȡģ���������������������1e-5��ʾ10��-5�η�
																										//������һ�����ĽǶ�������(1-12)   
				theta_array[i][j] = 1 + floor(theta_array[i][j] * nbins_theta / (2 * PI));		//floorȡ��
			}
		}
	}


	//��ʼ��r_bins_edges����(0.125,0.25,0.5,1,2)
	nDist = (log10(r_outer) - log10(r_inner)) / (nbins_r - 1);
	for (int i = 0; i < nbins_r; i++)
		r_bins_edges[i] = pow(10, log10(r_inner) + nDist*i);

	//��������ݾ�����з���(1-6)
	for (int i = 0; i < nPointNeed; i++)
	{
		for (int j = 0; j < nPointNeed; j++)
		{
			r_array[i][j] /= mean_dist;
			int k = 0;
			for (; k < nbins_r; k++)
			{
				if (r_array[i][j] <= r_bins_edges[k])
					break;
			}
			r_array[i][j] = nbins_r + 1 - k;
		}
	}


	SC = new double*[nPointNeed];
	for (size_t k = 0; k < nPointNeed; k++)
	{
		SC[k] = new double[nbins_theta*nbins_r];
	}
	for (size_t i = 0; i < nPointNeed; i++)
	{
		for (size_t j = 0; j < nbins_theta*nbins_r; j++)
		{
			SC[i][j] = 0;
		}
	}
	for (int i = 0; i < nPointNeed; i++)
	{
		//ZeroMemory���ܵ��³������
		//ZeroMemory(SC[i], sizeof(double)*nbins_r*nbins_theta);
		for (int j = 0; j < nPointNeed; j++)
		{
			if (i == j || r_array[i][j] == 6)	//Do not count the point itself. This is a bug in the original shape context.
			{
				continue;
			}
			else if (r_array[i][j] < 6)
			{
				int temp = ((r_array[i][j] - 1) * nbins_theta + theta_array[i][j]) - 1;  //temp����[0-59]
				SC[i][temp]++;
			}
		}
	}


	//�����ڴ�
	for (size_t i = 0; i < nPointNeed; i++)
	{
		delete[] theta_array[i];
	}
	delete[] theta_array;

	for (size_t i = 0; i < nPointNeed; i++)
	{
		delete[] r_array[i];
	}
	delete[] r_array;
	//free(mypoints);
	//mypoints = NULL;
	//free(treatPoints);
	//treatPoints = NULL;

	//cout << "�㼯ƽ�������ǣ�" << mean_dist << endl;
	return SC;
}

double ** HistCost(double **SC1, double **SC2, int nPointNeed)
{
	double **costmat;
	costmat = new double*[nPointNeed];
	for (size_t i = 0; i < nPointNeed; i++)
	{
		costmat[i] = new double[nPointNeed];
	}
	int i, j, k;
	//Normalization
	double	nsum;
	for (i = 0; i<nPointNeed; i++)
	{
		nsum = eps;
		for (j = 0; j<60; j++)
			nsum += SC1[i][j];
		for (j = 0; j<60; j++)
			SC1[i][j] /= nsum;
	}
	for (i = 0; i<nPointNeed; i++)
	{
		nsum = eps;
		for (j = 0; j<60; j++)
			nsum += SC2[i][j];
		for (j = 0; j<60; j++)
			SC2[i][j] /= nsum;
	}

	//Calculate distance
	for (i = 0; i<nPointNeed; i++)
	{
		for (j = 0; j<nPointNeed; j++)
		{
			nsum = 0;
			for (k = 0; k<60; k++)
			{
				nsum += (SC1[i][k] - SC2[j][k]) * (SC1[i][k] - SC2[j][k]) /
					(SC1[i][k] + SC2[j][k] + eps);
			}
			costmat[i][j] = nsum / 2;
		}
	}
	return costmat;
}

double hungarian(double **assigncost, int dim, int *colsol, int *rowsol)
// input:
// dim        - problem size
// assigncost - cost matrix

// output:
// rowsol     - column assigned to row in solution
// colsol     - row assigned to column in solution
{
	unsigned char unassignedfound;
	int  i, imin, numfree = 0, prvnumfree, f, i0, k, freerow, *pred, *free;
	int  j, j1, j2, endofpath, last, low, up, *collist, *matches;
	double min, h, umin, usubmin, v2, *d, *v;

	free = new int[dim];       // list of unassigned rows.
	collist = new int[dim];    // list of columns to be scanned in various ways.
	matches = new int[dim];    // counts how many times a row could be assigned.
	d = new double[dim];         // 'cost-distance' in augmenting path calculation.
	pred = new int[dim];       // row-predecessor of column in augmenting/alternating path.
	v = new double[dim];

	// init how many times a row will be assigned in the column reduction.
	for (i = 0; i < dim; i++)
		matches[i] = 0;

	// COLUMN REDUCTION 
	for (j = dim - 1; j >= 0; j--)    // reverse order gives better results.
	{
		// find minimum cost over rows.
		min = assigncost[0][j];
		imin = 0;
		for (i = 1; i < dim; i++)
			if (assigncost[i][j] < min)
			{
				min = assigncost[i][j];
				imin = i;
			}
		v[j] = min;

		if (++matches[imin] == 1)
		{
			// init assignment if minimum row assigned for first time.
			rowsol[imin] = j;
			colsol[j] = imin;
		}
		else
			colsol[j] = -1;        // row already assigned, column not assigned.
	}

	// REDUCTION TRANSFER
	for (i = 0; i < dim; i++)
	{
		if (matches[i] == 0)     // fill list of unassigned 'free' rows.
			free[numfree++] = i;
		else
			if (matches[i] == 1)   // transfer reduction from rows that are assigned once.
			{
				j1 = rowsol[i];
				min = BIG;
				for (j = 0; j < dim; j++)
					if (j != j1)
						if (assigncost[i][j] - v[j] < min)
							min = assigncost[i][j] - v[j];
				v[j1] = v[j1] - min;
			}
	}
	// AUGMENTING ROW REDUCTION 
	int loopcnt = 0;           // do-loop to be done twice.
	do
	{
		loopcnt++;

		// scan all free rows.
		// in some cases, a free row may be replaced with another one to be scanned next.
		k = 0;
		prvnumfree = numfree;
		numfree = 0;             // start list of rows still free after augmenting row reduction.
		while (k < prvnumfree)
		{
			i = free[k];
			k++;

			// find minimum and second minimum reduced cost over columns.
			umin = assigncost[i][0] - v[0];
			j1 = 0;
			usubmin = BIG;
			for (j = 1; j < dim; j++)
			{
				h = assigncost[i][j] - v[j];
				if (h < usubmin)
					if (h >= umin)
					{
						usubmin = h;
						j2 = j;
					}
					else
					{
						usubmin = umin;
						umin = h;
						j2 = j1;
						j1 = j;
					}
			}

			i0 = colsol[j1];

			/* Begin modification by Yefeng Zheng 03/07/2004 */
			//if( umin < usubmin )
			if (fabs(umin - usubmin) > 1e-10)
				/* End modification by Yefeng Zheng 03/07/2004 */

				// change the reduction of the minimum column to increase the minimum
				// reduced cost in the row to the subminimum.
				v[j1] = v[j1] - (usubmin - umin);
			else                   // minimum and subminimum equal.
				if (i0 >= 0)         // minimum column j1 is assigned.
				{
					// swap columns j1 and j2, as j2 may be unassigned.
					j1 = j2;
					i0 = colsol[j2];
				}

			// (re-)assign i to j1, possibly de-assigning an i0.
			rowsol[i] = j1;
			colsol[j1] = i;

			if (i0 >= 0)           // minimum column j1 assigned earlier.

								   /* Begin modification by Yefeng Zheng 03/07/2004 */
								   //if( umin < usubmin )
				if (fabs(umin - usubmin) > 1e-10)
					/* End modification by Yefeng Zheng 03/07/2004 */

					// put in current k, and go back to that k.
					// continue augmenting path i - j1 with i0.
					free[--k] = i0;
				else
					// no further augmenting reduction possible.
					// store i0 in list of free rows for next phase.
					free[numfree++] = i0;
		}
	} while (loopcnt < 2);       // repeat once.

								 // AUGMENT SOLUTION for each free row.
	for (f = 0; f < numfree; f++)
	{
		freerow = free[f];       // start row of augmenting path.

								 // Dijkstra shortest path algorithm.
								 // runs until unassigned column added to shortest path tree.
		for (j = 0; j < dim; j++)
		{
			d[j] = assigncost[freerow][j] - v[j];
			pred[j] = freerow;
			collist[j] = j;        // init column list.
		}

		low = 0; // columns in 0..low-1 are ready, now none.
		up = 0;  // columns in low..up-1 are to be scanned for current minimum, now none.
				 // columns in up..dim-1 are to be considered later to find new minimum, 
				 // at this stage the list simply contains all columns 
		unassignedfound = FALSE;
		do
		{
			if (up == low)         // no more columns to be scanned for current minimum.
			{
				last = low - 1;

				// scan columns for up..dim-1 to find all indices for which new minimum occurs.
				// store these indices between low..up-1 (increasing up). 
				min = d[collist[up++]];
				for (k = up; k < dim; k++)
				{
					j = collist[k];
					h = d[j];
					if (h <= min)
					{
						if (h < min)     // new minimum.
						{
							up = low;      // restart list at index low.
							min = h;
						}
						// new index with same minimum, put on undex up, and extend list.
						collist[k] = collist[up];
						collist[up++] = j;
					}
				}

				// check if any of the minimum columns happens to be unassigned.
				// if so, we have an augmenting path right away.
				for (k = low; k < up; k++)
					if (colsol[collist[k]] < 0)
					{
						endofpath = collist[k];
						unassignedfound = TRUE;
						break;
					}
			}

			if (!unassignedfound)
			{
				// update 'distances' between freerow and all unscanned columns, via next scanned column.
				j1 = collist[low];
				low++;
				i = colsol[j1];
				h = assigncost[i][j1] - v[j1] - min;

				for (k = up; k < dim; k++)
				{
					j = collist[k];
					v2 = assigncost[i][j] - v[j] - h;
					if (v2 < d[j])
					{
						pred[j] = i;
						if (v2 == min)   // new column found at same minimum value
							if (colsol[j] < 0)
							{
								// if unassigned, shortest augmenting path is complete.
								endofpath = j;
								unassignedfound = TRUE;
								break;
							}
						// else add to list to be scanned right away.
							else
							{
								collist[k] = collist[up];
								collist[up++] = j;
							}
						d[j] = v2;
					}
				}
			}
		} while (!unassignedfound);

		// update column prices.
		for (k = 0; k <= last; k++)
		{
			j1 = collist[k];
			v[j1] = v[j1] + d[j1] - min;
		}

		// reset row and column assignments along the alternating path.
		do
		{
			i = pred[endofpath];
			colsol[endofpath] = i;
			j1 = endofpath;
			endofpath = rowsol[i];
			rowsol[i] = j1;
		} while (i != freerow);
	}

	// calculate optimal cost.
	double lapcost = 0;
	for (i = 0; i < dim; i++)
	{
		j = rowsol[i];
		lapcost = lapcost + assigncost[i][j];
	}

	// free reserved memory.
	delete[] pred;
	delete[] free;
	delete[] collist;
	delete[] matches;
	delete[] d;
	delete[] v;
	return lapcost;
}


/*
���ڼ���ģ�����״������
compareView��ȡģ��ͼ����ͨ���Ҷ�ͼ��
compareView1��ŵ�ͨ����ֵ��ģ��ͼ��
compareView2��Ź�һ�������ͨ����ֵ��ģ��ͼ��
compareView3��Ź�һ����ĵ�ͨ����ֵ��ģ��ͼ��
*/
double ** tempSC(IplImage * compareView, int contourSize)
{
	//cout << "------��ʼ����ģ��SC------" << endl;
	CvSize cs;
	double scale;
	int  ** Cdata;
	int  ** Ddata;
	IplImage * compareView1;
	IplImage * compareView2;
	IplImage * compareView3;
	compareView1 = cvCreateImage(cvGetSize(compareView), 8, 1);

	//cvErode(compareView, compareView, NULL, 1);
	//cvDilate(compareView, compareView, NULL, 1); 
	cvSmooth(compareView, compareView, CV_MEDIAN, 3, 1);
	cvThreshold(compareView, compareView, 120, 255, CV_THRESH_BINARY);
	Ddata = new int*[compareView->height];
	for (int i = 0; i < compareView->height; i++)
		Ddata[i] = new int[compareView->width];
	for (size_t i = 0; i < compareView->height; i++)
	{
		for (size_t j = 0; j < compareView->width; j++)
		{
			Ddata[i][j] = cvGet2D(compareView, i, j).val[0];
			cvSetReal2D(compareView1, i, j, Ddata[i][j]);
		}
	}

	for (size_t i = 0; i < compareView->height; i++)
	{
		delete[] Ddata[i];
	}
	delete[] Ddata;

	CvMemStorage* comStorage1 = cvCreateMemStorage(0);
	CvSeq* comContour1 = 0;
	CvSeq* comAllpointsSeq1 = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour), sizeof(CvPoint), comStorage1);
	int comContours_num1 = cvFindContours(compareView1, comStorage1, &comContour1, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));
	double comContourlenth1 = comContour1->total;
	scale = static_cast<double>(contourSize / comContourlenth1);

	//scale = 0.5;
	cs.height = int(compareView->height*scale);
	cs.width = int(compareView->width*scale);
	compareView2 = cvCreateImage(cs, 8, 3);
	compareView3 = cvCreateImage(cs, 8, 1);
	//ģ��ͼ���һ��
	cvResize(compareView, compareView2, CV_INTER_CUBIC);
	//��ģ��ͼ����ͨ���Ҷ�ͼ��ת��Ϊ��ͨ����ֵͼ��
	cvThreshold(compareView2, compareView2, 120, 255, CV_THRESH_BINARY);

	Cdata = new int*[cs.height];
	for (int i = 0; i < cs.height; i++)
		Cdata[i] = new int[cs.width];
	for (size_t i = 0; i < cs.height; i++)
	{
		for (size_t j = 0; j < cs.width; j++)
		{
			Cdata[i][j] = cvGet2D(compareView2, i, j).val[0];
			cvSetReal2D(compareView3, i, j, Cdata[i][j]);
		}
	}
	
	double ** tempSC;								//�ֱ����ڴ洢ģ���ԭ��������״������
	double optSC;									   //ֵԽ�ʹ������ƶ�Խ��
	double mean_dist = 0;
	CvMemStorage* comStorage2 = cvCreateMemStorage(0);
	CvSeq* comContour2 = 0;
	CvSeq* comtreatContour = 0;
	CvSeq* comAllpointsSeq2 = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour), sizeof(CvPoint), comStorage2);
	//ע��cvFindContours����ֻ���ܵ�ͨ���Ķ�ֵͼ��
	int comContours_num2 = cvFindContours(compareView3, comStorage2, &comContour2, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));
	int comContourlenth2 = comContour2->total;
	//cout << "ģ���С:" << comContourlenth2 << endl;
	CvPoint *comPoints = (CvPoint *)malloc(sizeof(CvPoint) * comContourlenth2);
	CvSeqReader comReader;										//-- ��������������
	CvPoint cpt = cvPoint(0, 0);
	cvStartReadSeq(comContour2, &comReader);        //��ʼ��ȡ   
	for (int i = 0; i < comContourlenth2; i++)
	{
		CV_READ_SEQ_ELEM(cpt, comReader);				//������һ�������е�һ��Ԫ�ص�
		comPoints[i] = cpt;
	}
	tempSC = calSC(comPoints, comStorage2, comContour2);
	//cout << "------����ģ��SC����------" << endl<<endl;
	for (size_t i = 0; i < compareView2->height; i++)
	{
		delete[] Cdata[i];
	}
	delete[] Cdata;

	return tempSC;
}

double random(double start, double end)
{
	return start + (end - start)*rand() / (RAND_MAX + 1.0);
}
