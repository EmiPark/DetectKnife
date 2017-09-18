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
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>
using namespace cv;
using namespace std;

double ** theta_array;
double ** r_array;			//��������ľ�������
double ** total_r_array;	//����֮ǰ�ľ�������
double ** SC;
typedef int * intArrayPtr;

double ** calSC(CvPoint *points, CvMemStorage *storage, CvSeq *contour);
double ** HistCost(double **SC1, double **SC2, int nPointNeed);
double hungarian(double **assigncost, int dim, int *colsol, int *rowsol);
double ** tempSC(IplImage * compareView, int contourSize);
#define BIG 1e+10
#define PI  3.1415926
#ifndef		eps			//a value used to avoid dividing zero
#define	eps 2.2204e-016
#endif	//eps

/*
#ifndef _MYCODE_H_
#define _MYCODE_H_
#ifdef DLLDEMO1_EXPORTS
#define EXPORTS_DEMO _declspec(dllexport )
#else
#define EXPORTS_DEMO _declspec(dllimport)
#endif
extern "C" EXPORTS_DEMO IplImage * _stdcall DetectKnife(IplImage *image);
#endif
*/
typedef struct MyPoint
{
	int x;
	int y;
	int mask;
}
MyPoint;
double random(double, double);

IplImage *DetectKnife(const char * imageName)
{
	clock_t startTime, endTime;
	startTime = clock();
	void arrDilate(IplImage * src, IplImage * dst);
	void FillInternalContours(IplImage *pBinary, double dAreaThre);
	void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode); //CheckMode: 0����ȥ��������1����ȥ��������; NeihborMode��0����4����1����8����;
	int Two_Pass(const cv::Mat& binImg, cv::Mat& lableImg);		 //������ͨ����ĸ���
	int icvprCcaBySeedFill(const cv::Mat& _binImg, cv::Mat& _lableImg);
	void labelRectangle(IplImage* pic, int &left, int &right, int &up, int &down);
	void MatchingMethod(int, void*);
	int nPointNeed = 80;
	int width = 0, height = 0, count = 0;
	intArrayPtr * Bdata, *Gdata, *Rdata, *Vdata;
	//IplImage * image = cvLoadImage("C:\\CodingWorkspace\\Resource\\a4.bmp");
	IplImage *image = cvLoadImage(imageName);
	int edgeThresh = 1;
	int lowThreshold = 30;
	int const max_lowThreshold = 100;
	int ratio = 3;
	int kernel_size = 3;
	IplImage * view;
	width = image->width;
	height = image->height;
	//cvNamedWindow("ԭͼ");
	//cvShowImage("ԭͼ", image);

	//��̬����RGB��ά����height*width
	Bdata = new intArrayPtr[height];
	Gdata = new intArrayPtr[height];
	Rdata = new intArrayPtr[height];
	Vdata = new intArrayPtr[height];

	for (int i = 0; i < height; i++)
	{
		Bdata[i] = new int[width];
		Gdata[i] = new int[width];
		Rdata[i] = new int[width];
		Vdata[i] = new int[width];
	}



	for (int i = 0; i < height; i++)		  //ע��cvGet2D�Ķ�ȡ˳������height��width
	{
		for (int j = 0; j < width; j++)		//RGB��������Ϊheight����Ϊwidth
		{
			//Bdata[i][j] = static_cast<int>(cvGet2D(image, j, i).val[0]);
			//Gdata[i][j] = static_cast<int>(cvGet2D(image, j, i).val[1]);
			//Rdata[i][j] = static_cast<int>(cvGet2D(image, j, i).val[2]);		//cvGet2D���������⣿��

			//Bdata[i][j]=CV_IMAGE_ELEM(image, uchar, i, 3 * j);
			//Gdata[i][j] = CV_IMAGE_ELEM(image, uchar, i, 3 * j + 1);
			//Rdata[i][j] = CV_IMAGE_ELEM(image, uchar, i, 3 * j + 2);

			Bdata[i][j] = static_cast<int>(((uchar *)image->imageData)[i*(image->widthStep / sizeof(uchar)) + j*image->nChannels + 0]);
			Gdata[i][j] = static_cast<int>(((uchar *)image->imageData)[i*(image->widthStep / sizeof(uchar)) + j*image->nChannels + 1]);
			Rdata[i][j] = static_cast<int>(((uchar *)image->imageData)[i*(image->widthStep / sizeof(uchar)) + j*image->nChannels + 2]);
			Vdata[i][j] = 255;		//����һ��ȫΪ255���ж�����
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if ((Rdata[i][j] <= 90) && (Rdata[i][j] >= 0) && (Gdata[i][j] <= 145) && (Gdata[i][j] >= 55) && (Bdata[i][j] <= 255) && (Bdata[i][j] >= 180))
			{
				Vdata[i][j] = 255;
			}
			else
			{
				Vdata[i][j] = 0;
			}
		}
	}

	//cv::Mat matImage(image->height, image->width, CV_8UC1, Bdata);
	//view = &IplImage(matImage);

	//����
	view = cvCreateImage(cvGetSize(image), 8, 1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cvSetReal2D(view, i, j, Vdata[i][j]);
		}
	}
	for (size_t i = 0; i < height; i++)
	{
		delete[] Rdata[i];
		delete[] Gdata[i];
		delete[] Bdata[i];
		delete[] Vdata[i];
	}
	delete[] Rdata;
	delete[] Gdata;
	delete[] Bdata;
	delete[] Vdata;


	//cvSetData(view, Vdata, view->widthStep);
	//view->imageData = (char*)Vdata;

	IplImage * treatView = cvCreateImage(cvGetSize(view), view->depth, view->nChannels);
	IplImage * dilateView = cvCreateImage(cvGetSize(view), view->depth, view->nChannels);
	IplImage * removeView = cvCreateImage(cvGetSize(view), view->depth, view->nChannels);
	IplImage * drawContour = cvCreateImage(cvGetSize(view), view->depth, view->nChannels);

	cvDilate(view, treatView, NULL, 3); //����
	cvErode(treatView, treatView, NULL, 3);//��ʴ
	cvSmooth(treatView, treatView, CV_MEDIAN, 3, dilateView->nChannels);

	cvDilate(treatView, treatView, NULL, 3);
	cvErode(treatView, treatView, NULL, 3);
	cvSmooth(treatView, treatView, CV_MEDIAN, 3, dilateView->nChannels);

	FillInternalContours(treatView, 200); //����ڲ�����



	double **tempsc1, **tempsc2, **tempsc3, **tempsc4, **tempsc5, **tempsc6, **tempsc7, **tempsc8, **tempsc9, **tempsc10, **tempsc11, **tempsc12, **tempsc13, **tempsc14, **tempsc15, **tempsc16, **tempsc17, **tempsc18;

	IplImage * compareView1 = cvLoadImage("C:\\DetectKnife\\Resource\\temp1.jpg");
	IplImage * compareView2 = cvLoadImage("C:\\DetectKnife\\Resource\\temp2.jpg");
	IplImage * compareView3 = cvLoadImage("C:\\DetectKnife\\Resource\\temp3.jpg");
	IplImage * compareView4 = cvLoadImage("C:\\DetectKnife\\Resource\\temp4.jpg");
	IplImage * compareView5 = cvLoadImage("C:\\DetectKnife\\Resource\\temp5.jpg");
	IplImage * compareView6 = cvLoadImage("C:\\DetectKnife\\Resource\\temp6.jpg");
	IplImage * compareView7 = cvLoadImage("C:\\DetectKnife\\Resource\\temp7.jpg");
	IplImage * compareView8 = cvLoadImage("C:\\DetectKnife\\Resource\\temp8.jpg");
	IplImage * compareView9 = cvLoadImage("C:\\DetectKnife\\Resource\\temp9.jpg");
	IplImage * compareView10 = cvLoadImage("C:\\DetectKnife\\Resource\\temp10.jpg");
	IplImage * compareView11 = cvLoadImage("C:\\DetectKnife\\Resource\\temp11.jpg");
	IplImage * compareView12 = cvLoadImage("C:\\DetectKnife\\Resource\\temp12.jpg");
	IplImage * compareView13 = cvLoadImage("C:\\DetectKnife\\Resource\\temp13.jpg");
	IplImage * compareView14 = cvLoadImage("C:\\DetectKnife\\Resource\\temp14.jpg");
	IplImage * compareView15 = cvLoadImage("C:\\DetectKnife\\Resource\\temp15.jpg");
	IplImage * compareView16 = cvLoadImage("C:\\DetectKnife\\Resource\\temp16.jpg");
	IplImage * compareView17 = cvLoadImage("C:\\DetectKnife\\Resource\\temp17.jpg");
	IplImage * compareView18 = cvLoadImage("C:\\DetectKnife\\Resource\\temp18.jpg");

	double ** orisc;
	double  ** costMat;
	double minOpt;

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	IplImage *dst = cvCreateImage(cvSize(view->width, view->height), 8, 1);		//���Ʋ���ͼ
	CvSeq* treatContour = 0;
	//CvSeq::total��ָ�����ڲ���ЧԪ�صĸ�������h_next��h_prev������ָ��CvSeq�ڲ�Ԫ �ص�ָ�룬������ָ������CvSeq
	//���´�������ÿһ��������ÿһ�����ص�ı���
	int num = 0;
	int rect_num = 0;
	int onetourlength = 0;
	int contours_num = cvFindContours(treatView, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));       // contours_num ��ʾ����һ���ж�����������

	CvPoint **rect_array;
	rect_array = new CvPoint *[6];
	for (size_t i = 0; i < 6; i++)
	{
		rect_array[i] = new CvPoint[4];
	}

	//vector<CvPoint *> vecPoint;
	//vector<vector<CvPoint> > vecPoint;

	for (; contour != 0; contour = contour->h_next)     //-- ָ����һ����������
	{
		//�������CvSeq�����Ԫ�صķ����ܹ���
		num++;
		onetourlength = contour->total;
		if (onetourlength < 120)
		{
			cvClearSeq(contour);
			num--;
			continue;
		}

		/*
		cout << onetourlength << "======" << endl;
		scale = 120.0 / onetourlength;
		cs.height = int(treatView->height*scale);
		cs.width = int(treatView->width*scale);
		treatViewRs = cvCreateImage(cs, treatView->depth, treatView->nChannels);
		cvResize(treatView, treatViewRs, CV_INTER_CUBIC);
		*/

		/*
		//��ɸѡ������и��Ƹ�������
		if (treatContour == NULL)
		treatContour = cvCloneSeq(contour, cvCreateMemStorage(0));
		else
		{
		treatContour = treatContour->h_next;
		treatContour = cvCloneSeq(contour, cvCreateMemStorage(0));
		}
		*/
		//�����������ռ䣬�ǵ��ͷ�   
		CvPoint *points = (CvPoint *)malloc(sizeof(CvPoint) * onetourlength);
		CvSeqReader reader;       //-- ������һ����������
		CvPoint pt = cvPoint(0, 0);
		cvStartReadSeq(contour, &reader);       //��ʼ��ȡ   
		for (int i = 0; i < onetourlength; i++) {
			CV_READ_SEQ_ELEM(pt, reader);     //--������һ�������е�һ��Ԫ�ص�
			points[i] = pt;
		}
		//��������
		orisc = calSC(points, storage, contour);
		//minOpt = calMinOpt(SCTD, orisc);

		int colsol[15], rowsol[15];
		int dim = 10;		//�������㷨�ĸ��Ӷ�
		minOpt = 50;
		orisc = calSC(points, storage, contour);
		tempsc1 = tempSC(compareView1, onetourlength);
		tempsc2 = tempSC(compareView2, onetourlength);
		tempsc3 = tempSC(compareView3, onetourlength);
		tempsc4 = tempSC(compareView4, onetourlength);
		tempsc5 = tempSC(compareView5, onetourlength);
		tempsc6 = tempSC(compareView6, onetourlength);
		tempsc7 = tempSC(compareView7, onetourlength);
		tempsc8 = tempSC(compareView8, onetourlength);
		tempsc9 = tempSC(compareView9, onetourlength);
		tempsc10 = tempSC(compareView10, onetourlength);
		tempsc11 = tempSC(compareView11, onetourlength);
		tempsc12 = tempSC(compareView12, onetourlength);
		tempsc13 = tempSC(compareView13, onetourlength);
		tempsc14 = tempSC(compareView14, onetourlength);
		tempsc15 = tempSC(compareView15, onetourlength);
		tempsc16 = tempSC(compareView16, onetourlength);
		tempsc17 = tempSC(compareView17, onetourlength);
		tempsc18 = tempSC(compareView18, onetourlength);

		costMat = HistCost(tempsc1, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc2, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc3, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc4, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc5, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc6, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc7, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc8, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc9, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc10, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc11, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc12, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc13, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc14, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc15, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc16, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc17, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		costMat = HistCost(tempsc18, orisc, nPointNeed);
		if (minOpt > hungarian(costMat, dim, colsol, rowsol))
		{
			minOpt = hungarian(costMat, dim, colsol, rowsol);
		}
		cout << "��С����ֵ��" << minOpt << endl;



		//minOpt = calMinOpt(SCTD, orisc);
		if (minOpt > 3.5)
		{
			cvClearSeq(contour);
			num--;
			cout << "δ���" << endl << endl << endl;
			continue;
		}

		/*
		ifstream input;
		input.open(SCTD);
		string s;
		double tdata;
		while (input >> s)
		{
		tdata = double(std::atoi(s.c_str()));
		if (tdata == -1) cout << "һ��ģ��SC�Ѿ��ɹ�����" << endl;
		else cout << tdata;
		}
		*/

		/*
		double ***scTest = new double **[nTemp];
		for (int i = 0; i < nTemp; i++)
		{
		scTest[i] = new double*[nPointNeed];
		for (int j = 0; j < nPointNeed; j++)
		{
		scTest[i][j] = new double[nbins_r*nbins_theta];
		}
		}
		*/
		//����ע�Ͳ����е�scTest����������whileѭ��֮��洢�����ݾ���ʧ�ˣ���֪Ϊ�Ρ�
		/*
		ifstream input;
		input.open(SCTD);
		string s;
		double tdata;
		for (size_t i = 0; i < nTemp; i++)
		{
		for (size_t j = 0; j < nPointNeed; j++)
		{
		for (size_t k = 0; k < nbins_r*nbins_theta;k++)
		{
		while ((input >> s)&&(std::atoi(s.c_str())!=-1))
		{
		tdata = static_cast<double>(std::atoi(s.c_str()));
		//scTest[i][j][k] = static_cast<double>(std::atoi(s.c_str()));
		scTest[i][j][k] = tdata;
		//cout<< typeid(scTest[i][j][k]).name() << "-";
		//cout << typeid(static_cast<double>(std::atoi(s.c_str()))).name()<<"+";
		}
		}
		}
		}
		input.close();

		cout << "-------------------����----------------------------------" << endl<<endl;
		*/


		CvBox2D rect = cvMinAreaRect2(contour, storage);
		CvPoint2D32f rect_pts0[4];
		cvBoxPoints(rect, rect_pts0);
		int dist_a = 0, dist_b = 0, k = 0;
		CvPoint rect_pts[4];
		//CvPoint *pt1 = rect_pts;
		//cout << "��ͨ������С��Ӿ��󶥵�����ֱ��ǣ�" << endl;
		for (size_t i = 0; i < 4; i++)
		{
			rect_pts[i] = cvPointFrom32f(rect_pts0[i]);
			//cout << rect_pts[i].x << " " << rect_pts[i].y << endl;
			dist_a = (int)sqrt((pow((rect_pts[0].x - rect_pts[1].x), 2) + pow((rect_pts[0].y - rect_pts[1].y), 2)));
			dist_b = (int)sqrt((pow((rect_pts[0].x - rect_pts[3].x), 2) + pow((rect_pts[0].y - rect_pts[3].y), 2)));
			if (dist_a < dist_b)
			{
				k = dist_a;
				dist_a = dist_b;
				dist_b = k;
			}
		}
		//�������У���������Ϊָ��ֻ��ָ�����һ��rect_pts
		//rect_array[rect_num] = rect_pts;
		//vecPoint.push_back(rect_pts);

		rect_array[rect_num][0] = rect_pts[0];
		rect_array[rect_num][1] = rect_pts[1];
		rect_array[rect_num][2] = rect_pts[2];
		rect_array[rect_num++][3] = rect_pts[3];
		//cout << "��ɱ��" << endl << endl << endl;
		//cvPolyLine(image, &pt1, &npts, 1, 1, CV_RGB(255, 0, 0), 2);
	}

	cvNamedWindow("���ͼ", 1);
	CvPoint *pt1 = &cvPoint(0, 0);
	int npts = 4;
	srand(unsigned(time(0)));
	cout << "��ɱ��" << endl << endl << endl;

	while (true)
	{
		for (size_t i = 0; i < 6; i++)
		{
			//�������dll����ʱĪ��������쳣����
			if (rect_array[i][0].x > 10000 || rect_array[i][0].y > 10000 || rect_array[i][1].x > 10000 || rect_array[i][1].y > 10000 || rect_array[i][2].x > 10000 || rect_array[i][2].y > 10000 || rect_array[i][3].x > 10000 || rect_array[i][3].y > 10000) continue;
			else
			{
				pt1 = rect_array[i];
				cvPolyLine(image, &pt1, &npts, 1, 1, CV_RGB(int(random(0, 255)), int(random(0, 255)), int(random(0, 255))), 3);
			}
		}
		cvShowImage("���ͼ", image);
		cvWaitKey(200);
	}
	endTime = clock();
	cout << "������ʱ : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cvReleaseImage(&view);
	cvReleaseImage(&treatView);
	//cvReleaseImage(&erodeView);
	//system("pause");
	return image;
}
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
	int nPointNeed = 80;

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
void FillInternalContours(IplImage *pBinary, double dAreaThre)
{
	double dConArea;
	CvSeq *pContour = NULL;
	CvSeq *pConInner = NULL;
	CvMemStorage *pStorage = NULL;
	// ִ������   
	if (pBinary)
	{
		// ������������   
		pStorage = cvCreateMemStorage(0);
		cvFindContours(pBinary, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		// �����������   
		cvDrawContours(pBinary, pContour, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 2, CV_FILLED, 8, cvPoint(0, 0));
		// ������ѭ��   
		int wai = 0;
		int nei = 0;
		for (; pContour != NULL; pContour = pContour->h_next)
		{
			wai++;
			// ������ѭ��   
			for (pConInner = pContour->v_next; pConInner != NULL; pConInner = pConInner->h_next)
			{
				nei++;
				// ���������   
				dConArea = fabs(cvContourArea(pConInner, CV_WHOLE_SEQ));
				//printf("%f\n", dConArea);
				if (dConArea <= dAreaThre)
				{
					cvDrawContours(pBinary, pConInner, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 0, CV_FILLED, 8, cvPoint(0, 0));
				}
			}
		}
		//printf("outer = %d, inner = %d\n", wai, nei);
		cvReleaseMemStorage(&pStorage);
		pStorage = NULL;
	}
}