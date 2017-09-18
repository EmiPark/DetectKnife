// deKnife.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "sc.h"

typedef int * intArrayPtr;

int main()
{
	clock_t startTime, endTime;
	startTime = clock();
	void FillInternalContours(IplImage *pBinary, double dAreaThre);
	int nPointNeed = 30;
	int width = 0, height = 0, count = 0;
	intArrayPtr * Bdata, *Gdata, *Rdata, *Vdata;
	intArrayPtr * Ydata, *CBdata, *CRdata, *Wdata;
	vector<vector<int>> matData;

	IplImage * image = cvLoadImage(".\\Resource\\a9.bmp");
	int edgeThresh = 1;
	int lowThreshold = 30;
	int const max_lowThreshold = 100;
	int ratio = 3;
	int kernel_size = 3;
	IplImage * view;
	IplImage * testView;
	IplImage * testChannel1;
	IplImage * testChannel2;
	IplImage * testChannel3;
	width = image->width;
	height = image->height;
	//cvNamedWindow("ԭͼ");
	//cvShowImage("ԭͼ", image);
	
	testView = cvCreateImage(cvGetSize(image), 8, 3);
	cvCvtColor(image, testView, CV_RGB2YCrCb);
	//�任ͨ�������������
	IplImage * Channel1;
	IplImage * Channel2;
	IplImage * Channel3;
	Channel1 = cvCreateImage(cvGetSize(testView), 8, 1);
	Channel2 = cvCreateImage(cvGetSize(testView), 8, 1);
	Channel3 = cvCreateImage(cvGetSize(testView), 8, 1);
	testChannel1 = cvCreateImage(cvGetSize(testView), 8, 3);
	testChannel2 = cvCreateImage(cvGetSize(testView), 8, 3);
	testChannel3 = cvCreateImage(cvGetSize(testView), 8, 3);
	cvSplit(image, Channel3, Channel2, Channel1 ,0);			//ע����������˳���������
	cvMerge(Channel3, Channel2, 0, 0, testChannel1);			//��XYZ�е�X������ʾ����
	cvMerge(Channel3, 0, Channel1, 0, testChannel2);
	cvMerge(0, Channel2, Channel1, 0, testChannel3);

	//����YCbCr�Ķ�ά����
	Ydata = new intArrayPtr[height];
	CRdata = new intArrayPtr[height];
	CBdata = new intArrayPtr[height];
	Wdata = new intArrayPtr[height];
	for (int i = 0; i < height; i++)
	{
		Ydata[i] = new int[width];
		CRdata[i] = new int[width];
		CBdata[i] = new int[width];
		Wdata[i] = new int[width];
	}

	for (int i = 0; i < height; i++)		  //ע��cvGet2D�Ķ�ȡ˳������height��width
	{
		for (int j = 0; j < width; j++)		//RGB��������Ϊheight����Ϊwidth
		{
			Ydata[i][j] = static_cast<int>(((uchar *)testView->imageData)[i*(testView->widthStep / sizeof(uchar)) + j*testView->nChannels + 0]);
			CRdata[i][j] = static_cast<int>(((uchar *)testView->imageData)[i*(testView->widthStep / sizeof(uchar)) + j*testView->nChannels + 1]);
			CBdata[i][j] = static_cast<int>(((uchar *)testView->imageData)[i*(testView->widthStep / sizeof(uchar)) + j*testView->nChannels + 2]);
			Wdata[i][j] = 255;		//����һ��ȫΪ255���ж�����
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if ((Ydata[i][j] < 180) && (Ydata[i][j] >130) && (CRdata[i][j] < 210) && (CRdata[i][j] > 150) && (CBdata[i][j] < 90) && (CBdata[i][j] > 60))
				Wdata[i][j] = 255;
			else
				Wdata[i][j] = 0;
		}
	}
	view = cvCreateImage(cvGetSize(testView), 8, 1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			cvSetReal2D(view, i, j, Wdata[i][j]);

	for (size_t i = 0; i < height; i++)
	{
		delete[] Ydata[i];
		delete[] CBdata[i];
		delete[] CRdata[i];
		delete[] Wdata[i];
	}
	delete[] Ydata;
	delete[] CBdata;
	delete[] CRdata;
	delete[] Wdata;

	//��̬����RGB��ά����height*width
	//Bdata = new intArrayPtr[height];
	//Gdata = new intArrayPtr[height];
	//Rdata = new intArrayPtr[height];
	//Vdata = new intArrayPtr[height];

	//for (int i = 0; i < height; i++)
	//{
	//	Bdata[i] = new int[width];
	//	Gdata[i] = new int[width];
	//	Rdata[i] = new int[width];
	//	Vdata[i] = new int[width];
	//}


	//for (int i = 0; i < height; i++)		  //ע��cvGet2D�Ķ�ȡ˳������height��width
	//{
	//	for (int j = 0; j < width; j++)		//RGB��������Ϊheight����Ϊwidth
	//	{
	//		//Bdata[i][j] = static_cast<int>(cvGet2D(image, j, i).val[0]);
	//		//Gdata[i][j] = static_cast<int>(cvGet2D(image, j, i).val[1]);
	//		//Rdata[i][j] = static_cast<int>(cvGet2D(image, j, i).val[2]);		//cvGet2D���������⣿��

	//		//Bdata[i][j]=CV_IMAGE_ELEM(image, uchar, i, 3 * j);
	//		//Gdata[i][j] = CV_IMAGE_ELEM(image, uchar, i, 3 * j + 1);
	//		//Rdata[i][j] = CV_IMAGE_ELEM(image, uchar, i, 3 * j + 2);

	//		Bdata[i][j] = static_cast<int>(((uchar *)image->imageData)[i*(image->widthStep / sizeof(uchar)) + j*image->nChannels + 0]);
	//		Gdata[i][j] = static_cast<int>(((uchar *)image->imageData)[i*(image->widthStep / sizeof(uchar)) + j*image->nChannels + 1]);
	//		Rdata[i][j] = static_cast<int>(((uchar *)image->imageData)[i*(image->widthStep / sizeof(uchar)) + j*image->nChannels + 2]);
	//		Vdata[i][j] = 255;		//����һ��ȫΪ255���ж�����
	//	}
	//}

	//for (int i = 0; i < height; i++)
	//{
	//	for (int j = 0; j < width; j++)
	//	{
	//		if ((Rdata[i][j] <= 90) && (Rdata[i][j] >= 0) && (Gdata[i][j] <= 145) && (Gdata[i][j] >= 55) && (Bdata[i][j] <= 255) && (Bdata[i][j] >= 180))
	//		{
	//			Vdata[i][j] = 255;
	//		}
	//		else
	//		{
	//			Vdata[i][j] = 0;
	//		}
	//	}
	//}

	////cv::Mat matImage(image->height, image->width, CV_8UC1, Bdata);
	////view = &IplImage(matImage);

	////����
	//view = cvCreateImage(cvGetSize(image), 8, 1);
	//for (int i = 0; i<height; i++)
	//{
	//	for (int j = 0; j<width; j++)
	//	{
	//		cvSetReal2D(view, i, j, Vdata[i][j]);
	//	}
	//}
	//for (size_t i = 0; i < height; i++)
	//{
	//	delete[] Rdata[i];
	//	delete[] Gdata[i];
	//	delete[] Bdata[i];
	//	delete[] Vdata[i];
	//}
	//delete[] Rdata;
	//delete[] Gdata;
	//delete[] Bdata;
	//delete[] Vdata;

	//cvSetData(view, Vdata, view->widthStep);
	//view->imageData = (char*)Vdata;

	IplImage * treatView = cvCreateImage(cvGetSize(view), view->depth, view->nChannels);
	IplImage * dilateView = cvCreateImage(cvGetSize(view), view->depth, view->nChannels);
	IplImage * removeView = cvCreateImage(cvGetSize(view), view->depth, view->nChannels);
	IplImage * drawContour = cvCreateImage(cvGetSize(view), view->depth, view->nChannels);


	//arrDilate(view, dilateView);
	//cvDilate(image, view, NULL, 1);
	//cvErode(view, erodeView, NULL, 1);   

	/*
	�ر�ע�⣺cvShowImage����RGBɫϵ��ʾͼ���
	*/
	//cvNamedWindow("ԭͼ");
	//cvNamedWindow("��ȡͼ");
	//cvNamedWindow("����ͼ");
	//cvNamedWindow("��ʴͼ");
	//cvNamedWindow("���ͼ");
	//cvNamedWindow("����ͼ");
	//cvNamedWindow("ʵ����ȡͼ");

	//cvShowImage("��ȡͼ", view);
	//cvShowImage("ʵ����ȡͼ",testView);
	//cvShowImage("����ͼ", treatView);
	//cvShowImage("��ʴͼ", erodeView);
	//cvShowImage("���ͼ", dilateView);

	//cvNamedWindow("Channel1");
	//cvShowImage("Channel1", Channel1);
	//cvNamedWindow("Channel2");
	//cvShowImage("Channel2", Channel2);
	//cvNamedWindow("Channel3");
	//cvShowImage("Channel3", Channel3);

	//cvNamedWindow("testChannel1");
	//cvShowImage("testChannel1", testChannel1);
	//cvNamedWindow("testChannel2");
	//cvShowImage("testChannel2", testChannel2);
	//cvNamedWindow("testChannel3");
	//cvShowImage("testChannel3", testChannel3);

	//cvDilate(view, treatView, NULL, 3); //����
	//cvErode(treatView, treatView, NULL, 3);//��ʴ
	//cvSmooth(treatView, treatView, CV_MEDIAN, 3, dilateView->nChannels);

	cvErode(view, treatView, NULL, 1);
	cvDilate(treatView, treatView, NULL, 2);
	cvSmooth(treatView, treatView, CV_MEDIAN, 1, dilateView->nChannels);

	/*
	cvDilate(treatView, treatView, NULL, 6);
	cvErode(treatView, treatView, NULL, 6);
	cvSmooth(treatView, treatView, CV_MEDIAN, 3, dilateView->nChannels);

	cvDilate(treatView, treatView, NULL, 6);
	cvErode(treatView, treatView, NULL, 6);
	cvSmooth(treatView, treatView, CV_MEDIAN, 3, dilateView->nChannels);

	cvDilate(treatView, treatView, NULL, 6);
	cvErode(treatView, treatView, NULL, 6);
	cvSmooth(treatView, treatView, CV_MEDIAN, 3, dilateView->nChannels);
	*/

	//cvDilate(treatView, treatView, NULL, 4);
	//FillInternalContours(treatView, 100); //����ڲ�����

	//cvShowImage("����ͼ", treatView);

	//Mat mat(treatView, true);		//IplImageתMat
	//Mat mat1 = Mat::zeros(width, height, CV_8UC1);

	//MatתIplImage
	//cv::Mat img2;
	//IplImage imgTmp = img2;
	//IplImage *input = cvCloneImage(&imgTmp);


	//threshold(mat, mat, 100, 255, THRESH_BINARY);	//���ҽ�ͼ��ת��Ϊ��ֵͼ��
	//namedWindow("MATͼ��", CV_WINDOW_AUTOSIZE);
	//imshow("MATͼ��", mat);
	//count=Two_Pass(mat, mat1);
	//namedWindow("MATͼ��", CV_WINDOW_AUTOSIZE);
	//imshow("MATͼ��", mat);



	//�������δ�����������������ȡ�;��α��
	/*
	vector<vector<Point>> contours;		//�洢������������
	vector<Vec4i> hierarchy;					//�洢ÿһ�������Ĳ�Σ��ֱ��ʾ��һ��������ǰһ������������������Ƕ������������ţ�
	cvThreshold(treatView, treatView, 120, 255, CV_THRESH_BINARY);
	Mat mat(treatView, true);

	//�������ģ��ƥ��
	IplImage *result;
	if (!treatView || !templat)
	{
	cout << "��ͼ��ʧ��" << endl;
	//return 0;
	}
	int treatW, treatH, templatW, templatH, resultH, resultW;
	treatW = treatView->width;
	treatH = treatView->height;
	templatW = templat->width;
	templatH = templat->height;
	if (treatW < templatW || treatH < templatH)
	{
	cout << "ģ�岻�ܱ�ԭͼ���" << endl;
	//return 0;
	}
	resultW = treatW - templatW + 1;
	resultH = treatH - templatH + 1;
	result = cvCreateImage(cvSize(resultW, resultH), 8, 1);
	cvMatchTemplate(treatView, templat, result, CV_TM_SQDIFF);
	double minValue, maxValue;
	CvPoint minLoc, maxLoc;
	cvMinMaxLoc(result, &minValue, &maxValue, &minLoc, &maxLoc);
	cvRectangle(view, minLoc, cvPoint(minLoc.x + templatW, minLoc.y + templatH), cvScalar(0, 0, 255));
	cvNamedWindow("srcResult", 0);
	cvNamedWindow("templat", 0);
	cvShowImage("srcResult", view);
	cvShowImage("templat", templat);
	*/



	//��������ע�͵Ĵ���Ϊ��״�����ĵĲ���

	//ͨ��canny������ȡģ�������
	/*
	Mat compareImage = imread("C:\\CodingWorkspace\\Resource\\a4.bmp");
	Mat compareDst, detected_edges;
	compareDst.create(compareImage.size(), compareImage.type());
	namedWindow("888", CV_WINDOW_AUTOSIZE);
	blur(compareImage, detected_edges, Size(3, 3));
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
	compareDst = Scalar::all(0);
	compareImage.copyTo(compareDst, detected_edges);
	imshow("888", compareDst);
	*/

	//����ģ�����ģ�����״������

	double **tempsc1, **tempsc2, **tempsc3, **tempsc4, **tempsc5, **tempsc6, **tempsc7, **tempsc8, **tempsc9, **tempsc10, **tempsc11, **tempsc12, **tempsc13, **tempsc14, **tempsc15, **tempsc16, **tempsc17, **tempsc18;

	IplImage * compareView1 = cvLoadImage(".\\Resource\\temp1.jpg");
	IplImage * compareView2 = cvLoadImage(".\\Resource\\temp2.jpg");
	IplImage * compareView3 = cvLoadImage(".\\Resource\\temp3.jpg");
	IplImage * compareView4 = cvLoadImage(".\\Resource\\temp4.jpg");
	IplImage * compareView5 = cvLoadImage(".\\Resource\\temp5.jpg");
	IplImage * compareView6 = cvLoadImage(".\\Resource\\temp6.jpg");
	IplImage * compareView7 = cvLoadImage(".\\Resource\\temp7.jpg");
	IplImage * compareView8 = cvLoadImage(".\\Resource\\temp8.jpg");
	IplImage * compareView9 = cvLoadImage(".\\Resource\\temp9.jpg");
	IplImage * compareView10 = cvLoadImage(".\\Resource\\temp10.jpg");
	IplImage * compareView11 = cvLoadImage(".\\Resource\\temp11.jpg");
	IplImage * compareView12 = cvLoadImage(".\\Resource\\temp12.jpg");
	IplImage * compareView13 = cvLoadImage(".\\Resource\\temp13.jpg");
	IplImage * compareView14 = cvLoadImage(".\\Resource\\temp14.jpg");
	IplImage * compareView15 = cvLoadImage(".\\Resource\\temp15.jpg");
	IplImage * compareView16 = cvLoadImage(".\\Resource\\temp16.jpg");
	IplImage * compareView17 = cvLoadImage(".\\Resource\\temp17.jpg");
	IplImage * compareView18 = cvLoadImage(".\\Resource\\temp18.jpg");


	double ** orisc;
	double  ** costMat;
	double minOpt;

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	IplImage *dst = cvCreateImage(cvSize(view->width, view->height), 8, 1);		//���Ʋ���ͼ
	IplImage *treatViewRs;
	CvSeq* treatContour = 0;
	//CvSeq::total��ָ�����ڲ���ЧԪ�صĸ�������h_next��h_prev������ָ��CvSeq�ڲ�Ԫ �ص�ָ�룬������ָ������CvSeq
	//���´�������ÿһ��������ÿһ�����ص�ı���
	int num = 0;
	int rect_num = 0;
	int onetourlength = 0;
	int contours_num = cvFindContours(treatView, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));       // contours_num ��ʾ����һ���ж�����������
	double scale = 0;
	CvSize cs;

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
		if (minOpt > 3.8)
		{
			cvClearSeq(contour);
			num--;
			cout << "δ���" << endl << endl << endl;
			continue;
		}

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
			cout << rect_pts[i].x << " " << rect_pts[i].y << endl;
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
		cout << "dist_a:" << dist_a << endl << "dist_b:" << dist_b << endl;
		rect_array[rect_num][0] = rect_pts[0];
		rect_array[rect_num][1] = rect_pts[1];
		rect_array[rect_num][2] = rect_pts[2];
		rect_array[rect_num++][3] = rect_pts[3];

		matData.resize(rect_num);
		matData[rect_num-1].resize(8);
		matData[rect_num-1][0] = rect_pts[0].x;
		matData[rect_num-1][1] = rect_pts[0].y;
		matData[rect_num-1][2] = rect_pts[1].x;
		matData[rect_num-1][3] = rect_pts[1].y;
		matData[rect_num-1][4] = rect_pts[2].x;
		matData[rect_num-1][5] = rect_pts[2].y;
		matData[rect_num-1][6] = rect_pts[3].x;
		matData[rect_num-1][7] = rect_pts[3].y;

		//cout << "��ɱ��" << endl << endl << endl;
		//cvPolyLine(image, &pt1, &npts, 1, 1, CV_RGB(255, 0, 0), 2);
	}

	cvNamedWindow("���ͼ", 1);
	CvPoint *pt1 = &cvPoint(0, 0);
	int npts = 4;
	srand(unsigned(time(0)));
	cout << "��ɱ��" << endl << endl << endl;

	vector<vector<int>>::iterator iter;

	for (size_t i = 0; i < matData.size(); i++)
	{
		for (size_t j = 0; j < 8; j++)
			cout << matData[i][j] << " ";
		cout << endl;
	}
		
	    


	//for (size_t i = 0; i < 6; i++)
	//{
	//	for (size_t j = 0; j < 4; j++)
	//	{
	//		cout << rect_array[i][j].x << " " << rect_array[i][j].y << "	";
	//	}
	//	cout << endl;
	//}
	endTime = clock();
	cout << "������ʱ : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	while (true)
	{
		for (size_t i = 0; i < 6; i++)
		{
			if (rect_array[i][1].x == 0) continue;
			else
			{
				pt1 = rect_array[i];
				cvPolyLine(image, &pt1, &npts, 1, 1, CV_RGB(int(random(0, 255)), int(random(0, 255)), int(random(0, 255))), 3);
			}
		}
		cvShowImage("���ͼ", image);
		cvWaitKey(200);
	}
	cvWaitKey(-1);
	//cvDestroyWindow("ԭͼ");//���ٴ���
	//cvDestroyWindow("��ȡͼ");
	//cvDestroyWindow("����ͼ");
	//cvDestroyWindow("��ʴͼ");
	//cvDestroyWindow("���ͼ");
	//cvDestroyWindow("����ͼ");
	//cvReleaseImage(&image); //�ͷ�ͼ��
	//cvReleaseImage(&view);
	//cvReleaseImage(&treatView);
	//cvReleaseImage(&erodeView);
	//system("pause");
	return 0;
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
