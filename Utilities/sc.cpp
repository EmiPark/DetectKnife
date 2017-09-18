#include "stdafx.h"
#include "sc.h"

//#define PI  3.1415926
typedef int * intArrayPtr;
//double ** theta_array;
//double ** r_array;	//采样过后的距离数组
//double ** total_r_array;	//采样之前的距离数组
//double ** SC;
/// 全局变量
Mat img;
Mat templ;
Mat result;
int match_method;
int max_Trackbar = 5;
//typedef struct MyPoint
//{
//	int x ;
//	int y ;
//	int mask;
//}
//MyPoint;



//int main(int argc, char** argv)
//{
//IplImage * _stdcall  DetectKnife(IplImage *image)
//IplImage * _stdcall  DetectKnife(const char * imageName) //返回Iplimage *的图像的最终版本
int _stdcall DetectKnife(const char * imageName,int ** result)  //dll之间传递STL数据(vector)可能出错
{
	//clock_t startTime, endTime;
	//startTime = clock();
	void arrDilate(IplImage * src, IplImage * dst);
	void FillInternalContours(IplImage *pBinary, double dAreaThre);
	void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode); //CheckMode: 0代表去除黑区域，1代表去除白区域; NeihborMode：0代表4邻域，1代表8邻域;
	int Two_Pass(const cv::Mat& binImg, cv::Mat& lableImg);		 //返回连通区域的个数
	int icvprCcaBySeedFill(const cv::Mat& _binImg, cv::Mat& _lableImg);
	void labelRectangle(IplImage* pic, int &left, int &right, int &up, int &down);
	void MatchingMethod(int, void*);
	int nPointNeed = 30;
	int width = 0, height = 0, count = 0;
	intArrayPtr * Ydata, *CBdata, *CRdata, *Wdata;
	vector<vector<int>> matData;
	//IplImage * image = cvLoadImage("C:\\CodingWorkspace\\Resource\\a4.bmp");
	IplImage *image = cvLoadImage(imageName);
	int edgeThresh = 1;
	int lowThreshold = 30;
	int const max_lowThreshold = 100;
	int ratio = 3;
	int kernel_size = 3;
	IplImage * testView;
	IplImage * testChannel1;
	IplImage * testChannel2;
	IplImage * testChannel3;
	IplImage * view;
	if (!image)
	{
		cout << "读入图像有误！" << endl;
		return 0;
	}
	width = image->width;
	height = image->height;
	//cvNamedWindow("原图");
	//cvShowImage("原图", image);
	testView = cvCreateImage(cvGetSize(image), 8, 3);
	cvCvtColor(image, testView, CV_RGB2YCrCb);
	//变换通道后的三个分量
	IplImage * Channel1;
	IplImage * Channel2;
	IplImage * Channel3;
	Channel1 = cvCreateImage(cvGetSize(testView), 8, 1);
	Channel2 = cvCreateImage(cvGetSize(testView), 8, 1);
	Channel3 = cvCreateImage(cvGetSize(testView), 8, 1);
	testChannel1 = cvCreateImage(cvGetSize(testView), 8, 3);
	testChannel2 = cvCreateImage(cvGetSize(testView), 8, 3);
	testChannel3 = cvCreateImage(cvGetSize(testView), 8, 3);
	cvSplit(image, Channel3, Channel2, Channel1, 0);			//注意分离出来的顺序是逆序的
	cvMerge(Channel3, Channel2, 0, 0, testChannel1);			//将XYZ中的X分量显示出来
	cvMerge(Channel3, 0, Channel1, 0, testChannel2);
	cvMerge(0, Channel2, Channel1, 0, testChannel3);


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

	for (int i = 0; i < height; i++)		  //注意cvGet2D的读取顺序是先height再width
	{
		for (int j = 0; j < width; j++)		//RGB数组是行为height，列为width
		{
			Ydata[i][j] = static_cast<int>(((uchar *)testView->imageData)[i*(testView->widthStep / sizeof(uchar)) + j*testView->nChannels + 0]);
			CRdata[i][j] = static_cast<int>(((uchar *)testView->imageData)[i*(testView->widthStep / sizeof(uchar)) + j*testView->nChannels + 1]);
			CBdata[i][j] = static_cast<int>(((uchar *)testView->imageData)[i*(testView->widthStep / sizeof(uchar)) + j*testView->nChannels + 2]);
			Wdata[i][j] = 255;		//建立一个全为255的判定矩阵
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
	//cvSetData(view, Vdata, view->widthStep);
	//view->imageData = (char*)Vdata;

	IplImage * treatView = cvCreateImage(cvGetSize(view), view->depth, view->nChannels);
	IplImage * dilateView = cvCreateImage(cvGetSize(view), view->depth, view->nChannels);
	IplImage * removeView = cvCreateImage(cvGetSize(view), view->depth, view->nChannels);
	IplImage * drawContour = cvCreateImage(cvGetSize(view), view->depth, view->nChannels);

	cvErode(view, treatView, NULL, 1);
	cvDilate(treatView, treatView, NULL, 2);
	cvSmooth(treatView, treatView, CV_MEDIAN, 1, dilateView->nChannels);

	//cvDilate(treatView, treatView, NULL, 3);
	//cvErode(treatView, treatView, NULL, 3);
	//cvSmooth(treatView, treatView, CV_MEDIAN, 3, dilateView->nChannels);
	
	//FillInternalContours(treatView, 200); //填充内部区域
	

	//Mat mat(treatView, true);		//IplImage转Mat
	//Mat mat1 = Mat::zeros(width, height, CV_8UC1);

	//Mat转IplImage
	//cv::Mat img2;
	//IplImage imgTmp = img2;
	//IplImage *input = cvCloneImage(&imgTmp);


	//threshold(mat, mat, 100, 255, THRESH_BINARY);	//将灰阶图像转换为二值图像
	//namedWindow("MAT图像", CV_WINDOW_AUTOSIZE);
	//imshow("MAT图像", mat);
	//count=Two_Pass(mat, mat1);
	//namedWindow("MAT图像", CV_WINDOW_AUTOSIZE);
	//imshow("MAT图像", mat);



	//以下两段代码可以完成轮廓的提取和矩形标记
	/*
	vector<vector<Point>> contours;		//存储检测的轮廓数组
	vector<Vec4i> hierarchy;					//存储每一个轮廓的层次（分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号）
	cvThreshold(treatView, treatView, 120, 255, CV_THRESH_BINARY);
	Mat mat(treatView, true);

	//下面进行模板匹配
	IplImage *result;
	if (!treatView || !templat)
	{
	cout << "打开图像失败" << endl;
	//return 0;
	}
	int treatW, treatH, templatW, templatH, resultH, resultW;
	treatW = treatView->width;
	treatH = treatView->height;
	templatW = templat->width;
	templatH = templat->height;
	if (treatW < templatW || treatH < templatH)
	{
	cout << "模板不能比原图像大" << endl;
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



	//以下整体注释的代码为形状上下文的操作

	//通过canny算子提取模板的轮廓
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

	//载入模板计算模板的形状上下文

	double **tempsc1, **tempsc2, **tempsc3, **tempsc4, **tempsc5, **tempsc6, **tempsc7, **tempsc8, **tempsc9, **tempsc10, **tempsc11, **tempsc12, **tempsc13, **tempsc14, **tempsc15, **tempsc16, **tempsc17, **tempsc18;

	IplImage * compareView1 = cvLoadImage(".\\Temp\\temp1.jpg");
	IplImage * compareView2 = cvLoadImage(".\\Temp\\temp2.jpg");
	IplImage * compareView3 = cvLoadImage(".\\Temp\\temp3.jpg");
	IplImage * compareView4 = cvLoadImage(".\\Temp\\temp4.jpg");
	IplImage * compareView5 = cvLoadImage(".\\Temp\\temp5.jpg");
	IplImage * compareView6 = cvLoadImage(".\\Temp\\temp6.jpg");
	IplImage * compareView7 = cvLoadImage(".\\Temp\\temp7.jpg");
	IplImage * compareView8 = cvLoadImage(".\\Temp\\temp8.jpg");
	IplImage * compareView9 = cvLoadImage(".\\Temp\\temp9.jpg");
	IplImage * compareView10 = cvLoadImage(".\\Temp\\temp10.jpg");
	IplImage * compareView11 = cvLoadImage(".\\Temp\\temp11.jpg");
	IplImage * compareView12 = cvLoadImage(".\\Temp\\temp12.jpg");
	IplImage * compareView13 = cvLoadImage(".\\Temp\\temp13.jpg");
	IplImage * compareView14 = cvLoadImage(".\\Temp\\temp14.jpg");
	IplImage * compareView15 = cvLoadImage(".\\Temp\\temp15.jpg");
	IplImage * compareView16 = cvLoadImage(".\\Temp\\temp16.jpg");
	IplImage * compareView17 = cvLoadImage(".\\Temp\\temp17.jpg");
	IplImage * compareView18 = cvLoadImage(".\\Temp\\temp18.jpg");

	double ** orisc;
	double  ** costMat;
	double minOpt;

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	IplImage *dst = cvCreateImage(cvSize(view->width, view->height), 8, 1);		//绘制采样图
	IplImage *treatViewRs;
	CvSeq* treatContour = 0;
	//CvSeq::total是指序列内部有效元素的个数；而h_next和h_prev并不是指向CvSeq内部元 素的指针，它们是指向其它CvSeq
	//以下代码可完成每一个轮廓上每一个像素点的遍历
	int num = 0;
	int rect_num = 0;
	int onetourlength = 0;
	int contours_num = cvFindContours(treatView, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));       // contours_num 表示的是一共有多少条轮廓线
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

	for (; contour != 0; contour = contour->h_next)     //-- 指向下一个轮廓序列
	{
		//这里遍历CvSeq里面的元素的方法很怪异
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
		//将筛选后的序列复制给新序列
		if (treatContour == NULL)
		treatContour = cvCloneSeq(contour, cvCreateMemStorage(0));
		else
		{
		treatContour = treatContour->h_next;
		treatContour = cvCloneSeq(contour, cvCreateMemStorage(0));
		}
		*/
		//给点数组分配空间，记得释放   
		CvPoint *points = (CvPoint *)malloc(sizeof(CvPoint) * onetourlength);
		CvSeqReader reader;       //-- 读其中一个轮廓序列
		CvPoint pt = cvPoint(0, 0);
		cvStartReadSeq(contour, &reader);       //开始提取   
		for (int i = 0; i < onetourlength; i++) {
			CV_READ_SEQ_ELEM(pt, reader);     //--读其中一个序列中的一个元素点
			points[i] = pt;
		}
		//绘制轮廓
		orisc = calSC(points, storage, contour);
		//minOpt = calMinOpt(SCTD, orisc);

		int colsol[15], rowsol[15];
		int dim = 10;		//匈牙利算法的复杂度
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
		//cout << "最小最优值：" << minOpt << endl;



		//minOpt = calMinOpt(SCTD, orisc);
		if (minOpt > 3.8)
		{
			cvClearSeq(contour);
			num--;
			//cout << "未标记" << endl;
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
		if (tdata == -1) cout << "一个模板SC已经成功载入" << endl;
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
		//以下注释部分中的scTest数组在跳出while循环之后存储的数据就消失了，不知为何。
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

		cout << "-------------------存完----------------------------------" << endl<<endl;
		*/


		CvBox2D rect = cvMinAreaRect2(contour, storage);
		CvPoint2D32f rect_pts0[4];
		cvBoxPoints(rect, rect_pts0);
		int dist_a = 0, dist_b = 0, k = 0;
		CvPoint rect_pts[4];
		//CvPoint *pt1 = rect_pts;
		//cout << "连通区域最小外接矩阵顶点坐标分别是：" << endl;
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
		//这样不行，可能是因为指针只会指向最后一个rect_pts
		//rect_array[rect_num] = rect_pts;
		//vecPoint.push_back(rect_pts);

		//rect_array[rect_num][0] = rect_pts[0];
		//rect_array[rect_num][1] = rect_pts[1];
		//rect_array[rect_num][2] = rect_pts[2];
		//rect_array[rect_num++][3] = rect_pts[3];

		matData.resize(++rect_num);
		matData[rect_num - 1].resize(8);
		matData[rect_num - 1][0] = rect_pts[0].x;
		matData[rect_num - 1][1] = rect_pts[0].y;
		matData[rect_num - 1][2] = rect_pts[1].x;
		matData[rect_num - 1][3] = rect_pts[1].y;
		matData[rect_num - 1][4] = rect_pts[2].x;
		matData[rect_num - 1][5] = rect_pts[2].y;
		matData[rect_num - 1][6] = rect_pts[3].x;
		matData[rect_num - 1][7] = rect_pts[3].y;

		//cout << "已标记" << endl;
		//cvPolyLine(image, &pt1, &npts, 1, 1, CV_RGB(255, 0, 0), 2);
	}

	
	
	//intArrayPtr * ReturnData;
	//ReturnData = new intArrayPtr[matData.size()+1];
	//for (int i = 0; i <= matData.size(); i++)
	//	ReturnData[i] = new int[8];
	//ReturnData[0][0] = matData.size();
	for (size_t i = 0; i < matData.size(); i++)
		for (size_t j = 0; j < 8; j++)
			result[i][j] = matData[i][j];
	
	//cvNamedWindow("标记图", 1);
	//CvPoint *pt1 = &cvPoint(0, 0);
	//int npts = 4;
	//srand(unsigned(time(0)));
	//cout << endl << "完成标记" << endl;
	//endTime = clock();
	//cout << "程序用时 : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	//下面是标记刀具矩形框闪动的代码
	/*
	while (true)
	{
		for (size_t i = 0; i < 6; i++)
		{
			//清理调用dll运行时莫名其妙的异常数据
			if (rect_array[i][0].x >10000|| rect_array[i][0].y>10000 || rect_array[i][1].x >10000 || rect_array[i][1].y>10000 || rect_array[i][2].x >10000 || rect_array[i][2].y>10000 || rect_array[i][3].x >10000 || rect_array[i][3].y>10000) continue;
			else
			{
				pt1 = rect_array[i];
				cvPolyLine(image, &pt1, &npts, 1, 1, CV_RGB(int(random(0, 255)), int(random(0, 255)), int(random(0, 255))), 3);
			}
		}
		cvShowImage("标记图", image);
		cvWaitKey(200);
	}
	*/
	
	
	//cvNamedWindow("标记图", 1);
	//cvShowImage("标记图", image);

	//以下矩形标记的方法用于保存点集所使用的容器是Vector<Vector<Point>>，不是CvSeq，在本程序中暂不适用
	/*
	Mat oriMat(image, true);
	Mat matContours = Mat::zeros(mat.size(), CV_8UC1);
	findContours(mat, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());		//从mat中提取轮廓，只检查外轮廓，存储所有的轮廓点

	for (size_t i = 0; i < contours.size(); i++)
	{
	if (contours[i].size() < 50)
	continue;
	drawContours(matContours, contours, i, Scalar(255), 1, 8, hierarchy);
	//绘制轮廓的最小外结矩形
	RotatedRect rect = minAreaRect(contours[i]);
	Point2f P[4];
	rect.points(P);
	for (int j = 0; j <= 3; j++)
	{
	line(oriMat, P[j], P[(j + 1) % 4], Scalar(0, 0, 255), 3);
	}
	}
	imshow("标记图", oriMat);
	imshow("Mat", mat);
	*/

	/*
	以下代码是实现轮廓提取的标准方法，但在矩形标记时会有困难，且没有实现像素点遍历
	*/
	/*
	int contour_num = cvFindContours(treatView, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//cvZero(drawContour);
	CvSeq* _contour = contour;
	double maxArea = 0;
	double minArea = 300;
	int m=0;
	for (; contour!=0; contour=contour->h_next)
	{
	double tmpArea = fabs(cvContourArea(contour));
	if (tmpArea < minArea)
	{
	cvSeqRemove(contour, 0);
	continue;
	}
	CvRect aRect = cvBoundingRect(contour, 0);
	//if ((aRect.width / aRect.height) < 1)	//限制宽高比例
	//{
	//	cvSeqRemove(contour, 0);
	//	continue;
	//}
	if (tmpArea > maxArea)
	maxArea = tmpArea;
	cout << tmpArea << endl;
	m++;
	CvScalar color = CV_RGB(255, 0, 0);
	cvDrawContours(drawContour, contour, color, color, -1, 1, 8);
	}

	//标注出最大的一个连接区域
	contour = _contour;
	for (; contour != 0; contour = contour->h_next)
	{
	count++;
	double tmparea = fabs(cvContourArea(contour));
	if (tmparea == maxArea)
	{
	CvScalar color = CV_RGB(255, 0, 0);
	cvDrawContours(drawContour, contour, color, color, -1, 1, 8);
	}
	}
	cvShowImage("轮廓图", drawContour);
	*/
	
	cvReleaseImage(&view);
	cvReleaseImage(&treatView);
	//cvReleaseImage(&erodeView);
	//system("pause");
	//return image;
	return rect_num;
}

//void arrErode(intArrayPtr* src, intArrayPtr* tar,IplImage *image)
void erode_image(IplImage * src, IplImage * dst)
{
	if (src == NULL || dst == NULL)
		return;

	int width = src->width;
	int height = src->height;

	//水平方向的腐蚀  
	for (int i = 0; i < src->height; i++)
	{
		for (int j = 1; j < src->width - 1; j++)
		{
			//  data = ((uchar *)(src->imageData + src->widthStep * i))[j];  
			if (((uchar *)(src->imageData + src->widthStep * i))[j] == 0)
			{
				((uchar *)(dst->imageData + dst->widthStep * i))[j] = 0;
				for (int k = 0; k < 3; k++)
				{
					//  data = ((uchar *)(src->imageData + src->widthStep * i))[j + k -1];  
					if (((uchar *)(src->imageData + src->widthStep * i))[j + k - 1] == 255)
					{
						((uchar *)(dst->imageData + dst->widthStep * i))[j] = 255;
						break;
					}
				}
			}
			else
				((uchar *)(dst->imageData + dst->widthStep * i))[j] = 255;
		}
	}
	//垂直方向的腐蚀  
	for (int i = 0; i < dst->width; i++)
	{
		for (int j = 1; j < dst->height - 1; j++)
		{
			//  data = ((uchar *)(src->imageData + src->widthStep * i))[j];  
			if (((uchar *)(dst->imageData + dst->widthStep * j))[i] == 0)
			{
				((uchar *)(src->imageData + src->widthStep * j))[i] = 0;
				for (int k = 0; k < 3; k++)
				{
					//  data = ((uchar *)(src->imageData + src->widthStep * i))[j + k -1];  
					if (((uchar *)(dst->imageData + dst->widthStep * (j + k - 1)))[i] == 255)
					{
						((uchar *)(src->imageData + src->widthStep * j))[i] = 255;
						break;
					}
				}
			}
			else
				((uchar *)(src->imageData + src->widthStep * j))[i] = 255;
		}
	}
}
void arrDilate(IplImage * src, IplImage * dst)
{
	if (src == NULL || dst == NULL)
		return;

	int width = src->width;
	int height = src->height;

	//水平方向的膨胀  
	for (int i = 0; i < src->height; i++)
	{
		for (int j = 1; j < src->width - 1; j++)
		{
			if (((uchar *)(src->imageData + src->widthStep * i))[j] == 255)
			{
				((uchar *)(dst->imageData + dst->widthStep * i))[j] = 255;
				for (int k = 0; k < 3; k++)
				{
					if (((uchar *)(src->imageData + src->widthStep * i))[j + k - 1] == 0)
					{
						((uchar *)(dst->imageData + dst->widthStep * i))[j] = 0;
						break;
					}
				}
			}
			else
				((uchar *)(dst->imageData + dst->widthStep * i))[j] = 0;
		}
	}

	//垂直方向的膨胀  

	for (int i = 0; i < dst->width; i++)
	{
		for (int j = 1; j < dst->height - 1; j++)
		{
			//  data = ((uchar *)(src->imageData + src->widthStep * i))[j];  
			if (((uchar *)(dst->imageData + dst->widthStep * j))[i] == 255)
			{
				((uchar *)(src->imageData + src->widthStep * j))[i] = 255;
				for (int k = 0; k < 3; k++)
				{
					//  data = ((uchar *)(src->imageData + src->widthStep * i))[j + k -1];  
					if (((uchar *)(dst->imageData + dst->widthStep * (j + k - 1)))[i] == 0)
					{
						((uchar *)(src->imageData + src->widthStep * j))[i] = 0;
						break;
					}
				}
			}
			else
				((uchar *)(src->imageData + src->widthStep * j))[i] = 0;
		}
	}
}

void FillInternalContours(IplImage *pBinary, double dAreaThre)
{
	double dConArea;
	CvSeq *pContour = NULL;
	CvSeq *pConInner = NULL;
	CvMemStorage *pStorage = NULL;
	// 执行条件   
	if (pBinary)
	{
		// 查找所有轮廓   
		pStorage = cvCreateMemStorage(0);
		cvFindContours(pBinary, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		// 填充所有轮廓   
		cvDrawContours(pBinary, pContour, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 2, CV_FILLED, 8, cvPoint(0, 0));
		// 外轮廓循环   
		int wai = 0;
		int nei = 0;
		for (; pContour != NULL; pContour = pContour->h_next)
		{
			wai++;
			// 内轮廓循环   
			for (pConInner = pContour->v_next; pConInner != NULL; pConInner = pConInner->h_next)
			{
				nei++;
				// 内轮廓面积   
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

void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	int RemoveCount = 0;       //记录除去的个数  
							   //记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查  
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
		cout << "Mode: 去除小区域. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] < 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}
	else
	{
		cout << "Mode: 去除孔洞. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] > 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}

	vector<Point2i> NeihborPos;  //记录邻域点位置  
	NeihborPos.push_back(Point2i(-1, 0));
	NeihborPos.push_back(Point2i(1, 0));
	NeihborPos.push_back(Point2i(0, -1));
	NeihborPos.push_back(Point2i(0, 1));
	if (NeihborMode == 1)
	{
		cout << "Neighbor mode: 8邻域." << endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	else cout << "Neighbor mode: 4邻域." << endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//开始检测  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********开始该点处的检查**********  
				vector<Point2i> GrowBuffer;                                      //堆栈，用于存储生长点  
				GrowBuffer.push_back(Point2i(j, i));
				Pointlabel.at<uchar>(i, j) = 1;
				int CheckResult = 0;                                               //用于判断结果（是否超出大小），0为未超出，1为超出  

				for (size_t z = 0; z<GrowBuffer.size(); z++)
				{

					for (int q = 0; q<NeihborCount; q++)                                      //检查四个邻域点  
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX<Src.cols&&CurrY >= 0 && CurrY<Src.rows)  //防止越界  
						{
							if (Pointlabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //邻域点加入buffer  
								Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //更新邻域点的检查标签，避免重复检查  
							}
						}
					}

				}
				if (GrowBuffer.size()>AreaLimit) CheckResult = 2;                 //判断结果（是否超出限定的大小），1为未超出，2为超出  
				else { CheckResult = 1;   RemoveCount++; }
				for (size_t z = 0; z<GrowBuffer.size(); z++)                         //更新Label记录  
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
				}
				//********结束该点处的检查**********  
			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//开始反转面积过小的区域  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iData = Src.ptr<uchar>(i);
		uchar* iDstData = Dst.ptr<uchar>(i);
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 2)
			{
				iDstData[j] = CheckMode;
			}
			else if (iLabel[j] == 3)
			{
				iDstData[j] = iData[j];
			}
		}
	}
	cout << RemoveCount << " objects removed." << endl;
}

int Two_Pass(const cv::Mat& binImg, cv::Mat& lableImg)    //二维图像连通区域标记，两遍扫描法，返回连通区域的个数
{
	if (binImg.empty() ||
		binImg.type() != CV_8UC1)
	{
		return 99;
	}

	// 第一个通路

	lableImg.release();
	binImg.convertTo(lableImg, CV_32SC1);

	int label = 1;
	std::vector<int> labelSet;
	labelSet.push_back(0);
	labelSet.push_back(1);

	int rows = binImg.rows - 1;
	int cols = binImg.cols - 1;
	for (int i = 1; i < rows; i++)
	{
		int* data_preRow = lableImg.ptr<int>(i - 1);
		int* data_curRow = lableImg.ptr<int>(i);
		for (int j = 1; j < cols; j++)
		{
			if (data_curRow[j] == 1)
			{
				std::vector<int> neighborLabels;
				neighborLabels.reserve(2);
				int leftPixel = data_curRow[j - 1];
				int upPixel = data_preRow[j];
				if (leftPixel > 1)
				{
					neighborLabels.push_back(leftPixel);
				}
				if (upPixel > 1)
				{
					neighborLabels.push_back(upPixel);
				}

				if (neighborLabels.empty())
				{
					labelSet.push_back(++label);  // 不连通，标签+1
					data_curRow[j] = label;
					labelSet[label] = label;
				}
				else
				{
					std::sort(neighborLabels.begin(), neighborLabels.end());
					int smallestLabel = neighborLabels[0];
					data_curRow[j] = smallestLabel;

					// 保存最小等价表
					for (size_t k = 1; k < neighborLabels.size(); k++)
					{
						int tempLabel = neighborLabels[k];
						int& oldSmallestLabel = labelSet[tempLabel];
						if (oldSmallestLabel > smallestLabel)
						{
							labelSet[oldSmallestLabel] = smallestLabel;
							oldSmallestLabel = smallestLabel;
						}
						else if (oldSmallestLabel < smallestLabel)
						{
							labelSet[smallestLabel] = oldSmallestLabel;
						}
					}
				}
			}
		}
	}
	// 更新等价对列表
	// 将最小标号给重复区域
	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];
		int preLabel = labelSet[curLabel];
		while (preLabel != curLabel)
		{
			curLabel = preLabel;
			preLabel = labelSet[preLabel];
		}
		labelSet[i] = curLabel;
	};

	for (int i = 0; i < rows; i++)
	{
		int* data = lableImg.ptr<int>(i);
		for (int j = 0; j < cols; j++)
		{
			int& pixelLabel = data[j];
			pixelLabel = labelSet[pixelLabel];
		}
	}
	return labelSet.size() - 1;
}

int icvprCcaBySeedFill(const cv::Mat& _binImg, cv::Mat& _lableImg)		//种子填充方法
{
	// connected component analysis (4-component)  
	// use seed filling algorithm  
	// 1. begin with a foreground pixel and push its foreground neighbors into a stack;  
	// 2. pop the top pixel on the stack and label it with the same label until the stack is empty  
	//   
	// foreground pixel: _binImg(x,y) = 1  
	// background pixel: _binImg(x,y) = 0  


	if (_binImg.empty() ||
		_binImg.type() != CV_8UC1)
	{
		return 1;
	}

	_lableImg.release();
	_binImg.convertTo(_lableImg, CV_32SC1);

	int label = 1;  // start by 2  
	int count = 0;

	int rows = _binImg.rows - 1;
	int cols = _binImg.cols - 1;
	std::stack<std::pair<int, int>> neighborPixels;
	for (int i = 1; i < rows - 1; i++)
	{
		int* data = _lableImg.ptr<int>(i);
		for (int j = 1; j < cols - 1; j++)
		{
			if (data[j] == 1)
			{
				neighborPixels.push(std::pair<int, int>(i, j));     // pixel position: <i,j>  
				++label;  // begin with a new label  
				while (!neighborPixels.empty())
				{
					// get the top pixel on the stack and label it with the same label  
					std::pair<int, int> curPixel = neighborPixels.top();
					int curX = curPixel.first;
					int curY = curPixel.second;
					_lableImg.at<int>(curX, curY) = label;

					// pop the top pixel  
					neighborPixels.pop();

					// push the 4-neighbors (foreground pixels)  
					if (_lableImg.at<int>(curX, curY - 1) == 1)
					{// left pixel  
						neighborPixels.push(std::pair<int, int>(curX, curY - 1));
					}
					if (_lableImg.at<int>(curX, curY + 1) == 1)
					{// right pixel  
						neighborPixels.push(std::pair<int, int>(curX, curY + 1));
					}
					if (_lableImg.at<int>(curX - 1, curY) == 1)
					{// up pixel  
						neighborPixels.push(std::pair<int, int>(curX - 1, curY));
					}
					if (_lableImg.at<int>(curX + 1, curY) == 1)
					{// down pixel  
						neighborPixels.push(std::pair<int, int>(curX + 1, curY));
					}
				}
			}
		}
	}
	return label;
}

void labelRectangle(IplImage* pic, int &left, int &right, int &up, int &down)
{
	int n = pic->width;
	int m = pic->height;
	intArrayPtr *data = new intArrayPtr[m];
	for (size_t i = 0; i < m; i++)
	{
		data[i] = new int[n];
	}

	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < m; j++)
		{

		}
	}


	for (size_t j = 0; j < n; j++)
	{
		for (size_t i = 0; i < m; i++)
		{
			if (data[i][j] != 0)
			{
				left = j;
				break;
			}
		}
		if (left != -1) break;
	}

	for (size_t j = n - 1; j >= 0; j--)
	{
		for (size_t i = 0; i < m; i++)
		{
			if (data[i][j] != 0)
			{
				right = j;
				break;
			}
		}
		if (right != -1) break;
	}

	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			if (data[i][j] != 0)
			{
				up = i;
				break;
			}
		}
		if (up != -1) break;
	}

	for (size_t i = m - 1; i >= 0; i--)
	{
		for (size_t j = 0; j < n; j++)
		{
			if (data[i][j] != 0)
			{
				down = i;
				break;
			}
		}
		if (down != -1) break;
	}

	if (left == -1)	left = 1;
	if (right == -1) right = n;
	if (up == -1) up = 1;
	if (down == -1) down = n;
}


void MatchingMethod(int, void*)
{
	/// 将被显示的原图像
	Mat img_display;
	img.copyTo(img_display);

	/// 创建输出结果的矩阵
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;

	result.create(result_cols, result_rows, CV_32FC1);

	/// 进行匹配和标准化
	matchTemplate(img, templ, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// 通过函数 minMaxLoc 定位最匹配的位置
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	/// 对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值代表更高的匹配结果. 而对于其他方法, 数值越大匹配越好
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	/// 让我看看您的最终结果
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);

	imshow("image_window", img_display);
	imshow("result_window", result);

	return;
}

//根据直方图数据绘制并显示直方图图像  
void myShow_Histogram(MatND &hist, int scale) {

	int hist_height = 256;
	int bins = 256;
	double max_val;

	minMaxLoc(hist, 0, &max_val, 0, 0);

	Mat hist_img = Mat::zeros(hist_height, bins*scale, CV_8UC3);

	cout << "max_val = " << max_val << endl;
	for (int i = 0; i<bins; i++)
	{
		float bin_val = hist.at<float>(i); //  


		int intensity = cvRound(bin_val*hist_height / max_val);  //要绘制的高度    

		cv::rectangle(hist_img, Point(i*scale, hist_height - 1),
			Point((i + 1)*scale - 1, hist_height - intensity),
			CV_RGB(255, 255, 255));
	}
	imshow("Gray Histogram2", hist_img);
}
