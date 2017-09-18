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

/*
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
*/
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
