#include "stdafx.h"
#include "sc.h"

//#define PI  3.1415926
typedef int * intArrayPtr;
//double ** theta_array;
//double ** r_array;	//��������ľ�������
//double ** total_r_array;	//����֮ǰ�ľ�������
//double ** SC;
/// ȫ�ֱ���
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

	//ˮƽ����ĸ�ʴ  
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
	//��ֱ����ĸ�ʴ  
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

	//ˮƽ���������  
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

	//��ֱ���������  

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
*/
void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	int RemoveCount = 0;       //��¼��ȥ�ĸ���  
							   //��¼ÿ�����ص����״̬�ı�ǩ��0����δ��飬1�������ڼ��,2�����鲻�ϸ���Ҫ��ת��ɫ����3������ϸ������  
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
		cout << "Mode: ȥ��С����. ";
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
		cout << "Mode: ȥ���׶�. ";
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

	vector<Point2i> NeihborPos;  //��¼�����λ��  
	NeihborPos.push_back(Point2i(-1, 0));
	NeihborPos.push_back(Point2i(1, 0));
	NeihborPos.push_back(Point2i(0, -1));
	NeihborPos.push_back(Point2i(0, 1));
	if (NeihborMode == 1)
	{
		cout << "Neighbor mode: 8����." << endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	else cout << "Neighbor mode: 4����." << endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//��ʼ���  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********��ʼ�õ㴦�ļ��**********  
				vector<Point2i> GrowBuffer;                                      //��ջ�����ڴ洢������  
				GrowBuffer.push_back(Point2i(j, i));
				Pointlabel.at<uchar>(i, j) = 1;
				int CheckResult = 0;                                               //�����жϽ�����Ƿ񳬳���С����0Ϊδ������1Ϊ����  

				for (size_t z = 0; z<GrowBuffer.size(); z++)
				{

					for (int q = 0; q<NeihborCount; q++)                                      //����ĸ������  
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX<Src.cols&&CurrY >= 0 && CurrY<Src.rows)  //��ֹԽ��  
						{
							if (Pointlabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //��������buffer  
								Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //���������ļ���ǩ�������ظ����  
							}
						}
					}

				}
				if (GrowBuffer.size()>AreaLimit) CheckResult = 2;                 //�жϽ�����Ƿ񳬳��޶��Ĵ�С����1Ϊδ������2Ϊ����  
				else { CheckResult = 1;   RemoveCount++; }
				for (size_t z = 0; z<GrowBuffer.size(); z++)                         //����Label��¼  
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
				}
				//********�����õ㴦�ļ��**********  
			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//��ʼ��ת�����С������  
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

int Two_Pass(const cv::Mat& binImg, cv::Mat& lableImg)    //��άͼ����ͨ�����ǣ�����ɨ�跨��������ͨ����ĸ���
{
	if (binImg.empty() ||
		binImg.type() != CV_8UC1)
	{
		return 99;
	}

	// ��һ��ͨ·

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
					labelSet.push_back(++label);  // ����ͨ����ǩ+1
					data_curRow[j] = label;
					labelSet[label] = label;
				}
				else
				{
					std::sort(neighborLabels.begin(), neighborLabels.end());
					int smallestLabel = neighborLabels[0];
					data_curRow[j] = smallestLabel;

					// ������С�ȼ۱�
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
	// ���µȼ۶��б�
	// ����С��Ÿ��ظ�����
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

int icvprCcaBySeedFill(const cv::Mat& _binImg, cv::Mat& _lableImg)		//������䷽��
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
	/// ������ʾ��ԭͼ��
	Mat img_display;
	img.copyTo(img_display);

	/// �����������ľ���
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;

	result.create(result_cols, result_rows, CV_32FC1);

	/// ����ƥ��ͱ�׼��
	matchTemplate(img, templ, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// ͨ������ minMaxLoc ��λ��ƥ���λ��
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	/// ���ڷ��� SQDIFF �� SQDIFF_NORMED, ԽС����ֵ������ߵ�ƥ����. ��������������, ��ֵԽ��ƥ��Խ��
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	/// ���ҿ����������ս��
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);

	imshow("image_window", img_display);
	imshow("result_window", result);

	return;
}

//����ֱ��ͼ���ݻ��Ʋ���ʾֱ��ͼͼ��  
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


		int intensity = cvRound(bin_val*hist_height / max_val);  //Ҫ���Ƶĸ߶�    

		cv::rectangle(hist_img, Point(i*scale, hist_height - 1),
			Point((i + 1)*scale - 1, hist_height - intensity),
			CV_RGB(255, 255, 255));
	}
	imshow("Gray Histogram2", hist_img);
}
