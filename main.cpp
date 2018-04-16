#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>  
#include <opencv2/core/core.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <cstdio> 

using namespace std;
using namespace cv;

////////////////////////////////
//二值化方法  
Mat IntersectBinaryMat(Mat & m1, Mat & m2) {
	assert(m1.rows == m2.rows && m1.cols == m2.cols);
	assert(m1.type() == CV_8U && m2.type() == CV_8U);
	Mat m3;
	m3.create(m1.size(), m1.type());
	int IMAGE_WIDTH = m1.cols;
	int IMAGE_HEIGHT = m1.rows;
	 
	for (int i = 0; i<IMAGE_HEIGHT; i++) {
		for (int j = 0; j < IMAGE_WIDTH; j++) {
			int index = i*IMAGE_WIDTH + j;
			if (m1.data[index] == 255 && m2.data[index] == 255)
				m3.data[index] = 255;
			else
				m3.data[index] = 0;
		}
	}
	return m3;
}

Mat SubtractMat(Mat & m1, Mat & m2) {
	assert(m1.rows == m2.rows && m1.cols == m2.cols);
	assert(m1.type() == CV_8U && m2.type() == CV_8U);
	Mat m3;
	m3.create(m1.size(), m1.type());
	int IMAGE_WIDTH = m1.cols;
	int IMAGE_HEIGHT = m1.rows;

	for (int i = 0; i<IMAGE_HEIGHT; i++) {
		for (int j = 0; j < IMAGE_WIDTH; j++) {
			int index = i*IMAGE_WIDTH + j;
			m3.data[index] = m1.data[index] - m2.data[index];
		}
	}
	return m3;
}

void get_background(unsigned char * greyImage, unsigned char * foreImage, unsigned char * backImage, int w, int h, int windowSize) {
	int IMAGE_WIDTH = w;
	int IMAGE_HEIGHT = h;
	int whalf = windowSize >> 1;

	for (int i = 0; i<IMAGE_HEIGHT; i++) {
		for (int j = 0; j < IMAGE_WIDTH; j++) {
			int index = i*IMAGE_WIDTH + j;

			if (foreImage[index]) {
				//cout << "[Foreground] " << i << ", " << j << endl;
				/*Is foreground*/
				int xmin = max(0, i - whalf);
				int ymin = max(0, j - whalf);
				int xmax = min(IMAGE_HEIGHT - 1, i + whalf);
				int ymax = min(IMAGE_WIDTH - 1, j + whalf);
				int sum = 0;
				int cnt = 0;
				for (int ii = xmin; ii <= xmax; ii++) {
					for (int jj = xmin; jj <= ymax; jj++) {
						int index2 = ii*IMAGE_WIDTH + jj;
						if (foreImage[index2] == 0) {
							sum += greyImage[index2];
							cnt++;
						}
					}
				}
				if (cnt != 0)
					backImage[index] = sum / cnt;
				else
					backImage[index] = 0;
			}
			else {
				//cout << "[Background] " << i << ", " << j << endl;
				/*Is background*/
				backImage[index] = greyImage[index];
			}
		}
	}
}

void sauvola_kernel(unsigned char * greyImage, unsigned char * biImage, int w, int h, float k, int windowSize)
{
	int whalf = windowSize >> 1;

	int i, j;
	int IMAGE_WIDTH = w;
	int IMAGE_HEIGHT = h;
	// create the integral image  
	unsigned long * integralImg = (unsigned long*)malloc(IMAGE_WIDTH*IMAGE_HEIGHT * sizeof(unsigned long*));
	unsigned long * integralImgSqrt = (unsigned long*)malloc(IMAGE_WIDTH*IMAGE_HEIGHT * sizeof(unsigned long*));
	int sum = 0;
	int sqrtsum = 0;
	int index;
	for (i = 0; i<IMAGE_HEIGHT; i++)
	{
		// reset this column sum  
		sum = 0;
		sqrtsum = 0;

		for (j = 0; j<IMAGE_WIDTH; j++)
		{
			index = i*IMAGE_WIDTH + j;

			sum += greyImage[index];
			sqrtsum += greyImage[index] * greyImage[index];

			if (i == 0)
			{
				integralImg[index] = sum;
				integralImgSqrt[index] = sqrtsum;
			}
			else
			{
				integralImgSqrt[index] = integralImgSqrt[(i - 1)*IMAGE_WIDTH + j] + sqrtsum;
				integralImg[index] = integralImg[(i - 1)*IMAGE_WIDTH + j] + sum;
			}
		}
	}

	//Calculate the mean and standard deviation using the integral image  
	int xmin, ymin, xmax, ymax;
	double mean, std, threshold;
	double diagsum, idiagsum, diff, sqdiagsum, sqidiagsum, sqdiff, area;

	for (i = 0; i<IMAGE_WIDTH; i++) {
		for (j = 0; j<IMAGE_HEIGHT; j++) {
			xmin = max(0, i - whalf);
			ymin = max(0, j - whalf);
			xmax = min(IMAGE_WIDTH - 1, i + whalf);
			ymax = min(IMAGE_HEIGHT - 1, j + whalf);

			area = (xmax - xmin + 1) * (ymax - ymin + 1);
			if (area <= 0)
			{
				biImage[i * IMAGE_WIDTH + j] = 255;
				continue;
			}

			if (xmin == 0 && ymin == 0) {
				diff = integralImg[ymax * IMAGE_WIDTH + xmax];
				sqdiff = integralImgSqrt[ymax * IMAGE_WIDTH + xmax];
			}
			else if (xmin > 0 && ymin == 0) {
				diff = integralImg[ymax * IMAGE_WIDTH + xmax] - integralImg[ymax * IMAGE_WIDTH + xmin - 1];
				sqdiff = integralImgSqrt[ymax * IMAGE_WIDTH + xmax] - integralImgSqrt[ymax * IMAGE_WIDTH + xmin - 1];
			}
			else if (xmin == 0 && ymin > 0) {
				diff = integralImg[ymax * IMAGE_WIDTH + xmax] - integralImg[(ymin - 1) * IMAGE_WIDTH + xmax];
				sqdiff = integralImgSqrt[ymax * IMAGE_WIDTH + xmax] - integralImgSqrt[(ymin - 1) * IMAGE_WIDTH + xmax];;
			}
			else {
				diagsum = integralImg[ymax * IMAGE_WIDTH + xmax] + integralImg[(ymin - 1) * IMAGE_WIDTH + xmin - 1];
				idiagsum = integralImg[(ymin - 1) * IMAGE_WIDTH + xmax] + integralImg[ymax * IMAGE_WIDTH + xmin - 1];
				diff = diagsum - idiagsum;

				sqdiagsum = integralImgSqrt[ymax * IMAGE_WIDTH + xmax] + integralImgSqrt[(ymin - 1) * IMAGE_WIDTH + xmin - 1];
				sqidiagsum = integralImgSqrt[(ymin - 1) * IMAGE_WIDTH + xmax] + integralImgSqrt[ymax * IMAGE_WIDTH + xmin - 1];
				sqdiff = sqdiagsum - sqidiagsum;
			}

			mean = diff / area;
			std = sqrt((sqdiff - diff*diff / area) / (area - 1));
			threshold = mean*(1 + k*((std / 128) - 1));
			//cout << "pixel[" << i << "][" << j << "]" << ", val: " << int(greyImage[j*IMAGE_WIDTH + i])  << ",threshold: " << threshold << endl;
			if (greyImage[j*IMAGE_WIDTH + i] < threshold)
				biImage[j*IMAGE_WIDTH + i] = 0;
			else
				biImage[j*IMAGE_WIDTH + i] = 255;
		}
	}

	free(integralImg);
	free(integralImgSqrt);
}
/* plate_color_type: true - blue plate, false - yellow plate */
Mat SauvolaBasedBinaryzation(Mat & srcImage, bool plate_color_type) {

	//assert(foreBImage.rows == srcImage.rows && foreBImage.cols == srcImage.cols);
	//assert(backBImage.rows == foreBImage.rows && foreBImage.cols == backBImage.cols);

	/* parameters: optimum values */
	float k_big = 0.05;
	float k_small = 0.01;
	int windowSize_big = 48;
	int windowSize_small = 4;
	int bilateralFilterKernelSize = 8;

	/* intermediate images */
	Mat greyImage;
	Mat foreBImage, backBImage; // foreground and background using big/coarse parameters
	Mat foreSImage, backSImage; // foreground and background using small/fine parameters
	Mat residualImage, residualImageB;
	Mat intersectImage;

	foreBImage.create(srcImage.size(), srcImage.type());
	backBImage.create(srcImage.size(), srcImage.type());

	foreSImage.create(srcImage.size(), srcImage.type());
	backSImage.create(srcImage.size(), srcImage.type());

	residualImageB.create(srcImage.size(), srcImage.type());

	// Bilateral Filter
	bilateralFilter(srcImage, greyImage, bilateralFilterKernelSize, bilateralFilterKernelSize * 2, bilateralFilterKernelSize / 2);

	if (!plate_color_type) {
		Mat fullImage;
		fullImage.create(greyImage.size(), greyImage.type());
		fullImage = Scalar::all(255);
		greyImage = SubtractMat(fullImage, greyImage);
	}

	//cvtColor( bilateralFilteredImage, greyImage, CV_BGR2GRAY );
	blur(greyImage, greyImage, Size(3, 3));

	int w = greyImage.cols;
	int h = greyImage.rows;

	assert(greyImage.type() == CV_8U && foreBImage.type() == CV_8U);

	sauvola_kernel(greyImage.data, foreBImage.data, w, h, k_big, windowSize_big);
	get_background(greyImage.data, foreBImage.data, backBImage.data, w, h, windowSize_big);
	sauvola_kernel(greyImage.data, foreSImage.data, w, h, k_small, windowSize_small);

	residualImage = SubtractMat(greyImage, backBImage);

	/*TODO: This number is hard coded right now. Needs a better way*/
	unsigned threshold_low = 25;
	threshold(residualImage, residualImageB, threshold_low, 255, THRESH_BINARY);
	intersectImage = IntersectBinaryMat(residualImageB, foreSImage);

	return intersectImage;
}

////////////////////////////////////////////////////////
vector<int> getHorProjImage(Mat threshold_mat) {   //水平投影 为了判断单层还是双层车牌
	int height = threshold_mat.rows;
	int width = threshold_mat.cols;
	int tmp = 0;
	vector < int>sum_point;
	for (int i = 0; i < height; ++i)
	{
		tmp = 0;
		for (int j = 0; j < width; ++j)
		{
			if (threshold_mat.at<uchar>(i, j) == 255)
			{
				++tmp;
			}
		}
		sum_point.push_back(tmp);
	}
	return sum_point;
}

int JudgeDouble(Mat threshold_mat) {  //判断车牌是双层还是单层
	vector<int> sum_point = getHorProjImage(threshold_mat);
	int height = threshold_mat.rows;
	int width = threshold_mat.cols;

	int tmp_num = 0;
	double a = 0.4;
	double b = 0.6;
	cout << a << endl;
	cout << b << endl;
	for (int i = a*height; i < b*height ; i++) {
		cout << sum_point[i] << endl;
		if (sum_point[i] < 20) tmp_num++;
	}
	if (tmp_num >0) {
		return 0;     //返回0 则是双层车牌
	}
	else {
		return 1;    //返回1 则是单层车牌
	}
}
//////////////////////////////////////////
//围绕矩形中心缩放  目的是为了让取到的字符边框大一点，比如“1”这样的，方便后面识别
Rect rectCenterScale(Rect rect, Size size){
	rect = rect + size;
	Point pt;
	pt.x = cvRound(size.width / 2.0);
	pt.y = cvRound(size.height / 2.0);
	return (rect - pt);
}
////////////////////////////////////////////////
bool LessSort(Rect a, Rect b) {   //快排比较函数 上下排序
	Point pa = a.tl();
	Point pb = b.tl();
	return (pa.y<pb.y); 
}

bool HorSort(Rect a, Rect b) {   //快排比较函数 左右排序
	Point pa = a.tl();
	Point pb = b.tl();
	return (pa.x<pb.x);
}
/*
Rect chineseChar(vector<Rect> rects) {
	Rect chinese_char;
	if (rects.size() > 7) {    //中文字符被划成两半， 则总计的轮廓会大于7 这排序后的第一二两个轮廓就是一个
		chinese_char = rects[0] | rects[1];
	}
	return chinese_char;
}

*/
Rect chineseChar() {  //经验值位置
	Rect rect(45, 3, 70, 50);	
	return rect;
}

vector<Rect> sortContours(vector<vector<Point> >contours) { //通过这个方法得到除了中文字符之外的其他六个字符
	vector<Rect> rects;
	for (int i = 0; i < contours.size(); i++) {
		Rect rect = boundingRect(Mat(contours[i]));
		Point tl = rect.tl();
		Point br = rect.br();
		if (rect.area() < 250) {  //去掉小面积的轮廓， 比如中文与城市英文之间的点 或者柳丁
			continue;
		}
		if (tl.y < 3 || br.y > 128 || tl.x < 3 || br.x>265) { //去掉轮廓周边
			continue;
		}
		rects.push_back(rect);
	}
	//利用快排给 tl坐标排序，上到下 然后左到右排序
	sort(rects.begin(), rects.end(), LessSort);
	sort(rects.begin(), rects.end()-5, HorSort);
	sort(rects.end()-5, rects.end(), HorSort);
	Rect tmp = chineseChar();
	vector<Rect> rects_new;
	rects_new.push_back(tmp);
	int size_rects = rects.size();

	for (int i = 6; i > 0; i--) {
		rects_new.push_back(rects[size_rects - i]);
	}
	return rects_new;
}


///////////
//图片归一化
Mat getSameMat(Mat mat) {   //参数带调整，若这里调整，这上面中文字符识别参数也要修改，同时 sortcounters函数里面的去掉边缘等数据也要修改
	Size dsize = Size(270, 130);
	Mat image2 = Mat(dsize, CV_32S);
	resize(mat, image2, dsize);
	return image2;
}




int main() {
	Mat srcImg = imread("double1.png"); // 读取源图像  
	cout << srcImg.size() << endl;
	if (!srcImg.data) {
		cout<<"double1.png不存在！";
		return -1;
	}
	imshow("srcImage", srcImg);
	waitKey();
    srcImg = getSameMat(srcImg);
	imshow("same", srcImg);
    waitKey();


	Mat gray, edge, dst; // 
	cvtColor(srcImg, gray, CV_BGR2GRAY); // 转为灰度图像 
	imshow("gray", gray);
	waitKey();
	destroyWindow("gray");
	//GaussianBlur(gray, edge, Size(3, 3),0,0);  
	dst = SauvolaBasedBinaryzation(gray, 0);
	cout << JudgeDouble(dst) << endl;  //判断车牌是双层还是单层 0是双层 1 是单层
	imshow("dst", dst);
	waitKey();

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(dst, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	/////
	for (int i = 0; i < contours.size(); i++) {
		Rect rect = boundingRect(Mat(contours[i]));
		Point tl = rect.tl();
		Point br = rect.br();
		cout << tl << endl;
		cout << br << endl;
		cout << endl;
	}
	//////
	vector<Rect> rects = sortContours(contours);
	Point p1(2, 2);//矩形框显示缩放
	Size size1(2, 2);//矩形缩放
	Mat img1;
	for (int i = 0; i < rects.size(); i++) {
		rectangle(srcImg, rects[i].tl() - p1, rects[i].br() + p1, Scalar(255, 0, 0));
		rects[i] = rectCenterScale(rects[i], size1); //将轮廓扩大点
		cout << rects[i].size() << "第" << i << "张b" << endl;
		Mat roi = dst(rects[i]);
	    roi.convertTo(img1, roi.type());
		imshow("show", img1);
		waitKey();
		destroyWindow("show");

	}
	imshow("contoursImg", srcImg);
	waitKey();
	return 0;





}