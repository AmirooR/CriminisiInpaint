#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

int win_size = 9;
int stride = 3;

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        printf("Usage: %s src mask output\n",argv[0]); 
        return -1;
    }
    Mat src, src_gray, mask, mask_gray, grad_x, grad_y, inpaint_me;

    // init src, mask
    src = imread(argv[1]);
    mask = imread(argv[2]);
    if(!src.data || !mask.data){return -1;}
    //Experimental resize!
    //resize(src, src, Size(0,0), 0.5, 0.5);
    //resize(mask, mask, Size(0,0), 0.5, 0.5);
    src.copyTo(inpaint_me, mask);// inpaint_me is a bgr mat
    cvtColor(inpaint_me, inpaint_me, CV_BGR2Lab);//inpaint_me is now in Lab space

    //Computing gradients
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT);
    cvtColor(src, src_gray, CV_BGR2GRAY);
    cvtColor(mask, mask_gray, CV_BGR2GRAY);
    Mat src_gray_f;
    src_gray.convertTo(src_gray_f, CV_32FC1, 1./255.);
    Sobel(src_gray_f, grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Sobel(src_gray_f, grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);


    // create patches inside I - \omega, and find boundary
    // patches si matrix, each row is a single patch vector (casted to float)
    int num_patches = 0;
    int num_vec = (mask.rows - win_size)*(mask.cols - win_size)/(stride * stride);
    int f_vec_len = (win_size * win_size)*src.channels();
    Mat patches(num_vec, f_vec_len, CV_8UC1);
    for(int y =0; y < mask.rows - win_size; y+=stride)
    {
        for(int x =0; x < mask.cols - win_size; x+=stride)
        {
            //create patch
            Mat patch_mat = inpaint_me(Range(y,y+win_size), Range(x,x+win_size)).clone();
            Mat mask_patch = mask_gray(Range(y,y+win_size), Range(x,x+win_size)).clone();
            patch_mat = patch_mat.reshape(1,1);

            if( countNonZero(mask_patch) == win_size * win_size )
            {
                for(int j =0; j < f_vec_len; j++)
                    patches.at<uchar>(num_patches,j) = patch_mat.at<uchar>(0,j);
                num_patches++;

            }
        }
    }
    cout<<num_patches<<" patches created!"<<endl;

    //compute C(p) i.e. by mean filtering
    Mat mask_gray_f, C_p, filled_f, filled;
    mask_gray.convertTo(mask_gray_f, CV_32FC1, 1./255.);
    filled = mask_gray.clone();


    int iter = 0;
    while(countNonZero(filled) != mask.rows * mask.cols)
    {


        filled.convertTo(filled_f, CV_32FC1, 1./255.); 
        blur(filled_f, C_p, Size(win_size, win_size), Point(-1,-1) );

        // find boundaries!
        vector<vector<Point> > contours;
        Mat not_filled = 255 - filled;
        findContours(not_filled, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

        // find pixel with the highest priority
        float max_p_p = -1;
        int max_c_idx = -1;
        int max_p_idx = -1;

        
        for(int c = 0; c < contours.size(); c++)
        {
            for(int p = 0; p < contours[c].size(); p++)
            {
                Point this_p = contours[c][p];
                int last_point_idx = (p-1) < 0 ? contours[c].size() - 1: p-1;
                int next_point_idx = (p+1)%contours[c].size();
                Point last_p = contours[c][last_point_idx];
                Point next_p = contours[c][next_point_idx];
                Point boundary_dir_p = next_p - last_p;
                Point2f grad_p(grad_x.at<float>(this_p), grad_y.at<float>(this_p) );
                float this_c_p = C_p.at<float>(this_p);
                float priority = this_c_p * fabs(grad_p.x*boundary_dir_p.x + grad_p.y*boundary_dir_p.y);
                if(priority > max_p_p)
                {
                    max_p_p = priority;
                    max_c_idx = c;
                    max_p_idx = p;
                }

            }
        }

        // find min distance
        Point start = contours[max_c_idx][max_p_idx] - Point(win_size/2, win_size/2);
        Mat max_mask_patch = filled(Range(start.y, start.y+win_size), Range(start.x, start.x+win_size) ).clone();
        Mat max_patch_mat  = inpaint_me(Range(start.y, start.y+win_size),Range(start.x, start.x+win_size) ).clone();

        max_mask_patch = max_mask_patch.reshape(1,1);
        max_patch_mat =  max_patch_mat.reshape(1,1);
        int min_dist = 255*255*3 + 1;
        int min_dist_idx = 0;
        for(int y = 0; y < num_patches; y++)
        {
            int dist = 0;
            for(int x = 0; x < f_vec_len; x++)
            {
                if(max_mask_patch.at<uchar>(0,x/3) > 0 )
                {
                    int diff = patches.at<uchar>(y,x) - max_patch_mat.at<uchar>(0,x);
                    dist += diff*diff;
                }
            }
            if(dist < min_dist)
            {
                min_dist = dist;
                min_dist_idx = y;
            }
        }
        
        Point fill_me = contours[max_c_idx][max_p_idx];

        filled.at<uchar>(fill_me) = 255;
        inpaint_me.at<Vec3b>(fill_me) = patches.at<Vec3b>(min_dist_idx, (win_size*win_size/2));
        iter++;
        cout<<"Iter: "<<iter<<endl;

        //Save temporary results in save/ folder
        if(iter % 100 == 0)
        {
            Mat temp = inpaint_me.clone();
            cvtColor(temp, temp, CV_Lab2BGR);
            char name[256]={0};
            sprintf(name,"save/%d.png",iter);
            imwrite(name, temp);
        }
    }

    cvtColor(inpaint_me, inpaint_me, CV_Lab2BGR);
    imwrite(argv[3], inpaint_me);
    return 0;
}
