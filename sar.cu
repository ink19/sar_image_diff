#include <stdio.h>  
#include <cuda_runtime.h>  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libbmp.h"

#define B_D_2 (4)
#define B_R_2 (0.01)
#define DOUBLE_FILTER_N (10)

//rgb灰度化
__global__ void rgb2gray_kernel(int *gray_data, int *rgb_data, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        gray_data[tid] = (rgb_data[tid * 3]*76 + rgb_data[tid * 3 + 1]*150 + rgb_data[tid * 3 + 2]*30) >> 8;
        tid += blockDim.x * gridDim.x;
    }
}
//扩展图像
__global__ void extimg_kernel(int *ext_data, int *data, int width, int heigth, int ext_length) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int ext_heigth = (heigth + ext_length * 2), ext_width = (width + ext_length * 2);
    int size = ext_heigth * ext_width;
    int now_width = 0, now_heigth = 0, shadow_width, shadow_heigth, shadow_index;
    while (index < size) {
        now_width = index % ext_width;
        now_heigth = index / ext_width;
        shadow_width = abs(now_width -ext_length);
        shadow_heigth = abs(now_heigth -ext_length);
        shadow_width = width - 1 - abs(shadow_width - width + 1);
        shadow_heigth = heigth - 1 - abs(shadow_heigth - heigth + 1);
        shadow_index = shadow_heigth * width + shadow_width;
        ext_data[index] = data[shadow_index];
        index += blockDim.x * gridDim.x;
    }
}

//差异图
__global__ void diffimg_kernel(double *dst_data, int *src1_data, int *src2_data, int width, int height, int r) {
    int min_sum, max_sum;
    int index = threadIdx.x + blockIdx.x * blockDim.x, size = width * height;
    int ext_width = width + 2*r, loop_h, loop_w, loop_index, now_height, now_width;
    while (index < size) {
        min_sum = max_sum= 0;
        now_height = index / width + r;
        now_width = index % width + r;
        for(loop_h = -r; loop_h <= r; ++loop_h) {
            for(loop_w = -r; loop_w <= r; ++loop_w) {
                loop_index = (loop_h + now_height) * ext_width + loop_w + now_width; 
                min_sum += src1_data[loop_index] + src2_data[loop_index] - abs(src1_data[loop_index] - src2_data[loop_index]);
                max_sum += src1_data[loop_index] + src2_data[loop_index] + abs(src1_data[loop_index] - src2_data[loop_index]);
            }
        }
        dst_data[index] = (min_sum + 1) / (double)(max_sum + 1);
        index += blockDim.x * gridDim.x;
    }
}

//扩展浮点型
__global__ void extdouble_kernel(double *ext_data, double *data, int width, int heigth, int ext_length) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int ext_heigth = (heigth + ext_length * 2), ext_width = (width + ext_length * 2);
    int size = ext_heigth * ext_width;
    int now_width = 0, now_heigth = 0, shadow_width, shadow_heigth, shadow_index;
    while (index < size) {
        now_width = index % ext_width;
        now_heigth = index / ext_width;
        shadow_width = abs(now_width -ext_length);
        shadow_heigth = abs(now_heigth -ext_length);
        shadow_width = width - 1 - abs(shadow_width - width + 1);
        shadow_heigth = heigth - 1 - abs(shadow_heigth - heigth + 1);
        shadow_index = shadow_heigth * width + shadow_width;
        ext_data[index] = data[shadow_index];
        index += blockDim.x * gridDim.x;
    }
}

//读取图像
int rgb2gray(const char *filename, int **dev_data, int *out_width, int *out_height) {
    bmp_img bmp_info;
    int length;
    int *pix_array;
    int *dev_gray, *dev_rgb;

    bmp_img_read(&bmp_info ,filename);
    int height = bmp_info.img_header.biHeight;
    int width = bmp_info.img_header.biWidth;
    length = bmp_info.img_header.biWidth * bmp_info.img_header.biHeight;
    *out_width = width;
    *out_height = height;

    pix_array = (int *)malloc(sizeof(int) * length * 3);
    
    for(int loop_h = 0; loop_h < height; ++loop_h) {
        for(int loop_w = 0; loop_w < width; ++loop_w) {
            *(pix_array + ((loop_h) * (width) + loop_w) * 3) = (int)(bmp_info.img_pixels[loop_h][loop_w].red);
            *(pix_array + ((loop_h) * (width) + loop_w) * 3 + 1) = (int)(bmp_info.img_pixels[loop_h][loop_w].green);
            *(pix_array + ((loop_h) * (width) + loop_w) * 3 + 2) = (int)(bmp_info.img_pixels[loop_h][loop_w].blue);
        }
    }
    
    cudaMalloc((void **)&dev_gray, sizeof(int) * length);
    cudaMalloc((void **)&dev_rgb, sizeof(int) * length * 3);
    cudaMemcpy(dev_rgb, pix_array, sizeof(int) * length * 3, cudaMemcpyHostToDevice);
    rgb2gray_kernel<<<128, 512>>>(dev_gray, dev_rgb, length);
    cudaThreadSynchronize();
    cudaFree(dev_rgb);
    free(pix_array);
    bmp_img_free(&bmp_info);
    *dev_data = dev_gray;
    return 0;
}

__global__ void double_filter_kernel(double *gray_data, double *result, int width, int height) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int now_height, now_width, now_index, size = width * height;
    int loop_width, loop_height, loop_index, ext_width = width + 2 * DOUBLE_FILTER_N;
    double W, sum_W_with_f, sum_W;
    while (tid < size) {
        now_height = tid / width + DOUBLE_FILTER_N;
        now_width = tid % width + DOUBLE_FILTER_N;
        now_index = now_height * ext_width + now_width;
        sum_W_with_f = 0;
        sum_W = 0;
        for(loop_height = -DOUBLE_FILTER_N; loop_height < DOUBLE_FILTER_N; ++loop_height) {
            for(loop_width = -DOUBLE_FILTER_N; loop_width < DOUBLE_FILTER_N; ++loop_width) {
                loop_index = (now_width + loop_width) + (now_height + loop_height) * ext_width;
                W = exp(-(loop_width * loop_width + loop_height * loop_height)/(2.0 * B_D_2) - (gray_data[loop_index] - gray_data[now_index]) * (gray_data[loop_index] - gray_data[now_index])/(2.0 * B_R_2));
                sum_W += W;
                sum_W_with_f += gray_data[loop_index] * W;
            }
        }
        result[tid] = (sum_W_with_f/sum_W);
        tid += blockDim.x * gridDim.x;
    }
}



int main(int argc, char *argv[]) {  
    if(argc <= 3) {
        printf("参数不足\n");
        return 0;
    }
    int *img1_data, img1_width, img1_height, *img2_data, img2_width, img2_height, *img1_ext, *img2_ext;
    double *diff_img, *ext_diff_img;
    int filter_times;
    rgb2gray(argv[1], &img1_data, &img1_width, &img1_height);
    rgb2gray(argv[2], &img2_data, &img2_width, &img2_height);
    cudaMalloc((void **)&img1_ext, sizeof(int) * (img1_width + 6) * (img1_height + 6));
    cudaMalloc((void **)&img2_ext, sizeof(int) * (img2_width + 6) * (img2_height + 6));
    extimg_kernel<<<128, 512>>>(img1_ext, img1_data, img1_width, img1_height, 3);
    extimg_kernel<<<128, 512>>>(img2_ext, img2_data, img2_width, img2_height, 3);
    cudaMalloc((void **)&diff_img, sizeof(double) * img1_width * img2_height);
    cudaMalloc((void **)&ext_diff_img, sizeof(double) * (img2_width + DOUBLE_FILTER_N * 2) * (img2_height + DOUBLE_FILTER_N * 2));
    cudaThreadSynchronize();
    diffimg_kernel<<<128, 512>>>(diff_img, img1_ext, img2_ext, img1_width, img1_height, 3);
    cudaFree(img2_data);
    cudaFree(img1_data);
    cudaThreadSynchronize();
    filter_times = atoi(argv[3]);
    for(int loop = 0; loop < filter_times; ++loop) {
        extdouble_kernel<<<128, 512>>>(ext_diff_img, diff_img, img1_width, img1_height, DOUBLE_FILTER_N);
        cudaThreadSynchronize();
        double_filter_kernel<<<128, 512>>>(ext_diff_img, diff_img, img1_width, img1_height);
        cudaThreadSynchronize();
    }
    
    cudaFree(img1_ext);
    cudaFree(img2_ext);
    cudaFree(ext_diff_img);
    // cudaFree(diff_img);
    double *final_data = (double *)malloc(sizeof(double) * img1_height * img1_width);
    cudaMemcpy(final_data, diff_img, sizeof(double) * img1_height * img1_width, cudaMemcpyDeviceToHost);
    // for(int loop_h = 0; loop_h < img1_height; ++loop_h) {
    //     for(int loop_w = 0; loop_w < img1_width; ++loop_w) {
    //         printf((loop_w == (img1_width - 1))?"%lf":"%lf,", final_data[loop_h * img1_width + loop_w]);
    //     }
    //     printf("\n");
    // }
    // free(final_data);
    cudaFree(diff_img);
    int *result = (int *)malloc(sizeof(int) * img1_height * img1_width);
    double *distence = (double *)malloc(sizeof(double) * img1_height * img1_width * 2);
    double center[2],center_sum[2], center_sum2[2],tmp_data_center[2];
    double sdist = 0, sdist2 = 0;
    center[0] = final_data[0];
    center[1] = final_data[1];
    
    do{
        sdist2 = sdist;
        center_sum[0] = 0;
        center_sum[1] = 0;
        center_sum2[0] = 0;
        center_sum2[1] = 0;
        for(int loop = 0; loop < img1_height * img1_width; ++loop) {
            tmp_data_center[0] = fabs(center[0] - final_data[loop]);
            tmp_data_center[1] = fabs(center[1] - final_data[loop]);
            distence[loop * 2] = (tmp_data_center[0] + tmp_data_center[1])/(tmp_data_center[0] + 0.0000001);
            center_sum[0] += distence[loop * 2];
            center_sum2[0] +=  distence[loop * 2] * final_data[loop];
            distence[loop * 2 + 1] = (tmp_data_center[0] + tmp_data_center[1])/(tmp_data_center[1] + 0.0000001);
            center_sum[1] += distence[loop * 2 + 1];
            center_sum2[1] +=  distence[loop * 2 + 1] * final_data[loop];
        }
        sdist = center[0] + center[1];
        center[0] = center_sum2[0]/center_sum[0];
        center[1] = center_sum2[1]/center_sum[1];
        sdist = abs(sdist - center[0] - center[1]);
        fprintf(stderr, "%lf\n", fabs(sdist - sdist2));
    } while(fabs(sdist - sdist2) > 0.00001);
    
    for(int loop = 0; loop < img1_height * img1_width; ++loop) {
        result[loop] = distence[loop * 2] > distence[loop * 2 + 1]? 1: 0;
    }

	bmp_img img;
	bmp_img_init_df (&img, img1_width, img1_height);
	char filename[20];
    sprintf(filename, "diff_%d.bmp", filter_times);
	// Draw a checkerboard pattern:
	for (int y = 0; y < img1_height; y++)
	{
		for (int x = 0; x < img1_width; x++)
		{
			if (result[y * img1_width + x]) {
				bmp_pixel_init (&img.img_pixels[y][x], 255, 255, 255);
			} else {
				bmp_pixel_init (&img.img_pixels[y][x], 0, 0, 0);
			}
		}
	}
	free(result);
    free(distence);
	bmp_img_write (&img, filename);
	bmp_img_free (&img);
    return 0;
}
