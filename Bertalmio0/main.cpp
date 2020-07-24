//#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include "opencv2/opencv.hpp";
#include <thread>  

void perform_bertalmio_pde_inpainting_0(
	cv::Mat& input_array, cv::Mat& mask_array, cv::Mat& output_array,
	int* total_iters, int* total_inpaint_iters, int* total_anidiffuse_iters, int total_stages,
	float* delta_ts, float* sensitivities, int diffuse_coef);

void imshowOutput(std::string output_window_name, cv::Mat& output_array);


enum { PIXEL_WHITE = 1, PIXEL_RED = 2 };
typedef struct coord {
	int i;
	int j;
	int color;
}Coord;

using namespace cv;

int main(int argc, char* argv[])
{
	/* name of the images */
	std::string image_name = "simpleimg.png";
	std::string mask_name = "simpleimg_mask.png";
	std::string window_name = image_name;
	std::string mask_window_name = mask_name;
	std::string output_window_name = "output_array";
	
	cv::Mat output_array;
	
	/* Load and normalize the image */
	cv::Mat image_array = cv::imread(image_name,  cv::IMREAD_GRAYSCALE);
	image_array.convertTo(image_array, CV_32FC1);
	cv::normalize(image_array, image_array, 0, 1, cv::NORM_MINMAX, CV_32FC1);

	/* Load the mask */
	cv::Mat mask_array = cv::imread(mask_name, cv::IMREAD_GRAYSCALE);
	
	/* Create output_array */
	/*output_array.create(image_array.size(), CV_32FC1);
	typedef unsigned char logical_type;*/
	
	
	std::cout << image_array.channels();

	/* Create the windows */
	cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(mask_window_name, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(output_window_name, cv::WINDOW_AUTOSIZE);
	

	
	int total_iters[] = { 500 };
	int total_inpaint_iters[] = { 6 };
	int total_anidiffuse_iters[] = { 6 };
	int total_stages = 2;
	float delta_ts[] = { 0.02f };
	float sensitivites[] = { 100 };
	int diffuse_coef = 1;
	perform_bertalmio_pde_inpainting_0(
		image_array, mask_array, output_array,
		total_iters, total_inpaint_iters, total_anidiffuse_iters, total_stages,
		delta_ts, sensitivites, diffuse_coef);

	/* Display the output */
	/*cv::imshow(window_name, image_array);
	cv::imshow(mask_window_name, mask_array);
	cv::imshow(output_window_name, output_array);
	cv::waitKey(0);*/
}

void imshowOutput(std::string output_window_name, cv::Mat& output_array) {
	cv::imshow(output_window_name, output_array);
	cv::waitKey(10);
}

void perform_bertalmio_pde_inpainting_0(
	cv::Mat& input_array, cv::Mat& mask_array, cv::Mat& output_array,
	int* total_iters, int* total_inpaint_iters, int* total_anidiffuse_iters, int total_stages,
	float* delta_ts, float* sensitivities, int diffuse_coef) {

	/* Other declarations */
	typedef unsigned char logical_type;

	/* Matrix declarations */
	cv::Mat image_grad_row;
	cv::Mat image_grad_col;
	cv::Mat image_grad_norm;
	cv::Mat image_iso_row;
	cv::Mat image_iso_col;
	cv::Mat image_iso_norm;
	cv::Mat image_laplacian;
	cv::Mat image_laplacian_grad_row;
	cv::Mat image_laplacian_grad_col;
	cv::Mat diffuse_coefs;
	cv::Mat temp;

	/* Initialize output */
	input_array.copyTo(output_array);

	/*for (int row = 0; row < output_array.rows; row++) {
		logical_type* mask_array_ptr = mask_array.ptr<logical_type>(row);
		float* output_array_ptr = output_array.ptr<float>(row);
		for (int col = 0; col < output_array.cols; col++) {
			if ((mask_array_ptr[col] != 0)) {
				output_array_ptr[col] = 0.55;
			}
		}
	}*/
	cv::namedWindow("Teste", cv::WINDOW_AUTOSIZE);
	/* Compute bertalmio for each stage */
	for (int stage = 0; stage < total_stages; stage++) {
		cv::imshow("Teste", output_array);
		
		/* Grab data */
		int total_iter = total_iters[stage];
		int total_inpaint_iter = total_inpaint_iters[stage];
		int total_anidiffuse_iter = total_anidiffuse_iters[stage];
		float sensitivity = sensitivities[stage];
		float delta_t = delta_ts[stage];

		/* Run stage of algorithm */
		for (int iter = 0; iter < total_iter; iter++) {
			/* Perform anisotropic diffusion (there's probably a function for this, but wutevs) */
			for (int iter_aniffuse = 0; iter_aniffuse < total_anidiffuse_iter; iter_aniffuse++) {
				cv::Sobel(output_array, image_grad_row, -1, 0, 1);
				cv::Sobel(output_array, image_grad_col, -1, 1, 0);
				cv::magnitude(image_grad_row, image_grad_col, image_grad_norm);
				if (diffuse_coef == 0) {
					cv::exp(-(image_grad_norm.mul(1 / sensitivity)), diffuse_coefs);
				}
				else {
					cv::pow(image_grad_norm.mul(1 / sensitivity), 2, temp);
					diffuse_coefs = 1 / (1 + temp);
				}
				cv::Laplacian(output_array, image_laplacian, -1);
				for (int row = 0; row < output_array.rows; row++) {
					float* output_array_ptr = output_array.ptr<float>(row);
					float* diffuse_coefs_ptr = diffuse_coefs.ptr<float>(row);
					float* image_laplacian_ptr = image_laplacian.ptr<float>(row);
					logical_type* mask_array_ptr = mask_array.ptr<logical_type>(row);
					for (int col = 0; col < output_array.cols; col++) {
						if (mask_array_ptr[col] != 0) {
							output_array_ptr[col] +=
								delta_t * (diffuse_coefs_ptr[col] * image_laplacian_ptr[col]);
						}
					}
				}
			}

			/* Perform inpainting */
			for (int total_inpaint_iters = 0; total_inpaint_iters < total_inpaint_iter; total_inpaint_iters++) {
				cv::Sobel(output_array, image_iso_row, -1, 1, 0);
				cv::Sobel(output_array, image_iso_col, -1, 0, 1);
				image_iso_row *= -1;
				cv::sqrt(image_iso_row.mul(image_iso_row) + image_iso_col.mul(image_iso_col), image_iso_norm);
				cv::Laplacian(output_array, image_laplacian, -1);
				cv::Sobel(image_laplacian, image_laplacian_grad_row, -1, 0, 1);
				cv::Sobel(image_laplacian, image_laplacian_grad_col, -1, 1, 0);
				for (int row = 0; row < output_array.rows; row++) {
					logical_type* mask_array_ptr = mask_array.ptr<logical_type>(row);
					float* image_iso_norm_ptr = image_iso_norm.ptr<float>(row);
					float* image_iso_row_ptr = image_iso_row.ptr<float>(row);
					float* image_iso_col_ptr = image_iso_col.ptr<float>(row);
					float* image_laplacian_grad_row_ptr = image_laplacian_grad_row.ptr<float>(row);
					float* image_laplacian_grad_col_ptr = image_laplacian_grad_col.ptr<float>(row);
					float* output_array_ptr = output_array.ptr<float>(row);
					for (int col = 0; col < output_array.cols; col++) {
						if ((mask_array_ptr[col] != 0) && (image_iso_norm_ptr[col] != 0)) {
							output_array_ptr[col] -= delta_t * (
								image_iso_row_ptr[col] * image_laplacian_grad_row_ptr[col] +
								image_iso_col_ptr[col] * image_laplacian_grad_col_ptr[col]) /
								image_iso_norm_ptr[col];
							//std::cout << image_iso_norm_ptr[col] << std::endl;
							output_array_ptr[col] = (output_array_ptr[col] > 1.0f) ? 1 : output_array_ptr[col];
							output_array_ptr[col] = (output_array_ptr[col] < 0.0f) ? 0 : output_array_ptr[col];
							//std::cout << output_array_ptr[col] << std::endl;
						}
					}
				}
			}
		}
	}
	cv::waitKey(0);
}

