//#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include "opencv2/opencv.hpp";
#include <thread>  

enum { PIXEL_WHITE = 1, PIXEL_RED = 2 };
typedef struct coord {
	int i;
	int j;
	int color;
}Coord;

std::vector<Coord> create_mask(cv::Mat& mask); 
void perform_bertalmio_pde_inpainting_0(
	cv::Mat& input_array, std::vector<Coord>&  mask_array, cv::Mat& output_array,
	int* total_iters, int* total_inpaint_iters, int* total_anidiffuse_iters, int total_stages,
	float* delta_ts, float* sensitivities, int diffuse_coef);

void imshowOutput(std::string output_window_name, cv::Mat& output_array);



int main(int argc, char* argv[])
{
	/* name of the images */
	std::string dir = "img/";
	std::string name = "lena";
	std::string image_name = dir + name + ".png";
	std::string mask_name = dir + name + "_mask.png";
	std::string window_name = image_name;
	std::string mask_window_name = mask_name;
	std::string output_window_name = "output_array";
	

	cv::Mat output_array;


	/* Load and normalize the image */
	cv::Mat image_array = cv::imread(image_name);
	image_array.convertTo(image_array, CV_32FC1);
	cv::cvtColor(image_array, image_array, cv::COLOR_BGR2GRAY);
	cv::normalize(image_array, image_array, 0, 1, cv::NORM_MINMAX, CV_32FC1);
		

	/* Load the mask and fill the vector*/
	cv::Mat mask_array = cv::imread(mask_name);
	std::vector<Coord> mask_data = create_mask(mask_array);
	

	/* Create the windows */
	cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(mask_window_name, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(output_window_name, cv::WINDOW_AUTOSIZE);

	/*
		Bertalmio PDE Inpainting.	
	*/
	int total_iters[] = { 500 };
	int total_inpaint_iters[] = { 6 };
	int total_anidiffuse_iters[] = { 6 };
	int total_stages = 2;
	float delta_ts[] = { 0.02f };
	float sensitivites[] = { 100 };
	int diffuse_coef = 1;

	perform_bertalmio_pde_inpainting_0(
		image_array, mask_data, output_array,
		total_iters, total_inpaint_iters, total_anidiffuse_iters, total_stages,
		delta_ts, sensitivites, diffuse_coef);

	/* Display the output */
	cv::imshow(window_name, image_array);
	cv::imshow(mask_window_name, mask_array);
	cv::imshow(output_window_name, output_array);
	cv::waitKey(0);
}

void imshowOutput(std::string output_window_name, cv::Mat& output_array) {
	cv::imshow(output_window_name, output_array);
	cv::waitKey(10);
}

void perform_bertalmio_pde_inpainting_0(
	cv::Mat& input_array, std::vector<Coord>& mask_array, cv::Mat& output_array,
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
	/* Size mask*/
	int size_mask = mask_array.size();
	/* Compute bertalmio for each stage */
	for (int stage = 0; stage < total_stages; stage++) {
		
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
				
				for (int cont = 0; cont < size_mask; cont++) {
					Coord coord = mask_array.at(cont);
					int row = coord.i;
					int col = coord.j;

					output_array.at<float>(row, col) +=
						delta_t * (diffuse_coefs.at<float>(row, col) * image_laplacian.at<float>(row, col));
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

				for (int cont = 0; cont < size_mask; cont++) {
					Coord coord = mask_array.at(cont);
					int row = coord.i;
					int col = coord.j;
					if (image_iso_norm.at<float>(row, col) != 0) {
						output_array.at<float>(row, col) -= delta_t * (
							image_iso_row.at<float>(row, col) * image_laplacian_grad_row.at<float>(row, col) +
							image_iso_col.at<float>(row, col) * image_laplacian_grad_col.at<float>(row, col)) /
							image_iso_norm.at<float>(row, col);
						output_array.at<float>(row, col) = (output_array.at<float>(row, col) > 1.0f ? 1 : output_array.at<float>(row, col));
						output_array.at<float>(row, col) = (output_array.at<float>(row, col) < 0.0f ? 0 : output_array.at<float>(row, col));
					}			
				}
			}
		}
	}
}



/*
	Save the inpainting domain to dinamic vector
*/
std::vector<Coord> create_mask(cv::Mat& mask) {
	std::vector<Coord> mask_data;
	for (int i = 1; i < mask.rows - 1; i++) {
		for (int j = 1; j < mask.cols - 1; j++) {
			if (mask.at<cv::Vec3b>(i, j)[0] != 0) { //BLUE GREEN RED --> white (255,255,255) 
				Coord xy;
				xy.i = i;
				xy.j = j;
				xy.color = PIXEL_WHITE;
				mask_data.push_back(xy);
			}
			else if (mask.at<cv::Vec3b>(i, j)[0] == 0 && mask.at<cv::Vec3b>(i, j)[2] == 255) { //BLUE GREEN RED --> red (0,0,255) 
				Coord xy;
				xy.i = i;
				xy.j = j;
				xy.color = PIXEL_RED;
				mask_data.push_back(xy);
			}
		}
	}
	return mask_data;
}