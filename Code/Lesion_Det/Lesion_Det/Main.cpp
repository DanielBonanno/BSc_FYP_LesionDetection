//Definition for PI, used if using shape index
//#define _USE_MATH_DEFINES

//DCMTK includes
#include <dcmtk\dcmimgle\dcmimage.h>
#include <dcmtk\dcmjpeg\ddpiimpl.h>
#include <dcmtk\dcmdata\dcdatset.h>
#include <dcmtk\dcmdata\dctag.h>
#include <dcmtk\dcmdata\dctagkey.h>
#include <dcmtk\dcmdata\dcdeftag.h>

//OpenCV includes
#include <opencv2\opencv.hpp>

//C++ includes
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <cmath>

//Eigen includes, used if using shape index
//#include "Eigen/Eigen/Dense"
//#include "Eigen/Eigen/src/Eigenvalues/EigenSolver.h"


//Function used to get file names from a from a folder - WORKS ONLY ON WINDOWS
//Input: path for folder
//Output: vector with file names
bool Get_Files(std::string folder, std::vector<std::string>& files) {

	//Create a data object
	WIN32_FIND_DATA FindFileData;

	//Get the full path
	std::string full_path = folder + "/*.*";

	//Change it to widestring (due to function parameter)
	std::wstring full_path_wide = std::wstring(full_path.begin(), full_path.end());

	//Search folder with the path given. Note that wildcards are used since the full name is not known
	HANDLE hFind = FindFirstFile(full_path_wide.c_str(), &FindFileData);

	//create a char* to hold the file name It's length is 50, since it is assumed that a file name will not be longer than that
	char file_name[50];

	//if no files exist, return false
	if (hFind == INVALID_HANDLE_VALUE) {
		return false;
	}

	//otherwise, put the file in the vector, until there are no more files to read
	else do {
		std::wcstombs(file_name, FindFileData.cFileName, sizeof(file_name));

		//Note that this check is done to eliminate current directory and parent directory, which are stored in every folder
		if ((std::strcmp(file_name, ".")) && (std::strcmp(file_name, "..")))
		{
			files.push_back(folder + file_name);
		}

	} while (FindNextFile(hFind, &FindFileData));

	//close the filehande
	FindClose(hFind);

	//sort the vector
	std::sort(files.begin(), files.end());
	return true;
}


//Function used to load the DICOM file
//Input:  path to file
//Output: the index of the file in the sequence, the pixel spacing along the x and y, the slice thickness, the image
void Load_Image(int& index, float& resolution_x, float& resolution_y, float& resolution_z, cv::Mat &dst, const char* path)
{
	//window centre and width
	int centre = -700;
	int width = 1000;

	//variable to store the data from the DICOM file
	OFString tag_data;

	//the following variables are used to load the file
	DcmFileFormat dcmFileFormat;
	DcmDataset* dataset;

	//load the dicom file and get the dataset
	dcmFileFormat.loadFile(path);
	dataset = dcmFileFormat.getDataset();


	//get the image
	DicomImage *dicomImage = new DicomImage(dataset, EXS_Unknown);

	//set the window parameters which were defined above to the image. 
	//Note that these will modift the image such that only volumes of interest are shown
	dicomImage->setWindow(centre, width);

	//re-create the image using the window parameters
	DicomImage *windowed_image = dicomImage->createDicomImage();

	//get the output in 8 bit form. since grayscale images will be used, 8 bits are enough
	uchar *window_dat = (uchar *)(windowed_image->getOutputData(8));

	//create the opencv mat image
	cv::Mat cv_image(int(windowed_image->getHeight()), int(windowed_image->getWidth()), CV_8U, window_dat);
	dst = cv_image.clone();

	//Get the index
	dataset->findAndGetOFString(DCM_InstanceNumber, tag_data);
	index = atoi(tag_data.c_str()) - 1;

	//Get the pixel spacing
	dataset->findAndGetOFString(DCM_PixelSpacing, tag_data, 0);
	resolution_x = atof(tag_data.c_str());

	dataset->findAndGetOFString(DCM_PixelSpacing, tag_data, 0);
	resolution_y = atof(tag_data.c_str());

	//Get the slice thickness
	dataset->findAndGetOFString(DCM_SliceThickness, tag_data, 0);
	resolution_z = atof(tag_data.c_str());
}

//Function used to perform the complex DFT of a grayscale image
//Input:  image
//Output: DFT (complex numbers)
void Complex_Gray_DFT(const cv::Mat &src, cv::Mat &dst)
{
	//Convert the original grayscale image into a normalised floating point image with only 1 channel
	cv::Mat src_float;
	src.convertTo(src_float, CV_32FC1);
	cv::normalize(src_float, src_float, 0, 1, cv::NORM_MINMAX);

	//Run the DFT
	cv::dft(src_float, dst, cv::DFT_COMPLEX_OUTPUT);
}

//Function used to obtain a magnitude image from a complex DFT
//Input:  Complex DFT
//Output: Magnitude Image
void DFT_Magnitude(const cv::Mat &dft_src, cv::Mat &dft_magnitude)
{
	//The dft_src mat is split into 2 separate channels, one for the real and the other for the imaginary
	cv::Mat channels[2] = { cv::Mat::zeros(dft_src.size(),CV_32F), cv::Mat::zeros(dft_src.size(), CV_32F) };
	cv::split(dft_src, channels);

	//Compute the Magnitude
	cv::magnitude(channels[0], channels[1], dft_magnitude);

	//Obtain the log and normalise it between 0 and 1. +1 since log(0) is undefined.
	cv::log(dft_magnitude + cv::Scalar::all(1), dft_magnitude);
	cv::normalize(dft_magnitude, dft_magnitude, 0, 1, CV_MINMAX);
}

//Function used to re-centre the Magnitude of the DFT, such that the centre shows the low frequencies
//Input/Output: Magnitude image
void Recentre(cv::Mat &dft_magnitude)
{
	//define the centre points. These will be used when selecting the quadrants
	int centre_x = dft_magnitude.cols / 2;
	int centre_y = dft_magnitude.rows / 2;

	//create the quadrant objects and a temporary object that will be used in the swapping process
	cv::Mat temp;
	//NB: The way the quadrants are created means that they all reference the same matrix. (shallow copy!)
	//    A change in the quadrants results in a change in the source
	cv::Mat quad_1(dft_magnitude, cv::Rect(0, 0, centre_x, centre_y));
	cv::Mat quad_2(dft_magnitude, cv::Rect(centre_x, 0, centre_x, centre_y));
	cv::Mat quad_3(dft_magnitude, cv::Rect(0, centre_y, centre_x, centre_y));
	cv::Mat quad_4(dft_magnitude, cv::Rect(centre_x, centre_y, centre_x, centre_y));

	//swapping process
	quad_1.copyTo(temp);
	quad_4.copyTo(quad_1);
	temp.copyTo(quad_4);

	quad_2.copyTo(temp);
	quad_3.copyTo(quad_2);
	temp.copyTo(quad_3);
}

//Function to perform the inverse DFT of a Gray Scale image and obtain an image.
//Note that the values will be real, not complex
//Input:  DFT of an image
//Output: Image in spatial domain
void Real_Gray_IDFT(const cv::Mat &src, cv::Mat &dst)
{
	//Perform the inverse dft. Note that, by the flags passed, it will return a 
	//real output (since an image is made out of real components) and will also 
	//scale the values so that information can be visualised.
	cv::idft(src, dst, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

	//Note that for consistency purpose, it is recomended to convert the dst to the original image's type and normalise it
	//Here, it is assumed that the image is 8 bits and unsigned
	dst.convertTo(dst, CV_8U, 255);
	cv::normalize(dst, dst, 0, 255, CV_MINMAX);
}

//Function used to perform bilateral Filtering
//Input: src image, neighbourhood size, standard dev for colour and space
//Output: filtered image
void Bilateral_Filtering(const cv::Mat&  src, cv::Mat& dst, int neighbourhood, double sigma_colour, double sigma_space)
{
	cv::bilateralFilter(src, dst, neighbourhood, sigma_colour, sigma_space);
}

//Function used to perform Wiener Denoising. Note that this function does not reverse the blur
//Input: src image, constant to multiply with (inverse of PSNR)
//Output: filtered image
void Wiener_Denoising(const cv::Mat &src, cv::Mat &dst, double PSNR_inverse)
{
	cv::Mat src_dft;

	//Obtain the DFT of the src image
	Complex_Gray_DFT(src, src_dft);

	//Multiply it by the constant
	cv::Mat output_dft;
	output_dft = (PSNR_inverse)*src_dft;

	//Perform the inverse PSNR to get the filtered output
	Real_Gray_IDFT(output_dft, dst);
}

//Function used to perform anisotropic diffusion
//Input: src image, lambda (used for converance), number of iterations that should run, 
//		 k values to define what is an edge and what is not (x2: one for north/south, one for east/west)
//Output: filtered image
void Anisotrpic_Diffusion(const cv::Mat& src, cv::Mat& dst, double lambda, int iterations, float k_ns, float k_we)
{
	//copy the src to dst as initilisation and convert dst to a 32 bit float mat, for accuracy when processing
	dst = src.clone();
	dst.convertTo(dst, CV_32F);

	//Create the kernels to be used to calculate edges in North, South, East and West directions
	//These are similar to Laplacian kernels and defined in [24], equation (8)
	cv::Mat_<int> kernel_n = cv::Mat::zeros(3, 3, CV_8U);
	kernel_n(1, 1) = -1;
	cv::Mat_<int> kernel_s, kernel_w, kernel_e;
	kernel_s = kernel_n.clone();
	kernel_w = kernel_n.clone();
	kernel_e = kernel_n.clone();

	kernel_n(0, 1) = 1;
	kernel_s(2, 1) = 1;
	kernel_w(1, 0) = 1;
	kernel_e(1, 2) = 1;

	//Create the parameters that will be used in the loop:

	// grad_x = gradient obtained by using the directional edge detection kernel defined above

	// param_x = mat which containes the parameter to be fed to the exp function, used when calculating the conduction coefficient value (c_x)

	// c_x = conduction coefficient values which help to control the amount of diffusion

	// k_ns and k_we are gradient magnitude parameters - they serve as thresholds to define what is a gradient and what is noise. 
	// therefore, they are used in the function to define the c_x values (one for north and south directions, the other fo west and east)

	cv::Mat_<float> grad_n, grad_s, grad_w, grad_e;
	cv::Mat_<float> param_n, param_s, param_w, param_e;
	cv::Mat_<float> c_n, c_s, c_w, c_e;

	//perform the number of iterations requested
	for (int time = 0; time < iterations; time++)
	{
		//perform the filtering
		cv::filter2D(dst, grad_n, -1, kernel_n);
		cv::filter2D(dst, grad_s, -1, kernel_s);
		cv::filter2D(dst, grad_w, -1, kernel_w);
		cv::filter2D(dst, grad_e, -1, kernel_e);

		//obtain the iteration values as per the conduction function used
		//in this case, the conduction function is found in [52], eq 11
		param_n = (std::sqrt(5) / k_ns) * grad_n;
		cv::pow(param_n, 2, param_n);
		cv::exp(-(param_n), c_n);

		param_s = (std::sqrt(5) / k_ns) * grad_s;
		cv::pow(param_s, 2, param_s);
		cv::exp(-(param_s), c_s);

		param_w = (std::sqrt(5) / k_we) * grad_w;
		cv::pow(param_w, 2, param_w);
		cv::exp(-(param_w), c_w);

		param_e = (std::sqrt(5) / k_we) * grad_e;
		cv::pow(param_e, 2, param_e);
		cv::exp(-(param_e), c_e);

		//update the image
		dst = dst + lambda*(c_n.mul(grad_n) + c_s.mul(grad_s) + c_w.mul(grad_w) + c_e.mul(grad_e));
	}
	//convert the output to 8 bits
	dst.convertTo(dst, CV_8U);
}

//Function used to perform Gamma correction
//Input:  NORMALISED src image, gamma
//Output: Modified image
void Gamma_Correction(const cv::Mat&  src, cv::Mat& dst, double gamma)
{
	//Obtain the inverse of the gamma provided
	double inverse_gamma = 1.0 / gamma;

	//Create a lookup table, to reduce processing time
	cv::Mat Lookup(1, 256, CV_8U);			//Lookup table having values from 0 to 255
	uchar* ptr = Lookup.ptr<uchar>(0);		//pointer access is fast in opencv, since it does not perform a range check for each call
											//uchar since 1 byte unsigned integer, in the range 0 - 255
	for (int i = 0; i < 256; i++)
	{
		ptr[i] = (int)(std::pow((double)i / 255.0, inverse_gamma) * 255.0);

	}
	//Apply the lookup table to the source image
	cv::LUT(src, Lookup, dst);

}

//Function used to perform contrast enhancement by the technique defined
//Input: src image, gamma used for gamma correction, clip limit and grid sizez sed in CLAHE
//Output: contrast enhanced image
void Contrast_Enhancement(const cv::Mat&  src, cv::Mat& dst, double gamma, double clip_limit, int grid_size_x, int grid_size_y)
{
	//Create a CLAHE_grid 
	cv::Size CLAHE_grid(grid_size_x, grid_size_y);

	//Perform gamma correction
	Gamma_Correction(src, dst, gamma);

	//Create a CLAHE object with the specified parameters
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, CLAHE_grid);

	//Apply CLAHE to the given image
	clahe->apply(dst, dst);
}

//Function: Used to remove the background around the chest cavity.
//          If in search_dist pixels below or above the pixel, there are at less than counter_threshold pixels which have
//          a higher intenisty than intensity_threshold, change that pixel to white (255)
//          Ex: if in search_dist pixels below pixel (0,0), less than 2 pixels are more than 50, change that pixel to white. 
//Input: Image to be modified, search radius, threshold for the number of pixels, threshold for the intensity
//Output: Modified image
void Remove_Background_Chest(cv::Mat& src, int search_dist, int counter_threshold, int intensity_threshold)
{
	cv::Mat_<uchar> image = src.clone();
	cv::Mat_<uchar> output = src.clone();		//needed so that changes made do not influence the procedure

	int width = src.cols;
	int height = src.rows;
	int counter = 0;

	//perform search column wise (ie vertically) since most contouring is found along the width

	//For Top
	//Iterate over the first half the rows in a column (bottom half will be handled later on)
	for (int col = 0; col < width; col++)
	{
		for (int row = 0; row <= height / 2; row++)
		{
			//for every pixel, set the counter to 0
			counter = 0;
			//perform the search over the defined search area
			for (int add = 1; add < search_dist; add++)
			{
				//increment the counter if a pixel over the specified intensity is found
				if (image(row + add, col) > intensity_threshold)
				{
					counter++;
				}
			}

			//if for that particular pixel's search area, less than counter_threshold have been found with the required intensity
			//set it to white
			if (counter < counter_threshold)
			{
				output(row, col) = 255;
			}
			//otherwise, it means that the chest cavity is close, therefore set the pixel to white and move on to the next column
			else
			{
				output(row, col) = 255;
				break;
			}
		}
	}

	//Repeat the procedure for the bottom half
	for (int col = 0; col < width; col++)
	{
		for (int row = height - 1; row >= height / 2; row--)
		{
			counter = 0;
			for (int add = 1; add < search_dist; add++)
			{
				if (image(row - add, col) > intensity_threshold)
				{
					counter++;
				}
			}

			if (counter < counter_threshold)
			{
				output(row, col) = 255;
			}
			else
			{
				output(row, col) = 255;
				break;
			}
		}
	}

	src = output.clone();
}

//Function: Similar to the above function, this removes the background around the lungs. Note that here the logic is inverted, since the intensity is inverted as well
//Input: Image to be modified, search radius, threshold for the number of pixels, threshold for the intensity
//Output: Modified image
void Remove_Background_Lungs(cv::Mat& src, int search_dist, int counter_threshold, int intensity_threshold)
{

	cv::Mat_<uchar> image = src.clone();
	cv::Mat_<uchar> output = src.clone();		//needed so that changes made do not influence the procedure

	int width = src.cols;
	int height = src.rows;
	int counter = 0;

	//perform search row wise (ie horizontally) since most contorted along the height

	//For Left
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col <= width / 2; col++)
		{
			counter = 0;
			for (int add = 1; add < search_dist; add++)
			{
				if (image(row, col + add) < intensity_threshold)
				{
					counter++;
				}
			}

			if (counter < counter_threshold)
			{
				output(row, col) = 255;
			}
			else
			{
				output(row, col) = 255;
				break;
			}
		}
	}


	//For Right
	for (int row = 0; row < height; row++)
	{
		for (int col = width - 1; col >= width / 2; col--)
		{
			counter = 0;
			for (int add = 1; add < search_dist; add++)
			{
				if (image(row, col - add) < intensity_threshold)
				{
					counter++;
				}
			}

			if (counter < counter_threshold)
			{
				output(row, col) = 255;
			}
			else
			{
				output(row, col) = 255;
				break;
			}
		}
	}

	src = output.clone();
}

void Segment(const cv::Mat&  src, cv::Mat& fine_binary_mask)
{
	cv::Mat src_copy = src.clone();

#pragma region blob	
	// Create a structuring element and perform morpholoical opening. This removes details inside the lungs 
	// and artefacts around the chest cavity and returns blobs. The image is useful to define the watershed markers
	cv::Mat image_blobbed = src.clone();
	cv::Mat disk_10 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10));
	cv::morphologyEx(image_blobbed, image_blobbed, cv::MORPH_OPEN, disk_10);
#pragma endregion

#pragma region remove background
	//Black background and any residual noise are set to white, such that only the chest cavity, lined with a black 
	//border remains. (This will be removed later on by dilation.) Note that this function does NOT enter the chest cavity.
	Remove_Background_Chest(image_blobbed, 5, 3, 200);

	//Remove any residual noise inside the chest cavity that would have been left by the previous removal.
	Remove_Background_Lungs(image_blobbed, 50, 3, 70);

#pragma endregion

#pragma region foreground markers
	// Dilation is used create a mask to define the foreground markers. Dilation achieves 3 purposes: 
	// 1.) remove the remaining faint line along the chest cavity
	// 2.) reduce the area that is occupied by the lungs in the mask, such  that the watershed algorithm can define the exact boundaries. 
	// 3.) it also helps to remove the trachea in certain images, especially in images where it is clearly detached from the lungs. 
	//     this was a consideration when choosing the size of element. 
	// Dilation is used since there is a larger white area around the edges and dilation outputs the maximum value.
	cv::Mat image_markers_foreground;
	cv::Mat disk_25 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25));
	cv::morphologyEx(image_blobbed, image_markers_foreground, cv::MORPH_DILATE, disk_25);

	//The mask is then inverted and thresholded, such that they it turned into a binary image. 
	//In the foreground markers, white represents the areas which we are sure are ROI.
	//if pixel > 0 --> set to white, since it is a foreground pixel

	cv::Mat binary_markers_foreground = 255 - image_markers_foreground;
	cv::threshold(binary_markers_foreground, binary_markers_foreground, 0, 255, CV_THRESH_BINARY);

	//fill any holes
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binary_markers_foreground, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	cv::drawContours(binary_markers_foreground, contours, -1, 255, -1);

#pragma endregion

#pragma region uncertain markers
	//The same procedure as the foreground markers is repeated but with a smaller element, such that less dilation occurs. Once this is done, Erosion is performed 
	//using a large element. This will aid to set regions which we are uncertain if they are background or foreground to a grey level of 128 (further on).
	//This tells the watershed algorithm to classify these values. Note that dilation is used to remove the faint line at the chest cavity and the trachea 
	//in some cases (as with the foreground marks).
	cv::Mat image_markers_uncertain;
	cv::Mat disk_7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
	cv::morphologyEx(image_blobbed, image_markers_uncertain, cv::MORPH_DILATE, disk_7);
	cv::Mat disk_12 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(12, 12));
	cv::morphologyEx(image_markers_uncertain, image_markers_uncertain, cv::MORPH_ERODE, disk_12);
	
	//The marker mask is then inverted and thresholded, such that it is turned into a binary image. 
	//In the uncertain mask, white represents the possibility of being foreground. 
	//	cv::imshow("Uncertain", image_markers_uncertain);
	//if pixel > 0 --> set to white, since it MIGHT be a foreground pixel
	cv::Mat binary_markers_uncertain = 255 - image_markers_uncertain;
	cv::threshold(binary_markers_uncertain, binary_markers_uncertain, 0, 255, CV_THRESH_BINARY);

	//fill any holes
	cv::findContours(binary_markers_uncertain, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	cv::drawContours(binary_markers_uncertain, contours, -1, 255, -1);

#pragma endregion
	
#pragma region final marker mask
	//The 2 masks are then combined to achieve a final marker mask. In this marker mask:
	// a white value (255) represents a sure foreground pixel
	// a grey value (128) represents a sure background pixel
	// a black value (0) represents an uncertainty, these will be evaluated by the watershed.
	cv::Mat final_markers = (binary_markers_foreground / 2 + binary_markers_uncertain / 2);
	for (int row = 0; row < final_markers.rows; row++)
	{
		for (int col = 0; col < final_markers.cols; col++)
		{
			if (final_markers.at<uchar>(row, col) == 0) {
				final_markers.at<uchar>(row, col) = 128;
			}
			else if (final_markers.at<uchar>(row, col) != 255)
			{
				final_markers.at<uchar>(row, col) = 0;
			}
		}
	}
#pragma endregion

#pragma region watershed
	//Use watershed algorithm, along with the final markers mask created above to obtain segmentation


	final_markers.convertTo(final_markers, CV_32SC1);
	cv::cvtColor(src_copy, src_copy, CV_GRAY2BGR, 3);
	src_copy.convertTo(src_copy, CV_8UC3);

	cv::watershed(src_copy, final_markers);
	final_markers.convertTo(final_markers, CV_8UC1);

	cv::morphologyEx(final_markers, fine_binary_mask, cv::MORPH_CLOSE, disk_25);

	//threshold the final mask, such that it is a binary mask and an AND operator will result in the segmented image
	cv::threshold(fine_binary_mask, fine_binary_mask, 200, 255, CV_THRESH_BINARY);

#pragma endregion
}


//Function: Obtain a 3D Mat from a vector of 2D Images -- No Longer Used In Implementation
//Input: Vector of Images
//Output: 3D Mat - 16BIT, SIGNED
/*void Mat_3D_From_Vector(const std::vector<cv::Mat>& src_vec, cv::Mat& dst)
{
//obtain the sizes of the vector and the images inside it
int size_z = src_vec.size();
int size_y = src_vec[0].rows;
int size_x = src_vec[0].cols;

//set the sizes of the 3D Matrix
int sizes[3] = { size_z, size_y, size_x };
cv::Mat output(3, sizes, CV_16S);

//Set the matrix values
for (int plane = 0; plane < size_z; plane++) {
short *data_ptr = (short*)src_vec[plane].data;
std::memcpy((short*)output.data + plane* size_x * size_y, data_ptr, sizeof(short)*size_x * size_y);
}

dst = output.clone();
}*/

//Function: Create a vector fo 2D Images from a 3D Mat -- No Longer Used In Implementation
//Input: 3D Mat - 16BIT, SIGNED
//Output: Vector of 2D Images - 16BIT, SIGNED
/*void Vector_From_Mat_3D(const cv::Mat& src, std::vector<cv::Mat>& dst_vec)
{
std::vector<cv::Mat> output;


//get the sizes of the 3D Matrix
const int* sizes = src.size;

//set the new sizes
int size_z = sizes[0];
int size_y = sizes[1];
int size_x = sizes[2];

//set the values for every image in the vector
for (int plane = 0; plane < size_z; plane++) {
short *data_ptr = (short*)src.data + plane * size_x * size_y; // sub-matrix pointer		//+ plane... ensures that, with every iteration, you move to the next 2D matrix
output.push_back(cv::Mat(size_y, size_x, CV_16S, data_ptr).clone());
}
dst_vec.clear();
dst_vec = output;

}*/

//Function: Used to switch the axis of a vector of 2D Images, as per Figure 3.14  -- No Longer Used In Implementation
//Input: Src Vector - 16BIT, SIGNED
//Output: Dst Vector - 16BIT, SIGNED
/*void Switch_XZ_Axis_Vector(const std::vector<cv::Mat>& src_vec, std::vector<cv::Mat>& dst_vec)
{
//Change from Vector to 3D Matrix
cv::Mat src_3D;
Mat_3D_From_Vector(src_vec, src_3D);

//length, rows, cols of the new output are equal to the cols, rows and length of the source respectively.

const int* sizes = src_3D.size;
int size_z = sizes[0];
int size_y = sizes[1];
int size_x = sizes[2];
int new_sizes[3] = { size_x, size_y,size_z };
cv::Mat dst_3D(3, new_sizes, CV_16S);

//perform the axis change
for (int col = 0; col < size_x; col++)
{
for (int plane = 0; plane < size_z; plane++)
{
for (int row = 0; row < size_y; row++)
{
dst_3D.at<short>(col, row, plane) = src_3D.at<short>(plane, row, col);
}
}
}

Vector_From_Mat_3D(dst_3D, dst_vec);
}*/

//Function: Used to switch back the axis of a vector of 2D Images, as per Figure 3.14  -- No Longer Used In Implementation
//Input: Src Vector - 16BIT, SIGNED
//Output: Dst Vector - 16BIT, SIGNED
/*void Restore_XZ_Axis_Vector(const std::vector<cv::Mat>& src_vec, std::vector<cv::Mat>& dst_vec)
{
//Change from Vector to 3D Matrix
cv::Mat src_3D;
Mat_3D_From_Vector(src_vec, src_3D);


//length, rows, cols of the new output are equal to the cols, rows and length of the source rescpecively.

const int* sizes = src_3D.size;
int size_z = sizes[0];
int size_y = sizes[1];
int size_x = sizes[2];

int new_sizes[3] = { size_x, size_y,size_z };
cv::Mat dst_3D(3, new_sizes, CV_16S);

//perform the rotation
for (int col = 0; col < size_x; col++)
{
for (int plane = 0; plane < size_z; plane++)
{
for (int row = 0; row < size_y; row++)
{
dst_3D.at<short>(col, row, plane) = src_3D.at<short>(plane, row, col);
}
}
}

Vector_From_Mat_3D(dst_3D, dst_vec);
}*/


//Function: Perform derivation along the x-axis for a set of images in a vector -- No Longer Used in Implementation
//Input: Vector of 2D Images 
//Output: Vector of 2D gradient images - 16BIT, SIGNED
/*void Derivate_Vector_x(const std::vector<cv::Mat>& src_vec, std::vector<cv::Mat>& dst_vec)
{
dst_vec.clear();
dst_vec.resize(src_vec.size());
for (std::vector<cv::Mat>::size_type i = 0; i < src_vec.size(); i++)
{
cv::Scharr(src_vec[i], dst_vec[i], CV_16S, 1, 0);
}
}*/

//Function: Perform derivation along the y-axis for a set of images in a vector -- No Longer Used in Implementation
//Input: Vector of 2D Images
//Output: Vector of 2D gradient images - 16BIT, SIGNED
/*void Derivate_Vector_y(const std::vector<cv::Mat>& src_vec, std::vector<cv::Mat>& dst_vec)
{
dst_vec.clear();
dst_vec.resize(src_vec.size());
for (std::vector<cv::Mat>::size_type i = 0; i < src_vec.size(); i++)
{
cv::Scharr(src_vec[i], dst_vec[i], CV_16S, 0, 1);

}
}*/

//Function: Perform derivation along the z-axis for a set of images in a vector -- No Longer Used in Implementation
//Input: Vector of 2D Images - 16BIT, SIGNED
//Output: Vector of 2D gradient images - 16BIT, SIGNED
//NB: INPUT IMAGES MUST BE 16BIT, SIGNED. THIS IS DUE TO THE WAY SWITCH AND RESTORE FUNCTIONS ARE WRITTEN	
/*void Derivate_Vector_z(const std::vector<cv::Mat>& src_vec, std::vector<cv::Mat>& dst_vec)
{
dst_vec.clear();
dst_vec.resize(src_vec.size());

//Natively, OpenCV does not suppost filtering of a 3D Mat along the z axis, therefore the images are stored as a vector,
//Their x and z axis are switched and derivation is performed along the x (now z) axis
std::vector<cv::Mat> src_new_axis;
Switch_XZ_Axis_Vector(src_vec, src_new_axis);
std::vector<cv::Mat> dst_new_axis(src_new_axis.size());
for (std::vector<cv::Mat>::size_type i = 0; i < src_new_axis.size(); i++)
{
cv::Scharr(src_new_axis[i], dst_new_axis[i], CV_16S, 1, 0);

}
//once derivation is complete, the axis are resotred back
Restore_XZ_Axis_Vector(dst_new_axis, dst_vec);
}*/


//Function: Change all the image types in a vector. This is usefull for the above functions which perform derivations or rotations. - No Longer used in Implementation
//Input: Vector of 2D Images, new type
//Output: Vector of 2D Images with new type
/*void Convert_Vector_Image_Type(std::vector<cv::Mat>& src_vector, int type)
{
for (int plane = 0; plane < src_vector.size(); plane++)
{
src_vector[plane].convertTo(src_vector[plane], type);
}
}*/

//Function: Get Sphericity Mask of the image. Note that the sphericity mask is calculated on the Enhanced images and only 
//          for the points that are white in the intensity mask -- No Longer Used in the Implementation
//Input: vector of intensity masks, vector of enahnced images
//Output: vector of spehericity masks
/*void Get_Sphericity_Mask(const std::vector<cv::Mat>& Intensity_Mask, const std::vector<cv::Mat>& Enhanced_Image, std::vector<cv::Mat>& Sphericity_Mask)
{
Sphericity_Mask.clear();
Sphericity_Mask.resize(Intensity_Mask.size());

//Create the vectors for the derivateive images
std::vector<cv::Mat> I_x, I_xx, I_xy, I_xz, I_y, I_yy, I_yx, I_yz, I_z, I_zz, I_zx, I_zy;

//obtain the 2nd order derivatives in every direction for every image
Derivate_Vector_x(Enhanced_Image, I_x);
Derivate_Vector_x(I_x, I_xx);
Derivate_Vector_y(I_x, I_xy);
Derivate_Vector_z(I_x, I_xz);

Derivate_Vector_y(Enhanced_Image, I_y);
Derivate_Vector_y(I_y, I_yy);
Derivate_Vector_x(I_y, I_yx);
Derivate_Vector_z(I_y, I_yz);

Derivate_Vector_z(Enhanced_Image, I_z);
Derivate_Vector_z(I_z, I_zz);
Derivate_Vector_x(I_z, I_zx);
Derivate_Vector_y(I_z, I_zy);


//For every lesion candidate point in every image, obtain a shape index by creating a hessian matrix for that point composed of the 9
//2nd order derivatives. The principle curvatures are represented by the largest and smallest eignen values of the
//hessian matrix. These are fed into an equation to calculate the shape index at every point
Eigen::Matrix3d Hessian;
double max_eigen_val, min_eigen_val;
for (int plane = 0; plane < Enhanced_Image.size(); plane++)
{
//use a 64 bit floating point for accuracy
cv::Mat temp(Enhanced_Image[plane].rows, Enhanced_Image[plane].cols, CV_64F);
for (int row = 0; row < Enhanced_Image[plane].rows; row++)
{
for (int col = 0; col < Enhanced_Image[plane].cols; col++)
{
//temp.convertTo(temp, CV_64F);
if (Intensity_Mask[plane].at<uchar>(row, col) != 0)
{
//if, according to the segmentation mask, the point is white (that is, it is part of the region of interest), calcualte the Shape Index
Hessian(0, 0) = I_xx[plane].at<short>(row, col);
Hessian(0, 1) = I_xy[plane].at<short>(row, col);
Hessian(0, 2) = I_xz[plane].at<short>(row, col);
Hessian(1, 0) = I_yx[plane].at<short>(row, col);
Hessian(1, 1) = I_yy[plane].at<short>(row, col);
Hessian(1, 2) = I_yz[plane].at<short>(row, col);
Hessian(2, 0) = I_zx[plane].at<short>(row, col);
Hessian(2, 1) = I_zy[plane].at<short>(row, col);
Hessian(2, 2) = I_zz[plane].at<short>(row, col);

Eigen::EigenSolver<Eigen::Matrix3d> Eigen_Solver(Hessian, false);
Eigen::Vector3d solutions = Eigen_Solver.eigenvalues().real();
min_eigen_val = solutions(0);
max_eigen_val = solutions(2);

temp.at<double>(row, col) = 0.5 - M_1_PI*atan2((max_eigen_val + min_eigen_val),(max_eigen_val - min_eigen_val));
}
//otherwise set the shape index to 0
else
{
temp.at<double>(row, col) = 0;
}
}
}

//convert back to unsigned 8 bits and copy the temp to the sphericity mask. Note that here the range is no longer 0 - 1 but 0 - 255
temp.convertTo(temp, CV_8U, 255);
Sphericity_Mask[plane] = temp.clone();
}
}*/


//Function: Check if a point is in a list of points
//Input: Point, List of Points
//Output: True if found, Flase if not
bool is_in_list(const cv::Point& point, std::vector<cv::Point> list)
{
	//Check every point in the list
	for (int index = 0; index < list.size(); index++)
	{
		//if they are equal, then the point in the parameters is in the list
		if (point == list[index])
		{
			return true;
		}
		//otherwise it is not
	}
	return false;
}

//Function: A recursive function used to expand regions of interest of an image by expanding into a point's 8 neighbours
//Input: Image (Mask), row and column of pixel which was checked, list of already visited points, group of points that define a region
//Output: Group (that is, group of points that define a region)
void Expand_ROI(const cv::Mat &Mask, int row, int col, std::vector<cv::Point> &visited_list, std::vector<cv::Point> &group)
{
	//a region > 2700 white pixels is most probably an error in the segmentation, it is too large!
	if (visited_list.size() > 2700)
	{
		//therefore clear the group and return
		group.clear();
		return;
	}

	//if the pixel is white and not already vistied
	if ((Mask.at<uchar>(row, col) != 0) && (!is_in_list(cv::Point(col, row), visited_list)))
	{
		//add it to the visited list
		visited_list.push_back(cv::Point(col, row));

		//add it to the group that defines the region
		group.push_back(cv::Point(col, row));

		//for its 8 neighbours, call again this function
		for (int search_row = -1; search_row < 2; search_row++)
		{
			if ((row + search_row >= 0) && (row + search_row < Mask.rows))
			{
				for (int search_col = -1; search_col < 2; search_col++)
				{
					if ((col + search_col >= 0) && (row + search_row < Mask.cols))
					{
						Expand_ROI(Mask, row + search_row, col + search_col, visited_list, group);
					}
				}
			}

		}
	}
}

//Function:A function to obtain the regions in an image - makes use of Expand_ROI
//Input: Image
//Output: Vector of all the Regions of interest
void Get_Image_Regions(const cv::Mat& image, std::vector<std::vector<cv::Point>>& roi_list)
{
	//a vector of all the white (255) points visited in the image so far. This is required since we are checking neighbours recursively, 
	//which would result in an infinite loop if this is not set
	std::vector<cv::Point> visited_list;

	//iterate through every pixel in the image
	std::vector<cv::Point> group;
	for (int row = 0; row < image.rows; row++)
	{
		for (int col = 0; col < image.cols; col++)
		{
			//create a new group (a set of points to define the region)
			group.clear();
			//expand the roi (nb: the function checks if the current pixel is white or not)
			Expand_ROI(image, row, col, visited_list, group);

			//if a group has been identified, add it to the list
			if (!group.empty())
			{
				roi_list.push_back(group);
			}
		}
	}
}


//Function: Reduce False Positives in the images by taking into cosideration the connectivity along the z axis
//Input: Vector of Masks (Images), resolution along the x and y axis (pixel spacing), resolution along the z axis (slice thickness)
//Output: Vector of new masks (with False Positives Reduced)
void False_Positive_Reduction_Z(const std::vector<cv::Mat>& Mask_Vector, std::vector<cv::Mat>& new_Mask_Vector, float resolution_x, float resolution_y, float resolution_z)
{
	std::vector<cv::Mat> output(Mask_Vector.size());
	output = Mask_Vector;
	cv::Mat new_mask;

	//for every mask in the vector
	for (int plane = 0; plane < Mask_Vector.size(); plane++)
	{
		//set the new mask to all black
		new_mask = cv::Mat::zeros(Mask_Vector[plane].rows, Mask_Vector[plane].cols, CV_8U);
		std::vector<std::vector<cv::Point>> roi_list, final_roi_list;

		//Get the regions inside this plane
		Get_Image_Regions(Mask_Vector[plane], roi_list);

		//for every region, get the minimum and maximum x and y
		for (int current_region_index = 0; current_region_index < roi_list.size(); current_region_index++)
		{
			std::vector<cv::Point> current_region = roi_list[current_region_index];

			int min_x = Mask_Vector[plane].cols;
			int max_x = 0;
			int min_y = Mask_Vector[plane].rows;
			int max_y = 0;
			int region_width = 0;
			int region_height = 0;

			for (int current_point = 0; current_point < current_region.size(); current_point++)
			{
				if (current_region[current_point].x < min_x)
				{
					min_x = current_region[current_point].x;
				}

				if (current_region[current_point].x > max_x)
				{
					max_x = current_region[current_point].x;
				}

				if (current_region[current_point].y < min_y)
				{
					min_y = current_region[current_point].y;
				}

				if (current_region[current_point].y > max_y)
				{
					max_y = current_region[current_point].y;
				}
			}

			//obtain its width and height

			region_width = max_x - min_x;
			region_height = max_y - min_y;

			//define the amount of planes that should be searched according to the smallest dimention between the width and height
			float size;
			if (region_width < region_height)
			{
				size = region_width*resolution_x;
			}
			else
			{
				size = region_height*resolution_y;
			}

			//According to the thickness of the slice, determine the number of planes that need to be checked.
			//Note that this is divided by 2, since this will be both top and bottom planes
			//Ex: if slice thickness = 2.5mm, => for a 10mm lesion (determined by width and height), 4 planes need to be checked: 2 above and 2 below.

			int plane_check = floor((round(size / resolution_z)) / 2);

			if (plane_check < 1)
			{
				plane_check = 1;
			}

			//Get the images of the top planes and the bottom planes
			int top_count = 0;
			int bot_count = 0;
			cv::Mat top = cv::Mat(output[plane].rows, output[plane].cols, CV_8U, cv::Scalar(255));
			cv::Mat bottom = cv::Mat(output[plane].rows, output[plane].cols, CV_8U, cv::Scalar(255));

			//Use bitwise AND on the series of top images and the series of bottom images
			//Ex: if 2 planes need to be checked on top and 2 on bottom => Bitwise AND between n-1 and n-2; Bitwise AND between n+1 and n+2
			for (int search_plane = 1; search_plane <= plane_check; search_plane++)
			{
				if (plane - search_plane >= 0)
				{
					cv::bitwise_and(top, output[plane - search_plane], top);
					top_count++;
				}

				if (plane + search_plane < Mask_Vector.size())
				{
					cv::bitwise_and(bottom, output[plane + search_plane], bottom);
					bot_count++;
				}
			}

			//Necessary so that the bitwise OR (next step) is not affected if there were no planes at the top or bottom
			if (top_count == 0)
			{
				top = cv::Mat::zeros(Mask_Vector[plane].rows, Mask_Vector[plane].cols, CV_8U);
			}
			if (bot_count == 0)
			{
				bottom = cv::Mat::zeros(Mask_Vector[plane].rows, Mask_Vector[plane].cols, CV_8U);
			}


			//Combine them into one final image using a bitiwise OR
			cv::Mat final_image;
			cv::bitwise_or(top, bottom, final_image);

			//Get the regions in the final image
			Get_Image_Regions(final_image, final_roi_list);


			//iterate through every final region in the final regions list
			for (int final_region_index = 0; final_region_index < final_roi_list.size(); final_region_index++)
			{
				int counter = 0;	//counter for overlapped pixels

				std::vector<cv::Point> final_region = final_roi_list[final_region_index];
				//iterate through every point in the region in the current plane
				for (int current_point = 0; current_point < current_region.size(); current_point++)
				{
					//if one point in the current region is in the final region, the counter of overlapped pixels is incremented 
					if (is_in_list(current_region[current_point], final_region))
					{
						counter++;
					}
				}



				//get the percentage overlap between both images. Note that ths is done for both regions
				float connected_points_current = float(counter) / float(current_region.size());
				float connected_points_final = float(counter) / float(final_region.size());

				//if there is 85% overlap in at least one region
				//the region will be kept, therefore set the mask points contained in the region to white (255)
				if ((connected_points_current>0.85) || (connected_points_final > 0.85))
				{
					for (int current_point = 0; current_point < current_region.size(); current_point++)
					{
						new_mask.at<uchar>(current_region[current_point].y, current_region[current_point].x) = 255;
					}
				}

			}
		}
		//set the vector output to the new mask
		output[plane] = new_mask.clone();
	}
	//copy the output to the vector passed in the parameters
	new_Mask_Vector.clear();
	new_Mask_Vector = output;
}

//Function: Reduce False Positives in the images by taking into cosideration the size and area of a region in the xy plane
//Input: Vector of Masks (Images), resolution along the x and y axis (pixel spacing)
//Output: Vector of new masks (with False Positives Reduced)
void False_Positive_Reduction_XY(const std::vector<cv::Mat>& Mask_Vector, std::vector<cv::Mat>& new_Mask_Vector, float resolution_x, float resolution_y)
{
	std::vector<cv::Mat> output(Mask_Vector.size());

	cv::Mat new_mask;

	//According to the pixel spacing, determine the minimum width and height acceptable for a 5mm lesion.
	//Note that this a delta of 2 is allowed, since a lesion is only 5mm in diameter at it's largest slice (more details in the report, Figure 3.16)
	int min_width = (round(float(5) / resolution_x)) - 2;
	int min_height = (round(float(5) / resolution_y)) - 2;

	//for every plane, start with an empty mask
	for (int plane = 0; plane < Mask_Vector.size(); plane++)
	{
		new_mask = cv::Mat::zeros(Mask_Vector[plane].rows, Mask_Vector[plane].cols, CV_8U);
		std::vector<std::vector<cv::Point>> roi_list;

		//Get the regions inside this plane
		Get_Image_Regions(Mask_Vector[plane], roi_list);

		for (int current_region_index = 0; current_region_index < roi_list.size(); current_region_index++)
		{

			int min_x = Mask_Vector[plane].cols;
			int max_x = 0;
			int min_y = Mask_Vector[plane].rows;
			int max_y = 0;
			int region_width = 0;
			int region_height = 0;
			int region_area = 0;

			//for every region in the current plane 
			std::vector<cv::Point> current_region = roi_list[current_region_index];

			//iterate through every point in the region to get the minimum and maximum x and y. This create a bounding box around a region
			for (int current_point = 0; current_point < current_region.size(); current_point++)
			{
				if (current_region[current_point].x < min_x)
				{
					min_x = current_region[current_point].x;
				}

				if (current_region[current_point].x > max_x)
				{
					max_x = current_region[current_point].x;
				}

				if (current_region[current_point].y < min_y)
				{
					min_y = current_region[current_point].y;
				}

				if (current_region[current_point].y > max_y)
				{
					max_y = current_region[current_point].y;
				}
			}
			//get the width, height and area of the bounding box
			region_width = max_x - min_x;
			region_height = max_y - min_y;
			region_area = region_width*region_height;

			//get the area that the region covers inside the bounding box
			float area_covered = float(current_region.size()) / float(region_area);


			//to keep the region, it must have its length and width span over at least 5mm and the area which it covers must be larger than 70%
			//the 70% threshold is introduced to avoid eliptical regions
			//if the region will be kept, set the mask points contained in the region to white (255)
			if ((region_width >= min_width) && (region_height >= min_height) && (area_covered>0.7))
			{
				for (int current_point = 0; current_point < current_region.size(); current_point++)
				{
					new_mask.at<uchar>(current_region[current_point].y, current_region[current_point].x) = 255;
				}
			}

		}

		//set the vector output to the new mask
		output[plane] = new_mask.clone();

	}
	//copy the output to the vector passed in the parameters
	new_Mask_Vector.clear();
	new_Mask_Vector = output;
}


void Set_Markers(cv::Mat &image, int channel, const cv::Mat& mask) {
	for (int col = 0; col < mask.cols; col++)
	{
		for (int row = 0; row < mask.rows; row++)
		{
			if (mask.at<uchar>(row, col) != 0)
			{
				image.at<cv::Vec3b>(row, col)[channel] = image.at<cv::Vec3b>(row, col)[channel] + 20;
			}
		}
	}
}


//Function: Used to test parameters and functions on a single image - Not used in the Implementation
//Input: /
//Output: /
void Single_Image(std::string path)
{
	//path to image
	//char* path = "C:\\Users\\danie\\Documents\\Final Year Project 2016 - Testing 2\\DICOM_Input\\IMG00030";

	//Original Image
	cv::Mat image;
	//Index in Vector
	int index = 0;
	//Resolutions read from DICOM
	float res_x, res_y, res_z;

	//Load the file and display the image
	Load_Image(index, res_x, res_y, res_z, image, path.c_str());
	cv::imshow("Original Image", image);
	cv::waitKey();

	//Perform Bilateral Filtering and Show the Image
#pragma region bilateral
	//**************************************//
	//			     BILATERAL				//
	//**************************************//

	int neighbourhood = 5;
	double sigma_colour = 40;
	double sigma_space = 70;

	cv::Mat image_bilateral;
	Bilateral_Filtering(image, image_bilateral, 5, 40, 70);
	cv::imshow("Bilateral", image_bilateral);

	//**************************************//
#pragma endregion

	//Perform Weiner Filtering and Show the Image
#pragma region weiner
	//**************************************//
	//			     Weiner 				//
	//**************************************//
	cv::Mat image_wiener;
	Wiener_Denoising(image, image_wiener, 0.018);
	cv::imshow("Wiener", image_wiener);

	//**************************************/
#pragma endregion

	//Perform Anisotropic Diffusion and Show the Image
#pragma region anisotropic
	//**************************************//
	//		  ANISOTROPIC DIFFUSION			//
	//**************************************//

	cv::Mat image_anisotropic;
	Anisotrpic_Diffusion(image, image_anisotropic, 0.25, 10, 45, 45);
	cv::imshow("Anistorpic", image_anisotropic);
	cv::waitKey();
	//**************************************//
#pragma endregion

	//Perform Contrast Enhancement and Show the Image
#pragma region contrast
	//**************************************//
	//        Contrast Enhancement 			//
	//**************************************//

	cv::Mat contrast_image;
	Contrast_Enhancement(image_bilateral, contrast_image, 0.5, 2, 6, 6);
	cv::imshow("Contrast Enhanced", contrast_image);
	cv::waitKey();

	//**************************************//
#pragma endregion

	//Perform Segmentationand Show the Image
#pragma region segmentation
	//**************************************//
	//			  Segmentation  			//
	//**************************************//

	cv::Mat segmented_image, segmentation_mask;
	Segment(contrast_image, segmentation_mask);
	contrast_image.copyTo(segmented_image, segmentation_mask);
	cv::imshow("Segmented Image", segmented_image);

	//**************************************//

	cv::imshow("Segmentation Mask", segmentation_mask);
	cv::waitKey();
#pragma endregion
}

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		std::cout << "Not enough input arguments\n" << std::flush;
		return 0;
	}

	std::string path_in(argv[1]);
	std::string path_out(argv[2]);

#pragma region timing
	// Current date/time based on current system
	time_t now = time(0);
	// Convert now to tm struct for local timezone
	tm* localtm = localtime(&now);
	std::cout << "The local date and time is: " << asctime(localtm) << std::endl;
#pragma endregion

	//Single_Image(path_in);
	//return 0;

	//Get the files in the specified directory
	std::vector<std::string> files;
	Get_Files(path_in + "\\", files);

	//the numebr of planes (slices) is equal to the number of files in the folder
	int planes = files.size();

	//Create vectors to store the original images, enhanced images, segmentation masks, intensity masks and the final lesion masks
	std::vector<cv::Mat> Original_Image(planes), Enhanced_Image(planes), Segmentation_Mask(planes), Intensity_Mask(planes), Lesion_Mask(planes);

	//Create a temporary Mat to hold images, a variable to hold the index of the image and variables for the resolution along all axes
	cv::Mat temp;
	int index = -1;
	float resolution_x, resolution_y, resolution_z;

	//for every image in the folder
	for (int plane = 0; plane < planes; plane++)
	{

		//Load the image into the appropriate vector location
#pragma region load_original
		Load_Image(index, resolution_x, resolution_y, resolution_z, temp, files[plane].c_str());
		Original_Image[index] = temp.clone();
#pragma endregion

		//Perform Bilateral Filtering
#pragma region bilateral
		int neighbourhood = 5;
		double sigma_colour = 40;
		double sigma_space = 70;
		Bilateral_Filtering(Original_Image[index], Enhanced_Image[index], neighbourhood, sigma_colour, sigma_space);
#pragma endregion

		/*#pragma region wiener
		Wiener_Denoising(image_vector[index].Original_Image, image_vector[index].Enhanced_Image, 9, 10);
		#pragma endregion

		#pragma region anisotropic
		double lambda = 0.25;
		int iterations = 35;
		Anisotrpic_Diffusion(image_vector[index].Original_Image, image_vector[index].Enhanced_Image, lambda, iterations);
		#pragma endregion*/

		//Enhance its contrast
#pragma region contrast
		double gamma = 0.55;
		double clip_limit = 2.5;
		int grid_x = 10;
		int grid_y = 10;
		Contrast_Enhancement(Enhanced_Image[index], Enhanced_Image[index], gamma, clip_limit, grid_x, grid_y);
#pragma endregion

		//Segment the Enhanced Image and place the mask in the vector
#pragma region segmentation
		Segment(Enhanced_Image[index], Segmentation_Mask[index]);
#pragma endregion
	}

	/*for (int plane = 0; plane < planes; plane++)
	{
	//Required since vector rotations (used in Segment_Lungs) work on type short
	Enhanced_Image[plane].convertTo(Enhanced_Image[plane], CV_16S);
	}*/

#pragma region detection  

	for (int plane = 0; plane < planes; plane++)
	{
		cv::Mat temp;

		//obtain the segmented images by applying the mask to the enhanced images
		Enhanced_Image[plane].copyTo(temp, Segmentation_Mask[plane]);

		//obtain the intensity mask by thresholding and remove any noise
		cv::threshold(temp, Intensity_Mask[plane], 185, 255, CV_THRESH_BINARY);

	}

	//THE FOLLLOWING ARE NO LONGER NEEDED SINCE A SPHERICITY MASK IS NOT BEING GENERATED	
	/*	Convert_Vector_Image_Type(Enhanced_Image, CV_16S);
	Get_Sphericity_Mask(Intensity_Mask, Enhanced_Image, Sphericity_Mask);

	//Threshold also the sphericity mask, to keep only the points which are spherical
	for (int plane = 0; plane < planes; plane++)
	{
	//Recall: sphericity mask is no longer in range 0-1 but 0-255
	cv::threshold(Sphericity_Mask[plane], Lesion_Mask[plane],0.78*255, 255, CV_THRESH_BINARY);

	}*/


	//The order of connectivity checking is important. Whereas x and y axis connectivity is only dependent on local image
	//information, connectivity along the z axis is dependent on more than 1 image.

	//Reduce false positives along the xy plane
	False_Positive_Reduction_XY(Intensity_Mask, Lesion_Mask, resolution_x, resolution_y);

	//Check connectivity along the z axis
	False_Positive_Reduction_Z(Lesion_Mask, Lesion_Mask, resolution_x, resolution_y, resolution_z);

	//set the output images by applying markers where the lesion mask indicates and save the output to file
	for (int plane = 0; plane < planes; plane++)
	{
		cv::Mat temp;
		//convert the images to RGB so that a particular channel can be increased
		cv::cvtColor(Original_Image[plane], temp, CV_GRAY2RGB);
		//	colorAdjust(temp, 20, 1, Lesion_Mask[plane]);
		Set_Markers(temp, 1, Lesion_Mask[plane]);

		std::string full_path = path_out + "\\" + std::to_string(plane) + ".png";
		cv::imwrite(full_path, temp);
	}

#pragma endregion

#pragma region endtime
	// Current date/time based on current system
	now = time(0);
	// Convert now to tm struct for local timezone
	localtm = localtime(&now);
	std::cout << "The local date and time is: " << asctime(localtm) << std::endl;
#pragma endregion

	return 0;
}
