/* The project is about realization of heterogeneous material microstructures with multiple phases
	using an algroithm based on Monte-Carlo and Cellular Automata methods.
	This is the main part of the program which runs the code given a list of input variables initialized later in the "main" function.  */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <vector>
#include <cmath>
// #include <string>
#include <cstring>
#include <omp.h>
#include <random>

#ifdef __linux__
#include <sys/stat.h>
#include <sys/types.h>
#else
#include <direct.h>
#endif

#include "sobol.hpp"

using namespace std;

// The maximum number of characters for paths to files is assumed to be "MAX_CHAR_PATH" in this application.
#define MAX_CHAR_PATH 128
// The number of pixels in each physical direction of the microstructure realization
#define RESOLUTION_X 180
#define RESOLUTION_X_FLOAT 180.00
#define RESOLUTION_Y 180
#define RESOLUTION_Y_FLOAT 180.00
#define RESOLUTION_Z 180
#define RESOLUTION_Z_FLOAT 180.00
// The maximum radius of neighborhood checked for Cellular Automata 
// It will be added to the dimensions (RESOLUTION_X,Y,Z) of data matrices of microstructure at the beginnings and at the ends so that no out-of-boundary memory access can happen.
#define MAX_NEIGHBORHOOD 15
// DPI is originally the number of pixels per inch for printing/viewing purposes, but it should be prescribed per meter for BMP images
#define DPI_x 2835
#define DPI_y 2835
// For static memory allocation, the maximum file size for each image will be defined
#define MAX_FILE_SIZE 100000

// Definition of a structure used for bmp image writing
typedef struct Image_info {
	// "pad" is the number of bytes needed to make each row of pixel array a mutiplier of 4 bytes. "width_in_bytes" is the number of bytes in each row of pixel array.
	// static unsigned int pad = (4 - (1 * RESOLUTION_X) % 4) % 4;
	unsigned int pad;
	// static unsigned int width_in_bytes = 1 * RESOLUTION_X + pad;
	unsigned int width_in_bytes;
	// static unsigned int image_size = width_in_bytes * RESOLUTION_Y;
	unsigned int image_size;
	// "filesize" is the total bytes of the image file based on the image_size which is the number of bytes describing the pixel array.
	// static unsigned int filesize = 4 * 8 + 54 + image_size;
	unsigned int filesize;
	// Please refer to the Wikipedia page for BMP image format. Here a raw 8-bit color Windows BMP is considered. The number of colors in the color table is also 8 (each defined with 4 bytes).
	// static unsigned char header[54] = { 'B','M', 0,0,0,0, 0,0,0,0, 4 * 8 + 54,0,0,0, 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,8,0, 0,0,0,0, 0,0,0,0, 8,0,0,0, 0,0,0,0 };
	unsigned char header[54];
	// The color table used to determine the color of each pixel in the following pixel array.
	// Each color is determined with 4 bytes of information: Blue intensity (0-255), Green intensity (0-255), Red intensity (0-255), zero in the chosen BMP format
	// static unsigned char color_table[4 * 8] = { 0,0,0,0, 0,0,255,0, 0,255,0,0, 225,0,0,0, 255,255,255,0, 0,255,255,0, 255,0,255,0, 255,255,0,0 };
	unsigned char color_table[4 * (8+7)];
} Image_info;

// Definition of a structure used for the input parameters of the realization algorithm
typedef struct Input_struct {
	// Path to the results directory
	std::string results_directory;

	// Microstructure phase variables
	std::vector<float> volume_fraction;

	// Microstructure seeding variables for cell generations
	int number_of_seeds_initial;
	int number_of_seeds_increment;
	int frequency_of_seed_addition;

	// Initializing parameters for coalescence of the particles	(cells) using the colony algorithm
	int n; // The power of the bundling distribution function
	double omega; // The input probability criterion
	
	// Initializing parameters for stochasitc cell growth (Cellular Automata)
	std::vector<float> p; // The growth probablity threshold in 6 Neumann neighborhoods of a voxel
	std::vector<int> neighborhood_radius; // The min number of empty/void pixels/voxels between different seeds; it must be 0 if contacts between the seeds are allowed.
	std::vector<bool> clustered; // Whether the specific phase should be clustered or not
	std::vector<float> decay_p; // Controls the decay/growth in the probabilty of the seed growth as the microstructure evolution continues. It can be set so that the growth stops after a number of iterations (probablity converges to zero).
} Input_struct;

// Function declaration for the realization algorithm
int realize(Input_struct* input_struct, Image_info* image_info, int* microstructure_id);

// Functions declarations for the auxiliary algorithm of traversing the seed connection graph

// Function to process command line inputs
void get_input(int argc, char* argv[], int* N);

// It will be used in the percolation/clustering section of the realization algorithm to find the connected seeds represented by a graph.
void traverse(int u, const std::vector<std::vector<int>>& connection_array, std::vector<bool>& visited);

// Function to generate image files from the microstructural phase information
void microstructure_image_generator(char* bbuffer, unsigned char (&pixel_array)[MAX_FILE_SIZE], Image_info* image_info, uint8_t (&image_matrix_phase)[RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD][RESOLUTION_Y + 2 + 2 * MAX_NEIGHBORHOOD][RESOLUTION_Z + 2 + 2 * MAX_NEIGHBORHOOD]);



// Driver code
int main(int argc, char* argv[]) {
	int number_of_threads;
	get_input(argc, argv, &number_of_threads);
	int ii; // counter variable
	// Initializing the image info structure
	Image_info image_info = {
		(4 - (1 * RESOLUTION_X) % 4) % 4,
		1 * RESOLUTION_X + (image_info.pad),
		(image_info.width_in_bytes)* RESOLUTION_Y,
		4 * (8 + 7) + 54 + (image_info.image_size),
		{ 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 4 * 8 + 54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0 },
		{ 0,0,0,0, 255,255,255,0, 0,0,255,0, 0,255,0,0, 225,0,0,0, 0,255,255,0, 255,0,255,0, 255,255,0,0, 0,0,128,0, 0,128,0,0, 128,0,0,0, 128,128,128,0, 0,128,128,0, 128,0,128,0, 128,128,0,0 }
	};
	for (ii = 0; ii < 4; ii++) {
		image_info.header[2 + ii] = (unsigned char)((image_info.filesize >> (8 * ii)) & 255);
		image_info.header[18 + ii] = (unsigned char)((RESOLUTION_X >> (8 * ii)) & 255);
		image_info.header[22 + ii] = (unsigned char)((RESOLUTION_Y >> (8 * ii)) & 255);
		image_info.header[38 + ii] = (unsigned char)((DPI_x >> (8 * ii)) & 255);
		image_info.header[42 + ii] = (unsigned char)((DPI_y >> (8 * ii)) & 255);
	}



	/* Parsing the input file of the program */
	char* buffer = new char[MAX_CHAR_PATH]; char* buffer2 = new char[MAX_CHAR_PATH];
	int line_number = 0;
	int errnum; // Error number
	int return_value;

	double time_start = omp_get_wtime();
	FILE* summary_file = fopen("summary_openmp.txt", "a");
	fprintf(summary_file, "%i\t", number_of_threads);

	FILE* input_file = fopen("input.txt", "r"); // Trying to open/read the input file in the program directory with file name "input.txt"
	if (input_file == NULL) { // The code block will be executed if opening the above input file is not successful.
		input_file = fopen("input.csv", "r"); // Since no "input.txt" is found, this line checks whether there is "input.csv" file as input.
		if (input_file == NULL) {
			errnum = errno;
			//fprintf(stderr, "Value of errno: %d\n", errno);
			//perror("Error printed by perror");
			fprintf(stderr, "Error opening the input file (input.txt or input.csv): %s\n", strerror(errnum));
			exit(EXIT_FAILURE);
		} else { // The code block will be executed if there is "input.csv" in the program directory.
			// Counting the number of lines
			while (fgets(buffer, MAX_CHAR_PATH, input_file))
				line_number++;
			if (line_number == 0) {
				fprintf(stderr, "Error: the input file is empty.\n");
				exit(EXIT_FAILURE);
			}
			rewind(input_file); // Going back to the beginning of the input file
			fgets(buffer, MAX_CHAR_PATH, input_file); // Reading one complete line (including end of line character) from the file
			if (buffer[strlen(buffer) - 1] != '\n') {
				fprintf(stderr, "Error: the results path in the first line of input file is too long (>260 chars).\n");
				exit(EXIT_FAILURE);
			}
			if (strlen(buffer) == 1 || (strlen(buffer) > 1 && buffer[0] == ' ')) {
				fprintf(stderr, "Error: the results path in the first line of input file is empty or the first character(s) in the line is(are) space(s). Begin the text file with input parameters if you want to use the default directory path for results storage.\n");
				exit(EXIT_FAILURE);
			}
			// Check whether the first line of input file contains an alphabetic string (path to the results directory)
			for (ii = 0; ii < MAX_CHAR_PATH && buffer[ii] != '\0'; ii++) {
				if ((buffer[ii] >= 'a' && buffer[ii] <= 'z') || (buffer[ii] >= 'A' && buffer[ii] <= 'Z')) {
					if (line_number == 1) {
						fprintf(stderr, "Error: the input file has no information about the microstructure parameters.\n");
						exit(EXIT_FAILURE);
					}
					else {
						line_number--; // Subtracting one line number from the total would result in the total number of microstructures in the input file.
						break;
					}
				}
			}
			if (ii == MAX_CHAR_PATH || buffer[ii] == '\0') { // This code block will be executed if the first line does not have any alphabetic characters for the results directory.
				rewind(input_file); // Going back to the beginning of the input file to read the microstructure parameters from the beginning
				strcpy(buffer, "Results"); // Copying the default value for the results directory into "buffer"
			}
		}
	} else { // The code block will be executed if there is "input.txt" in the program directory.
		// Counting the number of lines
		while (fgets(buffer, MAX_CHAR_PATH, input_file))
			line_number++;
		if (line_number == 0) {
			fprintf(stderr, "Error: the input file is empty.\n");
			exit(EXIT_FAILURE);
		}
		rewind(input_file); // Going back to the beginning of the input file
		fgets(buffer, MAX_CHAR_PATH, input_file); // Reading one complete line (including end of line character) from the file
		if (buffer[strlen(buffer) - 1] != '\n') {
			fprintf(stderr, "Error: the results path in the first line of input file is too long (>260 chars).\n");
			exit(EXIT_FAILURE);
		}
		if (strlen(buffer) == 1 || (strlen(buffer) > 1 && buffer[0] == ' ')) {
			fprintf(stderr, "Error: the results path in the first line of input file is empty or the first character(s) in the line is(are) space(s). Begin the text file with input parameters if you want to use the default directory path for results storage.\n");
			exit(EXIT_FAILURE);
		}
		// Check whether the first line of input file contains an alphabetic string (path to the results directory)
		for (ii = 0; ii < MAX_CHAR_PATH && buffer[ii] != '\0'; ii++) {
			if ((buffer[ii] >= 'a' && buffer[ii] <= 'z') || (buffer[ii] >= 'A' && buffer[ii] <= 'Z')) {
				if (line_number == 1) {
					fprintf(stderr, "Error: the input file has no information about the microstructure parameters.\n");
					exit(EXIT_FAILURE);
				}
				else {
					line_number--; // Subtracting one line number from the total would result in the total number of microstructures in the input file.
					break;
				}
			}
		}
		if (ii == MAX_CHAR_PATH || buffer[ii] == '\0') { // This code block will be executed if the first line does not have any alphabetic characters for the results directory.
			rewind(input_file); // Going back to the beginning of the input file to read the microstructure parameters from the beginning
			strcpy(buffer, "Results"); // Copying the default value for the results directory into "buffer"
		}
		//fclose(input_file);
		//exit(EXIT_SUCCESS);
	}
	
	// Making the results directory
	sscanf(buffer, "%s", buffer2);
#ifdef __linux__
	return_value = mkdir(buffer2, 777);
#else
	return_value = _mkdir(buffer2);
#endif
	if (return_value != 0) {
		if (errno == EEXIST)
			printf("Warning: the results directory already exists. No new folder was created.\n");
		else {
			fprintf(stderr, "Error: the results path in the first line of input file could not be found/created.\n");
			exit(EXIT_FAILURE);
		}
	}
	


	/* Extracting numeric microstructure data from each line of the input file and submitting a new task of "realization" based on each set of parameters */
	Input_struct input_struct;
	string directory = buffer2;
	//int number_of_threads = omp_get_max_threads(); // The number of utilized CPU workers (threads) for parallelization
	// int number_of_threads = 10; // The number of utilized CPU workers (threads) for parallelization
	printf("\n----------------\nRealization code execution...\nomp_get_max_threads(): %d; num_threads(): %d\n", omp_get_max_threads(), number_of_threads); // omp_get_num_threads()
#pragma omp parallel for num_threads(number_of_threads) private(input_struct) firstprivate(image_info, buffer, buffer2, directory) schedule(static, 1)
	for (int i = 1; i <= line_number; i++) {
		fgets(buffer, MAX_CHAR_PATH, input_file); // Reading one complete line (including end of line character) from the file
		// Path to the results directory
		input_struct.results_directory = directory;
		int microstructure_id, number_of_phases; // "number_of_phases" does not include the background/black phase (so the total number is a unit larger than this parameter).
		sscanf(buffer, "%d, %d, %s", &microstructure_id, &number_of_phases, buffer2);
		char* pointer_to_char;
		pointer_to_char = strchr(buffer, ',');
		pointer_to_char = strchr(pointer_to_char + 1, ',');

		// Microstructure phase variables
		float temp;
		input_struct.volume_fraction.clear();
		for (int j = 0; j < number_of_phases; j++) {
			sscanf(pointer_to_char + 1, "%f, %s", &temp, buffer2);
			input_struct.volume_fraction.push_back(temp);
			pointer_to_char = strchr(pointer_to_char + 1, ',');
		}
		
		// Microstructure seeding variables for cell generations
		//input_struct.number_of_seeds_initial = (int)(1 * (RESOLUTION_X_FLOAT + RESOLUTION_Y_FLOAT + RESOLUTION_Z_FLOAT) / 3.0);
		sscanf(pointer_to_char + 1, "%d, %s", &input_struct.number_of_seeds_initial, buffer2);
		pointer_to_char = strchr(pointer_to_char + 1, ',');
		sscanf(pointer_to_char + 1, "%d, %s", &input_struct.number_of_seeds_increment, buffer2);
		pointer_to_char = strchr(pointer_to_char + 1, ',');
		sscanf(pointer_to_char + 1, "%d, %s", &input_struct.frequency_of_seed_addition, buffer2);
		pointer_to_char = strchr(pointer_to_char + 1, ',');
		
		// Initializing parameters for coalescence of the particles	(cells) using the colony algorithm
		//input_struct.n = 2; // The power of the bundling distribution function
		//input_struct.omega = 0.5; // The input probability criterion
		sscanf(pointer_to_char + 1, "%d, %s", &input_struct.n, buffer2);
		pointer_to_char = strchr(pointer_to_char + 1, ',');
		sscanf(pointer_to_char + 1, "%lf, %s", &input_struct.omega, buffer2);
		pointer_to_char = strchr(pointer_to_char + 1, ',');

		// Initializing parameters for stochasitc cell growth (Cellular Automata)
		int temp2;
		input_struct.neighborhood_radius.clear();
		for (int j = 0; j < number_of_phases; j++) {
			sscanf(pointer_to_char + 1, "%d, %s", &temp2, buffer2);
			input_struct.neighborhood_radius.push_back(temp2);
			pointer_to_char = strchr(pointer_to_char + 1, ',');
		}

		input_struct.clustered.clear();
		for (int j = 0; j < number_of_phases; j++) {
			sscanf(pointer_to_char + 1, "%d, %s", &temp2, buffer2);
			input_struct.clustered.push_back((bool)temp2);
			pointer_to_char = strchr(pointer_to_char + 1, ',');
		}

		input_struct.decay_p.clear();
		for (int j = 0; j < number_of_phases; j++) {
			sscanf(pointer_to_char + 1, "%f, %s", &temp, buffer2);
			input_struct.decay_p.push_back(temp);
			pointer_to_char = strchr(pointer_to_char + 1, ',');
		}

		input_struct.p.clear();
		for (int j = 0; j < 6*number_of_phases; j++) {
			sscanf(pointer_to_char + 1, "%f, %s", &temp, buffer2);
			input_struct.p.push_back(temp);
			pointer_to_char = strchr(pointer_to_char + 1, ',');
		}
		// Running the main realization algorithm based on the given inputs
		realize(&input_struct, &image_info, &microstructure_id);
	}
	delete[] buffer;
	delete[] buffer2;
	fclose(input_file);
	printf("Realization code finished!\n----------------\n");

	fprintf(summary_file, "%9.2f\n", omp_get_wtime() - time_start);
	fclose(summary_file);
	return 0;
}



/* "realize" function makes a microstructure based on the microstructure parameters contained by "input_struct" structure. */
int realize(Input_struct* input_struct, Image_info* image_info, int* microstructure_id) {
	// Showing/registering the passed parameters to this function
	int i = 0;
	char* bbuffer = new char[MAX_CHAR_PATH]; // To contain the string of file name.
	snprintf(bbuffer, MAX_CHAR_PATH, "%s/log_thread_%02d.txt", input_struct->results_directory.c_str(), omp_get_thread_num()); // Creating a log file for each processing thread/instantiation of this function
// #pragma warning(suppress : 4996) // Similarly done for the whole enviroment by adding "_CRT_SECURE_NO_WARNING=1" to the preprocessor definitions  
	FILE* log_file = fopen(bbuffer, "a");
	fprintf(log_file, "\n\n%06d- number_of_phases = %lu; volume_fraction(s) = %.3f", *microstructure_id, input_struct->volume_fraction.size(), input_struct->volume_fraction[0]);
	for (i = 1; i < input_struct->volume_fraction.size(); i++)
		fprintf(log_file, ", %.3f", input_struct->volume_fraction[i]);
	fprintf(log_file, "; number_of_seeds_initial = %d; number_of_seeds_increment = %d; frequency_of_seed_addition = %d; n = %d; omega = %.3f", input_struct->number_of_seeds_initial, input_struct->number_of_seeds_increment, input_struct->frequency_of_seed_addition, input_struct->n, input_struct->omega);
	fprintf(log_file, ";\n%06d-", *microstructure_id);
	for (i = 0; i < input_struct->volume_fraction.size(); i++)
		fprintf(log_file, " Phase %2d: neighborhood_radius = %2d; clustered = %s; decay_p = %.4f; p[] = %.3f, %.3f, %.3f, %.3f, %.3f, %.3f;", i + 1, input_struct->neighborhood_radius[i], input_struct->clustered[i] ? "true" : "false", input_struct->decay_p[i], input_struct->p[0 + 6 * i], input_struct->p[1 + 6 * i], input_struct->p[2 + 6 * i], input_struct->p[3 + 6 * i], input_struct->p[4 + 6 * i], input_struct->p[5 + 6 * i]);
	fprintf(log_file, "\n");
	
	double start_time, start_time_trial, end_time;
	start_time = omp_get_wtime(); // Start time of the current microstructure realization (thread-safe/private)
	// Seeding the RNG based on the time of code execution to get different random numbers in each execution
	// srand(time(NULL));
	// std::mt19937 mt(time(NULL));
	// std::mt19937 mt((((unsigned)time(NULL)) & 0xFFFFFFF0) | (omp_get_thread_num() + 1));
	std::mt19937 mt(((unsigned)time(NULL) & 0xFFFFFFF0) | (omp_get_thread_num() + 1));
	// std::mt19937 mt((unsigned long)time(NULL));
	std::uniform_real_distribution<float> dist(0, 1);
	int j = 0;
	int k = 0;
	int x, y, z, r_x, r_y;
	int seed_x = MAX_NEIGHBORHOOD;
	int seed_y = MAX_NEIGHBORHOOD;
	int seed_z = MAX_NEIGHBORHOOD;
	bool gotoMainLoop = false;
	bool gotoMainLoop2 = false;
	// Initializing the quasirandom generator of Sobol sequence as a space-filling Design of Experiment for new seed positions (3D vectors)
	// Skipping the few initial outputs of Sobol for better randomness
	long long int skip_sobol_sequence;
	skip_sobol_sequence = 100;
	long long int* seed_sobol_quasirandom;
	seed_sobol_quasirandom = &skip_sobol_sequence;
	// Defining a 3D vector for a new seed position inside the microstructure.
	// It is "float" since the imported sobol library gives vectors only in 0<x_i<1 format. This will be scaled to the resolution of microstructure as "int".
	double quasirandom_vector[3];
	


	/* Initializing/declaring the input varibales needed to make a microstructure  */
	int loop_threshold;
	loop_threshold = (RESOLUTION_X) * (RESOLUTION_Y) * (RESOLUTION_Z);
	// loop_threshold = (RESOLUTION_X + 2 + 2*(input_struct->neighborhood_radius-1)) * (RESOLUTION_Y + 2 + 2 * (input_struct->neighborhood_radius - 1)) * (RESOLUTION_Z + 2 + 2 * (input_struct->neighborhood_radius - 1));
	static uint8_t image_matrix_phase[RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD][RESOLUTION_Y + 2 + 2 * MAX_NEIGHBORHOOD][RESOLUTION_Z + 2 + 2 * MAX_NEIGHBORHOOD];
#pragma omp threadprivate(image_matrix_phase)
	static uint8_t is_grown_voxel[RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD][RESOLUTION_Y + 2 + 2 * MAX_NEIGHBORHOOD][RESOLUTION_Z + 2 + 2 * MAX_NEIGHBORHOOD];
#pragma omp threadprivate(is_grown_voxel)
	static unsigned int image_matrix_cluster[RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD][RESOLUTION_Y + 2 + 2 * MAX_NEIGHBORHOOD][RESOLUTION_Z + 2 + 2 * MAX_NEIGHBORHOOD];
#pragma omp threadprivate(image_matrix_cluster)
	static unsigned char pixel_array[MAX_FILE_SIZE]; // 1D array compression of all 2D pixels 
#pragma omp threadprivate(pixel_array)
	// unsigned char* pixel_array = new unsigned char[image_info->image_size];
	std::vector<int> seed_volume;
	// Initializing the image matrix voxels with the void/background phase "0"
	for (i = 0; i < RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD; i++) {
		for (j = 0; j < RESOLUTION_Y + 2 + 2 * MAX_NEIGHBORHOOD; j++) {
			for (k = 0; k < RESOLUTION_Z + 2 + 2 * MAX_NEIGHBORHOOD; k++) {
				image_matrix_phase[i][j][k] = 0;
				is_grown_voxel[i][j][k] = 0;
				image_matrix_cluster[i][j][k] = 0;
			}
		}
	}

	// Microstructure phase variables
	// Defining a boolian array to keep track of active phases, which will be developed in the microstructure evolution code.
	// The inactive phases are those which have reached the volume fraction of their phases during microstructure evolution.
	int number_of_phases;
	number_of_phases = (int)(input_struct->volume_fraction.size()) + 1;
	std::vector<bool> is_completed_phase(number_of_phases - 1, false);
	std::vector<int> phase_volume(number_of_phases - 1, 0); // Initialization of each phase volume in terms of pixels/voxels
	std::vector<int> phase_volume_threshold(number_of_phases - 1); // Definition: the total volume of each phase at the end of microstructure realization in terms of pixels/voxels
	for (i = 0; i < number_of_phases - 1; i++) {
		if (input_struct->neighborhood_radius[i] == 0 || !input_struct->clustered[i])
			phase_volume_threshold[i] = (int)(input_struct->volume_fraction[i] * loop_threshold);
		else // Lowering the threshold of volume fraction so that the final clustering can be performed before reaching the threshold
			phase_volume_threshold[i] = (int)(input_struct->volume_fraction[i] * loop_threshold * 0.80);
	}
	std::vector<int> phase_volume_previous(number_of_phases - 1, 0);
	std::vector<float> phase_volume_min(number_of_phases - 1, 0.20);
	for (i = 0; i < number_of_phases - 1; i++) {
		if (input_struct->volume_fraction[i] < 0.20)
			phase_volume_min[i] = 0.0;
		else
			phase_volume_min[i] = 0.20;
	}
	std::vector<float> phase_volume_max(number_of_phases - 1, 0.55);
	for (i = 0; i < number_of_phases - 1; i++) {
		if (input_struct->volume_fraction[i] > 0.55)
			phase_volume_max[i] = 1.0;
		else
			phase_volume_max[i] = 0.55;
	}
	std::vector<int> stagnant_iteration(number_of_phases - 1, 0);
	int stagnant_iteration_threshold = 5;

	// Microstructure seeding variables for cell generations
	int number_of_seeds = input_struct->number_of_seeds_initial; // Can be changed in the microstructure evolution
	float random_fraction;
	int phase = 0;
	int cluster = 0;
	int seed_location[3];



	/* Cellular Automata evolution of the microstructure with synchronous generation and growth of material cells inside the microstructure  */
	// Infinite loop of microstructure evolution will be broken if all the Phase_Volume[:] reach the Thresholds_Phase_Volume_Threshold[:].
	int iteration, seed_number, iteration_seeding;
	iteration = 0;
	seed_number = 1;
	std::vector<int> seed_generation_start;
	seed_generation_start.clear();
	iteration_seeding = 0;
	float sum_f;
	int sum;
	bool desired_microstructure_generated = false;
	int iteration_realization = 0;
	std::vector<int> phase_volume_copy(number_of_phases - 1, -1); // Initialization of each phase volume in terms of pixels/voxels

	while (!desired_microstructure_generated && (iteration_realization < 6)) { // Keep regenerating the microsrcuture all over again until some conditions are satisfied in the end (e.g., the volume fractions are within some bounds.)
	
	iteration_realization++;
	if (iteration_realization % 6 == 0) skip_sobol_sequence = 100;
	start_time_trial = omp_get_wtime(); // Start time of the current microstructure realization (thread-safe/private)
	// Initialization
	std::mt19937 mt(((unsigned)time(NULL) & 0xFFFFFFF0) | (omp_get_thread_num() + 1));
	// std::mt19937 mt(((unsigned long)time(NULL))+(unsigned long)(1000000*dist(mt)));
	std::uniform_real_distribution<float> dist(0, 1);
	// fprintf(log_file, "%06d- %f\n", *microstructure_id, dist(mt));
	// fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
	int seed_x = MAX_NEIGHBORHOOD;
	int seed_y = MAX_NEIGHBORHOOD;
	int seed_z = MAX_NEIGHBORHOOD;
	// skip_sobol_sequence = skip_sobol_sequence + 1000;
	seed_volume.clear();
	for (i = 0; i < RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD; i++) {
		for (j = 0; j < RESOLUTION_Y + 2 + 2 * MAX_NEIGHBORHOOD; j++) {
			for (k = 0; k < RESOLUTION_Z + 2 + 2 * MAX_NEIGHBORHOOD; k++) {
				image_matrix_phase[i][j][k] = 0;
				is_grown_voxel[i][j][k] = 0;
				image_matrix_cluster[i][j][k] = 0;
			}
		}
	}
	for (i = 0; i < number_of_phases - 1; i++) {
		is_completed_phase[i] = false;
		phase_volume[i] = 0;
		if (input_struct->neighborhood_radius[i] == 0 || !input_struct->clustered[i])
			phase_volume_threshold[i] = (int)(input_struct->volume_fraction[i] * loop_threshold);
		else // Lowering the threshold of volume fraction so that the final clustering can be performed before reaching the threshold
			phase_volume_threshold[i] = (int)(input_struct->volume_fraction[i] * loop_threshold * 0.80);
		phase_volume_previous[i] = 0;
		stagnant_iteration[i] = 0;
	}
	iteration = 0;
	seed_number = 1;
	seed_generation_start.clear();
	iteration_seeding = 0;

	fprintf(log_file, "%06d- Start...\n", *microstructure_id);
	fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
	while (1 > 0) {
		// Check if the current process has been taking too long to finish, or if it has taken more than the time-out limit (2100 sec)
		end_time = omp_get_wtime(); // End time of the current microstructure realization (thread-safe/private)
		if (end_time - start_time > 3600.0) {
			fprintf(log_file, "%06d- Timeout: time = %02ld:%02ld:%02ld\n\n", *microstructure_id, (long int)(end_time - start_time) / 3600, ((long int)(end_time - start_time) % 3600) / 60, ((long int)(end_time - start_time) % 3600) % 60);
			fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
			return 0;
		}
		std::vector<uint8_t> phase_labels_active(number_of_phases - 1, 0);
		for (i = 0; i < number_of_phases - 1; i++) {
			phase_labels_active[i] = i + 1;
		}
		std::vector<float> volume_fraction_active(input_struct->volume_fraction.begin(), input_struct->volume_fraction.end());
		std::vector<float> biased_interval(number_of_phases, 0.0);
		sum_f = 0;
		for (auto& el : input_struct->volume_fraction)
			sum_f += el;
		for (i = 1; i < number_of_phases; i++)
			biased_interval[i] = biased_interval[i - 1] + input_struct->volume_fraction[i - 1] / sum_f;
		// Remove/Avoid phases which have reached/surpassed their volume fraction limits for microstructure evolution (seed generation/growth)
		sum = 0;
		for (i = 0; i < number_of_phases - 1; i++) {
			if (phase_volume[i] > phase_volume_threshold[i] || stagnant_iteration[i] > stagnant_iteration_threshold || (stagnant_iteration[i] == stagnant_iteration_threshold && phase_volume[i] <= phase_volume_previous[i])) {
				fprintf(log_file, "%06d- Phase %2d: Volume Fraction(VF) = %.3f; Realization VF = %.3f; stagnant_iteration = %d\n", *microstructure_id, i + 1, volume_fraction_active[i - sum], ((double)phase_volume[i] / (double)loop_threshold), stagnant_iteration[i]);
				fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
				is_completed_phase[i] = true;
				phase_labels_active.erase(phase_labels_active.begin() + i - sum);
				biased_interval.erase(biased_interval.begin() + 1 + i - sum);
				volume_fraction_active.erase(volume_fraction_active.begin() + i - sum);
				++sum;
			} else if (phase_volume[i] <= phase_volume_previous[i])
				++stagnant_iteration[i];
			else {
				stagnant_iteration[i] = 0;
				phase_volume_previous[i] = phase_volume[i];
			}
		}
		if (sum == number_of_phases - 1) break; // If all phases have reached their designated volume fractions, break the infinite evolution loop.

		// Generate new cell(s)/seed(s) if this is the right increment.
		if (iteration % input_struct->frequency_of_seed_addition == 0 && seed_x + seed_y + seed_z < RESOLUTION_X + MAX_NEIGHBORHOOD + RESOLUTION_Y + MAX_NEIGHBORHOOD + RESOLUTION_Z + MAX_NEIGHBORHOOD) {
			sum_f = 0.0;
			for (auto& el : volume_fraction_active)
				sum_f += el;
			for (i = 1; i < number_of_phases - sum; i++)
				biased_interval[i] = biased_interval[i - 1] + volume_fraction_active[i - 1] / sum_f;
			if (iteration != 0)
				number_of_seeds = (number_of_seeds + input_struct->number_of_seeds_increment) < 1 ? 1 : (number_of_seeds + input_struct->number_of_seeds_increment);
			else
				number_of_seeds = input_struct->number_of_seeds_initial;

			for (i = 0; i < number_of_seeds; i++) {
				// random_fraction = rand() / ((float)(RAND_MAX + 1.0));
				random_fraction = dist(mt);
				for (j = 1; j < number_of_phases - sum; j++) {
					if (biased_interval[j - 1] <= random_fraction && random_fraction < biased_interval[j]) {
						phase = phase_labels_active[j - 1];
						break;
					}
				}

				if (iteration_seeding < 5000) { // Check whether a new seed has been found in the Sobol sequence after 5000 iterations. Otherwise, an exhaustive search will begin in the else clause.  
					iteration_seeding = 0;
					while (iteration_seeding < 2000) {
						//seed_location[0] = rand() % (RESOLUTION_X + 2);
						//seed_location[1] = rand() % (RESOLUTION_Y + 2);
						//seed_location[2] = rand() % (RESOLUTION_Z + 2);
						i8_sobol(3, seed_sobol_quasirandom, quasirandom_vector);
						seed_location[0] = 1 + MAX_NEIGHBORHOOD + (int)(quasirandom_vector[0] * (RESOLUTION_X_FLOAT));
						seed_location[1] = 1 + MAX_NEIGHBORHOOD + (int)(quasirandom_vector[1] * (RESOLUTION_Y_FLOAT));
						seed_location[2] = 1 + MAX_NEIGHBORHOOD + (int)(quasirandom_vector[2] * (RESOLUTION_Z_FLOAT));
						// if (iteration_seeding == 0) {
						// 	fprintf(log_file, "%06d- %d\n", *microstructure_id, skip_sobol_sequence);
						// 	fprintf(log_file, "%06d- %d\n", *microstructure_id, seed_sobol_quasirandom);
						// 	fprintf(log_file, "%06d- %d, %d, %d\n", *microstructure_id, seed_location[0], seed_location[1], seed_location[2]);
						// 	fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
						// }
						gotoMainLoop = false;
						for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
							r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
							for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
								r_y = r_x - abs(y);
								for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
									if (image_matrix_phase[seed_location[0] + x][seed_location[1] + y][seed_location[2] + z])
										gotoMainLoop = true;
								}
							}
						}
						if (!gotoMainLoop)
							//if (!image_matrix_phase[seed_location[0]][seed_location[1]][seed_location[2]] && !image_matrix_phase[seed_location[0] + 1][seed_location[1]][seed_location[2]] && !image_matrix_phase[seed_location[0] - 1][seed_location[1]][seed_location[2]] && !image_matrix_phase[seed_location[0]][seed_location[1] + 1][seed_location[2]] && !image_matrix_phase[seed_location[0]][seed_location[1] - 1][seed_location[2]] && !image_matrix_phase[seed_location[0]][seed_location[1]][seed_location[2] + 1] && !image_matrix_phase[seed_location[0]][seed_location[1]][seed_location[2] - 1] && // 6 Neumann neighborhood  
							//	!image_matrix_phase[seed_location[0] + 1][seed_location[1] + 1][seed_location[2] + 1] && !image_matrix_phase[seed_location[0] - 1][seed_location[1] + 1][seed_location[2] + 1] && !image_matrix_phase[seed_location[0] + 1][seed_location[1] - 1][seed_location[2] + 1] && !image_matrix_phase[seed_location[0] + 1][seed_location[1] + 1][seed_location[2] - 1] && !image_matrix_phase[seed_location[0] - 1][seed_location[1] - 1][seed_location[2] + 1] && !image_matrix_phase[seed_location[0] + 1][seed_location[1] - 1][seed_location[2] - 1] && !image_matrix_phase[seed_location[0] - 1][seed_location[1] + 1][seed_location[2] - 1] && !image_matrix_phase[seed_location[0] - 1][seed_location[1] - 1][seed_location[2] - 1] && // 8 diagonal elements  
							//	!image_matrix_phase[seed_location[0] + 1][seed_location[1] + 1][seed_location[2]] && !image_matrix_phase[seed_location[0] - 1][seed_location[1] + 1][seed_location[2]] && !image_matrix_phase[seed_location[0] + 1][seed_location[1] - 1][seed_location[2]] && !image_matrix_phase[seed_location[0] - 1][seed_location[1] - 1][seed_location[2]] && !image_matrix_phase[seed_location[0] + 1][seed_location[1]][seed_location[2] + 1] && !image_matrix_phase[seed_location[0] + 1][seed_location[1]][seed_location[2] - 1] && !image_matrix_phase[seed_location[0] - 1][seed_location[1]][seed_location[2] + 1] && !image_matrix_phase[seed_location[0] - 1][seed_location[1]][seed_location[2] - 1] && !image_matrix_phase[seed_location[0]][seed_location[1] + 1][seed_location[2] + 1] && !image_matrix_phase[seed_location[0]][seed_location[1] - 1][seed_location[2] + 1] && !image_matrix_phase[seed_location[0]][seed_location[1] + 1][seed_location[2] - 1] && !image_matrix_phase[seed_location[0]][seed_location[1] - 1][seed_location[2] - 1]) // 12 remaining elements  
						{
							image_matrix_phase[seed_location[0]][seed_location[1]][seed_location[2]] = phase;
							image_matrix_cluster[seed_location[0]][seed_location[1]][seed_location[2]] = seed_number;
							seed_volume.push_back(1);
							seed_generation_start.push_back(iteration);
							++seed_number;
							phase_volume[phase - 1]++;
							break;
							//random_fraction = dist(mt);
							//if (pow(random_fraction, input_struct->n) > (input_struct->omega) * exp((input_struct->omega) * (input_struct->volume_fraction[phase - 1]))) {
							//	image_matrix_phase[seed_location[0]][seed_location[1]][seed_location[2]] = phase;
							//	image_matrix_cluster[seed_location[0]][seed_location[1]][seed_location[2]] = seed_number;
							//	++seed_number;
							//	phase_volume[phase - 1]++;
							//	break;
							//}
						}
						++iteration_seeding;
					}
				}
				else {
					//if (seed_x + seed_y + seed_z >= RESOLUTION_X + MAX_NEIGHBORHOOD + RESOLUTION_Y + MAX_NEIGHBORHOOD + RESOLUTION_Z + MAX_NEIGHBORHOOD) printf("ssss!\n");
					gotoMainLoop2 = false;
					while (seed_x < RESOLUTION_X + MAX_NEIGHBORHOOD && !gotoMainLoop2) {
						seed_x++;
						while (seed_y < RESOLUTION_Y + MAX_NEIGHBORHOOD && !gotoMainLoop2) {
							seed_y++;
							while (seed_z < RESOLUTION_Z + MAX_NEIGHBORHOOD && !gotoMainLoop2) {
								seed_z++;
								seed_location[0] = seed_x;
								seed_location[1] = seed_y;
								seed_location[2] = seed_z;
								gotoMainLoop = false;
								for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
									r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
									for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
										r_y = r_x - abs(y);
										for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
											if (image_matrix_phase[seed_location[0] + x][seed_location[1] + y][seed_location[2] + z])
												gotoMainLoop = true;
										}
									}
								}
								if (!gotoMainLoop) {
									image_matrix_phase[seed_location[0]][seed_location[1]][seed_location[2]] = phase;
									image_matrix_cluster[seed_location[0]][seed_location[1]][seed_location[2]] = seed_number;
									seed_generation_start.push_back(iteration);
									seed_volume.push_back(1);
									++seed_number;
									phase_volume[phase - 1]++;
									gotoMainLoop2 = true;
								}
							}
						}
					}
				}
			}
		}
		
		// Grow the cells for one iteration of cellular automata
		for (i = 1 + MAX_NEIGHBORHOOD; i <= RESOLUTION_X + MAX_NEIGHBORHOOD; i++) {
			for (j = 1 + MAX_NEIGHBORHOOD; j <= RESOLUTION_Y + MAX_NEIGHBORHOOD; j++) {
				for (k = 1 + MAX_NEIGHBORHOOD; k <= RESOLUTION_Z + MAX_NEIGHBORHOOD; k++) {
					phase = image_matrix_phase[i][j][k];
					cluster = image_matrix_cluster[i][j][k];
					if (phase != 0 && !is_completed_phase[phase - 1] && is_grown_voxel[i][j][k] == 0 && stagnant_iteration[phase - 1] != stagnant_iteration_threshold) {
						//if (image_matrix_phase[i + 1][j][k] == 0 && (input_struct->p[0] - dist(mt)) > 0) { image_matrix_phase[i + 1][j][k] = phase; is_grown_voxel[i + 1][j][k] = 1; phase_volume[phase - 1]++; }
						//if (image_matrix_phase[i - 1][j][k] == 0 && (input_struct->p[1] - dist(mt)) > 0) { image_matrix_phase[i - 1][j][k] = phase; is_grown_voxel[i - 1][j][k] = 1; phase_volume[phase - 1]++; }
						//if (image_matrix_phase[i][j + 1][k] == 0 && (input_struct->p[2] - dist(mt)) > 0) { image_matrix_phase[i][j + 1][k] = phase; is_grown_voxel[i][j + 1][k] = 1; phase_volume[phase - 1]++; }
						//if (image_matrix_phase[i][j - 1][k] == 0 && (input_struct->p[3] - dist(mt)) > 0) { image_matrix_phase[i][j - 1][k] = phase; is_grown_voxel[i][j - 1][k] = 1; phase_volume[phase - 1]++; }
						//if (image_matrix_phase[i][j][k + 1] == 0 && (input_struct->p[4] - dist(mt)) > 0) { image_matrix_phase[i][j][k + 1] = phase; is_grown_voxel[i][j][k + 1] = 1; phase_volume[phase - 1]++; }
						//if (image_matrix_phase[i][j][k - 1] == 0 && (input_struct->p[5] - dist(mt)) > 0) { image_matrix_phase[i][j][k - 1] = phase; is_grown_voxel[i][j][k - 1] = 1; phase_volume[phase - 1]++; }
						//if ((input_struct->p[0] - dist(mt)) > 0 && !image_matrix_phase[i + 1][j][k] && (!image_matrix_phase[i + 2][j][k] || !(image_matrix_cluster[i + 2][j][k] - cluster)) && (!image_matrix_phase[i + 1][j + 1][k] || !(image_matrix_cluster[i + 1][j + 1][k] - cluster)) && (!image_matrix_phase[i + 1][j - 1][k] || !(image_matrix_cluster[i + 1][j - 1][k] - cluster)) && (!image_matrix_phase[i + 1][j][k + 1] || !(image_matrix_cluster[i + 1][j][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j][k - 1] || !(image_matrix_cluster[i + 1][j][k - 1] - cluster)) &&
						//	(!image_matrix_phase[i + 2][j + 1][k + 1] || !(image_matrix_cluster[i + 2][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i][j + 1][k + 1] || !(image_matrix_cluster[i][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i + 2][j - 1][k + 1] || !(image_matrix_cluster[i + 2][j - 1][k + 1] - cluster)) && (!image_matrix_phase[i + 2][j + 1][k - 1] || !(image_matrix_cluster[i + 2][j + 1][k - 1] - cluster)) && (!image_matrix_phase[i][j - 1][k + 1] || !(image_matrix_cluster[i][j - 1][k + 1] - cluster)) && (!image_matrix_phase[i + 2][j - 1][k - 1] || !(image_matrix_cluster[i + 2][j - 1][k - 1] - cluster)) && (!image_matrix_phase[i][j + 1][k - 1] || !(image_matrix_cluster[i][j + 1][k - 1] - cluster)) && (!image_matrix_phase[i][j - 1][k - 1] || !(image_matrix_cluster[i][j - 1][k - 1] - cluster)) && // 8 diagonal elements  
						//	(!image_matrix_phase[i + 2][j + 1][k] || !(image_matrix_cluster[i + 2][j + 1][k] - cluster)) && (!image_matrix_phase[i][j + 1][k] || !(image_matrix_cluster[i][j + 1][k] - cluster)) && (!image_matrix_phase[i + 2][j - 1][k] || !(image_matrix_cluster[i + 2][j - 1][k] - cluster)) && (!image_matrix_phase[i][j - 1][k] || !(image_matrix_cluster[i][j - 1][k] - cluster)) && (!image_matrix_phase[i + 2][j][k + 1] || !(image_matrix_cluster[i + 2][j][k + 1] - cluster)) && (!image_matrix_phase[i + 2][j][k - 1] || !(image_matrix_cluster[i + 2][j][k - 1] - cluster)) && (!image_matrix_phase[i][j][k + 1] || !(image_matrix_cluster[i][j][k + 1] - cluster)) && (!image_matrix_phase[i][j][k - 1] || !(image_matrix_cluster[i][j][k - 1] - cluster)) && (!image_matrix_phase[i + 1][j + 1][k + 1] || !(image_matrix_cluster[i + 1][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i + 2][j + 1][k + 1] || !(image_matrix_cluster[i + 2][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j + 1][k - 1] || !(image_matrix_cluster[i + 1][j + 1][k - 1] - cluster)) && (!image_matrix_phase[i + 1][j - 1][k - 1] || !(image_matrix_cluster[i + 1][j - 1][k - 1] - cluster)) ) // 12 remaining elements  
						//{ image_matrix_phase[i + 1][j][k] = phase; image_matrix_cluster[i + 1][j][k] = cluster; is_grown_voxel[i + 1][j][k] = 1; phase_volume[phase - 1]++; }
						//if ((input_struct->p[1] - dist(mt)) > 0 && !image_matrix_phase[i - 1][j][k] && (!image_matrix_phase[i - 2][j][k] || !(image_matrix_cluster[i - 2][j][k] - cluster)) && (!image_matrix_phase[i - 1][j + 1][k] || !(image_matrix_cluster[i - 1][j + 1][k] - cluster)) && (!image_matrix_phase[i - 1][j - 1][k] || !(image_matrix_cluster[i - 1][j - 1][k] - cluster)) && (!image_matrix_phase[i - 1][j][k + 1] || !(image_matrix_cluster[i - 1][j][k + 1] - cluster)) && (!image_matrix_phase[i - 1][j][k - 1] || !(image_matrix_cluster[i - 1][j][k - 1] - cluster)) &&
						//	(!image_matrix_phase[i][j + 1][k + 1] || !(image_matrix_cluster[i][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i - 2][j + 1][k + 1] || !(image_matrix_cluster[i - 2][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i][j - 1][k + 1] || !(image_matrix_cluster[i][j - 1][k + 1] - cluster)) && (!image_matrix_phase[i][j + 1][k - 1] || !(image_matrix_cluster[i][j + 1][k - 1] - cluster)) && (!image_matrix_phase[i - 2][j - 1][k + 1] || !(image_matrix_cluster[i - 2][j - 1][k + 1] - cluster)) && (!image_matrix_phase[i][j - 1][k - 1] || !(image_matrix_cluster[i][j - 1][k - 1] - cluster)) && (!image_matrix_phase[i - 2][j + 1][k - 1] || !(image_matrix_cluster[i - 2][j + 1][k - 1] - cluster)) && (!image_matrix_phase[i - 2][j - 1][k - 1] || !(image_matrix_cluster[i - 2][j - 1][k - 1] - cluster)) && // 8 diagonal elements  
						//	(!image_matrix_phase[i][j + 1][k] || !(image_matrix_cluster[i][j + 1][k] - cluster)) && (!image_matrix_phase[i - 2][j + 1][k] || !(image_matrix_cluster[i - 2][j + 1][k] - cluster)) && (!image_matrix_phase[i][j - 1][k] || !(image_matrix_cluster[i][j - 1][k] - cluster)) && (!image_matrix_phase[i - 2][j - 1][k] || !(image_matrix_cluster[i - 2][j - 1][k] - cluster)) && (!image_matrix_phase[i][j][k + 1] || !(image_matrix_cluster[i][j][k + 1] - cluster)) && (!image_matrix_phase[i][j][k - 1] || !(image_matrix_cluster[i][j][k - 1] - cluster)) && (!image_matrix_phase[i - 2][j][k + 1] || !(image_matrix_cluster[i - 2][j][k + 1] - cluster)) && (!image_matrix_phase[i - 2][j][k - 1] || !(image_matrix_cluster[i - 2][j][k - 1] - cluster)) && (!image_matrix_phase[i - 1][j + 1][k + 1] || !(image_matrix_cluster[i - 1][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i][j + 1][k + 1] || !(image_matrix_cluster[i][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i - 1][j + 1][k - 1] || !(image_matrix_cluster[i - 1][j + 1][k - 1] - cluster)) && (!image_matrix_phase[i - 1][j - 1][k - 1] || !(image_matrix_cluster[i - 1][j - 1][k - 1] - cluster))) // 12 remaining elements  
						//{ image_matrix_phase[i - 1][j][k] = phase; image_matrix_cluster[i - 1][j][k] = cluster; is_grown_voxel[i - 1][j][k] = 1; phase_volume[phase - 1]++; }
						//if ((input_struct->p[2] - dist(mt)) > 0 && !image_matrix_phase[i][j + 1][k] && (!image_matrix_phase[i][j + 2][k] || !(image_matrix_cluster[i][j + 2][k] - cluster)) && (!image_matrix_phase[i + 1][j + 1][k] || !(image_matrix_cluster[i + 1][j + 1][k] - cluster)) && (!image_matrix_phase[i - 1][j + 1][k] || !(image_matrix_cluster[i - 1][j + 1][k] - cluster)) && (!image_matrix_phase[i][j + 1][k + 1] || !(image_matrix_cluster[i][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i][j + 1][k - 1] || !(image_matrix_cluster[i][j + 1][k - 1] - cluster)) &&
						//	(!image_matrix_phase[i + 1][j + 2][k + 1] || !(image_matrix_cluster[i + 1][j + 2][k + 1] - cluster)) && (!image_matrix_phase[i - 1][j + 2][k + 1] || !(image_matrix_cluster[i - 1][j + 2][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j][k + 1] || !(image_matrix_cluster[i + 1][j][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j + 2][k - 1] || !(image_matrix_cluster[i + 1][j + 2][k - 1] - cluster)) && (!image_matrix_phase[i - 1][j][k + 1] || !(image_matrix_cluster[i - 1][j][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j][k - 1] || !(image_matrix_cluster[i + 1][j][k - 1] - cluster)) && (!image_matrix_phase[i - 1][j + 2][k - 1] || !(image_matrix_cluster[i - 1][j + 2][k - 1] - cluster)) && (!image_matrix_phase[i - 1][j][k - 1] || !(image_matrix_cluster[i - 1][j][k - 1] - cluster)) && // 8 diagonal elements  
						//	(!image_matrix_phase[i + 1][j + 2][k] || !(image_matrix_cluster[i + 1][j + 2][k] - cluster)) && (!image_matrix_phase[i - 1][j + 2][k] || !(image_matrix_cluster[i - 1][j + 2][k] - cluster)) && (!image_matrix_phase[i + 1][j][k] || !(image_matrix_cluster[i + 1][j][k] - cluster)) && (!image_matrix_phase[i - 1][j][k] || !(image_matrix_cluster[i - 1][j][k] - cluster)) && (!image_matrix_phase[i + 1][j + 1][k + 1] || !(image_matrix_cluster[i + 1][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j + 1][k - 1] || !(image_matrix_cluster[i + 1][j + 1][k - 1] - cluster)) && (!image_matrix_phase[i - 1][j + 1][k + 1] || !(image_matrix_cluster[i - 1][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i - 1][j + 1][k - 1] || !(image_matrix_cluster[i - 1][j + 1][k - 1] - cluster)) && (!image_matrix_phase[i][j + 2][k + 1] || !(image_matrix_cluster[i][j + 2][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j + 2][k + 1] || !(image_matrix_cluster[i + 1][j + 2][k + 1] - cluster)) && (!image_matrix_phase[i][j + 2][k - 1] || !(image_matrix_cluster[i][j + 2][k - 1] - cluster)) && (!image_matrix_phase[i][j][k - 1] || !(image_matrix_cluster[i][j][k - 1] - cluster))) // 12 remaining elements  
						//{ image_matrix_phase[i][j + 1][k] = phase; image_matrix_cluster[i][j + 1][k] = cluster; is_grown_voxel[i][j + 1][k] = 1; phase_volume[phase - 1]++; }
						//if ((input_struct->p[3] - dist(mt)) > 0 && !image_matrix_phase[i][j - 1][k] && (!image_matrix_phase[i][j - 2][k] || !(image_matrix_cluster[i][j - 2][k] - cluster)) && (!image_matrix_phase[i + 1][j - 1][k] || !(image_matrix_cluster[i + 1][j - 1][k] - cluster)) && (!image_matrix_phase[i - 1][j - 1][k] || !(image_matrix_cluster[i - 1][j - 1][k] - cluster)) && (!image_matrix_phase[i][j - 1][k + 1] || !(image_matrix_cluster[i][j - 1][k + 1] - cluster)) && (!image_matrix_phase[i][j - 1][k - 1] || !(image_matrix_cluster[i][j - 1][k - 1] - cluster)) &&
						//	(!image_matrix_phase[i + 1][j][k + 1] || !(image_matrix_cluster[i + 1][j][k + 1] - cluster)) && (!image_matrix_phase[i - 1][j][k + 1] || !(image_matrix_cluster[i - 1][j][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j][k + 1] || !(image_matrix_cluster[i + 1][j - 2][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j][k - 1] || !(image_matrix_cluster[i + 1][j][k - 1] - cluster)) && (!image_matrix_phase[i - 1][j - 2][k + 1] || !(image_matrix_cluster[i - 1][j - 2][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j - 2][k - 1] || !(image_matrix_cluster[i + 1][j - 2][k - 1] - cluster)) && (!image_matrix_phase[i - 1][j][k - 1] || !(image_matrix_cluster[i - 1][j][k - 1] - cluster)) && (!image_matrix_phase[i - 1][j - 2][k - 1] || !(image_matrix_cluster[i - 1][j - 2][k - 1] - cluster)) && // 8 diagonal elements  
						//	(!image_matrix_phase[i + 1][j][k] || !(image_matrix_cluster[i + 1][j][k] - cluster)) && (!image_matrix_phase[i - 1][j - 1 + 1][k] || !(image_matrix_cluster[i - 1][j][k] - cluster)) && (!image_matrix_phase[i + 1][j - 2][k] || !(image_matrix_cluster[i + 1][j - 2][k] - cluster)) && (!image_matrix_phase[i - 1][j - 2][k] || !(image_matrix_cluster[i - 1][j - 2][k] - cluster)) && (!image_matrix_phase[i + 1][j - 1][k + 1] || !(image_matrix_cluster[i + 1][j - 1][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j - 1][k - 1] || !(image_matrix_cluster[i + 1][j - 1][k - 1] - cluster)) && (!image_matrix_phase[i - 1][j - 1][k + 1] || !(image_matrix_cluster[i - 1][j - 1][k + 1] - cluster)) && (!image_matrix_phase[i - 1][j - 1][k - 1] || !(image_matrix_cluster[i - 1][j - 1][k - 1] - cluster)) && (!image_matrix_phase[i][j][k + 1] || !(image_matrix_cluster[i][j][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j][k + 1] || !(image_matrix_cluster[i + 1][j][k + 1] - cluster)) && (!image_matrix_phase[i][j][k - 1] || !(image_matrix_cluster[i][j][k - 1] - cluster)) && (!image_matrix_phase[i][j - 2][k - 1] || !(image_matrix_cluster[i][j - 2][k - 1] - cluster))) // 12 remaining elements  
						//{ image_matrix_phase[i][j - 1][k] = phase; image_matrix_cluster[i][j - 1][k] = cluster; is_grown_voxel[i][j - 1][k] = 1; phase_volume[phase - 1]++; }
						//if ((input_struct->p[4] - dist(mt)) > 0 && !image_matrix_phase[i][j][k + 1] && (!image_matrix_phase[i][j][k + 2] || !(image_matrix_cluster[i][j][k + 2] - cluster)) && (!image_matrix_phase[i + 1][j][k + 1] || !(image_matrix_cluster[i + 1][j][k + 1] - cluster)) && (!image_matrix_phase[i - 1][j][k + 1] || !(image_matrix_cluster[i - 1][j][k + 1] - cluster)) && (!image_matrix_phase[i][j + 1][k + 1] || !(image_matrix_cluster[i][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i][j - 1][k + 1] || !(image_matrix_cluster[i][j - 1][k + 1] - cluster)) &&
						//	(!image_matrix_phase[i + 1][j + 1][k + 2] || !(image_matrix_cluster[i + 1][j + 1][k + 2] - cluster)) && (!image_matrix_phase[i - 1][j + 1][k + 2] || !(image_matrix_cluster[i - 1][j + 1][k + 2] - cluster)) && (!image_matrix_phase[i + 1][j - 1][k + 2] || !(image_matrix_cluster[i + 1][j - 1][k + 2] - cluster)) && (!image_matrix_phase[i + 1][j + 1][k] || !(image_matrix_cluster[i + 1][j + 1][k] - cluster)) && (!image_matrix_phase[i - 1][j - 1][k + 2] || !(image_matrix_cluster[i - 1][j - 1][k + 2] - cluster)) && (!image_matrix_phase[i + 1][j - 1][k] || !(image_matrix_cluster[i + 1][j - 1][k] - cluster)) && (!image_matrix_phase[i - 1][j + 1][k] || !(image_matrix_cluster[i - 1][j + 1][k] - cluster)) && (!image_matrix_phase[i - 1][j - 1][k] || !(image_matrix_cluster[i + 1 - 2][j - 1][k] - cluster)) && // 8 diagonal elements  
						//	(!image_matrix_phase[i + 1][j + 1][k + 1] || !(image_matrix_cluster[i + 1][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i - 1][j + 1][k + 1] || !(image_matrix_cluster[i - 1][j + 1][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j - 1][k + 1] || !(image_matrix_cluster[i + 1][j - 1][k + 1] - cluster)) && (!image_matrix_phase[i - 1][j - 1][k + 1] || !(image_matrix_cluster[i - 1][j - 1][k + 1] - cluster)) && (!image_matrix_phase[i + 1][j][k + 2] || !(image_matrix_cluster[i + 1][j][k + 2] - cluster)) && (!image_matrix_phase[i + 1][j][k] || !(image_matrix_cluster[i + 1][j][k] - cluster)) && (!image_matrix_phase[i - 1][j][k + 2] || !(image_matrix_cluster[i - 1][j][k + 2] - cluster)) && (!image_matrix_phase[i - 1][j][k] || !(image_matrix_cluster[i - 1][j][k] - cluster)) && (!image_matrix_phase[i][j + 1][k + 2] || !(image_matrix_cluster[i][j + 1][k + 2] - cluster)) && (!image_matrix_phase[i + 1][j + 1][k + 2] || !(image_matrix_cluster[i + 1][j + 1][k + 2] - cluster)) && (!image_matrix_phase[i][j + 1][k] || !(image_matrix_cluster[i][j + 1][k] - cluster)) && (!image_matrix_phase[i][j - 1][k] || !(image_matrix_cluster[i][j - 1][k] - cluster))) // 12 remaining elements  
						//{ image_matrix_phase[i][j][k + 1] = phase; image_matrix_cluster[i][j][k + 1] = cluster; is_grown_voxel[i][j][k + 1] = 1; phase_volume[phase - 1]++; }
						//if ((input_struct->p[5] - dist(mt)) > 0 && !image_matrix_phase[i][j][k - 1] && (!image_matrix_phase[i][j][k - 2] || !(image_matrix_cluster[i][j][k - 2] - cluster)) && (!image_matrix_phase[i + 1][j][k - 1] || !(image_matrix_cluster[i + 1][j][k - 1] - cluster)) && (!image_matrix_phase[i - 1][j][k - 1] || !(image_matrix_cluster[i - 1][j][k - 1] - cluster)) && (!image_matrix_phase[i][j + 1][k - 1] || !(image_matrix_cluster[i][j + 1][k - 1] - cluster)) && (!image_matrix_phase[i][j - 1][k - 1] || !(image_matrix_cluster[i][j - 1][k - 1] - cluster)) &&
						//	(!image_matrix_phase[i + 1][j + 1][k] || !(image_matrix_cluster[i + 1][j + 1][k] - cluster)) && (!image_matrix_phase[i - 1][j + 1][k] || !(image_matrix_cluster[i - 1][j + 1][k] - cluster)) && (!image_matrix_phase[i + 1][j - 1][k] || !(image_matrix_cluster[i + 1][j - 1][k] - cluster)) && (!image_matrix_phase[i + 1][j + 1][k - 2] || !(image_matrix_cluster[i + 1][j + 1][k - 2] - cluster)) && (!image_matrix_phase[i - 1][j - 1][k] || !(image_matrix_cluster[i - 1][j - 1][k] - cluster)) && (!image_matrix_phase[i + 1][j - 1][k - 2] || !(image_matrix_cluster[i + 1][j - 1][k - 2] - cluster)) && (!image_matrix_phase[i - 1][j + 1][k - 2] || !(image_matrix_cluster[i - 1][j + 1][k - 2] - cluster)) && (!image_matrix_phase[i - 1][j - 1][k - 2] || !(image_matrix_cluster[i + 1 - 2][j - 1][k - 2] - cluster)) && // 8 diagonal elements  
						//	(!image_matrix_phase[i + 1][j + 1][k - 1] || !(image_matrix_cluster[i + 1][j + 1][k - 1] - cluster)) && (!image_matrix_phase[i - 1][j + 1][k - 1] || !(image_matrix_cluster[i - 1][j + 1][k - 1] - cluster)) && (!image_matrix_phase[i + 1][j - 1][k - 1] || !(image_matrix_cluster[i + 1][j - 1][k - 1] - cluster)) && (!image_matrix_phase[i - 1][j - 1][k - 1] || !(image_matrix_cluster[i - 1][j - 1][k - 1] - cluster)) && (!image_matrix_phase[i + 1][j][k] || !(image_matrix_cluster[i + 1][j][k] - cluster)) && (!image_matrix_phase[i + 1][j][k - 2] || !(image_matrix_cluster[i + 1][j][k - 2] - cluster)) && (!image_matrix_phase[i - 1][j][k] || !(image_matrix_cluster[i - 1][j][k] - cluster)) && (!image_matrix_phase[i - 1][j][k - 2] || !(image_matrix_cluster[i - 1][j][k - 2] - cluster)) && (!image_matrix_phase[i][j + 1][k] || !(image_matrix_cluster[i][j + 1][k] - cluster)) && (!image_matrix_phase[i + 1][j + 1][k] || !(image_matrix_cluster[i + 1][j + 1][k] - cluster)) && (!image_matrix_phase[i][j + 1][k - 2] || !(image_matrix_cluster[i][j + 1][k - 2] - cluster)) && (!image_matrix_phase[i][j - 1][k - 2] || !(image_matrix_cluster[i][j - 1][k - 2] - cluster))) // 12 remaining elements  
						//{ image_matrix_phase[i][j][k - 1] = phase; image_matrix_cluster[i][j][k - 1] = cluster; is_grown_voxel[i][j][k - 1] = 1; phase_volume[phase - 1]++; }
						if (image_matrix_phase[i + 1][j][k] == 0 && (input_struct->p[0 + 6 * (phase - 1)] * exp(-input_struct->decay_p[phase - 1] * (float)(iteration - seed_generation_start[cluster - 1]) * (float)(iteration - seed_generation_start[cluster - 1])) - dist(mt)) > 0) {
							gotoMainLoop = false;
							for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
								r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
								for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
									r_y = r_x - abs(y);
									for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
										if (!(!image_matrix_phase[i + 1 + x][j + y][k + z] || !(image_matrix_cluster[i + 1 + x][j + y][k + z] - cluster)))
											gotoMainLoop = true;
									}
								}
							}
							if (!gotoMainLoop) { image_matrix_phase[i + 1][j][k] = phase; image_matrix_cluster[i + 1][j][k] = cluster; is_grown_voxel[i + 1][j][k] = 1; if (i + 1 < RESOLUTION_X + 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
						}
						if (image_matrix_phase[i - 1][j][k] == 0 && (input_struct->p[1 + 6 * (phase - 1)] * exp(-input_struct->decay_p[phase - 1] * (float)(iteration - seed_generation_start[cluster - 1]) * (float)(iteration - seed_generation_start[cluster - 1])) - dist(mt)) > 0) {
							gotoMainLoop = false;
							for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
								r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
								for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
									r_y = r_x - abs(y);
									for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
										if (!(!image_matrix_phase[i - 1 + x][j + y][k + z] || !(image_matrix_cluster[i - 1 + x][j + y][k + z] - cluster)))
											gotoMainLoop = true;
									}
								}
							}
							if (!gotoMainLoop) { image_matrix_phase[i - 1][j][k] = phase; image_matrix_cluster[i - 1][j][k] = cluster; is_grown_voxel[i - 1][j][k] = 1; if (i - 1 >= 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
						}
						if (image_matrix_phase[i][j + 1][k] == 0 && (input_struct->p[2 + 6 * (phase - 1)] * exp(-input_struct->decay_p[phase - 1] * (float)(iteration - seed_generation_start[cluster - 1]) * (float)(iteration - seed_generation_start[cluster - 1])) - dist(mt)) > 0) {
							gotoMainLoop = false;
							for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
								r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
								for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
									r_y = r_x - abs(y);
									for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
										if (!(!image_matrix_phase[i + x][j + 1 + y][k + z] || !(image_matrix_cluster[i + x][j + 1 + y][k + z] - cluster)))
											gotoMainLoop = true;
									}
								}
							}
							if (!gotoMainLoop) { image_matrix_phase[i][j + 1][k] = phase; image_matrix_cluster[i][j + 1][k] = cluster; is_grown_voxel[i][j + 1][k] = 1; if (j + 1 < RESOLUTION_Y + 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
						}
						if (image_matrix_phase[i][j - 1][k] == 0 && (input_struct->p[3 + 6 * (phase - 1)] * exp(-input_struct->decay_p[phase - 1] * (float)(iteration - seed_generation_start[cluster - 1]) * (float)(iteration - seed_generation_start[cluster - 1])) - dist(mt)) > 0) {
							gotoMainLoop = false;
							for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
								r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
								for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
									r_y = r_x - abs(y);
									for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
										if (!(!image_matrix_phase[i + x][j - 1 + y][k + z] || !(image_matrix_cluster[i + x][j - 1 + y][k + z] - cluster)))
											gotoMainLoop = true;
									}
								}
							}
							if (!gotoMainLoop) { image_matrix_phase[i][j - 1][k] = phase; image_matrix_cluster[i][j - 1][k] = cluster; is_grown_voxel[i][j - 1][k] = 1; if (j - 1 >= 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
						}
						if (image_matrix_phase[i][j][k + 1] == 0 && (input_struct->p[4 + 6 * (phase - 1)] * exp(-input_struct->decay_p[phase - 1] * (float)(iteration - seed_generation_start[cluster - 1]) * (float)(iteration - seed_generation_start[cluster - 1])) - dist(mt)) > 0) {
							gotoMainLoop = false;
							for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
								r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
								for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
									r_y = r_x - abs(y);
									for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
										if (!(!image_matrix_phase[i + x][j + y][k + 1 + z] || !(image_matrix_cluster[i + x][j + y][k + 1 + z] - cluster)))
											gotoMainLoop = true;
									}
								}
							}
							if (!gotoMainLoop) { image_matrix_phase[i][j][k + 1] = phase; image_matrix_cluster[i][j][k + 1] = cluster; is_grown_voxel[i][j][k + 1] = 1; if (k + 1 < RESOLUTION_Z + 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
						}
						if (image_matrix_phase[i][j][k - 1] == 0 && (input_struct->p[5 + 6 * (phase - 1)] * exp(-input_struct->decay_p[phase - 1] * (float)(iteration - seed_generation_start[cluster - 1]) * (float)(iteration - seed_generation_start[cluster - 1])) - dist(mt)) > 0) {
							gotoMainLoop = false;
							for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
								r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
								for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
									r_y = r_x - abs(y);
									for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
										if (!(!image_matrix_phase[i + x][j + y][k - 1 + z] || !(image_matrix_cluster[i + x][j + y][k - 1 + z] - cluster)))
											gotoMainLoop = true;
									}
								}
							}
							if (!gotoMainLoop) { image_matrix_phase[i][j][k - 1] = phase; image_matrix_cluster[i][j][k - 1] = cluster; is_grown_voxel[i][j][k - 1] = 1; if (k - 1 >= 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
						}
					}
					else if (phase != 0 && !is_completed_phase[phase - 1] && is_grown_voxel[i][j][k] == 0 && stagnant_iteration[phase - 1] == stagnant_iteration_threshold) {
						if (image_matrix_phase[i + 1][j][k] == 0) {
							gotoMainLoop = false;
							for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
								r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
								for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
									r_y = r_x - abs(y);
									for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
										if (!(!image_matrix_phase[i + 1 + x][j + y][k + z] || !(image_matrix_cluster[i + 1 + x][j + y][k + z] - cluster)))
											gotoMainLoop = true;
									}
								}
							}
							if (!gotoMainLoop) { image_matrix_phase[i + 1][j][k] = phase; image_matrix_cluster[i + 1][j][k] = cluster; is_grown_voxel[i + 1][j][k] = 1; if (i + 1 < RESOLUTION_X + 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
						}
						if (image_matrix_phase[i - 1][j][k] == 0) {
							gotoMainLoop = false;
							for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
								r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
								for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
									r_y = r_x - abs(y);
									for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
										if (!(!image_matrix_phase[i - 1 + x][j + y][k + z] || !(image_matrix_cluster[i - 1 + x][j + y][k + z] - cluster)))
											gotoMainLoop = true;
									}
								}
							}
							if (!gotoMainLoop) { image_matrix_phase[i - 1][j][k] = phase; image_matrix_cluster[i - 1][j][k] = cluster; is_grown_voxel[i - 1][j][k] = 1; if (i - 1 >= 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
						}
						if (image_matrix_phase[i][j + 1][k] == 0) {
							gotoMainLoop = false;
							for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
								r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
								for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
									r_y = r_x - abs(y);
									for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
										if (!(!image_matrix_phase[i + x][j + 1 + y][k + z] || !(image_matrix_cluster[i + x][j + 1 + y][k + z] - cluster)))
											gotoMainLoop = true;
									}
								}
							}
							if (!gotoMainLoop) { image_matrix_phase[i][j + 1][k] = phase; image_matrix_cluster[i][j + 1][k] = cluster; is_grown_voxel[i][j + 1][k] = 1; if (j + 1 < RESOLUTION_Y + 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
						}
						if (image_matrix_phase[i][j - 1][k] == 0) {
							gotoMainLoop = false;
							for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
								r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
								for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
									r_y = r_x - abs(y);
									for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
										if (!(!image_matrix_phase[i + x][j - 1 + y][k + z] || !(image_matrix_cluster[i + x][j - 1 + y][k + z] - cluster)))
											gotoMainLoop = true;
									}
								}
							}
							if (!gotoMainLoop) { image_matrix_phase[i][j - 1][k] = phase; image_matrix_cluster[i][j - 1][k] = cluster; is_grown_voxel[i][j - 1][k] = 1; if (j - 1 >= 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
						}
						if (image_matrix_phase[i][j][k + 1] == 0) {
							gotoMainLoop = false;
							for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
								r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
								for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
									r_y = r_x - abs(y);
									for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
										if (!(!image_matrix_phase[i + x][j + y][k + 1 + z] || !(image_matrix_cluster[i + x][j + y][k + 1 + z] - cluster)))
											gotoMainLoop = true;
									}
								}
							}
							if (!gotoMainLoop) { image_matrix_phase[i][j][k + 1] = phase; image_matrix_cluster[i][j][k + 1] = cluster; is_grown_voxel[i][j][k + 1] = 1; if (k + 1 < RESOLUTION_Z + 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
						}
						if (image_matrix_phase[i][j][k - 1] == 0) {
							gotoMainLoop = false;
							for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
								r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
								for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
									r_y = r_x - abs(y);
									for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
										if (!(!image_matrix_phase[i + x][j + y][k - 1 + z] || !(image_matrix_cluster[i + x][j + y][k - 1 + z] - cluster)))
											gotoMainLoop = true;
									}
								}
							}
							if (!gotoMainLoop) { image_matrix_phase[i][j][k - 1] = phase; image_matrix_cluster[i][j][k - 1] = cluster; is_grown_voxel[i][j][k - 1] = 1; if (k - 1 >= 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
						}
					}
				}
			}
		}
		// Intializing growth mask matrix
		for (i = 0; i < RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD; i++) {
			for (j = 0; j < RESOLUTION_Y + 2 + 2 * MAX_NEIGHBORHOOD; j++) {
				for (k = 0; k < RESOLUTION_Z + 2 + 2 * MAX_NEIGHBORHOOD; k++) {
					is_grown_voxel[i][j][k] = 0;
				}
			}
		}
		++iteration; // Incrementing the iteration number of the microstructure evolution
	}
	seed_number--; // The last increment in "seed_number" happens after the last seed is added.  
	fprintf(log_file, "%06d- Microstructure evolution finished after %4d iterations with %5d seeds added in the process.\n", *microstructure_id, iteration, seed_number);
	fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution


	
	/* Connecting the foreground phase(s) for a cellular material with completely connected material phases if desired and creating the images if the microstrcutre conform to some conditions */
	// Resetting the limits/thresholds of volume fractions to the original ones, and zeroing the counter of stagnant iterations for all phases in case they should be regrown
	for (i = 0; i < number_of_phases - 1; i++) {
		phase_volume_threshold[i] = (int)(input_struct->volume_fraction[i] * loop_threshold);
		stagnant_iteration[i] = 0;
	}

	bool clustered = false;
	//std::vector<std::vector<int>> connection_array(seed_number, std::vector<int>(seed_number, 0));
	std::vector<std::vector<int>> connection_array;
	std::vector<bool> visited;
	bool connection_array_expanded;
	bool found_two_clusters;
	int density;
	int density_threshold;
	int radius;
	int radius2;
	int max;
	int max_index;
	bool condition_satisfied = true;
	gotoMainLoop = false;
	for (phase = 1; !gotoMainLoop && phase < number_of_phases; phase++) {
		desired_microstructure_generated = false;
		// Check if the volume fraction is within an acceptable range for clustering 
		if ((double)phase_volume[phase - 1] / (double)loop_threshold > phase_volume_max[phase - 1]) { // || phase_volume[phase - 1] > phase_volume_threshold[phase - 1] + (int)(loop_threshold * 0.05)
			fprintf(log_file, "%06d- Phase %2d: Realization VF = %.3f > Max Volume Fraction(VF) = %.3f -> The realization VF is too high, so no outputs will be generated due to failure.\n", *microstructure_id, phase, ((double)phase_volume[phase - 1] / (double)loop_threshold), phase_volume_max[phase - 1]);
			fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
			break;
		}
		if (phase_volume[phase - 1] < phase_volume_threshold[phase - 1] - (int)(loop_threshold * 0.30)) { // && input_struct->neighborhood_radius[phase - 1] < 4
			fprintf(log_file, "%06d- Phase %2d: Realization VF = %.3f < Volume Fraction(VF) = %.3f - 0.30 -> The realization VF is too low, so no outputs will be generated due to failure.\n", *microstructure_id, phase, ((double)phase_volume[phase - 1] / (double)loop_threshold), input_struct->volume_fraction[phase - 1]);
			fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
			break;
		}

		connection_array.clear();
		visited.clear();
		for (i = 0; i < seed_number; i++) {
			connection_array.push_back({});
			visited.push_back(false);
		}

		// If clustering algorithm should not be run for the current phase
		if (!input_struct->neighborhood_radius[phase - 1] || !input_struct->clustered[phase - 1]) {
			// Checking whether the current volume fraction falls into an acceptable range around the target volume fraction
			if ((double)phase_volume[phase - 1] / (double)loop_threshold < phase_volume_min[phase - 1] || (double)phase_volume[phase - 1] / (double)loop_threshold > phase_volume_max[phase - 1]) {
				fprintf(log_file, "%06d- Phase %2d: Realization VF = %.3f is out of bounds, so no outputs will be generated due to failure.\n", *microstructure_id, phase, ((double)phase_volume[phase - 1] / (double)loop_threshold));
				fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
				condition_satisfied = false;
				desired_microstructure_generated = false;
				break;
			}
			else {
				if (phase_volume[phase - 1] < phase_volume_threshold[phase - 1] - (int)(loop_threshold * 0.05) || phase_volume[phase - 1] > phase_volume_threshold[phase - 1] + (int)(loop_threshold * 0.05)) {
					fprintf(log_file, "%06d- Phase %2d: |Realization VF = %.3f - Volume Fraction(VF) = %.3f| > 0.05 -> The difference is high; however, the images may be generated in directory with '_' prefix.\n", *microstructure_id, phase, ((double)phase_volume[phase - 1] / (double)loop_threshold), input_struct->volume_fraction[phase - 1]);
					fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
					condition_satisfied = false;
					desired_microstructure_generated = false;
				}
				if (phase == number_of_phases - 1) { // If the current phase is the last phase of the microstructure
					if (condition_satisfied) {
						snprintf(bbuffer, MAX_CHAR_PATH, "%s/%06d", input_struct->results_directory.c_str(), *microstructure_id);
						desired_microstructure_generated = true;
						// Write image files of the microstructure in the results directory
						microstructure_image_generator(bbuffer, pixel_array, image_info, image_matrix_phase);
						end_time = omp_get_wtime(); // End time of the current microstructure realization (thread-safe/private)  
						fprintf(log_file, "%06d- Done: time = %02ld:%02ld:%02ld\n", *microstructure_id, (long int)(end_time-start_time_trial) / 3600, ((long int)(end_time - start_time_trial) % 3600) / 60, ((long int)(end_time - start_time_trial) % 3600) % 60);
						fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
					}
					else {
						snprintf(bbuffer, MAX_CHAR_PATH, "%s/_%06d", input_struct->results_directory.c_str(), *microstructure_id);
						desired_microstructure_generated = false;
						if (abs(phase_volume_copy[phase - 1] - phase_volume_threshold[phase - 1]) > abs(phase_volume[phase - 1] - phase_volume_threshold[phase - 1])) {
							// Write image files of the microstructure in the results directory
							fprintf(log_file, "%06d- Writing the images temporarily.\n", *microstructure_id);
							fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
							microstructure_image_generator(bbuffer, pixel_array, image_info, image_matrix_phase);
							phase_volume_copy = phase_volume;
						}
						end_time = omp_get_wtime(); // End time of the current microstructure realization (thread-safe/private)  
						fprintf(log_file, "%06d- End: time = %02ld:%02ld:%02ld\n", *microstructure_id, (long int)(end_time-start_time_trial) / 3600, ((long int)(end_time - start_time_trial) % 3600) / 60, ((long int)(end_time - start_time_trial) % 3600) % 60);
						fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
					}
				}
			}
			continue;
		}

		// If clustering algorithm should be run for the current phase
		if (input_struct->neighborhood_radius[phase - 1] < 4)
			radius = 1;
		else
			radius = input_struct->neighborhood_radius[phase - 1] / 2 + 1; // + 1
		density_threshold = 0;
		for (x = -radius; x <= radius; x++) {
			r_x = radius - abs(x);
			for (y = -r_x; y <= r_x; y++) {
				r_y = r_x - abs(y);
				for (z = -r_y; z <= r_y; z++) {
					density_threshold++;
				}
			}
		}
		density_threshold = density_threshold / (int)2 + 1;
		iteration = 0;
		end_time = omp_get_wtime(); // End time of the current microstructure realization (thread-safe/private)  
		fprintf(log_file, "%06d- time = %02ld:%02ld:%02ld; Clustering...\n", *microstructure_id, (long int)(end_time - start_time) / 3600, ((long int)(end_time - start_time) % 3600) / 60, ((long int)(end_time - start_time) % 3600) % 60);
		fprintf(log_file, "%06d- Phase %2d: iteration = %4d; radius = %2d; density_threshold = %3d\n", *microstructure_id, phase, iteration, radius, density_threshold);
		fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
		clustered = false;
		while (!clustered) { // && (phase_volume[phase - 1] < phase_volume_threshold[phase - 1])
			// Check if the current process has been taking too long to finish, or if it has taken more than the time-out limit (2100 sec)
			end_time = omp_get_wtime(); // End time of the current microstructure realization (thread-safe/private)
			if (end_time - start_time > 3600.0) {
				fprintf(log_file, "%06d- Timeout: time = %02ld:%02ld:%02ld\n", *microstructure_id, (long int)(end_time - start_time) / 3600, ((long int)(end_time - start_time) % 3600) / 60, ((long int)(end_time - start_time) % 3600) % 60);
				fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
				return 0;
			}
			if (radius > 2*input_struct->neighborhood_radius[phase - 1]) {
				fprintf(log_file, "%06d- Clustering not successful: too large neighborhood radius\n", *microstructure_id);
				fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
				gotoMainLoop = true;
				break;
			}
			// Search for the background voxels to change them into the foreground phase if the next conditons are met  
			//gotoMainLoop = false;
			for (i = 1 + MAX_NEIGHBORHOOD; i <= RESOLUTION_X + MAX_NEIGHBORHOOD; i++) {
				for (j = 1 + MAX_NEIGHBORHOOD; j <= RESOLUTION_Y + MAX_NEIGHBORHOOD; j++) {
					for (k = 1 + MAX_NEIGHBORHOOD; k <= RESOLUTION_Z + MAX_NEIGHBORHOOD; k++) {
						if (image_matrix_phase[i][j][k] == 0) { //  && !is_completed_phase[phase - 1] && is_grown_voxel[i][j][k] == 0
							//if (phase_volume[phase - 1] >= phase_volume_threshold[phase - 1]) { gotoMainLoop = true; break; }
							density = 0;
							for (x = -radius; x <= radius; x++) {
								r_x = radius - abs(x);
								for (y = -r_x; y <= r_x; y++) {
									r_y = r_x - abs(y);
									for (z = -r_y; z <= r_y; z++) {
										if ((i + x >= 0) && (j + y >= 0) && (k + z >= 0) && (i + x < RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD) && (j + y < RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD) && (k + z < RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD)) {
											if (image_matrix_phase[i + x][j + y][k + z] == phase) density++;
										}
										else
											density++;
									}
								}
							}
							connection_array_expanded = false;
							if (density >= density_threshold) {
								density = 0;
								for (x = -radius; x <= radius; x++) {
									r_x = radius - abs(x);
									for (y = -r_x; y <= r_x; y++) {
										r_y = r_x - abs(y);
										for (z = -r_y; z <= r_y; z++) {
											if ((i + x >= 0) && (j + y >= 0) && (k + z >= 0) && (i + x < RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD) && (j + y < RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD) && (k + z < RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD))
												if (image_matrix_phase[i + x][j + y][k + z] == phase) {
												density++;
												if (density == 1) {
													cluster = image_matrix_cluster[i + x][j + y][k + z];
												}
												else if (image_matrix_cluster[i + x][j + y][k + z] != cluster) {
													found_two_clusters = true;
													for (int l = 0; l < connection_array[cluster - 1].size(); l++) {
														if (!connection_array[cluster - 1].size() || connection_array[cluster - 1][l] == image_matrix_cluster[i + x][j + y][k + z] - 1) {
															found_two_clusters = false;
															break;
														}
													}
													if (found_two_clusters) {
														connection_array[cluster - 1].push_back(image_matrix_cluster[i + x][j + y][k + z] - 1);
														connection_array[image_matrix_cluster[i + x][j + y][k + z] - 1].push_back(cluster - 1);
														connection_array_expanded = true;
													}
												}
											}
										}
									}
								}
							}
							if (connection_array_expanded) {
								//radius2 = (int)((float)radius * ((dist(mt) * 3.0) + 1.0));
								if (input_struct->neighborhood_radius[phase - 1] < 4)
									//radius = 1;
									radius2 = (radius >= 1 + 1) ? radius : (int)((float)radius * ((dist(mt) * 2.2) + 1.0));
								else
									//radius = input_struct->neighborhood_radius[phase - 1] / 2 + 1;
									radius2 = (radius >= input_struct->neighborhood_radius[phase - 1] / 2) ? radius : (int)((float)radius * ((dist(mt) * 2.2) + 1.0));
								for (x = -radius2; x <= radius2; x++) {
									r_x = radius2 - abs(x);
									for (y = -r_x; y <= r_x; y++) {
										r_y = r_x - abs(y);
										for (z = -r_y; z <= r_y; z++) {
											if ((i + x >= 0) && (j + y >= 0) && (k + z >= 0) && (i + x < RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD) && (j + y < RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD) && (k + z < RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD) && image_matrix_phase[i + x][j + y][k + z] == 0) {
												image_matrix_phase[i + x][j + y][k + z] = phase;
												image_matrix_cluster[i + x][j + y][k + z] = cluster;
												if ((i + x < RESOLUTION_X + 1 + MAX_NEIGHBORHOOD) && (i + x >= 1 + MAX_NEIGHBORHOOD) && (j + y < RESOLUTION_X + 1 + MAX_NEIGHBORHOOD) && (j + y >= 1 + MAX_NEIGHBORHOOD) && (k + z < RESOLUTION_X + 1 + MAX_NEIGHBORHOOD) && (k + z >= 1 + MAX_NEIGHBORHOOD)) {
													seed_volume[cluster - 1]++;
													phase_volume[phase - 1]++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
			iteration++;

			// Find the percentage of connected seeds for the current phase
			max = 0;
			max_index = 0;
			for (i = 0; i < seed_number; i++) {
				if (connection_array[i].size()) {
					for (j = 0; j < seed_number; j++)
						visited[j] = false; // Initialize as if no seeds are visited 
					traverse(i, connection_array, visited); // Indeces of "visited" which are connected to seed i will get true values by this recursive function 
					sum = 0;
					for (j = 0; j < seed_number; j++)
						if (visited[j]) sum = sum + seed_volume[j];
					//if ((float)sum / (float)seed_number > 0.80) {
					//	clustered = true;
					//	break;
					//}
					if (sum > max) {
						max = sum;
						max_index = i;
					}
				}
			}
			if (connection_array[max_index].size() && ((float)max / (float)(phase_volume[phase - 1]) > 0.95))
				clustered = true;
			else if (density_threshold > 1)
				density_threshold--;
			else {
				density_threshold = 0;
				radius++;
				for (x = -radius; x <= radius; x++) {
					r_x = radius - abs(x);
					for (y = -r_x; y <= r_x; y++) {
						r_y = r_x - abs(y);
						for (z = -r_y; z <= r_y; z++) {
							density_threshold++;
						}
					}
				}
				density_threshold = density_threshold / (int)2;
				fprintf(log_file, "%06d- Phase %2d: iteration = %4d; radius = %2d; density_threshold = %3d\n", *microstructure_id, phase, iteration, radius, density_threshold);
				fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
			}
		}
		// Calculating the total number of voxels in a von Nuemann's "radius"-neighborhood  
		sum = 0; // The summation variable  
		for (x = -radius; x <= radius; x++) {
			r_x = radius - abs(x);
			for (y = -r_x; y <= r_x; y++) {
				r_y = r_x - abs(y);
				for (z = -r_y; z <= r_y; z++) {
					sum++;
				}
			}
		}
		fprintf(log_file, "%06d- Phase %2d with %5d seeds was clustered after %4d iterations with the final volume fraction of %.3f, the clustered percentage of %.3f, and the final density threshold of %.2f.\n", *microstructure_id, phase, seed_number, iteration, ((double)(phase_volume[phase - 1]) / (double)loop_threshold), ((double)max / (double)(phase_volume[phase - 1])), ((double)density_threshold / (double)sum));
		fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution

		// Checking whether the current volume fraction falls into an acceptable range around the target volume fraction
		if ((double)phase_volume[phase - 1] / (double)loop_threshold > phase_volume_max[phase - 1]) { // || phase_volume[phase - 1] > phase_volume_threshold[phase - 1] + (int)(loop_threshold * 0.05)
			fprintf(log_file, "%06d- Phase %2d: Realization VF = %.3f > Max Volume Fraction(VF) = %.3f -> The realization VF is too high, so no outputs will be generated due to failure.\n", *microstructure_id, phase, ((double)phase_volume[phase - 1] / (double)loop_threshold), phase_volume_max[phase - 1]);
			fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
			break;
		}
		if (phase_volume[phase - 1] < phase_volume_threshold[phase - 1] - (int)(loop_threshold * 0.05)) {
			fprintf(log_file, "%06d- Phase %2d: Realization VF = %.3f < Volume Fraction(VF) = %.3f - 0.05 -> begin growing the phase...\n", *microstructure_id, phase, ((double)phase_volume[phase - 1] / (double)loop_threshold), input_struct->volume_fraction[phase - 1]);
			fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
			while (1 > 0) {
				// Check if the current process has been taking too long to finish, or if it has taken more than the time-out limit (2100 sec)
				end_time = omp_get_wtime(); // End time of the current microstructure realization (thread-safe/private)
				if (end_time - start_time > 3600.0) {
					fprintf(log_file, "%06d- Timeout: time = %02ld:%02ld:%02ld\n", *microstructure_id, (long int)(end_time - start_time) / 3600, ((long int)(end_time - start_time) % 3600) / 60, ((long int)(end_time - start_time) % 3600) % 60);
					fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
					return 0;
				}
				// Stop evolution is the current phase has reached/surpassed its volume fraction limit during microstructure evolution
				if (phase_volume[phase - 1] > phase_volume_threshold[phase - 1] || stagnant_iteration[phase - 1] > stagnant_iteration_threshold || (stagnant_iteration[phase - 1] == stagnant_iteration_threshold && phase_volume[phase - 1] <= phase_volume_previous[phase - 1])) {
					fprintf(log_file, "%06d- Phase %2d: Volume Fraction(VF) = %.3f; Realization VF = %.3f; stagnant_iteration = %d\n", *microstructure_id, phase, input_struct->volume_fraction[phase - 1], ((double)phase_volume[phase - 1] / (double)loop_threshold), stagnant_iteration[phase - 1]);
					fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
					break;
				}
				else if (phase_volume[phase - 1] <= phase_volume_previous[phase - 1])
					++stagnant_iteration[phase - 1];
				else {
					stagnant_iteration[phase - 1] = 0;
					phase_volume_previous[phase - 1] = phase_volume[phase - 1];
				}

				// Grow the cells for one iteration of Cellular Automata
				for (i = 1 + MAX_NEIGHBORHOOD; i <= RESOLUTION_X + MAX_NEIGHBORHOOD; i++) {
					for (j = 1 + MAX_NEIGHBORHOOD; j <= RESOLUTION_Y + MAX_NEIGHBORHOOD; j++) {
						for (k = 1 + MAX_NEIGHBORHOOD; k <= RESOLUTION_Z + MAX_NEIGHBORHOOD; k++) {
							cluster = image_matrix_cluster[i][j][k];
							if (phase == image_matrix_phase[i][j][k] && is_grown_voxel[i][j][k] == 0 && stagnant_iteration[phase - 1] != stagnant_iteration_threshold) {
								if (image_matrix_phase[i + 1][j][k] == 0 && (input_struct->p[0 + 6 * (phase - 1)] * exp(-input_struct->decay_p[phase - 1] * (float)(iteration - seed_generation_start[cluster - 1]) * (float)(iteration - seed_generation_start[cluster - 1])) - dist(mt)) > 0) {
									gotoMainLoop = false;
									for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
										r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
										for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
											r_y = r_x - abs(y);
											for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
												if (!(!image_matrix_phase[i + 1 + x][j + y][k + z] || !(image_matrix_cluster[i + 1 + x][j + y][k + z] - cluster)))
													gotoMainLoop = true;
											}
										}
									}
									if (!gotoMainLoop) { image_matrix_phase[i + 1][j][k] = phase; image_matrix_cluster[i + 1][j][k] = cluster; is_grown_voxel[i + 1][j][k] = 1; if (i + 1 < RESOLUTION_X + 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
								}
								if (image_matrix_phase[i - 1][j][k] == 0 && (input_struct->p[1 + 6 * (phase - 1)] * exp(-input_struct->decay_p[phase - 1] * (float)(iteration - seed_generation_start[cluster - 1]) * (float)(iteration - seed_generation_start[cluster - 1])) - dist(mt)) > 0) {
									gotoMainLoop = false;
									for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
										r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
										for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
											r_y = r_x - abs(y);
											for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
												if (!(!image_matrix_phase[i - 1 + x][j + y][k + z] || !(image_matrix_cluster[i - 1 + x][j + y][k + z] - cluster)))
													gotoMainLoop = true;
											}
										}
									}
									if (!gotoMainLoop) { image_matrix_phase[i - 1][j][k] = phase; image_matrix_cluster[i - 1][j][k] = cluster; is_grown_voxel[i - 1][j][k] = 1; if (i - 1 >= 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
								}
								if (image_matrix_phase[i][j + 1][k] == 0 && (input_struct->p[2 + 6 * (phase - 1)] * exp(-input_struct->decay_p[phase - 1] * (float)(iteration - seed_generation_start[cluster - 1]) * (float)(iteration - seed_generation_start[cluster - 1])) - dist(mt)) > 0) {
									gotoMainLoop = false;
									for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
										r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
										for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
											r_y = r_x - abs(y);
											for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
												if (!(!image_matrix_phase[i + x][j + 1 + y][k + z] || !(image_matrix_cluster[i + x][j + 1 + y][k + z] - cluster)))
													gotoMainLoop = true;
											}
										}
									}
									if (!gotoMainLoop) { image_matrix_phase[i][j + 1][k] = phase; image_matrix_cluster[i][j + 1][k] = cluster; is_grown_voxel[i][j + 1][k] = 1; if (j + 1 < RESOLUTION_Y + 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
								}
								if (image_matrix_phase[i][j - 1][k] == 0 && (input_struct->p[3 + 6 * (phase - 1)] * exp(-input_struct->decay_p[phase - 1] * (float)(iteration - seed_generation_start[cluster - 1]) * (float)(iteration - seed_generation_start[cluster - 1])) - dist(mt)) > 0) {
									gotoMainLoop = false;
									for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
										r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
										for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
											r_y = r_x - abs(y);
											for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
												if (!(!image_matrix_phase[i + x][j - 1 + y][k + z] || !(image_matrix_cluster[i + x][j - 1 + y][k + z] - cluster)))
													gotoMainLoop = true;
											}
										}
									}
									if (!gotoMainLoop) { image_matrix_phase[i][j - 1][k] = phase; image_matrix_cluster[i][j - 1][k] = cluster; is_grown_voxel[i][j - 1][k] = 1; if (j - 1 >= 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
								}
								if (image_matrix_phase[i][j][k + 1] == 0 && (input_struct->p[4 + 6 * (phase - 1)] * exp(-input_struct->decay_p[phase - 1] * (float)(iteration - seed_generation_start[cluster - 1]) * (float)(iteration - seed_generation_start[cluster - 1])) - dist(mt)) > 0) {
									gotoMainLoop = false;
									for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
										r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
										for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
											r_y = r_x - abs(y);
											for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
												if (!(!image_matrix_phase[i + x][j + y][k + 1 + z] || !(image_matrix_cluster[i + x][j + y][k + 1 + z] - cluster)))
													gotoMainLoop = true;
											}
										}
									}
									if (!gotoMainLoop) { image_matrix_phase[i][j][k + 1] = phase; image_matrix_cluster[i][j][k + 1] = cluster; is_grown_voxel[i][j][k + 1] = 1; if (k + 1 < RESOLUTION_Z + 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
								}
								if (image_matrix_phase[i][j][k - 1] == 0 && (input_struct->p[5 + 6 * (phase - 1)] * exp(-input_struct->decay_p[phase - 1] * (float)(iteration - seed_generation_start[cluster - 1]) * (float)(iteration - seed_generation_start[cluster - 1])) - dist(mt)) > 0) {
									gotoMainLoop = false;
									for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
										r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
										for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
											r_y = r_x - abs(y);
											for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
												if (!(!image_matrix_phase[i + x][j + y][k - 1 + z] || !(image_matrix_cluster[i + x][j + y][k - 1 + z] - cluster)))
													gotoMainLoop = true;
											}
										}
									}
									if (!gotoMainLoop) { image_matrix_phase[i][j][k - 1] = phase; image_matrix_cluster[i][j][k - 1] = cluster; is_grown_voxel[i][j][k - 1] = 1; if (k - 1 >= 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
								}
							}
							else if (phase == image_matrix_phase[i][j][k] && is_grown_voxel[i][j][k] == 0 && stagnant_iteration[phase - 1] == stagnant_iteration_threshold) {
								if (image_matrix_phase[i + 1][j][k] == 0) {
									gotoMainLoop = false;
									for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
										r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
										for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
											r_y = r_x - abs(y);
											for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
												if (!(!image_matrix_phase[i + 1 + x][j + y][k + z] || !(image_matrix_cluster[i + 1 + x][j + y][k + z] - cluster)))
													gotoMainLoop = true;
											}
										}
									}
									if (!gotoMainLoop) { image_matrix_phase[i + 1][j][k] = phase; image_matrix_cluster[i + 1][j][k] = cluster; is_grown_voxel[i + 1][j][k] = 1; if (i + 1 < RESOLUTION_X + 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
								}
								if (image_matrix_phase[i - 1][j][k] == 0) {
									gotoMainLoop = false;
									for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
										r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
										for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
											r_y = r_x - abs(y);
											for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
												if (!(!image_matrix_phase[i - 1 + x][j + y][k + z] || !(image_matrix_cluster[i - 1 + x][j + y][k + z] - cluster)))
													gotoMainLoop = true;
											}
										}
									}
									if (!gotoMainLoop) { image_matrix_phase[i - 1][j][k] = phase; image_matrix_cluster[i - 1][j][k] = cluster; is_grown_voxel[i - 1][j][k] = 1; if (i - 1 >= 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
								}
								if (image_matrix_phase[i][j + 1][k] == 0) {
									gotoMainLoop = false;
									for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
										r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
										for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
											r_y = r_x - abs(y);
											for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
												if (!(!image_matrix_phase[i + x][j + 1 + y][k + z] || !(image_matrix_cluster[i + x][j + 1 + y][k + z] - cluster)))
													gotoMainLoop = true;
											}
										}
									}
									if (!gotoMainLoop) { image_matrix_phase[i][j + 1][k] = phase; image_matrix_cluster[i][j + 1][k] = cluster; is_grown_voxel[i][j + 1][k] = 1; if (j + 1 < RESOLUTION_Y + 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
								}
								if (image_matrix_phase[i][j - 1][k] == 0) {
									gotoMainLoop = false;
									for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
										r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
										for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
											r_y = r_x - abs(y);
											for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
												if (!(!image_matrix_phase[i + x][j - 1 + y][k + z] || !(image_matrix_cluster[i + x][j - 1 + y][k + z] - cluster)))
													gotoMainLoop = true;
											}
										}
									}
									if (!gotoMainLoop) { image_matrix_phase[i][j - 1][k] = phase; image_matrix_cluster[i][j - 1][k] = cluster; is_grown_voxel[i][j - 1][k] = 1; if (j - 1 >= 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
								}
								if (image_matrix_phase[i][j][k + 1] == 0) {
									gotoMainLoop = false;
									for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
										r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
										for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
											r_y = r_x - abs(y);
											for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
												if (!(!image_matrix_phase[i + x][j + y][k + 1 + z] || !(image_matrix_cluster[i + x][j + y][k + 1 + z] - cluster)))
													gotoMainLoop = true;
											}
										}
									}
									if (!gotoMainLoop) { image_matrix_phase[i][j][k + 1] = phase; image_matrix_cluster[i][j][k + 1] = cluster; is_grown_voxel[i][j][k + 1] = 1; if (k + 1 < RESOLUTION_Z + 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
								}
								if (image_matrix_phase[i][j][k - 1] == 0) {
									gotoMainLoop = false;
									for (x = -input_struct->neighborhood_radius[phase - 1]; x <= input_struct->neighborhood_radius[phase - 1] && !gotoMainLoop; x++) {
										r_x = input_struct->neighborhood_radius[phase - 1] - abs(x);
										for (y = -r_x; y <= r_x && !gotoMainLoop; y++) {
											r_y = r_x - abs(y);
											for (z = -r_y; z <= r_y && !gotoMainLoop; z++) {
												if (!(!image_matrix_phase[i + x][j + y][k - 1 + z] || !(image_matrix_cluster[i + x][j + y][k - 1 + z] - cluster)))
													gotoMainLoop = true;
											}
										}
									}
									if (!gotoMainLoop) { image_matrix_phase[i][j][k - 1] = phase; image_matrix_cluster[i][j][k - 1] = cluster; is_grown_voxel[i][j][k - 1] = 1; if (k - 1 >= 1 + MAX_NEIGHBORHOOD) { seed_volume[cluster - 1]++; phase_volume[phase - 1]++; } }
								}
							}
						}
					}
				}
				// Intializing growth mask matrix
				for (i = 0; i < RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD; i++) {
					for (j = 0; j < RESOLUTION_Y + 2 + 2 * MAX_NEIGHBORHOOD; j++) {
						for (k = 0; k < RESOLUTION_Z + 2 + 2 * MAX_NEIGHBORHOOD; k++) {
							is_grown_voxel[i][j][k] = 0;
						}
					}
				}
				++iteration; // Incrementing the iteration number of the microstructure evolution
				//printf("%f\n", phase_volume[0] / ((float)phase_volume_threshold[0]));
			}
			fprintf(log_file, "%06d- Microstructure evolution finished after %4d iterations with no more seeds added in the process.\n", *microstructure_id, iteration);
			fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
		}
		// Checking whether the current volume fraction falls into an acceptable range around the target volume fraction
		if ((double)phase_volume[phase - 1] / (double)loop_threshold < phase_volume_min[phase - 1] || (double)phase_volume[phase - 1] / (double)loop_threshold > phase_volume_max[phase - 1]) {
			fprintf(log_file, "%06d- Phase %2d: Realization VF = %.3f is out of bounds, so no outputs will be generated due to failure.\n", *microstructure_id, phase, ((double)phase_volume[phase - 1] / (double)loop_threshold));
			fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
			condition_satisfied = false;
			desired_microstructure_generated = false;
			break;
		}
		else {
			if (phase_volume[phase - 1] < phase_volume_threshold[phase - 1] - (int)(loop_threshold * 0.05) || phase_volume[phase - 1] > phase_volume_threshold[phase - 1] + (int)(loop_threshold * 0.05)) {
				fprintf(log_file, "%06d- Phase %2d: |Realization VF = %.3f - Volume Fraction(VF) = %.3f| > 0.05 -> The difference is high; however, the images may be generated in directory with '_' prefix.\n", *microstructure_id, phase, ((double)phase_volume[phase - 1] / (double)loop_threshold), input_struct->volume_fraction[phase - 1]);
				fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
				condition_satisfied = false;
				desired_microstructure_generated = false;
			}
			if (phase == number_of_phases - 1) { // If the current phase is the last phase of the microstructure
				if (condition_satisfied) {
					snprintf(bbuffer, MAX_CHAR_PATH, "%s/%06d", input_struct->results_directory.c_str(), *microstructure_id);
					desired_microstructure_generated = true;
					// Write image files of the microstructure in the results directory
					microstructure_image_generator(bbuffer, pixel_array, image_info, image_matrix_phase);
					end_time = omp_get_wtime(); // End time of the current microstructure realization (thread-safe/private)  
					fprintf(log_file, "%06d- Done: time = %02ld:%02ld:%02ld\n", *microstructure_id, (long int)(end_time-start_time_trial) / 3600, ((long int)(end_time - start_time_trial) % 3600) / 60, ((long int)(end_time - start_time_trial) % 3600) % 60);
					fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
				}
				else {
					snprintf(bbuffer, MAX_CHAR_PATH, "%s/_%06d", input_struct->results_directory.c_str(), *microstructure_id);
					desired_microstructure_generated = false;
					if (abs(phase_volume_copy[phase - 1] - phase_volume_threshold[phase - 1]) > abs(phase_volume[phase - 1] - phase_volume_threshold[phase - 1])) {
						// Write image files of the microstructure in the results directory
						fprintf(log_file, "%06d- Writing the images temporarily.\n", *microstructure_id);
						fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
						microstructure_image_generator(bbuffer, pixel_array, image_info, image_matrix_phase);
						phase_volume_copy = phase_volume;
					}
					end_time = omp_get_wtime(); // End time of the current microstructure realization (thread-safe/private)  
					fprintf(log_file, "%06d- End: time = %02ld:%02ld:%02ld\n", *microstructure_id, (long int)(end_time-start_time_trial) / 3600, ((long int)(end_time - start_time_trial) % 3600) / 60, ((long int)(end_time - start_time_trial) % 3600) % 60);
					fflush(log_file); // Flush out the file stream so that the user can see what new things has been written to the log file during program execution
				}
			}
		}
	}
	}

	delete[] bbuffer;
	fclose(log_file);
	return 1;
}



// Get and preprocess the input of the main function
void get_input(int argc, char* argv[], int* N) {
	if (argc != 2) { fprintf(stderr, "usage: %s <N>\n", argv[0]); fprintf(stderr,"    N should be positive.\n"); exit(1); }

	*N = strtol(argv[1], NULL, 10);
	if (*N <= 0) { fprintf(stderr, "usage: %s <N>\n", argv[0]); fprintf(stderr,"    N should be positive.\n"); exit(1); }
	// if (*N%comm_sz!=0) { usage(argv[0]); }
}



// Traverse the 2D connection array to find out what nodes are directly or indirectly connected to node v. If there is a connection, the associated index in "visited" array gets a true value.
void traverse(int v, const std::vector<std::vector<int>> &connection_array, std::vector<bool> &visited) {
	visited[v] = true; // Mark the current node as visited  
	for (auto it = connection_array[v].begin(); it != connection_array[v].end(); ++it)
		if (!visited[*it]) traverse(*it, connection_array, visited);
}



// Convert the 3D "image_matrix_phase" into the sliced images of the microstructure
void microstructure_image_generator(char* bbuffer, unsigned char (&pixel_array)[MAX_FILE_SIZE], Image_info* image_info, uint8_t (&image_matrix_phase)[RESOLUTION_X + 2 + 2 * MAX_NEIGHBORHOOD][RESOLUTION_Y + 2 + 2 * MAX_NEIGHBORHOOD][RESOLUTION_Z + 2 + 2 * MAX_NEIGHBORHOOD]) {
	/* Making images from the realized microstructure (from the evolved 3D matrix "image_matrix_phase")  */
	// Based on the target machine OS, an appropriate function for making a new directory will be chosen to save the microstructure images in a seperate directory.  
#ifdef __linux__
	mkdir(bbuffer, 777);
#else
	_mkdir(bbuffer);
#endif
	// Finding the total number of digits, width_field, for 0 padding in image file names  
	// int width_field = 1;
	// int order_of_magnitude = 10;
	// k = RESOLUTION_Z;
	// while (k != 0) {
	// 	k = k / order_of_magnitude;
	// 	order_of_magnitude *= 10;
	// 	++width_field;
	// }
	int i, j, k;
	char* bbuffer2 = new char[MAX_CHAR_PATH];
	for (k = 1 + MAX_NEIGHBORHOOD; k < RESOLUTION_Z + 1 + MAX_NEIGHBORHOOD; k++) {
		for (j = 1 + MAX_NEIGHBORHOOD; j < RESOLUTION_Y + 1 + MAX_NEIGHBORHOOD; j++) {
			for (i = 1 + MAX_NEIGHBORHOOD; i < RESOLUTION_X + 1 + MAX_NEIGHBORHOOD; i++) {
				if (image_matrix_phase[i][j][k])
					//pixel_array[(i - 1) + (image_info->width_in_bytes) * (j - 1)] = (image_matrix_cluster[i][j][k] % 14) + 1;
					pixel_array[(i - 1 - MAX_NEIGHBORHOOD) + (image_info->width_in_bytes) * (j - 1 - MAX_NEIGHBORHOOD)] = image_matrix_phase[i][j][k];
				else
					pixel_array[(i - 1 - MAX_NEIGHBORHOOD) + (image_info->width_in_bytes) * (j - 1 - MAX_NEIGHBORHOOD)] = 0;
				//pixel_array[(i - 1) + (image_info->width_in_bytes) * (j - 1)] = image_matrix_phase[i][j][k];
				// Following commented lines are needed when the image color format is 24 bit (3 bytes of RGB for each pixel)
				//pixel_array[3 * (i - 1) + width_in_bytes * (j - 1) + 1] = 255 * image_matrix_phase[i][j][k];
				//pixel_array[3 * (i - 1) + width_in_bytes * (j - 1) + 2] = 255 * image_matrix_phase[i][j][k];
			}
			// Adding the padding bytes in each row of pixel array as many as needed.  
			for (i = 0; i < (image_info->pad); i++) pixel_array[(1 * RESOLUTION_X + i) + (image_info->width_in_bytes) * (j - 1 - MAX_NEIGHBORHOOD)] = 0;
		}

		// Writing the image information in a binary format byte by byte.
		// Making an appropriate string for file name based on the number of slices, RESOLUTION_Z, and enough 0 padding in file name based on width_field.
		snprintf(bbuffer2, MAX_CHAR_PATH, "%s/slice_%03d.bmp", bbuffer, k-MAX_NEIGHBORHOOD);
		FILE* image_file = fopen(bbuffer2, "wb");
		fwrite(image_info->header, 1, 54, image_file);
		fwrite(image_info->color_table, 1, 4 * 8, image_file);
		fwrite(pixel_array, 1, image_info->image_size, image_file);
		fclose(image_file);
	}
	delete[] bbuffer2;
}
