// compile in Linux with gcc:
// g++ hello_world.cpp -lOpenCL

#include "CL/cl.h"                              //includes open CL library to enable open CL functionality
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_SIZE   10                          //prepare matrix size
#define MEM_SIZE    DATA_SIZE * sizeof(float)   //prepares memory needed to store a matrix of given size containing float values

/** kernel's source text as string **/
const char* KernelSource =
"#define DATA_SIZE 10												\n"
"__kernel void test(__global float *input, __global float *output)  \n"
"{																	\n"
"	size_t i = get_global_id(0);									\n"
"	output[i] = input[i] * input[i];								\n"
"}																	\n"
"\n";

/** Body of the main code **/
int main(void)
{
	cl_int				err;                      //stores information about success or failure of commands
	cl_platform_id* platforms = NULL;         //id of the platform
	char			    platform_name[1024];      //name of the platform
	cl_device_id	    device_id = NULL;         //id of the device
	cl_uint			    num_of_platforms = 0,     //number of platforms
		num_of_devices = 0;       //number of devices
	cl_context 			context;                  //context
	cl_kernel 			kernel;                   //kernel
	cl_command_queue	command_queue;            //queue storing commands
	cl_program 			program;                  //stores generated program
	cl_mem				input, output;            //stores necessary available memory
	float				data[DATA_SIZE] =         //prepare storage for data
	{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	size_t				global[1] = { DATA_SIZE };  //global storage for data size
	float				results[DATA_SIZE] = { 0 }; //empty storage with enough space for future matrix data

	/* 1) */

	// gets the number of available platforms, returning if it was successful
	err = clGetPlatformIDs(0, NULL, &num_of_platforms);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}

	// gets the id of the current platform
	platforms = (cl_platform_id*)malloc(num_of_platforms);
	err = clGetPlatformIDs(num_of_platforms, platforms, NULL);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}
	else
	{
		int nvidia_platform = 0;

		// For every platform
		for (unsigned int i = 0; i < num_of_platforms; i++)
		{
			// Attempt to get its information
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
			if (err != CL_SUCCESS)
			{
				printf("Could not get information about platform. Error: %d\n", err);
				return 0;
			}

			// check if using nvidia
			if (strstr(platform_name, "NVIDIA") != NULL)
			{
				nvidia_platform = i;
				break;
			}
		}

		// Get ID of current device, and maximum available numbers
		err = clGetDeviceIDs(platforms[nvidia_platform], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
		if (err != CL_SUCCESS)
		{
			printf("Could not get device in platform. Error: %d\n", err);
			return 0;
		}
	}

	// Open Context
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create context. Error: %d\n", err);
		return 0;
	}

	// Creates a queue (FIFO)
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create command queue. Error: %d\n", err);
		return 0;
	}

	// Generate online program
	program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create program. Error: %d\n", err);
		return 0;
	}

	// Compile and link the kernel source text
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error building program. Error: %d\n", err);
		return 0;
	}

	// Create entry point for kernel
	kernel = clCreateKernel(program, "test", &err);
	if (err != CL_SUCCESS)
	{
		printf("Error setting kernel. Error: %d\n", err);
		return 0;
	}


	/* 2) */

	// Create Buffer for input and output
	input = clCreateBuffer(context, CL_MEM_READ_ONLY, MEM_SIZE, NULL, &err);
	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MEM_SIZE, NULL, &err);

	// Write related data from 'data' into 'input' buffer
	clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, MEM_SIZE, data, 0, NULL, NULL);

	// Define the order of operations of kernel arguments
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);


	/* 3)  */

	// Puts kernel into command queue and splits up instructions
	clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);

	// Wait for queue to complete
	clFinish(command_queue);

	// Read and store results of output buffer into results
	clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, MEM_SIZE, results, 0, NULL, NULL);

	// Display results
	for (unsigned int i = 0; i < DATA_SIZE; i++)
		printf("%f\n", results[i]);


	/* 4) */
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return 0;
}
