#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <nvrtc.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "hello.h"
#include "helper_math.h"

#include "gltf.h"

template<typename T>
struct SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

using RaygenSbtRecord = SbtRecord<RaygenData>;
using MissSbtRecord = SbtRecord<MissData>;
using HitgroupSbtRecord = SbtRecord<HitgroupData>;

void logFunc(unsigned int level, const char* tag, const char* message, void* cbdata) {
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << std::endl;
}

int main(int argc, char* argv[]) {
	constexpr int width = 640;
	constexpr int height = 480;

	GLFWwindow* window;

	if (!glfwInit()) exit(EXIT_FAILURE);

	window = glfwCreateWindow(width, height, "Optix Test", nullptr, nullptr);
	if (!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(window);

	glewInit();

	OptixDeviceContext context{};
	{
		cudaFree(0);
		CUcontext cuCtx = 0;

		optixInit();

		OptixDeviceContextOptions options{};
		options.logCallbackFunction = &logFunc;
		options.logCallbackLevel = 4;
		
		optixDeviceContextCreate(cuCtx, &options, &context);
	}

	OptixTraversableHandle gasHandle;
	CUdeviceptr d_gas;
	{
		OptixAccelBuildOptions options{};
		options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
		options.operation = OPTIX_BUILD_OPERATION_BUILD;

		std::vector<float3> vertices;
		std::vector<uint3> indices;
		readGLB("cornellbox.glb", vertices, indices);

		CUdeviceptr d_vert{};
		cudaMalloc((void**)&d_vert, sizeof(float3) * vertices.size());
		cudaMemcpy((void*)d_vert, vertices.data(), sizeof(float3) * vertices.size(), cudaMemcpyHostToDevice);

		CUdeviceptr d_indi{};
		cudaMalloc((void**)&d_indi, sizeof(uint3) * indices.size());
		cudaMemcpy((void*)d_indi, indices.data(), sizeof(uint3) * indices.size(), cudaMemcpyHostToDevice);

		const uint32_t triangleFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
		OptixBuildInput triangleInput{};
		triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInput.triangleArray.numVertices = vertices.size();
		triangleInput.triangleArray.vertexBuffers = &d_vert;
		triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInput.triangleArray.numIndexTriplets = indices.size();
		triangleInput.triangleArray.indexBuffer = d_indi;
		triangleInput.triangleArray.flags = triangleFlags;
		triangleInput.triangleArray.numSbtRecords = 1;

		OptixAccelBufferSizes gasSizes;
		optixAccelComputeMemoryUsage(context, &options, &triangleInput, 1, &gasSizes);

		CUdeviceptr d_temp;
		cudaMalloc((void**)&d_temp, gasSizes.tempSizeInBytes);
		cudaMalloc((void**)&d_gas, gasSizes.outputSizeInBytes);

		optixAccelBuild(context, 0, &options, &triangleInput, 1, d_temp, gasSizes.tempSizeInBytes, d_gas, gasSizes.outputSizeInBytes, &gasHandle, nullptr, 0);

		cudaFree((void*)d_temp);
	}

	OptixModule optixModule{};
	OptixPipelineCompileOptions pipelineOptions{};
	{
		std::ifstream ifs("hello.cu");
		if (ifs.fail()) {
			std::cerr << "Cannot read cu file." << std::endl;
			exit(EXIT_FAILURE);
		}
		std::string srcContent{ std::istreambuf_iterator<char>{ifs}, std::istreambuf_iterator<char>{} };

		nvrtcProgram program;
		nvrtcCreateProgram(&program, srcContent.c_str(), nullptr, 0, nullptr, nullptr);


		std::vector<const char*> options = {
			"-I C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 7.7.0\\include",
			"-I C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\include",
			"-I hello.h",
			"-I helper_math.h",
			"--optix-ir",
		};

		nvrtcCompileProgram(program, options.size(), options.data());

		size_t logSize;
		nvrtcGetProgramLogSize(program, &logSize);
		std::vector<char> log(logSize);
		nvrtcGetProgramLog(program, log.data());
		std::cout << log.data() << std::endl;

		size_t optixIrSize;
		nvrtcGetOptiXIRSize(program, &optixIrSize);

		std::vector<char> optixIr(optixIrSize);
		nvrtcGetOptiXIR(program, optixIr.data());

		OptixModuleCompileOptions moduleOptions{};
		moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

		pipelineOptions.usesMotionBlur = false;
		pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipelineOptions.numPayloadValues = 3;
		pipelineOptions.numAttributeValues = 3;
		pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		pipelineOptions.pipelineLaunchParamsVariableName = "params";
		pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

		optixModuleCreate(context, &moduleOptions, &pipelineOptions, optixIr.data(), optixIrSize, nullptr, nullptr, &optixModule);
	}

	OptixProgramGroup raygenGroup;
	OptixProgramGroup missGroup;
	OptixProgramGroup hitgroupGroup;
	{
		OptixProgramGroupOptions options{};

		OptixProgramGroupDesc raygenDesc{};
		raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygenDesc.raygen.module = optixModule;
		raygenDesc.raygen.entryFunctionName = "__raygen__rg";
		optixProgramGroupCreate(context, &raygenDesc, 1, &options, nullptr, nullptr, &raygenGroup);

		OptixProgramGroupDesc missDesc{};
		missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		missDesc.miss.module = optixModule;
		missDesc.miss.entryFunctionName = "__miss__ms";
		optixProgramGroupCreate(context, &missDesc, 1, &options, nullptr, nullptr, &missGroup);

		OptixProgramGroupDesc hitgroupDesc{};
		hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroupDesc.hitgroup.moduleCH = optixModule;
		hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
		optixProgramGroupCreate(context, &hitgroupDesc, 1, &options, nullptr, nullptr, &hitgroupGroup);
	}

	OptixPipeline pipeline;
	{
		uint32_t maxDepth = 1;

		std::vector<OptixProgramGroup> programGroups = { raygenGroup, missGroup, hitgroupGroup };

		OptixPipelineLinkOptions linkOptions{};
		linkOptions.maxTraceDepth = maxDepth;
		optixPipelineCreate(context, &pipelineOptions, &linkOptions, programGroups.data(), programGroups.size(), nullptr, nullptr, &pipeline);

		OptixStackSizes stackSizes{};
		for (auto& group : programGroups) {
			optixUtilAccumulateStackSizes(group, &stackSizes, pipeline);
		}

		uint32_t stackSizeTraversal;
		uint32_t stackSizeState;
		uint32_t stackSizeContinuation;
		optixUtilComputeStackSizes(&stackSizes, maxDepth, 0, 0, &stackSizeTraversal, &stackSizeState, &stackSizeContinuation);
		optixPipelineSetStackSize(pipeline, stackSizeTraversal, stackSizeState, stackSizeContinuation, 1);
	}

	OptixShaderBindingTable sbt{};
	{
		CUdeviceptr raygenRecord;
		size_t raygenRecordSize = sizeof(RaygenSbtRecord);
		cudaMalloc((void**)&raygenRecord, raygenRecordSize);
		RaygenSbtRecord raygenData;
		optixSbtRecordPackHeader(raygenGroup, &raygenData);
		cudaMemcpy((void*)raygenRecord, &raygenData, raygenRecordSize, cudaMemcpyHostToDevice);

		CUdeviceptr missRecord;
		size_t missRecordSize = sizeof(MissSbtRecord);
		cudaMalloc((void**)&missRecord, missRecordSize);
		MissSbtRecord missData;
		optixSbtRecordPackHeader(missGroup, &missData);
		cudaMemcpy((void*)missRecord, &missData, missRecordSize, cudaMemcpyHostToDevice);

		CUdeviceptr hitgroupRecord;
		size_t hitgroupRecordSize = sizeof(HitgroupSbtRecord);
		cudaMalloc((void**)&hitgroupRecord, hitgroupRecordSize);
		HitgroupSbtRecord hitgroupData;
		optixSbtRecordPackHeader(hitgroupGroup, &hitgroupData);
		cudaMemcpy((void*)hitgroupRecord, &hitgroupData, hitgroupRecordSize, cudaMemcpyHostToDevice);

		sbt.raygenRecord = raygenRecord;
		sbt.missRecordBase = missRecord;
		sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
		sbt.missRecordCount = 1;
		sbt.hitgroupRecordBase = hitgroupRecord;
		sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupSbtRecord);
		sbt.hitgroupRecordCount = 1;
	}

	uchar4* d_output;
	
	GLuint pbo;
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(uchar4) * width * height, nullptr, GL_DYNAMIC_DRAW);

	cudaGraphicsResource* resource;
	cudaGraphicsGLRegisterBuffer(&resource, pbo, cudaGraphicsMapFlagsWriteDiscard);

	{
		CUstream stream;
		cudaStreamCreate(&stream);

		size_t size;
		cudaGraphicsMapResources(1, &resource, stream);
		cudaGraphicsResourceGetMappedPointer((void**)&d_output, &size, resource);

		float3 lookFrom = { 0.0f, 3.0f, -1.0f };
		float3 lookAt = { 0.0f, 0.0f, -1.0f };
		float3 lookUp = { 0.0f, 0.0f, -1.0f };
		float fov = 60.0f;
		float aspect = (float)width / (float)height;

		float3 w = lookAt - lookFrom;
		float wLen = length(w);
		float3 u = normalize(cross(w, lookUp));
		float3 v = normalize(cross(u, w));
		float vLen = wLen * tanf(0.5f * fov * 3.14159265358979323846f / 180.f);
		v *= vLen;
		float uLen = vLen * aspect;
		u *= uLen;

		Params params;
		params.image = d_output;
		params.width = width;
		params.height = height;
		params.handle = gasHandle;
		params.eye = lookFrom;
		params.u = u;
		params.v = v;
		params.w = w;

		CUdeviceptr d_param;
		cudaMalloc((void**)&d_param, sizeof(Params));
		cudaMemcpy((void*)d_param, &params, sizeof(Params), cudaMemcpyHostToDevice);

		optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, 1);
		cudaDeviceSynchronize();
		
		cudaStreamSynchronize(stream);

		cudaGraphicsUnmapResources(1, &resource, stream);

		cudaFree((void*)d_param);
	}

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);

		glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();

	return 0;
}