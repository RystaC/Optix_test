#pragma once

#include <optix.h>
#include <cuda_runtime.h>

struct Params {
	uchar4* image;
	unsigned int width;
	unsigned int height;
	float3 eye;
	float3 u, v, w;
	OptixTraversableHandle handle;
};

struct RaygenData {
};

struct MissData {
};

struct HitgroupData {
};

struct Payload {
	float3 color;
};