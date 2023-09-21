#include <optix.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "hello.h"
#include "helper_math.h"

extern "C" {
	__constant__ Params params;
	__device__ curandState randState;
}

static __forceinline__ __device__ uchar4 makeColor(float r, float g, float b) {
	return make_uchar4((unsigned char)(r * 255.0f), (unsigned char)(g * 255.0f), (unsigned char)(b * 255.0f), 255);
}

static __forceinline__ __device__ Payload getPayload() {
	float r = __uint_as_float(optixGetPayload_0());
	float g = __uint_as_float(optixGetPayload_1());
	float b = __uint_as_float(optixGetPayload_2());
	return Payload{ make_float3(r, g, b)};
}

static __forceinline__ __device__ void setPayload(Payload p) {
	optixSetPayload_0(__float_as_uint(p.color.x));
	optixSetPayload_1(__float_as_uint(p.color.y));
	optixSetPayload_2(__float_as_uint(p.color.z));
}

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction) {
	const float3 u = params.u;
	const float3 v = params.v;
	const float3 w = params.w;
	const float2 d = 2.0f * make_float2((float)idx.x / (float)dim.x, (float)idx.y / (float)dim.y) - 1.0f;

	origin = params.eye;
	direction = normalize(d.x * u + d.y * v + w);
}

static __forceinline__ __device__ float3 randUnitSphere() {
	while (true) {
		float3 p = make_float3(curand_uniform(&randState), curand_uniform(&randState), curand_uniform(&randState)) * 2.0f - 1.0f;
		if (dot(p, p) >= 1) continue;
		return p;
	}
}

/*static __forceinline__ __device__ void computeOnb(float3 normal, float3 onb[3]) {
	onb[2] = normalize(normal);
	float3 a = (fabs(onb[2].x) > 0.9) ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
	onb[1] = normalize(cross(onb[2], a));
	onb[0] = cross(onb[2], onb[1]);
}*/

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	unsigned long threadIdx = idx.y * dim.x + idx.x;
	curand_init(0, threadIdx, 0, &randState);

	float3 origin, direction;
	computeRay(idx, dim, origin, direction);

	unsigned int p0, p1, p2;
	optixTrace(
		params.handle,
		origin, direction,
		0.0f,
		1e16f,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		0,
		1,
		0,
		p0, p1, p2
	);

	params.image[idx.y * params.width + idx.x] = makeColor(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
}

extern "C" __global__ void __miss__ms() {
	float3 unitDir = normalize(optixGetWorldRayDirection());
	float t = 0.5f * (unitDir.z + 1.0);
	
	float3 background = t * make_float3(1.0f) + (1.0f - t) * make_float3(0.5f, 0.7f, 1.0f);

	setPayload(Payload{ background });
}

extern "C" __global__ void __closesthit__ch() {
	OptixTraversableHandle gas = optixGetGASTraversableHandle();
	uint primIdx = optixGetPrimitiveIndex();
	uint sbtGASIdx = optixGetSbtGASIndex();

	float3 vertices[3];

	optixGetTriangleVertexData(gas, primIdx, sbtGASIdx, 0.0f, vertices);

	float3 e0 = vertices[1] - vertices[0];
	float3 e1 = vertices[2] - vertices[0];
	float3 normal = normalize(cross(e0, e1));

	float3 nextRayOrigin = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
	float3 nextRayDirection = normal + randUnitSphere();

	setPayload(Payload{ normal * 0.5 + 0.5 });
}