#pragma once

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

void readGLB(std::string, std::vector<float3>&, std::vector<uint3>&);

class glTF {
};