#include "gltf.h"

using namespace nlohmann;

void readGLB(std::string path, std::vector<float3>& vertices, std::vector<uint3>& indices) {

	std::ifstream file(path, std::ios::in | std::ios::binary);
	if (!file) {
		std::cerr << "Cannot read file: " << path << std::endl;
		return;
	}

	{
		uint32_t magic;
		file.read((char*)&magic, sizeof(uint32_t));
		if (magic != 0x46546C67) {
			std::cerr << "Input file is not binary glTF." << std::endl;
			return;
		}

		uint32_t version;
		file.read((char*)&version, sizeof(uint32_t));
		if (version != 2) {
			std::cerr << "Input file is not glTF 2.x." << std::endl;
			return;
		}
	}

	uint32_t length;
	file.read((char*)&length, sizeof(uint32_t));

	json jsonData;

	{
		uint32_t chunkLength;
		file.read((char*)&chunkLength, sizeof(uint32_t));
		uint32_t chunkType;
		file.read((char*)&chunkType, sizeof(uint32_t));
		// JSON
		if (chunkType != 0x4E4F534A) {
			std::cerr << "This file does not have JSON chunk first." << std::endl;
			return;
		}
		std::string jsonString(chunkLength, ' ');
		file.read((char*)jsonString.data(), sizeof(unsigned char) * chunkLength);
		jsonData = json::parse(jsonString);
	}

	if (file.tellg() == length) {
		std::cerr << "This file has no BIN chunk." << std::endl;
		return;
	}

	{
		uint32_t chunkLength;
		file.read((char*)&chunkLength, sizeof(uint32_t));
		uint32_t chunkType;
		file.read((char*)&chunkType, sizeof(uint32_t));
		// BIN
		if (chunkType != 0x004E4942) {
			std::cerr << "This file has invalid chunk, expected BIN chunk." << std::endl;
			return;
		}
	}

	auto binBegin = file.tellg();

	{
		auto gltfVerison = jsonData["asset"]["version"];
		if (gltfVerison != "2.0") {
			std::cerr << "This file is not glTF 2.0." << std::endl;
			return;
		}
	}

	uint32_t verticesIdx = jsonData["meshes"][0]["primitives"][0]["attributes"]["POSITION"];
	uint32_t indicesIdx = jsonData["meshes"][0]["primitives"][0]["indices"];

	auto verticesAccessor = jsonData["accessors"][verticesIdx];
	auto indicesAccessor = jsonData["accessors"][indicesIdx];

	uint32_t verticesBufferViewIdx = verticesAccessor["bufferView"];
	uint32_t indicesBufferViewIdx = indicesAccessor["bufferView"];

	auto verticesBufferView = jsonData["bufferViews"][verticesBufferViewIdx];
	auto indicesBufferView = jsonData["bufferViews"][indicesBufferViewIdx];

	uint32_t verticesBufferLength = verticesBufferView["byteLength"];
	uint32_t verticesBufferOffset = verticesBufferView["byteOffset"];

	uint32_t indicesBufferLength = indicesBufferView["byteLength"];
	uint32_t indicesBufferOffset = indicesBufferView["byteOffset"];

	// vertices
	file.seekg(binBegin);
	file.seekg(sizeof(unsigned char) * verticesBufferOffset, std::ios::cur);

	vertices.reserve(verticesBufferLength / 4 / 3);
	for (int i = 0; i < verticesBufferLength / 4 / 3; ++i) {
		float x, y, z;
		file.read((char*)&x, sizeof(float));
		file.read((char*)&y, sizeof(float));
		file.read((char*)&z, sizeof(float));
		vertices.push_back({x, y, z});
	}

	// indices
	file.seekg(binBegin);
	file.seekg(sizeof(unsigned char) * indicesBufferOffset, std::ios::cur);

	indices.reserve(indicesBufferLength / 2 / 3);
	for (int i = 0; i < indicesBufferLength / 2 / 3; ++i) {
		uint16_t i0, i1, i2;
		file.read((char*)&i0, sizeof(uint16_t));
		file.read((char*)&i1, sizeof(uint16_t));
		file.read((char*)&i2, sizeof(uint16_t));
		indices.push_back({i0, i1, i2});
	}
}