#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <glm/glm.hpp>
#include "particle_primatives.hpp"
#include "math_utils.hpp"

// Struct for 45-element vector
struct Vec45
{
    float r[45];

    float &operator[](size_t i) { return r[i]; }
    const float &operator[](size_t i) const { return r[i]; }
};

class GRTModel
{
public:
    GRTModel(const std::string &path, bool printInfo = false)
    {
        _load(path, printInfo);
    }

    int N = 0;
    int pos_dim = 0, scales_dim = 0, rotations_dim = 0;
    int albedo_dim = 0, specular_dim = 0, densities_dim = 0;

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> scales;
    std::vector<glm::vec4> rotations;
    std::vector<glm::vec3> albedos;
    std::vector<Vec45> speculars;
    std::vector<float> densities;
    std::vector<glm::vec3> vertices;
    std::vector<glm::ivec3> triangles;

private:
    void _load(const std::string &filepath, bool printInfo)
    {
        std::ifstream fin(filepath, std::ios::binary);
        if (!fin)
        {
            std::cerr << "Failed to open file: " << filepath << "\n";
            return;
        }

        int header[7];
        fin.read(reinterpret_cast<char *>(header), 7 * sizeof(int));
        if (fin.gcount() != 7 * sizeof(int))
        {
            std::cerr << "Failed to read header.\n";
            return;
        }

        N = header[0];
        pos_dim = header[1];
        scales_dim = header[2];
        rotations_dim = header[3];
        albedo_dim = header[4];
        specular_dim = header[5];
        densities_dim = header[6];

        // read helpers
        auto readVec3 = [&](std::vector<glm::vec3> &vec)
        {
            vec.resize(N);
            for (int i = 0; i < N; ++i)
            {
                float buffer[3];
                fin.read(reinterpret_cast<char *>(buffer), 3 * sizeof(float));
                vec[i] = glm::vec3(buffer[0], buffer[1], buffer[2]);
            }
        };

        auto readVec4 = [&](std::vector<glm::vec4> &vec)
        {
            vec.resize(N);
            for (int i = 0; i < N; ++i)
            {
                float buffer[4];
                fin.read(reinterpret_cast<char *>(buffer), 4 * sizeof(float));
                vec[i] = glm::vec4(buffer[0], buffer[1], buffer[2], buffer[3]);
            }
        };

        auto readVec45 = [&](std::vector<Vec45> &vec)
        {
            vec.resize(N);
            for (int i = 0; i < N; ++i)
            {
                fin.read(reinterpret_cast<char *>(vec[i].r), 45 * sizeof(float));
            }
        };

        auto readFloat = [&](std::vector<float> &vec)
        {
            vec.resize(N);
            fin.read(reinterpret_cast<char *>(vec.data()), N * sizeof(float));
        };

        // read all tensors
        readVec3(positions);
        readVec3(scales);    
        readVec4(rotations); 
        readVec3(albedos);    
        readVec45(speculars);
        readFloat(densities);

        // apply activations
        auto& scalesNormalized = scales;
        auto& rotationsNormalized = rotations;
        auto& densitiesNormalized = densities;

        normalize(rotationsNormalized);
        exp(scalesNormalized);
        sigmoid(densitiesNormalized);
        
        // load vertices and triangles
        vertices = std::vector<glm::vec3>(N * ICOSAHEDRON_NUM_VERT);
        triangles = std::vector<glm::ivec3>(N* ICOSAHEDRON_NUM_TRI);
        const float kernelMinResponse = 0.0113000004f;
        const uint32_t opts = 0;
        const float degree = 4;

        computeGaussianEnclosingIcosahedron(
            N,
            positions.data(),
            rotationsNormalized.data(),
            scalesNormalized.data(),
            densitiesNormalized.data(),
            kernelMinResponse,
            opts,
            degree,
            vertices.data(),
            triangles.data()
        );

        if (printInfo)
        {
            _printInfo();
        }
    }

    void _printInfo() const
    {
        std::cout << "Shapes:\n";
        std::cout << "N: " << N << "\n";
        std::cout << "positions shape: (" << N << ", " << pos_dim << ")\n";
        std::cout << "scales shape: (" << N << ", " << scales_dim << ")\n";
        std::cout << "rotations shape: (" << N << ", " << rotations_dim << ")\n";
        std::cout << "albedos shape: (" << N << ", " << albedo_dim << ")\n";
        std::cout << "speculars shape: (" << N << ", " << specular_dim << ")\n";
        std::cout << "densities shape: (" << N << ", " << densities_dim << ")\n\n";

        auto printFirstLast = [](const std::string &name, const auto &vec, int dim)
        {
            if (vec.empty())
                return;

            std::cout << name << " first: [";
            for (int i = 0; i < dim; ++i)
                std::cout << vec[0][i] << (i < dim - 1 ? ", " : "");
            std::cout << "]\n";

            std::cout << name << " last:  [";
            for (int i = 0; i < dim; ++i)
                std::cout << vec.back()[i] << (i < dim - 1 ? ", " : "");
            std::cout << "]\n\n";
        };

        printFirstLast("positions", positions, 3);
        printFirstLast("scales", scales, 3);
        printFirstLast("rotations", rotations, 4);
        printFirstLast("albedos", albedos, 3);
        printFirstLast("speculars", speculars, 45);
        std::cout << "densities first: " << densities[0] << "\n";
        std::cout << "densities last:  " << densities.back() << "\n";
    }
};
