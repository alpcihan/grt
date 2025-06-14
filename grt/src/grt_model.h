#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

class GRTModel {
public:
    GRTModel(const std::string& path, bool printInfo = false) {
        load(path, printInfo);
    }

    // Member variables
    int N = 0;
    int pos_dim = 0, scale_dim = 0, rotate_dim = 0;
    int albedo_dim = 0, specular_dim = 0, density_dim = 0;

    std::vector<float> position;
    std::vector<float> scale;
    std::vector<float> rotate;
    std::vector<float> features_albedo;
    std::vector<float> features_specular;
    std::vector<float> density;

private:
    void load(const std::string& filepath, bool printInfo) {
        std::ifstream fin(filepath, std::ios::binary);
        if (!fin) {
            std::cerr << "Failed to open file: " << filepath << "\n";
            return;
        }

        int header[7];
        fin.read(reinterpret_cast<char*>(header), 7 * sizeof(int));
        if (fin.gcount() != 7 * sizeof(int)) {
            std::cerr << "Failed to read header.\n";
            return;
        }

        N = header[0];
        pos_dim = header[1];
        scale_dim = header[2];
        rotate_dim = header[3];
        albedo_dim = header[4];
        specular_dim = header[5];
        density_dim = header[6];

        if (printInfo) {
            std::cout << "Shapes:\n";
            std::cout << "N: " << N << "\n";
            std::cout << "position shape: (" << N << ", " << pos_dim << ")\n";
            std::cout << "scale shape: (" << N << ", " << scale_dim << ")\n";
            std::cout << "rotate shape: (" << N << ", " << rotate_dim << ")\n";
            std::cout << "features_albedo shape: (" << N << ", " << albedo_dim << ")\n";
            std::cout << "features_specular shape: (" << N << ", " << specular_dim << ")\n";
            std::cout << "density shape: (" << N << ", " << density_dim << ")\n";
            std::cout << std::endl;
        }

        auto readTensor = [&](int rows, int cols, std::vector<float>& vec) {
            size_t count = size_t(rows) * cols;
            vec.resize(count);
            fin.read(reinterpret_cast<char*>(vec.data()), count * sizeof(float));
            if (fin.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
                std::cerr << "Failed to read tensor data\n";
                vec.clear();
            }
        };

        readTensor(N, pos_dim, position);
        readTensor(N, scale_dim, scale);
        readTensor(N, rotate_dim, rotate);
        readTensor(N, albedo_dim, features_albedo);
        readTensor(N, specular_dim, features_specular);
        readTensor(N, density_dim, density);

        if (printInfo) {
            auto printFirstLast = [](const std::string& name, const std::vector<float>& data, int dim) {
                if (data.empty()) return;
                std::cout << name << " first element: [";
                for (int i = 0; i < dim; ++i) {
                    std::cout << data[i];
                    if (i < dim - 1) std::cout << ", ";
                }
                std::cout << "]\n";

                std::cout << name << " last element:  [";
                size_t start = data.size() - dim;
                for (int i = 0; i < dim; ++i) {
                    std::cout << data[start + i];
                    if (i < dim - 1) std::cout << ", ";
                }
                std::cout << "]\n\n";
            };

            printFirstLast("position", position, pos_dim);
            printFirstLast("scale", scale, scale_dim);
            printFirstLast("rotate", rotate, rotate_dim);
            printFirstLast("features_albedo", features_albedo, albedo_dim);
            printFirstLast("features_specular", features_specular, specular_dim);
            printFirstLast("density", density, density_dim);
        }
    }
};
