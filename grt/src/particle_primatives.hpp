#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>

constexpr uint32_t ICOSAHEDRON_NUM_VERT = 12;
constexpr uint32_t ICOSAHEDRON_NUM_TRI = 20;
constexpr float GOLDEN_RATIO   = 1.618033988749895f;
constexpr float ICOSAHEDRON_EDGE     = 1.323169076499215f;
constexpr float ICOSAHEDRON_VERT_SCALE = 0.5f * ICOSAHEDRON_EDGE;

#ifndef MOGRenderAdaptiveKernelClamping
#define MOGRenderAdaptiveKernelClamping 1
#endif

using Vec3 = glm::vec3;
using Mat3 = glm::mat3;
using Quat = glm::quat;
using Int3 = glm::ivec3;

inline std::array<Vec3, ICOSAHEDRON_NUM_VERT> getIcosahedronVertices() {
    return {
        Vec3(-1, GOLDEN_RATIO, 0), Vec3(1, GOLDEN_RATIO, 0), Vec3(0, 1, -GOLDEN_RATIO),
        Vec3(-GOLDEN_RATIO, 0, -1), Vec3(-GOLDEN_RATIO, 0, 1), Vec3(0, 1, GOLDEN_RATIO),
        Vec3(GOLDEN_RATIO, 0, 1), Vec3(0, -1, GOLDEN_RATIO), Vec3(-1, -GOLDEN_RATIO, 0),
        Vec3(0, -1, -GOLDEN_RATIO), Vec3(GOLDEN_RATIO, 0, -1), Vec3(1, -GOLDEN_RATIO, 0)
    };
}

inline std::array<Int3, ICOSAHEDRON_NUM_TRI> getIcosahedronTriangles() {
    return {
        Int3(0, 1, 2), Int3(0, 2, 3), Int3(0, 3, 4), Int3(0, 4, 5), Int3(0, 5, 1),
        Int3(6, 1, 5), Int3(6, 5, 7), Int3(6, 7, 11), Int3(6, 11, 10), Int3(6, 10, 1),
        Int3(8, 4, 3), Int3(8, 3, 9), Int3(8, 9, 11), Int3(8, 11, 7), Int3(8, 7, 4),
        Int3(9, 3, 2), Int3(9, 2, 10), Int3(9, 10, 11),
        Int3(5, 4, 7), Int3(1, 10, 2)
    };
}

inline float kernelScale(float density, float modulatedMinResponse, uint32_t opts, float kernelDegree) {
    const float responseModulation = opts & MOGRenderAdaptiveKernelClamping ? density : 1.0f;
    const float minResponse        = std::fmin(modulatedMinResponse / responseModulation, 0.97f);

    // bump kernel
    if (kernelDegree < 0) {
        const float k  = std::fabs(kernelDegree);
        const float s  = 1.0f / std::pow(3.0f, k);
        const float ks = std::pow((1.0f / (std::log(minResponse) - 1.0f) + 1.0f) / s, 1.0f / k);
        return ks;
    }

    // linear kernel
    if (kernelDegree == 0) {
        return ((1.0f - minResponse) / 3.0f) / -0.329630334487f;
    }

    // generalized gaussian of degree b : scaling a = -4.5/3^b
    // e^{a*|x|^b}
    const float b = kernelDegree;
    const float a = -4.5f / std::pow(3.0f, b);
    // find distance r (>0) st e^{a*r^b} = minResponse
    return std::pow(std::log(minResponse) / a, 1.0f / b);
}

inline void quaternionWXYZToMatrixTranspose(const glm::vec4& q, Mat3& outMat) {
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;

    // Compute rotation matrix from quaternion
    outMat[0] = glm::vec3((1.f - 2.f * (y * y + z * z)), 2.f * (x * y - r * z), 2.f * (x * z + r * y));
    outMat[1] = glm::vec3(2.f * (x * y + r * z), (1.f - 2.f * (x * x + z * z)), 2.f * (y * z - r * x));
    outMat[2] = glm::vec3(2.f * (x * z - r * y), 2.f * (y * z + r * x), (1.f - 2.f * (x * x + y * y)));
}

inline void computeGaussianEnclosingIcosahedron(
    uint32_t gNum,
    const Vec3* gPos,
    const glm::vec4* gRot,
    const Vec3* gScl,
    const float* gDns,
    float kernelMinResponse,
    uint32_t opts,
    float degree,
    Vec3* gPrimVrt,
    Int3* gPrimTri)
{
    const auto icosaVrt = getIcosahedronVertices();
    const auto icosaTri = getIcosahedronTriangles();
    for (uint32_t idx = 0; idx < gNum; ++idx) {
        uint32_t sVertIdx = ICOSAHEDRON_NUM_VERT * idx;
        uint32_t sTriIdx  = ICOSAHEDRON_NUM_TRI * idx;

        Mat3 rot;
        quaternionWXYZToMatrixTranspose(gRot[idx], rot);
        Vec3 scl   = gScl[idx];
        Vec3 trans = gPos[idx];

        Vec3 kscl = kernelScale(gDns[idx], kernelMinResponse, opts, degree) * scl * ICOSAHEDRON_VERT_SCALE;
        for (uint32_t i = 0; i < ICOSAHEDRON_NUM_VERT; ++i) {
            gPrimVrt[sVertIdx + i] = (icosaVrt[i] * kscl) * rot + trans;
        }
        Int3 triIdxOffset(sVertIdx, sVertIdx, sVertIdx);
        for (uint32_t i = 0; i < ICOSAHEDRON_NUM_TRI; ++i) {
            gPrimTri[sTriIdx + i] = icosaTri[i] + triIdxOffset;
        }
    }
}

