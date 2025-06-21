#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <vector>

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline void sigmoid(std::vector<float>& vecs) {
    for (auto& v : vecs) {
        v = sigmoid(v);
    }
}

template<glm::length_t L, glm::qualifier Q = glm::defaultp>
inline void normalize(std::vector<glm::vec<L, float, Q>>& vecs, float epsilon = 1e-12f) {
    for (auto& v : vecs) {
        float norm = std::sqrt(glm::dot(v, v));
        if (norm > epsilon) v /= norm;
        else v = glm::vec<L, float, Q>(0.0f);
    }
}

template<glm::length_t L, glm::qualifier Q = glm::defaultp>
inline void exp(std::vector<glm::vec<L, float, Q>>& vecs) {
    for (auto& v : vecs) {
        v = glm::exp(v);
    }
}