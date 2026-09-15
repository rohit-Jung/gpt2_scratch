#pragma once

#include <span>
#include <vector>

using tensor = std::vector<float>;
using TensorView = std::span<const float>;

tensor matMul(TensorView a, int an, int am, TensorView b, int bn, int bm);

tensor transpose(TensorView a, int n, int m);

tensor layerNorm(TensorView ogEmbeddings, TensorView weights,
                 TensorView biases);

tensor softmax(TensorView input);

float dotProduct(TensorView a, TensorView b);

tensor addVectors(TensorView a, TensorView b);

float gelu(float x);
