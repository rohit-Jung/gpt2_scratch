#pragma once

#include "tensor_math.hpp"

tensor attention(TensorView qWeights, TensorView kWeights, TensorView vWeights,
                 TensorView qBiases, TensorView kBiases, TensorView vBiases,
                 int numTokens, int embedDim, int headDim,
                 TensorView embeddings);

tensor multiHeadAttention(int numTokens, int embedDim, int headDim,
                          TensorView embeddings, TensorView qWeights,
                          TensorView kWeights, TensorView vWeights,
                          TensorView qBiases, TensorView kBiases,
                          TensorView vBiases, TensorView oWeights,
                          TensorView oBiases);
