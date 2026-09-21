#pragma once

#include "tensor_math.hpp"

struct TransformerInput {
  // attention
  tensor qWeights, kWeights, vWeights;
  tensor qBiases, kBiases, vBiases;

  // mlp
  tensor l1Weights, l2Weights;
  tensor l1Biases, l2Biases;

  // attention output
  tensor oWeights, oBiases;

  // layer norm
  tensor lnAttnWeights, lnAttnBiases;
  tensor lnMlpWeights, lnMlpBiases;
};

constexpr double EMBEDDING_DIM = 768;

tensor forwardPass(TensorView weights, TensorView biases, TensorView inputs,
                   bool useGelu = false);

tensor mlp(int numTokens, int dimensions, TensorView embeddings,
           TensorView l1Weights, TensorView l2Weights, TensorView l1Biases,
           TensorView l2Biases);

tensor transformer(TransformerInput &input, int numTokens,
                   const tensor &embeddings);
