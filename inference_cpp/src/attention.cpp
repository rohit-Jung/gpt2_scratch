#include "attention.hpp"
#include "tensor_math.hpp"
#include <math.h>

tensor attention(TensorView qWeights, TensorView kWeights, TensorView vWeights,
                 TensorView qBiases, TensorView kBiases, TensorView vBiases,
                 int numTokens, int embedDim, int headDim,
                 TensorView embeddings) {
  tensor qProjections(numTokens * headDim);
  tensor kProjections(numTokens * headDim);
  tensor vProjections(numTokens * headDim);

  // Project embeddings into Q, K, and V.
  for (int t = 0; t < numTokens; t++) {
    auto tokenEmb = embeddings.subspan(t * embedDim, embedDim);

    for (int d = 0; d < headDim; d++) {
      auto qRow = qWeights.subspan(d * embedDim, embedDim);
      auto kRow = kWeights.subspan(d * embedDim, embedDim);
      auto vRow = vWeights.subspan(d * embedDim, embedDim);

      qProjections[t * headDim + d] = dotProduct(tokenEmb, qRow) + qBiases[d];
      kProjections[t * headDim + d] = dotProduct(tokenEmb, kRow) + kBiases[d];
      vProjections[t * headDim + d] = dotProduct(tokenEmb, vRow) + vBiases[d];
    }
  }

  // q @ k^t
  auto kTranspose = transpose(kProjections, numTokens, headDim);

  auto attentionScores =
      matMul(qProjections, numTokens, headDim, kTranspose, headDim, numTokens);

  // scale by sqrt(headdim).
  float scale = 1.0f / sqrt(static_cast<float>(headDim));

  for (auto &score : attentionScores)
    score *= scale;

  // causal mask: token i cannot attend to future tokens j > i.
  for (int i = 0; i < numTokens; i++) {
    for (int j = i + 1; j < numTokens; j++) {
      attentionScores[i * numTokens + j] =
          -std::numeric_limits<float>::infinity();
    }
  }

  // Softmax each attention row.
  for (int i = 0; i < numTokens; i++) {
    auto row = std::span(attentionScores).subspan(i * numTokens, numTokens);
    auto softrow = softmax(row);

    for (int j = 0; j < numTokens; j++)
      attentionScores[i * numTokens + j] = softrow[j];
  }

  // attention @ v
  return matMul(attentionScores, numTokens, numTokens, vProjections, numTokens,
                headDim);
}

tensor multiHeadAttention(int numTokens, int embedDim, int headDim,
                          TensorView embeddings, TensorView qWeights,
                          TensorView kWeights, TensorView vWeights,
                          TensorView qBiases, TensorView kBiases,
                          TensorView vBiases, TensorView oWeights,
                          TensorView oBiases) {

  tensor result(numTokens * embedDim, 0);

  int numHeads = embedDim / headDim;
  int weightBlockSize = embedDim * headDim;

  for (int h = 0; h < numHeads; h++) {
    auto qHead = qWeights.subspan(h * weightBlockSize, weightBlockSize);
    auto kHead = kWeights.subspan(h * weightBlockSize, weightBlockSize);
    auto vHead = vWeights.subspan(h * weightBlockSize, weightBlockSize);

    auto qHeadBiases = qBiases.subspan(h * headDim, headDim);
    auto kHeadBiases = kBiases.subspan(h * headDim, headDim);
    auto vHeadBiases = vBiases.subspan(h * headDim, headDim);

    auto curResult =
        attention(qHead, kHead, vHead, qHeadBiases, kHeadBiases, vHeadBiases,
                  numTokens, embedDim, headDim, embeddings);

    // concatenate
    for (int i = 0; i < numTokens; i++) {
      for (int j = 0; j < headDim; j++) {

        result[i * embedDim + h * headDim + j] = curResult[i * headDim + j];
      }
    }
  }

  auto projectionResult =
      matMul(result, numTokens, embedDim, oWeights, embedDim, embedDim);

  for (int i = 0; i < numTokens; i++) {
    for (int j = 0; j < embedDim; j++) {
      projectionResult[i * embedDim + j] += oBiases[j];
    }
  }

  return projectionResult;
}
