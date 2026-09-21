#include "transformer.hpp"
#include "attention.hpp"
#include "tensor_math.hpp"

tensor forwardPass(TensorView weights, TensorView biases, TensorView inputs,
                   bool useGelu) {
  int countNeurons = biases.size();
  tensor output(biases.begin(), biases.end());

  for (int i = 0; i < countNeurons; i++) {
    for (int j = 0; j < inputs.size(); j++) {
      output[i] += weights[i * inputs.size() + j] * inputs[j];
    }
  }

  if (useGelu) {
    for (auto &v : output)
      v = gelu(v);
  }

  return output;
}

tensor mlp(int numTokens, int dimensions, TensorView embeddings,
           TensorView l1Weights, TensorView l2Weights, TensorView l1Biases,
           TensorView l2Biases) {
  tensor result(numTokens * dimensions);

  for (int i = 0; i < numTokens; i++) {
    auto tokenEmb = embeddings.subspan(i * dimensions, dimensions);
    auto hiddenOut = forwardPass(l1Weights, l1Biases, tokenEmb, true);
    auto out = forwardPass(l2Weights, l2Biases, hiddenOut);

    for (int j = 0; j < dimensions; j++) {
      result[i * dimensions + j] = out[j];
    }
  }

  return result;
}

tensor transformer(TransformerInput &input, int numTokens,
                   const tensor &embeddings) {
  tensor layerNormEmbeddings(numTokens * EMBEDDING_DIM);

  for (int i = 0; i < numTokens; i++) {
    tensor tokenEmbedding(embeddings.begin() + i * EMBEDDING_DIM,
                          embeddings.begin() + (i + 1) * EMBEDDING_DIM);

    auto normed =
        layerNorm(tokenEmbedding, input.lnAttnWeights, input.lnAttnBiases);

    for (int j = 0; j < EMBEDDING_DIM; j++) {
      layerNormEmbeddings[i * EMBEDDING_DIM + j] = normed[j];
    }
  }

  auto attentionResult = multiHeadAttention(
      numTokens, EMBEDDING_DIM, 64, layerNormEmbeddings, input.qWeights,
      input.kWeights, input.vWeights, input.qBiases, input.kBiases,
      input.vBiases, input.oWeights, input.oBiases);

  // residual 1: x = x + attention(LN1(x))
  auto afterAttention = addVectors(embeddings, attentionResult);

  tensor layerNormMlp(numTokens * EMBEDDING_DIM);
  for (int i = 0; i < numTokens; i++) {
    tensor tokenEmbedding(afterAttention.begin() + i * EMBEDDING_DIM,
                          afterAttention.begin() + (i + 1) * EMBEDDING_DIM);

    auto normed =
        layerNorm(tokenEmbedding, input.lnMlpWeights, input.lnMlpBiases);

    for (int j = 0; j < EMBEDDING_DIM; j++) {
      layerNormMlp[i * EMBEDDING_DIM + j] = normed[j];
    }
  }

  auto mlpResult = mlp(numTokens, EMBEDDING_DIM, layerNormMlp, input.l1Weights,
                       input.l2Weights, input.l1Biases, input.l2Biases);

  // residual 2: out = x + mlp(LN2(x))
  return addVectors(afterAttention, mlpResult);
}
