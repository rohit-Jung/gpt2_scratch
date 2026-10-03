#include "gpt_weights.hpp"
#include "tensor_math.hpp"
#include "transformer.hpp"
#include <algorithm>

// transformer - attention and mlp
// tokens and embeddings according to token id (use gpt2)
// positional meaning

// qkv are different projection of same embeddings how multiplying with q, k, v
// weights softmax(q * k(transpose) / sqrt(d)) * v attention
tensor gpt(const vector<int> &tokenIds, const gptWeights &weights) {
  constexpr int embeddingDim = 768;

  int numTokens = tokenIds.size();

  tensor embeddings(numTokens * embeddingDim);

  // token + positional embeddings.
  for (int i = 0; i < numTokens; i++) {
    int tokenId = tokenIds[i];

    int tokenStart = tokenId * embeddingDim;
    int positionStart = i * embeddingDim;

    for (int j = 0; j < embeddingDim; j++) {
      embeddings[i * embeddingDim + j] =
          weights.embeddingWeights[tokenStart + j] +
          weights.positionalEmbeddingWeights[positionStart + j];
    }
  }

  // transformer blocks.
  tensor hidden = embeddings;

  for (const auto &transformerWeights : weights.transformerWeights) {
    hidden = transformer(const_cast<TransformerInput &>(transformerWeights),
                         numTokens, hidden);
  }

  // layernorm, applied per token (ln_f).
  tensor normed(numTokens * embeddingDim);

  for (int i = 0; i < numTokens; i++) {
    TensorView tokenHidden(hidden.data() + i * embeddingDim, embeddingDim);
    auto n = layerNorm(tokenHidden, weights.finalWeights, weights.finalBias);
    std::copy(n.begin(), n.end(), normed.begin() + i * embeddingDim);
  }

  // Project hidden states -> vocabulary logits. GPT-2 ties the output
  // projection to the token embedding table (wte) and has no output bias.
  constexpr int vocabSize = 50257;

  tensor logits(numTokens * vocabSize);

  for (int token = 0; token < numTokens; token++) {
    TensorView hiddenToken(normed.data() + token * embeddingDim, embeddingDim);

    for (int vocab = 0; vocab < vocabSize; vocab++) {
      float value = 0.0f;

      for (int j = 0; j < embeddingDim; j++) {
        value +=
            weights.embeddingWeights[vocab * embeddingDim + j] * hiddenToken[j];
      }

      logits[token * vocabSize + vocab] = value;
    }
  }

  return logits;
}
