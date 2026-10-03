#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "tensor_math.hpp"
#include "transformer.hpp"

using std::ifstream;
using std::string;
using std::to_string;
using std::vector;

struct gptWeights {
  // embedding, positional embedding
  tensor embeddingWeights;
  tensor positionalEmbeddingWeights;

  // transformer blocks
  vector<TransformerInput> transformerWeights;

  // final layer normalization
  tensor finalBias;
  tensor finalWeights;

  gptWeights();
};
