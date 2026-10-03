#include "gpt_weights.hpp"

gptWeights::gptWeights() : transformerWeights(12) {
  auto readFromStream = [](vector<float> &vec, ifstream &stream,
                           int maxCount = 0) {
    float input;

    if (maxCount > 0) {
      while (maxCount > 0 && stream >> input) {
        vec.push_back(input);
        maxCount--;
      }
    } else {
      while (stream >> input) {
        vec.push_back(input);
      }
    }
  };

  for (int i = 0; i < 12; i++) {
    string prefix = "../weights/transformer.h." + to_string(i);

    // layernorm 1
    ifstream ln1_weight(prefix + ".ln_1.weight.txt");
    ifstream ln1_bias(prefix + ".ln_1.bias.txt");

    readFromStream(transformerWeights[i].lnAttnWeights, ln1_weight);
    readFromStream(transformerWeights[i].lnAttnBiases, ln1_bias);

    // layernorm 2
    ifstream ln2_weight(prefix + ".ln_2.weight.txt");
    ifstream ln2_bias(prefix + ".ln_2.bias.txt");

    readFromStream(transformerWeights[i].lnMlpWeights, ln2_weight);
    readFromStream(transformerWeights[i].lnMlpBiases, ln2_bias);

    // attention qkv
    tensor cAttnWeights;
    tensor cAttnBiases;

    ifstream cAttnWeight(prefix + ".attn.c_attn.weight.txt");
    ifstream cAttnBias(prefix + ".attn.c_attn.bias.txt");

    readFromStream(cAttnWeights, cAttnWeight);
    readFromStream(cAttnBiases, cAttnBias);

    const int hiddenSize = 768;
    const int mlpHiddenSize = hiddenSize * 4;
    const int qkvSize = hiddenSize * hiddenSize;

    // PyTorch's Conv1D (used by GPT-2 for c_attn/c_proj/c_fc, unlike
    // nn.Linear) stores weight as [in_features, out_features]. Our
    // matmul/attention code expects [out_features, in_features], so we
    // transpose right after loading. c_attn.weight is [768, 2304]; after
    // transposing to [2304, 768] the Q/K/V blocks are contiguous rows.
    cAttnWeights = transpose(cAttnWeights, hiddenSize, hiddenSize * 3);

    transformerWeights[i].qWeights.assign(cAttnWeights.begin(),
                                          cAttnWeights.begin() + qkvSize);

    transformerWeights[i].kWeights.assign(cAttnWeights.begin() + qkvSize,
                                          cAttnWeights.begin() + 2 * qkvSize);

    transformerWeights[i].vWeights.assign(cAttnWeights.begin() + 2 * qkvSize,
                                          cAttnWeights.end());

    transformerWeights[i].qBiases.assign(cAttnBiases.begin(),
                                         cAttnBiases.begin() + hiddenSize);

    transformerWeights[i].kBiases.assign(cAttnBiases.begin() + hiddenSize,
                                         cAttnBiases.begin() + 2 * hiddenSize);

    transformerWeights[i].vBiases.assign(cAttnBiases.begin() + 2 * hiddenSize,
                                         cAttnBiases.end());

    // attention output projection
    ifstream cProjWeight(prefix + ".attn.c_proj.weight.txt");
    ifstream cProjBias(prefix + ".attn.c_proj.bias.txt");

    readFromStream(transformerWeights[i].oWeights, cProjWeight);
    readFromStream(transformerWeights[i].oBiases, cProjBias);

    // Unlike qWeights/kWeights/vWeights (consumed via a per-output-row
    // dotProduct, which needs [out, in]), oWeights is consumed via matMul's
    // C[i][j] = sum_k A[i][k]*B[k][j] convention, which wants B laid out
    // [in, out] -- exactly the raw Conv1D layout. No transpose here.

    // mlp
    ifstream fcWeight(prefix + ".mlp.c_fc.weight.txt");
    ifstream fcBias(prefix + ".mlp.c_fc.bias.txt");

    ifstream mlpProjectionWeight(prefix + ".mlp.c_proj.weight.txt");
    ifstream mlpProjectionBias(prefix + ".mlp.c_proj.bias.txt");

    tensor rawL1Weights;
    readFromStream(rawL1Weights, fcWeight);
    readFromStream(transformerWeights[i].l1Biases, fcBias);

    tensor rawL2Weights;
    readFromStream(rawL2Weights, mlpProjectionWeight);
    readFromStream(transformerWeights[i].l2Biases, mlpProjectionBias);

    // c_fc.weight is [in=768, out=3072] -> transpose to [3072, 768].
    transformerWeights[i].l1Weights =
        transpose(rawL1Weights, hiddenSize, mlpHiddenSize);

    // c_proj.weight (mlp) is [in=3072, out=768] -> transpose to [768, 3072].
    transformerWeights[i].l2Weights =
        transpose(rawL2Weights, mlpHiddenSize, hiddenSize);
  }

  // final layer normalization
  ifstream finalWeight("../weights/transformer.ln_f.weight.txt");
  ifstream finalBiasFile("../weights/transformer.ln_f.bias.txt");

  readFromStream(finalWeights, finalWeight);
  readFromStream(finalBias, finalBiasFile);

  // token embeddings
  ifstream embeddingFile("../weights/transformer.wte.weight.txt");
  ifstream positionalEmbeddingFile("../weights/transformer.wpe.weight.txt");

  readFromStream(embeddingWeights, embeddingFile);
  readFromStream(positionalEmbeddingWeights, positionalEmbeddingFile);
}
