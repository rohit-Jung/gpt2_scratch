#include "encoder.hpp"
#include "gpt_model.hpp"
#include "gpt_weights.hpp"
#include "tensor_math.hpp"
#include "tokenization.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <utility>
using namespace std;

int main() {
  // load gpt-2 weights.
  gptWeights weights;

  // load tokenizer vocabulary and bpe merges.
  loadVocab();

  while (true) {

    // get prompt.
    string prompt;

    cout << "Prompt: ";
    getline(cin, prompt);

    // get number of tokens to generate.
    int maxNewTokens;

    cout << "Number of tokens to generate: ";
    cin >> maxNewTokens;

    // Convert prompt to GPT-2 token IDs.
    vector<int> tokenIds = tokenize(prompt);

    cout << "\nPrompt tokens: ";

    for (int id : tokenIds) {
      cout << id << " ";
    }

    cout << "\n\n";

    constexpr int vocabSize = 50257;

    // Top-k + temperature sampling. Greedy (always the highest-logit token)
    constexpr int topK = 40;
    constexpr float temperature = 0.8f;

    mt19937 rng(random_device{}());

    // autoregressive generation.
    for (int step = 0; step < maxNewTokens; step++) {
      // Run the entire sequence through GPT-2.
      tensor logits = gpt(tokenIds, weights);

      // Logits for the last token.
      int lastTokenOffset = (tokenIds.size() - 1) * vocabSize;

      vector<pair<float, int>> scored(vocabSize);
      for (int i = 0; i < vocabSize; i++) {
        scored[i] = {logits[lastTokenOffset + i], i};
      }

      // keep only the topk highest-logit tokens.
      partial_sort(
          scored.begin(), scored.begin() + topK, scored.end(),
          [](const auto &a, const auto &b) { return a.first > b.first; });
      scored.resize(topK);

      // softmax (with temperature) over just those topk logits to get
      // sampling probabilities.
      tensor topLogits(topK);
      for (int i = 0; i < topK; i++) {
        topLogits[i] = scored[i].first / temperature;
      }
      tensor probs = softmax(topLogits);

      discrete_distribution<int> dist(probs.begin(), probs.end());
      int bestToken = scored[dist(rng)].second;

      // add predicted token to the sequence.
      tokenIds.push_back(bestToken);

      // convert token id back to text.
      string token = getTokenFromTokenId(bestToken);

      cout << byteDecode(token) << flush;
    }

    cout << "\n";
  }

  return 0;
}
