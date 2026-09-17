#include "tokenization.hpp"
#include "encoder.hpp"

#include "json.hpp"
#include "tensor_math.hpp"
#include <cassert>
#include <climits>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

vector<string> gpt2Tokens;
unordered_map<string, int> gpt2TokenToTokenId;
map<pair<string, string>, int> merges;

string getTokenFromTokenId(int tokenId) {
  assert(tokenId >= 0 && tokenId < gpt2Tokens.size());
  return gpt2Tokens[tokenId];
}

void loadVocab() {
  using json = nlohmann::json;
  json conf = json::parse("../../weights/tokenizer/tokenizer.json");

  // get the vocab and map tokens
  const auto &vocab = conf["model"]["vocab"];
  gpt2Tokens.resize(vocab.size());

  for (const auto &[token, tokenId] : vocab.items()) {
    gpt2Tokens[tokenId] = token;
    gpt2TokenToTokenId[token] = tokenId;
  }

  // BPE = Byte Pair Encoding merges
  const auto &mergesList = conf["model"]["merges"];

  int rank = 0;
  for (const auto &merge : mergesList) {
    assert(merge.is_array());
    assert(merge.size() == 2);

    string first = merge[0].get<string>();
    string second = merge[0].get<string>();
    merges[{first, second}] = rank++;
  }
}

// convert a token id into its embedding vector.
tensor getTokenEmbedding(const tensor &embeddingWeights, int tokenId) {
  constexpr int embeddingDim = 768;

  int start = tokenId * embeddingDim;
  return tensor(embeddingWeights.begin() + start,
                embeddingWeights.begin() + start + embeddingDim);
}

vector<string> splitTokens(const string &encoded) {
  vector<string> tokens;
  string current;

  for (int i = 0; i < encoded.size();) {
    // UTF-8 encoding of Ġ (U+0120) = C4 A0
    if (i + 1 < encoded.size() &&
        static_cast<unsigned char>(encoded[i]) == 0xC4 &&
        static_cast<unsigned char>(encoded[i + 1]) == 0xA0) {

      if (!current.empty()) {
        tokens.push_back(current);
        current.clear();
      }

      current = "Ġ";

      i += 2;
      continue;
    }

    current += encoded[i];
    i++;
  }

  if (!current.empty()) {
    tokens.push_back(current);
  }

  return tokens;
}

// apply merges according to map
vector<string> applyMerges(const string &piece) {
  vector<string> symbols;

  // split the piece into utf-8 characters.
  for (size_t i = 0; i < piece.size();) {
    unsigned char c = piece[i];

    size_t length = 1;

    if ((c & 0x80) == 0) {
      length = 1;
    } else if ((c & 0xE0) == 0xC0) {
      length = 2;
    } else if ((c & 0xF0) == 0xE0) {
      length = 3;
    } else if ((c & 0xF8) == 0xF0) {
      length = 4;
    }

    symbols.push_back(piece.substr(i, length));
    i += length;
  }

  while (symbols.size() > 1) {
    int bestRank = INT_MAX;
    size_t bestIndex = 0;
    bool found = false;

    // find the lowest-rank merge.
    for (size_t i = 0; i + 1 < symbols.size(); i++) {
      pair<string, string> pair = {symbols[i], symbols[i + 1]};

      auto it = merges.find(pair);

      if (it != merges.end() && it->second < bestRank) {
        bestRank = it->second;
        bestIndex = i;
        found = true;
      }
    }

    // No more merge rules apply.
    if (!found) {
      break;
    }

    // Merge the pair.
    symbols[bestIndex] = symbols[bestIndex] + symbols[bestIndex + 1];
    symbols.erase(symbols.begin() + bestIndex + 1);
  }

  return symbols;
}

// GPT-2 BPE tokenizer.
vector<int> tokenize(const string &text) {
  vector<int> tokenIds;

  // gpt-2 byte-level representation.
  string encoded = byteEncode(text);

  // split tokens
  vector<string> pieces = splitTokens(encoded);

  // BPE merges.
  for (const auto &piece : pieces) {
    vector<string> bpeTokens = applyMerges(piece);

    for (const string &token : bpeTokens) {
      auto it = gpt2TokenToTokenId.find(token);
      if (it == gpt2TokenToTokenId.end()) {
        throw runtime_error("Token not found in GPT-2 vocabulary: " + token);
      }

      tokenIds.push_back(it->second);
    }
  }

  return tokenIds;
}
