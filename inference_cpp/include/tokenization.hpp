#pragma once

#include "tensor_math.hpp"
#include <string>
#include <vector>

std::string getTokenFromTokenId(int tokenId);
void loadVocab();

std::vector<int> tokenize(const std::string &text);
std::vector<std::string> applyMerges(const std::string &piece);
std::vector<std::string> splitTokens(const std::string &encoded);
tensor getTokenEmbedding(const tensor &embeddingWeights, int tokenId);
