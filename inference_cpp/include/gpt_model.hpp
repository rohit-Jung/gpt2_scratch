#pragma once

#include "gpt_weights.hpp"
#include "tensor_math.hpp"
#include <vector>

tensor gpt(const std::vector<int> &tokenIds, const gptWeights &weights);
