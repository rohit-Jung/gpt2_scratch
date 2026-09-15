#include "tensor_math.hpp"
#include <algorithm>
#include <cassert>
#include <math.h>
#include <numeric>
using namespace std;

using tensor = vector<float>;
constexpr double EPSILON = 0.00001;

vector<float> matMul(TensorView a, int an, int am, TensorView b, int bn,
                     int bm) {
  assert(am == bn);
  vector<float> result(an * bm, 0);

  for (int i = 0; i < an; i++) {
    for (int j = 0; j < bm; j++) {
      for (int k = 0; k < bn; k++) {
        result[i * bm + j] += a[i * am + k] * b[k * bm + j];
      }
    }
  }

  return result;
}

vector<float> transpose(TensorView a, int n, int m) {
  vector<float> result(m * n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      result[i * n + j] = a[j * m + i];
    }
  }
  return result;
}

float dotProduct(TensorView a, TensorView b) {
  float result = 0.00;
  for (int i = 0; i < a.size(); i++) {
    result += a[i] * b[i];
  }
  return result;
}

tensor addVectors(TensorView a, TensorView b) {
  assert(a.size() == b.size());
  tensor result(a.begin(), a.end());
  for (int i = 0; i < a.size(); i++) {
    result[i] += b[i];
  }
  return result;
}

tensor layerNorm(TensorView ogEmbeddings, TensorView weights,
                 TensorView biases) {
  float mean = 0, variance = 0;
  int n = ogEmbeddings.size();

  // mean calculation : 1/n * sum of X
  for (auto &i : ogEmbeddings)
    mean += i;
  mean /= n;

  // variance claculation: 1/n * (x - mean)^2
  for (auto &i : ogEmbeddings)
    variance += pow((i - mean), 2);
  variance /= n;

  // standard deviation
  float stdDev = sqrt(variance + EPSILON);

  // normalize
  tensor output(n);
  for (int i = 0; i < n; i++) {
    float normalizedVlaue = (ogEmbeddings[i] - mean) / stdDev;
    output[i] = normalizedVlaue * weights[i] + biases[i];
  }

  return output;
}

tensor softmax(TensorView input) {
  int n = input.size();
  float nx = *max_element(input.begin(), input.end());

  tensor output(n);
  for (int i = 0; i < n; i++) {
    output[i] = exp(input[i] - nx);
  }

  float sum = accumulate(output.begin(), output.end(), 0.0f);

  for (int i = 0; i < n; i++) {
    output[i] /= sum;
  }

  return output;
}

// gelu activation function
float gelu(float x) {
  return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x * x * x)));
}
