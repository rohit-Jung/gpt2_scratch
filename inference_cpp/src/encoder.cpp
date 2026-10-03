#include "encoder.hpp"
#include "gpt_weights.hpp"
using namespace std;

// encode a unicode code point as utf-8.
// ref: RFC 3629, https://www.rfc-editor.org/rfc/rfc3629.html
string utf8Encode(int codepoint) {
  string result;

  if (codepoint <= 0x7F) {
    result += static_cast<char>(codepoint);
  } else if (codepoint <= 0x7FF) {
    result += static_cast<char>(0xC0 | (codepoint >> 6));
    result += static_cast<char>(0x80 | (codepoint & 0x3F));
  } else if (codepoint <= 0xFFFF) {
    result += static_cast<char>(0xE0 | (codepoint >> 12));
    result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (codepoint & 0x3F));
  } else {
    result += static_cast<char>(0xF0 | (codepoint >> 18));
    result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
    result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (codepoint & 0x3F));
  }

  return result;
}

// ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
unordered_map<unsigned char, string> createByteEncoder() {
  unordered_map<unsigned char, string> encoder;

  vector<int> bytes;
  vector<int> unicode;

  // bytes that gpt-2 represents directly.
  for (int i = '!'; i <= '~'; i++) {
    bytes.push_back(i);
  }

  for (int i = 0xA1; i <= 0xAC; i++) {
    bytes.push_back(i);
  }

  for (int i = 0xAE; i <= 0xFF; i++) {
    bytes.push_back(i);
  }

  unicode = bytes;

  // add remaining byte values
  int extra = 0;

  for (int i = 0; i < 256; i++) {
    bool exists = false;

    for (int b : bytes) {
      if (b == i) {
        exists = true;
        break;
      }
    }

    if (!exists) {
      bytes.push_back(i);
      unicode.push_back(256 + extra);
      extra++;
    }
  }

  // byte to utf8 character mapping
  for (int i = 0; i < bytes.size(); i++) {
    encoder[static_cast<unsigned char>(bytes[i])] = utf8Encode(unicode[i]);
  }

  return encoder;
}

string byteEncode(const string &text) {
  auto encoder = createByteEncoder();

  string result;
  for (unsigned char byte : text) {
    result += encoder[byte];
  }

  return result;
}

// reverse of byteEncode: turn the GPT-2 byte-level unicode string (Ġ, Ċ,
// etc.) back into the original bytes.
string byteDecode(const string &text) {
  auto encoder = createByteEncoder();

  unordered_map<string, unsigned char> reverseEncoder;
  for (const auto &[byte, encoded] : encoder) {
    reverseEncoder[encoded] = byte;
  }

  string result;

  for (size_t i = 0; i < text.size();) {
    unsigned char c = text[i];
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

    string codepoint = text.substr(i, length);

    auto it = reverseEncoder.find(codepoint);
    if (it != reverseEncoder.end()) {
      result += static_cast<char>(it->second);
    }

    i += length;
  }

  return result;
}
