#pragma once
#include <string>
#include <unordered_map>

std::string utf8Encode(int codepoint);
std::unordered_map<unsigned char, std::string> createByteEncoder();
std::string byteEncode(const std::string &text);
std::string byteDecode(const std::string &text);
