// SPDX-License-Identifier: MIT
#include "embedding/workflow.h"

#include <exception>
#include <iostream>

int main(int argc, char **argv) {
  try {
    return embedding::run_cli(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "embedding: " << error.what() << '\n';
    return 1;
  }
}
