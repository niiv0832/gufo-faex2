// vision_encode_dump MMPROJ WIDTH IMAGE OUT.bin [REPEAT] — encodes one image with
// the Qwen vision tower, writes the merged embedding (float32, rows x width) and
// prints the warm encode time. Used to compare encoder variants end to end.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/qwen/vision/encoder.hpp"
#include "src/models/qwen/vision/prompt.hpp"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: vision_encode_dump MMPROJ WIDTH IMAGE OUT.bin [REPEAT]\n";
    return 2;
  }
  try {
    namespace vision = gufo::models::qwen::vision;
    const auto width = static_cast<std::uint32_t>(std::stoul(argv[2]));
    const int repeat = argc > 5 ? std::stoi(argv[5]) : 3;
    const auto bytes = gufo::core::ReadImageFile(argv[3]);
    const auto image = vision::ResizeImage(gufo::core::DecodeImage(bytes));
    vision::Encoder encoder(argv[1], width);
    encoder.Upload();
    std::shared_ptr<const vision::Encoder::Embedding> embedding;
    double best = 1e30;
    for (int i = 0; i <= repeat; ++i) {  // the first pass warms kernels up
      const auto start = std::chrono::steady_clock::now();
      embedding = encoder.Encode(image);
      if (hipDeviceSynchronize() != hipSuccess)
        throw std::runtime_error("device synchronization failed");
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
      if (i > 0 && ms < best)
        best = ms;
    }
    std::vector<float> host(std::size_t{embedding->rows()} * embedding->width());
    if (hipMemcpy(host.data(), embedding->data(), host.size() * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
      throw std::runtime_error("embedding download failed");
    std::ofstream(argv[4], std::ios::binary)
        .write(reinterpret_cast<const char*>(host.data()),
               static_cast<std::streamsize>(host.size() * sizeof(float)));
    std::cout << "image " << image.width << "x" << image.height << " rows "
              << embedding->rows() << " width " << embedding->width()
              << " encode_ms " << best << "\n";
  } catch (const std::exception& e) {
    std::cerr << "vision_encode_dump: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
