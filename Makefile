CXX := g++
LIBTORCH := $(CURDIR)/.external/libtorch
CPPFLAGS := -Iinclude -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$$ORIGIN/../.external/libtorch/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread

.PHONY: all test baseline smoke smoke-cuda
all: .build/embedding

.build/embedding: .build/main.o .build/workflow.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

.build/%.o: src/%.cpp | .build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.build/model_test: tests/model_test.cpp | .build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

.build/workflow_test: .build/workflow_test.o .build/workflow.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

.build/workflow_test.o: tests/workflow_test.cpp | .build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.build/baseline_test: tests/baseline_test.cpp .build/reference/reference_model.h | .build
	$(CXX) $(CPPFLAGS) -I.build/reference $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

test: .build/model_test .build/workflow_test
	.build/model_test
	.build/workflow_test

# Optional extraction audit; the old header is a local test fixture, never a runtime dependency.
baseline: .build/baseline_test
	.build/baseline_test

smoke: .build/embedding
	bash tests/cli_smoke.sh cpu

smoke-cuda: .build/embedding
	bash tests/cli_smoke.sh cuda

.build:
	mkdir -p $@

-include $(wildcard .build/*.d)
