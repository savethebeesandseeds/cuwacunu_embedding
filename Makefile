CXX := g++
CODE_ROOT := code
ENCODER_ROOT := $(CODE_ROOT)/encoders/mtf_jepa_mae_vicreg
BUILD_DIR ?= /opt/cuwacunu_embedding/build/baseline
OBJECT_DIR := $(BUILD_DIR)/code
LIBTORCH ?= $(CURDIR)/.external/libtorch
REFERENCE_DIR ?= $(CURDIR)/.build/reference
RUN_ROOT ?= $(CURDIR)/output/runs/baseline
BIN := $(BUILD_DIR)/embedding
COMMON_CPPFLAGS := -I$(CODE_ROOT)/shared/include -I$(CODE_ROOT)/shared/tests -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPPFLAGS := -I$(ENCODER_ROOT)/include -I$(ENCODER_ROOT)/tests $(COMMON_CPPFLAGS)
CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$(abspath $(LIBTORCH))/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread

SHARED_OBJECTS := $(OBJECT_DIR)/shared/data.o $(OBJECT_DIR)/shared/evaluation.o
ENCODER_OBJECTS := $(OBJECT_DIR)/encoders/mtf_jepa_mae_vicreg/workflow.o $(OBJECT_DIR)/encoders/mtf_jepa_mae_vicreg/evaluation.o
ENCODER_TEST_DIR := $(OBJECT_DIR)/tests/encoders/mtf_jepa_mae_vicreg
SHARED_TEST_DIR := $(OBJECT_DIR)/tests/shared
ENCODER_TESTS := $(addprefix $(ENCODER_TEST_DIR)/,model_test workflow_test numerics_test masking_test evaluation_test)
SHARED_TESTS := $(addprefix $(SHARED_TEST_DIR)/,data_test objectives_test evaluation_test)

.PHONY: all test baseline smoke smoke-cuda evaluate periodic-check
all: $(BIN)

$(BIN): $(OBJECT_DIR)/encoders/mtf_jepa_mae_vicreg/main.o $(SHARED_OBJECTS) $(ENCODER_OBJECTS)
	mkdir -p "$(@D)"
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

# Shared code compiles without encoder include directories.
$(OBJECT_DIR)/shared/%.o: $(CODE_ROOT)/shared/src/%.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OBJECT_DIR)/encoders/mtf_jepa_mae_vicreg/%.o: $(ENCODER_ROOT)/src/%.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(ENCODER_TEST_DIR)/%_test: $(ENCODER_ROOT)/tests/%_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(ENCODER_TEST_DIR)/%_test.o: $(ENCODER_ROOT)/tests/%_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(ENCODER_TEST_DIR)/workflow_test: $(ENCODER_TEST_DIR)/workflow_test.o $(SHARED_OBJECTS) $(ENCODER_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(ENCODER_TEST_DIR)/evaluation_test: $(ENCODER_TEST_DIR)/evaluation_test.o $(SHARED_OBJECTS) $(ENCODER_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(ENCODER_TEST_DIR)/baseline_test: $(ENCODER_ROOT)/tests/baseline_test.cpp $(REFERENCE_DIR)/reference_model.h
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) -I$(REFERENCE_DIR) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/objectives_test: $(CODE_ROOT)/shared/tests/objectives_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/evaluation_test.o: $(CODE_ROOT)/shared/tests/evaluation_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/evaluation_test: $(SHARED_TEST_DIR)/evaluation_test.o $(SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/data_test.o: $(CODE_ROOT)/shared/tests/data_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/data_test: $(SHARED_TEST_DIR)/data_test.o $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test: $(ENCODER_TESTS) $(SHARED_TESTS)
	$(ENCODER_TEST_DIR)/model_test
	$(ENCODER_TEST_DIR)/workflow_test
	$(ENCODER_TEST_DIR)/numerics_test
	$(ENCODER_TEST_DIR)/masking_test
	$(ENCODER_TEST_DIR)/evaluation_test
	$(SHARED_TEST_DIR)/data_test
	$(SHARED_TEST_DIR)/objectives_test
	$(SHARED_TEST_DIR)/evaluation_test

baseline: $(ENCODER_TEST_DIR)/baseline_test
	$(ENCODER_TEST_DIR)/baseline_test

smoke: $(BIN)
	EMBEDDING_BIN="$(abspath $(BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(ENCODER_ROOT)/tests/cli_smoke.sh cpu

smoke-cuda: $(BIN)
	EMBEDDING_BIN="$(abspath $(BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(ENCODER_ROOT)/tests/cli_smoke.sh cuda

evaluate: $(BIN)
	EMBEDDING_BIN="$(abspath $(BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(ENCODER_ROOT)/scripts/evaluate.sh

periodic-check: $(BIN)
	EMBEDDING_BIN="$(abspath $(BIN))" bash $(ENCODER_ROOT)/tests/periodic_checkpoint.sh

# Build products use their own code/ namespace so retained dependencies from
# earlier layouts cannot refer to source paths that have moved.
-include $(wildcard $(OBJECT_DIR)/shared/*.d $(OBJECT_DIR)/encoders/mtf_jepa_mae_vicreg/*.d $(ENCODER_TEST_DIR)/*.d $(SHARED_TEST_DIR)/*.d)

# Independent model and evaluation boundaries. Baseline targets above retain
# their historical evaluator; ordinary RPB train/embed links neither evaluator.
RPB_ROOT := $(CODE_ROOT)/encoders/raw_patch_bottleneck_mae
RPB_OBJECT_DIR := $(OBJECT_DIR)/encoders/raw_patch_bottleneck_mae
RPB_TEST_DIR := $(OBJECT_DIR)/tests/encoders/raw_patch_bottleneck_mae
RPB_BIN ?= $(BUILD_DIR)/embedding_raw_patch_bottleneck_mae
EVALUATION_ROOT := $(CODE_ROOT)/evaluation
EVALUATION_OBJECT_DIR := $(OBJECT_DIR)/evaluation
EVALUATION_BIN ?= $(BUILD_DIR)/embedding_evaluate
HARNESS_BIN ?= $(BUILD_DIR)/feature_harness

RPB_CORE_HEADERS := $(filter-out $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/evaluation_adapter.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/reconstruction_adapter.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/native_curve_gate.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/paired_pooling_adapter.h,$(wildcard $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/*.h))
RPB_PROVENANCE_INPUTS := $(sort $(RPB_CORE_HEADERS) $(RPB_ROOT)/src/workflow.cpp $(RPB_ROOT)/src/main.cpp $(wildcard $(RPB_ROOT)/config/*.conf) $(CODE_ROOT)/shared/include/embedding/shared/data.h $(CODE_ROOT)/shared/include/embedding/shared/types.h $(CODE_ROOT)/shared/include/embedding/shared/tensor_ops.h $(CODE_ROOT)/shared/src/data.cpp Makefile dependencies.lock)
RPB_SOURCE_ID := $(shell sha256sum $(RPB_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
SOURCE_GIT_HEAD := $(shell git -c safe.directory=$(CURDIR) rev-parse HEAD 2>/dev/null || printf unrecorded)
SOURCE_GIT_DIRTY := $(shell if source_status=$$(git -c safe.directory=$(CURDIR) status --porcelain 2>/dev/null); then if test -n "$$source_status"; then printf dirty; else printf clean; fi; else printf unrecorded; fi)
RPB_CPPFLAGS := -I$(RPB_ROOT)/include -I$(RPB_ROOT)/tests $(COMMON_CPPFLAGS) -DRPB_SOURCE_ID=\"$(RPB_SOURCE_ID)\" -DRPB_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DRPB_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"

MINIMUM_PROVENANCE_INPUTS := $(sort $(wildcard $(CODE_ROOT)/shared/include/embedding/shared/*.h $(CODE_ROOT)/shared/src/*.cpp $(ENCODER_ROOT)/include/embedding/encoders/mtf_jepa_mae_vicreg/*.h $(ENCODER_ROOT)/src/*.cpp $(ENCODER_ROOT)/config/*.conf $(EVALUATION_ROOT)/src/*.cpp $(EVALUATION_ROOT)/include/*.h) Makefile dependencies.lock)
EVALUATION_PROVENANCE_INPUTS := $(sort $(MINIMUM_PROVENANCE_INPUTS) $(RPB_PROVENANCE_INPUTS) $(RPB_ROOT)/src/evaluation_adapter.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/evaluation_adapter.h $(RPB_ROOT)/src/reconstruction_adapter.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/reconstruction_adapter.h $(RPB_ROOT)/src/learning_curve_adapter.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h)
MINIMUM_SOURCE_ID := $(shell sha256sum $(MINIMUM_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
EVALUATION_SOURCE_ID := $(shell sha256sum $(EVALUATION_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
EVALUATION_PROVENANCE_CPPFLAGS := -DEVALUATION_SOURCE_ID=\"$(EVALUATION_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
EVALUATION_CPPFLAGS := -I$(ENCODER_ROOT)/include $(COMMON_CPPFLAGS) $(EVALUATION_PROVENANCE_CPPFLAGS)
MINIMUM_CPPFLAGS := -I$(ENCODER_ROOT)/include $(COMMON_CPPFLAGS) -DEVALUATION_SOURCE_ID=\"$(MINIMUM_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"

HARNESS_OBJECT := $(OBJECT_DIR)/shared/feature_harness.o
FEATURE_EVALUATION_OBJECT := $(OBJECT_DIR)/shared/feature_evaluation.o
FEATURE_STRESS_OBJECT := $(OBJECT_DIR)/shared/feature_stress.o
EVALUATION_COMMON_OBJECTS := $(SHARED_OBJECTS) $(HARNESS_OBJECT) $(FEATURE_EVALUATION_OBJECT) $(FEATURE_STRESS_OBJECT)
RECONSTRUCTION_OBJECT := $(OBJECT_DIR)/shared/reconstruction_evaluation.o
RPB_WORKFLOW_OBJECT := $(RPB_OBJECT_DIR)/workflow.o
RPB_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/evaluation_adapter.o
RPB_RECONSTRUCTION_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/reconstruction_adapter.o
BASELINE_ADAPTER_OBJECT := $(EVALUATION_OBJECT_DIR)/baseline_adapter.o
MINIMUM_ADAPTER_OBJECT := $(EVALUATION_OBJECT_DIR)/minimum_baseline_adapter.o
RPB_CORE_TESTS := $(addprefix $(RPB_TEST_DIR)/,preprocessing_test numerics_test masking_test model_test)

.PHONY: rpb-mae evaluation feature-harness test-rpb-core test-rpb-mae test-rpb-reconstruction test-rpb-feature-adapter test-feature-harness test-feature-evaluation test-feature-stress test-reconstruction-evaluation smoke-rpb-mae smoke-rpb-mae-cuda periodic-rpb-mae evaluate-minimum evaluate-rpb-mae reconstruct-rpb-mae
rpb-mae: $(RPB_BIN)
evaluation: $(EVALUATION_BIN)
feature-harness: $(HARNESS_BIN)

$(RPB_OBJECT_DIR)/%.o: $(RPB_ROOT)/src/%.cpp
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_WORKFLOW_OBJECT) $(RPB_OBJECT_DIR)/main.o: $(RPB_PROVENANCE_INPUTS)

$(RPB_BIN): $(RPB_OBJECT_DIR)/main.o $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	mkdir -p "$(@D)"
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(EVALUATION_OBJECT_DIR)/main.o: $(EVALUATION_ROOT)/src/main.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) -I$(EVALUATION_ROOT)/include -I$(RPB_ROOT)/include -DEMBEDDING_WITH_RPB=1 $(CXXFLAGS) -c $< -o $@

$(EVALUATION_OBJECT_DIR)/minimum_main.o: $(EVALUATION_ROOT)/src/main.cpp $(MINIMUM_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MINIMUM_CPPFLAGS) -I$(EVALUATION_ROOT)/include -DEMBEDDING_WITH_RPB=0 $(CXXFLAGS) -c $< -o $@

$(EVALUATION_OBJECT_DIR)/reconstruction_main.o: $(EVALUATION_ROOT)/src/reconstruction_main.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) -I$(EVALUATION_ROOT)/include -I$(RPB_ROOT)/include -DEMBEDDING_WITH_RPB=1 $(CXXFLAGS) -c $< -o $@

$(EVALUATION_OBJECT_DIR)/minimum_reconstruction_main.o: $(EVALUATION_ROOT)/src/reconstruction_main.cpp $(MINIMUM_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MINIMUM_CPPFLAGS) -I$(EVALUATION_ROOT)/include -DEMBEDDING_WITH_RPB=0 $(CXXFLAGS) -c $< -o $@

$(BASELINE_ADAPTER_OBJECT): $(ENCODER_ROOT)/src/evaluation_adapter.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(MINIMUM_ADAPTER_OBJECT): $(ENCODER_ROOT)/src/evaluation_adapter.cpp $(MINIMUM_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MINIMUM_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_ADAPTER_OBJECT): $(RPB_ROOT)/src/evaluation_adapter.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(EVALUATION_PROVENANCE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_RECONSTRUCTION_ADAPTER_OBJECT): $(RPB_ROOT)/src/reconstruction_adapter.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(EVALUATION_PROVENANCE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(EVALUATION_BIN): $(EVALUATION_OBJECT_DIR)/main.o $(EVALUATION_OBJECT_DIR)/reconstruction_main.o $(EVALUATION_COMMON_OBJECTS) $(RECONSTRUCTION_OBJECT) $(ENCODER_OBJECTS) $(BASELINE_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_RECONSTRUCTION_ADAPTER_OBJECT)
	mkdir -p "$(@D)"
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(HARNESS_BIN): $(EVALUATION_OBJECT_DIR)/minimum_main.o $(EVALUATION_OBJECT_DIR)/minimum_reconstruction_main.o $(EVALUATION_COMMON_OBJECTS) $(ENCODER_OBJECTS) $(MINIMUM_ADAPTER_OBJECT)
	mkdir -p "$(@D)"
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/%_test.o: $(RPB_ROOT)/tests/%_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/%_test: $(RPB_TEST_DIR)/%_test.o $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/workflow_test: $(RPB_TEST_DIR)/workflow_test.o $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/reconstruction_adapter_test: $(RPB_TEST_DIR)/reconstruction_adapter_test.o $(RPB_RECONSTRUCTION_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/evaluation_adapter_test: $(RPB_TEST_DIR)/evaluation_adapter_test.o $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/feature_harness_test.o: $(CODE_ROOT)/shared/tests/feature_harness_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/feature_harness_test: $(SHARED_TEST_DIR)/feature_harness_test.o $(SHARED_OBJECTS) $(HARNESS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/feature_evaluation_test.o: $(CODE_ROOT)/shared/tests/feature_evaluation_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/feature_evaluation_test: $(SHARED_TEST_DIR)/feature_evaluation_test.o $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/feature_stress_test.o: $(CODE_ROOT)/shared/tests/feature_stress_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/feature_stress_test: $(SHARED_TEST_DIR)/feature_stress_test.o $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/reconstruction_evaluation_test.o: $(CODE_ROOT)/shared/tests/reconstruction_evaluation_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/reconstruction_evaluation_test: $(SHARED_TEST_DIR)/reconstruction_evaluation_test.o $(RECONSTRUCTION_OBJECT) $(SHARED_OBJECTS) $(HARNESS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-core: $(RPB_CORE_TESTS)
	$(RPB_TEST_DIR)/preprocessing_test
	$(RPB_TEST_DIR)/numerics_test
	$(RPB_TEST_DIR)/masking_test
	$(RPB_TEST_DIR)/model_test

test-rpb-mae: test-rpb-core $(RPB_TEST_DIR)/workflow_test
	$(RPB_TEST_DIR)/workflow_test

test-rpb-reconstruction: $(RPB_TEST_DIR)/reconstruction_adapter_test
	$(RPB_TEST_DIR)/reconstruction_adapter_test

test-rpb-feature-adapter: $(RPB_TEST_DIR)/evaluation_adapter_test
	$(RPB_TEST_DIR)/evaluation_adapter_test

test-feature-evaluation: $(SHARED_TEST_DIR)/feature_evaluation_test
	$(SHARED_TEST_DIR)/feature_evaluation_test

test-feature-stress: $(SHARED_TEST_DIR)/feature_stress_test
	$(SHARED_TEST_DIR)/feature_stress_test

test-feature-harness: $(SHARED_TEST_DIR)/feature_harness_test test-feature-evaluation test-feature-stress
	$(SHARED_TEST_DIR)/feature_harness_test

test-reconstruction-evaluation: $(SHARED_TEST_DIR)/reconstruction_evaluation_test
	$(SHARED_TEST_DIR)/reconstruction_evaluation_test

smoke-rpb-mae: $(RPB_BIN)
	EMBEDDING_BIN="$(abspath $(RPB_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(RPB_ROOT)/tests/cli_smoke.sh cpu

smoke-rpb-mae-cuda: $(RPB_BIN)
	EMBEDDING_BIN="$(abspath $(RPB_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(RPB_ROOT)/tests/cli_smoke.sh cuda

periodic-rpb-mae: $(RPB_BIN)
	EMBEDDING_BIN="$(abspath $(RPB_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(RPB_ROOT)/tests/periodic_checkpoint.sh

evaluate-minimum: $(HARNESS_BIN)
	EMBEDDING_BIN="$(abspath $(HARNESS_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(CODE_ROOT)/scripts/evaluate-minimum.sh

evaluate-rpb-mae: $(EVALUATION_BIN)
	EVALUATION_BIN="$(abspath $(EVALUATION_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(CODE_ROOT)/scripts/evaluate-rpb-mae.sh

reconstruct-rpb-mae: $(EVALUATION_BIN)
	EVALUATION_BIN="$(abspath $(EVALUATION_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(CODE_ROOT)/scripts/reconstruct-rpb-mae.sh

-include $(wildcard $(RPB_OBJECT_DIR)/*.d $(RPB_TEST_DIR)/*.d $(EVALUATION_OBJECT_DIR)/*.d)

# Learning-curve selection/scoring stays shared; only training and compact
# reconstruction live in the RPB adapter. Existing executables retain their CLI.
LEARNING_CURVE_BIN := $(BUILD_DIR)/embedding_learning_curve
LEARNING_CURVE_OBJECT := $(OBJECT_DIR)/shared/learning_curve.o
RPB_CURVE_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/learning_curve_adapter.o
.PHONY: learning-curve test-learning-curve test-rpb-learning-curve
learning-curve: $(LEARNING_CURVE_BIN)

$(EVALUATION_OBJECT_DIR)/learning_curve_main.o: $(EVALUATION_ROOT)/src/learning_curve_main.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) -I$(RPB_ROOT)/include $(CXXFLAGS) -c $< -o $@

$(RPB_CURVE_ADAPTER_OBJECT): $(RPB_ROOT)/src/learning_curve_adapter.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(EVALUATION_PROVENANCE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(LEARNING_CURVE_BIN): $(EVALUATION_OBJECT_DIR)/learning_curve_main.o $(LEARNING_CURVE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/learning_curve_test.o: $(CODE_ROOT)/shared/tests/learning_curve_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/learning_curve_test: $(SHARED_TEST_DIR)/learning_curve_test.o $(LEARNING_CURVE_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/learning_curve_adapter_test: $(RPB_TEST_DIR)/learning_curve_adapter_test.o $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-learning-curve: $(SHARED_TEST_DIR)/learning_curve_test
	$(SHARED_TEST_DIR)/learning_curve_test

test-rpb-learning-curve: $(RPB_TEST_DIR)/learning_curve_adapter_test
	$(RPB_TEST_DIR)/learning_curve_adapter_test

# Frozen projection diagnostics link only shared math/archive code.
PROJECTION_DIAGNOSTIC_BIN := $(BUILD_DIR)/embedding_projection_diagnostic
PROJECTION_DIAGNOSTIC_OBJECT := $(OBJECT_DIR)/shared/projection_diagnostic.o
GLOBAL_BOTTLENECK_BIN := $(BUILD_DIR)/embedding_global_bottleneck
GLOBAL_BOTTLENECK_OBJECT := $(OBJECT_DIR)/shared/global_bottleneck_experiment.o
.PHONY: projection-diagnostic test-projection-diagnostic global-bottleneck test-global-bottleneck
projection-diagnostic: $(PROJECTION_DIAGNOSTIC_BIN)
global-bottleneck: $(GLOBAL_BOTTLENECK_BIN)

$(EVALUATION_OBJECT_DIR)/projection_diagnostic_main.o: $(EVALUATION_ROOT)/src/projection_diagnostic_main.cpp $(MINIMUM_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(EVALUATION_PROVENANCE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(EVALUATION_OBJECT_DIR)/global_bottleneck_main.o: $(EVALUATION_ROOT)/src/global_bottleneck_main.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) -I$(RPB_ROOT)/include $(CXXFLAGS) -c $< -o $@

$(PROJECTION_DIAGNOSTIC_BIN): $(EVALUATION_OBJECT_DIR)/projection_diagnostic_main.o $(PROJECTION_DIAGNOSTIC_OBJECT) $(OBJECT_DIR)/shared/feature_harness.o $(SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(GLOBAL_BOTTLENECK_BIN): $(EVALUATION_OBJECT_DIR)/global_bottleneck_main.o $(GLOBAL_BOTTLENECK_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/projection_diagnostic_test.o: $(CODE_ROOT)/shared/tests/projection_diagnostic_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/projection_diagnostic_test: $(SHARED_TEST_DIR)/projection_diagnostic_test.o $(PROJECTION_DIAGNOSTIC_OBJECT) $(OBJECT_DIR)/shared/feature_harness.o $(SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/global_bottleneck_experiment_test.o: $(CODE_ROOT)/shared/tests/global_bottleneck_experiment_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/global_bottleneck_experiment_test: $(SHARED_TEST_DIR)/global_bottleneck_experiment_test.o $(GLOBAL_BOTTLENECK_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-projection-diagnostic: $(SHARED_TEST_DIR)/projection_diagnostic_test
	$(SHARED_TEST_DIR)/projection_diagnostic_test

test-global-bottleneck: $(SHARED_TEST_DIR)/global_bottleneck_experiment_test
	$(SHARED_TEST_DIR)/global_bottleneck_experiment_test

.PHONY: test-rpb-legacy-checkpoint
$(RPB_TEST_DIR)/legacy_checkpoint_parity_test: $(RPB_TEST_DIR)/legacy_checkpoint_parity_test.o $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-legacy-checkpoint: $(RPB_TEST_DIR)/legacy_checkpoint_parity_test
	$(RPB_TEST_DIR)/legacy_checkpoint_parity_test

# Native archive controls use shared fitting only; no encoder is linked.
ARCHIVE_READOUT_BIN := $(BUILD_DIR)/embedding_archive_readout
ARCHIVE_READOUT_OBJECT := $(OBJECT_DIR)/shared/archive_readout.o
ARCHIVE_READOUT_PROVENANCE_INPUTS := $(sort $(wildcard $(CODE_ROOT)/shared/include/embedding/shared/*.h $(CODE_ROOT)/shared/src/*.cpp) $(EVALUATION_ROOT)/src/archive_readout_main.cpp $(EVALUATION_ROOT)/cards/archive_readout_v1.md Makefile dependencies.lock)
ARCHIVE_READOUT_SOURCE_ID := $(shell sha256sum $(ARCHIVE_READOUT_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
ARCHIVE_READOUT_CPPFLAGS := $(COMMON_CPPFLAGS) -DEVALUATION_SOURCE_ID=\"$(ARCHIVE_READOUT_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
.PHONY: archive-readout test-archive-readout evaluate-archive
archive-readout: $(ARCHIVE_READOUT_BIN)

$(EVALUATION_OBJECT_DIR)/archive_readout_main.o: $(EVALUATION_ROOT)/src/archive_readout_main.cpp $(ARCHIVE_READOUT_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(ARCHIVE_READOUT_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(ARCHIVE_READOUT_BIN): $(EVALUATION_OBJECT_DIR)/archive_readout_main.o $(ARCHIVE_READOUT_OBJECT) $(HARNESS_OBJECT) $(SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/archive_readout_test.o: $(CODE_ROOT)/shared/tests/archive_readout_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/archive_readout_test: $(SHARED_TEST_DIR)/archive_readout_test.o $(ARCHIVE_READOUT_OBJECT) $(HARNESS_OBJECT) $(SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-archive-readout: $(SHARED_TEST_DIR)/archive_readout_test
	$(SHARED_TEST_DIR)/archive_readout_test

evaluate-archive: $(ARCHIVE_READOUT_BIN)
	ARCHIVE_READOUT_BIN="$(abspath $(ARCHIVE_READOUT_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(CODE_ROOT)/scripts/evaluate-archive.sh $(ARCHIVE_ARGS)

# Native-only curves select and score the served export. Historical curve
# executables/cards above retain their projection-based protocol semantics.
NATIVE_CURVE_BIN := $(BUILD_DIR)/embedding_native_curve
NATIVE_CURVE_OBJECT := $(OBJECT_DIR)/shared/native_curve.o
RPB_NATIVE_GATE_OBJECT := $(RPB_OBJECT_DIR)/native_curve_gate.o
NATIVE_CURVE_PROVENANCE_INPUTS := $(sort $(wildcard $(CODE_ROOT)/shared/include/embedding/shared/*.h $(CODE_ROOT)/shared/src/*.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/*.h) $(RPB_ROOT)/src/workflow.cpp $(RPB_ROOT)/src/evaluation_adapter.cpp $(RPB_ROOT)/src/learning_curve_adapter.cpp $(RPB_ROOT)/src/native_curve_gate.cpp $(RPB_ROOT)/config/learned_global.conf $(EVALUATION_ROOT)/src/native_curve_main.cpp $(EVALUATION_ROOT)/cards/native_curve_v1.md $(CODE_ROOT)/scripts/evaluate-native-curve.sh Makefile dependencies.lock)
NATIVE_CURVE_SOURCE_ID := $(shell sha256sum $(NATIVE_CURVE_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
NATIVE_CURVE_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(NATIVE_CURVE_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
.PHONY: native-curve test-native-curve test-rpb-native-gate evaluate-native-curve
native-curve: $(NATIVE_CURVE_BIN)

$(EVALUATION_OBJECT_DIR)/native_curve_main.o: $(EVALUATION_ROOT)/src/native_curve_main.cpp $(NATIVE_CURVE_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(NATIVE_CURVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_NATIVE_GATE_OBJECT): $(RPB_ROOT)/src/native_curve_gate.cpp $(NATIVE_CURVE_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(NATIVE_CURVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(NATIVE_CURVE_BIN): $(EVALUATION_OBJECT_DIR)/native_curve_main.o $(NATIVE_CURVE_OBJECT) $(RPB_NATIVE_GATE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/native_curve_test.o: $(CODE_ROOT)/shared/tests/native_curve_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/native_curve_test: $(SHARED_TEST_DIR)/native_curve_test.o $(NATIVE_CURVE_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/native_curve_gate_test: $(RPB_TEST_DIR)/native_curve_gate_test.o $(RPB_NATIVE_GATE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-native-curve: $(SHARED_TEST_DIR)/native_curve_test
	$(SHARED_TEST_DIR)/native_curve_test

test-rpb-native-gate: $(RPB_TEST_DIR)/native_curve_gate_test
	$(RPB_TEST_DIR)/native_curve_gate_test

evaluate-native-curve: $(NATIVE_CURVE_BIN)
	NATIVE_CURVE_BIN="$(abspath $(NATIVE_CURVE_BIN))" NATIVE_CURVE_SOURCE_INPUTS="$(NATIVE_CURVE_PROVENANCE_INPUTS)" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(CODE_ROOT)/scripts/evaluate-native-curve.sh

# One pooling candidate versus immutable retained native-only reference assets.
PAIRED_POOLING_BIN := $(BUILD_DIR)/embedding_paired_pooling
PAIRED_POOLING_OBJECT := $(OBJECT_DIR)/shared/paired_pooling.o
RPB_PAIRED_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/paired_pooling_adapter.o
RPB_PAIRED_GATE_OBJECT := $(RPB_OBJECT_DIR)/paired_pooling_gate.o
PAIRED_POOLING_PROVENANCE_INPUTS := $(sort $(NATIVE_CURVE_PROVENANCE_INPUTS) $(RPB_ROOT)/src/paired_pooling_adapter.cpp $(RPB_ROOT)/config/learned_patch_global.conf $(EVALUATION_ROOT)/src/paired_pooling_main.cpp $(EVALUATION_ROOT)/cards/paired_pooling_v1.md $(CODE_ROOT)/scripts/evaluate-paired-pooling.sh)
PAIRED_POOLING_SOURCE_ID := $(shell sha256sum $(PAIRED_POOLING_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
PAIRED_POOLING_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(PAIRED_POOLING_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
.PHONY: paired-pooling test-paired-pooling evaluate-paired-pooling
paired-pooling: $(PAIRED_POOLING_BIN)

$(EVALUATION_OBJECT_DIR)/paired_pooling_main.o: $(EVALUATION_ROOT)/src/paired_pooling_main.cpp $(PAIRED_POOLING_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(PAIRED_POOLING_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_PAIRED_ADAPTER_OBJECT): $(RPB_ROOT)/src/paired_pooling_adapter.cpp $(PAIRED_POOLING_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(PAIRED_POOLING_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_PAIRED_GATE_OBJECT): $(RPB_ROOT)/src/native_curve_gate.cpp $(PAIRED_POOLING_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(PAIRED_POOLING_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(PAIRED_POOLING_BIN): $(EVALUATION_OBJECT_DIR)/paired_pooling_main.o $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(RPB_PAIRED_ADAPTER_OBJECT) $(RPB_PAIRED_GATE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/paired_pooling_test.o: $(CODE_ROOT)/shared/tests/paired_pooling_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/paired_pooling_test: $(SHARED_TEST_DIR)/paired_pooling_test.o $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-paired-pooling: $(SHARED_TEST_DIR)/paired_pooling_test
	$(SHARED_TEST_DIR)/paired_pooling_test

evaluate-paired-pooling: $(PAIRED_POOLING_BIN)
	PAIRED_POOLING_BIN="$(abspath $(PAIRED_POOLING_BIN))" PAIRED_POOLING_SOURCE_INPUTS="$(PAIRED_POOLING_PROVENANCE_INPUTS)" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(CODE_ROOT)/scripts/evaluate-paired-pooling.sh
