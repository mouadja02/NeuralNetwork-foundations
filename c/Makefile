CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math -Iinclude
LDFLAGS = -lm

SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build
INCLUDE_DIR = include

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Test files
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.c)
TEST_EXECUTABLES = $(TEST_SOURCES:$(TEST_DIR)/%.c=$(BUILD_DIR)/%)

# Default target
all: $(BUILD_DIR) $(TEST_EXECUTABLES)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Build test executables
$(BUILD_DIR)/test_%: $(TEST_DIR)/test_%.c $(OBJECTS)
	$(CC) $(CFLAGS) $< $(OBJECTS) -o $@ $(LDFLAGS)

# Run tests
test: $(TEST_EXECUTABLES)
	@for test in $(TEST_EXECUTABLES); do \
		echo ""; \
		$$test; \
	done

# Benchmark
benchmark: $(BUILD_DIR)/benchmark
	$(BUILD_DIR)/benchmark

$(BUILD_DIR)/benchmark: benchmarks/benchmark.c $(OBJECTS)
	$(CC) $(CFLAGS) $< $(OBJECTS) -o $@ $(LDFLAGS)

# Clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all test benchmark clean