# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Rust library implementing exponential histograms for efficient statistical data collection and analysis.

## Build Commands

```bash
# Build the project
cargo build

# Build for release
cargo build --release

# Run tests
cargo test

# Run a specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture

# Format code
cargo fmt

# Lint with clippy
cargo clippy

# Generate documentation
cargo doc --open
```

## Architecture

This is a library crate that wraps the `histogram` crate to provide an API-compatible replacement for the `exponential_histogram` crate. The project uses Rust 2024 edition.

### Key Components

- **ExponentialHistogram**: Pure stateless wrapper around two `histogram::Histogram` instances
  - Uses separate histograms for positive and negative values
  - **Zero cached state** - all statistics computed on-demand from buckets
  - Every query method (count, sum, min, max) iterates buckets
  - Trade-off: slightly slower reads for zero memory overhead and perfect consistency
  - Full support for negative values

- **SharedExponentialHistogram**: Pure stateless wrapper around two `histogram::AtomicHistogram` instances
  - Uses separate atomic histograms for positive and negative values
  - **Zero additional state** - just wraps two AtomicHistograms directly
  - All statistics (count, sum, min, max) computed at read-time from histogram buckets
  - Lock-free implementation - can be shared across threads via `Arc`
  - Extremely efficient accumulate() - single atomic increment, no additional counters
  - Guarantees consistency since all stats come from the same snapshot
  - Full support for negative values

### Implementation Notes

- Both wrappers use dual histograms (positive and negative) for proper sign handling
- The wrapper maintains API compatibility while using `histogram` crate internally
- Scale parameter is mapped to grouping_power (capped at 7)
- All values (including zero and fractional values) are stored in histogram buckets
- Positive values go to the positive histogram, negative values go to the negative histogram
- Statistics are derived from bucket iteration at read-time:
  - Count: sum of all bucket counts from both histograms
  - Sum: approximate using bucket midpoints (positive buckets add, negative buckets subtract)
  - Min: most negative value from negative histogram, or smallest positive value
  - Max: largest positive value from positive histogram, or smallest negative value
  - has_negatives(): checks if negative histogram has any data
- Performance trade-off: reads iterate buckets, but writes are minimal and consistency is guaranteed
