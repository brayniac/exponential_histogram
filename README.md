# exponential_histogram

A wrapper around the `histogram` crate that provides an API-compatible replacement for the `exponential_histogram` crate.

## Purpose

This crate wraps the high-performance `histogram` crate to provide the same public API as `exponential_histogram`, allowing you to drop it in as a replacement using Cargo's override mechanism.

## Implementation Details

Both histogram types are **pure wrappers** with zero additional state:

- **ExponentialHistogram**: Pure wrapper around two `histogram::Histogram` instances
  - Uses separate histograms for positive and negative values
  - No cached statistics - everything computed from buckets on-demand
  - All methods (`count()`, `sum()`, `min()`, `max()`) iterate buckets to compute values
  - Consistent API with zero memory overhead beyond the histograms themselves
  - Full support for negative values with `has_negatives()` working correctly

- **SharedExponentialHistogram**: Pure wrapper around two `histogram::AtomicHistogram` instances
  - Uses separate atomic histograms for positive and negative values
  - **No atomic counters** - completely stateless wrapper!
  - **All statistics** (count, sum, min, max) computed at read-time by walking histogram buckets
  - Provides zero-cost sharing via `Arc` wrapping
  - Thread-safe without any mutex overhead
  - `accumulate()` is just a single atomic increment to one histogram - incredibly fast!
  - `snapshot()` and `snapshot_and_reset()` return stateless `ExponentialHistogram` instances
  - Sum is approximate (uses bucket midpoints) matching the original `exponential_histogram` API
  - Guarantees consistency - all stats are derived from the same histogram snapshot
  - Full support for negative values

### Key Features

- Proper negative value support with separate positive/negative histograms
- `has_negatives()` correctly detects if negative values were recorded
- Min/max work correctly across negative and positive ranges
- Sum accounts for sign (positive buckets add, negative buckets subtract)
- `take_counts()` returns separate positive and negative bucket counts
- Min/max reflect histogram bucket boundaries (approximate but consistent)
- Sum is approximate based on bucket midpoints

## Usage as a Cargo Override

To use this as a replacement for `exponential_histogram` in your project, add the following to your project's `.cargo/config.toml`:

```toml
[patch.crates-io]
exponential_histogram = { path = "/path/to/this/crate" }
```

Or in your `Cargo.toml`:

```toml
[patch.crates-io]
exponential_histogram = { path = "/path/to/this/crate" }
```

Alternatively, you can use a Git repository:

```toml
[patch.crates-io]
exponential_histogram = { git = "https://github.com/yourusername/exponential_histogram" }
```

## API Compatibility

This wrapper provides the following types and methods matching the `exponential_histogram` crate:

### ExponentialHistogram

- `new(desired_scale: u8) -> Self`
- `new_with_max_buckets(desired_scale: u8, max_buckets: u16) -> Self`
- `reset(&mut self)`
- `accumulate<T: Into<f64>>(&mut self, value: T)`
- `is_empty(&self) -> bool`
- `count(&self) -> usize`
- `sum(&self) -> f64`
- `min(&self) -> f64`
- `max(&self) -> f64`
- `scale(&self) -> u8`
- `bucket_start_offset(&self) -> usize`
- `has_negatives(&self) -> bool`
- `take_counts(self) -> (VecDeque<usize>, VecDeque<usize>)`
- `value_counts(&self) -> impl Iterator<Item = (f64, usize)>`

### SharedExponentialHistogram

- `new(desired_scale: u8) -> Self`
- `new_with_max_buckets(desired_scale: u8, max_buckets: u16) -> Self`
- `accumulate(&self, value: f64)`
- `snapshot(&self) -> ExponentialHistogram`
- `snapshot_and_reset(&self) -> ExponentialHistogram`

## Testing

Run the test suite:

```bash
cargo test
```

## License

This project uses the same license as the underlying `histogram` crate.
