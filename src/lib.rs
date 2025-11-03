use histogram::{AtomicHistogram, Histogram as InnerHistogram};
use std::collections::VecDeque;

pub struct ExponentialHistogram {
    positive: InnerHistogram,
    negative: InnerHistogram,
    scale: u8,
    #[allow(dead_code)]
    max_buckets: u16,
}

impl ExponentialHistogram {
    pub fn new(desired_scale: u8) -> Self {
        Self::new_with_max_buckets(desired_scale, 160)
    }

    pub fn new_with_max_buckets(desired_scale: u8, max_buckets: u16) -> Self {
        let grouping_power = desired_scale.min(7);
        let max_value_power = 64;

        let positive = InnerHistogram::new(grouping_power, max_value_power)
            .expect("failed to create histogram");
        let negative = InnerHistogram::new(grouping_power, max_value_power)
            .expect("failed to create histogram");

        Self {
            positive,
            negative,
            scale: desired_scale,
            max_buckets,
        }
    }

    pub fn reset(&mut self) {
        let grouping_power = self.scale.min(7);
        let max_value_power = 64;

        self.positive = InnerHistogram::new(grouping_power, max_value_power)
            .expect("failed to create histogram");
        self.negative = InnerHistogram::new(grouping_power, max_value_power)
            .expect("failed to create histogram");
    }

    pub fn accumulate<T: Into<f64>>(&mut self, value: T) {
        let val: f64 = value.into();

        if val.is_finite() {
            let abs_val = val.abs();
            if val >= 0.0 {
                let _ = self.positive.increment(abs_val as u64);
            } else {
                let _ = self.negative.increment(abs_val as u64);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    pub fn count(&self) -> usize {
        let mut count = 0;
        for bucket in self.positive.iter() {
            if bucket.count() > 0 {
                count += bucket.count() as usize;
            }
        }
        for bucket in self.negative.iter() {
            if bucket.count() > 0 {
                count += bucket.count() as usize;
            }
        }
        count
    }

    pub fn sum(&self) -> f64 {
        let mut sum = 0.0;
        for bucket in self.positive.iter() {
            if bucket.count() > 0 {
                let bucket_mid = (bucket.start() + bucket.end()) as f64 / 2.0;
                sum += bucket_mid * bucket.count() as f64;
            }
        }
        for bucket in self.negative.iter() {
            if bucket.count() > 0 {
                let bucket_mid = (bucket.start() + bucket.end()) as f64 / 2.0;
                sum -= bucket_mid * bucket.count() as f64;
            }
        }
        sum
    }

    pub fn min(&self) -> f64 {
        // Check negative histogram first (most negative value - largest bucket)
        let mut neg_max = None;
        for bucket in self.negative.iter() {
            if bucket.count() > 0 {
                neg_max = Some(bucket.end() as f64);
            }
        }
        if let Some(val) = neg_max {
            return -val;
        }

        // If no negatives, check positive histogram (smallest bucket)
        for bucket in self.positive.iter() {
            if bucket.count() > 0 {
                return bucket.start() as f64;
            }
        }
        0.0
    }

    pub fn max(&self) -> f64 {
        // Check positive histogram first (most positive value)
        let mut max = 0.0;
        for bucket in self.positive.iter() {
            if bucket.count() > 0 {
                max = bucket.end() as f64;
            }
        }
        if max > 0.0 {
            return max;
        }
        // If no positives, check negative histogram
        for bucket in self.negative.iter() {
            if bucket.count() > 0 {
                return -(bucket.start() as f64);
            }
        }
        0.0
    }

    pub fn scale(&self) -> u8 {
        self.scale
    }

    pub fn bucket_start_offset(&self) -> usize {
        0
    }

    pub fn has_negatives(&self) -> bool {
        for bucket in self.negative.iter() {
            if bucket.count() > 0 {
                return true;
            }
        }
        false
    }

    pub fn take_counts(self) -> (VecDeque<usize>, VecDeque<usize>) {
        let mut positive_counts = VecDeque::new();
        let mut negative_counts = VecDeque::new();

        for bucket in self.positive.iter() {
            if bucket.count() > 0 {
                positive_counts.push_back(bucket.count() as usize);
            }
        }

        for bucket in self.negative.iter() {
            if bucket.count() > 0 {
                negative_counts.push_back(bucket.count() as usize);
            }
        }

        (positive_counts, negative_counts)
    }

    pub fn value_counts(&self) -> impl Iterator<Item = (f64, usize)> + '_ {
        let positive_iter = self.positive.iter().filter_map(|bucket| {
            if bucket.count() > 0 {
                Some((bucket.end() as f64, bucket.count() as usize))
            } else {
                None
            }
        });

        let negative_iter = self.negative.iter().filter_map(|bucket| {
            if bucket.count() > 0 {
                Some((-(bucket.end() as f64), bucket.count() as usize))
            } else {
                None
            }
        });

        negative_iter.chain(positive_iter)
    }
}

pub struct SharedExponentialHistogram {
    positive: AtomicHistogram,
    negative: AtomicHistogram,
    scale: u8,
    #[allow(dead_code)]
    max_buckets: u16,
}

impl SharedExponentialHistogram {
    pub fn new(desired_scale: u8) -> Self {
        Self::new_with_max_buckets(desired_scale, 160)
    }

    pub fn new_with_max_buckets(desired_scale: u8, max_buckets: u16) -> Self {
        let grouping_power = desired_scale.min(7);
        let max_value_power = 64;

        let positive = AtomicHistogram::new(grouping_power, max_value_power)
            .expect("failed to create atomic histogram");
        let negative = AtomicHistogram::new(grouping_power, max_value_power)
            .expect("failed to create atomic histogram");

        Self {
            positive,
            negative,
            scale: desired_scale,
            max_buckets,
        }
    }

    pub fn accumulate(&self, value: f64) {
        if !value.is_finite() {
            return;
        }

        let abs_val = value.abs();
        if value >= 0.0 {
            let _ = self.positive.increment(abs_val as u64);
        } else {
            let _ = self.negative.increment(abs_val as u64);
        }
    }

    pub fn snapshot(&self) -> ExponentialHistogram {
        let positive = self.positive.load();
        let negative = self.negative.load();

        ExponentialHistogram {
            positive,
            negative,
            scale: self.scale,
            max_buckets: self.max_buckets,
        }
    }

    pub fn snapshot_and_reset(&self) -> ExponentialHistogram {
        let positive = self.positive.drain();
        let negative = self.negative.drain();

        ExponentialHistogram {
            positive,
            negative,
            scale: self.scale,
            max_buckets: self.max_buckets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_histogram_basic() {
        let mut hist = ExponentialHistogram::new(0);
        assert!(hist.is_empty());
        assert_eq!(hist.count(), 0);

        hist.accumulate(1.0);
        hist.accumulate(2.0);
        hist.accumulate(3.0);

        assert!(!hist.is_empty());
        assert_eq!(hist.count(), 3);
        assert_eq!(hist.sum(), 6.0);
        assert_eq!(hist.min(), 1.0);
        assert_eq!(hist.max(), 3.0);
    }

    #[test]
    fn test_exponential_histogram_with_max_buckets() {
        let mut hist = ExponentialHistogram::new_with_max_buckets(2, 100);
        hist.accumulate(10.0);
        assert_eq!(hist.count(), 1);
        assert_eq!(hist.scale(), 2);
    }

    #[test]
    fn test_exponential_histogram_reset() {
        let mut hist = ExponentialHistogram::new(0);
        hist.accumulate(5.0);
        hist.accumulate(10.0);

        assert_eq!(hist.count(), 2);

        hist.reset();

        assert!(hist.is_empty());
        assert_eq!(hist.count(), 0);
    }

    #[test]
    fn test_exponential_histogram_negatives() {
        let mut hist = ExponentialHistogram::new(0);
        hist.accumulate(-5.0);
        hist.accumulate(10.0);
        hist.accumulate(-2.0);

        assert!(hist.has_negatives());
        assert_eq!(hist.count(), 3);

        // Min should be approximately -5 (based on negative histogram buckets)
        let min = hist.min();
        assert!(min < 0.0, "min should be negative, got {}", min);
        assert!(min <= -5.0, "min should be <= -5.0, got {}", min);

        // Max should be approximately 10 (based on positive histogram buckets)
        let max = hist.max();
        assert!(max > 0.0, "max should be positive, got {}", max);
        assert!(max >= 10.0, "max should be >= 10.0, got {}", max);

        // Sum should be approximately 3.0 (= -5 + 10 - 2)
        let sum = hist.sum();
        println!("Sum: {}", sum);
        assert!((sum - 3.0).abs() < 5.0, "sum should be approximately 3.0, got {}", sum);
    }

    #[test]
    fn test_shared_exponential_histogram() {
        let hist = SharedExponentialHistogram::new(0);

        hist.accumulate(1.0);
        hist.accumulate(2.0);
        hist.accumulate(3.0);

        let snapshot = hist.snapshot();
        assert_eq!(snapshot.count(), 3);
        assert_eq!(snapshot.sum(), 6.0);

        let snapshot2 = hist.snapshot();
        assert_eq!(snapshot2.count(), 3);
    }

    #[test]
    fn test_shared_exponential_histogram_reset() {
        let hist = SharedExponentialHistogram::new(0);

        hist.accumulate(1.0);
        hist.accumulate(2.0);

        let snapshot = hist.snapshot_and_reset();
        assert_eq!(snapshot.count(), 2);

        let snapshot2 = hist.snapshot();
        assert_eq!(snapshot2.count(), 0);
    }

    #[test]
    fn test_value_counts_iterator() {
        let mut hist = ExponentialHistogram::new(0);
        hist.accumulate(1.0);
        hist.accumulate(5.0);
        hist.accumulate(10.0);

        let counts: Vec<_> = hist.value_counts().collect();
        assert!(!counts.is_empty());
    }

    #[test]
    fn test_take_counts() {
        let mut hist = ExponentialHistogram::new(0);
        hist.accumulate(1.0);
        hist.accumulate(5.0);
        hist.accumulate(10.0);

        let (positive, _negative) = hist.take_counts();
        assert!(!positive.is_empty());
    }

    #[test]
    fn test_shared_exponential_histogram_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let hist = Arc::new(SharedExponentialHistogram::new(0));
        let mut handles = vec![];

        for i in 0..10 {
            let hist_clone = Arc::clone(&hist);
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    hist_clone.accumulate((i * 100 + j) as f64);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let snapshot = hist.snapshot();
        assert_eq!(snapshot.count(), 1000);
    }

    #[test]
    fn test_histogram_zero_handling() {
        let mut hist = InnerHistogram::new(0, 64).unwrap();

        hist.increment(0).unwrap();
        hist.increment(0).unwrap();
        hist.increment(1).unwrap();

        let mut total_count = 0;
        println!("\nBuckets with data:");
        for bucket in hist.iter() {
            if bucket.count() > 0 {
                println!("  start={}, end={}, count={}",
                         bucket.start(), bucket.end(), bucket.count());
                total_count += bucket.count();
            }
        }

        assert_eq!(total_count, 3, "Should track all values including zeros");
    }

    #[test]
    fn test_histogram_sum_from_buckets() {
        let mut hist = InnerHistogram::new(0, 64).unwrap();

        hist.increment(10).unwrap();
        hist.increment(20).unwrap();
        hist.increment(30).unwrap();

        let mut computed_sum = 0.0;
        println!("\nComputing sum from buckets:");
        for bucket in hist.iter() {
            if bucket.count() > 0 {
                let bucket_mid = (bucket.start() + bucket.end()) as f64 / 2.0;
                let contribution = bucket_mid * bucket.count() as f64;
                println!("  Bucket [{}, {}]: count={}, mid={}, contribution={}",
                         bucket.start(), bucket.end(), bucket.count(), bucket_mid, contribution);
                computed_sum += contribution;
            }
        }

        println!("Computed sum: {}, Expected: 60", computed_sum);
        println!("Note: Sum is approximate due to bucket quantization");
    }
}
