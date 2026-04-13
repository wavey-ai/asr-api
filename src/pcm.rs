pub fn rms_level(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum_squares = samples
        .iter()
        .map(|sample| {
            let sample = *sample;
            sample * sample
        })
        .sum::<f32>();
    (sum_squares / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_level_returns_zero_for_silence() {
        assert_eq!(rms_level(&[]), 0.0);
        assert_eq!(rms_level(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn rms_level_matches_expected_value() {
        let rms = rms_level(&[1.0, -1.0, 1.0, -1.0]);
        assert!((rms - 1.0).abs() < 1e-6);
    }
}
