pub fn median_u8(values: &[u8]) -> u8 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted: Vec<u8> = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 != 0 {
        sorted[mid]
    } else {
        ((sorted[mid - 1] as u16 + sorted[mid] as u16) / 2) as u8
    }
}

pub fn median_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 != 0 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

pub fn mode_u8(values: &[u8]) -> u8 {
    if values.is_empty() {
        return 0;
    }
    use std::collections::HashMap;
    let mut counts = HashMap::new();
    let mut best = values[0];
    let mut best_n = 0u32;
    for &v in values {
        let c = counts.entry(v).or_insert(0u32);
        *c += 1;
        if *c > best_n {
            best_n = *c;
            best = v;
        }
    }
    best
}

pub fn mean_u8(values: &[u8]) -> u8 {
    if values.is_empty() {
        return 0;
    }
    let sum: u32 = values.iter().map(|&v| v as u32).sum();
    (sum / values.len() as u32) as u8
}
