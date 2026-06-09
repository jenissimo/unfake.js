//! Runs-based scale detection (GCD of color run lengths).

pub fn runs_based_detect(rgba: &[u8], width: u32, height: u32) -> u32 {
    let w = width as usize;
    let h = height as usize;
    let mut runs = Vec::new();

    for y in 0..h {
        collect_runs_horizontal(rgba, w, y, &mut runs);
    }
    for x in 0..w {
        collect_runs_vertical(rgba, w, h, x, &mut runs);
    }

    if runs.len() < 10 {
        return 1;
    }
    gcd_array(&runs).max(1)
}

fn collect_runs_horizontal(rgba: &[u8], w: usize, y: usize, out: &mut Vec<u32>) {
    let mut len = 1u32;
    for x in 1..w {
        if same_pixel(rgba, y * w + x, y * w + (x - 1)) {
            len += 1;
        } else {
            if len > 1 {
                out.push(len);
            }
            len = 1;
        }
    }
    if len > 1 {
        out.push(len);
    }
}

fn collect_runs_vertical(rgba: &[u8], w: usize, h: usize, x: usize, out: &mut Vec<u32>) {
    let mut len = 1u32;
    for y in 1..h {
        if same_pixel(rgba, y * w + x, (y - 1) * w + x) {
            len += 1;
        } else {
            if len > 1 {
                out.push(len);
            }
            len = 1;
        }
    }
    if len > 1 {
        out.push(len);
    }
}

fn same_pixel(rgba: &[u8], a: usize, b: usize) -> bool {
    let ia = a * 4;
    let ib = b * 4;
    rgba[ia..ia + 4] == rgba[ib..ib + 4]
}

fn gcd_array(values: &[u32]) -> u32 {
    if values.is_empty() {
        return 1;
    }
    let mut g = values[0];
    for &v in &values[1..] {
        g = gcd(g, v);
        if g == 1 {
            return 1;
        }
    }
    g
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
