// Mandelbrot set rendered to stdout as ASCII shading.
//
// For each cell (x, y) on a 60x24 grid, we map it to a complex number
// c = (x_real, y_imag) in [-2.5, 1.0] x [-1.0, 1.0], then iterate
//
//   z_{n+1} = z_n^2 + c
//
// starting from z_0 = 0. We count how many iterations before |z| > 2
// (the "escape time"). Points that don't escape after MAX_ITER are
// inside the set; points that escape quickly are outside.
//
// The escape count is mapped to one of 10 ASCII shading characters,
// from " " (escapes immediately) to "@" (in the set). The result
// is a recognizable ASCII picture of the Mandelbrot fractal printed
// directly to stdout.
//
// Exercises: nested loops, mutable lets, while + conditional, float
// arithmetic, print_str with literal dispatch.

@pure
fn shade(escape: i32) -> i32 {
    // Print one of 10 shade characters based on escape count.
    // Higher escape = closer to/inside the set = denser char.
    if escape <  5 { print_str(" "); }
    else { if escape < 10 { print_str("."); }
    else { if escape < 15 { print_str(":"); }
    else { if escape < 20 { print_str("-"); }
    else { if escape < 25 { print_str("="); }
    else { if escape < 30 { print_str("+"); }
    else { if escape < 35 { print_str("*"); }
    else { if escape < 40 { print_str("#"); }
    else { if escape < 50 { print_str("%"); }
    else { print_str("@"); }}}}}}}}}
    0
}

@pure
fn iterate(cx: f32, cy: f32, max_iter: i32) -> i32 {
    let mut zx: f32 = 0.0;
    let mut zy: f32 = 0.0;
    let mut i: i32 = 0;
    let mut count: i32 = max_iter;  // assume in-set unless we escape
    while i < max_iter {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > 4.0 {
            // Escaped; record iteration count and exit loop.
            count = i;
            i = max_iter;
        } else {
            // z = z^2 + c. (a+bi)^2 = (a^2 - b^2) + 2abi.
            let new_zx = zx2 - zy2 + cx;
            let new_zy = 2.0 * zx * zy + cy;
            zx = new_zx;
            zy = new_zy;
            i = i + 1;
        };
    }
    count
}

fn render_row(y: i32, width: i32, height: i32, max_iter: i32) -> i32 {
    let mut x: i32 = 0;
    while x < width {
        // Map x in [0, width) to cx in [-2.5, 1.0].
        let cx_num = (x as f32) * 3.5 - 2.5 * (width as f32);
        let cx = cx_num / (width as f32);
        // Map y in [0, height) to cy in [-1.0, 1.0].
        let cy_num = (y as f32) * 2.0 - 1.0 * (height as f32);
        let cy = cy_num / (height as f32);
        // iterate() returns iteration count at escape, or max_iter if
        // the point stayed bounded ("in the set"). High count = denser
        // shade (close to/in set); low count = lighter shade.
        let escape = iterate(cx, cy, max_iter);
        shade(escape);
        x = x + 1;
    }
    print_str("\n");
    0
}

fn main() -> i32 {
    let width: i32 = 60;
    let height: i32 = 22;
    let max_iter: i32 = 50;
    let mut y: i32 = 0;
    while y < height {
        render_row(y, width, height, max_iter);
        y = y + 1;
    };
    0
}
