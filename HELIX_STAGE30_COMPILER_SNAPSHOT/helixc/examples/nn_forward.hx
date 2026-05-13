// nn_forward.hx — forward pass of a tiny neural network in Helix.
//
// Computes:  y = relu(W * x + b)  for a 1-hidden-layer network
// where:
//   x = [1.0, 2.0, 3.0, 4.0]   (input, 4 features)
//   W = 4x4 matrix (weights)
//   b = [0.0; 4]                (bias)
//   y = output, summed and truncated to i32 for the exit code
//
// All in floats, exercising: arrays, for-loops, indexing, float arith,
// comparisons, branches, casts.

fn main() -> i32 {
    let x = [1.0, 2.0, 3.0, 4.0];

    // 4x4 identity matrix (so W*x = x)
    let w = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ];

    // Output buffer
    let y = [0.0, 0.0, 0.0, 0.0];

    // y[i] = sum_j W[i,j] * x[j]   (matrix-vector product)
    for i in 0 .. 4 {
        let mut acc = 0.0;
        for j in 0 .. 4 {
            acc += w[i * 4 + j] * x[j];
        }
        // ReLU activation: max(acc, 0.0)
        if acc < 0.0 {
            y[i] = 0.0;
        } else {
            y[i] = acc;
        }
    }

    // Sum the outputs: 1 + 2 + 3 + 4 = 10
    // (we want 42 as exit code, so add 32 to it for the demo)
    let mut total = 32.0;
    for i in 0 .. 4 {
        total += y[i];
    }

    // Cast to i32 and return as exit code: 42
    total as i32
}
