// helixc/stdlib/mnist.hx — IDX-format binary blob reader.
//
// Stage 55 Inc 6: pure Helix stdlib for parsing MNIST-style IDX
// headers + bounds-checking the body. No file I/O — caller is
// expected to have already loaded the IDX bytes into the arena
// (Inc 3 file I/O will provide this at runtime once shipped).
//
// IDX format (LeCun's MNIST format):
//   bytes 0..1   : 0x00 0x00 (zero padding)
//   byte  2      : dtype code (0x08=u8, 0x09=i8, 0x0B=i16, 0x0C=i32,
//                              0x0D=f32, 0x0E=f64)
//   byte  3      : ndims  (1, 2, 3, ...)
//   bytes 4..4+(ndims*4) : ndims big-endian u32 dimension sizes
//   body         : product(dims) * dtype_size bytes
//
// MNIST canonical files:
//   train-images-idx3-ubyte: magic 0x00000803, ndims=3, dims=[60000, 28, 28]
//   train-labels-idx1-ubyte: magic 0x00000801, ndims=1, dims=[60000]
//   t10k-images-idx3-ubyte:  magic 0x00000803, ndims=3, dims=[10000, 28, 28]
//   t10k-labels-idx1-ubyte:  magic 0x00000801, ndims=1, dims=[10000]
//
// Convention: caller loads the IDX blob into arena bytes
// [blob_start..blob_start+blob_len). All API functions take that
// pair and return derived offsets/values without further allocation.
//
// API:
//   mnist_idx_magic_ok(blob, blob_len)        -> i32   1 if bytes 0+1 are both 0
//                                                      AND ndims >= 1
//   mnist_idx_dtype(blob, blob_len)           -> i32   byte 2 (dtype code), 0 if invalid
//   mnist_idx_ndims(blob, blob_len)           -> i32   byte 3 (ndims), 0 if invalid
//   mnist_idx_header_size(blob, blob_len)     -> i32   4 + (ndims * 4) — body start offset
//   mnist_idx_dim(blob, blob_len, i)          -> i32   i-th dimension (0-indexed)
//                                                      as i32 big-endian decoded
//   mnist_idx_body_offset(blob, blob_len)     -> i32   alias for header_size
//   mnist_idx_body_len_bytes(blob, blob_len)  -> i32   blob_len - header_size
//   mnist_idx_dtype_size(dtype)               -> i32   bytes per element (0 if unknown)
//   mnist_idx_expected_body_len(blob, blob_len) -> i32  product(dims) * dtype_size
//   mnist_idx_validate(blob, blob_len)        -> i32   1 if header+body lengths match;
//                                                      0 if malformed/truncated
//   mnist_idx_u8_at(blob, blob_len, i)        -> i32   i-th body byte (u8 widened to i32);
//                                                      no bounds-check
//   mnist_idx_image_pixel(blob, blob_len, img_idx, row, col) -> i32
//                                                      pixel for the 3D u8 case
//                                                      (train/t10k images);
//                                                      dim[1],dim[2] looked up inside
//                                                      via mnist_idx_dim. Hot loops
//                                                      should cache (h, w) and call
//                                                      mnist_idx_u8_at directly with
//                                                      a pre-computed offset.
//
// License: Apache 2.0

@pure
fn mnist_idx_magic_ok(blob: i32, blob_len: i32) -> i32 {
    if blob_len < 4 { 0 }
    else {
        let b0 = __str_byte_at(blob, 0);
        let b1 = __str_byte_at(blob, 1);
        let nd = __str_byte_at(blob, 3);
        if b0 != 0 { 0 }
        else { if b1 != 0 { 0 }
        else { if nd < 1 { 0 } else { 1 } } }
    }
}

@pure
fn mnist_idx_dtype(blob: i32, blob_len: i32) -> i32 {
    if blob_len < 4 { 0 }
    else { __str_byte_at(blob, 2) }
}

@pure
fn mnist_idx_ndims(blob: i32, blob_len: i32) -> i32 {
    if blob_len < 4 { 0 }
    else { __str_byte_at(blob, 3) }
}

@pure
fn mnist_idx_header_size(blob: i32, blob_len: i32) -> i32 {
    let nd = mnist_idx_ndims(blob, blob_len);
    4 + nd * 4
}

// mnist_idx_dim: read the i-th 4-byte big-endian u32 dimension
// (bytes [4 + i*4 .. 4 + i*4 + 4)). Returns 0 if i is out of range.
// IDX dims are always non-negative and fit in i31, so the i32 cast
// is safe for canonical MNIST files.
@pure
fn mnist_idx_dim(blob: i32, blob_len: i32, i: i32) -> i32 {
    let nd = mnist_idx_ndims(blob, blob_len);
    if i < 0 { 0 }
    else { if i >= nd { 0 }
    else {
        let pos = 4 + i * 4;
        if pos + 4 > blob_len { 0 }
        else {
            let b0 = __str_byte_at(blob, pos);
            let b1 = __str_byte_at(blob, pos + 1);
            let b2 = __str_byte_at(blob, pos + 2);
            let b3 = __str_byte_at(blob, pos + 3);
            // Cycle 3 R1 fix batch 20 (RT HIGH-3): big-endian decode of u32
            // into i32 silently wraps for dims with high byte >= 128 (i.e.
            // dim > 2^31 / 2^24 = 128 in high byte). b0=128, b1=b2=b3=0
            // = 2^31 = INT32_MIN. For dims <= 2^31-1 (which covers all
            // realistic IDX files including ImageNet-scale), this works
            // correctly. For dims > 2^31-1, returns INT32_MIN sentinel —
            // since IDX dims are unsigned i32 on disk, the sentinel is
            // distinguishable from any legitimate small dim.
            // Big-endian: byte 0 is MSB.
            if b0 >= 128 {
                // Top bit set: u32 value > 2^31-1, doesn't fit in i32.
                // Return INT32_MIN as out-of-band sentinel.
                0 - 2147483647 - 1
            } else {
                b0 * 16777216 + b1 * 65536 + b2 * 256 + b3
            }
        }
    } }
}

@pure
fn mnist_idx_body_offset(blob: i32, blob_len: i32) -> i32 {
    mnist_idx_header_size(blob, blob_len)
}

@pure
fn mnist_idx_body_len_bytes(blob: i32, blob_len: i32) -> i32 {
    let h = mnist_idx_header_size(blob, blob_len);
    if h >= blob_len { 0 } else { blob_len - h }
}

// mnist_idx_dtype_size: bytes per element. Returns 0 for unrecognized
// dtype codes so callers can fail-closed.
@pure
fn mnist_idx_dtype_size(dtype: i32) -> i32 {
    if dtype == 8 { 1 }       // 0x08 u8
    else { if dtype == 9 { 1 }   // 0x09 i8
    else { if dtype == 11 { 2 }  // 0x0B i16
    else { if dtype == 12 { 4 }  // 0x0C i32
    else { if dtype == 13 { 4 }  // 0x0D f32
    else { if dtype == 14 { 8 }  // 0x0E f64
    else { 0 } } } } } }
}

// mnist_idx_expected_body_len: product(dims) * dtype_size.
// Phase-0 supports up to 4 dims (covers all canonical MNIST + most
// IDX use cases). Higher-rank tensors saturate to 0 to fail-closed
// the validate check.
//
// Cycle 3 R1 fix batch 20 (RT HIGH-4): every multiplication step now
// guards against i32 overflow. Returns INT32_MIN sentinel on overflow.
// Pre-fix: [1000, 1000, 1000, 1000] u8 = 1e12 silently wraps to small
// positive value, validate() passes a corrupt file. Post-fix: overflow
// detected → INT32_MIN propagates to validate, which now rejects.
@pure
fn mnist_idx_expected_body_len(blob: i32, blob_len: i32) -> i32 {
    let nd = mnist_idx_ndims(blob, blob_len);
    let dt = mnist_idx_dtype(blob, blob_len);
    let elem_size = mnist_idx_dtype_size(dt);
    if elem_size == 0 { 0 }
    else { if nd < 1 { 0 }
    else { if nd > 4 { 0 }
    else {
        let d0 = mnist_idx_dim(blob, blob_len, 0);
        if d0 == 0 - 2147483647 - 1 { 0 - 2147483647 - 1 }
        else if d0 < 0 { 0 - 2147483647 - 1 }
        else {
        let prod = if nd >= 2 {
            let d1 = mnist_idx_dim(blob, blob_len, 1);
            if d1 < 0 { 0 - 2147483647 - 1 }
            else if d0 != 0 && d1 > 2147483647 / d0 { 0 - 2147483647 - 1 }
            else {
            let p2 = d0 * d1;
            if nd >= 3 {
                let d2 = mnist_idx_dim(blob, blob_len, 2);
                if d2 < 0 { 0 - 2147483647 - 1 }
                else if p2 != 0 && d2 > 2147483647 / p2 { 0 - 2147483647 - 1 }
                else {
                let p3 = p2 * d2;
                if nd >= 4 {
                    let d3 = mnist_idx_dim(blob, blob_len, 3);
                    if d3 < 0 { 0 - 2147483647 - 1 }
                    else if p3 != 0 && d3 > 2147483647 / p3 { 0 - 2147483647 - 1 }
                    else { p3 * d3 }
                } else { p3 }
                }
            } else { p2 }
            }
        } else { d0 };
        // Final elem_size multiplication overflow guard.
        if prod == 0 - 2147483647 - 1 { 0 - 2147483647 - 1 }
        else if prod != 0 && elem_size > 2147483647 / prod { 0 - 2147483647 - 1 }
        else { prod * elem_size }
        }
    } } }
}

@pure
fn mnist_idx_validate(blob: i32, blob_len: i32) -> i32 {
    if mnist_idx_magic_ok(blob, blob_len) == 0 { 0 }
    else {
        let dt = mnist_idx_dtype(blob, blob_len);
        if mnist_idx_dtype_size(dt) == 0 { 0 }
        else {
            let header = mnist_idx_header_size(blob, blob_len);
            if header > blob_len { 0 }
            else {
                let expect = mnist_idx_expected_body_len(blob, blob_len);
                // Cycle 3 R1 fix batch 20 (RT HIGH-4): honor INT32_MIN
                // overflow sentinel from expected_body_len — corrupt large
                // file would have silently passed before this guard.
                if expect == 0 - 2147483647 - 1 { 0 }
                else {
                    let actual = blob_len - header;
                    if expect == actual { 1 } else { 0 }
                }
            }
        }
    }
}

// mnist_idx_u8_at: i-th body byte widened to i32. No bounds-check —
// caller is expected to have called validate first. Body starts at
// mnist_idx_body_offset and contains body_len_bytes total bytes.
@pure
fn mnist_idx_u8_at(blob: i32, blob_len: i32, i: i32) -> i32 {
    let body = mnist_idx_header_size(blob, blob_len);
    __str_byte_at(blob, body + i)
}

// mnist_idx_image_pixel: thin shim for the canonical 3D u8 image
// case (60000 x 28 x 28 etc.). Looks up dim[1] (h) and dim[2] (w)
// from the header on every call. Hot loops should bypass this and
// call mnist_idx_u8_at directly with a pre-computed offset.
// img_idx is the 0-indexed image; row, col the in-image coordinates.
// Returns 0..255 as i32.
@pure
fn mnist_idx_image_pixel(blob: i32, blob_len: i32,
                          img_idx: i32, row: i32, col: i32) -> i32 {
    let body = mnist_idx_header_size(blob, blob_len);
    let h = mnist_idx_dim(blob, blob_len, 1);
    let w = mnist_idx_dim(blob, blob_len, 2);
    let off = img_idx * (h * w) + row * w + col;
    __str_byte_at(blob, body + off)
}

// Cycle 2 Batch RT fix batch 17 (silent-failure HIGH-2):
// Pre-fix: mnist_idx_u8_at and mnist_idx_image_pixel had ZERO
// bounds-check; OOB returned whatever arena byte was at body+i
// (tensor magic, freed memory, another image's pixels). Training
// loops iterating past body silently consumed adjacent garbage as
// valid pixel input. Same defect class as pre-fix string_get OOB.
// Post-fix: _checked variants return -1 sentinel on OOB. Originals
// preserved for backward compat with existing hot-loop callers.

@pure
fn mnist_idx_u8_at_checked(blob: i32, blob_len: i32, body_len: i32, i: i32) -> i32 {
    if i < 0 { 0 - 1 }
    else { if i >= body_len { 0 - 1 }
    else { mnist_idx_u8_at(blob, blob_len, i) } }
}

@pure
fn mnist_idx_image_pixel_checked(blob: i32, blob_len: i32, n_images: i32,
                                  img_idx: i32, row: i32, col: i32) -> i32 {
    let h = mnist_idx_dim(blob, blob_len, 1);
    let w = mnist_idx_dim(blob, blob_len, 2);
    if img_idx < 0 { 0 - 1 }
    else { if img_idx >= n_images { 0 - 1 }
    else { if row < 0 { 0 - 1 }
    else { if row >= h { 0 - 1 }
    else { if col < 0 { 0 - 1 }
    else { if col >= w { 0 - 1 }
    else { mnist_idx_image_pixel(blob, blob_len, img_idx, row, col) } } } } } }
}
