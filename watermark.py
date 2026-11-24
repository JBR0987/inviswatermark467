from pathlib import Path
import cv2
import numpy as np
import pywt

# Mid-frequency (row, col) coordinates inside an 8x8 DCT block (picked some random numbers 1-5, 1-5 and jumbled them up)
MID_FREQ_COORDS = [
    (1, 2), (2, 1), (2, 2), (1, 3), (3, 1),
    (1, 4), (2, 3), (3, 2), (4, 1), (2, 4),
    (3, 3), (4, 2), (1, 5), (5, 1), (3, 4),
    (4, 3), (2, 5), (5, 2), (3, 5), (5, 3),
]


def _arnold_step(grid: np.ndarray):
    """Applies one step of the Arnold Cat Map to an n x n grid."""

    n = grid.shape[0]
    out = np.zeros_like(grid)
    for r in range(n):
        for c in range(n):
            out[(r + 2 * c) % n, (r + c) % n] = grid[r, c]
    return out


def _arnold_period(n: int):
    """Finds the cycle length of the Arnold Cat Map for an n x n grid."""

    # Track unique values so returning to the probe marks a full cycle
    probe = np.arange(n * n, dtype=np.int32).reshape(n, n)
    curr = _arnold_step(probe)
    period = 1
    while not np.array_equal(curr, probe):
        curr = _arnold_step(curr)
        period += 1
    return period


def scramble_watermark(wm: np.ndarray, iters: int = 5):
    """Scrambles a square binary (0/1) watermark using Arnold Cat Map."""

    n = wm.shape[0]
    period = _arnold_period(n)
    out = wm.copy()

    # Reduce the iteration count because the map repeats after its period
    for _ in range(iters % period):
        out = _arnold_step(out)
    return out


def unscramble_watermark(scrambled: np.ndarray, iters: int = 5):
    """Reverses Arnold Cat Map scrambling."""

    n = scrambled.shape[0]
    period = _arnold_period(n)
    inv_steps = (period - (iters % period)) % period
    out = scrambled.copy()
    for _ in range(inv_steps):
        out = _arnold_step(out)
    return out


def get_pn_keys(seed, length=None):
    """Generate the +1/-1 patterns used to store 0 and 1 bits."""

    if length is None:
        length = len(MID_FREQ_COORDS)

    rng = np.random.default_rng(seed)

    # Reusing the seed lets extraction regenerate the same bit patterns
    pn0 = rng.choice([-1.0, 1.0], size=length).astype(np.float32)
    pn1 = rng.choice([-1.0, 1.0], size=length).astype(np.float32)
    return pn0, pn1


def nc_score(a: np.ndarray, b: np.ndarray):
    """Normalized Cross-Correlation between two arrays."""

    va = a.astype(np.float64).ravel()
    vb = b.astype(np.float64).ravel()
    denom = np.sqrt(np.sum(va ** 2) * np.sum(vb ** 2)) + 1e-12
    return float(np.sum(va * vb) / denom)


def embed_watermark(
    cover_bgr,
    wm_binary,
    seed=42,
    alpha=10.0,
    wavelet="haar",
    iters=5,
):
    """Embed a square binary watermark with DWT and DCT."""

    h, w = cover_bgr.shape[:2]

    # Convert to YCrCb and extract the Y channel
    ycrcb = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    y = ycrcb[:, :, 0]

    # Perform DWT on the Y channel
    ll1, (lh1, hl1, hh1) = pywt.dwt2(y, wavelet)
    hl1_mod = hl1.copy()

    
    sub_h, sub_w = hl1_mod.shape
    block_rows, block_cols = sub_h // 8, sub_w // 8
    max_bits = block_rows * block_cols
    
    wm_side = min(wm_binary.shape[0], wm_binary.shape[1])
    while wm_side * wm_side > max_bits and wm_side > 1:
        wm_side -= 1
    
    wm_sq = cv2.resize(
        wm_binary,
        (wm_side, wm_side),
        interpolation=cv2.INTER_NEAREST
    )
    wm_sq = (wm_sq > 0).astype(np.uint8)
    bits = scramble_watermark(wm_sq, iters=iters).flatten()
    pn0, pn1 = get_pn_keys(seed, len(MID_FREQ_COORDS))

    # Embed the bits into the DCT coefficients of 8x8 blocks in the HL1 subband
    idx = 0
    for br in range(block_rows):
        for bc in range(block_cols):
            if idx >= len(bits):
                break
            r0, c0 = br * 8, bc * 8
            block = hl1_mod[r0:r0 + 8, c0:c0 + 8].astype(np.float32)
            dct_block = cv2.dct(block)

            if bits[idx] == 1:
                pn = pn1
            else:
                pn = pn0

            for k, (mr, mc) in enumerate(MID_FREQ_COORDS):
                dct_block[mr, mc] += alpha * pn[k]

            hl1_mod[r0:r0 + 8, c0:c0 + 8] = cv2.idct(dct_block).astype(np.float64)
            idx += 1

    # Reconstruct the Y channel and convert back to BGR
    y_rec = pywt.idwt2((ll1, (lh1, hl1_mod, hh1)), wavelet)[:h, :w]
    ycrcb[:, :, 0] = np.clip(y_rec, 0, 255)
    watermarked_bgr = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    # Keep extraction settings and the resized reference watermark together
    info = {
        "seed": seed,
        "wavelet": wavelet,
        "iters": iters,
        "wm_shape": (wm_side, wm_side),
        "wm_ref": wm_sq,
    }
    return watermarked_bgr, info


def extract_watermark(suspect_bgr, info):
    """Extract the watermark from a suspected image."""

    wm_h, wm_w = info["wm_shape"]
    total_bits = wm_h * wm_w

    # Convert to YCrCb and extract the Y channel
    ycrcb = cv2.cvtColor(suspect_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    _, (_, hl1, _) = pywt.dwt2(ycrcb[:, :, 0], info["wavelet"])

    sub_h, sub_w = hl1.shape
    block_rows, block_cols = sub_h // 8, sub_w // 8
    pn0, pn1 = get_pn_keys(info["seed"], len(MID_FREQ_COORDS))

    # Classify each block by whichever saved pattern correlates more strongly
    bits = []
    for br in range(block_rows):
        for bc in range(block_cols):
            if len(bits) >= total_bits:
                break
            r0, c0 = br * 8, bc * 8
            dct_block = cv2.dct(
                hl1[r0:r0 + 8, c0:c0 + 8].astype(np.float32)
            )
            coeffs = []
            for mr, mc in MID_FREQ_COORDS:
                coeffs.append(dct_block[mr, mc])
            coeffs = np.array(coeffs, dtype=np.float32)

            s1 = nc_score(coeffs, pn1)
            s0 = nc_score(coeffs, pn0)
            if s1 >= s0:
                bits.append(1)
            else:
                bits.append(0)
        if len(bits) >= total_bits:
            break

    while len(bits) < total_bits:
        bits.append(0)

    # Restore the original watermark layout after reading scrambled bits
    scrambled_est = np.array(bits[:total_bits], dtype=np.uint8).reshape((wm_h, wm_w))
    return unscramble_watermark(scrambled_est, iters=info["iters"])


def compute_psnr(orig_bgr, wm_bgr):
    """Compute the PSNR between two images."""

    mse = np.mean((orig_bgr.astype(np.float64) - wm_bgr.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compute_ssim(orig_bgr, wm_bgr):
    """Compute the SSIM between two images."""

    a = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(wm_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2

    # Compute local means, variances, and covariance using Gaussian blur
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sig_a2 = cv2.GaussianBlur(a ** 2, (11, 11), 1.5) - mu_a ** 2
    sig_b2 = cv2.GaussianBlur(b ** 2, (11, 11), 1.5) - mu_b ** 2
    sig_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b

    # Compute the SSIM map and return the mean value
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (sig_a2 + sig_b2 + c2)
    return float(np.mean(num / (den + 1e-12)))


def jpeg_sample_test(img_bgr, quality=70):
    """Simulate JPEG compression and decompression."""

    # Encode/decode in memory to simulate compression without altering the source
    ok, enc = cv2.imencode(
        ".jpg",
        img_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise RuntimeError("JPEG encode failed.")
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


if __name__ == "__main__":

    # Create a simple 32x32 watermark pattern.
    wm = np.zeros((32, 32), dtype=np.uint8)
    wm[4:28, 4:28] = 1
    wm[10:22, 10:22] = 0

    # Run tests on two sample images from COCO dataset
    for image_name in ("dogtest.png", "wrtest.png"):
        image_path = Path(__file__).with_name(image_name)
        cover = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if cover is None:
            raise FileNotFoundError(f"Could not read cover image: {image_path}")

        wm_img, info = embed_watermark(
            cover,
            wm,
            seed=42,
            alpha=10.0,
            wavelet="haar",
            iters=5,
        )

        print(f"\nTest image: {image_name}")
        print(f"PSNR: {compute_psnr(cover, wm_img):.2f} dB")
        print(f"SSIM: {compute_ssim(cover, wm_img):.4f}")

        clean_ext = extract_watermark(wm_img, info)
        clean_ber = float(np.mean(info["wm_ref"] != clean_ext))
        print(
            f"Clean Extraction -> NC: "
            f"{nc_score(info['wm_ref'], clean_ext):.4f}, "
            f"BER: {clean_ber:.4f}"
        )

        jpg_img = jpeg_sample_test(wm_img, quality=70)
        jpg_ext = extract_watermark(jpg_img, info)
        jpg_ber = float(np.mean(info["wm_ref"] != jpg_ext))
        print(
            f"JPEG Q=70 Sample Test -> NC: "
            f"{nc_score(info['wm_ref'], jpg_ext):.4f}, "
            f"BER: {jpg_ber:.4f}"
        )