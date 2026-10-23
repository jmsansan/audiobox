/**
 * IMA/DVI ADPCM decoding.
 *
 * A 4:1 lossy compression that stores each sample as a 4-bit delta against an
 * adaptive step size. Long obsolete for music, but still the default in a lot of
 * telephony, game-asset, and embedded-recorder output — exactly the awkward
 * files that make a "convert my audio" tool useful. Decode is supported;
 * encoding to ADPCM is not, because nothing benefits from new ADPCM in 2026.
 *
 * Data is organised in fixed-size blocks. Each block restarts the predictor from
 * a stored value, so a corrupt block cannot poison the rest of the stream — and
 * it means blocks can be decoded independently.
 */

import { DecodeError } from '../errors.ts';

/** Step-index adjustment per nibble magnitude. */
const INDEX_TABLE = new Int8Array([-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]);

/** The 89-entry logarithmic step-size ladder from the IMA specification. */
const STEP_TABLE = new Int32Array([
  7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73,
  80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494,
  544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499,
  2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487,
  12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
]);

/** Frames stored in one ADPCM block. */
export function adpcmSamplesPerBlock(blockAlign: number, channels: number): number {
  // Each channel spends 4 bytes on its preamble; the rest is two samples per byte.
  const payload = blockAlign - 4 * channels;
  if (payload < 0) {
    throw new DecodeError(
      `ADPCM blockAlign ${blockAlign} is too small for ${channels} channel(s).`,
    );
  }
  return 1 + Math.floor((payload * 2) / channels);
}

/**
 * Decodes IMA ADPCM into planar float channels.
 *
 * @param bytes Complete ADPCM data (all blocks, concatenated).
 * @param channels Channel count.
 * @param blockAlign Bytes per block, from the container's `fmt ` chunk.
 * @param totalFrames Frames to produce; the tail of a final partial block is dropped.
 */
export function decodeImaAdpcm(
  bytes: Uint8Array,
  channels: number,
  blockAlign: number,
  totalFrames: number,
): Float32Array[] {
  const out: Float32Array[] = [];
  for (let c = 0; c < channels; c++) out.push(new Float32Array(totalFrames));

  const perBlock = adpcmSamplesPerBlock(blockAlign, channels);
  const predictors = new Int32Array(channels);
  const stepIndices = new Int32Array(channels);

  let written = 0;
  for (let base = 0; base + blockAlign <= bytes.byteLength && written < totalFrames; base += blockAlign) {
    // Block preamble: one 4-byte header per channel.
    for (let c = 0; c < channels; c++) {
      const p = base + c * 4;
      let predictor = bytes[p]! | (bytes[p + 1]! << 8);
      if (predictor & 0x8000) predictor -= 0x10000;
      predictors[c] = predictor;
      // A corrupt index would index outside STEP_TABLE; clamping is cheaper than
      // a bounds check in the inner loop and matches what real decoders do.
      stepIndices[c] = Math.max(0, Math.min(88, bytes[p + 2]!));

      if (written < totalFrames) out[c]![written] = predictor / 32768;
    }
    written++;

    // Payload: 4-byte groups, cycling through channels.
    let offset = base + 4 * channels;
    let produced = 1;

    while (produced < perBlock && written < totalFrames && offset + 4 * channels <= bytes.byteLength) {
      for (let c = 0; c < channels; c++) {
        const groupStart = offset + c * 4;
        for (let byteIdx = 0; byteIdx < 4; byteIdx++) {
          const byte = bytes[groupStart + byteIdx]!;
          const slot = written + byteIdx * 2;
          // Low nibble first, then high.
          const lo = decodeNibble(byte & 0x0f, c, predictors, stepIndices);
          if (slot < totalFrames && produced + byteIdx * 2 < perBlock) {
            out[c]![slot] = lo / 32768;
          }
          const hi = decodeNibble((byte >> 4) & 0x0f, c, predictors, stepIndices);
          if (slot + 1 < totalFrames && produced + byteIdx * 2 + 1 < perBlock) {
            out[c]![slot + 1] = hi / 32768;
          }
        }
      }
      const advanced = Math.min(8, perBlock - produced);
      produced += advanced;
      written += advanced;
      offset += 4 * channels;
    }
  }

  return out;
}

/** Advances one channel's predictor by a single 4-bit delta. */
function decodeNibble(
  nibble: number,
  channel: number,
  predictors: Int32Array,
  stepIndices: Int32Array,
): number {
  const step = STEP_TABLE[stepIndices[channel]!]!;
  // The IMA reference sums separately truncated shifts, as ffmpeg 8.1+ does. A single multiply,
  // ((2 * (nibble & 7) + 1) * step) >> 3, runs up to 2 LSB larger and the error accumulates.
  let diff = step >> 3;
  if (nibble & 4) diff += step;
  if (nibble & 2) diff += step >> 1;
  if (nibble & 1) diff += step >> 2;

  let predictor = predictors[channel]! + (nibble & 8 ? -diff : diff);
  if (predictor > 32767) predictor = 32767;
  else if (predictor < -32768) predictor = -32768;
  predictors[channel] = predictor;

  const nextIndex = stepIndices[channel]! + INDEX_TABLE[nibble]!;
  stepIndices[channel] = nextIndex < 0 ? 0 : nextIndex > 88 ? 88 : nextIndex;

  return predictor;
}
