import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

import { decodeWav, encodeWav } from '../src/formats/wav.ts';
import {
  decodeInterleaved,
  encodeInterleaved,
  bytesPerSample,
  linearToALaw,
  linearToULaw,
} from '../src/pcm/convert.ts';
import { Writer } from '../src/io/writer.ts';
import { decodeImaAdpcm } from '../src/pcm/adpcm.ts';
import { DecodeError, LimitExceededError } from '../src/errors.ts';
import type { SampleFormat } from '../src/types.ts';
import {
  ALAW_DECODE_REFERENCE,
  ULAW_DECODE_REFERENCE,
  ENCODE_INPUTS,
  ALAW_ENCODE_REFERENCE,
  ULAW_ENCODE_REFERENCE,
} from './fixtures/g711-reference.ts';

/** Every distinct integer value representable at `bits`, as floats. */
function fullIntegerSweep(bits: number): Float32Array {
  const scale = 2 ** (bits - 1);
  const count = Math.min(2 ** bits, 4096);
  const step = Math.max(1, Math.floor((2 ** bits) / count));
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = (-scale + i * step) / scale;
  }
  return out;
}

describe('PCM conversion', () => {
  // The gate for this milestone: anything up to 24 bits must survive a
  // round-trip with no change at all. float32 has a 24-bit mantissa, so this is
  // exactly the range where bit-exactness is achievable.
  const exactFormats: Array<[SampleFormat, number]> = [
    ['u8', 8],
    ['s8', 8],
    ['s16', 16],
    ['s24', 24],
  ];

  for (const [format, bits] of exactFormats) {
    test(`${format} round-trips every value bit-exactly`, () => {
      const original = fullIntegerSweep(bits);
      for (const littleEndian of [true, false]) {
        const packed = encodeInterleaved([original], format, littleEndian);
        const [decoded] = decodeInterleaved(packed, format, littleEndian, 1, original.length);
        assert.deepEqual(
          Array.from(decoded!),
          Array.from(original),
          `${format} ${littleEndian ? 'LE' : 'BE'}`,
        );
      }
    });
  }

  test('f32 round-trips exactly; f64 round-trips within float32 precision', () => {
    const signal = new Float32Array(512);
    for (let i = 0; i < signal.length; i++) signal[i] = Math.sin(i / 7) * 0.9;

    const f32 = decodeInterleaved(encodeInterleaved([signal], 'f32', true), 'f32', true, 1, 512)[0]!;
    assert.deepEqual(Array.from(f32), Array.from(signal));

    const f64 = decodeInterleaved(encodeInterleaved([signal], 'f64', true), 'f64', true, 1, 512)[0]!;
    assert.deepEqual(Array.from(f64), Array.from(signal));
  });

  test('s32 round-trips to float32 precision, not bit-exactly', () => {
    // Documented limitation: float32 cannot hold 32 significant bits. Assert the
    // real behaviour so a future change to the storage type is caught here.
    const signal = new Float32Array([0, 0.5, -0.5, 0.25, -1]);
    const out = decodeInterleaved(
      encodeInterleaved([signal], 's32', true),
      's32',
      true,
      1,
      signal.length,
    )[0]!;
    for (let i = 0; i < signal.length; i++) {
      assert.ok(Math.abs(out[i]! - signal[i]!) < 1e-7, `sample ${i}: ${out[i]} vs ${signal[i]}`);
    }
  });

  test('clamps out-of-range input instead of wrapping', () => {
    // Wrapping would turn a loud transient into full-scale noise of the opposite
    // sign — the single most audible failure mode in a converter.
    const hot = new Float32Array([2, -2, 1.5, -1.5]);
    const out = decodeInterleaved(encodeInterleaved([hot], 's16', true), 's16', true, 1, 4)[0]!;
    assert.ok(out[0]! > 0.99 && out[0]! <= 1);
    assert.equal(out[1], -1);
    assert.ok(out[2]! > 0.99);
    assert.equal(out[3], -1);
  });

  test('interleaves and de-interleaves multi-channel data correctly', () => {
    const left = new Float32Array([0.1, 0.2, 0.3]);
    const right = new Float32Array([-0.1, -0.2, -0.3]);
    const packed = encodeInterleaved([left, right], 's16', true);
    assert.equal(packed.byteLength, 3 * 2 * bytesPerSample('s16'));

    const [l, r] = decodeInterleaved(packed, 's16', true, 2, 3);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(l![i]! - left[i]!) < 1e-4);
      assert.ok(Math.abs(r![i]! - right[i]!) < 1e-4);
    }
  });

  test('A-law and µ-law survive a round-trip within their companding error', () => {
    const signal = new Float32Array(256);
    for (let i = 0; i < signal.length; i++) signal[i] = Math.sin(i / 12) * 0.8;

    for (const format of ['alaw', 'ulaw'] as const) {
      const out = decodeInterleaved(
        encodeInterleaved([signal], format, true),
        format,
        true,
        1,
        signal.length,
      )[0]!;
      let worst = 0;
      for (let i = 0; i < signal.length; i++) {
        worst = Math.max(worst, Math.abs(out[i]! - signal[i]!));
      }
      // G.711 is ~8 bits of perceptual resolution; 2% peak error is expected.
      assert.ok(worst < 0.02, `${format} peak error ${worst}`);
    }
  });

  test('dither is off by default and deterministic when enabled', () => {
    const signal = new Float32Array(64).fill(0.30001);

    const plain = encodeInterleaved([signal], 's16', true);
    assert.deepEqual(plain, encodeInterleaved([signal], 's16', true));

    const a = encodeInterleaved([signal], 's16', true, { type: 'tpdf', seed: 42 });
    const b = encodeInterleaved([signal], 's16', true, { type: 'tpdf', seed: 42 });
    assert.deepEqual(a, b, 'same seed must produce identical output');
    assert.notDeepEqual(a, plain, 'dither must actually change the output');
  });

  test('rejects channels of differing lengths', () => {
    assert.throws(
      () => encodeInterleaved([new Float32Array(10), new Float32Array(9)], 's16', true),
      /same length/,
    );
  });
});

describe('WAV container', () => {
  const rate = 44100;

  function testSignal(frames: number, channels: number): Float32Array[] {
    return Array.from({ length: channels }, (_, c) => {
      const a = new Float32Array(frames);
      for (let i = 0; i < frames; i++) a[i] = Math.sin((i / rate) * 440 * 2 * Math.PI * (c + 1)) * 0.5;
      return a;
    });
  }

  test('round-trips through every writable sample format', () => {
    const formats: SampleFormat[] = ['u8', 's16', 's24', 's32', 'f32', 'f64', 'alaw', 'ulaw'];
    for (const sampleFormat of formats) {
      const original = testSignal(1000, 2);
      const bytes = encodeWav(original, rate, { sampleFormat });
      const decoded = decodeWav(bytes);

      assert.equal(decoded.sampleRate, rate, sampleFormat);
      assert.equal(decoded.channelData.length, 2, sampleFormat);
      assert.equal(decoded.channelData[0]!.length, 1000, sampleFormat);
      assert.equal(decoded.info.sampleFormat, sampleFormat, sampleFormat);
      assert.equal(decoded.info.format, 'wav');
    }
  });

  test('16-bit round-trip is bit-exact', () => {
    const original = testSignal(500, 1);
    const once = decodeWav(encodeWav(original, rate, { sampleFormat: 's16' }));
    const twice = decodeWav(encodeWav(once.channelData, rate, { sampleFormat: 's16' }));
    // A second pass must be a no-op; if it isn't, quantization is drifting.
    assert.deepEqual(Array.from(twice.channelData[0]!), Array.from(once.channelData[0]!));
  });

  test('bitDepth shorthand maps to the expected formats', () => {
    assert.equal(decodeWav(encodeWav(testSignal(10, 1), rate, { bitDepth: 8 })).info.sampleFormat, 'u8');
    assert.equal(decodeWav(encodeWav(testSignal(10, 1), rate, { bitDepth: 16 })).info.sampleFormat, 's16');
    assert.equal(decodeWav(encodeWav(testSignal(10, 1), rate, { bitDepth: 24 })).info.sampleFormat, 's24');
    assert.equal(decodeWav(encodeWav(testSignal(10, 1), rate, { bitDepth: 32 })).info.sampleFormat, 'f32');
  });

  test('writes EXTENSIBLE automatically for multichannel and high bit depth', () => {
    const surround = decodeWav(encodeWav(testSignal(64, 6), rate, { sampleFormat: 's16' }));
    assert.equal(surround.info.details?.formatTag, 1, 'subformat tag is unwrapped on read');
    assert.equal(surround.channelData.length, 6);
    assert.equal(surround.info.details?.channelMask, 0x3f, '5.1 speaker mask');

    // Plain stereo 16-bit needs no extension.
    const stereo = decodeWav(encodeWav(testSignal(64, 2), rate, { sampleFormat: 's16' }));
    assert.equal(stereo.info.details?.channelMask, undefined);
  });

  test('preserves metadata through a round-trip', () => {
    const bytes = encodeWav(testSignal(64, 1), rate, {
      metadata: {
        title: 'Test Tone',
        artist: 'audiobox',
        comment: 'A 440 Hz sine',
        trackNumber: 7,
      },
    });
    const { info } = decodeWav(bytes);
    assert.equal(info.metadata.title, 'Test Tone');
    assert.equal(info.metadata.artist, 'audiobox');
    assert.equal(info.metadata.comment, 'A 440 Hz sine');
    assert.equal(info.metadata.trackNumber, 7);
  });

  test('output length is even, as RIFF requires', () => {
    // An odd-length data chunk must be padded, or every strict reader rejects the file.
    const odd = encodeWav([new Float32Array(3)], rate, { sampleFormat: 'u8' });
    assert.equal(odd.byteLength % 2, 0);
    assert.equal(decodeWav(odd).channelData[0]!.length, 3);
  });

  test('skips unknown chunks instead of failing', () => {
    const valid = encodeWav(testSignal(32, 1), rate);
    const w = new Writer();
    w.ascii('RIFF');
    const sizeAt = w.reserve(4);
    const start = w.length;
    w.ascii('WAVE');
    // A junk chunk of odd length, which also exercises the pad-byte path.
    w.ascii('JUNK');
    w.u32(5, true);
    w.write(new Uint8Array([1, 2, 3, 4, 5]));
    w.align2();
    w.write(valid.subarray(12)); // everything after "RIFF<size>WAVE"
    w.patchU32(sizeAt, w.length - start, true);

    assert.equal(decodeWav(w.finish()).channelData[0]!.length, 32);
  });

  test('recovers audio from a truncated data chunk', () => {
    // Interrupted recordings are common; discarding the whole file would be worse
    // than returning the frames that survived.
    const full = encodeWav(testSignal(1000, 2), rate, { sampleFormat: 's16' });
    const truncated = full.subarray(0, full.byteLength - 1000);
    const decoded = decodeWav(truncated);
    assert.ok(decoded.channelData[0]!.length > 700);
    assert.ok(decoded.channelData[0]!.length < 1000);
  });

  describe('hostile input', () => {
    test('rejects a chunk size that overruns the buffer', () => {
      const w = new Writer();
      w.ascii('RIFF');
      w.u32(1000, true);
      w.ascii('WAVE');
      w.ascii('fmt ');
      w.u32(0x7fffffff, true); // absurd declared size
      w.zeros(16);
      assert.throws(() => decodeWav(w.finish()), DecodeError);
    });

    test('rejects a header claiming an implausible allocation', () => {
      // 44-byte file, 4-billion-frame claim. The limit must trip before any
      // buffer is allocated.
      const w = new Writer();
      w.ascii('RIFF');
      w.u32(36, true);
      w.ascii('WAVE');
      w.ascii('fmt ');
      w.u32(16, true);
      w.u16(1, true);
      w.u16(64, true); // 64 channels
      w.u32(768000, true); // maximum permitted rate
      w.u32(0, true);
      w.u16(128, true);
      w.u16(16, true);
      w.ascii('data');
      w.u32(0, true);
      const bytes = w.finish();
      // Shape is legal but the limits must still be enforced on a real payload.
      assert.doesNotThrow(() => decodeWav(bytes));
      assert.throws(
        () => decodeWav(bytes, { limits: { maxChannels: 2 } }),
        LimitExceededError,
      );
    });

    test('rejects a reserved sample rate of zero', () => {
      const w = new Writer();
      w.ascii('RIFF');
      w.u32(36, true);
      w.ascii('WAVE');
      w.ascii('fmt ');
      w.u32(16, true);
      w.u16(1, true);
      w.u16(1, true);
      w.u32(0, true); // division hazard downstream
      w.u32(0, true);
      w.u16(2, true);
      w.u16(16, true);
      w.ascii('data');
      w.u32(2, true);
      w.u16(0, true);
      assert.throws(() => decodeWav(w.finish()), LimitExceededError);
    });

    test('reports a missing fmt or data chunk clearly', () => {
      const noFmt = new Writer();
      noFmt.ascii('RIFF');
      noFmt.u32(4, true);
      noFmt.ascii('WAVE');
      assert.throws(() => decodeWav(noFmt.finish()), /no "fmt " chunk/);
    });

    test('rejects non-RIFF and non-WAVE input', () => {
      assert.throws(() => decodeWav(new TextEncoder().encode('NOPEnope')), /Not a RIFF file/);

      const avi = new Writer();
      avi.ascii('RIFF');
      avi.u32(4, true);
      avi.ascii('AVI ');
      assert.throws(() => decodeWav(avi.finish()), /form type is "AVI "/);
    });

    test('names an unsupported format tag instead of failing opaquely', () => {
      const w = new Writer();
      w.ascii('RIFF');
      w.u32(36, true);
      w.ascii('WAVE');
      w.ascii('fmt ');
      w.u32(16, true);
      w.u16(0x0161, true); // WMA
      w.u16(2, true);
      w.u32(44100, true);
      w.u32(0, true);
      w.u16(4, true);
      w.u16(16, true);
      w.ascii('data');
      w.u32(0, true);
      assert.throws(() => decodeWav(w.finish()), /Unsupported WAV format tag 0x0161/);
    });
  });
});

describe('G.711 interop against the ITU reference', () => {
  test('every A-law and µ-law byte decodes to the reference value', () => {
    const allBytes = new Uint8Array(256).map((_, i) => i);

    const alaw = decodeInterleaved(allBytes, 'alaw', true, 1, 256)[0]!;
    const ulaw = decodeInterleaved(allBytes, 'ulaw', true, 1, 256)[0]!;

    for (let i = 0; i < 256; i++) {
      assert.equal(Math.round(alaw[i]! * 32768), ALAW_DECODE_REFERENCE[i], `A-law byte ${i}`);
      assert.equal(Math.round(ulaw[i]! * 32768), ULAW_DECODE_REFERENCE[i], `µ-law byte ${i}`);
    }
  });

  test('encoding matches the reference across the full 16-bit range', () => {
    for (let i = 0; i < ENCODE_INPUTS.length; i++) {
      const input = ENCODE_INPUTS[i]!;
      assert.equal(linearToALaw(input), ALAW_ENCODE_REFERENCE[i], `A-law of ${input}`);
      assert.equal(linearToULaw(input), ULAW_ENCODE_REFERENCE[i], `µ-law of ${input}`);
    }
  });
});

describe('IMA ADPCM', () => {
  test('reconstruction sums separately truncated shifts, as the IMA reference does', () => {
    // Predictor 0, step 7, nibbles 1 then 7. A single multiply would give [0, 2, 15].
    const block = Uint8Array.of(0, 0, 0, 0, 0x71, 0, 0, 0);
    const [mono] = decodeImaAdpcm(block, 1, 8, 3);
    assert.deepEqual(Array.from(mono!, (v) => v * 32768), [0, 1, 12]);
  });
});
