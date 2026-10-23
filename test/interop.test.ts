/**
 * Interop tests against ffmpeg.
 *
 * Round-tripping through our own code proves only internal consistency: an
 * inverted sign convention or a non-standard reconstruction formula round-trips
 * perfectly and still sounds wrong in every other program. These tests check the
 * thing that actually matters — that files audiobox writes decode correctly
 * elsewhere, and that files written elsewhere decode correctly here.
 *
 * ffmpeg is not a dependency. When it is missing the whole suite skips, so
 * contributors without it still get a green run; CI installs it so the checks
 * genuinely execute.
 */

import { test, describe, before } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { decodeWav, encodeWav } from '../src/formats/wav.ts';
import type { SampleFormat } from '../src/types.ts';

function hasFfmpeg(): boolean {
  try {
    execFileSync('ffmpeg', ['-version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

const FFMPEG = hasFfmpeg();

/** True when ffmpeg decodes 4-bit IMA ADPCM with the IMA reference formula, as 8.1 and later do. */
function ffmpegDecodesImaReference(): boolean {
  const banner = execFileSync('ffmpeg', ['-version']).toString();
  const release = /^ffmpeg version n?(\d+)\.(\d+)/.exec(banner);
  if (release) return Number(release[1]) * 100 + Number(release[2]) >= 801;
  // Git builds count commits since tag N; the change landed as N-121180.
  // Any other banner gets the loose check, which both formulas pass.
  const git = /^ffmpeg version N-(\d+)-/.exec(banner);
  return git !== null && Number(git[1]) >= 121180;
}

let dir: string;

/** Reads a raw little-endian float32 file as interleaved samples. */
function readRawF32(path: string): Float32Array {
  const b = readFileSync(path);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}

/** Largest absolute difference between planar channels and interleaved reference. */
function maxDiff(planar: readonly Float32Array[], interleaved: Float32Array): number {
  const channels = planar.length;
  const frames = planar[0]!.length;
  // A shorter reference would compare against undefined, and NaN never counts as worse.
  assert.ok(interleaved.length >= frames * channels, 'the reference has fewer frames');
  let worst = 0;
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < channels; c++) {
      const d = Math.abs(planar[c]![i]! - interleaved[i * channels + c]!);
      if (d > worst) worst = d;
    }
  }
  return worst;
}

function sine(frames: number, rate: number, hz: number, amplitude = 0.8): Float32Array {
  const a = new Float32Array(frames);
  for (let i = 0; i < frames; i++) a[i] = Math.sin((2 * Math.PI * hz * i) / rate) * amplitude;
  return a;
}

describe('ffmpeg interop', { skip: FFMPEG ? false : 'ffmpeg not installed' }, () => {
  before(() => {
    dir = mkdtempSync(join(tmpdir(), 'audiobox-interop-'));
  });

  test.after(() => {
    if (dir) rmSync(dir, { recursive: true, force: true });
  });

  describe('files we write decode correctly in ffmpeg', () => {
    // A-law and µ-law are lossy, but the *decoded* result is deterministic, so
    // even those must match ffmpeg exactly rather than approximately.
    const formats: SampleFormat[] = ['u8', 's16', 's24', 's32', 'f32', 'f64', 'alaw', 'ulaw'];

    for (const sampleFormat of formats) {
      test(sampleFormat, () => {
        const rate = 44100;
        const frames = 4410;
        const source = [sine(frames, rate, 440), sine(frames, rate, 880, 0.4)];

        const wavPath = join(dir, `out-${sampleFormat}.wav`);
        const rawPath = join(dir, `out-${sampleFormat}.raw`);
        writeFileSync(wavPath, encodeWav(source, rate, { sampleFormat }));

        execFileSync('ffmpeg', ['-v', 'error', '-i', wavPath, '-f', 'f32le', rawPath, '-y']);

        const ours = decodeWav(readFileSync(wavPath)).channelData;
        const theirs = readRawF32(rawPath);

        assert.equal(
          maxDiff(ours, theirs),
          0,
          `${sampleFormat}: ffmpeg and audiobox must decode our file identically`,
        );
      });
    }

    test('ffmpeg reports the container fields we intended', () => {
      const wavPath = join(dir, 'probe.wav');
      writeFileSync(
        wavPath,
        encodeWav([sine(480, 48000, 440), sine(480, 48000, 440)], 48000, { sampleFormat: 's24' }),
      );

      const out = execFileSync('ffprobe', [
        '-v', 'error',
        '-show_entries', 'stream=codec_name,sample_rate,channels',
        '-of', 'csv=p=0',
        wavPath,
      ]).toString().trim();

      assert.equal(out, 'pcm_s24le,48000,2');
    });
  });

  describe('files ffmpeg writes decode correctly here', () => {
    /** Renders a tone with ffmpeg in `codec`, returning the file path. */
    function ffmpegWrite(name: string, codec: string, rate: number, channels: number, extra: string[] = []): string {
      const path = join(dir, name);
      execFileSync('ffmpeg', [
        '-v', 'error',
        '-f', 'lavfi',
        '-i', `sine=frequency=440:sample_rate=${rate}:duration=0.1`,
        '-ac', String(channels),
        '-acodec', codec,
        ...extra,
        path,
        '-y',
      ]);
      return path;
    }

    test('24-bit PCM', () => {
      const decoded = decodeWav(readFileSync(ffmpegWrite('in-s24.wav', 'pcm_s24le', 44100, 2)));
      assert.equal(decoded.info.sampleFormat, 's24');
      assert.equal(decoded.sampleRate, 44100);
      assert.equal(decoded.channelData.length, 2);
      assert.equal(decoded.channelData[0]!.length, 4410);
    });

    test('A-law at telephony rates', () => {
      const decoded = decodeWav(readFileSync(ffmpegWrite('in-alaw.wav', 'pcm_alaw', 8000, 1)));
      assert.equal(decoded.info.sampleFormat, 'alaw');
      assert.equal(decoded.sampleRate, 8000);
      assert.equal(decoded.channelData[0]!.length, 800);
      assert.equal(decoded.info.lossy, true);
    });

    test('RF64, the 64-bit RIFF extension', () => {
      const path = ffmpegWrite('in-rf64.wav', 'pcm_f32le', 44100, 2, ['-rf64', 'always']);
      const decoded = decodeWav(readFileSync(path));
      assert.equal(decoded.info.sampleFormat, 'f32');
      assert.equal(decoded.channelData[0]!.length, 4410);
    });

    test('IMA ADPCM matches ffmpeg, exactly from 8.1', () => {
      const path = ffmpegWrite('in-adpcm.wav', 'adpcm_ima_wav', 44100, 2);
      const rawPath = join(dir, 'in-adpcm.raw');
      execFileSync('ffmpeg', ['-v', 'error', '-i', path, '-f', 'f32le', rawPath, '-y']);

      const decoded = decodeWav(readFileSync(path));
      assert.equal(decoded.info.sampleFormat, 'ima-adpcm');
      assert.equal(decoded.info.lossy, true);
      const diff = maxDiff(decoded.channelData, readRawF32(rawPath));
      if (ffmpegDecodesImaReference()) {
        assert.equal(diff, 0, 'ADPCM must match ffmpeg exactly, not approximately');
      } else {
        // Older ffmpeg drifts 29 LSB from the reference on this input; a real error is far larger.
        const lsb = diff * 32768;
        assert.ok(lsb < 64, `ADPCM is ${lsb} LSB away from ffmpeg's pre-8.1 decode`);
      }
    });
  });
});
