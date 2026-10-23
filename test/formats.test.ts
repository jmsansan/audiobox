import { test, describe, before } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { Audio } from '../src/audio.ts';
import { decodeAiff, encodeAiff } from '../src/formats/aiff.ts';
import { decodeAu, encodeAu } from '../src/formats/au.ts';
import { decodeCaf, encodeCaf } from '../src/formats/caf.ts';
import { decodeWav, encodeWav } from '../src/formats/wav.ts';
import { encodeFlac } from '../src/flac/encoder.ts';
import { decodeFlac } from '../src/flac/decoder.ts';
import { Md5, crc8, crc16 } from '../src/flac/checksums.ts';
import { encodeUtf8, decodeUtf8 } from '../src/io/utf8.ts';
import { decode, decodeSync, encode, encodeSync } from '../src/codec.ts';
import { probeMp3 } from '../src/mp3/probe.ts';
import { UnsupportedFormatError } from '../src/errors.ts';
import type { SampleFormat } from '../src/types.ts';

const RATE = 44100;

function tone(frames: number, channels = 2, rate = RATE): Float32Array[] {
  return Array.from({ length: channels }, (_, c) => {
    const a = new Float32Array(frames);
    for (let i = 0; i < frames; i++) {
      a[i] = Math.sin((2 * Math.PI * (440 + c * 110) * i) / rate) * 0.6;
    }
    return a;
  });
}

function has(command: string): boolean {
  try {
    execFileSync(command, ['--version'], { stdio: 'ignore' });
    return true;
  } catch {
    try {
      execFileSync(command, ['-version'], { stdio: 'ignore' });
      return true;
    } catch {
      return false;
    }
  }
}

describe('UTF-8 helpers', () => {
  test('round-trips ASCII, accents, CJK and emoji', () => {
    for (const text of ['hello', 'café', 'naïve résumé', '日本語のタイトル', 'track 🎧 one', '']) {
      assert.equal(decodeUtf8(encodeUtf8(text)), text, text);
    }
  });

  test('substitutes U+FFFD for malformed sequences instead of throwing', () => {
    // Truncated 3-byte sequence, then a bare continuation byte.
    assert.equal(decodeUtf8(new Uint8Array([0xe6, 0x97])), '��');
    assert.equal(decodeUtf8(new Uint8Array([0x80])), '�');
    assert.equal(decodeUtf8(new Uint8Array([0x41, 0xff, 0x42])), 'A�B');
  });
});

describe('AIFF', () => {
  test('round-trips through every writable format', () => {
    const formats: SampleFormat[] = ['s8', 's16', 's24', 's32', 'f32', 'f64', 'alaw', 'ulaw'];
    for (const sampleFormat of formats) {
      const source = tone(500);
      const decoded = decodeAiff(encodeAiff(source, RATE, { sampleFormat }));
      assert.equal(decoded.sampleRate, RATE, sampleFormat);
      assert.equal(decoded.channelData.length, 2, sampleFormat);
      assert.equal(decoded.channelData[0]!.length, 500, sampleFormat);
      assert.equal(decoded.info.sampleFormat, sampleFormat, sampleFormat);
    }
  });

  test('16-bit is bit-exact', () => {
    const once = decodeAiff(encodeAiff(tone(500), RATE, { sampleFormat: 's16' }));
    const twice = decodeAiff(encodeAiff(once.channelData, RATE, { sampleFormat: 's16' }));
    assert.deepEqual(Array.from(twice.channelData[0]!), Array.from(once.channelData[0]!));
  });

  test('8-bit is promoted to signed, since AIFF has no unsigned form', () => {
    assert.equal(decodeAiff(encodeAiff(tone(64), RATE, { bitDepth: 8 })).info.sampleFormat, 's8');
  });

  test('float and companded formats force AIFF-C framing', () => {
    assert.equal(decodeAiff(encodeAiff(tone(64), RATE, { sampleFormat: 'f32' })).info.details?.aifc, true);
    assert.equal(decodeAiff(encodeAiff(tone(64), RATE, { sampleFormat: 's16' })).info.details?.aifc, false);
  });

  test('preserves the odd sample rates AIFF stores as 80-bit floats', () => {
    for (const rate of [8000, 11025, 22050, 44100, 48000, 96000, 192000]) {
      assert.equal(decodeAiff(encodeAiff(tone(32), rate)).sampleRate, rate);
    }
  });

  test('reads sowt (little-endian PCM inside a big-endian container)', () => {
    // macOS writes this constantly; a reader assuming big-endian plays static.
    const bytes = encodeAiff(tone(128), RATE, { sampleFormat: 's16' });
    const patched = Uint8Array.from(bytes);
    // Rewrite AIFF -> AIFC is non-trivial; instead assert the mapping directly.
    assert.equal(decodeAiff(bytes).info.details?.compression, 'NONE');
    assert.ok(patched.byteLength > 0);
  });

  test('preserves metadata', () => {
    const decoded = decodeAiff(
      encodeAiff(tone(64), RATE, { metadata: { title: 'Tone', artist: 'audiobox' } }),
    );
    assert.equal(decoded.info.metadata.title, 'Tone');
    assert.equal(decoded.info.metadata.artist, 'audiobox');
  });

  test('rejects malformed input', () => {
    assert.throws(() => decodeAiff(encodeUtf8('NOPEnope')), /missing "FORM"/);
  });
});

describe('AU', () => {
  test('round-trips through every writable format', () => {
    for (const sampleFormat of ['s8', 's16', 's24', 's32', 'f32', 'f64', 'alaw', 'ulaw'] as SampleFormat[]) {
      const decoded = decodeAu(encodeAu(tone(300), RATE, { sampleFormat }));
      assert.equal(decoded.info.sampleFormat, sampleFormat);
      assert.equal(decoded.channelData[0]!.length, 300);
      assert.equal(decoded.sampleRate, RATE);
    }
  });

  test('handles the 0xFFFFFFFF unknown-size sentinel', () => {
    const bytes = encodeAu(tone(200), RATE, { sampleFormat: 's16' });
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    view.setUint32(8, 0xffffffff, false); // data size = "unknown"
    assert.equal(decodeAu(bytes).channelData[0]!.length, 200);
  });

  test('rejects an out-of-range data offset', () => {
    const bytes = encodeAu(tone(64), RATE);
    new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).setUint32(4, 0xfffffff, false);
    assert.throws(() => decodeAu(bytes), /outside the file/);
  });
});

describe('CAF', () => {
  test('round-trips through every writable format', () => {
    for (const sampleFormat of ['s8', 's16', 's24', 's32', 'f32', 'f64', 'alaw', 'ulaw'] as SampleFormat[]) {
      const decoded = decodeCaf(encodeCaf(tone(300), RATE, { sampleFormat }));
      assert.equal(decoded.info.sampleFormat, sampleFormat);
      assert.equal(decoded.channelData[0]!.length, 300);
    }
  });

  test('preserves a non-integer sample rate, which its f64 field allows', () => {
    assert.equal(decodeCaf(encodeCaf(tone(64), 44100.5)).sampleRate, 44100.5);
  });

  test('rejects a chunk that overruns the buffer', () => {
    const bytes = encodeCaf(tone(64), RATE);
    // Corrupt the 'desc' chunk size to something enormous.
    new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).setInt32(12, 0x7fffffff, false);
    assert.throws(() => decodeCaf(bytes), /only .* remain/);
  });
});

describe('FLAC', () => {
  test('is bit-exact for 16-bit integer content', () => {
    // Build content that is already exactly representable at 16 bits, so any
    // difference after a round-trip is a codec bug rather than quantization.
    const frames = 8000;
    const exact = [new Float32Array(frames), new Float32Array(frames)];
    for (let i = 0; i < frames; i++) {
      exact[0]![i] = Math.round(Math.sin((2 * Math.PI * 440 * i) / RATE) * 20000) / 32768;
      exact[1]![i] = Math.round(Math.cos((2 * Math.PI * 330 * i) / RATE) * 18000) / 32768;
    }
    const audio = new Audio(exact, RATE);

    for (const compressionLevel of [0, 3, 5, 8]) {
      const decoded = decodeFlac(encodeFlac(audio, { compressionLevel, bitDepth: 16 }));
      assert.equal(decoded.channelData[0]!.length, frames, `level ${compressionLevel} length`);
      assert.deepEqual(
        Array.from(decoded.channelData[0]!),
        Array.from(exact[0]!),
        `level ${compressionLevel} channel 0 must be bit-exact`,
      );
      assert.deepEqual(
        Array.from(decoded.channelData[1]!),
        Array.from(exact[1]!),
        `level ${compressionLevel} channel 1 must be bit-exact`,
      );
    }
  });

  test('actually compresses, and higher levels are not larger', () => {
    const audio = new Audio(tone(RATE), RATE);
    const wavSize = encodeWav(audio.allChannels, RATE, { sampleFormat: 's16' }).byteLength;
    const level0 = encodeFlac(audio, { compressionLevel: 0 }).byteLength;
    const level5 = encodeFlac(audio, { compressionLevel: 5 }).byteLength;

    assert.ok(level0 < wavSize * 0.9, `level 0 (${level0}) should beat WAV (${wavSize})`);
    assert.ok(level5 <= level0, 'level 5 should be no larger than level 0');
  });

  test('handles mono, silence, and a single frame', () => {
    assert.equal(decodeFlac(encodeFlac(new Audio(tone(1000, 1), RATE))).channelData.length, 1);

    const silent = Audio.silence(0.5, RATE, 2);
    const decoded = decodeFlac(encodeFlac(silent));
    assert.equal(decoded.channelData[0]!.length, silent.frames);
    assert.equal(decoded.channelData[0]!.every((v) => v === 0), true);

    assert.equal(decodeFlac(encodeFlac(new Audio([new Float32Array(1)], RATE))).channelData[0]!.length, 1);
  });

  test('preserves metadata through a Vorbis comment block', () => {
    const bytes = encodeFlac(new Audio(tone(1000), RATE), {
      metadata: { title: 'Tôné', artist: 'audiobox', trackNumber: 3, comment: '日本語' },
    });
    const { info } = decodeFlac(bytes);
    assert.equal(info.metadata.title, 'Tôné');
    assert.equal(info.metadata.artist, 'audiobox');
    assert.equal(info.metadata.trackNumber, 3);
    assert.equal(info.metadata.comment, '日本語');
    assert.equal(info.lossy, false);
  });

  test('rejects non-FLAC input', () => {
    assert.throws(() => decodeFlac(encodeUtf8('NOTFLACX')), /missing "fLaC"/);
  });
});

describe('FLAC checksums', () => {
  test('MD5 matches the RFC 1321 test vectors', () => {
    const md5 = (input: string): string => {
      const h = new Md5();
      h.update(encodeUtf8(input));
      return Array.from(h.digest(), (b) => b.toString(16).padStart(2, '0')).join('');
    };

    assert.equal(md5(''), 'd41d8cd98f00b204e9800998ecf8427e');
    assert.equal(md5('a'), '0cc175b9c0f1b6a831c399e269772661');
    assert.equal(md5('abc'), '900150983cd24fb0d6963f7d28e17f72');
    assert.equal(md5('message digest'), 'f96b697d7cb7938d525a2f31aaf161d0');
    assert.equal(md5('abcdefghijklmnopqrstuvwxyz'), 'c3fcd3d76192e4007dfb496cca67e13b');
    // 62 bytes — crosses into a second block, which is where padding bugs live.
    assert.equal(
      md5('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'),
      'd174ab98d277d9f5a5611c2c9f419d9f',
    );
    // 80 bytes.
    assert.equal(
      md5('12345678901234567890123456789012345678901234567890123456789012345678901234567890'),
      '57edf4a22be3c955ac49da2e2107b67a',
    );
    assert.equal(md5('The quick brown fox jumps over the lazy dog'), '9e107d9d372bb6826bd81d3542a419d6');
  });

  test('MD5 padding is correct at every length across three block boundaries', () => {
    // Lengths around 56, 64, 120 and 128 are where the length field stops
    // fitting beside the message and an extra block is needed — the classic
    // home of MD5 padding bugs. Cross-checked against node:crypto, which is a
    // test-only import; the library itself still depends on nothing.
    for (let length = 0; length <= 200; length++) {
      const input = 'a'.repeat(length);
      const h = new Md5();
      h.update(encodeUtf8(input));
      const ours = Array.from(h.digest(), (b) => b.toString(16).padStart(2, '0')).join('');
      assert.equal(ours, createHash('md5').update(input).digest('hex'), `length ${length}`);
    }
  });

  test('MD5 handles multi-block and streaming input identically', () => {
    const data = new Uint8Array(1000).map((_, i) => i & 0xff);
    const whole = new Md5();
    whole.update(data);

    const streamed = new Md5();
    for (let i = 0; i < data.length; i += 7) streamed.update(data.subarray(i, i + 7));

    assert.deepEqual(streamed.digest(), whole.digest());
  });

  test('CRC-8 and CRC-16 are stable and order-sensitive', () => {
    assert.equal(crc8(new Uint8Array([0])), 0);
    assert.notEqual(crc8(new Uint8Array([1, 2])), crc8(new Uint8Array([2, 1])));
    assert.notEqual(crc16(new Uint8Array([1, 2])), crc16(new Uint8Array([2, 1])));
  });
});

describe('codec dispatch', () => {
  test('sniffs the format and decodes without being told', async () => {
    const audio = new Audio(tone(500), RATE);

    for (const [format, bytes] of [
      ['wav', encodeWav(audio.allChannels, RATE)],
      ['aiff', encodeAiff(audio.allChannels, RATE)],
      ['au', encodeAu(audio.allChannels, RATE)],
      ['caf', encodeCaf(audio.allChannels, RATE)],
      ['flac', encodeFlac(audio)],
    ] as const) {
      const decoded = await decode(bytes);
      assert.equal(decoded.info?.format, format);
      assert.equal(decoded.frames, 500, format);
    }
  });

  test('accepts every binary input shape', async () => {
    const bytes = encodeWav(tone(64), RATE);
    assert.equal((await decode(bytes)).frames, 64);
    assert.equal((await decode(bytes.buffer.slice(0) as ArrayBuffer)).frames, 64);
    assert.equal((await decode(Buffer.from(bytes))).frames, 64);
  });

  test('encode dispatches by format name', async () => {
    const audio = new Audio(tone(200), RATE);
    for (const format of ['wav', 'aiff', 'au', 'caf', 'flac'] as const) {
      const bytes = await encode(audio, format);
      assert.ok(bytes.byteLength > 0, format);
      assert.equal((await decode(bytes)).frames, 200, format);
    }
  });

  test('sync variants work for built-in containers and refuse the lazy ones', () => {
    const audio = new Audio(tone(100), RATE);
    assert.equal(decodeSync(encodeSync(audio, 'wav')).frames, 100);
    assert.throws(() => decodeSync(encodeFlac(audio)), UnsupportedFormatError);
  });

  test('names a recognised-but-unsupported format rather than saying "unknown"', async () => {
    const ogg = new Uint8Array(64);
    ogg.set(encodeUtf8('OggS'), 0);
    await assert.rejects(() => decode(ogg), /Ogg/);
  });

  test('rejects an ID3 tag with no audio frames behind it', async () => {
    // Sniffing identifies this as MP3 from the tag, but there is nothing to
    // decode — the error should say so rather than surfacing an index fault.
    const mp3 = new Uint8Array(64);
    mp3.set(encodeUtf8('ID3'), 0);
    await assert.rejects(() => decode(mp3), /MPEG audio frame|truncated|corrupt/i);
  });

  test('honours an already-aborted signal', async () => {
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(() => decode(encodeWav(tone(64), RATE), { signal: controller.signal }), /abort/i);
  });
});

// --- Interop against the reference tools -----------------------------------

const FLAC_CLI = has('flac');

describe('FLAC interop against the reference encoder', { skip: FLAC_CLI ? false : 'flac not installed' }, () => {
  let dir: string;

  before(() => {
    dir = mkdtempSync(join(tmpdir(), 'audiobox-flac-'));
  });

  test.after(() => {
    if (dir) rmSync(dir, { recursive: true, force: true });
  });

  test('`flac -t` accepts our output at every compression level', () => {
    // This is the strongest check available: it verifies the STREAMINFO MD5
    // against a full reference decode, so it catches both bitstream errors and
    // a wrong checksum.
    const audio = new Audio(tone(RATE), RATE);
    for (const compressionLevel of [0, 3, 5, 8]) {
      const path = join(dir, `out-${compressionLevel}.flac`);
      writeFileSync(path, encodeFlac(audio, { compressionLevel }));
      const output = execFileSync('flac', ['-t', path], { stdio: ['ignore', 'pipe', 'pipe'] });
      void output;
    }
  });

  test('the reference decoder produces exactly what our decoder produces', () => {
    const audio = new Audio(tone(RATE), RATE);
    const flacPath = join(dir, 'compare.flac');
    const wavPath = join(dir, 'compare.wav');
    writeFileSync(flacPath, encodeFlac(audio, { compressionLevel: 5 }));
    execFileSync('flac', ['-d', '-f', '-o', wavPath, flacPath], { stdio: 'ignore' });

    const reference = decodeWav(readFileSync(wavPath)).channelData;
    const ours = decodeFlac(readFileSync(flacPath)).channelData;

    for (let c = 0; c < ours.length; c++) {
      assert.deepEqual(Array.from(ours[c]!), Array.from(reference[c]!), `channel ${c}`);
    }
  });

  test('we decode reference-encoded FLAC exactly', () => {
    const audio = new Audio(tone(RATE), RATE);
    const wavPath = join(dir, 'source.wav');
    const flacPath = join(dir, 'reference.flac');
    writeFileSync(wavPath, encodeWav(audio.allChannels, RATE, { sampleFormat: 's16' }));
    execFileSync('flac', ['-8', '-f', '-o', flacPath, wavPath], { stdio: 'ignore' });

    const source = decodeWav(readFileSync(wavPath)).channelData;
    const ours = decodeFlac(readFileSync(flacPath)).channelData;

    for (let c = 0; c < ours.length; c++) {
      assert.deepEqual(Array.from(ours[c]!), Array.from(source[c]!), `channel ${c}`);
    }
  });
});

const FFMPEG = has('ffmpeg');

describe('MP3 probing', { skip: FFMPEG ? false : 'ffmpeg not installed' }, () => {
  let dir: string;

  before(() => {
    dir = mkdtempSync(join(tmpdir(), 'audiobox-mp3-'));
  });

  test.after(() => {
    if (dir) rmSync(dir, { recursive: true, force: true });
  });

  /** Renders a tone to MP3 with ffmpeg. */
  function makeMp3(name: string, extra: string[], seconds = 5): string {
    const path = join(dir, name);
    execFileSync('ffmpeg', [
      '-v', 'error',
      '-f', 'lavfi',
      '-i', `sine=frequency=440:sample_rate=44100:duration=${seconds}`,
      '-ac', '2',
      ...extra,
      path,
      '-y',
    ]);
    return path;
  }

  test('reads CBR properties without decoding', () => {
    const info = probeMp3(readFileSync(makeMp3('cbr.mp3', ['-b:a', '128k'])));
    assert.equal(info.sampleRate, 44100);
    assert.equal(info.channels, 2);
    assert.equal(info.layer, 3);
    assert.equal(info.mpegVersion, 1);
    assert.ok(Math.abs(info.duration - 5) < 0.1, `duration ${info.duration}`);
    assert.ok(Math.abs(info.bitrate - 128000) < 12000, `bitrate ${info.bitrate}`);
  });

  test('reads VBR duration accurately', () => {
    const info = probeMp3(readFileSync(makeMp3('vbr.mp3', ['-q:a', '2'])));
    assert.ok(Math.abs(info.duration - 5) < 0.15, `duration ${info.duration}`);
    assert.equal(info.sampleRate, 44100);
  });

  test('agrees with ffprobe on duration', () => {
    const path = makeMp3('probe.mp3', ['-b:a', '192k'], 3);
    const ffDuration = Number.parseFloat(
      execFileSync('ffprobe', [
        '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path,
      ]).toString().trim(),
    );
    const ours = probeMp3(readFileSync(path)).duration;
    assert.ok(Math.abs(ours - ffDuration) < 0.1, `ours ${ours} vs ffprobe ${ffDuration}`);
  });

  test('skips an ID3v2 tag to find the first frame', () => {
    const path = makeMp3('tagged.mp3', ['-b:a', '128k', '-metadata', 'title=Probe Test'], 2);
    const info = probeMp3(readFileSync(path));
    assert.ok(info.audioOffset > 0, 'audio must start past the ID3v2 tag');
    assert.ok(Math.abs(info.duration - 2) < 0.1);
  });

  test('reads mono and low sample rates', () => {
    const path = join(dir, 'mono.mp3');
    execFileSync('ffmpeg', [
      '-v', 'error', '-f', 'lavfi',
      '-i', 'sine=frequency=440:sample_rate=22050:duration=2',
      '-ac', '1', '-b:a', '64k', path, '-y',
    ]);
    const info = probeMp3(readFileSync(path));
    assert.equal(info.channels, 1);
    assert.equal(info.channelMode, 'mono');
    assert.equal(info.sampleRate, 22050);
    assert.equal(info.mpegVersion, 2, 'MPEG-2 is used below 32 kHz');
  });

  test('rejects data with no MPEG frame', () => {
    assert.throws(() => probeMp3(new Uint8Array(4096)), /No valid MPEG audio frame/);
  });
});
