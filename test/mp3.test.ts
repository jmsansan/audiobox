/**
 * MP3 tests.
 *
 * Decoding is still under construction, so these cover what is implemented:
 * frame headers, tags, probing, and — the part most likely to hide a subtle bug
 * — side-information parsing and the bit reservoir.
 *
 * The reservoir check is the valuable one. If `main_data_begin` handling or the
 * MPEG-1/2 field widths are wrong, granules will claim more bits than the
 * assembled buffer holds, which shows up here rather than as noise in the audio
 * much later.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { BitReader, BitWriter } from '../src/io/bits.ts';
import {
  parseFrameHeader,
  findFrame,
  id3v2Size,
  readVbrHeader,
  MPEG1_LAYER3_BITRATE_VALUES,
} from '../src/mp3/frame.ts';
import { parseSideInfo, BitReservoir } from '../src/mp3/sideinfo.ts';
import type { GranuleInfo, SideInfo } from '../src/mp3/sideinfo.ts';
import { writeFrameHeader, writeSideInfo, FrameAssembler } from '../src/mp3/bitstream.ts';
import {
  createScalefactors,
  decodeScalefactorsMpeg1,
  decodeScalefactorsMpeg2,
  decodeHuffman,
} from '../src/mp3/huffman.ts';
import {
  HUFFMAN_TABS,
  HUFFMAN_TABLE_INDEX,
  COUNT1_TABLE_A,
  COUNT1_TABLE_B,
} from '../src/mp3/huffman-tables.ts';
import { probeMp3 } from '../src/mp3/probe.ts';
import { decodeMp3 } from '../src/mp3/decoder.ts';
import { AnalysisFilterbank, analyseGranule, ANALYSIS_WINDOW } from '../src/mp3/analysis.ts';
import {
  pairCost, writePair, regionCost, chooseTable,
  count1Cost, writeCount1,
} from '../src/mp3/huffman-encode.ts';
import { HUFFMAN_ENCODE_XLEN, HUFFMAN_ENCODE_YLEN } from '../src/mp3/huffman-encode-tables.ts';
import { reduceAlias, imdct, invertFrequency, SynthesisFilterbank } from '../src/mp3/imdct.ts';
import { decode, encode } from '../src/codec.ts';
import {
  SFB_LONG,
  SFB_SHORT,
  POW43,
  IMDCT_WINDOWS,
  SYNTH_WINDOW,
  ALIAS_CS,
  ALIAS_CA,
  HUFFMAN_LINBITS,
  SLEN1,
  SLEN2,
} from '../src/mp3/tables.ts';
import {
  createWorkspace,
  quantizeGranule,
  writeGranuleData,
  partition,
  chooseScalefacCompress,
} from '../src/mp3/quantize.ts';
import type { GranuleCoding } from '../src/mp3/quantize.ts';
import { requantize } from '../src/mp3/requantize.ts';
import { PsychoacousticModel } from '../src/mp3/psychoacoustic.ts';
import { fft } from '../src/analyze/fft.ts';
import { encodeMp3Channels, ENCODER_DELAY } from '../src/mp3/encoder.ts';
import { lameCrc } from '../src/mp3/xing.ts';
import { Audio } from '../src/audio.ts';

/** Presents a quantiser result to the decoder's requantiser. */
function codingToGranuleInfo(coding: GranuleCoding): GranuleInfo {
  return {
    part2_3_length: coding.part2_3Length,
    bigValues: coding.bigValues,
    globalGain: coding.globalGain,
    scalefacCompress: coding.scalefacCompress,
    windowSwitching: false,
    blockType: 0,
    mixedBlock: false,
    tableSelect: coding.tableSelect,
    subblockGain: [0, 0, 0],
    region0Count: coding.region0Count,
    region1Count: coding.region1Count,
    preflag: coding.preflag,
    scalefacScale: coding.scalefacScale,
    count1TableSelect: coding.count1TableSelect,
  };
}

function hasFfmpeg(): boolean {
  try {
    execFileSync('ffmpeg', ['-version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

const FFMPEG = hasFfmpeg();

describe('MP3 tables', () => {
  test('every sample rate has consistent scalefactor bands', () => {
    for (const rate of [44100, 48000, 32000, 22050, 24000, 16000, 11025, 12000, 8000]) {
      const long = SFB_LONG[rate]!;
      const short = SFB_SHORT[rate]!;

      assert.equal(long.length, 23, `${rate} long band count`);
      assert.equal(short.length, 14, `${rate} short band count`);
      assert.equal(long[0], 0, `${rate} long starts at 0`);
      assert.equal(long[22], 576, `${rate} long covers all 576 lines`);
      assert.equal(short[0], 0, `${rate} short starts at 0`);
      assert.equal(short[13], 192, `${rate} short covers 192 lines per window`);

      // Boundaries must be non-decreasing, or band widths go negative.
      for (let i = 1; i < long.length; i++) {
        assert.ok(long[i]! >= long[i - 1]!, `${rate} long band ${i} decreases`);
      }
      for (let i = 1; i < short.length; i++) {
        assert.ok(short[i]! >= short[i - 1]!, `${rate} short band ${i} decreases`);
      }
    }
  });

  test('computed tables have the right shape and values', () => {
    assert.equal(POW43[0], 0);
    assert.equal(POW43[1], 1);
    assert.ok(Math.abs(POW43[8]! - Math.pow(8, 4 / 3)) < 1e-4);

    assert.equal(SYNTH_WINDOW.length, 512);
    assert.equal(IMDCT_WINDOWS.length, 4);
    for (const window of IMDCT_WINDOWS) assert.equal(window.length, 36);

    // The long window is a half-sine: zero-ish at both ends, peak in the middle.
    assert.ok(IMDCT_WINDOWS[0]![17]! > 0.99);
    assert.ok(IMDCT_WINDOWS[0]![0]! < 0.05);
  });

  test('alias-reduction coefficients satisfy cs^2 + ca^2 = 1', () => {
    // They are built from a normalised butterfly, so this identity must hold
    // exactly; a typo in the c[] constants shows up here.
    for (let i = 0; i < 8; i++) {
      const sum = ALIAS_CS[i]! ** 2 + ALIAS_CA[i]! ** 2;
      assert.ok(Math.abs(sum - 1) < 1e-6, `band ${i}: cs²+ca² = ${sum}`);
    }
  });

  test('linbits table matches the specification', () => {
    assert.equal(HUFFMAN_LINBITS.length, 32);
    // Tables 0-15 never use escape bits.
    for (let i = 0; i < 16; i++) assert.equal(HUFFMAN_LINBITS[i], 0, `table ${i}`);
    assert.equal(HUFFMAN_LINBITS[16], 1);
    assert.equal(HUFFMAN_LINBITS[23], 13);
    assert.equal(HUFFMAN_LINBITS[24], 4);
    assert.equal(HUFFMAN_LINBITS[31], 13);
  });
});

describe('MP3 frame headers', () => {
  test('parses a known MPEG-1 Layer III header', () => {
    // FF FB 90 00 = MPEG-1, Layer III, no CRC, 128 kbps, 44.1 kHz, stereo.
    const header = parseFrameHeader(new Uint8Array([0xff, 0xfb, 0x90, 0x00]), 0);
    assert.ok(header);
    assert.equal(header.version, 1);
    assert.equal(header.layer, 3);
    assert.equal(header.bitrate, 128000);
    assert.equal(header.sampleRate, 44100);
    assert.equal(header.channels, 2);
    assert.equal(header.samplesPerFrame, 1152);
    assert.equal(header.sideInfoSize, 32);
    assert.equal(header.frameLength, 417);
  });

  test('parses an MPEG-2 header with its halved geometry', () => {
    // FF F3 = MPEG-2, Layer III. 0x50 => 64 kbps at 22.05 kHz.
    const header = parseFrameHeader(new Uint8Array([0xff, 0xf3, 0x50, 0xc0]), 0);
    assert.ok(header);
    assert.equal(header.version, 2);
    assert.equal(header.samplesPerFrame, 576, 'MPEG-2 Layer III halves the granule count');
    assert.equal(header.channels, 1);
    assert.equal(header.sideInfoSize, 9, 'mono MPEG-2 side info is 9 bytes');
  });

  test('rejects reserved and invalid field combinations', () => {
    assert.equal(parseFrameHeader(new Uint8Array([0xff, 0xeb, 0x90, 0x00]), 0), null, 'reserved version');
    assert.equal(parseFrameHeader(new Uint8Array([0xff, 0xf9, 0x90, 0x00]), 0), null, 'reserved layer');
    assert.equal(parseFrameHeader(new Uint8Array([0xff, 0xfb, 0x00, 0x00]), 0), null, 'free-form bitrate');
    assert.equal(parseFrameHeader(new Uint8Array([0xff, 0xfb, 0xf0, 0x00]), 0), null, 'invalid bitrate index');
    assert.equal(parseFrameHeader(new Uint8Array([0xff, 0xfb, 0x9c, 0x00]), 0), null, 'reserved sample rate');
    assert.equal(parseFrameHeader(new Uint8Array([0x00, 0x00, 0x00, 0x00]), 0), null, 'no sync');
  });

  test('sizes an ID3v2 tag using syncsafe integers', () => {
    const tag = new Uint8Array(30);
    tag.set([0x49, 0x44, 0x33, 0x04, 0x00, 0x00], 0); // "ID3", v2.4
    tag[6] = 0x00;
    tag[7] = 0x00;
    tag[8] = 0x01;
    tag[9] = 0x0a; // syncsafe 0x8A = 138
    assert.equal(id3v2Size(tag, 0), 10 + 138);
    assert.equal(id3v2Size(new Uint8Array(30), 0), 0, 'absent tag reports zero');
  });
});

describe('MP3 side information and bit reservoir', () => {
  test('the reservoir returns null until it has enough history', () => {
    const reservoir = new BitReservoir();
    // A frame that references 100 bytes of history it cannot have yet.
    assert.equal(reservoir.read(100, new Uint8Array(50)), null);

    // A self-contained frame always resolves.
    const selfContained = new Uint8Array([1, 2, 3]);
    assert.deepEqual(reservoir.read(0, selfContained), selfContained);
  });

  test('the reservoir concatenates history ahead of the current frame', () => {
    const reservoir = new BitReservoir();
    reservoir.read(0, new Uint8Array([1, 2, 3, 4]));

    const current = new Uint8Array([9, 9]);
    const assembled = reservoir.read(2, current);
    assert.ok(assembled);
    // Two bytes of history (3, 4) then the current payload.
    assert.deepEqual(Array.from(assembled), [3, 4, 9, 9]);
  });

  test('the reservoir stays bounded no matter how much is pushed', () => {
    const reservoir = new BitReservoir(1024);
    for (let i = 0; i < 500; i++) reservoir.read(0, new Uint8Array(400));
    assert.ok(reservoir.length <= 1024, `grew to ${reservoir.length}`);
  });

  test('an oversized payload keeps only the referenceable tail', () => {
    const reservoir = new BitReservoir(64);
    reservoir.read(0, new Uint8Array(200).fill(7));
    assert.equal(reservoir.length, 64);
  });
});

describe('MP3 against real files', { skip: FFMPEG ? false : 'ffmpeg not installed' }, () => {
  let dir: string;

  test('setup', () => {
    dir = mkdtempSync(join(tmpdir(), 'audiobox-mp3-'));
  });

  test.after(() => {
    if (dir) rmSync(dir, { recursive: true, force: true });
  });

  function render(name: string, args: string[]): Uint8Array {
    const path = join(dir, name);
    execFileSync('ffmpeg', ['-v', 'error', '-f', 'lavfi', ...args, path, '-y']);
    return new Uint8Array(readFileSync(path));
  }

  /** Walks every frame, parsing side info and resolving the reservoir. */
  function walk(data: Uint8Array) {
    let offset = id3v2Size(data, 0);
    const reservoir = new BitReservoir();
    let frames = 0;
    let resolved = 0;
    let overBudget = 0;
    let maxMainDataBegin = 0;

    for (;;) {
      const found = findFrame(data, offset);
      if (!found) break;
      const header = found.header;

      const sideInfo = parseSideInfo(data, found.offset + 4 + (header.hasCrc ? 2 : 0), header);
      frames++;
      maxMainDataBegin = Math.max(maxMainDataBegin, sideInfo.mainDataBegin);

      const start = found.offset + 4 + (header.hasCrc ? 2 : 0) + header.sideInfoSize;
      const end = Math.min(found.offset + header.frameLength, data.byteLength);
      const assembled = reservoir.read(sideInfo.mainDataBegin, data.subarray(start, end));

      if (assembled) {
        resolved++;
        let bits = 0;
        for (const granule of sideInfo.granules) {
          for (const channel of granule) bits += channel.part2_3_length;
        }
        if (bits > assembled.byteLength * 8) overBudget++;
      }

      offset = found.offset + header.frameLength;
    }

    return { frames, resolved, overBudget, maxMainDataBegin };
  }

  test('CBR stereo: every granule fits inside the data the reservoir assembled', () => {
    const data = render('cbr.mp3', [
      '-i', 'sine=f=440:d=3:r=44100', '-ac', '2', '-b:a', '128k',
    ]);
    const result = walk(data);

    assert.ok(result.frames > 100, `only found ${result.frames} frames`);
    assert.equal(result.resolved, result.frames, 'every frame should resolve');
    // This is the assertion that actually proves the parsing is right: if any
    // field width or the reservoir arithmetic were wrong, granules would claim
    // more bits than exist.
    assert.equal(result.overBudget, 0, 'no granule may exceed its available bits');
    assert.ok(result.maxMainDataBegin <= 511, 'MPEG-1 main_data_begin is 9 bits');
  });

  test('VBR stereo', () => {
    const result = walk(render('vbr.mp3', ['-i', 'sine=f=440:d=3:r=44100', '-ac', '2', '-q:a', '2']));
    assert.equal(result.resolved, result.frames);
    assert.equal(result.overBudget, 0);
  });

  test('MPEG-2 mono uses an 8-bit main_data_begin', () => {
    const result = walk(render('mono2.mp3', [
      '-i', 'sine=f=440:d=3:r=22050', '-ac', '1', '-b:a', '64k',
    ]));
    assert.equal(result.resolved, result.frames);
    assert.equal(result.overBudget, 0);
    assert.ok(result.maxMainDataBegin <= 255, 'MPEG-2 main_data_begin is 8 bits');
  });

  test('joint stereo, the most common real-world mode', () => {
    const result = walk(render('joint.mp3', [
      '-i', 'sine=f=440:d=3:r=44100', '-ac', '2', '-b:a', '192k', '-joint_stereo', '1',
    ]));
    assert.equal(result.resolved, result.frames);
    assert.equal(result.overBudget, 0);
  });

  test('probe agrees with ffprobe on duration', () => {
    const path = join(dir, 'probe.mp3');
    execFileSync('ffmpeg', [
      '-v', 'error', '-f', 'lavfi', '-i', 'sine=f=440:d=4:r=44100',
      '-ac', '2', '-b:a', '160k', path, '-y',
    ]);
    const expected = Number.parseFloat(
      execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path])
        .toString()
        .trim(),
    );
    const info = probeMp3(new Uint8Array(readFileSync(path)));
    assert.ok(Math.abs(info.duration - expected) < 0.1, `${info.duration} vs ${expected}`);
  });
});

describe('MP3 Huffman tables', () => {
  test('the packed trie is internally consistent', () => {
    assert.equal(HUFFMAN_TABLE_INDEX.length, 32);
    assert.equal(COUNT1_TABLE_A.length, 28);
    assert.equal(COUNT1_TABLE_B.length, 16);

    for (let table = 0; table < 32; table++) {
      const offset = HUFFMAN_TABLE_INDEX[table]!;
      assert.ok(offset < HUFFMAN_TABS.length, `table ${table} offset out of range`);
    }
  });

  test('count1 table B is a flat 4-bit code', () => {
    // Every entry must be a leaf (bit 3 set) with length 4, or the two-level
    // path would be taken for a table that has no second level.
    for (let i = 0; i < COUNT1_TABLE_B.length; i++) {
      const entry = COUNT1_TABLE_B[i]!;
      assert.ok((entry & 8) !== 0, `entry ${i} is not a leaf`);
      assert.equal(entry & 7, 4, `entry ${i} should be a 4-bit code`);
    }
  });
});

describe('MP3 Huffman decoding against real files', { skip: FFMPEG ? false : 'ffmpeg not installed' }, () => {
  let dir: string;

  test('setup', () => {
    dir = mkdtempSync(join(tmpdir(), 'audiobox-huff-'));
  });

  test.after(() => {
    if (dir) rmSync(dir, { recursive: true, force: true });
  });

  /**
   * Decodes every granule and checks the bit accounting.
   *
   * `part2_3_length` declares exactly how many bits a granule's scalefactors
   * plus Huffman data occupy, which makes the bitstream self-checking: if the
   * tables, the trie walk, the scalefactor widths, or the region boundaries were
   * wrong, the bits consumed would not land on the declared total. Getting this
   * to 100% is strong evidence of correctness without needing reference PCM.
   */
  function checkBitAccounting(data: Uint8Array): { granules: number; exact: number; overshoot: number } {
    let offset = id3v2Size(data, 0);
    const reservoir = new BitReservoir();
    const values = new Int32Array(576);
    let granules = 0;
    let exact = 0;
    let overshoot = 0;

    for (;;) {
      const found = findFrame(data, offset);
      if (!found) break;
      const header = found.header;

      const sideInfo = parseSideInfo(data, found.offset + 4 + (header.hasCrc ? 2 : 0), header);
      const start = found.offset + 4 + (header.hasCrc ? 2 : 0) + header.sideInfoSize;
      const end = Math.min(found.offset + header.frameLength, data.byteLength);
      const main = reservoir.read(sideInfo.mainDataBegin, data.subarray(start, end));
      offset = found.offset + header.frameLength;
      if (!main) continue;

      const br = new BitReader(main);
      const previous = createScalefactors();

      for (let gr = 0; gr < sideInfo.granules.length; gr++) {
        for (let ch = 0; ch < header.channels; ch++) {
          const granule = sideInfo.granules[gr]![ch]!;
          const scalefactors = createScalefactors();
          const before = br.bitsRead;

          const scalefactorBits =
            header.version === 1
              ? decodeScalefactorsMpeg1(br, granule, sideInfo.scfsi[ch]!, gr, previous, scalefactors)
              : decodeScalefactorsMpeg2(br, granule, false, scalefactors);

          const result = decodeHuffman(
            br,
            granule,
            header.sampleRate,
            granule.part2_3_length - scalefactorBits,
            values,
          );

          const consumed = scalefactorBits + result.bitsUsed;
          granules++;
          if (consumed === granule.part2_3_length) exact++;
          if (consumed > granule.part2_3_length) overshoot++;

          br.seekBits(before + granule.part2_3_length);
          if (header.version === 1) previous.long.set(scalefactors.long);
        }
      }
    }

    return { granules, exact, overshoot };
  }

  function render(name: string, filter: string, args: string[]): Uint8Array {
    const path = join(dir, name);
    execFileSync('ffmpeg', ['-v', 'error', '-f', 'lavfi', '-i', filter, ...args, path, '-y']);
    return new Uint8Array(readFileSync(path));
  }

  const cases: Array<[string, string, string[]]> = [
    ['tone-128k.mp3', 'sine=f=440:d=3', ['-ac', '2', '-b:a', '128k']],
    ['pink-320k.mp3', 'anoisesrc=d=3:c=pink', ['-ac', '2', '-b:a', '320k']],
    ['brown-32k.mp3', 'anoisesrc=d=3:c=brown', ['-ac', '2', '-ar', '32000', '-b:a', '128k']],
    // Tremolo forces frequent long/short window switching, which exercises the
    // transition block types and their region boundaries.
    ['tremolo.mp3', 'sine=f=1000:d=3,aeval=val(0)*(1+0.9*sin(2*PI*17*t))', ['-ac', '2', '-ar', '48000', '-b:a', '256k']],
    ['mpeg2-mono.mp3', 'anoisesrc=d=3:c=white', ['-ac', '1', '-ar', '22050', '-b:a', '64k']],
    ['mpeg2-stereo.mp3', 'anoisesrc=d=3:c=pink', ['-ac', '2', '-ar', '24000', '-b:a', '96k']],
    ['mpeg25.mp3', 'sine=f=300:d=3', ['-ac', '1', '-ar', '11025', '-b:a', '32k']],
    ['low-16k.mp3', 'anoisesrc=d=3:c=pink', ['-ac', '2', '-ar', '16000', '-b:a', '48k']],
  ];

  for (const [name, filter, args] of cases) {
    test(`${name}: every granule consumes exactly part2_3_length`, () => {
      const result = checkBitAccounting(render(name, filter, args));

      assert.ok(result.granules > 50, `only ${result.granules} granules decoded`);
      assert.equal(
        result.overshoot,
        0,
        'a granule consumed more bits than it declared, which means the tables are wrong',
      );
      assert.equal(
        result.exact,
        result.granules,
        `${result.granules - result.exact}/${result.granules} granules missed their declared bit count`,
      );
    });
  }
});

describe('MP3 decoding against ffmpeg', { skip: FFMPEG ? false : 'ffmpeg not installed' }, () => {
  let dir: string;

  test('setup', () => {
    dir = mkdtempSync(join(tmpdir(), 'audiobox-mp3dec-'));
  });

  test.after(() => {
    if (dir) rmSync(dir, { recursive: true, force: true });
  });

  /**
   * Decodes with both audiobox and ffmpeg and compares sample by sample.
   *
   * Two ISO-compliant decoders never agree bit-for-bit — they differ in float
   * precision and rounding — so the bar is SNR, not equality. Above ~50 dB the
   * difference is far below the noise floor of the lossy encoding itself.
   * `lag` is asserted to be 0: a decoder can be spectrally perfect and still be
   * useless for editing if it does not agree on where the audio starts.
   */
  function compare(name: string, filter: string, args: string[]) {
    const mp3Path = join(dir, `${name}.mp3`);
    const rawPath = join(dir, `${name}.raw`);
    execFileSync('ffmpeg', ['-v', 'error', '-f', 'lavfi', '-i', filter, ...args, mp3Path, '-y']);
    execFileSync('ffmpeg', ['-v', 'error', '-i', mp3Path, '-f', 'f32le', rawPath, '-y']);

    const decoded = decodeMp3(new Uint8Array(readFileSync(mp3Path)));
    const buffer = readFileSync(rawPath);
    const reference = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    const channels = decoded.channelData.length;

    let bestSnr = -Infinity;
    let bestLag = 0;
    for (let lag = -200; lag <= 200; lag++) {
      let signal = 0;
      let noise = 0;
      const limit = Math.min(decoded.channelData[0]!.length - 4000, 50000);
      for (let i = 4000; i < limit; i++) {
        for (let c = 0; c < channels; c++) {
          const ours = decoded.channelData[c]![i]!;
          const theirs = reference[(i + lag) * channels + c];
          if (theirs === undefined) continue;
          signal += theirs * theirs;
          noise += (ours - theirs) ** 2;
        }
      }
      const snr = noise > 0 ? 10 * Math.log10(signal / noise) : Infinity;
      if (snr > bestSnr) {
        bestSnr = snr;
        bestLag = lag;
      }
    }

    return { decoded, snr: bestSnr, lag: bestLag, referenceFrames: reference.length / channels };
  }

  const cases: Array<[string, string, string[], number, number]> = [
    // name, filter, ffmpeg args, expected channels, expected rate
    ['cbr-128k-stereo', 'sine=f=440:d=3', ['-ac', '2', '-b:a', '128k'], 2, 44100],
    ['cbr-320k-pink', 'anoisesrc=d=3:c=pink', ['-ac', '2', '-b:a', '320k'], 2, 48000],
    ['vbr-stereo', 'anoisesrc=d=3:c=brown', ['-ac', '2', '-q:a', '2'], 2, 48000],
    // Tremolo forces window switching, exercising short and transition blocks.
    ['joint-192k', 'sine=f=800:d=3,aeval=val(0)*(1+0.8*sin(2*PI*11*t))', ['-ac', '2', '-b:a', '192k'], 2, 44100],
    ['mono-64k', 'anoisesrc=d=3:c=white', ['-ac', '1', '-b:a', '64k'], 1, 48000],
    ['mpeg2-22k', 'anoisesrc=d=3:c=pink', ['-ac', '1', '-ar', '22050', '-b:a', '64k'], 1, 22050],
    ['mpeg2-24k-stereo', 'anoisesrc=d=3:c=pink', ['-ac', '2', '-ar', '24000', '-b:a', '96k'], 2, 24000],
    ['mpeg25-11k', 'sine=f=300:d=3', ['-ac', '1', '-ar', '11025', '-b:a', '32k'], 1, 11025],
    ['48k-256k', 'anoisesrc=d=3:c=pink', ['-ac', '2', '-ar', '48000', '-b:a', '256k'], 2, 48000],
    ['32k-128k', 'anoisesrc=d=3:c=brown', ['-ac', '2', '-ar', '32000', '-b:a', '128k'], 2, 32000],
  ];

  for (const [name, filter, args, channels, rate] of cases) {
    test(`${name} matches ffmpeg`, () => {
      const { decoded, snr, lag, referenceFrames } = compare(name, filter, args);

      assert.equal(decoded.channelData.length, channels, 'channel count');
      assert.equal(decoded.sampleRate, rate, 'sample rate');
      assert.equal(decoded.info.format, 'mp3');
      assert.equal(decoded.info.lossy, true);

      assert.equal(lag, 0, `decoded audio is offset from ffmpeg by ${lag} samples`);
      assert.ok(snr > 45, `SNR against ffmpeg was only ${snr.toFixed(1)} dB`);

      // Exactly, not approximately. This was a 441-sample tolerance, which was
      // wide enough to hide a systematic 64-sample shortfall: the gapless trim
      // subtracted the implementation's filterbank offset at the head but not
      // at the tail, so every file lost 64 samples off the end.
      const drift = decoded.channelData[0]!.length - referenceFrames;
      assert.equal(drift, 0, `length differs from ffmpeg by ${drift} samples`);
    });
  }

  test('gapless trimming can be turned off', () => {
    const path = join(dir, 'gapless.mp3');
    execFileSync('ffmpeg', [
      '-v', 'error', '-f', 'lavfi', '-i', 'sine=f=440:d=2', '-ac', '2', '-b:a', '128k', path, '-y',
    ]);
    const bytes = new Uint8Array(readFileSync(path));

    const trimmed = decodeMp3(bytes);
    const untrimmed = decodeMp3(bytes, { gapless: false });
    assert.ok(
      untrimmed.channelData[0]!.length > trimmed.channelData[0]!.length,
      'disabling gapless should keep the encoder priming samples',
    );
  });

  test('a truncated file still decodes what survived', () => {
    const path = join(dir, 'truncated.mp3');
    execFileSync('ffmpeg', [
      '-v', 'error', '-f', 'lavfi', '-i', 'sine=f=440:d=3', '-ac', '2', '-b:a', '128k', path, '-y',
    ]);
    const full = new Uint8Array(readFileSync(path));
    const half = full.subarray(0, Math.floor(full.byteLength / 2));

    const decoded = decodeMp3(half);
    assert.ok(decoded.channelData[0]!.length > 1000, 'should recover the frames that are present');
    assert.ok(decoded.channelData[0]!.length < decodeMp3(full).channelData[0]!.length);
  });

  test('decode() routes MP3 automatically', async () => {
    const path = join(dir, 'auto.mp3');
    execFileSync('ffmpeg', [
      '-v', 'error', '-f', 'lavfi', '-i', 'sine=f=440:d=1', '-ac', '2', '-b:a', '128k', path, '-y',
    ]);
    const audio = await decode(readFileSync(path));
    assert.equal(audio.info?.format, 'mp3');
    assert.equal(audio.channels, 2);
    assert.ok(Math.abs(audio.duration - 1) < 0.05, `duration ${audio.duration}`);

    // And it composes with the rest of the library.
    const wav = await encode(audio.toMono().resample(16000), 'wav', { bitDepth: 16 });
    assert.ok(wav.byteLength > 0);
    assert.equal((await decode(wav)).sampleRate, 16000);
  });
});

describe('MP3 analysis filterbank (encoder side)', () => {
  const RATE = 44100;
  const BAND_WIDTH = RATE / 64;

  /** Runs the analysis filterbank alone and returns per-sub-band energy. */
  function subbandEnergy(signal: Float32Array): Float64Array {
    const filterbank = new AnalysisFilterbank();
    const slot = new Float32Array(32);
    const energy = new Float64Array(32);
    const frames = Math.floor(signal.length / 32);

    for (let n = 0; n < frames; n++) {
      filterbank.process(signal, n * 32, slot);
      // Skip the start-up transient, where the delay line is still filling.
      if (n < 40) continue;
      for (let k = 0; k < 32; k++) energy[k] += slot[k]! * slot[k]!;
    }
    return energy;
  }

  test('the analysis window takes its magnitudes from the synthesis window', () => {
    assert.equal(ANALYSIS_WINDOW.length, 512);

    // The two share a prototype filter, so every magnitude is the synthesis
    // window's, shifted 64 taps and scaled by 1/32.
    let worst = 0;
    for (let i = 0; i < 512; i++) {
      const expected = Math.abs(SYNTH_WINDOW[(i + 64) % 512]!) / 32;
      worst = Math.max(worst, Math.abs(Math.abs(ANALYSIS_WINDOW[i]!) - expected));
    }
    assert.ok(worst < 1e-4, `magnitudes drifted from the synthesis window by ${worst}`);

    // The signs, however, are not D's. 28 taps differ, and those 28 are worth
    // 31 dB to an independent decoder — which is why this table is solved and
    // generated rather than computed from D at module load.
    let flipped = 0;
    for (let i = 0; i < 512; i++) {
      const derived = SYNTH_WINDOW[(i + 64) % 512]! / 32;
      if (derived !== 0 && Math.sign(ANALYSIS_WINDOW[i]!) !== Math.sign(derived)) flipped++;
    }
    assert.equal(flipped, 28, 'the solved sign pattern changed');
  });

  test('a tone lands in its own sub-band and nowhere else', () => {
    // This is the gate that matters for an analysis filterbank. Reconstruction
    // SNR is not: it is bounded by our own synthesis window's accuracy, so it
    // measures the decoder rather than the encoder.
    for (const band of [0, 1, 5, 10, 15, 20, 28]) {
      const hz = (band + 0.5) * BAND_WIDTH;
      const signal = new Float32Array(16384);
      for (let i = 0; i < signal.length; i++) {
        signal[i] = Math.sin((2 * Math.PI * hz * i) / RATE) * 0.5;
      }

      const energy = subbandEnergy(signal);
      const total = energy.reduce((a, b) => a + b, 0);
      const inBand = energy[band]! / total;
      const leakDb = 10 * Math.log10((total - energy[band]!) / energy[band]!);

      assert.ok(inBand > 0.999, `band ${band}: only ${(inBand * 100).toFixed(2)}% in-band`);
      assert.ok(leakDb < -35, `band ${band}: leakage ${leakDb.toFixed(1)} dB`);
    }
  });

  test('analysis mirrors the decoder chain and reconstructs the signal', () => {
    // Push audio through the encoder's forward chain, then through the
    // decoder's own inverse path unchanged. If the two are mirror images, the
    // original comes back.
    const longBlock = {
      windowSwitching: false, blockType: 0, mixedBlock: false,
      subblockGain: [0, 0, 0] as [number, number, number],
      preflag: false, scalefacScale: 0.5, count1TableSelect: 0,
      part2_3_length: 0, bigValues: 0, globalGain: 0, scalefacCompress: 0,
      tableSelect: [0, 0, 0] as [number, number, number],
      region0Count: 0, region1Count: 0,
    };

    let seed = 777;
    const random = (): number => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x40000000 - 1;
    };
    const input = new Float32Array(30000);
    for (let i = 0; i < input.length; i++) input[i] = random() * 0.4;

    const filterbank = new AnalysisFilterbank();
    const encodeOverlap = new Float32Array(576);
    const subbands = new Float32Array(576);
    const slot = new Float32Array(32);
    const spectrum = new Float32Array(576);

    const synthesis = new SynthesisFilterbank();
    const decodeOverlap = new Float32Array(576);
    const timeDomain = new Float32Array(576);
    const decodeSlot = new Float32Array(32);
    const pcm = new Float32Array(576);

    const granules = Math.floor(input.length / 576);
    const out = new Float32Array(granules * 576);

    for (let g = 0; g < granules; g++) {
      analyseGranule(input, g * 576, filterbank, encodeOverlap, subbands, slot, spectrum);

      reduceAlias(spectrum, longBlock);
      imdct(spectrum, longBlock, decodeOverlap, timeDomain);
      invertFrequency(timeDomain);
      for (let t = 0; t < 18; t++) {
        for (let b = 0; b < 32; b++) decodeSlot[b] = timeDomain[b * 18 + t]!;
        synthesis.process(decodeSlot, pcm, t * 32);
      }
      out.set(pcm, g * 576);
    }

    let bestSnr = -Infinity;
    let bestScale = 0;
    for (let lag = 0; lag <= 1600; lag++) {
      let dot = 0;
      let outEnergy = 0;
      for (let i = 3000; i < 20000; i++) {
        const o = out[i + lag];
        if (o === undefined) continue;
        dot += o * input[i]!;
        outEnergy += o * o;
      }
      if (outEnergy <= 0) continue;
      const scale = dot / outEnergy;

      let signal = 0;
      let noise = 0;
      for (let i = 3000; i < 20000; i++) {
        const o = out[i + lag];
        if (o === undefined) continue;
        signal += input[i]! ** 2;
        noise += (o * scale - input[i]!) ** 2;
      }
      const snr = 10 * Math.log10(signal / noise);
      if (snr > bestSnr) {
        bestSnr = snr;
        bestScale = scale;
      }
    }

    // Scoring after fitting a gain measures shape and ignores level, and level
    // is exactly what a missing transform normalisation destroys: this chain
    // once reconstructed perfectly at nine times the input amplitude, because
    // the specification's IMDCT omits its 2/N and the forward MDCT has to carry
    // it. Asserting the fitted gain is unity is what catches that.
    assert.ok(
      Math.abs(bestScale - 1) < 0.01,
      `reconstruction gain was ${(1 / bestScale).toFixed(4)}x, not unity`,
    );

    // 50 dB. The remaining ceiling is the *synthesis* window — running analysis
    // straight into synthesis with no transform between them scores the same, so
    // the MDCT/IMDCT pair contributes nothing measurable. ffmpeg decodes our
    // output 19 dB better than this, which is how we know the residual is on the
    // decode side rather than in the analysis being tested here.
    assert.ok(bestSnr > 50, `round-trip SNR was only ${bestSnr.toFixed(1)} dB`);
  });
});

describe('MP3 Huffman encoding', () => {
  /** A long-block granule that uses one table for every region. */
  function granuleFor(table: number, bigValues: number, count1Table = 0) {
    return {
      windowSwitching: false, blockType: 0, mixedBlock: false,
      subblockGain: [0, 0, 0] as [number, number, number],
      preflag: false, scalefacScale: 0.5, count1TableSelect: count1Table,
      part2_3_length: 0, bigValues, globalGain: 0, scalefacCompress: 0,
      tableSelect: [table, table, table] as [number, number, number],
      region0Count: 15, region1Count: 6,
    };
  }

  test('every codeword in every table round-trips through the real decoder', () => {
    // The generator already proves the tables against its own trie walk. This
    // proves them against the decoder that actually ships — the two could only
    // agree by both being right.
    const values = new Int32Array(576);
    let checked = 0;

    for (let table = 1; table < 32; table++) {
      if (table === 4 || table === 14) continue;
      const xlen = HUFFMAN_ENCODE_XLEN[table]!;
      const ylen = HUFFMAN_ENCODE_YLEN[table]!;
      if (ylen === 0) continue;

      // Every (x, y) the table codes, with both signs where applicable.
      const pairs: Array<[number, number]> = [];
      for (let x = 0; x < xlen; x++) {
        for (let y = 0; y < ylen; y++) {
          if (pairCost(table, x, y) === Infinity) continue;
          pairs.push([x, y]);
          // Negate via a guard rather than unary minus: -0 is a distinct value
          // to Object.is, which is what assert.strictEqual compares with.
          if (x !== 0 || y !== 0) pairs.push([x === 0 ? 0 : -x, y === 0 ? 0 : -y]);
        }
      }
      // Keep each granule within the 576-line limit.
      const batch = pairs.slice(0, 288);

      const writer = new BitWriter();
      let expectedBits = 0;
      for (const [x, y] of batch) {
        writePair(writer, table, x, y);
        expectedBits += pairCost(table, x, y);
      }
      assert.equal(writer.bitLength, expectedBits, `table ${table}: pairCost disagrees with writePair`);

      const bytes = writer.finish();
      const reader = new BitReader(bytes);
      const result = decodeHuffman(reader, granuleFor(table, batch.length), 44100, expectedBits, values);

      assert.equal(result.bitsUsed, expectedBits, `table ${table}: decoder consumed a different number of bits`);
      for (let i = 0; i < batch.length; i++) {
        assert.equal(values[i * 2], batch[i]![0], `table ${table}: pair ${i} x`);
        assert.equal(values[i * 2 + 1], batch[i]![1], `table ${table}: pair ${i} y`);
      }
      checked += batch.length;
    }

    assert.ok(checked > 2000, `only ${checked} pairs exercised`);
  });

  test('escape coding carries large magnitudes', () => {
    // Tables 16-31 code anything above 14 as an escape plus linbits raw bits.
    // This is where an off-by-one in the escape threshold would hide.
    const values = new Int32Array(576);

    for (const table of [16, 20, 23, 24, 31]) {
      const linbits = HUFFMAN_LINBITS[table]!;
      const maxExtra = (1 << linbits) - 1;
      const batch: Array<[number, number]> = [
        [14, 14], [15, 0], [0, 15], [15, 15],
        [15 + maxExtra, 1], [1, 15 + maxExtra], [-(15 + maxExtra), -(15 + maxExtra)],
      ];

      const writer = new BitWriter();
      let bits = 0;
      for (const [x, y] of batch) {
        writePair(writer, table, x, y);
        bits += pairCost(table, x, y);
      }

      const reader = new BitReader(writer.finish());
      decodeHuffman(reader, granuleFor(table, batch.length), 44100, bits, values);

      for (let i = 0; i < batch.length; i++) {
        assert.equal(values[i * 2], batch[i]![0], `table ${table} (linbits ${linbits}): pair ${i} x`);
        assert.equal(values[i * 2 + 1], batch[i]![1], `table ${table} (linbits ${linbits}): pair ${i} y`);
      }
    }
  });

  test('a value beyond a table\'s range is rejected, not silently truncated', () => {
    // Table 1 codes only 0 and 1. Costing must say so rather than emitting
    // something the decoder would read back as a different number.
    assert.equal(pairCost(1, 2, 0), Infinity);
    assert.equal(pairCost(1, 1, 1) === Infinity, false);
    // Tables without linbits cannot escape.
    assert.equal(pairCost(15, 16, 0), Infinity);
    // Even an escaping table has a ceiling.
    assert.equal(pairCost(16, 15 + (1 << HUFFMAN_LINBITS[16]!), 0), Infinity);
  });

  test('count1 quadruples round-trip through the real decoder', () => {
    const values = new Int32Array(576);

    for (const countTable of [0, 1]) {
      // All 16 non-zero masks, with assorted signs.
      const quads: number[] = [];
      for (let mask = 0; mask < 16; mask++) {
        quads.push(
          mask & 8 ? (mask & 1 ? -1 : 1) : 0,
          mask & 4 ? -1 : 0,
          mask & 2 ? 1 : 0,
          mask & 1 ? -1 : 0,
        );
      }

      const writer = new BitWriter();
      let bits = 0;
      for (let i = 0; i < quads.length; i += 4) {
        writeCount1(writer, countTable, quads[i]!, quads[i + 1]!, quads[i + 2]!, quads[i + 3]!);
        bits += count1Cost(countTable, quads[i]!, quads[i + 1]!, quads[i + 2]!, quads[i + 3]!);
      }
      assert.equal(writer.bitLength, bits, `count1 table ${countTable}: cost disagrees with write`);

      const reader = new BitReader(writer.finish());
      decodeHuffman(reader, granuleFor(1, 0, countTable), 44100, bits, values);

      for (let i = 0; i < quads.length; i++) {
        assert.equal(values[i], quads[i], `count1 table ${countTable}: value ${i}`);
      }
    }
  });

  test('table selection picks a codable table and prefers cheaper ones', () => {
    // A region of small values should choose a small table, not an escaping one.
    const small = new Int32Array(32);
    for (let i = 0; i < small.length; i++) small[i] = i % 2;
    const chosen = chooseTable(small, 0, small.length);
    assert.ok(regionCost(small, 0, small.length, chosen) < Infinity);
    assert.equal(HUFFMAN_LINBITS[chosen], 0, `chose escaping table ${chosen} for tiny values`);

    // A region with a large value forces a table that can escape.
    const large = new Int32Array(32);
    large[0] = 200;
    const chosenLarge = chooseTable(large, 0, large.length);
    assert.ok(HUFFMAN_LINBITS[chosenLarge]! > 0, `table ${chosenLarge} cannot code 200`);
    assert.ok(regionCost(large, 0, large.length, chosenLarge) < Infinity);

    // An all-zero region costs nothing and uses table 0.
    assert.equal(chooseTable(new Int32Array(16), 0, 16), 0);
  });
});

describe('MP3 bitstream writing', () => {
  /** A plausible granule, varied by seed so fields are distinguishable. */
  function makeGranule(seed: number): GranuleInfo {
    return {
      part2_3_length: 0, // filled in by the caller
      bigValues: 40 + (seed % 30),
      globalGain: 100 + (seed % 50),
      scalefacCompress: seed % 16,
      windowSwitching: false,
      blockType: 0,
      mixedBlock: false,
      tableSelect: [1 + (seed % 10), 2 + (seed % 8), 3 + (seed % 6)] as [number, number, number],
      subblockGain: [0, 0, 0] as [number, number, number],
      region0Count: seed % 16,
      region1Count: seed % 8,
      preflag: seed % 2 === 1,
      scalefacScale: seed % 3 === 0 ? 1 : 0.5,
      count1TableSelect: seed % 2,
    };
  }

  function makeSideInfo(channels: number, seed: number, lengths: number[]): SideInfo {
    const granules: GranuleInfo[][] = [];
    let i = 0;
    for (let gr = 0; gr < 2; gr++) {
      const row: GranuleInfo[] = [];
      for (let ch = 0; ch < channels; ch++) {
        const g = makeGranule(seed + i);
        g.part2_3_length = lengths[i] ?? 0;
        row.push(g);
        i++;
      }
      granules.push(row);
    }
    const scfsi: number[][] = [];
    for (let ch = 0; ch < channels; ch++) scfsi.push([0, 0, 0, 0]);
    return { mainDataBegin: 0, scfsi, granules };
  }

  test('a written header parses back to the values it was given', () => {
    for (const bitrate of [64, 128, 192, 320]) {
      for (const rate of [32000, 44100, 48000]) {
        for (const [mode, channels] of [['mono', 1], ['stereo', 2], ['joint', 2]] as const) {
          const writer = new BitWriter();
          writeFrameHeader(writer, {
            bitrateKbps: bitrate, sampleRate: rate, channels,
            mode, modeExtension: 0, padding: false,
          });
          writer.align();

          const header = parseFrameHeader(writer.finish(), 0);
          assert.ok(header, `${bitrate}k ${rate} ${mode}: header did not parse`);
          assert.equal(header.version, 1);
          assert.equal(header.layer, 3);
          assert.equal(header.bitrate, bitrate * 1000);
          assert.equal(header.sampleRate, rate);
          assert.equal(header.mode, mode);
          assert.equal(header.channels, channels);
          assert.equal(header.sideInfoSize, channels === 1 ? 17 : 32);
        }
      }
    }
  });

  test('side information round-trips field for field', () => {
    for (const channels of [1, 2]) {
      const lengths = channels === 1 ? [400, 380] : [400, 380, 360, 340];
      const original = makeSideInfo(channels, 7, lengths);

      const writer = new BitWriter();
      // The header must precede it, since parseSideInfo is given the offset past it.
      writeFrameHeader(writer, {
        bitrateKbps: 128, sampleRate: 44100, channels,
        mode: channels === 1 ? 'mono' : 'joint', modeExtension: 0, padding: false,
      });
      original.mainDataBegin = 123;
      writeSideInfo(writer, original, channels);
      writer.align();

      const bytes = writer.finish();
      const header = parseFrameHeader(bytes, 0)!;
      const parsed = parseSideInfo(bytes, 4, header);

      assert.equal(parsed.mainDataBegin, 123);
      assert.equal(parsed.granules.length, 2);

      for (let gr = 0; gr < 2; gr++) {
        for (let ch = 0; ch < channels; ch++) {
          const a = original.granules[gr]![ch]!;
          const b = parsed.granules[gr]![ch]!;
          const where = `${channels}ch gr${gr} ch${ch}`;
          assert.equal(b.part2_3_length, a.part2_3_length, `${where} part2_3_length`);
          assert.equal(b.bigValues, a.bigValues, `${where} bigValues`);
          assert.equal(b.globalGain, a.globalGain, `${where} globalGain`);
          assert.equal(b.scalefacCompress, a.scalefacCompress, `${where} scalefacCompress`);
          assert.equal(b.windowSwitching, a.windowSwitching, `${where} windowSwitching`);
          assert.deepEqual(b.tableSelect, a.tableSelect, `${where} tableSelect`);
          assert.equal(b.region0Count, a.region0Count, `${where} region0Count`);
          assert.equal(b.region1Count, a.region1Count, `${where} region1Count`);
          assert.equal(b.preflag, a.preflag, `${where} preflag`);
          assert.equal(b.scalefacScale, a.scalefacScale, `${where} scalefacScale`);
          assert.equal(b.count1TableSelect, a.count1TableSelect, `${where} count1TableSelect`);
        }
      }
    }
  });

  test('the assembler produces a stream the decoder can walk and reassemble', () => {
    // The real gate: lay out frames with varying main-data sizes, then use the
    // decoder's own BitReservoir to put each frame's data back together. If the
    // reservoir bookkeeping is wrong in either direction, this diverges.
    const channels = 2;
    const params = {
      bitrateKbps: 128, sampleRate: 44100, channels,
      mode: 'joint' as const, modeExtension: 0,
    };
    const assembler = new FrameAssembler(params);

    const expected: Uint8Array[] = [];
    const sideInfos: SideInfo[] = [];

    for (let frame = 0; frame < 24; frame++) {
      // Uneven around the slot size (~386 bytes at 128 kbps stereo): some frames
      // borrow from the reservoir, some give back. Averaging near the slot is
      // what real rate control does — feeding consistently less makes the
      // reservoir grow past its 511-byte reach, which the assembler rejects.
      // Rate control asks what it may spend, then fits inside it. Overshooting
      // underflows the reservoir; consistently undershooting overflows it.
      const want = 386 + ((frame * 53) % 120) - 60;
      const size = Math.min(want, assembler.nextBudgetBytes());
      const data = new Uint8Array(size);
      for (let i = 0; i < size; i++) data[i] = (frame * 31 + i * 7) & 0xff;

      const info = makeSideInfo(channels, frame, [size * 2, size * 2, size * 2, size * 2]);
      assembler.addFrame(info, data);
      expected.push(data);
      sideInfos.push(info);
    }

    const stream = assembler.finish((n) => makeSideInfo(channels, n, [0, 0, 0, 0]));
    assert.ok(stream.byteLength > 0);

    // Walk it exactly as the decoder does.
    const reservoir = new BitReservoir();
    let offset = 0;
    let recovered = 0;

    for (let frame = 0; frame < expected.length; frame++) {
      const found = findFrame(stream, offset);
      assert.ok(found, `frame ${frame} not found`);

      const header = found.header;
      assert.equal(header.bitrate, 128000, `frame ${frame} bitrate`);
      assert.equal(header.channels, 2, `frame ${frame} channels`);

      const parsed = parseSideInfo(stream, found.offset + 4, header);
      const start = found.offset + 4 + header.sideInfoSize;
      const end = Math.min(found.offset + header.frameLength, stream.byteLength);
      const main = reservoir.read(parsed.mainDataBegin, stream.subarray(start, end));

      assert.ok(main, `frame ${frame}: reservoir could not resolve main_data_begin ${parsed.mainDataBegin}`);
      assert.ok(
        main.byteLength >= expected[frame]!.byteLength,
        `frame ${frame}: reassembled ${main.byteLength} bytes, needed ${expected[frame]!.byteLength}`,
      );
      // main_data_begin points back to where this frame's data starts, so the
      // reservoir hands it back beginning at offset zero.
      const slice = main.subarray(0, expected[frame]!.byteLength);
      assert.deepEqual(
        Array.from(slice),
        Array.from(expected[frame]!),
        `frame ${frame}: main data does not match what was written`,
      );

      recovered++;
      offset = found.offset + header.frameLength;
    }

    assert.equal(recovered, expected.length);
  });

  test('stuffing keeps main_data_begin inside its 9-bit field', () => {
    // Frames that use less than their slot push the reservoir further and
    // further ahead. Left alone it passes 511 and the stream becomes
    // unreadable, so the assembler inserts filler to cap it. Decoders skip
    // that filler: anything past a granule's Huffman data is ancillary.
    const assembler = new FrameAssembler({
      bitrateKbps: 320, sampleRate: 44100, channels: 2, mode: 'stereo', modeExtension: 0,
    });

    // 320 kbps gives ~1044-byte frames; ten bytes each drains nothing at all.
    for (let i = 0; i < 40; i++) {
      assembler.addFrame(makeSideInfo(2, i, [0, 0, 0, 0]), new Uint8Array(10));
    }
    const stream = assembler.finish((n) => makeSideInfo(2, n, [0, 0, 0, 0]));

    let offset = 0;
    let frames = 0;
    let worst = 0;
    for (;;) {
      const found = findFrame(stream, offset);
      if (!found) break;
      const parsed = parseSideInfo(stream, found.offset + 4, found.header);
      worst = Math.max(worst, parsed.mainDataBegin);
      frames++;
      offset = found.offset + found.header.frameLength;
    }

    assert.ok(frames >= 40, `only ${frames} frames`);
    assert.ok(worst <= 511, `main_data_begin reached ${worst}`);
  });

  test('overshooting the budget is rejected rather than silently corrupting', () => {
    // The opposite failure: a frame carrying more than the slot plus reservoir
    // can hold has nowhere to live, and would make main_data_begin negative.
    const assembler = new FrameAssembler({
      bitrateKbps: 64, sampleRate: 44100, channels: 2, mode: 'joint', modeExtension: 0,
    });
    assert.throws(
      () => assembler.addFrame(makeSideInfo(2, 0, [0, 0, 0, 0]), new Uint8Array(5000)),
      /underflow|nextBudgetBytes/i,
    );
  });

  test('the padding bit keeps frame sizes on average correct', () => {
    // 44.1 kHz at 128 kbps needs 417.96 bytes per frame, so roughly one frame in
    // every four must be padded or the stream drifts short.
    const assembler = new FrameAssembler({
      bitrateKbps: 128, sampleRate: 44100, channels: 2, mode: 'joint', modeExtension: 0,
    });
    for (let i = 0; i < 100; i++) {
      assembler.addFrame(makeSideInfo(2, i, [0, 0, 0, 0]), new Uint8Array(380));
    }
    const stream = assembler.finish((n) => makeSideInfo(2, n, [0, 0, 0, 0]));

    let padded = 0;
    let frames = 0;
    let offset = 0;
    for (;;) {
      const found = findFrame(stream, offset);
      if (!found) break;
      if (found.header.padding) padded++;
      frames++;
      offset = found.offset + found.header.frameLength;
    }

    assert.ok(frames >= 100, `only ${frames} frames`);
    // 144 * 128000 / 44100 = 417.959, so ~96% of frames carry the pad byte.
    // Anything far from that means the accumulator is drifting and the stream
    // would slowly desynchronise.
    const ratio = padded / frames;
    assert.ok(Math.abs(ratio - 0.959) < 0.03, `padding ratio ${ratio.toFixed(3)}, expected ~0.959`);
  });
});

describe('MP3 quantisation and rate control', () => {
  const RATE = 44100;

  function chord(n: number): Float32Array {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      out[i] =
        0.35 * Math.sin((2 * Math.PI * 220 * i) / RATE) +
        0.2 * Math.sin((2 * Math.PI * 660 * i) / RATE) +
        0.12 * Math.sin((2 * Math.PI * 3300 * i) / RATE);
    }
    return out;
  }

  function noise(n: number, amplitude = 0.35): Float32Array {
    let seed = 12345;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      out[i] = (seed / 0x40000000 - 1) * amplitude;
    }
    return out;
  }

  test('the spectrum splits into legal big_values, count1 and rzero regions', () => {
    const values = new Int32Array(576);
    // Large values low down, ±1 in the middle, silence above.
    for (let i = 0; i < 40; i++) values[i] = (i % 7) - 3;
    for (let i = 40; i < 84; i++) values[i] = (i % 3) - 1;
    values[80] = 1;

    const { bigValuesEnd, rzeroStart } = partition(values);

    assert.equal(bigValuesEnd % 2, 0, 'big_values must end on a pair boundary');
    assert.equal((rzeroStart - bigValuesEnd) % 4, 0, 'count1 codes whole quadruples');
    assert.ok(bigValuesEnd / 2 <= 288, 'big_values exceeds its 9-bit field');

    // Nothing above rzeroStart may be non-zero, and nothing in the count1
    // region may exceed 1 — the decoder cannot represent either.
    for (let i = rzeroStart; i < 576; i++) assert.equal(values[i], 0);
    for (let i = bigValuesEnd; i < rzeroStart; i++) assert.ok(Math.abs(values[i]!) <= 1);
  });

  test('scalefac_compress covers the values it is chosen for, as cheaply as possible', () => {
    const scalefactors = new Int32Array(23);
    assert.equal(chooseScalefacCompress(scalefactors)!.bits, 0, 'all-zero costs nothing');

    scalefactors[3] = 5; // needs 3 bits in group 1
    scalefactors[15] = 2; // needs 2 bits in group 2
    const chosen = chooseScalefacCompress(scalefactors)!;
    assert.ok(SLEN1[chosen.index]! >= 3, 'slen1 too narrow for the value');
    assert.ok(SLEN2[chosen.index]! >= 2, 'slen2 too narrow for the value');
    assert.equal(chosen.bits, 11 * SLEN1[chosen.index]! + 10 * SLEN2[chosen.index]!);

    // Nothing legal is cheaper.
    for (let index = 0; index < 16; index++) {
      if (SLEN1[index]! < 3 || SLEN2[index]! < 2) continue;
      assert.ok(11 * SLEN1[index]! + 10 * SLEN2[index]! >= chosen.bits);
    }

    // 16 needs five bits, which no slen provides.
    scalefactors[3] = 16;
    assert.equal(chooseScalefacCompress(scalefactors), null);
  });

  test('quantisation survives the decoder’s own requantiser', () => {
    // Measured against the quantiser's input spectrum rather than through the
    // filterbank, so this is not capped by the analysis/synthesis pair and
    // actually measures quantisation.
    const signal = chord(RATE);
    const filterbank = new AnalysisFilterbank();
    const overlap = new Float32Array(576);
    const subbands = new Float32Array(576);
    const slot = new Float32Array(32);
    const spectrum = new Float32Array(576);
    const workspace = createWorkspace();
    const back = new Float32Array(576);

    for (const [budgetBits, floor] of [[1636, 40], [4095, 60]] as const) {
      let signalEnergy = 0;
      let noiseEnergy = 0;

      for (let g = 0; g < 40; g++) {
        analyseGranule(signal, g * 576, filterbank, overlap, subbands, slot, spectrum);
        const coding = quantizeGranule({ spectrum, sampleRate: RATE, budgetBits }, workspace);

        assert.ok(
          coding.part2_3Length <= budgetBits,
          `granule used ${coding.part2_3Length} of ${budgetBits} bits`,
        );

        let nonZero = 576;
        while (nonZero > 0 && coding.values[nonZero - 1] === 0) nonZero--;

        requantize(
          coding.values,
          codingToGranuleInfo(coding),
          { long: coding.scalefactors, short: [new Int32Array(14), new Int32Array(14), new Int32Array(14)] },
          RATE,
          nonZero,
          back,
        );

        for (let i = 0; i < 576; i++) {
          signalEnergy += spectrum[i]! ** 2;
          noiseEnergy += (spectrum[i]! - back[i]!) ** 2;
        }
      }

      const snr = 10 * Math.log10(signalEnergy / noiseEnergy);
      assert.ok(snr > floor, `${budgetBits}-bit budget gave only ${snr.toFixed(1)} dB`);
    }
  });

  test('a coded granule writes exactly the bits it declares', () => {
    // The declared length is how the decoder finds the next granule. Counting
    // during table selection and counting again while writing must agree, or
    // everything after the granule desynchronises.
    const signal = chord(RATE);
    const filterbank = new AnalysisFilterbank();
    const overlap = new Float32Array(576);
    const subbands = new Float32Array(576);
    const slot = new Float32Array(32);
    const spectrum = new Float32Array(576);
    const workspace = createWorkspace();

    for (let g = 0; g < 30; g++) {
      analyseGranule(signal, g * 576, filterbank, overlap, subbands, slot, spectrum);
      const coding = quantizeGranule({ spectrum, sampleRate: RATE, budgetBits: 2464 }, workspace);

      const writer = new BitWriter(1024);
      const written = writeGranuleData(writer, coding, RATE);
      assert.equal(written, coding.huffmanBits, `granule ${g} miscounted its Huffman bits`);
    }
  });

  test('encoded frames report the parameters they were asked for', () => {
    for (const [bitrate, sampleRate, channels] of [
      [128, 44100, 1],
      [192, 44100, 2],
      [320, 48000, 2],
      [64, 32000, 1],
    ] as const) {
      const source = Array.from({ length: channels }, () => chord(sampleRate / 4));
      const mp3 = encodeMp3Channels(source, sampleRate, { bitrate });

      const header = parseFrameHeader(mp3, 0)!;
      assert.ok(header, 'first frame did not parse');
      assert.equal(header.version, 1);
      assert.equal(header.layer, 3);
      assert.equal(header.bitrate, bitrate * 1000);
      assert.equal(header.sampleRate, sampleRate);
      assert.equal(header.channels, channels);

      // Every frame, not just the first: a drifting padding bit or a bad frame
      // length shows up as the walk failing to land on the next sync word.
      let offset = 0;
      let frames = 0;
      while (offset + 4 <= mp3.byteLength) {
        const frame = parseFrameHeader(mp3, offset);
        assert.ok(frame, `frame ${frames} at ${offset} did not parse`);
        const sideInfo = parseSideInfo(mp3, offset + 4, frame);
        assert.ok(sideInfo.mainDataBegin <= 511);
        offset += frame.frameLength;
        frames++;
      }
      assert.ok(frames > 5, `only ${frames} frames`);
    }
  });

  test('the encoded stream holds the bitrate it declares', () => {
    const seconds = 2;
    const signal = chord(RATE * seconds);
    for (const bitrate of [128, 192, 320]) {
      const mp3 = encodeMp3Channels([signal], RATE, { bitrate });
      const actual = (mp3.byteLength * 8) / seconds / 1000;
      // Above nominal, never below: the trailing frames that flush the
      // filterbank carry no input but still occupy their slot.
      assert.ok(actual >= bitrate * 0.99, `${bitrate}k came out at ${actual.toFixed(1)}k`);
      assert.ok(actual <= bitrate * 1.05, `${bitrate}k came out at ${actual.toFixed(1)}k`);
    }
  });

  test('encode and decode round-trip at the level and delay they claim', () => {
    // Deliberately untagged. With the Info tag the decoder trims the priming
    // silence and the answer is zero by construction, which tests the tag
    // rather than the pipeline; ENCODER_DELAY is the raw figure the tag is
    // built from, so it has to be measured where nothing has compensated for it.
    const signal = chord(RATE * 2);
    const decoded = decodeMp3(
      encodeMp3Channels([signal], RATE, { bitrate: 192, tag: false }),
    ).channelData[0]!;

    let bestSnr = -Infinity;
    let bestLag = -1;
    let bestScale = 0;
    for (let lag = 0; lag <= 1200; lag++) {
      let dot = 0;
      let outEnergy = 0;
      for (let i = 8000; i < 60000; i++) {
        const o = decoded[i + lag];
        if (o === undefined) continue;
        dot += o * signal[i]!;
        outEnergy += o * o;
      }
      if (outEnergy <= 0) continue;
      const scale = dot / outEnergy;

      let signalEnergy = 0;
      let noiseEnergy = 0;
      for (let i = 8000; i < 60000; i++) {
        const o = decoded[i + lag];
        if (o === undefined) continue;
        signalEnergy += signal[i]! ** 2;
        noiseEnergy += (o * scale - signal[i]!) ** 2;
      }
      const snr = 10 * Math.log10(signalEnergy / noiseEnergy);
      if (snr > bestSnr) {
        bestSnr = snr;
        bestLag = lag;
        bestScale = scale;
      }
    }

    assert.equal(bestLag, ENCODER_DELAY, 'round-trip latency moved');
    assert.ok(
      Math.abs(bestScale - 1) < 0.02,
      `output level was ${(1 / bestScale).toFixed(4)}x the input`,
    );
    assert.ok(bestSnr > 50, `round-trip SNR was only ${bestSnr.toFixed(1)} dB`);
  });

  test('nothing is lost from the start or the end of the input', () => {
    // A burst at each edge: encoders that under-pad the flush silently truncate
    // the last fraction of a second, and one that primes its filterbank wrongly
    // loses the attack.
    const n = RATE;
    for (const at of [0, n - 2000]) {
      const signal = new Float32Array(n);
      for (let i = 0; i < 2000; i++) {
        const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / 2000);
        signal[at + i] = 0.6 * window * Math.sin((2 * Math.PI * 1000 * i) / RATE);
      }

      const decoded = decodeMp3(encodeMp3Channels([signal], RATE, { bitrate: 320 })).channelData[0]!;
      assert.equal(decoded.length, n, 'output length does not match the input');

      let input = 0;
      for (let i = at; i < at + 2000; i++) input += signal[i]! ** 2;
      let output = 0;
      for (let i = at; i < at + 2000; i++) output += decoded[i]! ** 2;
      assert.ok(output > input * 0.9, `burst at ${at}: ${output.toFixed(1)} of ${input.toFixed(1)}`);
    }
  });

  test('stereo channels stay independent', () => {
    const n = RATE;
    const left = new Float32Array(n);
    const right = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      left[i] = 0.4 * Math.sin((2 * Math.PI * 440 * i) / RATE);
      right[i] = 0.4 * Math.sin((2 * Math.PI * 1760 * i) / RATE);
    }

    const decoded = decodeMp3(encodeMp3Channels([left, right], RATE, { bitrate: 192 }));
    assert.equal(decoded.channelData.length, 2);

    // Correlating each output against both inputs: a channel swap or a leak
    // between them shows up as the wrong pairing scoring higher.
    const score = (out: Float32Array, source: Float32Array): number => {
      let dot = 0;
      for (let i = 5000; i < 30000; i++) dot += out[i]! * source[i]!;
      return Math.abs(dot);
    };
    assert.ok(score(decoded.channelData[0]!, left) > 20 * score(decoded.channelData[0]!, right));
    assert.ok(score(decoded.channelData[1]!, right) > 20 * score(decoded.channelData[1]!, left));
  });

  test('pathological input produces valid frames rather than exceptions', () => {
    const cases: Array<[string, Float32Array]> = [
      ['silence', new Float32Array(RATE)],
      ['DC', new Float32Array(RATE).fill(0.8)],
      ['full-scale square', Float32Array.from({ length: RATE }, (_, i) => (i % 64 < 32 ? 1 : -1))],
      ['full-scale noise', noise(RATE, 1)],
      ['single sample', Float32Array.of(0.5)],
      ['one frame', new Float32Array(1152)],
      ['empty', new Float32Array(0)],
    ];

    for (const [name, signal] of cases) {
      const mp3 = encodeMp3Channels([signal], RATE, { bitrate: 128 });
      assert.ok(mp3.byteLength > 0, `${name}: produced nothing`);

      let offset = 0;
      let frames = 0;
      while (offset + 4 <= mp3.byteLength) {
        const header = parseFrameHeader(mp3, offset);
        assert.ok(header, `${name}: frame ${frames} at ${offset} is not a valid header`);
        const sideInfo = parseSideInfo(mp3, offset + 4, header);
        assert.ok(sideInfo.mainDataBegin <= 511, `${name}: main_data_begin out of range`);
        for (const granule of sideInfo.granules.flat()) {
          assert.ok(granule.part2_3_length <= 4095);
          assert.ok(granule.bigValues <= 288);
        }
        offset += header.frameLength;
        frames++;
      }
      assert.equal(offset, mp3.byteLength, `${name}: frame walk ended at ${offset} of ${mp3.byteLength}`);

      // And it must decode without throwing.
      const decoded = decodeMp3(mp3);
      assert.equal(decoded.channelData.length, 1);
    }
  });

  test('supplying thresholds shapes the scalefactors', () => {
    // With thresholds the quantiser stops treating every band alike: the
    // scalefactors take their shape from the masking curve, and the result still
    // has to survive the decoder and write the bits it declares.
    const signal = chord(RATE);
    const filterbank = new AnalysisFilterbank();
    const model = new PsychoacousticModel(RATE);
    const overlap = new Float32Array(576);
    const subbands = new Float32Array(576);
    const slot = new Float32Array(32);
    const spectrum = new Float32Array(576);
    const workspace = createWorkspace();
    const thresholds = new Float32Array(22);
    const back = new Float32Array(576);

    let shaped = 0;
    for (let g = 0; g < 20; g++) {
      analyseGranule(signal, g * 576, filterbank, overlap, subbands, slot, spectrum);
      model.analyse(signal, g * 576, spectrum, thresholds);

      const coding = quantizeGranule(
        { spectrum, sampleRate: RATE, budgetBits: 2464, thresholds, maxOuterIterations: 6 },
        workspace,
      );

      if (coding.scalefactors.some((value) => value > 0)) shaped++;
      assert.ok(coding.part2_3Length <= 2464);
      assert.ok(coding.scalefactors.every((value) => value <= 15), 'scalefactor out of range');
      assert.notEqual(
        chooseScalefacCompress(coding.scalefactors),
        null,
        'chose scalefactors no scalefac_compress can carry',
      );

      let nonZero = 576;
      while (nonZero > 0 && coding.values[nonZero - 1] === 0) nonZero--;
      requantize(
        coding.values,
        codingToGranuleInfo(coding),
        { long: coding.scalefactors, short: [new Int32Array(14), new Int32Array(14), new Int32Array(14)] },
        RATE,
        nonZero,
        back,
      );

      const writer = new BitWriter(1024);
      assert.equal(writeGranuleData(writer, coding, RATE), coding.huffmanBits);
    }

    assert.ok(shaped > 10, `only ${shaped} of 20 granules had shaped scalefactors`);
  });

  test('unsupported parameters are refused with an explanation', () => {
    const signal = new Float32Array(1152);
    assert.throws(() => encodeMp3Channels([signal], 22050, { bitrate: 128 }), /22050|Resample/);
    assert.throws(() => encodeMp3Channels([signal], RATE, { bitrate: 130 }), /130|bitrate|Supported/i);
    assert.throws(
      () => encodeMp3Channels([signal, signal, signal], RATE, {}),
      /mono and stereo|3 channels/,
    );
    assert.throws(
      () => encodeMp3Channels([signal, new Float32Array(576)], RATE, {}),
      /same length/,
    );
  });
});

describe('MP3 encoding through the library', () => {
  const RATE = 44100;

  /**
   * Where the LAME extension's encoder string sits inside a mono Info frame:
   * 4-byte header, 17 bytes of side information, then "Info", the flag word,
   * and the two 32-bit counts those flags enable.
   */
  const TAG_ENCODER_OFFSET = 4 + 17 + 4 + 4 + 4 + 4;

  function music(n: number): Float32Array {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      out[i] =
        0.35 * Math.sin((2 * Math.PI * 220 * i) / RATE) +
        0.2 * Math.sin((2 * Math.PI * 660 * i) / RATE) +
        0.12 * Math.sin((2 * Math.PI * 3300 * i) / RATE);
    }
    return out;
  }

  test('encode() routes mp3 and decode() reads it back', async () => {
    const audio = new Audio([music(RATE), music(RATE)], RATE);
    const bytes = await encode(audio, 'mp3', { bitrate: 192 });

    const decoded = await decode(bytes);
    assert.equal(decoded.info?.format, 'mp3');
    assert.equal(decoded.info?.lossy, true);
    assert.equal(decoded.sampleRate, RATE);
    assert.equal(decoded.channels, 2);
    // Gapless, so the round trip preserves length through the public API too.
    assert.equal(decoded.frames, audio.frames);
  });

  test('the Info tag describes the stream that follows it', () => {
    const signal = music(RATE * 2);
    const bytes = encodeMp3Channels([signal], RATE, { bitrate: 128 });

    const first = findFrame(bytes, 0)!;
    assert.equal(first.offset, 0, 'the tag frame must lead the stream');
    const tag = readVbrHeader(bytes, first.offset, first.header)!;
    assert.ok(tag, 'no Xing/Info tag was written');

    assert.equal(tag.vbr, false, 'CBR streams are tagged Info, not Xing');
    assert.equal(tag.bytes, bytes.byteLength, 'declared byte count is wrong');
    assert.equal(tag.encoderDelay, ENCODER_DELAY);
    const lameAt = first.offset + TAG_ENCODER_OFFSET;
    assert.equal(bytes[lameAt + 9]! & 0x0f, 1, 'the VBR method does not say constant bitrate');
    assert.equal(bytes[lameAt + 20], 128, 'byte 20 is not the constant bitrate');
    const musicLength = new DataView(bytes.buffer, bytes.byteOffset).getUint32(lameAt + 28);
    assert.equal(musicLength, bytes.byteLength, 'the music length must count the tag frame');

    // The count excludes the tag frame itself, which is what a decoder assumes
    // when it skips that frame and then sizes its output buffer.
    let offset = first.header.frameLength;
    let audioFrames = 0;
    while (offset + 4 <= bytes.byteLength) {
      const header = parseFrameHeader(bytes, offset)!;
      assert.ok(header, `frame at ${offset} did not parse`);
      offset += header.frameLength;
      audioFrames++;
    }
    assert.equal(tag.frames, audioFrames, 'declared frame count is wrong');

    // Delay plus padding is exactly the silence the frames hold beyond the
    // input, or gapless trimming lands in the wrong place.
    assert.equal(
      audioFrames * 1152 - tag.encoderDelay! - tag.encoderPadding!,
      signal.length,
      'delay and padding do not account for the frame padding',
    );
  });

  test('a round trip returns exactly the samples it was given', () => {
    // The point of the tag. Without it a decoder cannot know where the audio
    // starts, and every round trip grows by a fraction of a second.
    for (const length of [RATE * 2, RATE * 2 + 577, 5000, 1152]) {
      const signal = music(length);
      const decoded = decodeMp3(encodeMp3Channels([signal], RATE, { bitrate: 192 }));
      const out = decoded.channelData[0]!;

      assert.equal(out.length, length, `${length}-sample input came back as ${out.length}`);

      // Aligned at lag zero, not merely the same length.
      let signalEnergy = 0;
      let noiseEnergy = 0;
      for (let i = 2000; i < length - 2000; i++) {
        signalEnergy += signal[i]! ** 2;
        noiseEnergy += (signal[i]! - out[i]!) ** 2;
      }
      if (signalEnergy > 0) {
        const snr = 10 * Math.log10(signalEnergy / noiseEnergy);
        assert.ok(snr > 50, `${length}-sample round trip aligned to only ${snr.toFixed(1)} dB`);
      }
    }
  });

  test('the tag can be turned off for stream fragments', () => {
    const signal = music(RATE);
    const tagged = encodeMp3Channels([signal], RATE, { bitrate: 128 });
    const untagged = encodeMp3Channels([signal], RATE, { bitrate: 128, tag: false });

    const first = findFrame(untagged, 0)!;
    assert.equal(readVbrHeader(untagged, first.offset, first.header), null);
    assert.ok(tagged.byteLength > untagged.byteLength, 'the tag should cost a frame');

    // Without the tag there is nothing to trim by, so the priming silence stays.
    assert.ok(decodeMp3(untagged).channelData[0]!.length > signal.length);
  });

  test('the LAME extension reports the channel mode and source rate', () => {
    // Stereo mode in bits 2-4 (0 mono, 1 stereo), source rate in bits 6-7 (0 32, 1 44.1, 2 48 kHz).
    for (const [channels, rate, misc] of [[1, 48000, 0x80], [2, 44100, 0x44], [2, 32000, 0x04]] as const) {
      const source = Array.from({ length: channels }, () => music(rate));
      const bytes = encodeMp3Channels(source, rate, { bitrate: 128 });
      const { offset, header } = findFrame(bytes, 0)!;
      const lameAt = offset + 4 + header.sideInfoSize + 4 + 4 + 4 + 4;
      assert.equal(bytes[lameAt + 24], misc, `byte 24 for ${channels} channels at ${rate} Hz`);
    }
  });

  test('the LAME extension carries the music and tag CRCs', () => {
    assert.equal(lameCrc(new TextEncoder().encode('123456789')), 0xbb3d, 'not CRC-16/ARC');
    for (const vbr of [false, true]) {
      const bytes = encodeMp3Channels([music(RATE)], RATE, { bitrate: 128, vbr });
      const { offset, header } = findFrame(bytes, 0)!;
      const lameAt = offset + TAG_ENCODER_OFFSET + (vbr ? 100 : 0);
      const field = (at: number) => (bytes[at]! << 8) | bytes[at + 1]!;
      const audio = bytes.subarray(offset + header.frameLength);
      assert.equal(field(lameAt + 32), lameCrc(audio), 'the music CRC is wrong');
      assert.equal(field(lameAt + 34), lameCrc(bytes.subarray(offset, lameAt + 34)), 'the tag CRC is wrong');
    }
  });

  test('an encoder name padded with NUL bytes is read, as LAME writes short names', () => {
    const signal = music(RATE);
    const bytes = encodeMp3Channels([signal], RATE, { bitrate: 128 });
    const first = findFrame(bytes, 0)!;
    bytes.set(new TextEncoder().encode('LAME4.0 \0'), first.offset + TAG_ENCODER_OFFSET);
    assert.equal(readVbrHeader(bytes, first.offset, first.header)!.encoderDelay, ENCODER_DELAY);
    assert.equal(decodeMp3(bytes).channelData[0]!.length, signal.length);
  });

  test('a tag from any encoder is read, not just LAME’s', () => {
    // Gating on a whitelist of encoder names is why other encoders label
    // themselves "LAME". audiobox names itself honestly, so the reader has to
    // accept an unfamiliar name — and reject implausible values instead.
    const bytes = encodeMp3Channels([music(RATE)], RATE, { bitrate: 128 });
    const first = findFrame(bytes, 0)!;
    assert.equal(
      String.fromCharCode(...bytes.subarray(first.offset + TAG_ENCODER_OFFSET, first.offset + TAG_ENCODER_OFFSET + 8)),
      'audiobox',
    );
    assert.equal(readVbrHeader(bytes, first.offset, first.header)!.encoderDelay, ENCODER_DELAY);

    // A wildly out-of-range delay means those bytes are not a LAME tag.
    const corrupt = Uint8Array.from(bytes);
    const delayOffset = first.offset + TAG_ENCODER_OFFSET + 21;
    corrupt[delayOffset] = 0xff;
    corrupt[delayOffset + 1] = 0xff;
    const tag = readVbrHeader(corrupt, first.offset, first.header)!;
    assert.equal(tag.encoderDelay, undefined, 'an implausible delay should be ignored');
  });
});

describe('MP3 encoding against ffmpeg', { skip: FFMPEG ? false : 'ffmpeg not installed' }, () => {
  let dir: string;

  test('setup', () => {
    dir = mkdtempSync(join(tmpdir(), 'audiobox-mp3enc-'));
  });

  test.after(() => {
    if (dir) rmSync(dir, { recursive: true, force: true });
  });

  const RATE = 44100;

  function music(n: number, offset = 0): Float32Array {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      out[i] =
        0.35 * Math.sin((2 * Math.PI * (220 + offset) * i) / RATE) +
        0.2 * Math.sin((2 * Math.PI * (660 + offset) * i) / RATE) +
        0.12 * Math.sin((2 * Math.PI * (3300 + offset) * i) / RATE);
    }
    return out;
  }

  test('ffmpeg accepts the tag CRC of a variable-bitrate stream', () => {
    // The seek table sits inside the CRC range only for VBR.
    const path = join(dir, 'vbr-crc.mp3');
    writeFileSync(path, encodeMp3Channels([music(RATE)], RATE, { bitrate: 192, vbr: true }));
    const args = ['-v', 'error', '-show_entries', 'stream_tags=encoder', '-of', 'default=nk=1:nw=1', path];
    assert.equal(execFileSync('ffprobe', args).toString().trim(), 'audiobox');
  });

  for (const [bitrate, channels] of [
    [128, 1],
    [192, 2],
    [320, 2],
  ] as const) {
    test(`ffmpeg decodes ${bitrate}k ${channels === 1 ? 'mono' : 'stereo'} without complaint`, () => {
      const source = Array.from({ length: channels }, (_, c) => music(RATE * 2, c * 110));
      const path = join(dir, `enc-${bitrate}-${channels}.mp3`);
      // Noise shaping off. This test exists to catch errors in the transform and
      // bitstream that our own decoder might undo, and for that it wants the
      // sharpest SNR it can get. Shaping deliberately trades SNR away, which
      // would force the threshold down to where it no longer catches anything.
      // What the default settings produce is checked separately below.
      const bytes = encodeMp3Channels(source, RATE, { bitrate, psychoacoustic: false });
      writeFileSync(path, bytes);

      // ffprobe must agree with what we asked for. A stream that plays but
      // misreports itself breaks every tool that reads metadata before decoding.
      const probe = execFileSync('ffprobe', [
        '-v', 'error',
        '-show_entries', 'stream=codec_name,sample_rate,channels,bit_rate:stream_tags=encoder',
        '-of', 'default=nk=1:nw=1',
        path,
      ]).toString().trim().split('\n').map((line) => line.trim());
      // ffmpeg reports the encoder name only when the tag CRC is right.
      const expected = ['mp3', String(RATE), String(channels), String(bitrate * 1000), 'audiobox'];
      assert.deepEqual(probe, expected);

      // Anything on stderr is a complaint: a bad header, a frame it had to
      // resynchronise past, a reservoir it could not resolve.
      const raw = join(dir, `enc-${bitrate}-${channels}.raw`);
      const run = spawnSync('ffmpeg', ['-v', 'warning', '-i', path, '-f', 'f32le', raw, '-y'], {
        encoding: 'utf8',
      });
      assert.equal(run.status, 0, 'ffmpeg exited non-zero');
      assert.equal(run.stderr.trim(), '', 'ffmpeg reported a problem with our stream');

      const buffer = readFileSync(raw);
      const reference = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);

      // ffmpeg has no reason to trust our encoder name, so it does not apply our
      // delay and padding; its output is the untrimmed stream. Ours is trimmed,
      // so the two differ by exactly the priming silence.
      const decoded = decodeMp3(bytes);
      const lag = ENCODER_DELAY + 529;

      let signal = 0;
      let noise = 0;
      for (let i = 4000; i < 60000; i++) {
        for (let c = 0; c < channels; c++) {
          const theirs = reference[(i + lag) * channels + c];
          if (theirs === undefined) continue;
          const ours = decoded.channelData[c]![i]!;
          signal += theirs * theirs;
          noise += (ours - theirs) ** 2;
        }
      }
      const snr = 10 * Math.log10(signal / noise);
      assert.ok(snr > 45, `our decode and ffmpeg's differ by ${snr.toFixed(1)} dB`);

      // And ffmpeg's decode must resemble the original. This is the only
      // measurement in the suite that does not involve our own decoder, which
      // makes it the one that can catch an encoder error our decoder happens to
      // undo. It is also the gate that settled the analysis window: a wrong
      // window scores 41 dB here while our own round trip still looks fine.
      let sourceEnergy = 0;
      let error = 0;
      for (let i = 4000; i < 60000; i++) {
        const theirs = reference[(i + lag) * channels];
        if (theirs === undefined) continue;
        sourceEnergy += source[0]![i]! ** 2;
        error += (source[0]![i]! - theirs) ** 2;
      }
      const quality = 10 * Math.log10(sourceEnergy / error);
      assert.ok(quality > 65, `ffmpeg heard only ${quality.toFixed(1)} dB against the source`);
    });
  }

  test('default settings also decode cleanly in ffmpeg', () => {
    // Everything above runs with shaping disabled, so this covers what callers
    // actually get. The bar is lower because shaping spends SNR on purpose; what
    // matters here is that ffmpeg reads the stream without complaint.
    const source = [music(RATE * 2), music(RATE * 2, 110)];
    const path = join(dir, 'enc-default.mp3');
    writeFileSync(path, encodeMp3Channels(source, RATE, { bitrate: 192 }));

    const raw = join(dir, 'enc-default.raw');
    const run = spawnSync('ffmpeg', ['-v', 'warning', '-i', path, '-f', 'f32le', raw, '-y'], {
      encoding: 'utf8',
    });
    assert.equal(run.status, 0, 'ffmpeg exited non-zero');
    assert.equal(run.stderr.trim(), '', 'ffmpeg reported a problem with our stream');

    const buffer = readFileSync(raw);
    const reference = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    const lag = ENCODER_DELAY + 529;

    let signal = 0;
    let error = 0;
    for (let i = 4000; i < 60000; i++) {
      const theirs = reference[(i + lag) * 2];
      if (theirs === undefined) continue;
      signal += source[0]![i]! ** 2;
      error += (source[0]![i]! - theirs) ** 2;
    }
    const quality = 10 * Math.log10(signal / error);
    assert.ok(quality > 40, `ffmpeg heard only ${quality.toFixed(1)} dB against the source`);
  });
});

describe('MP3 psychoacoustic model', () => {
  const RATE = 44100;
  const FFT = 1024;

  /** Dense, varied material — the case where a bit budget actually binds. */
  function music(n: number): Float32Array {
    let seed = 20250804;
    const random = (): number => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x40000000 - 1;
    };
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / RATE;
      const envelope = 0.5 + 0.5 * Math.sin(2 * Math.PI * 1.7 * t);
      out[i] =
        0.3 * Math.sin(2 * Math.PI * 196 * t) +
        0.22 * Math.sin(2 * Math.PI * 294 * t) +
        0.16 * Math.sin(2 * Math.PI * 392 * t) * envelope +
        0.09 * Math.sin(2 * Math.PI * 1567 * t) * envelope +
        0.05 * Math.sin(2 * Math.PI * 3136 * t) +
        0.03 * random();
    }
    return out;
  }

  /**
   * Noise-to-mask ratio, in dB, averaged over bands and frames.
   *
   * The measure the model is actually optimising. Total SNR cannot serve here:
   * shaping raises total error on purpose, moving it under the signal, so SNR
   * rates a working model as neutral or worse — which is exactly what it did
   * before this suite existed.
   *
   * The masker below is deliberately not the encoder's. Fixed bands, fixed
   * masking depth, no absolute threshold, no temporal term: it shares no
   * parameter with the model under test, so the model cannot satisfy it except
   * by genuinely putting less noise where the signal cannot hide it.
   */
  function noiseToMask(source: Float32Array, decoded: Float32Array): {
    mean: number;
    aboveMask: number;
  } {
    const bark = (hz: number): number =>
      13 * Math.atan(0.00076 * hz) + 3.5 * Math.atan((hz / 7500) ** 2);

    const edges: number[] = [];
    for (let b = 0; b <= 24; b++) {
      let low = 0;
      let high = RATE / 2;
      for (let i = 0; i < 40; i++) {
        const mid = (low + high) / 2;
        if (bark(mid) < b) low = mid;
        else high = mid;
      }
      edges.push(Math.round((low * FFT) / RATE));
    }
    const bands: Array<[number, number]> = [];
    for (let b = 0; b < 24; b++) {
      if (edges[b + 1]! > edges[b]!) bands.push([edges[b]!, Math.min(edges[b + 1]!, FFT / 2)]);
    }

    const spread = new Float64Array(bands.length * bands.length);
    for (let to = 0; to < bands.length; to++) {
      let sum = 0;
      for (let from = 0; from < bands.length; from++) {
        const dz = to - from;
        const weight = 10 ** ((dz >= 0 ? -25 * dz : 10 * dz) / 10);
        spread[to * bands.length + from] = weight;
        sum += weight;
      }
      for (let from = 0; from < bands.length; from++) spread[to * bands.length + from]! /= sum;
    }

    const window = Float64Array.from({ length: FFT }, (_, i) =>
      0.5 - 0.5 * Math.cos((2 * Math.PI * i) / FFT),
    );

    const powerAt = (signal: Float32Array, offset: number): Float64Array => {
      const real = new Float64Array(FFT);
      const imag = new Float64Array(FFT);
      for (let i = 0; i < FFT; i++) real[i] = (signal[offset + i] ?? 0) * window[i]!;
      fft(real, imag);
      const out = new Float64Array(FFT / 2);
      const scale = 2 / FFT;
      for (let k = 0; k < FFT / 2; k++) {
        out[k] = (real[k]! * scale) ** 2 + (imag[k]! * scale) ** 2;
      }
      return out;
    };

    const banded = (spectrum: Float64Array): Float64Array =>
      Float64Array.from(bands, ([low, high]) => {
        let sum = 0;
        for (let k = low; k < high; k++) sum += spectrum[k]!;
        return sum / Math.max(1, high - low);
      });

    const error = new Float32Array(source.length);
    for (let i = 0; i < source.length; i++) error[i] = source[i]! - (decoded[i] ?? 0);

    let total = 0;
    let count = 0;
    let above = 0;

    for (let offset = 4096; offset + FFT < source.length - 4096; offset += FFT) {
      const sourceBands = banded(powerAt(source, offset));
      const errorBands = banded(powerAt(error, offset));

      for (let b = 0; b < bands.length; b++) {
        if (sourceBands[b]! < 1e-12) continue;
        let masked = 0;
        for (let from = 0; from < bands.length; from++) {
          masked += spread[b * bands.length + from]! * sourceBands[from]!;
        }
        // 15 dB below the spread energy, flat: no tonality estimate, so this
        // cannot be gamed by agreeing with the encoder about what is tonal.
        const threshold = masked / 10 ** 1.5;
        const nmr = 10 * Math.log10((errorBands[b]! + 1e-30) / (threshold + 1e-30));
        total += nmr;
        count++;
        if (nmr > 0) above++;
      }
    }

    return { mean: total / count, aboveMask: (100 * above) / count };
  }

  test('shaping puts measurably less noise where it can be heard', () => {
    const source = music(RATE * 2);

    for (const bitrate of [128, 192]) {
      const flat = decodeMp3(
        encodeMp3Channels([source], RATE, { bitrate, psychoacoustic: false }),
      ).channelData[0]!;
      const shaped = decodeMp3(
        encodeMp3Channels([source], RATE, { bitrate }),
      ).channelData[0]!;

      const before = noiseToMask(source, flat);
      const after = noiseToMask(source, shaped);

      assert.ok(
        after.mean < before.mean - 1,
        `${bitrate}k: noise-to-mask ${before.mean.toFixed(2)} -> ${after.mean.toFixed(2)} dB`,
      );
      assert.ok(
        after.aboveMask < before.aboveMask,
        `${bitrate}k: audible bands ${before.aboveMask.toFixed(1)}% -> ${after.aboveMask.toFixed(1)}%`,
      );
    }
  });

  test('and pays for it in total SNR, which is why SNR is not the gate', () => {
    // Documents the trade rather than guarding against it. A change that
    // improved both would be welcome; one that improved SNR while worsening
    // noise-to-mask would mean the shaping had stopped working.
    const source = music(RATE * 2);
    const snr = (decoded: Float32Array): number => {
      let signal = 0;
      let noise = 0;
      for (let i = 4000; i < source.length - 4000; i++) {
        signal += source[i]! ** 2;
        noise += (source[i]! - (decoded[i] ?? 0)) ** 2;
      }
      return 10 * Math.log10(signal / noise);
    };

    const flat = snr(
      decodeMp3(encodeMp3Channels([source], RATE, { bitrate: 128, psychoacoustic: false }))
        .channelData[0]!,
    );
    const shaped = snr(
      decodeMp3(encodeMp3Channels([source], RATE, { bitrate: 128 })).channelData[0]!,
    );

    assert.ok(shaped < flat, 'shaping should cost total SNR while improving what is heard');
    assert.ok(shaped > 15, `shaping cost far too much: ${shaped.toFixed(1)} dB`);
  });

  test('thresholds stay inside the range scalefactors can express', () => {
    // slen2 never exceeds 3 bits, so bands 11-20 cap at 7 while bands 0-10 reach
    // 15. Exceeding either leaves no legal scalefac_compress at all, and the
    // granule is dropped — silently, and completely.
    const source = music(RATE);
    const filterbank = new AnalysisFilterbank();
    const model = new PsychoacousticModel(RATE);
    const overlap = new Float32Array(576);
    const subbands = new Float32Array(576);
    const slot = new Float32Array(32);
    const spectrum = new Float32Array(576);
    const thresholds = new Float32Array(22);
    const workspace = createWorkspace();

    let coded = 0;
    for (let g = 0; g < 60; g++) {
      analyseGranule(source, g * 576, filterbank, overlap, subbands, slot, spectrum);
      model.analyse(source, g * 576, spectrum, thresholds);

      for (let sfb = 0; sfb < 22; sfb++) {
        assert.ok(Number.isFinite(thresholds[sfb]!), `band ${sfb} threshold is not finite`);
        assert.ok(thresholds[sfb]! >= 0, `band ${sfb} threshold is negative`);
      }

      const coding = quantizeGranule(
        { spectrum, sampleRate: RATE, budgetBits: 1636, thresholds },
        workspace,
      );

      for (let sfb = 0; sfb < 11; sfb++) {
        assert.ok(coding.scalefactors[sfb]! <= 15, `band ${sfb} scalefactor out of range`);
      }
      for (let sfb = 11; sfb < 21; sfb++) {
        assert.ok(coding.scalefactors[sfb]! <= 7, `band ${sfb} exceeds what slen2 can carry`);
      }
      assert.notEqual(
        chooseScalefacCompress(coding.scalefactors),
        null,
        'no scalefac_compress can carry these scalefactors',
      );

      if (coding.part2_3Length > 0) coded++;
    }

    // The failure this guards against turns granules into silence, so a granule
    // count is the thing to assert.
    assert.ok(coded > 55, `only ${coded} of 60 granules carried any data`);
  });

  test('a loud tone raises the threshold of the bands around it', () => {
    // Masking, tested the way it is defined: the same quiet probe, measured with
    // and without a masker beside it. Reading thresholds off a single signal
    // does not work — they are capped at each band's own energy, so a near-empty
    // band reports its leakage rather than what is masking it.
    const bands = SFB_LONG[RATE]!;
    const bandOf = (hz: number): number => {
      const line = Math.round((hz * 576 * 2) / RATE);
      let sfb = 0;
      while (sfb < 21 && bands[sfb + 1]! <= line) sfb++;
      return sfb;
    };

    const measure = (masker: boolean, probeHz: number): number => {
      const n = RATE;
      const signal = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        signal[i] = 0.02 * Math.sin((2 * Math.PI * probeHz * i) / RATE);
        if (masker) signal[i] += 0.7 * Math.sin((2 * Math.PI * 1000 * i) / RATE);
      }

      const filterbank = new AnalysisFilterbank();
      const model = new PsychoacousticModel(RATE);
      const overlap = new Float32Array(576);
      const subbands = new Float32Array(576);
      const slot = new Float32Array(32);
      const spectrum = new Float32Array(576);
      const thresholds = new Float32Array(22);

      // A few granules in, so the filterbank and the temporal cap have settled.
      for (let g = 0; g < 12; g++) {
        analyseGranule(signal, g * 576, filterbank, overlap, subbands, slot, spectrum);
        model.analyse(signal, g * 576, spectrum, thresholds);
      }
      return thresholds[bandOf(probeHz)]!;
    };

    // A probe above the masker: its threshold must rise once the masker is there.
    const aloneAbove = measure(false, 1600);
    const maskedAbove = measure(true, 1600);
    assert.ok(
      maskedAbove > aloneAbove * 4,
      `1.6 kHz probe: threshold ${aloneAbove.toExponential(2)} -> ${maskedAbove.toExponential(2)}`,
    );

    // Masking spreads further upward than downward, so an equally distant probe
    // below the masker must be masked less.
    const aloneBelow = measure(false, 640);
    const maskedBelow = measure(true, 640);
    assert.ok(
      maskedAbove / aloneAbove > maskedBelow / aloneBelow,
      `masking should spread upward more than downward: ` +
        `${(maskedBelow / aloneBelow).toFixed(1)}x below, ${(maskedAbove / aloneAbove).toFixed(1)}x above`,
    );

    // And it must not reach the top of the spectrum.
    const aloneFar = measure(false, 15000);
    const maskedFar = measure(true, 15000);
    assert.ok(
      maskedFar < aloneFar * 4,
      `a 1 kHz tone should not mask 15 kHz: ${(maskedFar / aloneFar).toFixed(1)}x`,
    );
  });
});

describe('MP3 window switching', () => {
  const RATE = 44100;

  /** Near silence broken by hard attacks — the case pre-echo is audible in. */
  function castanets(n: number): { signal: Float32Array; hits: number[] } {
    let seed = 99;
    const random = (): number => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x40000000 - 1;
    };

    const signal = new Float32Array(n);
    const hits: number[] = [];
    for (let at = 12000; at + 6000 < n; at += 11025) hits.push(at);

    for (let i = 0; i < n; i++) signal[i] = 0.0004 * random();
    for (const at of hits) {
      for (let i = 0; i < 4000 && at + i < n; i++) {
        signal[at + i]! += 0.85 * Math.exp(-i / 380) * random();
      }
    }
    return { signal, hits };
  }

  /** Counts granules by block type, reading the side information back. */
  function blockTypes(bytes: Uint8Array): number[] {
    const counts = [0, 0, 0, 0];
    let offset = 0;
    while (offset + 4 <= bytes.byteLength) {
      const header = parseFrameHeader(bytes, offset);
      if (!header) break;
      const side = parseSideInfo(bytes, offset + 4, header);
      for (const granule of side.granules.flat()) {
        counts[granule.windowSwitching ? granule.blockType : 0]!++;
      }
      offset += header.frameLength;
    }
    return counts;
  }

  test('transients are coded with short blocks, in a legal sequence', () => {
    const { signal } = castanets(RATE * 3);
    const bytes = encodeMp3Channels([signal], RATE, { bitrate: 128 });
    const counts = blockTypes(bytes);

    assert.ok(counts[2]! > 10, `only ${counts[2]} short granules`);
    assert.ok(counts[0]! > counts[2]!, 'steady material should still be coded long');

    // A short run has to be entered through a start block and left through a
    // stop block, so the two must balance. They are the asymmetric windows whose
    // overlap adds back to unity; getting the sequence wrong decodes to a click.
    assert.equal(counts[1], counts[3], `${counts[1]} start blocks, ${counts[3]} stop blocks`);
    assert.ok(counts[1]! > 0, 'no transition blocks were emitted');

    // And every block type must appear at least once.
    for (let type = 0; type < 4; type++) {
      assert.ok(counts[type]! > 0, `block type ${type} never appears`);
    }
  });

  test('the block type sequence never jumps between window shapes', () => {
    const { signal } = castanets(RATE * 3);
    const bytes = encodeMp3Channels([signal], RATE, { bitrate: 128 });

    const order: number[] = [];
    let offset = 0;
    while (offset + 4 <= bytes.byteLength) {
      const header = parseFrameHeader(bytes, offset)!;
      const side = parseSideInfo(bytes, offset + 4, header);
      for (const granule of side.granules.flat()) {
        order.push(granule.windowSwitching ? granule.blockType : 0);
      }
      offset += header.frameLength;
    }

    // Legal successors, straight from the window shapes: a long window may be
    // followed by another long or by a start; a start must be followed by short.
    const allowed: Record<number, number[]> = {
      0: [0, 1],
      1: [2],
      2: [2, 3],
      3: [0, 1],
    };
    for (let i = 1; i < order.length; i++) {
      assert.ok(
        allowed[order[i - 1]!]!.includes(order[i]!),
        `granule ${i}: block type ${order[i - 1]} cannot be followed by ${order[i]}`,
      );
    }
  });

  test('short blocks reduce pre-echo on transients', () => {
    // The measurement that justifies the whole feature: error energy in the
    // silence *before* each attack, where a long block smears noise that belongs
    // to the attack itself.
    const { signal, hits } = castanets(RATE * 3);

    const preEcho = (decoded: Float32Array): number => {
      let noise = 0;
      let count = 0;
      for (const at of hits) {
        for (let i = Math.max(0, at - 1000); i < at - 60; i++) {
          const error = signal[i]! - (decoded[i] ?? 0);
          noise += error * error;
          count++;
        }
      }
      return 10 * Math.log10(noise / count + 1e-30);
    };

    for (const bitrate of [128, 192]) {
      const flat = decodeMp3(
        encodeMp3Channels([signal], RATE, { bitrate, windowSwitching: false }),
      ).channelData[0]!;
      const switched = decodeMp3(
        encodeMp3Channels([signal], RATE, { bitrate }),
      ).channelData[0]!;

      const before = preEcho(flat);
      const after = preEcho(switched);
      assert.ok(
        after < before - 3,
        `${bitrate}k: pre-echo ${before.toFixed(2)} -> ${after.toFixed(2)} dB`,
      );
    }
  });

  test('steady material is left on long blocks', () => {
    // Short blocks cost bits, so firing on sustained tones would be a loss with
    // nothing to show for it.
    const n = RATE * 2;
    const tone = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      // Faded in over 50 ms. A tone that starts at full amplitude from digital
      // silence *is* a transient, and the detector is right to say so — but that
      // says nothing about whether it leaves sustained material alone, which is
      // what this test is for.
      const fade = Math.min(1, i / (RATE * 0.05));
      tone[i] =
        fade *
        (0.4 * Math.sin((2 * Math.PI * 440 * i) / RATE) +
          0.2 * Math.sin((2 * Math.PI * 1320 * i) / RATE));
    }

    const counts = blockTypes(encodeMp3Channels([tone], RATE, { bitrate: 128 }));
    assert.equal(counts[2], 0, `${counts[2]} short granules on a steady tone`);
    assert.equal(counts[1], 0, 'no transition blocks should be needed either');
  });

  test('short granules survive a round trip', () => {
    // Everything about a short granule differs from a long one — band layout,
    // scalefactor grouping, region boundaries, transmission order — so it needs
    // its own end-to-end check rather than relying on the pre-echo measurement.
    const { signal } = castanets(RATE * 2);
    const bytes = encodeMp3Channels([signal], RATE, { bitrate: 192 });

    let offset = 0;
    let shortGranules = 0;
    while (offset + 4 <= bytes.byteLength) {
      const header = parseFrameHeader(bytes, offset)!;
      const side = parseSideInfo(bytes, offset + 4, header);
      for (const granule of side.granules.flat()) {
        if (granule.windowSwitching && granule.blockType === 2) {
          shortGranules++;
          assert.equal(granule.mixedBlock, false, 'mixed blocks are not emitted');
          for (const gain of granule.subblockGain) {
            assert.ok(gain >= 0 && gain <= 7, `subblock_gain ${gain} out of range`);
          }
        }
        assert.ok(granule.part2_3_length <= 4095);
        assert.ok(granule.bigValues <= 288);
      }
      offset += header.frameLength;
    }
    assert.ok(shortGranules > 0, 'no short granules to check');
    assert.equal(offset, bytes.byteLength, 'frame walk did not land on the end');

    const decoded = decodeMp3(bytes).channelData[0]!;
    assert.equal(decoded.length, signal.length);

    let signalEnergy = 0;
    let noise = 0;
    for (let i = 2000; i < signal.length - 2000; i++) {
      signalEnergy += signal[i]! ** 2;
      noise += (signal[i]! - decoded[i]!) ** 2;
    }
    const snr = 10 * Math.log10(signalEnergy / noise);
    assert.ok(snr > 5, `round trip with short blocks scored only ${snr.toFixed(1)} dB`);
  });
});

describe('MP3 variable bitrate', () => {
  const RATE = 44100;
  const SECONDS = 3;
  const N = RATE * SECONDS;

  function material(kind: 'quiet' | 'dense', length = N): Float32Array {
    let seed = 4242;
    const random = (): number => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x40000000 - 1;
    };
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      if (kind === 'quiet') out[i] = 0.15 * Math.sin((2 * Math.PI * 440 * i) / RATE);
      else out[i] = 0.3 * random();
    }
    return out;
  }

  function frameBitrates(bytes: Uint8Array): number[] {
    const rates: number[] = [];
    let offset = 0;
    while (offset + 4 <= bytes.byteLength) {
      const header = parseFrameHeader(bytes, offset);
      if (!header) break;
      rates.push(header.bitrate / 1000);
      offset += header.frameLength;
    }
    return rates;
  }

  test('frames take different bitrates, and the stream stays walkable', () => {
    const bytes = encodeMp3Channels([material('quiet')], RATE, { bitrate: 192, vbr: true });
    const rates = frameBitrates(bytes);

    assert.ok(rates.length > 50, `only ${rates.length} frames`);
    assert.ok(new Set(rates).size > 1, 'every frame took the same bitrate — that is not VBR');
    for (const kbps of rates) {
      assert.ok(kbps > 0 && kbps <= 192, `frame bitrate ${kbps} outside 1..192`);
    }

    // The walk landing exactly on the end is the real check: each frame's length
    // comes from its own header, so a wrong bitrate index desynchronises here.
    let offset = 0;
    while (offset + 4 <= bytes.byteLength) {
      offset += parseFrameHeader(bytes, offset)!.frameLength;
    }
    assert.equal(offset, bytes.byteLength);
  });

  test('easy material costs less than constant bitrate, hard material does not', () => {
    const rate = (signal: Float32Array, vbr: boolean): number => {
      const bytes = encodeMp3Channels([signal], RATE, { bitrate: 192, vbr });
      return (bytes.byteLength * 8) / SECONDS / 1000;
    };

    // A quiet tone is cheap to code, and VBR should stop paying for what it does
    // not need.
    const quietCbr = rate(material('quiet'), false);
    const quietVbr = rate(material('quiet'), true);
    assert.ok(quietVbr < quietCbr * 0.85, `quiet: ${quietCbr.toFixed(1)} -> ${quietVbr.toFixed(1)}`);

    // Noise genuinely needs the bits, so there is nothing to save and VBR should
    // not pretend otherwise.
    const denseVbr = rate(material('dense'), true);
    assert.ok(denseVbr > quietVbr * 1.2, `dense material should cost more than a quiet tone`);
  });

  test('the saving does not come out of quality', () => {
    const signal = material('quiet');
    const snr = (vbr: boolean): number => {
      const decoded = decodeMp3(
        encodeMp3Channels([signal], RATE, { bitrate: 192, vbr }),
      ).channelData[0]!;
      let sig = 0;
      let noise = 0;
      for (let i = 4000; i < N - 4000; i++) {
        sig += signal[i]! ** 2;
        noise += (signal[i]! - decoded[i]!) ** 2;
      }
      return 10 * Math.log10(sig / noise);
    };

    const constant = snr(false);
    const variable = snr(true);
    assert.ok(
      variable > constant - 2,
      `VBR gave up ${(constant - variable).toFixed(1)} dB for its saving`,
    );
  });

  test('quality trades size against fidelity monotonically', () => {
    const signal = material('dense');
    const sizes = [0, 3, 6, 9].map(
      (quality) =>
        encodeMp3Channels([signal], RATE, { bitrate: 320, vbr: true, quality }).byteLength,
    );

    for (let i = 1; i < sizes.length; i++) {
      assert.ok(
        sizes[i]! <= sizes[i - 1]!,
        `quality is not monotone: ${sizes.join(', ')}`,
      );
    }
    assert.ok(sizes[0]! > sizes[3]! * 1.3, `quality 0 and 9 differ too little: ${sizes.join(', ')}`);
  });

  test('a variable-bitrate stream is tagged Xing and carries a seek table', () => {
    const bytes = encodeMp3Channels([material('quiet')], RATE, { bitrate: 192, vbr: true });
    const first = findFrame(bytes, 0)!;
    const tag = readVbrHeader(bytes, first.offset, first.header)!;

    assert.equal(tag.vbr, true, 'a VBR stream must be tagged Xing, not Info');
    assert.equal(tag.bytes, bytes.byteLength);
    assert.equal(tag.encoderDelay, ENCODER_DELAY);

    // The seek table sits after the two counts; without it, players seeking a
    // VBR file assume byte position tracks time, which is what VBR breaks.
    const flagsAt = first.offset + 4 + 17 + 4;
    const flags = (bytes[flagsAt + 2]! << 8) | bytes[flagsAt + 3]!;
    assert.equal(flags & 0x04, 0x04, 'the seek table flag is not set');

    const tableAt = flagsAt + 4 + 4 + 4;
    let previous = -1;
    for (let i = 0; i < 100; i++) {
      const entry = bytes[tableAt + i]!;
      assert.ok(entry >= previous, `seek table is not monotone at entry ${i}`);
      previous = entry;
    }
  });

  test('the LAME extension describes a variable-bitrate stream as VBR', () => {
    const bytes = encodeMp3Channels([material('quiet')], RATE, { bitrate: 192, vbr: true });
    // After the header, side info, `Xing`, flags, both counts and the seek table.
    const lameAt = findFrame(bytes, 0)!.offset + 4 + 17 + 4 + 4 + 4 + 4 + 100;
    assert.equal(bytes[lameAt + 9]! & 0x0f, 4, 'the VBR method still says constant bitrate');
    assert.equal(bytes[lameAt + 20], 32, 'byte 20 is not the lowest bitrate a frame may take');
  });

  test('a variable-bitrate round trip is still gapless', () => {
    // A length that is not a whole number of granules, so the padding figure in
    // the tag has to be right rather than incidentally zero.
    for (const length of [N, N + 577]) {
      const signal = material('quiet', length);
      const decoded = decodeMp3(
        encodeMp3Channels([signal], RATE, { bitrate: 192, vbr: true }),
      ).channelData[0]!;
      assert.equal(decoded.length, length, `${length}-sample input came back as ${decoded.length}`);
    }
  });

  test('variable bitrate without the model is refused rather than ignored', () => {
    assert.throws(
      () => encodeMp3Channels([material('quiet')], RATE, { vbr: true, psychoacoustic: false }),
      /psychoacoustic|variable bitrate/i,
    );
  });

  test('the free-format and reserved bitrate slots are not offered as choices', () => {
    // The header field is 4 bits with two sentinels: 0 means the size is not
    // declared, 15 is reserved. Treating the raw table as a list of bitrates
    // accepts 0 as valid and reads -1 as the maximum.
    assert.ok(!MPEG1_LAYER3_BITRATE_VALUES.includes(0));
    assert.ok(!MPEG1_LAYER3_BITRATE_VALUES.includes(-1));
    assert.equal(MPEG1_LAYER3_BITRATE_VALUES.length, 14);
    assert.equal(Math.max(...MPEG1_LAYER3_BITRATE_VALUES), 320);

    const signal = new Float32Array(1152);
    assert.throws(() => encodeMp3Channels([signal], RATE, { bitrate: 0 }), /not an MPEG-1/);
    assert.throws(() => encodeMp3Channels([signal], RATE, { bitrate: -1 }), /not an MPEG-1/);
  });
});
