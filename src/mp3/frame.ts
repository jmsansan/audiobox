/**
 * MPEG audio frame headers, and the metadata tags that wrap an MP3 file.
 *
 * MP3 has no file-level structure: a `.mp3` is just a run of self-describing
 * frames, optionally bracketed by ID3 tags. Everything a decoder needs — bitrate,
 * sample rate, channel mode — is repeated in every 4-byte frame header, which is
 * what lets playback start from the middle of a stream.
 */

import { DecodeError } from '../errors.ts';

/** MPEG version, from the 2-bit version field. */
export type MpegVersion = 1 | 2 | 25;

/** Channel mode, from the 2-bit mode field. */
export type ChannelMode = 'stereo' | 'joint' | 'dual' | 'mono';

export interface FrameHeader {
  version: MpegVersion;
  layer: number;
  /** True when a 16-bit CRC follows the header. */
  hasCrc: boolean;
  bitrate: number;
  sampleRate: number;
  padding: boolean;
  mode: ChannelMode;
  /** Intensity/MS stereo flags, only meaningful in joint stereo. */
  modeExtension: number;
  channels: number;
  /** Total frame length in bytes, including the header. */
  frameLength: number;
  /** Samples produced per channel by this frame. */
  samplesPerFrame: number;
  /** Bytes of side information following the header (and CRC, if present). */
  sideInfoSize: number;
}

/** Bitrate tables in kbps, indexed by [versionGroup][layer][index]. */
const BITRATES_V1_L3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, -1];
const BITRATES_V2_L3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, -1];
const BITRATES_V1_L2 = [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, -1];
const BITRATES_V1_L1 = [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, -1];
const BITRATES_V2_L1 = [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, -1];

const SAMPLE_RATES: Record<MpegVersion, number[]> = {
  1: [44100, 48000, 32000],
  2: [22050, 24000, 16000],
  25: [11025, 12000, 8000],
};

/**
 * Bitrates the encoder can write, in kbps, indexed by the header's 4-bit field.
 *
 * Exported so the encoder writes headers from the same table the decoder reads
 * them with — the two disagreeing is exactly the kind of bug that produces files
 * only your own decoder can play.
 */
export const MPEG1_LAYER3_BITRATES: readonly number[] = BITRATES_V1_L3;

/**
 * The bitrates an MPEG-1 Layer III frame can actually be written at, ascending.
 *
 * {@link MPEG1_LAYER3_BITRATES} is indexed by the header's 4-bit field, so it
 * carries two sentinels: index 0 is "free format", where the size is not
 * declared at all, and index 15 is reserved. Neither is a bitrate, and treating
 * the raw table as a list of choices means accepting `0` as valid and reading
 * `-1` as the maximum.
 */
export const MPEG1_LAYER3_BITRATE_VALUES: readonly number[] = BITRATES_V1_L3.filter(
  (kbps) => kbps > 0,
);

/** Sample rates the encoder can write, indexed by the header's 2-bit field. */
export const MPEG1_SAMPLE_RATES: readonly number[] = SAMPLE_RATES[1];

/** Header field value for a bitrate in kbps, or -1 if it is not representable. */
export function bitrateIndex(kbps: number): number {
  return BITRATES_V1_L3.indexOf(kbps);
}

/** Header field value for a sample rate in Hz, or -1 if it is not representable. */
export function sampleRateIndex(hz: number): number {
  return SAMPLE_RATES[1].indexOf(hz);
}

/**
 * Frame length in bytes for MPEG-1 Layer III.
 *
 * 1152 samples at `bitrate` bits per second occupy a non-integer number of
 * bytes at most sample rates, so frames alternate between two sizes; the padding
 * bit says which. Getting this wrong desynchronises every following frame.
 */
export function mpeg1FrameLength(bitrateKbps: number, sampleRate: number, padding: boolean): number {
  return Math.floor((144 * bitrateKbps * 1000) / sampleRate) + (padding ? 1 : 0);
}

/** Side-information size in bytes for MPEG-1 Layer III. */
export function mpeg1SideInfoSize(channels: number): number {
  return channels === 1 ? 17 : 32;
}

/** Parses a 4-byte frame header. Returns `null` when the bytes are not a valid header. */
export function parseFrameHeader(data: Uint8Array, offset: number): FrameHeader | null {
  if (offset + 4 > data.byteLength) return null;

  const b0 = data[offset]!;
  const b1 = data[offset + 1]!;
  const b2 = data[offset + 2]!;
  const b3 = data[offset + 3]!;

  // 11-bit sync word.
  if (b0 !== 0xff || (b1 & 0xe0) !== 0xe0) return null;

  const versionBits = (b1 >> 3) & 0x03;
  if (versionBits === 1) return null; // reserved
  const version: MpegVersion = versionBits === 3 ? 1 : versionBits === 2 ? 2 : 25;

  const layerBits = (b1 >> 1) & 0x03;
  if (layerBits === 0) return null; // reserved
  const layer = 4 - layerBits;

  const hasCrc = (b1 & 0x01) === 0;

  const bitrateIndex = (b2 >> 4) & 0x0f;
  if (bitrateIndex === 0 || bitrateIndex === 0x0f) return null; // free-form or invalid

  const sampleRateIndex = (b2 >> 2) & 0x03;
  if (sampleRateIndex === 3) return null; // reserved

  const padding = ((b2 >> 1) & 0x01) === 1;

  const modeBits = (b3 >> 6) & 0x03;
  const mode: ChannelMode = (['stereo', 'joint', 'dual', 'mono'] as const)[modeBits]!;
  const modeExtension = (b3 >> 4) & 0x03;
  const channels = mode === 'mono' ? 1 : 2;

  let table: number[];
  if (layer === 1) table = version === 1 ? BITRATES_V1_L1 : BITRATES_V2_L1;
  else if (layer === 2) table = version === 1 ? BITRATES_V1_L2 : BITRATES_V2_L3;
  else table = version === 1 ? BITRATES_V1_L3 : BITRATES_V2_L3;

  const bitrate = table[bitrateIndex]! * 1000;
  if (bitrate <= 0) return null;

  const sampleRate = SAMPLE_RATES[version]![sampleRateIndex]!;

  // Layer I uses 384 samples per frame; Layer II always 1152; Layer III is 1152
  // for MPEG-1 but 576 for MPEG-2/2.5, which halves the frame length too.
  let samplesPerFrame: number;
  if (layer === 1) samplesPerFrame = 384;
  else if (layer === 2) samplesPerFrame = 1152;
  else samplesPerFrame = version === 1 ? 1152 : 576;

  const frameLength =
    layer === 1
      ? (Math.floor((12 * bitrate) / sampleRate) + (padding ? 1 : 0)) * 4
      : Math.floor((samplesPerFrame / 8) * bitrate / sampleRate) + (padding ? 1 : 0);

  if (frameLength < 4 || frameLength > 2880) return null;

  // Layer III side info: MPEG-1 uses 17 bytes mono / 32 stereo; MPEG-2 halves it.
  const sideInfoSize =
    layer === 3 ? (version === 1 ? (channels === 1 ? 17 : 32) : channels === 1 ? 9 : 17) : 0;

  return {
    version,
    layer,
    hasCrc,
    bitrate,
    sampleRate,
    padding,
    mode,
    modeExtension,
    channels,
    frameLength,
    samplesPerFrame,
    sideInfoSize,
  };
}

/**
 * Finds the next valid frame header at or after `from`.
 *
 * A single sync word is not proof of a frame — 0xFF bytes occur constantly in
 * audio data and in tags. The candidate is confirmed by checking that a second
 * valid header sits exactly `frameLength` bytes later, which makes a false
 * positive vanishingly unlikely.
 */
export function findFrame(
  data: Uint8Array,
  from: number,
  limit = data.byteLength,
): { offset: number; header: FrameHeader } | null {
  const end = Math.min(limit, data.byteLength) - 4;

  for (let i = Math.max(0, from); i <= end; i++) {
    if (data[i] !== 0xff) continue;
    const header = parseFrameHeader(data, i);
    if (!header) continue;

    const nextOffset = i + header.frameLength;
    if (nextOffset + 4 <= data.byteLength) {
      const next = parseFrameHeader(data, nextOffset);
      // Consecutive frames in one file always agree on version and rate.
      if (!next || next.version !== header.version || next.sampleRate !== header.sampleRate) {
        continue;
      }
    }
    return { offset: i, header };
  }

  return null;
}

/** Size of an ID3v2 tag at `offset`, or 0 if there isn't one. */
export function id3v2Size(data: Uint8Array, offset = 0): number {
  if (offset + 10 > data.byteLength) return 0;
  if (data[offset] !== 0x49 || data[offset + 1] !== 0x44 || data[offset + 2] !== 0x33) return 0;

  // Syncsafe integer: 7 bits per byte, so the size can never contain 0xFF and
  // be mistaken for a frame sync.
  const size =
    ((data[offset + 6]! & 0x7f) << 21) |
    ((data[offset + 7]! & 0x7f) << 14) |
    ((data[offset + 8]! & 0x7f) << 7) |
    (data[offset + 9]! & 0x7f);

  const footer = (data[offset + 5]! & 0x10) !== 0 ? 10 : 0;
  return 10 + size + footer;
}

/** Information carried by a Xing/Info/VBRI header in the first frame. */
export interface VbrInfo {
  /** Total frames in the file, when declared. */
  frames?: number;
  /** Total bytes of audio data, when declared. */
  bytes?: number;
  /** Samples to drop from the start, from the LAME tag. */
  encoderDelay?: number;
  /** Samples to drop from the end, from the LAME tag. */
  encoderPadding?: number;
  /** True for VBR (`Xing`), false for CBR (`Info`). */
  vbr: boolean;
}

/**
 * Reads a Xing, Info, or VBRI header from the first frame.
 *
 * These sit inside a normal (silent) frame, in the space the bit reservoir would
 * otherwise use. They matter for two reasons: they give an accurate duration for
 * a VBR file without scanning it, and the LAME extension records the encoder
 * delay and padding needed for gapless playback.
 */
export function readVbrHeader(
  data: Uint8Array,
  frameOffset: number,
  header: FrameHeader,
): VbrInfo | null {
  // Xing/Info sits immediately after the side information.
  const xingOffset = frameOffset + 4 + (header.hasCrc ? 2 : 0) + header.sideInfoSize;
  if (xingOffset + 12 > data.byteLength) return null;

  const tag = String.fromCharCode(
    data[xingOffset]!,
    data[xingOffset + 1]!,
    data[xingOffset + 2]!,
    data[xingOffset + 3]!,
  );

  if (tag === 'Xing' || tag === 'Info') {
    const flags = readU32(data, xingOffset + 4);
    let p = xingOffset + 8;
    const info: VbrInfo = { vbr: tag === 'Xing' };

    if (flags & 0x01) {
      info.frames = readU32(data, p);
      p += 4;
    }
    if (flags & 0x02) {
      info.bytes = readU32(data, p);
      p += 4;
    }
    if (flags & 0x04) p += 100; // seek table
    if (flags & 0x08) p += 4; // quality indicator

    // The LAME tag follows: 9 bytes of encoder name, then a fixed layout in
    // which bytes 21-23 hold 12-bit delay and 12-bit padding.
    //
    // Any printable name is accepted, not just the three that dominate. Gating
    // on a whitelist is what pushes other encoders into labelling themselves
    // "LAME", and it would make audiobox unable to read its own tag. The risk it
    // guarded against — reading delay and padding out of bytes that are not a
    // LAME tag at all — is handled instead by checking the values: a real
    // encoder delay is a filterbank's worth of samples, not thousands.
    if (p + 24 <= data.byteLength) {
      const named = isEncoderName(data, p);
      const delayOffset = p + 21;

      if (named && delayOffset + 3 <= data.byteLength) {
        const delay = (data[delayOffset]! << 4) | (data[delayOffset + 1]! >> 4);
        const padding = ((data[delayOffset + 1]! & 0x0f) << 8) | data[delayOffset + 2]!;

        // A frame is 1152 samples; priming beyond about two of them means these
        // bytes are something else. Padding cannot exceed a frame either, since
        // more than that would have been one fewer frame.
        if (delay <= 2 * 1152 && padding <= 3 * 1152) {
          info.encoderDelay = delay;
          info.encoderPadding = padding;
        }
      }
    }
    return info;
  }

  // VBRI (Fraunhofer) always sits 32 bytes past the header.
  const vbriOffset = frameOffset + 36;
  if (vbriOffset + 26 <= data.byteLength) {
    const vbriTag = String.fromCharCode(...data.subarray(vbriOffset, vbriOffset + 4));
    if (vbriTag === 'VBRI') {
      return {
        vbr: true,
        bytes: readU32(data, vbriOffset + 10),
        frames: readU32(data, vbriOffset + 14),
      };
    }
  }

  return null;
}

/** True for a 9-byte encoder name: printable ASCII, NUL-padded as LAME pads short names. */
function isEncoderName(data: Uint8Array, offset: number): boolean {
  let length = 9;
  while (length > 0 && data[offset + length - 1] === 0) length--;
  for (let i = 0; i < length; i++) {
    const byte = data[offset + i]!;
    if (byte < 0x20 || byte > 0x7e) return false;
  }
  return length > 0;
}

function readU32(data: Uint8Array, offset: number): number {
  if (offset + 4 > data.byteLength) {
    throw new DecodeError(`Truncated 32-bit field at offset ${offset}.`, { offset });
  }
  return (
    ((data[offset]! << 24) | (data[offset + 1]! << 16) | (data[offset + 2]! << 8) | data[offset + 3]!) >>>
    0
  );
}
