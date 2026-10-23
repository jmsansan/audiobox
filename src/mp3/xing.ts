/**
 * The Xing/Info header, and why an MP3 needs one.
 *
 * A Layer III stream has no container and no header describing the whole file.
 * Duration has to be guessed from the bitrate, seeking has to be guessed from
 * byte position, and — the part that is actually audible — there is no way to
 * say where the audio really starts and stops.
 *
 * That last one matters because MP3 cannot represent an arbitrary number of
 * samples. Frames hold exactly 1152 each, so any real recording gets padded to
 * the next frame boundary, and every encoder's filterbank prepends silence
 * before the first sample emerges. Decoded naively, a file gains silence at both
 * ends. Play two such files back to back and the join is audible: this is the
 * gap in "gapless".
 *
 * The fix is a convention rather than a standard: a first frame that codes no
 * audio and instead carries a tag with the frame count, the byte count, and the
 * exact number of samples to drop from each end. Decoders that recognise it skip
 * the frame entirely; decoders that do not simply decode 1152 samples of silence
 * and are none the worse.
 *
 * ## Naming the encoder honestly
 *
 * The delay and padding fields live in the "LAME tag" extension, and decoders
 * gate on the 9-byte encoder string that precedes them — many accept only
 * `LAME`, `Lavc` or `Lavf`. Writing one of those would buy wider gapless support
 * by claiming to be software this is not, so this writes `audiobox`. The cost is
 * real and worth stating plainly: decoders that check the name against a
 * whitelist will ignore our delay and padding and decode a fraction of a second
 * of extra silence at each end. The audio itself is unaffected, and audiobox
 * reads its own tag, so `decode(encode(x))` is sample-exact.
 */

import { BitWriter } from '../io/bits.ts';
import { EncodeError } from '../errors.ts';
import { writeFrameHeader } from './bitstream.ts';
import {
  mpeg1FrameLength,
  mpeg1SideInfoSize,
  MPEG1_LAYER3_BITRATES,
  MPEG1_LAYER3_BITRATE_VALUES,
  MPEG1_SAMPLE_RATES,
} from './frame.ts';
import type { ChannelMode } from './frame.ts';

/** The 9-byte encoder string. Exactly 9 bytes, space-padded. */
export const ENCODER_STRING = 'audiobox ';

/** Both delay and padding are 12-bit fields. */
const MAX_DELAY_FIELD = 4095;

export interface InfoTagParameters {
  bitrateKbps: number;
  sampleRate: number;
  channels: number;
  mode: ChannelMode;
  modeExtension: number;
  /** Audio frames that follow, not counting this one. */
  frameCount: number;
  /** Total stream length in bytes, including this frame. */
  streamBytes: number;
  /** Samples of priming silence before the first real sample. */
  encoderDelay: number;
  /** Samples of silence after the last real sample. */
  encoderPadding: number;
  /** Variable bitrate: tag as `Xing` and carry a seek table. */
  vbr?: boolean;
  /** The assembled audio frames, for the music CRC and the VBR seek table. */
  frames: Uint8Array;
  /** Size of this tag frame, since it precedes `frames` in the stream. */
  tagBytes?: number;
}

/**
 * Builds the complete leading frame, header and tag together: `Info` for a
 * constant-bitrate stream, `Xing` with a seek table for a variable-bitrate one.
 */
export function writeInfoTag(params: InfoTagParameters): Uint8Array {
  const { bitrateKbps, sampleRate, channels, encoderDelay, encoderPadding } = params;

  if (encoderDelay < 0 || encoderDelay > MAX_DELAY_FIELD) {
    throw new EncodeError(`Encoder delay ${encoderDelay} does not fit the tag's 12-bit field.`);
  }
  if (encoderPadding < 0 || encoderPadding > MAX_DELAY_FIELD) {
    throw new EncodeError(`Encoder padding ${encoderPadding} does not fit the tag's 12-bit field.`);
  }

  const frameBytes = infoTagFrameLength(bitrateKbps, sampleRate, channels, params.vbr);
  const headerBytes = 4 + mpeg1SideInfoSize(channels);
  const frame = new Uint8Array(frameBytes);

  const header = new BitWriter(64);
  writeFrameHeader(header, { ...params, bitrateKbps: tagBitrate(params), padding: false });
  frame.set(header.finish(), 0);
  // Side information stays zero: no granule claims any bits, so a decoder that
  // does not recognise the tag decodes silence rather than noise.

  let p = headerBytes;

  // `Info` and `Xing` are the same structure; the name is how a decoder tells
  // constant from variable bitrate without scanning the whole file.
  const seekTable = params.vbr ? buildSeekTable(params) : null;

  p = writeAscii(frame, p, params.vbr ? 'Xing' : 'Info');
  p = writeU32(frame, p, seekTable ? 0x0007 : 0x0003);
  p = writeU32(frame, p, params.frameCount);
  p = writeU32(frame, p, params.streamBytes);
  if (seekTable) {
    frame.set(seekTable, p);
    p += 100;
  }

  // --- LAME tag extension ---------------------------------------------------
  // Lowpass, replay gain and the remaining descriptors stay zero, which readers
  // take as "not reported".
  const lameStart = p;
  p = writeAscii(frame, p, ENCODER_STRING);

  // Bytes 9-20 of the extension: VBR method, lowpass, replay gain, flags.
  // Byte 9's low nibble is the VBR method: 1 constant, 4 variable bitrate.
  frame[lameStart + 9] = params.vbr ? 0x04 : 0x01;
  // Byte 20 is the bitrate for CBR, and for VBR the lowest one a frame may take.
  const lowestKbps = params.vbr ? MPEG1_LAYER3_BITRATE_VALUES[0]! : bitrateKbps;
  frame[lameStart + 20] = Math.min(255, lowestKbps);

  // Bytes 21-23: 12 bits of delay followed by 12 bits of padding.
  frame[lameStart + 21] = (encoderDelay >> 4) & 0xff;
  frame[lameStart + 22] = ((encoderDelay & 0x0f) << 4) | ((encoderPadding >> 8) & 0x0f);
  frame[lameStart + 23] = encoderPadding & 0xff;

  // Byte 24 holds the stereo mode in bits 2-4 and the source rate in bits 6-7.
  // Left zero, it would claim a mono source at 32 kHz or below.
  const stereoMode = ['mono', 'stereo', 'dual', 'joint'].indexOf(params.mode);
  const sourceRate = [32000, 44100, 48000].indexOf(sampleRate);
  frame[lameStart + 24] = (stereoMode << 2) | (sourceRate << 6);

  // Bytes 28-31: stream length from this frame on, as LAME and ffmpeg count it.
  writeU32(frame, lameStart + 28, params.streamBytes);
  // Bytes 32-35: CRC of the audio frames, then CRC of this frame up to that field.
  writeU16(frame, lameStart + 32, lameCrc(params.frames));
  writeU16(frame, lameStart + 34, lameCrc(frame.subarray(0, lameStart + 34)));

  return frame;
}

/**
 * The 100-entry seek table: where the file is, in 256ths, at each percent of it.
 *
 * Without one, seeking a variable-bitrate file means guessing that byte position
 * is proportional to time — which is exactly the assumption variable bitrate
 * breaks. Players that guess land in the wrong place and have to scan.
 *
 * Entry `i` is the byte offset at `i`% of the duration, scaled to 0–255 across
 * the whole file. The resolution is coarse by construction; it is enough to get
 * within a frame or two, after which a decoder resynchronises on the next sync
 * word anyway.
 */
function buildSeekTable(params: InfoTagParameters): Uint8Array | null {
  const audio = params.frames;
  const tagBytes = params.tagBytes ?? 0;
  if (params.frameCount <= 0) return null;

  // Walk the frames once, recording where each begins.
  const offsets = new Int32Array(params.frameCount + 1);
  let offset = 0;
  let count = 0;
  while (count < params.frameCount && offset + 4 <= audio.byteLength) {
    offsets[count++] = tagBytes + offset;
    const length = frameLengthAt(audio, offset);
    if (length <= 0) break;
    offset += length;
  }
  if (count < params.frameCount) return null;
  offsets[count] = tagBytes + offset;

  const total = params.streamBytes || tagBytes + audio.byteLength;
  const table = new Uint8Array(100);
  for (let i = 0; i < 100; i++) {
    const frame = Math.min(count - 1, Math.floor((i * count) / 100));
    table[i] = Math.min(255, Math.round((offsets[frame]! * 256) / total));
  }
  return table;
}

/** Frame length from a header at `offset`, or 0 if it is not one. */
function frameLengthAt(data: Uint8Array, offset: number): number {
  if (offset + 4 > data.byteLength) return 0;
  if (data[offset] !== 0xff || (data[offset + 1]! & 0xe0) !== 0xe0) return 0;

  const bitrateIndex = data[offset + 2]! >> 4;
  const rateIndex = (data[offset + 2]! >> 2) & 0x03;
  const padding = (data[offset + 2]! >> 1) & 1;
  if (bitrateIndex === 0 || bitrateIndex === 15 || rateIndex === 3) return 0;

  const kbps = MPEG1_LAYER3_BITRATES[bitrateIndex]!;
  const rate = MPEG1_SAMPLE_RATES[rateIndex]!;
  return mpeg1FrameLength(kbps, rate, padding === 1);
}

/**
 * Bitrate for the tag frame, which need not match the audio around it.
 *
 * The tag has to fit, and at low bitrates a frame is smaller than the tag —
 * 96 bytes at 32 kbps against 152 for a tag carrying a seek table. Since the
 * frame codes no audio, its bitrate field only decides its size, so it is chosen
 * for that rather than inherited. Decoders skip the frame either way.
 */
function tagBitrate(params: {
  bitrateKbps: number;
  sampleRate: number;
  channels: number;
  vbr?: boolean;
}): number {
  const needed = 4 + mpeg1SideInfoSize(params.channels) + 52 + (params.vbr ? 100 : 0);

  for (const kbps of MPEG1_LAYER3_BITRATES) {
    if (kbps < params.bitrateKbps) continue;
    if (mpeg1FrameLength(kbps, params.sampleRate, false) >= needed) return kbps;
  }
  for (const kbps of MPEG1_LAYER3_BITRATES) {
    if (mpeg1FrameLength(kbps, params.sampleRate, false) >= needed) return kbps;
  }

  throw new EncodeError(`No MPEG-1 frame at ${params.sampleRate} Hz can hold the tag.`);
}

/** Bytes the tag frame occupies, so the caller can size the stream first. */
export function infoTagFrameLength(
  bitrateKbps: number,
  sampleRate: number,
  channels: number,
  vbr?: boolean,
): number {
  return mpeg1FrameLength(tagBitrate({ bitrateKbps, sampleRate, channels, vbr }), sampleRate, false);
}

function writeAscii(target: Uint8Array, offset: number, text: string): number {
  for (let i = 0; i < text.length; i++) target[offset + i] = text.charCodeAt(i) & 0x7f;
  return offset + text.length;
}

/** CRC-16 with the reflected polynomial 0xA001, as LAME and ffmpeg compute both tag CRCs. */
export function lameCrc(data: Uint8Array): number {
  let crc = 0;
  for (let i = 0; i < data.length; i++) {
    crc ^= data[i]!;
    for (let bit = 0; bit < 8; bit++) crc = crc & 1 ? (crc >>> 1) ^ 0xa001 : crc >>> 1;
  }
  return crc;
}

function writeU16(target: Uint8Array, offset: number, value: number): number {
  target[offset] = (value >>> 8) & 0xff;
  target[offset + 1] = value & 0xff;
  return offset + 2;
}

function writeU32(target: Uint8Array, offset: number, value: number): number {
  target[offset] = (value >>> 24) & 0xff;
  target[offset + 1] = (value >>> 16) & 0xff;
  target[offset + 2] = (value >>> 8) & 0xff;
  target[offset + 3] = value & 0xff;
  return offset + 4;
}
