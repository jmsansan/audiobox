/**
 * MP3 encoding — MPEG-1 Layer III, constant bitrate, long blocks.
 *
 * The frame loop. Everything it needs already exists in its own module:
 * {@link analyseGranule} produces the spectrum, {@link quantizeGranule} decides
 * what survives the bit budget, and {@link FrameAssembler} handles the reservoir.
 * This is the part that runs them in the right order and divides the bits.
 *
 * ## Rate control across a frame
 *
 * A frame's slot is fixed by the bitrate, but its four granule-channels do not
 * need equal shares. Bits are handed out one unit at a time from a running pool,
 * so a granule that finishes under budget leaves the remainder to the ones after
 * it — within the frame, immediately, and across frames through the reservoir.
 *
 * The pool starts at the frame's own slot plus a bounded draw on the reservoir.
 * Bounded, because a granule that empties the reservoir leaves the next hard
 * passage with nothing to borrow; capping the draw at half a frame keeps a
 * buffer for the frames that need it more than this one does.
 *
 * ## What this does not do yet
 *
 * No joint stereo and no variable bitrate: the two channels are coded
 * independently, and every frame is the same size. Both cost bitrate rather than
 * correctness, and both are additions rather than changes to what is here.
 *
 * The output is valid MPEG-1 Layer III that any decoder reads. The README
 * publishes measured quality figures rather than adjectives.
 */

import { BitWriter } from '../io/bits.ts';
import { EncodeError, InvalidArgumentError } from '../errors.ts';
import { AnalysisFilterbank, analyseGranule } from './analysis.ts';
import { FrameAssembler, writeScalefactors, MAX_MAIN_DATA_BEGIN } from './bitstream.ts';
import {
  createWorkspace,
  quantizeGranule,
  writeGranuleData,
  MAX_PART2_3_LENGTH,
} from './quantize.ts';
import {
  MPEG1_LAYER3_BITRATE_VALUES,
  MPEG1_SAMPLE_RATES,
  mpeg1FrameLength,
  mpeg1SideInfoSize,
} from './frame.ts';
import { writeInfoTag, infoTagFrameLength } from './xing.ts';
import {
  PsychoacousticModel,
  detectAttacks,
  sequenceBlockTypes,
} from './psychoacoustic.ts';
import type { Audio } from '../audio.ts';
import type { GranuleCoding } from './quantize.ts';
import type { GranuleInfo, SideInfo } from './sideinfo.ts';
import type { Scalefactors } from './huffman.ts';

const GRANULE = 576;
const FRAME_SAMPLES = GRANULE * 2;

/**
 * Samples of latency between an encoder input sample and the corresponding
 * decoder output sample.
 *
 * Measured rather than derived: a windowed burst surrounded by silence, encoded
 * and decoded, moves its energy centroid by exactly this much. `test/mp3.test.ts`
 * repeats that measurement and fails if it drifts, because the figure is what
 * gapless playback subtracts and an error here is an audible offset.
 *
 * It comes out at the filterbank pair's own delay — the MDCT's granule of
 * lookahead is absorbed by feeding the transform the *previous* granule as the
 * first half of its window rather than delaying the input.
 */
export const ENCODER_DELAY = 528;

/**
 * Extra silence appended so the tail of the input reaches the output.
 *
 * Anything still inside the filterbank's delay line or the MDCT's overlap when
 * the input ends needs further granules to push it out.
 */
const FLUSH_SAMPLES = ENCODER_DELAY + GRANULE;

/**
 * Reservoir bytes kept in hand rather than spent on the current frame.
 *
 * Two thirds of what `main_data_begin` can reach back over, so a dense frame
 * still has somewhere to borrow from while the reservoir stays topped up.
 */
const RESERVOIR_WORKING_BYTES = 340;

export interface Mp3EncodeOptions {
  /**
   * Constant bitrate in kbit/s. One of 32, 40, 48, 56, 64, 80, 96, 112, 128,
   * 160, 192, 224, 256, 320.
   *
   * The default is 192 rather than the more familiar 128 because this encoder
   * has no psychoacoustic model yet: it spends bits evenly rather than where
   * they are least audible, and 192 is where that stops being noticeable.
   */
  bitrate?: number;
  /**
   * Write the leading Xing/Info frame carrying frame count, duration, and the
   * priming and padding figures gapless playback needs. On by default.
   *
   * Turn it off only when the output is a fragment that will be concatenated
   * into a larger stream, where a tag partway through would misreport the whole.
   */
  tag?: boolean;
  /**
   * Shape quantisation noise using the psychoacoustic model. On by default.
   *
   * Turning it off quantises flat — noise spread evenly across the spectrum
   * rather than hidden under the signal. Faster, and markedly worse below about
   * 256 kbps; useful mainly for measuring what the model is worth.
   */
  psychoacoustic?: boolean;
  /**
   * Switch to short blocks on transients. On by default.
   *
   * A long block spreads its quantisation noise over 24 ms, so noise from a
   * drum hit is audible before the hit itself — pre-echo. Short blocks confine
   * it to 4 ms at the cost of coding efficiency, so they are used only where an
   * attack is detected.
   */
  windowSwitching?: boolean;
  /**
   * Let each frame take the bitrate its content needs, instead of giving every
   * frame the same size.
   *
   * A constant-bitrate stream spends the same bytes on a cymbal crash and on a
   * held note, which means overspending on most of a track to afford the hardest
   * moments of it. Variable bitrate spends what transparency costs and stops.
   * `bitrate` then acts as a ceiling rather than a target.
   *
   * Requires the psychoacoustic model, which is what defines "enough".
   */
  vbr?: boolean;
  /**
   * Variable-bitrate quality, 0 (best) to 9 (smallest). Default 4.
   *
   * Scales the masking thresholds: a lower number demands the noise sit further
   * below what masks it, which costs bits. On dense material the range runs from
   * roughly 240 kbit/s at 0 to 90 at 9, but the whole point is that the figure
   * follows the material rather than the setting.
   *
   * Ignored unless `vbr` is on.
   */
  quality?: number;
  /**
   * How many passes the outer loop may take at shaping noise. Higher is slower
   * and slightly better. Ignored when `psychoacoustic` is off.
   */
  effort?: number;
}

/** One granule-channel's coded result, copied out of the shared workspace. */
interface CodedUnit {
  values: Int32Array;
  scalefactors: Scalefactors;
  granule: GranuleInfo;
  coding: GranuleCoding;
}

/**
 * Encodes an {@link Audio} to an MPEG-1 Layer III stream.
 *
 * ```ts
 * const mp3 = encodeMp3(audio, { bitrate: 192 });
 * ```
 *
 * MPEG-1 only, so the source must already be 32, 44.1 or 48 kHz and mono or
 * stereo. Resample or downmix first — silently changing the sample rate of
 * someone's audio is a worse outcome than refusing to.
 */
export function encodeMp3(audio: Audio, options: Mp3EncodeOptions = {}): Uint8Array {
  return encodeMp3Channels(audio.allChannels, audio.sampleRate, options);
}

/**
 * Encodes planar float PCM, for callers that have channels rather than an
 * `Audio` — a streaming source, or a worker that was handed raw buffers.
 *
 * @param channels One `Float32Array` per channel, nominally in [-1, 1], all the
 * same length. Mono and stereo only.
 * @param sampleRate 32000, 44100 or 48000 Hz.
 */
export function encodeMp3Channels(
  channels: readonly Float32Array[],
  sampleRate: number,
  options: Mp3EncodeOptions = {},
): Uint8Array {
  const channelCount = channels.length;
  if (channelCount !== 1 && channelCount !== 2) {
    throw new InvalidArgumentError(
      `MP3 encoding supports mono and stereo; got ${channelCount} channels.`,
    );
  }
  if (!MPEG1_SAMPLE_RATES.includes(sampleRate)) {
    throw new InvalidArgumentError(
      `${sampleRate} Hz cannot be encoded as MPEG-1 Layer III. ` +
        `Supported: ${MPEG1_SAMPLE_RATES.join(', ')} Hz. Resample first.`,
    );
  }

  const bitrateKbps = options.bitrate ?? 192;
  if (!MPEG1_LAYER3_BITRATE_VALUES.includes(bitrateKbps)) {
    throw new InvalidArgumentError(
      `${bitrateKbps} kbps is not an MPEG-1 Layer III bitrate. ` +
        `Supported: ${MPEG1_LAYER3_BITRATE_VALUES.join(', ')}.`,
    );
  }

  const length = channels[0]!.length;
  for (const channel of channels) {
    if (channel.length !== length) {
      throw new InvalidArgumentError('All channels must be the same length.');
    }
  }

  const frameCount = Math.max(1, Math.ceil((length + FLUSH_SAMPLES) / FRAME_SAMPLES));
  const mode = channelCount === 1 ? 'mono' : 'stereo';

  const assembler = new FrameAssembler({
    bitrateKbps,
    sampleRate,
    channels: channelCount,
    mode,
    modeExtension: 0,
  });

  const shaping = options.psychoacoustic !== false;
  const maxOuterIterations = Math.max(1, Math.round(options.effort ?? 6));

  if (options.vbr && !shaping) {
    throw new InvalidArgumentError(
      'Variable bitrate needs the psychoacoustic model to decide what "enough" means. ' +
        'Enable psychoacoustic, or use a constant bitrate.',
    );
  }
  const vbr = Boolean(options.vbr);

  // How far below the masking threshold the noise is asked to sit, in dB.
  //
  // The model's thresholds are a coarse estimate, and taken at face value they
  // call a granule transparent while it still measures 16 dB SNR — which is why
  // quality-targeted encoding needs a margin and constant-bitrate encoding, which
  // only ever used the *shape* of the same curve, did not. Beyond about 24 dB the
  // granule hits the format's per-granule bit ceiling and the setting stops
  // doing anything.
  const quality = Math.max(0, Math.min(9, Math.round(options.quality ?? 4)));
  const thresholdScale = vbr ? 10 ** (-(18 - 2 * quality) / 10) : 1;
  const switching = options.windowSwitching !== false;

  // Block types are decided for the whole input up front, because a short block
  // needs the granule before it to be a start block — one granule of lookahead
  // that the frame loop cannot supply while it is already coding.
  const granuleCount = frameCount * 2;
  const blockTypes = channels.map(() => new Uint8Array(granuleCount));
  if (switching) {
    const attacks = new Uint8Array(granuleCount);
    for (let channel = 0; channel < channelCount; channel++) {
      detectAttacks(channels[channel]!, granuleCount, attacks);
      sequenceBlockTypes(attacks, blockTypes[channel]!);
    }
  }

  const filterbanks = channels.map(() => new AnalysisFilterbank());
  const overlaps = channels.map(() => new Float32Array(GRANULE));
  const models = shaping ? channels.map(() => new PsychoacousticModel(sampleRate)) : null;
  const thresholds = new Float32Array(22);
  const workspace = createWorkspace();
  const spectrum = new Float32Array(GRANULE);
  const subbandScratch = new Float32Array(GRANULE);
  const slotScratch = new Float32Array(32);

  const unitCount = 2 * channelCount;
  const units: CodedUnit[] = [];
  for (let i = 0; i < unitCount; i++) units.push(createUnit());

  const writer = new BitWriter(4096);
  const emptyScfsi = [0, 0, 0, 0];

  for (let frame = 0; frame < frameCount; frame++) {
    // In variable-bitrate mode the granules are coded first and the frame is
    // sized around them afterwards. They are budgeted against `bitrate` — the
    // stated ceiling — so that the frame chosen below can always hold what they
    // produced. Budgeting against the format maximum instead lets a granule
    // exceed what any allowed frame can carry.
    let frameBitrate = bitrateKbps;

    // One byte held back: the frame's main data is byte-aligned, so the last
    // partial byte would otherwise be able to push it past the budget.
    let pool = assembler.nextBudgetBytes(vbr ? frameBitrate : undefined) * 8 - 8;
    let unitsLeft = unitCount;
    let index = 0;

    for (let granuleIndex = 0; granuleIndex < 2; granuleIndex++) {
      for (let channel = 0; channel < channelCount; channel++) {
        const startSample = frame * FRAME_SAMPLES + granuleIndex * GRANULE;
        const granuleNumber = frame * 2 + granuleIndex;
        const blockType = blockTypes[channel]![granuleNumber]!;

        analyseGranule(
          channels[channel]!,
          startSample,
          filterbanks[channel]!,
          overlaps[channel]!,
          subbandScratch,
          slotScratch,
          spectrum,
          blockType,
          sampleRate,
        );

        // The model reads the input directly rather than the spectrum: masking
        // is a property of what the listener hears, and the MDCT has already
        // folded two granules together by this point.
        models?.[channel]!.analyse(
          channels[channel]!,
          startSample,
          spectrum,
          thresholds,
          blockType === 2,
        );
        if (thresholdScale !== 1) {
          for (let i = 0; i < thresholds.length; i++) thresholds[i]! *= thresholdScale;
        }

        const budgetBits = Math.min(
          MAX_PART2_3_LENGTH,
          Math.max(0, Math.floor(pool / unitsLeft)),
        );
        const coding = quantizeGranule(
          {
            spectrum,
            sampleRate,
            budgetBits,
            thresholds: models ? thresholds : null,
            maxOuterIterations,
            blockType,
            minimiseBits: vbr,
          },
          workspace,
        );

        storeUnit(units[index]!, coding);
        pool -= coding.part2_3Length;
        unitsLeft--;
        index++;
      }
    }

    if (vbr) {
      // Now that the granules are coded, pick the smallest legal frame that
      // holds them. The reservoir absorbs the remainder, so this is a floor on
      // what the frame must carry rather than an exact fit.
      let needed = 0;
      for (const unit of units) needed += unit.granule.part2_3_length;
      frameBitrate = smallestBitrateFor(
        Math.ceil(needed / 8),
        sampleRate,
        channelCount,
        bitrateKbps,
        assembler.reservoirBytes,
      );
    }

    writer.reset();
    index = 0;
    for (let granuleIndex = 0; granuleIndex < 2; granuleIndex++) {
      for (let channel = 0; channel < channelCount; channel++) {
        const unit = units[index]!;
        const before = writer.bitLength;

        writeScalefactors(writer, unit.granule, unit.scalefactors, emptyScfsi, granuleIndex);
        writeGranuleData(writer, unit.coding, sampleRate);

        const written = writer.bitLength - before;
        if (written !== unit.granule.part2_3_length) {
          // The declared length is how the decoder finds the next granule, so a
          // mismatch corrupts everything after it. Catching it here rather than
          // shipping it is the point of counting bits twice.
          throw new EncodeError(
            `Granule ${granuleIndex} channel ${channel} wrote ${written} bits but ` +
              `declared ${unit.granule.part2_3_length}.`,
          );
        }
        index++;
      }
    }
    writer.align();

    assembler.addFrame(
      {
        mainDataBegin: 0,
        scfsi: Array.from({ length: channelCount }, () => [0, 0, 0, 0]),
        granules: [
          units.slice(0, channelCount).map((unit) => unit.granule),
          units.slice(channelCount, unitCount).map((unit) => unit.granule),
        ],
      },
      writer.finish(),
      vbr ? frameBitrate : undefined,
    );
  }

  const audio = assembler.finish(() => emptySideInfo(channelCount));
  if (options.tag === false) return audio;

  // Padding is whatever the frames hold beyond the real audio and the priming
  // silence in front of it. Deriving it from the frames actually emitted, rather
  // than from the frame count predicted earlier, keeps it right even though
  // draining the reservoir can add frames.
  const emittedFrames = assembler.frameCount;
  const tagBytes = infoTagFrameLength(bitrateKbps, sampleRate, channelCount, vbr);
  const encoderPadding = emittedFrames * FRAME_SAMPLES - length - ENCODER_DELAY;

  const tag = writeInfoTag({
    bitrateKbps,
    sampleRate,
    channels: channelCount,
    mode,
    modeExtension: 0,
    frameCount: emittedFrames,
    streamBytes: tagBytes + audio.byteLength,
    encoderDelay: ENCODER_DELAY,
    encoderPadding: Math.max(0, encoderPadding),
    vbr,
    frames: audio,
    tagBytes,
  });

  const out = new Uint8Array(tag.byteLength + audio.byteLength);
  out.set(tag, 0);
  out.set(audio, tag.byteLength);
  return out;
}

/**
 * Smallest legal bitrate whose frame holds `needed` bytes of main data.
 *
 * The reservoir counts towards it: a frame may borrow from the space earlier
 * frames left unused, so the bitrate only has to cover what is not already
 * banked. Capped at `ceiling` so a stated maximum is honoured even when the
 * material would rather have more.
 */
function smallestBitrateFor(
  needed: number,
  sampleRate: number,
  channels: number,
  ceiling: number,
  banked: number,
): number {
  const header = 4 + mpeg1SideInfoSize(channels);

  // Only the reservoir's *surplus* counts towards this frame. Subtracting all of
  // it makes every frame look nearly free, so every frame takes the smallest
  // bitrate, so the reservoir never refills — a runaway to 32 kbit/s whatever
  // the material. The reservoir is there to absorb variation between frames, not
  // to pay for them, so a working balance is held back.
  const surplus = Math.max(0, Math.min(banked, MAX_MAIN_DATA_BEGIN) - RESERVOIR_WORKING_BYTES);
  const available = Math.max(0, needed - surplus);

  for (const kbps of MPEG1_LAYER3_BITRATE_VALUES) {
    if (kbps > ceiling) break;
    const slot = mpeg1FrameLength(kbps, sampleRate, false) - header;
    if (slot >= available) return kbps;
  }
  return ceiling;
}

function createUnit(): CodedUnit {
  const values = new Int32Array(GRANULE);
  const long = new Int32Array(23);
  return {
    values,
    scalefactors: {
      long,
      short: [new Int32Array(14), new Int32Array(14), new Int32Array(14)],
    },
    granule: emptyGranuleInfo(),
    coding: {
      values,
      globalGain: 210,
      scalefactors: long,
      scalefacCompress: 0,
      scalefacScale: 0.5,
      preflag: false,
      bigValues: 0,
      tableSelect: [0, 0, 0],
      region0Count: 0,
      region1Count: 0,
      count1TableSelect: 0,
      count1End: 0,
      scalefactorBits: 0,
      huffmanBits: 0,
      part2_3Length: 0,
      blockType: 0,
      subblockGain: new Int32Array(3),
    },
  };
}

/**
 * Copies a quantiser result into a unit.
 *
 * The quantiser writes into shared scratch buffers, which the next granule
 * overwrites — so what the frame writer needs has to be taken out before then.
 */
function storeUnit(unit: CodedUnit, coding: GranuleCoding): void {
  unit.values.set(coding.values);

  const short = coding.blockType === 2;
  if (short) {
    // The quantiser keeps scalefactors in one flat array indexed window-major;
    // the bitstream writer wants them per window.
    unit.scalefactors.long.fill(0);
    for (let w = 0; w < 3; w++) {
      for (let band = 0; band < 13; band++) {
        unit.scalefactors.short[w]![band] = coding.scalefactors[w * 13 + band]!;
      }
    }
  } else {
    unit.scalefactors.long.set(coding.scalefactors.subarray(0, 23));
    for (const perWindow of unit.scalefactors.short) perWindow.fill(0);
  }

  unit.coding = { ...coding, values: unit.values };

  const granule = unit.granule;
  granule.part2_3_length = coding.part2_3Length;
  granule.bigValues = coding.bigValues;
  granule.globalGain = coding.globalGain;
  granule.scalefacCompress = coding.scalefacCompress;
  // Types 1 and 3 are transition windows, and the format treats them as window
  // switching too — they carry only two table selects and no region counts.
  granule.windowSwitching = coding.blockType !== 0;
  granule.blockType = coding.blockType;
  granule.mixedBlock = false;
  granule.tableSelect = [...coding.tableSelect];
  granule.subblockGain = [
    coding.subblockGain[0]!,
    coding.subblockGain[1]!,
    coding.subblockGain[2]!,
  ];
  // Implied rather than transmitted when window switching is on, but the
  // decoder derives the same values, so they have to match here.
  granule.region0Count = granule.windowSwitching ? (short ? 8 : 7) : coding.region0Count;
  granule.region1Count = granule.windowSwitching ? 20 - granule.region0Count : coding.region1Count;
  granule.preflag = coding.preflag;
  granule.scalefacScale = coding.scalefacScale;
  granule.count1TableSelect = coding.count1TableSelect;
}

function emptyGranuleInfo(): GranuleInfo {
  return {
    part2_3_length: 0,
    bigValues: 0,
    globalGain: 210,
    scalefacCompress: 0,
    windowSwitching: false,
    blockType: 0,
    mixedBlock: false,
    tableSelect: [0, 0, 0],
    subblockGain: [0, 0, 0],
    region0Count: 0,
    region1Count: 0,
    preflag: false,
    scalefacScale: 0.5,
    count1TableSelect: 0,
  };
}

/** Side information for a frame that carries no audio, used to drain the reservoir. */
function emptySideInfo(channels: number): SideInfo {
  return {
    mainDataBegin: 0,
    scfsi: Array.from({ length: channels }, () => [0, 0, 0, 0]),
    granules: [
      Array.from({ length: channels }, emptyGranuleInfo),
      Array.from({ length: channels }, emptyGranuleInfo),
    ],
  };
}
