# audiobox

**Audio decoding, editing and encoding for TypeScript and JavaScript — with zero dependencies.**

One API that reads and writes MP3, WAV, FLAC, AIFF, CAF and AU, and edits and converts between
them. The same code runs unchanged in the browser, Node, Deno, Bun and edge runtimes. No native
build step, no WebAssembly blob, nothing installed alongside it.

[![npm](https://img.shields.io/npm/v/audiobox.svg)](https://www.npmjs.com/package/audiobox)
[![core size](https://img.shields.io/badge/core-22.9%20kB%20gzip-blue.svg)](#bundle-size)
[![dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](#zero-dependencies-really)
[![types](https://img.shields.io/badge/types-included-blue.svg)](#typescript)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

```bash
npm install audiobox
```

```ts
import { decode, encode } from 'audiobox';

const audio = await decode(await file.arrayBuffer());   // format detected from content

const clip = audio
  .cut({ from: '0:12', to: '1:30' })   // keep 78 seconds
  .toMono()                            // downmix
  .resample(16_000)                    // 16 kHz for speech-to-text
  .normalize({ to: -1 });              // peak to -1 dBFS

const wav = await encode(clip, 'wav', { bitDepth: 16 });
```

---

## Contents

- [What it does](#what-it-does) · [Format support](#format-support)
- [Recipes](#recipes) — the things people actually build
- [API](#api)
- [Correctness](#correctness) · [Security](#security) · [Bundle size](#bundle-size)
- [Runtime support](#runtime-support) · [Roadmap](#roadmap)
- [Support and sponsorship](#support-and-sponsorship)

---

## What it does

**Read** — MP3, WAV (including RF64/BW64, `WAVE_FORMAT_EXTENSIBLE`, A-law, µ-law, IMA ADPCM),
FLAC, AIFF/AIFF-C, CAF, AU.

**Write** — WAV, FLAC, AIFF/AIFF-C, CAF, AU.

**Edit** — cut, trim, concatenate, pad, reverse, mix, crossfade.

**Convert** — sample rate, bit depth, channel layout, sample format.

**Process** — gain, fades, normalisation (peak / true-peak / **LUFS**), biquad filters, lookahead
limiter, silence trimming, DC-offset removal, time-stretch and pitch-shift.

**Measure** — peak, true peak, RMS, EBU R128 loudness, waveform peaks, FFT and spectrogram.

### Format support

| Format | Read | Write | Notes |
| --- | :---: | :---: | --- |
| **MP3** | ✅ | ✅ | Read: MPEG-1/2/2.5 · 8–48 kHz · CBR + VBR · joint stereo · gapless. Write: MPEG-1 · 32/44.1/48 kHz · 32–320 kbit/s · CBR + VBR |
| **WAV** / RF64 / BW64 | ✅ | ✅ | PCM 8/16/24/32, float 32/64, A-law, µ-law; IMA ADPCM read |
| **FLAC** | ✅ | ✅ | Lossless |
| **AIFF** / AIFF-C | ✅ | ✅ | Including `sowt` (little-endian PCM), `fl32`, `fl64` |
| **CAF** | ✅ | ✅ | 64-bit sizes, for files over 4 GB |
| **AU** / SND | ✅ | ✅ | Sun/NeXT |
| Ogg, MP4/M4A, WebM | — | — | Detected and named in the error, not decoded |

MP3 decoding covers every sample rate from 8 to 48 kHz, constant and variable bitrate, mono,
stereo, dual-channel and joint stereo (both mid/side and intensity), and honours Xing/VBRI/LAME
tags so playback is gapless — the encoder's priming and padding are removed rather than showing up
as a click at each end.

There is also `probeMp3()`, which reads duration, bitrate, channel mode and tags **without**
decoding. It is near-instant even on a large file and never allocates a decode buffer, which makes
it the right tool for validating an upload.

**MP3 encoding** writes MPEG-1 Layer III at 32–320 kbit/s, mono or stereo, at 32 / 44.1 / 48 kHz,
constant or variable bitrate. It has a psychoacoustic model, switches to short blocks on transients
to control pre-echo, uses the bit reservoir, and writes a Xing/Info tag so `decode(encode(x))`
returns exactly the samples it was given.

Measured, on this machine, at 44.1 kHz:

| | |
| --- | --- |
| ffmpeg's decode of our output, against the source | **73 dB** on tonal material |
| Noise shaping, on dense material at 128 kbit/s | noise-to-mask **−6.7 → −10.5 dB**; audible bands 49% → 20% |
| Short blocks, on transients | pre-echo **−27 → −34 dB** |
| VBR against CBR at matched quality | **15–25% smaller** on easy material, no saving on noise |
| Throughput | ~26×realtime CBR, ~13× VBR, ~110× decoding |

Encoding is not the equal of a mature encoder at low bitrates — the model is a simplification, and
stereo is coded as two independent channels rather than jointly. Above about 192 kbit/s that stops
being the limiting factor. The figures above are what the tests assert, not estimates.

---

## Recipes

### Convert a file a user uploaded

```ts
import { decode, encode } from 'audiobox';

const audio = await decode(await file.arrayBuffer());

const flac = await encode(audio, 'flac', { compressionLevel: 5 });   // lossless, ~50% of WAV
const wav  = await encode(audio, 'wav', { bitDepth: 24 });
```

`decode()` identifies the format from its **contents**, not its extension — uploads are routinely
mislabelled, and a `.wav` that is really an MP3 is common enough that content sniffing is the only
reliable approach.

### Record from the mic, upload something compact

```ts
import { Audio, encode } from 'audiobox';

// ...capture into a Float32Array per channel, then:
const audio = new Audio([left, right], audioContext.sampleRate);

const flac = await encode(
  audio.trimSilence({ paddingMs: 100 }).normalize({ to: -1 }),
  'flac',
);

await fetch('/upload', { method: 'POST', body: flac });
```

FLAC is lossless and typically 40–60 % of the WAV size, so nothing is thrown away on the way to
your server.

When the upload has to be small rather than exact — a voice note, a preview, anything a user will
listen to once — MP3 is the format everything already plays:

```ts
const mp3 = await encode(audio.normalize({ to: -1 }), 'mp3', { vbr: true, quality: 4 });
```

Variable bitrate spends bits where the material needs them and stops where it does not, so a quiet
recording comes out considerably smaller than a fixed bitrate would give you at the same quality.
Pass `{ bitrate: 128 }` instead when something downstream needs a predictable size.

### Prepare audio for speech-to-text

Whisper, Deepgram and most ASR APIs want 16 kHz mono 16-bit PCM. Sending 48 kHz stereo wastes
bandwidth and makes them resample it anyway.

```ts
const ready = audio.toMono().resample(16_000).normalize({ to: -3 });
const wav = await encode(ready, 'wav', { bitDepth: 16 });
```

The resampler is a windowed-sinc design with a proper anti-aliasing cutoff. Downsampling without
one folds high frequencies back into the speech band as noise, which measurably hurts transcription
accuracy.

### Trim, fade and normalise a podcast episode

```ts
const episode = audio
  .trimSilence({ threshold: -50, paddingMs: 250 })
  .fade({ in: 0.5, out: 2, curve: 'equalPower' })
  .normalize({ to: -16, unit: 'LUFS' });   // the podcast standard
```

`-16 LUFS` is *loudness*, not peak — two episodes normalised this way sound equally loud, which
peak normalisation does not achieve. A true-peak ceiling of -1 dBTP is applied automatically so the
result cannot clip.

```ts
// Did it actually reach the target?
const { audio: out, report } = audio.normalizeWithReport({ to: -16, unit: 'LUFS' });
if (report.limitedByPeak) {
  console.warn(`Held back to protect the peak ceiling; measured ${report.measured.toFixed(1)} LUFS`);
}
```

### Draw a waveform

```ts
const { min, max } = audio.waveform(canvas.width);

for (let x = 0; x < canvas.width; x++) {
  const top = (0.5 - max[x] / 2) * canvas.height;
  const bottom = (0.5 - min[x] / 2) * canvas.height;
  ctx.fillRect(x, top, 1, Math.max(1, bottom - top));
}
```

Returns per-bucket **extremes**, not averages. Averaging flattens transients and produces the
lifeless waveform displays where you cannot see where the beats are.

### Validate an upload without decoding it

```ts
import { probeMp3 } from 'audiobox/mp3';

const info = probeMp3(bytes);
if (info.duration > 600) throw new Error('Too long');
```

### Batch-convert a folder in Node

```ts
import { readAudioFile, writeAudioFile } from 'audiobox/node';
import { readdir } from 'node:fs/promises';

for (const name of await readdir('./in')) {
  const audio = await readAudioFile(`./in/${name}`);
  await writeAudioFile(`./out/${name.replace(/\.\w+$/, '.flac')}`, audio);
}
```

The output format is inferred from the extension.

### Keep the UI responsive

Encoding is seconds of solid CPU, and on the main thread that means a frozen page.

```ts
import { runChunked } from 'audiobox/worker';

await runChunked(items, (item) => processOne(item), {
  onProgress: (p) => setProgress(p),
  signal: controller.signal,
});
```

`runChunked` slices work into ~8 ms bursts and yields between them, so the browser can paint and
handle input. For true parallelism, `audiobox/worker` also exports a message protocol
(`handleWorkerMessage`) to drop into your own worker file — your bundler controls the worker URL,
so the library cannot create one for you without guessing wrong.

### Process a long file in constant memory

```ts
import { createPcmDecodeStream, createResampleStream, createPcmEncodeStream } from 'audiobox/stream';

await source
  .pipeThrough(createPcmDecodeStream({ sampleFormat: 's16', channels: 2, sampleRate: 48000 }))
  .pipeThrough(createResampleStream(16_000))
  .pipeThrough(createPcmEncodeStream({ sampleFormat: 's16' }))
  .pipeTo(destination);
```

Chunk boundaries that fall mid-frame are handled — a decoder that assumes whole frames per chunk
clicks at every boundary.

---

## API

### `decode(input, options?) → Promise<Audio>`

Accepts `Uint8Array`, `ArrayBuffer`, Node `Buffer`, or any typed-array view.

```ts
const audio = await decode(bytes, {
  limits: { maxDurationSeconds: 600 },   // reject anything longer
  signal: controller.signal,
});
```

`decodeSync` is available for the containers compiled into core (WAV, AIFF, AU, CAF).

### `encode(audio, format, options?) → Promise<Uint8Array>`

```ts
await encode(audio, 'wav',  { bitDepth: 24 });
await encode(audio, 'wav',  { sampleFormat: 'ulaw' });
await encode(audio, 'flac', { compressionLevel: 8 });
await encode(audio, 'aiff', { metadata: { title: 'Take 3' } });
await encode(audio, 'mp3',  { bitrate: 192 });
await encode(audio, 'mp3',  { vbr: true, quality: 2 });   // bitrate becomes a ceiling
```

MP3 options: `bitrate` (32–320, default 192), `vbr`, `quality` (0 best … 9 smallest, default 4),
and `tag` to suppress the leading Xing/Info frame when the output is a fragment to be concatenated.

### `Audio`

Immutable — every transform returns a new instance, so `const original = …` stays true.

**Shape** — `sampleRate`, `channels`, `frames`, `duration`, `durationFormatted`, `channelData(i)`,
`allChannels`, `toInterleaved()`

**Edit** — `cut`, `remove`, `concat`, `pad`, `reverse`, `mix`, `crossfadeTo`

**Convert** — `resample`, `toMono`, `toStereo`, `toChannels`, `mapChannels`, `pan`, `split`,
`Audio.merge`

**Level** — `gain`, `gainDb`, `normalize`, `normalizeWithReport`, `fade`

**Process** — `filter`, `limit`, `trimSilence`, `detectSilence`, `removeDcOffset`

**Time** — `speed` (changes pitch), `tempo` (preserves pitch), `pitch` (preserves duration)

**Measure** — `peak`, `peakDb`, `truePeakDb`, `rmsDb`, `loudness`, `waveform`

**Construct** — `new Audio(channels, sampleRate)`, `Audio.silence`, `Audio.fromInterleaved`,
`Audio.fromAudioBuffer`

### Time positions

Anywhere a position is accepted:

```ts
audio.cut({ from: 12.5 });               // seconds
audio.cut({ from: '1:30', to: '2:45' }); // m:ss  or  h:mm:ss.mmm
audio.cut({ from: '500ms' });            // suffixed
audio.cut({ from: { sample: 44100 } });  // exact frame index
audio.cut({ from: -10 });                // last 10 seconds
```

### Errors

Every error is an `AudioboxError` with a stable `code`. Match on the code, never the message.

```ts
import { AudioboxError } from 'audiobox';

try {
  await decode(bytes);
} catch (error) {
  if (error instanceof AudioboxError) {
    switch (error.code) {
      case 'UNSUPPORTED_FORMAT': return reply('That file type is not supported.');
      case 'DECODE_ERROR':       return reply('That file appears to be corrupt.');
      case 'LIMIT_EXCEEDED':     return reply('That file is too large.');
    }
  }
  throw error;
}
```

---

## Correctness

Round-tripping through your own code proves only that it is self-consistent. An inverted sign
convention round-trips perfectly and still sounds wrong everywhere else. So audiobox is checked
against the reference implementations, and those checks run in CI:

| Area | Checked against | Result |
| --- | --- | --- |
| WAV, all sample formats | ffmpeg decodes our files | **bit-exact** |
| A-law / µ-law | CPython `audioop` (the ITU G.711 reference) | **0 mismatches** in 2,028 vectors |
| IMA ADPCM | IMA reference formula; ffmpeg 8.1+ (older ffmpeg: within 64 LSB) | **sample-exact** |
| FLAC | reference `flac -t`, which verifies the embedded MD5 | **passes** at every level |
| FLAC decode | reference `flac` decoder | **identical** |
| EBU R128 loudness | ffmpeg `ebur128` | within **0.03 dB** |
| MP3 decode | ffmpeg, across a 10-case format matrix | **sample-aligned**, 52–64 dB SNR, exact length |
| MP3 encode | ffmpeg and mpg123 decode our files | **no warnings**, 73 dB against the source |

Two ISO-compliant MP3 decoders never agree bit-for-bit — they differ in float precision — so the
bar there is SNR, plus a hard assertion that the decoded audio starts at the *same sample* as the
reference and is the *same length*. A decoder can be spectrally perfect and still useless for
editing if it disagrees about where the audio begins.

The encoder is checked the same way round: what an independent decoder makes of our output. That
distinction is not academic — a round trip through a matched pair cannot tell a correct transform
from two compensating errors, and during development an analysis window that scored 41 dB through
our own decoder scored 41 dB through ffmpeg too, while the correct one scored 73.

The suite is ~270 tests and runs in a few seconds. Interop tests skip automatically when `ffmpeg`
and `flac` are not installed, so contributors without them still get a green run.

---

## Security

Audio decoders parse files from strangers, so failing *safely* matters more than failing rarely.

- **Every read is bounds-checked.** All parsing goes through a reader that throws a typed error
  with a byte offset rather than reading past a buffer. No parser touches a raw `DataView` offset.
- **Declared sizes are never trusted.** A chunk claiming more bytes than the file contains is
  rejected, which blocks both out-of-bounds reads and decompression bombs.
- **Allocation limits are on by default** — 512 MiB decoded, 6 hours, 64 channels. A 44-byte header
  cannot make the process allocate gigabytes. Raise them deliberately via `limits`.
- **No `eval`, no `new Function`**, and lookup tables use `Object.create(null)` so a crafted tag
  name cannot reach `__proto__`. Enforced by a CI check, not just convention.
- **Core reaches nothing** — no filesystem, no network, no `node:*` imports. Verified mechanically
  on every commit.
- **Fuzz-tested** — tens of thousands of malformed, mutated and truncated inputs per run, asserting
  that no decoder leaks an untyped error, hangs, or over-allocates.

Please report vulnerabilities privately — see [SECURITY.md](SECURITY.md).

---

## Zero dependencies, really

`"dependencies": {}`. Nothing vendored, nothing installed alongside it.

Development uses exactly two packages — `typescript` and `@types/node`. Tests run on Node's
built-in runner, the build is two `tsc` invocations with no bundler, and `npm audit` is
structurally clean because there is no tree to audit.

That extends to runtime APIs too: UTF-8 encoding, MD5, CRC and the FFT are implemented here rather
than reached for, because `TextEncoder` and `node:crypto` are not available identically across
every target.

---

## Bundle size

Measured gzipped, including everything each entry point transitively imports:

| Entry point | Gzipped |
| --- | --- |
| `audiobox` | **22.9 kB** |
| `audiobox/mp3` | 38.6 kB |
| `audiobox/stream` | 13.4 kB |
| `audiobox/flac` | 11.9 kB |
| `audiobox/analyze` | 3.7 kB |
| `audiobox/worker` | 1.4 kB |

A complete MP3 decoder *and* encoder — filterbanks, Huffman tables in both directions, and a
psychoacoustic model — in under 40 kB, and **not** in the core bundle: `decode()` and `encode()`
load it on demand, so a project that only handles WAV never downloads it. The package is
`sideEffects: false` and each subpath is a separate entry point, so bundlers drop what you do not
use. CI enforces a size budget per entry point.

---

## Runtime support

| Runtime | Core | `/node` | `/stream` | `/webcodecs` |
| --- | :---: | :---: | :---: | :---: |
| Browsers (Chrome, Firefox, Safari, Edge) | ✅ | — | ✅ | ✅ |
| Node 20+ | ✅ | ✅ | ✅ | — |
| Deno / Bun | ✅ | — | ✅ | — |
| Cloudflare Workers / edge | ✅ | — | ✅ | — |

**On WebCodecs.** It is an accelerator, never a requirement. Node has no WebCodecs —
`AudioEncoder`, `AudioDecoder` and `AudioData` are all undefined there — so a library built on it
cannot be isomorphic. The pure-TypeScript path is the real implementation everywhere;
`audiobox/webcodecs` feature-detects and lights up extra platform codecs (Opus, AAC) in browsers
that have them.

### TypeScript

Types are generated from source and shipped for **both** module formats, so a CJS `require()` gets
CJS types and an ESM `import` gets ESM types. Verified in CI with [publint](https://publint.dev)
and [arethetypeswrong](https://arethetypeswrong.github.io).

---

## Roadmap

This section says what is *not* here yet.

- **Joint stereo for MP3 encoding** — the two channels are currently coded independently, which
  costs bitrate on correlated material. Decoding already handles both mid/side and intensity.
- **Ogg/Opus and Vorbis** — under consideration. Large, and WebCodecs already covers the browser
  case.
- **A turnkey worker** — currently a protocol you wire up; a ready-made version needs per-bundler
  entry points.

### On MP3, patents and licensing

MP3 is **patent-free**. The last US patent expired on 16 April 2017 and the licensing programme was
terminated a week later. There is no royalty question.

The *copyright* question is the one that matters. A codec carries large tables, and where those
tables came from decides what licence the result can carry — code translated from a copyleft
implementation inherits its terms, however far the translation goes.

audiobox derives its own. The Huffman decode tables are generated from
[minimp3](https://github.com/lieff/minimp3) (CC0 / public domain) by a script in the repository;
the Huffman *encode* tables are then derived from those by walking the decode trie; the analysis and
synthesis filter windows are solved for numerically; and everything else follows the ISO
specification. Each generator verifies its own output — Kraft equality, decode round-trips, perfect
reconstruction — so the tables are checked rather than trusted.

audiobox is cleanly MIT and safe to use in commercial products.

---

## Support and sponsorship

Bugs and questions are welcome in [issues](https://github.com/jmsansan/audiobox/issues) and
[discussions](https://github.com/jmsansan/audiobox/discussions), handled on a best-effort basis.

For anything that doesn't belong in a public issue, email **jose.sansan@proton.me**. See
[SUPPORT.md](SUPPORT.md).

If audiobox saves your team time, sponsoring it is the cheapest way to keep it maintained:

- [GitHub Sponsors](https://github.com/sponsors/jmsansan) — recurring
- [Ko-fi](https://ko-fi.com/jmsansan) — one-off

---

## Contributing

```bash
git clone https://github.com/jmsansan/audiobox.git
cd audiobox
npm install

npm test          # ~220 tests, no build step needed
npm run check     # typecheck + portability + tests + size budgets
npm run build     # dual ESM/CJS build
```

Tests run straight from TypeScript source via Node's built-in type stripping, so there is no
watch-and-rebuild loop. That needs **Node 22.6+** to develop on — a contributor requirement only.
The published package is plain JavaScript and supports **Node 20+**, which CI verifies separately
against the built output.

Installing `ffmpeg` and `flac` enables the interop tests; without them those tests skip cleanly.

---

## Credits

MP3 Huffman tables are generated from [minimp3](https://github.com/lieff/minimp3) by lieff,
dedicated to the public domain under CC0 — with thanks, since a correct and unencumbered source for
that data is exactly what makes an MIT-licensed MP3 decoder possible.

Correctness was validated against [ffmpeg](https://ffmpeg.org), the reference
[FLAC](https://xiph.org/flac/) tools, and CPython's `audioop`.

## License

[MIT](LICENSE) © Jose Sanchez
