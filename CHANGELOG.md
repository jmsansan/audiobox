# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **MP3 VBR tag.** The LAME extension of a variable-bitrate stream claimed constant bitrate,
  so mpg123 reported such files as CBR. Its bitrate byte now holds the lowest bitrate a frame
  may take, as the layout defines for VBR, not the highest.
- **MP3 tag channel mode.** Byte 24 of the LAME extension stayed zero, which the layout reads
  as a mono source at 32 kHz or below. It now holds the channel mode and the sample rate.
- **MP3 tag CRCs.** The music and tag CRCs of the LAME extension were zero. Both are now
  computed, as LAME does. Apple Core Audio checks the tag CRC before it trims the priming and
  padding, so `afconvert` now decodes audiobox MP3s to the exact input length. ffmpeg checks it
  before it reports the encoder name.
- **MP3 tag music length.** Bytes 28-31 of the LAME extension left out the tag frame. They now
  count from it, as LAME and ffmpeg do, so they equal the Xing byte count.
- **MP3 gapless decoding of LAME files with a short encoder name.** The reader accepted only
  printable bytes in the 9-byte name, but LAME 4.0 pads `LAME4.0 ` with a NUL byte. Such files
  decoded with the priming and padding left in. Trailing NUL bytes are now accepted.

## [1.1.0] — 2026-08-04

### Added

- **MP3 encoding.** MPEG-1 Layer III at 32-320 kbit/s, mono or stereo, at 32 / 44.1 / 48 kHz,
  via `encode(audio, 'mp3', ...)` or `encodeMp3()` from `audiobox/mp3`. Includes a
  psychoacoustic model, window switching on transients, the bit reservoir, and a Xing/Info
  tag so `decode(encode(x))` returns exactly the samples it was given. ffmpeg and mpg123
  decode the output without warnings; ffmpeg measures it at 73 dB against the source on
  tonal material.
- **Variable bitrate**, with `{ vbr: true, quality }` where quality runs 0 (best) to 9.
  Each frame takes the bitrate its content needs, and the stream carries a Xing seek table
  so players can seek without assuming byte position tracks time. On easy material this is
  15-25% smaller than constant bitrate at matched quality; on noise there is nothing to
  save and it does not pretend otherwise.
- `MPEG1_LAYER3_BITRATE_VALUES` — the 14 bitrates a frame can actually be written at.
  The existing `MPEG1_LAYER3_BITRATES` is indexed by the header's 4-bit field and carries
  two sentinels (free format and reserved), which are not bitrates.

### Fixed

- **Gapless decoding lost the last 64 samples of every tagged file.** The trim subtracted
  this implementation's filterbank offset at the head but never emitted the samples still
  inside the filterbank at the tail. Files whose padding was at its minimum lost real audio.
  All ten cases in the ffmpeg comparison matrix now match its output length **exactly**;
  the assertion was previously a 441-sample tolerance, which is what hid it.
- The Xing/LAME tag reader no longer requires the encoder to be named `LAME`, `Lavc` or
  `Lavf`. Any printable encoder name is accepted, with the delay and padding values
  sanity-checked instead — a whitelist is why other encoders label themselves `LAME`, and
  it stopped audiobox reading its own tag.
- `npm run check` measured bundle sizes against a stale `dist/`, so the size budgets were
  not actually enforced against the code being checked.

### Changed

- `audiobox/mp3` is now 38.6 kB gzipped, up from 17.9 kB, and carries both a decoder and an
  encoder. It remains outside the core bundle and loads on demand, so the core is unchanged
  at 22.9 kB.

## [1.0.1] — 2026-08-03

### Changed

- **MP3 decoding is roughly 14x faster** (25 ms for 3 s of stereo audio on Node 23,
  about 120x realtime, up from ~9x). The synthesis filterbank wrapped its ring buffer
  with `%` 512 times per 32 output samples; the buffer length is a power of two and
  every index is non-negative, so `&` is equivalent and far cheaper. Index stepping and
  table lookups in the same loop were hoisted. Decoded output is unchanged — the full
  ffmpeg comparison matrix still reports sample-aligned results at 52-64 dB SNR.

### Added

- Companion site with an in-browser playground, documentation and recipes, published
  from `docs/` via GitHub Pages.

## [1.0.0] — 2026-08-03

First release.

### Added

- **Containers** — WAV (RIFF, RF64/BW64, `WAVE_FORMAT_EXTENSIBLE`), FLAC, AIFF/AIFF-C, CAF,
  and Sun/NeXT AU, all read and write.
- **Sample formats** — unsigned/signed 8-bit, 16-, 24- and 32-bit integer, 32- and 64-bit
  float, G.711 A-law and µ-law, plus IMA ADPCM decoding.
- **`Audio`** — an immutable value type with a fluent transform API.
- **Editing** — cut, remove, concat, pad, reverse, mix, crossfade.
- **Conversion** — windowed-sinc resampling with anti-aliasing, channel-layout conversion,
  bit-depth conversion with optional TPDF dither.
- **Processing** — gain, fades, peak/true-peak/LUFS normalisation, biquad filters, lookahead
  limiter, silence trimming, DC-offset removal, WSOLA time-stretch and pitch-shift.
- **Measurement** — peak, true peak, RMS, ITU-R BS.1770 / EBU R128 loudness, waveform peaks,
  FFT and spectrogram.
- **MP3 decoding** — MPEG-1, MPEG-2 and MPEG-2.5 Layer III at every sample rate from 8 to
  48 kHz, CBR and VBR, mono/stereo/dual-channel/joint stereo (mid-side and intensity), with
  gapless playback from Xing/VBRI/LAME tags. Verified sample-aligned against ffmpeg at
  52-64 dB SNR.
- **MP3 inspection** — frame headers, ID3v2 sizing, Xing/VBRI/LAME tags, and `probeMp3()` for
  duration, bitrate and channel mode without decoding a single sample.
- **Subpath entry points** — `audiobox/flac`, `/mp3`, `/analyze`, `/stream`, `/worker`,
  `/node`, `/webcodecs`, so unused codecs stay out of your bundle.
- **Safety limits** on every decode, enabled by default.
- Dual ESM + CJS build with per-format type declarations; zero runtime dependencies.

### Known limitations

- MP3 *encoding* is not implemented. Decoding is.
- `s32` and `f64` round-trips are accurate to float32 precision (~24 bits), because decoded
  audio is held as `Float32Array`. Depths up to 24 bits are bit-exact.
- Streaming resampling processes each chunk independently, so filter state does not carry
  across chunk boundaries.

[Unreleased]: https://github.com/jmsansan/audiobox/compare/v1.1.0...HEAD
[1.0.1]: https://github.com/jmsansan/audiobox/releases/tag/v1.0.1
[1.0.0]: https://github.com/jmsansan/audiobox/releases/tag/v1.0.0
