// Perceptually-ordered ramp for spectrogram intensity.
//
// Control points sampled from viridis: monotonic in lightness, so a formant
// band reads as brighter than its surroundings for viewers with any of the
// common colour-vision deficiencies. A rainbow ramp would invent contrast where
// the data has none.
const STOPS: [number, number, number][] = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [180, 222, 44],
  [253, 231, 37],
];

/** Map a normalised intensity in [0, 1] to an RGB triple. */
export function intensityToRgb(value: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, value));
  const position = clamped * (STOPS.length - 1);
  const index = Math.min(Math.floor(position), STOPS.length - 2);
  const t = position - index;
  const [r0, g0, b0] = STOPS[index];
  const [r1, g1, b1] = STOPS[index + 1];
  return [
    Math.round(r0 + (r1 - r0) * t),
    Math.round(g0 + (g1 - g0) * t),
    Math.round(b0 + (b1 - b0) * t),
  ];
}
