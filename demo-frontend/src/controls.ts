import type { TransformControls } from './types'

/**
 * Create a plain snapshot of the robustness controls.
 *
 * Vue exposes component props and reactive state through Proxy objects, which
 * cannot be passed directly to structuredClone in browsers. Keeping the copy
 * explicit also guarantees that only the six API-supported control groups are
 * sent to the local service.
 */
export function cloneTransformControls(source: TransformControls): TransformControls {
  return {
    jpeg: { enabled: source.jpeg.enabled, quality: source.jpeg.quality },
    blur: { enabled: source.blur.enabled, sigma: source.blur.sigma },
    resize: { enabled: source.resize.enabled, scale: source.resize.scale },
    noise: { enabled: source.noise.enabled, sigma: source.noise.sigma },
    jitter: {
      enabled: source.jitter.enabled,
      brightness: source.jitter.brightness,
      contrast: source.jitter.contrast,
      saturation: source.jitter.saturation,
    },
    crop: { enabled: source.crop.enabled, ratio: source.crop.ratio },
  }
}
