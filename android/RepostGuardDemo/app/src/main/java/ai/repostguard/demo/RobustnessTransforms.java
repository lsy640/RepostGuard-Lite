package ai.repostguard.demo;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import java.io.ByteArrayOutputStream;

/**
 * Deterministic, dependency-free image perturbations for the robustness workbench.
 *
 * <p>Every public transform leaves its input untouched and returns a new bitmap owned by the
 * caller. The caller is responsible for recycling the returned bitmap when it is no longer used.
 * {@link #apply(Bitmap, Parameters)} recycles only private intermediate bitmaps that it creates;
 * it never recycles the supplied source bitmap.</p>
 */
public final class RobustnessTransforms {
    private static final float EPSILON = 1.0e-6f;

    private RobustnessTransforms() {
    }

    /**
     * Applies enabled perturbations in a stable order: crop, resize, blur, color jitter, noise,
     * then JPEG. A JPEG quality of 100 means "disabled" for this composite operation.
     */
    public static Bitmap apply(Bitmap source, Parameters parameters) {
        requireBitmap(source);
        if (parameters == null) {
            throw new IllegalArgumentException("parameters must not be null");
        }

        Bitmap current = copyArgb8888(source);
        try {
            if (parameters.cropScale < 1.0f - EPSILON) {
                current = replaceOwned(
                        current,
                        centerCropAndResize(current, parameters.cropScale)
                );
            }
            if (parameters.resizeScale < 1.0f - EPSILON) {
                current = replaceOwned(
                        current,
                        resizeDownUp(current, parameters.resizeScale)
                );
            }
            if (parameters.blurSigma > EPSILON) {
                current = replaceOwned(current, gaussianBlur(current, parameters.blurSigma));
            }
            if (Math.abs(parameters.brightnessDelta) > EPSILON
                    || Math.abs(parameters.contrast - 1.0f) > EPSILON
                    || Math.abs(parameters.saturation - 1.0f) > EPSILON) {
                current = replaceOwned(
                        current,
                        applyColorJitter(
                                current,
                                parameters.brightnessDelta,
                                parameters.contrast,
                                parameters.saturation
                        )
                );
            }
            if (parameters.noiseStdDev > EPSILON) {
                current = replaceOwned(
                        current,
                        addDeterministicNoise(
                                current,
                                parameters.noiseStdDev,
                                parameters.noiseSeed
                        )
                );
            }
            if (parameters.jpegQuality < 100) {
                current = replaceOwned(current, applyJpeg(current, parameters.jpegQuality));
            }
            return current;
        } catch (RuntimeException | Error exception) {
            if (!current.isRecycled()) {
                current.recycle();
            }
            throw exception;
        }
    }

    /** Returns a JPEG encode/decode copy. Transparency is flattened onto white. */
    public static Bitmap applyJpeg(Bitmap source, int quality) {
        requireBitmap(source);
        requireRange("quality", quality, 1, 100);

        Bitmap opaque = Bitmap.createBitmap(
                source.getWidth(),
                source.getHeight(),
                Bitmap.Config.ARGB_8888
        );
        Canvas canvas = new Canvas(opaque);
        canvas.drawColor(Color.WHITE);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(source, 0.0f, 0.0f, paint);

        ByteArrayOutputStream encoded = new ByteArrayOutputStream();
        boolean compressed = opaque.compress(Bitmap.CompressFormat.JPEG, quality, encoded);
        opaque.recycle();
        if (!compressed) {
            throw new IllegalStateException("JPEG encoding failed");
        }

        byte[] bytes = encoded.toByteArray();
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.length, options);
        if (decoded == null) {
            throw new IllegalStateException("JPEG decoding failed");
        }
        if (decoded.getConfig() == Bitmap.Config.ARGB_8888) {
            return decoded;
        }
        Bitmap converted = copyArgb8888(decoded);
        decoded.recycle();
        return converted;
    }

    /** Applies a separable Gaussian blur with edge replication. */
    public static Bitmap gaussianBlur(Bitmap source, float sigma) {
        requireBitmap(source);
        requireFiniteRange("sigma", sigma, 0.0f, 25.0f);
        if (sigma <= EPSILON) {
            return copyArgb8888(source);
        }

        int width = source.getWidth();
        int height = source.getHeight();
        int radius = Math.max(1, (int) Math.ceil(3.0f * sigma));
        float[] kernel = gaussianKernel(radius, sigma);
        int[] input = readPixels(source);
        int[] horizontal = new int[input.length];
        int[] output = new int[input.length];

        for (int y = 0; y < height; y++) {
            int row = y * width;
            for (int x = 0; x < width; x++) {
                float a = 0.0f;
                float r = 0.0f;
                float g = 0.0f;
                float b = 0.0f;
                for (int offset = -radius; offset <= radius; offset++) {
                    int sampleX = clamp(x + offset, 0, width - 1);
                    int color = input[row + sampleX];
                    float weight = kernel[offset + radius];
                    a += Color.alpha(color) * weight;
                    r += Color.red(color) * weight;
                    g += Color.green(color) * weight;
                    b += Color.blue(color) * weight;
                }
                horizontal[row + x] = Color.argb(
                        clampByte(Math.round(a)),
                        clampByte(Math.round(r)),
                        clampByte(Math.round(g)),
                        clampByte(Math.round(b))
                );
            }
        }

        for (int y = 0; y < height; y++) {
            int row = y * width;
            for (int x = 0; x < width; x++) {
                float a = 0.0f;
                float r = 0.0f;
                float g = 0.0f;
                float b = 0.0f;
                for (int offset = -radius; offset <= radius; offset++) {
                    int sampleY = clamp(y + offset, 0, height - 1);
                    int color = horizontal[sampleY * width + x];
                    float weight = kernel[offset + radius];
                    a += Color.alpha(color) * weight;
                    r += Color.red(color) * weight;
                    g += Color.green(color) * weight;
                    b += Color.blue(color) * weight;
                }
                output[row + x] = Color.argb(
                        clampByte(Math.round(a)),
                        clampByte(Math.round(r)),
                        clampByte(Math.round(g)),
                        clampByte(Math.round(b))
                );
            }
        }
        return bitmapFromPixels(output, width, height);
    }

    /** Downscales by {@code scale}, then restores the original dimensions with bilinear filtering. */
    public static Bitmap resizeDownUp(Bitmap source, float scale) {
        requireBitmap(source);
        requireFiniteRange("scale", scale, 0.1f, 1.0f);
        if (scale >= 1.0f - EPSILON) {
            return copyArgb8888(source);
        }

        int width = source.getWidth();
        int height = source.getHeight();
        int downWidth = Math.max(1, Math.round(width * scale));
        int downHeight = Math.max(1, Math.round(height * scale));
        Bitmap downscaled = Bitmap.createScaledBitmap(source, downWidth, downHeight, true);
        Bitmap restored = Bitmap.createScaledBitmap(downscaled, width, height, true);
        if (downscaled != source && downscaled != restored) {
            downscaled.recycle();
        }
        if (restored == source) {
            return copyArgb8888(source);
        }
        if (restored.getConfig() == Bitmap.Config.ARGB_8888) {
            return restored;
        }
        Bitmap converted = copyArgb8888(restored);
        restored.recycle();
        return converted;
    }

    /** Adds deterministic, independent Gaussian noise to RGB channels. */
    public static Bitmap addDeterministicNoise(Bitmap source, float standardDeviation, long seed) {
        requireBitmap(source);
        requireFiniteRange("standardDeviation", standardDeviation, 0.0f, 255.0f);
        if (standardDeviation <= EPSILON) {
            return copyArgb8888(source);
        }

        int width = source.getWidth();
        int height = source.getHeight();
        int[] pixels = readPixels(source);
        DeterministicGaussian random = new DeterministicGaussian(seed);
        for (int index = 0; index < pixels.length; index++) {
            int color = pixels[index];
            int red = clampByte((int) Math.round(
                    Color.red(color) + standardDeviation * random.nextGaussian()
            ));
            int green = clampByte((int) Math.round(
                    Color.green(color) + standardDeviation * random.nextGaussian()
            ));
            int blue = clampByte((int) Math.round(
                    Color.blue(color) + standardDeviation * random.nextGaussian()
            ));
            pixels[index] = Color.argb(Color.alpha(color), red, green, blue);
        }
        return bitmapFromPixels(pixels, width, height);
    }

    /**
     * Applies additive brightness, multiplicative contrast, and saturation jitter.
     * Brightness is normalized: -1 means -255 and +1 means +255.
     */
    public static Bitmap applyColorJitter(
            Bitmap source,
            float brightnessDelta,
            float contrast,
            float saturation
    ) {
        requireBitmap(source);
        requireFiniteRange("brightnessDelta", brightnessDelta, -1.0f, 1.0f);
        requireFiniteRange("contrast", contrast, 0.0f, 3.0f);
        requireFiniteRange("saturation", saturation, 0.0f, 3.0f);

        int width = source.getWidth();
        int height = source.getHeight();
        int[] pixels = readPixels(source);
        float brightness = brightnessDelta * 255.0f;
        for (int index = 0; index < pixels.length; index++) {
            int color = pixels[index];
            float red = (Color.red(color) - 127.5f) * contrast + 127.5f + brightness;
            float green = (Color.green(color) - 127.5f) * contrast + 127.5f + brightness;
            float blue = (Color.blue(color) - 127.5f) * contrast + 127.5f + brightness;
            float luma = 0.2126f * red + 0.7152f * green + 0.0722f * blue;
            red = luma + saturation * (red - luma);
            green = luma + saturation * (green - luma);
            blue = luma + saturation * (blue - luma);
            pixels[index] = Color.argb(
                    Color.alpha(color),
                    clampByte(Math.round(red)),
                    clampByte(Math.round(green)),
                    clampByte(Math.round(blue))
            );
        }
        return bitmapFromPixels(pixels, width, height);
    }

    /** Center-crops by area-side scale, then restores the original dimensions. */
    public static Bitmap centerCropAndResize(Bitmap source, float scale) {
        requireBitmap(source);
        requireFiniteRange("scale", scale, 0.1f, 1.0f);
        if (scale >= 1.0f - EPSILON) {
            return copyArgb8888(source);
        }

        int width = source.getWidth();
        int height = source.getHeight();
        int cropWidth = Math.max(1, Math.round(width * scale));
        int cropHeight = Math.max(1, Math.round(height * scale));
        int left = (width - cropWidth) / 2;
        int top = (height - cropHeight) / 2;
        Bitmap crop = Bitmap.createBitmap(source, left, top, cropWidth, cropHeight);
        Bitmap restored = Bitmap.createScaledBitmap(crop, width, height, true);
        if (crop != source && crop != restored) {
            crop.recycle();
        }
        if (restored == source) {
            return copyArgb8888(source);
        }
        if (restored.getConfig() == Bitmap.Config.ARGB_8888) {
            return restored;
        }
        Bitmap converted = copyArgb8888(restored);
        restored.recycle();
        return converted;
    }

    private static Bitmap replaceOwned(Bitmap previous, Bitmap replacement) {
        if (previous != replacement && !previous.isRecycled()) {
            previous.recycle();
        }
        return replacement;
    }

    private static Bitmap copyArgb8888(Bitmap source) {
        Bitmap result = Bitmap.createBitmap(
                source.getWidth(),
                source.getHeight(),
                Bitmap.Config.ARGB_8888
        );
        Canvas canvas = new Canvas(result);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(source, 0.0f, 0.0f, paint);
        return result;
    }

    private static int[] readPixels(Bitmap source) {
        Bitmap readable = source;
        boolean ownsReadable = false;
        if (source.getConfig() == Bitmap.Config.HARDWARE) {
            readable = copyArgb8888(source);
            ownsReadable = true;
        }
        int[] pixels = new int[readable.getWidth() * readable.getHeight()];
        readable.getPixels(
                pixels,
                0,
                readable.getWidth(),
                0,
                0,
                readable.getWidth(),
                readable.getHeight()
        );
        if (ownsReadable) {
            readable.recycle();
        }
        return pixels;
    }

    private static Bitmap bitmapFromPixels(int[] pixels, int width, int height) {
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }

    private static float[] gaussianKernel(int radius, float sigma) {
        float[] kernel = new float[radius * 2 + 1];
        double denominator = 2.0 * sigma * sigma;
        double total = 0.0;
        for (int index = -radius; index <= radius; index++) {
            float value = (float) Math.exp(-(index * index) / denominator);
            kernel[index + radius] = value;
            total += value;
        }
        for (int index = 0; index < kernel.length; index++) {
            kernel[index] /= (float) total;
        }
        return kernel;
    }

    private static void requireBitmap(Bitmap source) {
        if (source == null) {
            throw new IllegalArgumentException("source bitmap must not be null");
        }
        if (source.isRecycled()) {
            throw new IllegalArgumentException("source bitmap is recycled");
        }
    }

    private static void requireRange(String name, int value, int minimum, int maximum) {
        if (value < minimum || value > maximum) {
            throw new IllegalArgumentException(
                    name + " must be in [" + minimum + ", " + maximum + "]"
            );
        }
    }

    private static void requireFiniteRange(
            String name,
            float value,
            float minimum,
            float maximum
    ) {
        if (!Float.isFinite(value) || value < minimum || value > maximum) {
            throw new IllegalArgumentException(
                    name + " must be finite and in [" + minimum + ", " + maximum + "]"
            );
        }
    }

    private static int clamp(int value, int minimum, int maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private static int clampByte(int value) {
        return clamp(value, 0, 255);
    }

    /** Immutable settings for {@link #apply(Bitmap, Parameters)}. */
    public static final class Parameters {
        private final int jpegQuality;
        private final float blurSigma;
        private final float resizeScale;
        private final float noiseStdDev;
        private final long noiseSeed;
        private final float brightnessDelta;
        private final float contrast;
        private final float saturation;
        private final float cropScale;

        private Parameters(Builder builder) {
            jpegQuality = builder.jpegQuality;
            blurSigma = builder.blurSigma;
            resizeScale = builder.resizeScale;
            noiseStdDev = builder.noiseStdDev;
            noiseSeed = builder.noiseSeed;
            brightnessDelta = builder.brightnessDelta;
            contrast = builder.contrast;
            saturation = builder.saturation;
            cropScale = builder.cropScale;
        }

        public static Builder builder() {
            return new Builder();
        }

        public int getJpegQuality() {
            return jpegQuality;
        }

        public float getBlurSigma() {
            return blurSigma;
        }

        public float getResizeScale() {
            return resizeScale;
        }

        public float getNoiseStdDev() {
            return noiseStdDev;
        }

        public long getNoiseSeed() {
            return noiseSeed;
        }

        public float getBrightnessDelta() {
            return brightnessDelta;
        }

        public float getContrast() {
            return contrast;
        }

        public float getSaturation() {
            return saturation;
        }

        public float getCropScale() {
            return cropScale;
        }

        /** Mutable construction helper; the built {@link Parameters} value is immutable. */
        public static final class Builder {
            private int jpegQuality = 100;
            private float blurSigma = 0.0f;
            private float resizeScale = 1.0f;
            private float noiseStdDev = 0.0f;
            private long noiseSeed = 20260831L;
            private float brightnessDelta = 0.0f;
            private float contrast = 1.0f;
            private float saturation = 1.0f;
            private float cropScale = 1.0f;

            public Builder setJpegQuality(int value) {
                requireRange("jpegQuality", value, 1, 100);
                jpegQuality = value;
                return this;
            }

            public Builder setBlurSigma(float value) {
                requireFiniteRange("blurSigma", value, 0.0f, 25.0f);
                blurSigma = value;
                return this;
            }

            public Builder setResizeScale(float value) {
                requireFiniteRange("resizeScale", value, 0.1f, 1.0f);
                resizeScale = value;
                return this;
            }

            public Builder setNoiseStdDev(float value) {
                requireFiniteRange("noiseStdDev", value, 0.0f, 255.0f);
                noiseStdDev = value;
                return this;
            }

            public Builder setNoiseSeed(long value) {
                noiseSeed = value;
                return this;
            }

            public Builder setBrightnessDelta(float value) {
                requireFiniteRange("brightnessDelta", value, -1.0f, 1.0f);
                brightnessDelta = value;
                return this;
            }

            public Builder setContrast(float value) {
                requireFiniteRange("contrast", value, 0.0f, 3.0f);
                contrast = value;
                return this;
            }

            public Builder setSaturation(float value) {
                requireFiniteRange("saturation", value, 0.0f, 3.0f);
                saturation = value;
                return this;
            }

            public Builder setCropScale(float value) {
                requireFiniteRange("cropScale", value, 0.1f, 1.0f);
                cropScale = value;
                return this;
            }

            public Parameters build() {
                return new Parameters(this);
            }
        }
    }

    /** Fast deterministic Gaussian source using SplitMix64 and the Marsaglia polar method. */
    private static final class DeterministicGaussian {
        private long state;
        private boolean hasSpare;
        private double spare;

        private DeterministicGaussian(long seed) {
            state = seed;
        }

        private double nextGaussian() {
            if (hasSpare) {
                hasSpare = false;
                return spare;
            }
            double first;
            double second;
            double radiusSquared;
            do {
                first = nextUnit() * 2.0 - 1.0;
                second = nextUnit() * 2.0 - 1.0;
                radiusSquared = first * first + second * second;
            } while (radiusSquared <= 0.0 || radiusSquared >= 1.0);
            double scale = Math.sqrt(-2.0 * Math.log(radiusSquared) / radiusSquared);
            spare = second * scale;
            hasSpare = true;
            return first * scale;
        }

        private double nextUnit() {
            state += 0x9E3779B97F4A7C15L;
            long value = state;
            value = (value ^ (value >>> 30)) * 0xBF58476D1CE4E5B9L;
            value = (value ^ (value >>> 27)) * 0x94D049BB133111EBL;
            value ^= value >>> 31;
            return (value >>> 11) * 0x1.0p-53;
        }
    }
}
