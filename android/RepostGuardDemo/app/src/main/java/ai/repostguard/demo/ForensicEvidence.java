package ai.repostguard.demo;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

/**
 * Lightweight forensic visualizations for the demo UI.
 *
 * <p><strong>Important:</strong> these maps are visual proxies. They are not model attribution,
 * causal localization, tamper segmentation, or proof of image provenance. The SRM-like view
 * exposes local high-frequency responses, while the NPR view exposes a simple down/up-sampling
 * reconstruction residual. Bright regions can also be caused by genuine texture, edges,
 * compression, resizing, sharpening, noise, or sensor processing.</p>
 */
public final class ForensicEvidence {
    private static final int HISTOGRAM_BINS = 1024;
    private static final float NORMALIZATION_PERCENTILE = 0.995f;
    private static final float[][] HEATMAP_STOPS = {
            {0.00f, 0.0f, 0.0f, 4.0f},
            {0.22f, 50.0f, 10.0f, 94.0f},
            {0.48f, 158.0f, 42.0f, 99.0f},
            {0.72f, 237.0f, 105.0f, 37.0f},
            {1.00f, 252.0f, 255.0f, 164.0f}
    };

    private ForensicEvidence() {
    }

    /**
     * Produces an SRM-like high-pass response heatmap.
     *
     * <p>The response combines horizontal, vertical, and diagonal second-order residuals. This is
     * a human-facing proxy and is not the trained forensic branch's actual activation map.</p>
     */
    public static Bitmap createSrmHeatmap(Bitmap source) {
        requireBitmap(source);
        int width = source.getWidth();
        int height = source.getHeight();
        int[] pixels = readPixels(source);
        float[] grayscale = grayscale(pixels);
        float[] response = new float[pixels.length];

        for (int y = 0; y < height; y++) {
            int up = Math.max(0, y - 1);
            int down = Math.min(height - 1, y + 1);
            int row = y * width;
            for (int x = 0; x < width; x++) {
                int left = Math.max(0, x - 1);
                int right = Math.min(width - 1, x + 1);
                float center = grayscale[row + x];
                float horizontal = grayscale[row + left] - 2.0f * center
                        + grayscale[row + right];
                float vertical = grayscale[up * width + x] - 2.0f * center
                        + grayscale[down * width + x];
                float diagonalDown = grayscale[up * width + left] - 2.0f * center
                        + grayscale[down * width + right];
                float diagonalUp = grayscale[up * width + right] - 2.0f * center
                        + grayscale[down * width + left];
                response[row + x] = (float) Math.sqrt(
                        horizontal * horizontal
                                + vertical * vertical
                                + 0.5f * diagonalDown * diagonalDown
                                + 0.5f * diagonalUp * diagonalUp
                );
            }
        }
        return heatmap(response, width, height);
    }

    /**
     * Produces an NPR-style reconstruction residual heatmap.
     *
     * <p>The proxy computes the RGB difference between the input and a bilinear half-resolution
     * down/up reconstruction. It illustrates local neighboring-pixel inconsistency only; it is not
     * the trained NPR tensor and must not be described as causal evidence.</p>
     */
    public static Bitmap createNprHeatmap(Bitmap source) {
        requireBitmap(source);
        int width = source.getWidth();
        int height = source.getHeight();
        int downWidth = Math.max(1, width / 2);
        int downHeight = Math.max(1, height / 2);

        Bitmap downsampled = Bitmap.createScaledBitmap(source, downWidth, downHeight, true);
        Bitmap reconstructed = Bitmap.createScaledBitmap(downsampled, width, height, true);
        int[] originalPixels = readPixels(source);
        int[] reconstructedPixels = readPixels(reconstructed);
        float[] residual = new float[originalPixels.length];
        for (int index = 0; index < originalPixels.length; index++) {
            int original = originalPixels[index];
            int approximation = reconstructedPixels[index];
            float red = Color.red(original) - Color.red(approximation);
            float green = Color.green(original) - Color.green(approximation);
            float blue = Color.blue(original) - Color.blue(approximation);
            residual[index] = (float) Math.sqrt(
                    (red * red + green * green + blue * blue) / 3.0f
            );
        }

        if (reconstructed != source && reconstructed != downsampled) {
            reconstructed.recycle();
        }
        if (downsampled != source && downsampled != reconstructed) {
            downsampled.recycle();
        }
        return heatmap(residual, width, height);
    }

    /**
     * Creates both proxy maps. The caller owns and must eventually recycle both returned bitmaps.
     */
    public static EvidenceMaps createMaps(Bitmap source) {
        requireBitmap(source);
        return new EvidenceMaps(createSrmHeatmap(source), createNprHeatmap(source));
    }

    /**
     * Creates one bitmap containing SRM-like (left) and NPR residual (right) maps.
     * The returned bitmap is caller-owned; private component maps are recycled before return.
     */
    public static Bitmap createSideBySide(Bitmap source, int gapPx) {
        requireBitmap(source);
        if (gapPx < 0) {
            throw new IllegalArgumentException("gapPx must be >= 0");
        }
        EvidenceMaps maps = createMaps(source);
        Bitmap combined = Bitmap.createBitmap(
                source.getWidth() * 2 + gapPx,
                source.getHeight(),
                Bitmap.Config.ARGB_8888
        );
        Canvas canvas = new Canvas(combined);
        canvas.drawColor(Color.rgb(18, 18, 18));
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(maps.srmHeatmap, 0.0f, 0.0f, paint);
        canvas.drawBitmap(maps.nprHeatmap, source.getWidth() + gapPx, 0.0f, paint);
        maps.srmHeatmap.recycle();
        maps.nprHeatmap.recycle();
        return combined;
    }

    /** Same as {@link #createSideBySide(Bitmap, int)} with no gap. */
    public static Bitmap createSideBySide(Bitmap source) {
        return createSideBySide(source, 0);
    }

    private static Bitmap heatmap(float[] values, int width, int height) {
        float normalization = percentileScale(values, NORMALIZATION_PERCENTILE);
        int[] colors = new int[values.length];
        if (normalization <= 1.0e-6f) {
            for (int index = 0; index < colors.length; index++) {
                colors[index] = Color.BLACK;
            }
        } else {
            for (int index = 0; index < colors.length; index++) {
                float normalized = Math.min(1.0f, Math.max(0.0f, values[index] / normalization));
                // A gentle gamma exposes low-amplitude residuals without letting outliers dominate.
                colors[index] = infernoColor((float) Math.sqrt(normalized));
            }
        }
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bitmap.setPixels(colors, 0, width, 0, 0, width, height);
        return bitmap;
    }

    private static float percentileScale(float[] values, float percentile) {
        float maximum = 0.0f;
        for (float value : values) {
            if (Float.isFinite(value)) {
                maximum = Math.max(maximum, value);
            }
        }
        if (maximum <= 1.0e-6f) {
            return 0.0f;
        }

        int[] histogram = new int[HISTOGRAM_BINS];
        for (float value : values) {
            float finite = Float.isFinite(value) ? Math.max(0.0f, value) : 0.0f;
            int bin = Math.min(
                    HISTOGRAM_BINS - 1,
                    Math.round((finite / maximum) * (HISTOGRAM_BINS - 1))
            );
            histogram[bin]++;
        }
        int target = Math.max(1, (int) Math.ceil(values.length * percentile));
        int cumulative = 0;
        for (int bin = 0; bin < histogram.length; bin++) {
            cumulative += histogram[bin];
            if (cumulative >= target) {
                float scaled = maximum * bin / (HISTOGRAM_BINS - 1.0f);
                return Math.max(scaled, maximum / (HISTOGRAM_BINS - 1.0f));
            }
        }
        return maximum;
    }

    private static int infernoColor(float value) {
        for (int index = 1; index < HEATMAP_STOPS.length; index++) {
            if (value <= HEATMAP_STOPS[index][0]) {
                float[] lower = HEATMAP_STOPS[index - 1];
                float[] upper = HEATMAP_STOPS[index];
                float fraction = (value - lower[0]) / (upper[0] - lower[0]);
                return Color.rgb(
                        Math.round(lerp(lower[1], upper[1], fraction)),
                        Math.round(lerp(lower[2], upper[2], fraction)),
                        Math.round(lerp(lower[3], upper[3], fraction))
                );
            }
        }
        return Color.rgb(252, 255, 164);
    }

    private static float lerp(float first, float second, float amount) {
        return first + (second - first) * amount;
    }

    private static float[] grayscale(int[] pixels) {
        float[] result = new float[pixels.length];
        for (int index = 0; index < pixels.length; index++) {
            int color = pixels[index];
            result[index] = 0.2126f * Color.red(color)
                    + 0.7152f * Color.green(color)
                    + 0.0722f * Color.blue(color);
        }
        return result;
    }

    private static int[] readPixels(Bitmap source) {
        Bitmap readable = source;
        boolean ownsReadable = false;
        if (source.getConfig() == Bitmap.Config.HARDWARE) {
            readable = Bitmap.createBitmap(
                    source.getWidth(),
                    source.getHeight(),
                    Bitmap.Config.ARGB_8888
            );
            Canvas canvas = new Canvas(readable);
            Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
            canvas.drawBitmap(source, 0.0f, 0.0f, paint);
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

    private static void requireBitmap(Bitmap source) {
        if (source == null) {
            throw new IllegalArgumentException("source bitmap must not be null");
        }
        if (source.isRecycled()) {
            throw new IllegalArgumentException("source bitmap is recycled");
        }
    }

    /** Immutable pair of caller-owned proxy heatmaps. */
    public static final class EvidenceMaps {
        private final Bitmap srmHeatmap;
        private final Bitmap nprHeatmap;

        private EvidenceMaps(Bitmap srmHeatmap, Bitmap nprHeatmap) {
            this.srmHeatmap = srmHeatmap;
            this.nprHeatmap = nprHeatmap;
        }

        public Bitmap getSrmHeatmap() {
            return srmHeatmap;
        }

        public Bitmap getNprHeatmap() {
            return nprHeatmap;
        }
    }
}
