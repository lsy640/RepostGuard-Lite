package ai.repostguard.demo;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.ColorSpace;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import ai.onnxruntime.OrtException;

public final class MainActivity extends Activity {
    private static final int REQUEST_PICK_IMAGE = 1001;
    private static final int MAX_PREVIEW_EDGE = 1024;
    private static final int MAX_EVIDENCE_EDGE = 512;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final AtomicInteger generation = new AtomicInteger();

    private Button selectImageButton;
    private Button resetTransformsButton;
    private ImageView inputImagePreview;
    private ImageView originalRobustnessPreview;
    private ImageView transformedPreview;
    private ImageView srmHeatmap;
    private ImageView nprHeatmap;
    private ProgressBar progressBar;
    private ProgressBar uncertaintyBar;
    private ProgressBar semanticGateBar;
    private ProgressBar forensicGateBar;
    private TextView statusText;
    private TextView resultText;
    private TextView scoreText;
    private TextView uncertaintyText;
    private TextView timingText;
    private TextView detailText;
    private TextView robustnessScoreText;
    private TextView robustnessStatusText;
    private TextView evidenceContextText;
    private TextView gateWeightsText;
    private TextView jpegLabel;
    private TextView blurLabel;
    private TextView resizeLabel;
    private TextView noiseLabel;
    private TextView jitterLabel;
    private TextView cropLabel;
    private SeekBar jpegSeek;
    private SeekBar blurSeek;
    private SeekBar resizeSeek;
    private SeekBar noiseSeek;
    private SeekBar jitterSeek;
    private SeekBar cropSeek;

    private volatile RepostGuardClassifier classifier;
    private volatile boolean busy;
    private volatile boolean destroyed;
    private boolean suppressSliderCallbacks;
    private Bitmap originalBitmap;
    private Bitmap transformedBitmap;
    private Bitmap srmBitmap;
    private Bitmap nprBitmap;
    private RepostGuardClassifier.Classification baselineClassification;
    private Runnable pendingTransform;
    private Future<?> transformFuture;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        bindViews();
        bindControls();
        updateControlLabels();
        selectImageButton.setEnabled(false);
        setWorkbenchEnabled(false);
        initializeModel();
    }

    private void bindViews() {
        selectImageButton = findViewById(R.id.selectImageButton);
        resetTransformsButton = findViewById(R.id.resetTransformsButton);
        inputImagePreview = findViewById(R.id.inputImagePreview);
        originalRobustnessPreview = findViewById(R.id.originalRobustnessPreview);
        transformedPreview = findViewById(R.id.transformedPreview);
        srmHeatmap = findViewById(R.id.srmHeatmap);
        nprHeatmap = findViewById(R.id.nprHeatmap);
        progressBar = findViewById(R.id.progressBar);
        uncertaintyBar = findViewById(R.id.uncertaintyBar);
        semanticGateBar = findViewById(R.id.semanticGateBar);
        forensicGateBar = findViewById(R.id.forensicGateBar);
        statusText = findViewById(R.id.statusText);
        resultText = findViewById(R.id.resultText);
        scoreText = findViewById(R.id.scoreText);
        uncertaintyText = findViewById(R.id.uncertaintyText);
        timingText = findViewById(R.id.timingText);
        detailText = findViewById(R.id.detailText);
        robustnessScoreText = findViewById(R.id.robustnessScoreText);
        robustnessStatusText = findViewById(R.id.robustnessStatusText);
        evidenceContextText = findViewById(R.id.evidenceContextText);
        gateWeightsText = findViewById(R.id.gateWeightsText);
        jpegLabel = findViewById(R.id.jpegLabel);
        blurLabel = findViewById(R.id.blurLabel);
        resizeLabel = findViewById(R.id.resizeLabel);
        noiseLabel = findViewById(R.id.noiseLabel);
        jitterLabel = findViewById(R.id.jitterLabel);
        cropLabel = findViewById(R.id.cropLabel);
        jpegSeek = findViewById(R.id.jpegSeek);
        blurSeek = findViewById(R.id.blurSeek);
        resizeSeek = findViewById(R.id.resizeSeek);
        noiseSeek = findViewById(R.id.noiseSeek);
        jitterSeek = findViewById(R.id.jitterSeek);
        cropSeek = findViewById(R.id.cropSeek);
    }

    private void bindControls() {
        selectImageButton.setOnClickListener(ignored -> openImagePicker());
        resetTransformsButton.setOnClickListener(ignored -> resetTransforms());
        SeekBar.OnSeekBarChangeListener listener = new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                updateControlLabels();
                if (!suppressSliderCallbacks && originalBitmap != null) {
                    scheduleTransform();
                }
            }

            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        };
        jpegSeek.setOnSeekBarChangeListener(listener);
        blurSeek.setOnSeekBarChangeListener(listener);
        resizeSeek.setOnSeekBarChangeListener(listener);
        noiseSeek.setOnSeekBarChangeListener(listener);
        jitterSeek.setOnSeekBarChangeListener(listener);
        cropSeek.setOnSeekBarChangeListener(listener);
    }

    private void initializeModel() {
        setBusy(true, getString(R.string.loading_model));
        worker.execute(() -> {
            long start = SystemClock.elapsedRealtimeNanos();
            RepostGuardClassifier loaded = null;
            try {
                loaded = RepostGuardClassifier.fromAssets(this);
                loaded.warmUp();
                if (destroyed) return;
                double loadMs = nanosToMillis(SystemClock.elapsedRealtimeNanos() - start);
                RepostGuardClassifier readyClassifier = loaded;
                classifier = readyClassifier;
                loaded = null;
                runOnUiThread(() -> {
                    if (isDestroyed()) return;
                    setBusy(false, getString(R.string.ready));
                    detailText.setText(getString(
                            R.string.model_details_format,
                            readyClassifier.getModelBytes() / 1_000_000.0,
                            loadMs,
                            RepostGuardClassifier.AIGI_THRESHOLD
                    ));
                });
            } catch (IOException | OrtException | RuntimeException error) {
                runOnUiThread(() -> showFatalError("Model load failed", error));
            } finally {
                closeQuietly(loaded);
            }
        });
    }

    private void openImagePicker() {
        if (busy || classifier == null) return;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivityForResult(intent, REQUEST_PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_PICK_IMAGE || resultCode != RESULT_OK || data == null) return;
        Uri uri = data.getData();
        if (uri == null) return;
        try {
            getContentResolver().takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (SecurityException ignored) {
            // Some document providers grant only a temporary read permission.
        }
        classifyNewImage(uri);
    }

    private void classifyNewImage(Uri uri) {
        int requestGeneration = generation.incrementAndGet();
        cancelPendingTransform();
        setBusy(true, getString(R.string.processing_image));
        setWorkbenchEnabled(false);
        resultText.setText(R.string.result_waiting);
        scoreText.setText(R.string.score_waiting);
        uncertaintyText.setText(R.string.uncertainty_waiting);
        timingText.setText(R.string.timing_waiting);

        worker.execute(() -> {
            Bitmap bitmap = null;
            Bitmap newSrm = null;
            Bitmap newNpr = null;
            try {
                bitmap = decodePreview(uri);
                RepostGuardClassifier current = classifier;
                if (current == null) throw new IllegalStateException(getString(R.string.model_not_ready));
                RepostGuardClassifier.Classification result = current.classify(bitmap);
                EvidenceImages evidence = createEvidenceImages(bitmap);
                newSrm = evidence.srm();
                newNpr = evidence.npr();
                Bitmap finalBitmap = bitmap;
                Bitmap finalSrm = newSrm;
                Bitmap finalNpr = newNpr;
                runOnUiThread(() -> {
                    if (isDestroyed() || requestGeneration != generation.get()) {
                        recycle(finalBitmap, finalSrm, finalNpr);
                        return;
                    }
                    showInitialClassification(finalBitmap, result, finalSrm, finalNpr);
                });
            } catch (IOException | OrtException | RuntimeException error) {
                recycle(bitmap, newSrm, newNpr);
                runOnUiThread(() -> showRecoverableError("Detection failed", error));
            }
        });
    }

    private void showInitialClassification(
            Bitmap bitmap,
            RepostGuardClassifier.Classification result,
            Bitmap newSrm,
            Bitmap newNpr
    ) {
        recycleOwnedImages();
        originalBitmap = bitmap;
        baselineClassification = result;
        srmBitmap = newSrm;
        nprBitmap = newNpr;
        inputImagePreview.setImageBitmap(bitmap);
        originalRobustnessPreview.setImageBitmap(bitmap);
        transformedPreview.setImageBitmap(bitmap);
        srmHeatmap.setImageBitmap(newSrm);
        nprHeatmap.setImageBitmap(newNpr);
        evidenceContextText.setText(R.string.evidence_context_original);
        showPrimaryResult(result);
        showGateEvidence(result);
        robustnessScoreText.setText(getString(
                R.string.robustness_format,
                result.score() * 100.0,
                result.score() * 100.0,
                0.0
        ));
        robustnessStatusText.setText(R.string.robustness_ready);
        resetTransformsWithoutScheduling();
        setBusy(false, getString(R.string.complete_offline));
        setWorkbenchEnabled(true);
    }

    private void showPrimaryResult(RepostGuardClassifier.Classification result) {
        resultText.setText(result.aiGenerated() ? R.string.result_ai : R.string.result_real);
        resultText.setTextColor(getColor(result.aiGenerated() ? R.color.result_ai : R.color.result_real));
        scoreText.setText(getString(R.string.score_format, result.score() * 100.0, result.logit()));
        uncertaintyText.setText(getString(R.string.uncertainty_format, result.uncertainty() * 100.0));
        uncertaintyBar.setProgress(Math.round(result.uncertainty() * 1000.0f));
        timingText.setText(getString(
                R.string.timing_format,
                result.preprocessMs(), result.inferenceMs(), result.totalMs()
        ));
    }

    private void showGateEvidence(RepostGuardClassifier.Classification result) {
        gateWeightsText.setText(getString(
                R.string.gate_format,
                result.semanticGate() * 100.0,
                result.forensicGate() * 100.0
        ));
        semanticGateBar.setProgress(Math.round(result.semanticGate() * 1000.0f));
        forensicGateBar.setProgress(Math.round(result.forensicGate() * 1000.0f));
    }

    private void scheduleTransform() {
        cancelPendingTransform();
        int requestGeneration = generation.incrementAndGet();
        RobustnessTransforms.Parameters parameters = currentParameters();
        String description = transformDescription();
        pendingTransform = () -> runTransform(requestGeneration, parameters, description);
        mainHandler.postDelayed(pendingTransform, 220L);
        robustnessStatusText.setText(R.string.robustness_running);
    }

    private void runTransform(
            int requestGeneration,
            RobustnessTransforms.Parameters parameters,
            String description
    ) {
        if (destroyed || requestGeneration != generation.get()) return;
        Bitmap source = originalBitmap;
        RepostGuardClassifier current = classifier;
        if (source == null || current == null) return;
        transformFuture = worker.submit(() -> {
            Bitmap perturbed = null;
            Bitmap newSrm = null;
            Bitmap newNpr = null;
            try {
                perturbed = RobustnessTransforms.apply(source, parameters);
                if (isStale(requestGeneration)) {
                    recycle(perturbed);
                    return;
                }
                RepostGuardClassifier.Classification result = current.classify(perturbed);
                if (isStale(requestGeneration)) {
                    recycle(perturbed);
                    return;
                }
                EvidenceImages evidence = createEvidenceImages(perturbed);
                newSrm = evidence.srm();
                newNpr = evidence.npr();
                if (isStale(requestGeneration)) {
                    recycle(perturbed, newSrm, newNpr);
                    return;
                }
                Bitmap finalPerturbed = perturbed;
                Bitmap finalSrm = newSrm;
                Bitmap finalNpr = newNpr;
                runOnUiThread(() -> {
                    if (isDestroyed() || requestGeneration != generation.get()) {
                        recycle(finalPerturbed, finalSrm, finalNpr);
                        return;
                    }
                    showTransformed(finalPerturbed, result, finalSrm, finalNpr, description);
                });
            } catch (IOException | OrtException | RuntimeException error) {
                recycle(perturbed, newSrm, newNpr);
                runOnUiThread(() -> {
                    if (requestGeneration == generation.get()) {
                        robustnessStatusText.setText(getString(
                                R.string.error_detail, "Robustness evaluation failed", safeMessage(error)
                        ));
                    }
                });
            }
        });
    }

    private void showTransformed(
            Bitmap perturbed,
            RepostGuardClassifier.Classification result,
            Bitmap newSrm,
            Bitmap newNpr,
            String description
    ) {
        if (transformedBitmap != null) transformedBitmap.recycle();
        if (srmBitmap != null) srmBitmap.recycle();
        if (nprBitmap != null) nprBitmap.recycle();
        transformedBitmap = perturbed;
        srmBitmap = newSrm;
        nprBitmap = newNpr;
        transformedPreview.setImageBitmap(perturbed);
        srmHeatmap.setImageBitmap(newSrm);
        nprHeatmap.setImageBitmap(newNpr);
        evidenceContextText.setText(R.string.evidence_context_transformed);
        showGateEvidence(result);
        RepostGuardClassifier.Classification baseline = baselineClassification;
        if (baseline != null) {
            double delta = (result.score() - baseline.score()) * 100.0;
            robustnessScoreText.setText(getString(
                    R.string.robustness_format,
                    baseline.score() * 100.0,
                    result.score() * 100.0,
                    delta
            ));
            robustnessStatusText.setText(getString(
                    R.string.robustness_stability_format,
                    description,
                    getString(result.aiGenerated() == baseline.aiGenerated()
                            ? R.string.label_stable : R.string.label_changed)
            ));
        }
    }

    private RobustnessTransforms.Parameters currentParameters() {
        int jpegQuality = 100 - jpegSeek.getProgress() * 10;
        float blurSigma = blurSeek.getProgress() * 0.3f;
        float resizeScale = 0.28f + resizeSeek.getProgress() * 0.08f;
        float noise = noiseSeek.getProgress() * 1.25f;
        int jitter = jitterSeek.getProgress() - 10;
        float cropScale = 0.5f + cropSeek.getProgress() * 0.05f;
        return RobustnessTransforms.Parameters.builder()
                .setJpegQuality(jpegQuality)
                .setBlurSigma(blurSigma)
                .setResizeScale(resizeScale)
                .setNoiseStdDev(noise)
                .setNoiseSeed(20260831L)
                .setBrightnessDelta(jitter / 50.0f)
                .setContrast(1.0f + jitter * 0.02f)
                .setSaturation(1.0f + jitter * 0.02f)
                .setCropScale(cropScale)
                .build();
    }

    private String transformDescription() {
        return String.format(
                Locale.US,
                "Q%d / blur %.1f / resize %.0f%% / noise %.1f / jitter %+d%% / crop %.0f%%",
                100 - jpegSeek.getProgress() * 10,
                blurSeek.getProgress() * 0.3f,
                (0.28f + resizeSeek.getProgress() * 0.08f) * 100.0f,
                noiseSeek.getProgress() * 1.25f,
                (jitterSeek.getProgress() - 10) * 2,
                (0.5f + cropSeek.getProgress() * 0.05f) * 100.0f
        );
    }

    private void updateControlLabels() {
        jpegLabel.setText(getString(R.string.jpeg_format, 100 - jpegSeek.getProgress() * 10));
        blurLabel.setText(getString(R.string.blur_format, blurSeek.getProgress() * 0.3f));
        resizeLabel.setText(getString(
                R.string.resize_format, (0.28f + resizeSeek.getProgress() * 0.08f) * 100.0f
        ));
        noiseLabel.setText(getString(R.string.noise_format, noiseSeek.getProgress() * 1.25f));
        jitterLabel.setText(getString(R.string.jitter_format, (jitterSeek.getProgress() - 10) * 2));
        cropLabel.setText(getString(
                R.string.crop_format, (0.5f + cropSeek.getProgress() * 0.05f) * 100.0f
        ));
    }

    private void resetTransforms() {
        resetTransformsWithoutScheduling();
        scheduleTransform();
    }

    private void resetTransformsWithoutScheduling() {
        suppressSliderCallbacks = true;
        jpegSeek.setProgress(0);
        blurSeek.setProgress(0);
        resizeSeek.setProgress(9);
        noiseSeek.setProgress(0);
        jitterSeek.setProgress(10);
        cropSeek.setProgress(10);
        suppressSliderCallbacks = false;
        updateControlLabels();
    }

    private void setWorkbenchEnabled(boolean enabled) {
        resetTransformsButton.setEnabled(enabled);
        jpegSeek.setEnabled(enabled);
        blurSeek.setEnabled(enabled);
        resizeSeek.setEnabled(enabled);
        noiseSeek.setEnabled(enabled);
        jitterSeek.setEnabled(enabled);
        cropSeek.setEnabled(enabled);
    }

    private Bitmap decodePreview(Uri uri) throws IOException {
        ImageDecoder.Source source = ImageDecoder.createSource(getContentResolver(), uri);
        return ImageDecoder.decodeBitmap(source, (decoder, info, ignored) -> {
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
            decoder.setTargetColorSpace(ColorSpace.get(ColorSpace.Named.SRGB));
            int maxEdge = Math.max(info.getSize().getWidth(), info.getSize().getHeight());
            int sample = Math.max(1, (int) Math.ceil(maxEdge / (double) MAX_PREVIEW_EDGE));
            if (sample > 1) decoder.setTargetSampleSize(sample);
        });
    }

    private static EvidenceImages createEvidenceImages(Bitmap source) {
        Bitmap evidenceInput = source;
        int maxEdge = Math.max(source.getWidth(), source.getHeight());
        if (maxEdge > MAX_EVIDENCE_EDGE) {
            float scale = MAX_EVIDENCE_EDGE / (float) maxEdge;
            evidenceInput = Bitmap.createScaledBitmap(
                    source,
                    Math.max(1, Math.round(source.getWidth() * scale)),
                    Math.max(1, Math.round(source.getHeight() * scale)),
                    true
            );
        }
        Bitmap newSrm = null;
        Bitmap newNpr = null;
        try {
            newSrm = ForensicEvidence.createSrmHeatmap(evidenceInput);
            newNpr = ForensicEvidence.createNprHeatmap(evidenceInput);
            return new EvidenceImages(newSrm, newNpr);
        } catch (RuntimeException error) {
            recycle(newSrm, newNpr);
            throw error;
        } finally {
            if (evidenceInput != source) evidenceInput.recycle();
        }
    }

    private boolean isStale(int requestGeneration) {
        return destroyed || requestGeneration != generation.get();
    }

    private void setBusy(boolean value, String message) {
        busy = value;
        progressBar.setVisibility(value ? View.VISIBLE : View.GONE);
        selectImageButton.setEnabled(!value && classifier != null);
        statusText.setText(message);
    }

    private void showRecoverableError(String title, Throwable error) {
        if (isDestroyed()) return;
        setBusy(false, getString(R.string.error_detail, title, safeMessage(error)));
        setWorkbenchEnabled(originalBitmap != null);
        Toast.makeText(this, title, Toast.LENGTH_LONG).show();
    }

    private void showFatalError(String title, Throwable error) {
        if (isDestroyed()) return;
        busy = false;
        progressBar.setVisibility(View.GONE);
        selectImageButton.setEnabled(false);
        setWorkbenchEnabled(false);
        statusText.setText(getString(R.string.error_detail, title, safeMessage(error)));
        resultText.setText(R.string.app_unavailable);
        Toast.makeText(this, title, Toast.LENGTH_LONG).show();
    }

    private void cancelPendingTransform() {
        if (pendingTransform != null) {
            mainHandler.removeCallbacks(pendingTransform);
            pendingTransform = null;
        }
        if (transformFuture != null) {
            transformFuture.cancel(false);
            transformFuture = null;
        }
    }

    private void recycleOwnedImages() {
        if (transformedBitmap != null) transformedBitmap.recycle();
        if (srmBitmap != null) srmBitmap.recycle();
        if (nprBitmap != null) nprBitmap.recycle();
        if (originalBitmap != null) originalBitmap.recycle();
        transformedBitmap = null;
        srmBitmap = null;
        nprBitmap = null;
        originalBitmap = null;
    }

    private static void recycle(Bitmap... bitmaps) {
        for (Bitmap bitmap : bitmaps) {
            if (bitmap != null && !bitmap.isRecycled()) bitmap.recycle();
        }
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.isBlank() ? error.getClass().getSimpleName() : message;
    }

    private static void closeQuietly(RepostGuardClassifier value) {
        if (value == null) return;
        try {
            value.close();
        } catch (OrtException ignored) {
            // A failed or abandoned initialization owns no UI state.
        }
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        generation.incrementAndGet();
        cancelPendingTransform();
        // Queue cleanup after any in-flight native inference or bitmap transform. Closing the
        // session or recycling its source bitmap on the UI thread can otherwise race native ORT.
        worker.execute(() -> {
            RepostGuardClassifier current = classifier;
            classifier = null;
            if (current != null) {
                try {
                    current.close();
                } catch (OrtException ignored) {
                    // Activity teardown.
                }
            }
            recycleOwnedImages();
        });
        worker.shutdown();
        super.onDestroy();
    }

    private record EvidenceImages(Bitmap srm, Bitmap npr) {}
}
