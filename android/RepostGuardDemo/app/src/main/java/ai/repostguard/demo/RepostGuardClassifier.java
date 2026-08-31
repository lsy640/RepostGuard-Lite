package ai.repostguard.demo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.SystemClock;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.Locale;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

final class RepostGuardClassifier implements AutoCloseable {
    static final int IMAGE_SIZE = 224;
    static final float AIGI_THRESHOLD = 0.060516357421875f;
    static final String MODEL_VERSION = "V3.2.1";
    static final int MODEL_PARAMETERS = 7_955_038;
    static final String MODEL_ASSET = "student_mnv3_fp32.onnx";
    static final String MODEL_SHA256 =
            "f52796946ed3e2a770a7500e77a07aeb7ae8c9312bf414ad14b0be1b252c0a9a";

    private static final long[] INPUT_SHAPE = {1, 3, IMAGE_SIZE, IMAGE_SIZE};
    private static final int PIXEL_COUNT = IMAGE_SIZE * IMAGE_SIZE;

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final OrtSession.SessionOptions sessionOptions;
    private final long modelBytes;

    private RepostGuardClassifier(
            OrtEnvironment environment,
            OrtSession session,
            OrtSession.SessionOptions sessionOptions,
            long modelBytes
    ) {
        this.environment = environment;
        this.session = session;
        this.sessionOptions = sessionOptions;
        this.modelBytes = modelBytes;
    }

    static RepostGuardClassifier fromAssets(Context context) throws IOException, OrtException {
        byte[] model;
        try (InputStream input = context.getAssets().open(MODEL_ASSET)) {
            model = readAll(input);
        }
        String actualHash = sha256(model);
        if (!MODEL_SHA256.equals(actualHash)) {
            throw new IOException("模型校验失败：" + actualHash);
        }

        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = null;
        boolean ownershipTransferred = false;
        try {
            options.setIntraOpNumThreads(4);
            options.setInterOpNumThreads(1);
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            session = environment.createSession(model, options);
            if (!session.getInputNames().contains("images")) {
                throw new OrtException("ONNX 输入名称不是 images");
            }
            if (!session.getOutputNames().contains("logits")
                    || !session.getOutputNames().contains("gate_fractions")) {
                throw new OrtException("ONNX 缺少 logits 或 gate_fractions 输出");
            }
            RepostGuardClassifier classifier = new RepostGuardClassifier(
                    environment, session, options, model.length
            );
            ownershipTransferred = true;
            return classifier;
        } finally {
            if (!ownershipTransferred) {
                try {
                    if (session != null) session.close();
                } finally {
                    options.close();
                }
            }
        }
    }

    void warmUp() throws OrtException {
        runModel(new float[3 * PIXEL_COUNT]);
    }

    Classification classify(Bitmap source) throws OrtException, IOException {
        long totalStart = SystemClock.elapsedRealtimeNanos();
        float[] input = preprocess(source);
        long preprocessEnd = SystemClock.elapsedRealtimeNanos();
        ModelOutput output = runModel(input);
        long inferenceEnd = SystemClock.elapsedRealtimeNanos();

        float score = sigmoid(output.logit());
        return new Classification(
                output.logit(), score, score >= AIGI_THRESHOLD,
                heuristicUncertainty(score), output.semanticGate(), output.forensicGate(),
                nanosToMillis(preprocessEnd - totalStart),
                nanosToMillis(inferenceEnd - preprocessEnd),
                nanosToMillis(inferenceEnd - totalStart)
        );
    }

    long getModelBytes() {
        return modelBytes;
    }

    private ModelOutput runModel(float[] input) throws OrtException {
        try (OnnxTensor tensor = OnnxTensor.createTensor(
                environment, FloatBuffer.wrap(input), INPUT_SHAPE
        ); OrtSession.Result outputs = session.run(Collections.singletonMap("images", tensor))) {
            float logit = parseScalar(outputs.get("logits").orElseThrow(
                    () -> new OrtException("缺少 logits 输出")
            ));
            float[] gates = parseGateFractions(outputs.get("gate_fractions").orElseThrow(
                    () -> new OrtException("缺少 gate_fractions 输出")
            ));
            return new ModelOutput(logit, gates[0], gates[1]);
        }
    }

    private static float parseScalar(OnnxValue output) throws OrtException {
        Object value = output.getValue();
        if (value instanceof float[] vector && vector.length == 1) {
            if (!Float.isFinite(vector[0])) throw new OrtException("ONNX logit 不是有限数值");
            return vector[0];
        }
        if (value instanceof float[][] matrix && matrix.length == 1 && matrix[0].length == 1) {
            if (!Float.isFinite(matrix[0][0])) throw new OrtException("ONNX logit 不是有限数值");
            return matrix[0][0];
        }
        throw new OrtException("无法解析 ONNX logit 输出");
    }

    private static float[] parseGateFractions(OnnxValue output) throws OrtException {
        Object value = output.getValue();
        if (value instanceof float[][] matrix && matrix.length == 1 && matrix[0].length == 2) {
            float semantic = matrix[0][0];
            float forensic = matrix[0][1];
            if (!Float.isFinite(semantic) || !Float.isFinite(forensic)) {
                throw new OrtException("门控权重不是有限数值");
            }
            return new float[]{semantic, forensic};
        }
        throw new OrtException("无法解析 ONNX gate_fractions 输出");
    }

    private static float[] preprocess(Bitmap source) throws IOException {
        Bitmap scaled = Bitmap.createScaledBitmap(source, IMAGE_SIZE, IMAGE_SIZE, true);
        Bitmap harmonized = null;
        try {
            byte[] jpeg;
            try (ByteArrayOutputStream buffer = new ByteArrayOutputStream(96 * 1024)) {
                if (!scaled.compress(Bitmap.CompressFormat.JPEG, 90, buffer)) {
                    throw new IOException("JPEG 质量统一处理失败");
                }
                jpeg = buffer.toByteArray();
            }
            harmonized = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
            if (harmonized == null) {
                throw new IOException("JPEG 质量统一处理后无法解码");
            }
            int[] pixels = new int[PIXEL_COUNT];
            harmonized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
            float[] input = new float[3 * PIXEL_COUNT];
            for (int index = 0; index < PIXEL_COUNT; index++) {
                int pixel = pixels[index];
                input[index] = ((pixel >> 16) & 0xff) / 255.0f;
                input[PIXEL_COUNT + index] = ((pixel >> 8) & 0xff) / 255.0f;
                input[2 * PIXEL_COUNT + index] = (pixel & 0xff) / 255.0f;
            }
            return input;
        } finally {
            if (harmonized != null) harmonized.recycle();
            if (scaled != source) scaled.recycle();
        }
    }

    private static float heuristicUncertainty(float score) {
        // UI stability proxy only; it is not a calibrated posterior uncertainty.
        return (float) Math.exp(-Math.abs(score - AIGI_THRESHOLD) / 0.15f);
    }

    private static float sigmoid(float value) {
        if (value >= 0.0f) {
            double exponent = Math.exp(-value);
            return (float) (1.0 / (1.0 + exponent));
        }
        double exponent = Math.exp(value);
        return (float) (exponent / (1.0 + exponent));
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    private static String sha256(byte[] value) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hashed = digest.digest(value);
            StringBuilder output = new StringBuilder(hashed.length * 2);
            for (byte item : hashed) {
                output.append(String.format(Locale.US, "%02x", item & 0xff));
            }
            return output.toString();
        } catch (NoSuchAlgorithmException error) {
            throw new IOException("设备不支持 SHA-256", error);
        }
    }

    private static byte[] readAll(InputStream input) throws IOException {
        try (ByteArrayOutputStream output = new ByteArrayOutputStream(32 * 1024 * 1024)) {
            byte[] chunk = new byte[64 * 1024];
            int count;
            while ((count = input.read(chunk)) != -1) output.write(chunk, 0, count);
            return output.toByteArray();
        }
    }

    @Override
    public void close() throws OrtException {
        try {
            session.close();
        } finally {
            sessionOptions.close();
        }
    }

    private record ModelOutput(float logit, float semanticGate, float forensicGate) {}

    record Classification(
            float logit, float score, boolean aiGenerated, float uncertainty,
            float semanticGate, float forensicGate,
            double preprocessMs, double inferenceMs, double totalMs
    ) {}
}
