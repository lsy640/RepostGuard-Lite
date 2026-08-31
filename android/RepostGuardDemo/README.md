# RepostGuard Lite V3.2.1 Android demo

Offline Galaxy S23 Ultra test application for the selected V3.2.1 distilled
Student checkpoint (`epoch=3`, `global_step=1545`).

## Demo layout

The app has three areas:

1. **Input and result** — select an image, then show the AIGC probability signal,
   the label produced by the frozen operating threshold, and a threshold-distance
   uncertainty proxy.
2. **Robustness workbench** — JPEG, blur, resize, noise, color jitter and crop
   controls; the original and transformed images, scores and score delta are
   shown side by side.
3. **Evidence and limitations** — model-exported semantic/forensic gate fractions
   plus Android-side SRM-like and NPR residual heatmap proxies. These visualizations
   are not model attribution and are not provenance evidence.

## Runtime contract

- Model: V3.2.1 MobileNetV3-Large semantic branch plus a lightweight
  EfficientNet-B0 high-frequency/NPR forensic branch
- Parameters: 7,955,038
- Runtime: ONNX Runtime Android 1.29.0, CPU baseline with four intra-op threads
- ABI: `arm64-v8a`
- Input: RGB float32 `[1,3,224,224]` in `[0,1]`
- Embedded model normalization: ImageNet mean/std
- App preprocessing: direct square resize, JPEG quality-90 round-trip, RGB tensor
- Outputs: `logits` and `[semantic, forensic] gate_fractions`
- Displayed probability signal: `sigmoid(AIGI logit)`
- Frozen decision threshold: `0.060516357421875`
- ONNX: FP32, 31,333,268 bytes, not quantized

The app requests no Internet or storage permission. Images are read through the
Android system document picker and inference remains on the device.

The displayed uncertainty is a threshold-distance heuristic, not a calibrated
posterior uncertainty. A score can be wrong or biased and must not be presented
as proof that an image came from a particular generator. It cannot replace
content credentials, watermark checks or source-chain provenance.

## Model integrity

Expected ONNX SHA-256:

```text
f52796946ed3e2a770a7500e77a07aeb7ae8c9312bf414ad14b0be1b252c0a9a
```

The export was checked against PyTorch on zero, one and random-batch inputs.
Maximum observed probability error was `0.000427231`; maximum gate error was
`0.000653714`.

## Build

This checkout uses JDK 17, Android API 35, Build Tools 35.0.0, Android Gradle
Plugin 8.9.2 and Gradle 8.11.1.

```powershell
$env:JAVA_HOME = 'D:\projects\tiktok26\.toolchains\jdk17\jdk-17.0.20.1+1'
$env:ANDROID_HOME = 'D:\projects\tiktok26\.android-sdk'
.\gradlew.bat :app:lintDebug :app:assembleDebug --offline --no-daemon
```

Install on a USB-debuggable Galaxy S23 Ultra:

```powershell
D:\projects\tiktok26\.android-sdk\platform-tools\adb.exe install -r .\app\build\outputs\apk\debug\app-debug.apk
```

The displayed timings separate preprocessing, model inference and total time.
They must be measured on the physical phone; a desktop build does not establish
phone latency.
