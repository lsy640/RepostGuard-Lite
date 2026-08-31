<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { Activity, AlertCircle, Cpu, ImageUp, LoaderCircle, ShieldCheck, Sparkles } from 'lucide-vue-next'
import { getHealth, inferImage, runRobustness } from './api'
import { cloneTransformControls } from './controls'
import BatchPanel from './components/BatchPanel.vue'
import EvidencePanel from './components/EvidencePanel.vue'
import RobustnessLab from './components/RobustnessLab.vue'
import type { HealthResponse, InferenceResult, Language, ModelName, TransformControls } from './types'

const defaultControls = (): TransformControls => ({
  jpeg: { enabled: false, quality: 70 },
  blur: { enabled: false, sigma: 1.0 },
  resize: { enabled: false, scale: 0.5 },
  noise: { enabled: false, sigma: 0.02 },
  jitter: { enabled: false, brightness: 1.2, contrast: 1.2, saturation: 1.2 },
  crop: { enabled: false, ratio: 0.8 },
})

const model = ref<ModelName>('m2')
const language = ref<Language>('zh')
const health = ref<HealthResponse | null>(null)
const healthError = ref('')
const dragging = ref(false)
const selectedFile = ref<File | null>(null)
const originalPreview = ref('')
const result = ref<InferenceResult | null>(null)
const robustnessResult = ref<InferenceResult | null>(null)
const inferenceLoading = ref(false)
const robustnessLoading = ref(false)
const errorMessage = ref('')
const history = ref<number[]>([])
const controls = reactive<TransformControls>(defaultControls())
let inferController: AbortController | null = null
let robustnessController: AbortController | null = null
let robustnessTimer: number | undefined
let robustnessSequence = 0
let healthTimer: number | undefined

const zh = computed(() => language.value === 'zh')
const copy = computed(() => zh.value ? {
  modelSelector: '选择推理模型', languageSelector: '选择界面语言', connecting: '正在连接', serviceOffline: '服务离线', modelReady: '模型就绪', modelIdle: '模型待机', activeModel: '当前模型',
  intro: '概率信号 · 取证证据 · 鲁棒性分析 · 100% 本地', localDisconnected: '本地推理服务未连接', analysisIncomplete: '分析未完成', sectionTitle: '输入与结果',
  sectionSummary: '原始 sigmoid 分数与研究温度校准分数并列展示；标签仍由冻结 validation 操作点决定。', dropImage: '拖入图片进行检测', selectImage: '选择图片', analysing: '正在本地分析',
  loadingModel: '正在加载模型权重', firstLoad: '首次加载可能需要数秒', waiting: '等待模型分析', localOnly: '图片只在当前 Mac 上处理。真实 checkpoint 未就绪时不会生成模拟结果。',
  sourceAlt: '上传的原始图片预览', modelAlt: '模型实际输入预览', modelInput: '模型输入 · JPEG Q90', resultLabel: '校准后标签', crossed: '分数超过冻结的 AIGC 操作点。', stayedBelow: '分数低于冻结的 AIGC 操作点。',
  rawScore: '原始 AIGC 分数', calibratedScore: '研究校准分数', scoreUncertainty: '分数不确定性', entropy: '归一化二元熵', latency: '本地推理耗时', verified: '已验证',
  rawThreshold: '分类阈值', calibratedThreshold: '校准阈值', rawThresholdHelp: '原始模型分数达到或超过该冻结操作点时，标签判定为 AIGC', calibratedThresholdHelp: '原始冻结操作点经过相同温度变换后的阈值，标签判定保持一致',
  firstFrame: '仅分析第一帧', calibrationNote: '研究校准只调整分数尺度，不扩大模型适用范围，也不代表目标流量中的真实发生概率。', footer: '本地研究界面 · 图片不会离开此设备', detecting: '检测设备',
} : {
  modelSelector: 'Select inference model', languageSelector: 'Select interface language', connecting: 'Connecting', serviceOffline: 'Service offline', modelReady: 'Model ready', modelIdle: 'Model idle', activeModel: 'Active model',
  intro: 'Probability signal · Forensic evidence · Robustness analysis · 100% local', localDisconnected: 'Local inference service disconnected', analysisIncomplete: 'Analysis incomplete', sectionTitle: 'Input & Result',
  sectionSummary: 'Raw sigmoid and research temperature-calibrated scores are shown together; the label still uses the frozen validation operating point.', dropImage: 'Drop an image to inspect', selectImage: 'Select image', analysing: 'Analysing locally',
  loadingModel: 'Loading model checkpoint', firstLoad: 'The first load may take a few seconds', waiting: 'Waiting for model analysis', localOnly: 'The image is processed only on this Mac. No simulated result is returned when the real checkpoint is unavailable.',
  sourceAlt: 'Uploaded source image preview', modelAlt: 'Actual model input preview', modelInput: 'MODEL INPUT · JPEG Q90', resultLabel: 'CALIBRATED LABEL', crossed: 'Score crossed the frozen AIGC operating point.', stayedBelow: 'Score stayed below the frozen AIGC operating point.',
  rawScore: 'RAW AIGC SCORE', calibratedScore: 'RESEARCH CALIBRATED', scoreUncertainty: 'SCORE UNCERTAINTY', entropy: 'normalized binary entropy', latency: 'LOCAL LATENCY', verified: 'verified',
  rawThreshold: 'Classification threshold', calibratedThreshold: 'Calibrated threshold', rawThresholdHelp: 'A raw model score at or above this frozen operating point is labeled AIGC', calibratedThresholdHelp: 'The frozen operating point after the same temperature transform; the label is unchanged',
  firstFrame: 'first frame analysed', calibrationNote: 'Research calibration changes only the score scale. It does not expand model coverage or represent the true prevalence in target traffic.', footer: 'Local research surface · no image leaves this device', detecting: 'detecting',
})
const evidenceResult = computed(() => robustnessResult.value || result.value)
const selectedHealth = computed(() => health.value?.models[model.value] ?? null)
const serviceLabel = computed(() => {
  if (healthError.value) return copy.value.serviceOffline
  if (!health.value) return copy.value.connecting
  return `${health.value.device.toUpperCase()} · ${health.value.models[model.value].loaded ? copy.value.modelReady : copy.value.modelIdle}`
})
const anyTransformEnabled = computed(() => Object.values(controls).some((control) => control.enabled))

function formatPercent(value: number | null | undefined) {
  return value == null ? 'N/A' : `${(value * 100).toFixed(2)}%`
}

function formatBytes(value: number | undefined) {
  if (value == null) return '—'
  return value >= 1024 * 1024 ? `${(value / (1024 * 1024)).toFixed(2)} MB` : `${Math.ceil(value / 1024)} KB`
}

async function refreshHealth() {
  try {
    health.value = await getHealth()
    healthError.value = ''
  } catch (error) {
    healthError.value = error instanceof Error ? error.message : String(error)
  }
}

function selectFile(file?: File) {
  if (!file) return
  if (originalPreview.value) URL.revokeObjectURL(originalPreview.value)
  selectedFile.value = file
  originalPreview.value = URL.createObjectURL(file)
  void analyseFile()
}

function onDrop(event: DragEvent) {
  dragging.value = false
  selectFile(event.dataTransfer?.files[0])
}

async function analyseFile() {
  const file = selectedFile.value
  if (!file) return
  inferController?.abort()
  robustnessController?.abort()
  inferController = new AbortController()
  inferenceLoading.value = true
  errorMessage.value = ''
  result.value = null
  robustnessResult.value = null
  history.value = []
  try {
    const response = await inferImage(file, model.value, inferController.signal)
    result.value = response
    history.value = [response.raw_score]
    await refreshHealth()
    if (anyTransformEnabled.value) scheduleRobustness()
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') return
    errorMessage.value = error instanceof Error ? error.message : String(error)
  } finally {
    inferenceLoading.value = false
  }
}

function setControls(next: TransformControls) {
  Object.assign(controls, next)
}

function resetControls() {
  Object.assign(controls, defaultControls())
  robustnessController?.abort()
  robustnessResult.value = null
  history.value = result.value ? [result.value.raw_score] : []
}

function scheduleRobustness() {
  window.clearTimeout(robustnessTimer)
  if (!result.value) return
  if (!anyTransformEnabled.value) {
    robustnessController?.abort()
    robustnessResult.value = null
    return
  }
  robustnessTimer = window.setTimeout(executeRobustness, 300)
}

async function executeRobustness() {
  const clean = result.value
  if (!clean) return
  robustnessController?.abort()
  robustnessController = new AbortController()
  const sequence = ++robustnessSequence
  robustnessLoading.value = true
  errorMessage.value = ''
  try {
    const response = await runRobustness(
      clean.image_id,
      model.value,
      cloneTransformControls(controls),
      robustnessController.signal,
    )
    if (sequence !== robustnessSequence) return
    robustnessResult.value = response
    history.value = [...history.value, response.raw_score].slice(-20)
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') return
    errorMessage.value = error instanceof Error ? error.message : String(error)
  } finally {
    if (sequence === robustnessSequence) robustnessLoading.value = false
  }
}

watch(model, () => {
  robustnessSequence += 1
  robustnessResult.value = null
  history.value = []
  if (selectedFile.value) void analyseFile()
})

watch(language, (value) => {
  document.documentElement.lang = value === 'zh' ? 'zh-CN' : 'en'
}, { immediate: true })

watch(controls, scheduleRobustness, { deep: true })

onMounted(() => {
  void refreshHealth()
  healthTimer = window.setInterval(refreshHealth, 10_000)
})

onBeforeUnmount(() => {
  inferController?.abort()
  robustnessController?.abort()
  window.clearTimeout(robustnessTimer)
  window.clearInterval(healthTimer)
  if (originalPreview.value) URL.revokeObjectURL(originalPreview.value)
})
</script>

<template>
  <main class="app-shell">
    <header class="topbar">
      <div class="brand-block">
        <div class="brand-mark" aria-hidden="true"><Sparkles :size="19" /></div>
        <div><p class="eyebrow">Tiktok TechJam 2026</p><h1>AIGI Detect Demo</h1></div>
      </div>
      <div class="header-controls">
        <div class="language-switch" :aria-label="copy.languageSelector">
          <button :class="{ active: language === 'zh' }" :aria-pressed="language === 'zh'" lang="zh-CN" @click="language = 'zh'">中文</button>
          <button :class="{ active: language === 'en' }" :aria-pressed="language === 'en'" lang="en" @click="language = 'en'">EN</button>
        </div>
        <div class="model-switch" :aria-label="copy.modelSelector">
          <button :class="{ active: model === 'm2' }" :disabled="inferenceLoading" @click="model = 'm2'">M2</button>
          <button :class="{ active: model === 'm3' }" :disabled="inferenceLoading" @click="model = 'm3'">M3</button>
        </div>
        <div class="status-chip" :class="{ offline: healthError }"><span class="status-dot"></span>{{ serviceLabel }}</div>
      </div>
    </header>

    <section class="intro-strip">
      <div><Activity :size="16" /> {{ copy.activeModel }} <strong>{{ model.toUpperCase() }}</strong><span class="divider">/</span><Cpu :size="14" />{{ health?.device || copy.detecting }}</div>
      <span>{{ copy.intro }}</span>
    </section>

    <div v-if="errorMessage || healthError" class="global-error">
      <AlertCircle :size="17" /><div><strong>{{ healthError ? copy.localDisconnected : copy.analysisIncomplete }}</strong><span>{{ errorMessage || healthError }}</span></div>
    </div>

    <section class="section-card primary-section">
      <div class="section-heading">
        <span class="section-index">01</span>
        <div><p class="section-kicker">INPUT & RESULT</p><h2>{{ copy.sectionTitle }}</h2></div>
        <p class="section-summary">{{ copy.sectionSummary }}</p>
      </div>

      <div class="primary-grid">
        <label class="upload-surface" :class="{ dragging, compact: !!originalPreview }" @dragenter.prevent="dragging = true" @dragover.prevent @dragleave.prevent="dragging = false" @drop.prevent="onDrop">
          <input type="file" accept=".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.gif,image/*" :disabled="inferenceLoading" @change="selectFile(($event.target as HTMLInputElement).files?.[0])" />
          <img v-if="originalPreview" :src="result?.source_preview || originalPreview" :alt="copy.sourceAlt" />
          <div v-else class="upload-copy"><div class="upload-icon"><ImageUp :size="26" /></div><strong>{{ copy.dropImage }}</strong><span>JPEG · PNG · WebP · BMP · TIFF · GIF</span><em>{{ copy.selectImage }}</em></div>
          <div v-if="inferenceLoading" class="upload-loading"><LoaderCircle class="spin" :size="24" /><strong>{{ selectedHealth?.loaded ? copy.analysing : `${copy.loadingModel} · ${model.toUpperCase()}` }}</strong><span>{{ copy.firstLoad }}</span></div>
          <p v-if="selectedFile" class="file-name">ORIGINAL · {{ selectedFile.name }}</p>
        </label>

        <div class="result-surface" :class="{ populated: result }">
          <div v-if="!result" class="result-empty"><div class="result-orbit"><ShieldCheck :size="27" /></div><p>{{ inferenceLoading ? copy.analysing : copy.waiting }}</p><span>{{ copy.localOnly }}</span></div>
          <template v-else>
            <div class="result-topline"><div><span>{{ copy.modelInput }}</span><strong>{{ result.model.toUpperCase() }} / {{ result.device.toUpperCase() }}</strong></div><img :src="result.preview" :alt="copy.modelAlt" /></div>
            <div class="result-hero" :class="result.label.toLowerCase()"><div><span>{{ copy.resultLabel }}</span><strong>{{ result.label }}</strong></div><p>{{ result.label === 'AIGC' ? copy.crossed : copy.stayedBelow }}</p></div>
            <div class="metric-grid">
              <article>
                <span>{{ copy.rawScore }}</span>
                <strong>{{ formatPercent(result.raw_score) }}</strong>
                <em class="metric-threshold" :title="copy.rawThresholdHelp">
                  <span>{{ copy.rawThreshold }}</span><b>AIGC ≥ {{ formatPercent(result.raw_threshold) }}</b>
                </em>
              </article>
              <article>
                <span>{{ copy.calibratedScore }}</span>
                <strong>{{ formatPercent(result.calibrated_score) }}</strong>
                <em class="metric-threshold calibrated" :title="copy.calibratedThresholdHelp">
                  <span>{{ copy.calibratedThreshold }}</span><b>AIGC ≥ {{ formatPercent(result.calibrated_threshold) }}</b>
                </em>
              </article>
              <article><span>{{ copy.scoreUncertainty }}</span><strong>{{ formatPercent(result.uncertainty_entropy) }}</strong><em>{{ copy.entropy }}</em></article>
              <article><span>{{ copy.latency }}</span><strong>{{ result.timing_ms.toFixed(0) }} ms</strong><em>{{ result.checkpoint_sha256.slice(0, 8) }}… {{ copy.verified }}</em></article>
            </div>
            <div v-if="result.file" class="file-metadata"><span>{{ result.file.format }}</span><span>{{ result.file.width }}×{{ result.file.height }}</span><span>{{ formatBytes(result.file.bytes) }}</span><span v-if="result.file.animated_first_frame">{{ copy.firstFrame }}</span></div>
            <p class="calibration-note">{{ copy.calibrationNote }}</p>
          </template>
        </div>
      </div>
    </section>

    <RobustnessLab :locale="language" :controls="controls" :clean="result" :current="robustnessResult" :history="history" :loading="robustnessLoading" :disabled="!result || inferenceLoading" @change="setControls" @reset="resetControls" />
    <EvidencePanel :locale="language" :result="evidenceResult" />
    <BatchPanel :locale="language" :model="model" />
    <footer class="footer-line"><span>AIGI Detect Demo</span><em>{{ copy.footer }}</em></footer>
  </main>
</template>
