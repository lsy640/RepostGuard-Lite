<script setup lang="ts">
import { computed } from 'vue'
import { RefreshCcw, SlidersHorizontal } from 'lucide-vue-next'
import { cloneTransformControls } from '../controls'
import type { InferenceResult, Language, TransformControls } from '../types'

const props = withDefaults(defineProps<{
  locale?: Language
  controls: TransformControls
  clean: InferenceResult | null
  current: InferenceResult | null
  history: number[]
  loading: boolean
  disabled: boolean
}>(), { locale: 'zh' })

const emit = defineEmits<{
  change: [controls: TransformControls]
  reset: []
}>()

const zh = computed(() => props.locale === 'zh')

const activeOrder = computed(() => {
  const names: string[] = []
  if (props.controls.crop.enabled) names.push(zh.value ? '裁剪' : 'crop')
  if (props.controls.resize.enabled) names.push(zh.value ? '缩放' : 'resize')
  if (props.controls.jitter.enabled) names.push(zh.value ? '色彩扰动' : 'jitter')
  if (props.controls.blur.enabled) names.push(zh.value ? '模糊' : 'blur')
  if (props.controls.noise.enabled) names.push(zh.value ? '噪声' : 'noise')
  if (props.controls.jpeg.enabled) names.push('JPEG')
  return names.length ? names.join(' → ') : (zh.value ? '干净输入 · 未启用扰动' : 'Clean input · no transform enabled')
})

const scoreDelta = computed(() => {
  if (!props.clean || !props.current) return null
  return props.current.raw_score - props.clean.raw_score
})

const historyChart = computed(() => {
  const left = 50
  const right = 348
  const top = 10
  const bottom = 70
  if (!props.history.length) {
    return {
      points: '',
      xTicks: [],
      yTicks: [],
      lastPoint: null,
      thresholdY: null,
      thresholdLabel: '',
      aigcLabelY: 0,
      realLabelY: 0,
    }
  }

  const threshold = props.clean?.raw_threshold ?? 0.5
  const domainValues = [...props.history, threshold]
  const rawMin = Math.min(...domainValues)
  const rawMax = Math.max(...domainValues)
  const padding = Math.max((rawMax - rawMin) * 0.2, 0.005)
  let domainMin = Math.max(0, rawMin - padding)
  let domainMax = Math.min(1, rawMax + padding)
  if (domainMax - domainMin < 0.001) {
    domainMin = Math.max(0, domainMin - 0.005)
    domainMax = Math.min(1, domainMax + 0.005)
  }
  const domainSpan = Math.max(domainMax - domainMin, 0.001)
  const xFor = (index: number) =>
    props.history.length === 1
      ? (left + right) / 2
      : left + (index / (props.history.length - 1)) * (right - left)
  const yFor = (value: number) => top + ((domainMax - value) / domainSpan) * (bottom - top)
  const coordinates = props.history.map((value, index) => ({
    x: xFor(index),
    y: yFor(value),
  }))
  const thresholdY = yFor(threshold)
  const xIndexes = Array.from(new Set([0, Math.floor((props.history.length - 1) / 2), props.history.length - 1]))

  return {
    points: coordinates.map(({ x, y }) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' '),
    xTicks: xIndexes.map((index) => ({ x: xFor(index), label: String(index + 1) })),
    yTicks: [domainMax, (domainMax + domainMin) / 2, domainMin].map((value) => ({
      y: yFor(value),
      label: `${(value * 100).toFixed(1)}%`,
    })),
    lastPoint: coordinates.at(-1) ?? null,
    thresholdY,
    thresholdLabel: `${(threshold * 100).toFixed(2)}% ${zh.value ? '阈值' : 'threshold'}`,
    aigcLabelY: (top + thresholdY) / 2,
    realLabelY: (thresholdY + bottom) / 2,
  }
})

function update(section: keyof TransformControls, field: string, value: boolean | number) {
  const next = cloneTransformControls(props.controls)
  ;(next[section] as unknown as Record<string, boolean | number>)[field] = value
  emit('change', next)
}

function percent(value: number | null | undefined) {
  return value == null ? '—' : `${(value * 100).toFixed(2)}%`
}
</script>

<template>
  <section class="section-card analysis-section" :class="{ muted: disabled }">
    <div class="section-heading">
      <span class="section-index">02</span>
      <div><p class="section-kicker">ROBUSTNESS LAB</p><h2>{{ zh ? '鲁棒性实验台' : 'Robustness Lab' }}</h2></div>
      <button class="ghost-button" :disabled="disabled" @click="emit('reset')"><RefreshCcw :size="14" /> {{ zh ? '全部重置' : 'Reset all' }}</button>
    </div>

    <div v-if="!clean" class="empty-section">
      <SlidersHorizontal :size="25" /><strong>{{ zh ? '上传图片后启用实验台' : 'Upload an image to enable the lab' }}</strong><span>{{ zh ? '滑块将在本地重建社交媒体压缩与重采样条件。' : 'The controls reproduce social-media compression and resampling conditions locally.' }}</span>
    </div>

    <template v-else>
      <div class="robustness-images">
        <figure>
          <span class="figure-label">{{ zh ? '干净输入' : 'CLEAN' }} · FORMAT-DEBIAS Q90</span>
          <img :src="clean.preview" :alt="zh ? '干净模型输入' : 'Clean model input'" />
          <figcaption>
            <span>{{ zh ? '原始分数' : 'Raw' }} {{ percent(clean.raw_score) }}</span>
            <strong class="prediction-badge" :class="clean.label.toLowerCase()">{{ clean.label }}</strong>
          </figcaption>
        </figure>
        <figure>
          <span class="figure-label">{{ zh ? '扰动输入 · 实时' : 'PERTURBED · LIVE' }}</span>
          <div class="image-loading-wrap">
            <img :src="current?.preview || clean.preview" :alt="zh ? '扰动后模型输入' : 'Perturbed model input'" />
            <div v-if="loading" class="image-loading">{{ zh ? '分析中' : 'ANALYSING' }}</div>
          </div>
          <figcaption>
            <span>{{ zh ? '原始分数' : 'Raw' }} {{ percent(current?.raw_score ?? clean.raw_score) }}</span>
            <strong class="prediction-badge" :class="(current?.label || clean.label).toLowerCase()">{{ current?.label || clean.label }}</strong>
          </figcaption>
        </figure>
      </div>

      <div class="order-strip"><span>{{ zh ? '执行顺序' : 'EXECUTION ORDER' }}</span><strong>{{ activeOrder }}</strong></div>

      <div class="transform-grid">
        <article class="control-card">
          <header><label><input type="checkbox" :checked="controls.jpeg.enabled" :disabled="disabled" @change="update('jpeg', 'enabled', ($event.target as HTMLInputElement).checked)" /> {{ zh ? 'JPEG 压缩' : 'JPEG Compression' }}</label><code>Q {{ controls.jpeg.quality }}</code></header>
          <input :aria-label="zh ? 'JPEG 质量' : 'JPEG quality'" type="range" min="30" max="100" step="1" :value="controls.jpeg.quality" :disabled="disabled || !controls.jpeg.enabled" @input="update('jpeg', 'quality', Number(($event.target as HTMLInputElement).value))" />
          <div class="ticks"><span>30</span><span>50</span><span>70</span><span>90</span><span>100</span></div><p>{{ zh ? '社交媒体重编码 · 即时通信' : 'Social-media re-encode · messaging' }}</p>
        </article>
        <article class="control-card">
          <header><label><input type="checkbox" :checked="controls.blur.enabled" :disabled="disabled" @change="update('blur', 'enabled', ($event.target as HTMLInputElement).checked)" /> {{ zh ? '高斯模糊' : 'Gaussian Blur' }}</label><code>σ {{ controls.blur.sigma.toFixed(1) }}</code></header>
          <input :aria-label="zh ? '高斯模糊 sigma' : 'Gaussian blur sigma'" type="range" min="0" max="2.5" step="0.1" :value="controls.blur.sigma" :disabled="disabled || !controls.blur.enabled" @input="update('blur', 'sigma', Number(($event.target as HTMLInputElement).value))" />
          <div class="ticks"><span>0</span><span>.5</span><span>1.0</span><span>2.0</span><span>2.5</span></div><p>{{ zh ? '失焦 · 柔化' : 'Out-of-focus · softening' }}</p>
        </article>
        <article class="control-card">
          <header><label><input type="checkbox" :checked="controls.resize.enabled" :disabled="disabled" @change="update('resize', 'enabled', ($event.target as HTMLInputElement).checked)" /> {{ zh ? '缩放往返' : 'Resize Roundtrip' }}</label><code>{{ controls.resize.scale.toFixed(2) }}×</code></header>
          <input :aria-label="zh ? '缩放比例' : 'Resize scale'" type="range" min="0.25" max="1" step="0.05" :value="controls.resize.scale" :disabled="disabled || !controls.resize.enabled" @input="update('resize', 'scale', Number(($event.target as HTMLInputElement).value))" />
          <div class="ticks"><span>.25×</span><span>.5×</span><span>.75×</span><span>1×</span></div><p>{{ zh ? '缩略图生成 · 放大恢复' : 'Thumbnail generation · upscale' }}</p>
        </article>
        <article class="control-card">
          <header><label><input type="checkbox" :checked="controls.noise.enabled" :disabled="disabled" @change="update('noise', 'enabled', ($event.target as HTMLInputElement).checked)" /> {{ zh ? '高斯噪声' : 'Gaussian Noise' }}</label><code>σ {{ controls.noise.sigma.toFixed(3) }}</code></header>
          <input :aria-label="zh ? '高斯噪声 sigma' : 'Gaussian noise sigma'" type="range" min="0" max="0.1" step="0.005" :value="controls.noise.sigma" :disabled="disabled || !controls.noise.enabled" @input="update('noise', 'sigma', Number(($event.target as HTMLInputElement).value))" />
          <div class="ticks"><span>0</span><span>.02</span><span>.05</span><span>.10</span></div><p>{{ zh ? '低照传感器噪声 · 固定种子 20260827' : 'Low-light sensor noise · seed 20260827' }}</p>
        </article>
        <article class="control-card jitter-card">
          <header><label><input type="checkbox" :checked="controls.jitter.enabled" :disabled="disabled" @change="update('jitter', 'enabled', ($event.target as HTMLInputElement).checked)" /> {{ zh ? '色彩扰动' : 'Color Jitter' }}</label><code>±20%</code></header>
          <div class="mini-range"><span>B</span><input :aria-label="zh ? '亮度' : 'Brightness'" type="range" min="0.8" max="1.2" step="0.01" :value="controls.jitter.brightness" :disabled="disabled || !controls.jitter.enabled" @input="update('jitter', 'brightness', Number(($event.target as HTMLInputElement).value))" /><code>{{ controls.jitter.brightness.toFixed(2) }}</code></div>
          <div class="mini-range"><span>C</span><input :aria-label="zh ? '对比度' : 'Contrast'" type="range" min="0.8" max="1.2" step="0.01" :value="controls.jitter.contrast" :disabled="disabled || !controls.jitter.enabled" @input="update('jitter', 'contrast', Number(($event.target as HTMLInputElement).value))" /><code>{{ controls.jitter.contrast.toFixed(2) }}</code></div>
          <div class="mini-range"><span>S</span><input :aria-label="zh ? '饱和度' : 'Saturation'" type="range" min="0.8" max="1.2" step="0.01" :value="controls.jitter.saturation" :disabled="disabled || !controls.jitter.enabled" @input="update('jitter', 'saturation', Number(($event.target as HTMLInputElement).value))" /><code>{{ controls.jitter.saturation.toFixed(2) }}</code></div>
          <p>{{ zh ? '滤镜应用 · 自动增强' : 'Filter apps · auto-enhance' }}</p>
        </article>
        <article class="control-card">
          <header><label><input type="checkbox" :checked="controls.crop.enabled" :disabled="disabled" @change="update('crop', 'enabled', ($event.target as HTMLInputElement).checked)" /> {{ zh ? '中心裁剪' : 'Center Crop' }}</label><code>{{ Math.round(controls.crop.ratio * 100) }}%</code></header>
          <input :aria-label="zh ? '中心裁剪比例' : 'Center crop ratio'" type="range" min="0.75" max="1" step="0.01" :value="controls.crop.ratio" :disabled="disabled || !controls.crop.enabled" @input="update('crop', 'ratio', Number(($event.target as HTMLInputElement).value))" />
          <div class="ticks"><span>75%</span><span>80%</span><span>90%</span><span>100%</span></div><p>{{ zh ? '头像裁剪 · 构图' : 'Profile-picture crop · framing' }}</p>
        </article>
      </div>

      <div class="stability-row">
        <div><span>{{ zh ? '原始分数变化' : 'RAW SCORE Δ' }}</span><strong :class="{ danger: scoreDelta && Math.abs(scoreDelta) > 0.1 }">{{ scoreDelta == null ? '—' : `${scoreDelta >= 0 ? '+' : ''}${(scoreDelta * 100).toFixed(2)} pp` }}</strong></div>
        <div><span>{{ zh ? '标签翻转' : 'LABEL FLIP' }}</span><strong>{{ current && clean.label !== current.label ? (zh ? '是' : 'YES') : (zh ? '否' : 'NO') }}</strong></div>
        <div><span>{{ zh ? '历史范围' : 'HISTORY RANGE' }}</span><strong>{{ history.length ? `${((Math.max(...history) - Math.min(...history)) * 100).toFixed(2)} pp` : '—' }}</strong></div>
        <svg class="sparkline" viewBox="0 0 360 100" role="img" :aria-label="zh ? '最近二十次扰动原始分数变化，横轴为响应序号，纵轴为原始分数百分比' : 'Raw score changes over the latest twenty perturbation responses; x-axis is response index and y-axis is raw score percentage'">
          <template v-if="historyChart.thresholdY != null">
            <rect class="score-zone-bg aigc-zone" x="50" y="10" width="298" :height="Math.max(historyChart.thresholdY - 10, 0)" />
            <rect class="score-zone-bg real-zone" x="50" :y="historyChart.thresholdY" width="298" :height="Math.max(70 - historyChart.thresholdY, 0)" />
          </template>
          <g v-for="tick in historyChart.yTicks" :key="`y-${tick.label}`">
            <line class="chart-grid" x1="50" :y1="tick.y" x2="348" :y2="tick.y" />
            <text class="chart-tick y-tick" x="43" :y="tick.y + 3" text-anchor="end">{{ tick.label }}</text>
          </g>
          <line class="chart-axis" x1="50" y1="10" x2="50" y2="70" />
          <line class="chart-axis" x1="50" y1="70" x2="348" y2="70" />
          <template v-if="historyChart.thresholdY != null">
            <line class="threshold-line" x1="50" :y1="historyChart.thresholdY" x2="348" :y2="historyChart.thresholdY" />
            <text class="threshold-label" x="54" :y="historyChart.thresholdY - 3">{{ historyChart.thresholdLabel }}</text>
            <text class="score-zone-label aigc-zone-label" x="342" :y="historyChart.aigcLabelY + 2" text-anchor="end">AIGC ↑</text>
            <text class="score-zone-label real-zone-label" x="342" :y="historyChart.realLabelY + 2" text-anchor="end">REAL ↓</text>
          </template>
          <g v-for="tick in historyChart.xTicks" :key="`x-${tick.label}`">
            <line class="chart-axis-tick" :x1="tick.x" y1="70" :x2="tick.x" y2="74" />
            <text class="chart-tick" :x="tick.x" y="83" text-anchor="middle">{{ tick.label }}</text>
          </g>
          <text class="chart-title x-title" x="199" y="96" text-anchor="middle">{{ zh ? '响应序号' : 'RESPONSE INDEX' }}</text>
          <text class="chart-title y-title" x="11" y="40" text-anchor="middle" transform="rotate(-90 11 40)">{{ zh ? '原始分数 (%)' : 'RAW SCORE (%)' }}</text>
          <polyline v-if="historyChart.points" :points="historyChart.points" />
          <circle v-if="historyChart.lastPoint" class="latest-point" :cx="historyChart.lastPoint.x" :cy="historyChart.lastPoint.y" r="3.5" />
        </svg>
      </div>
    </template>
  </section>
</template>
