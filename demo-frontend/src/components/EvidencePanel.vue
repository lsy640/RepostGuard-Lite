<script setup lang="ts">
import { computed, ref } from 'vue'
import { AlertTriangle, Fingerprint, ScanSearch } from 'lucide-vue-next'
import type { InferenceResult, Language } from '../types'

const props = withDefaults(defineProps<{ result: InferenceResult | null; locale?: Language }>(), { locale: 'zh' })
const heatmapMode = ref<'color' | 'gray'>('color')
const pct = (value: number) => `${(value * 100).toFixed(1)}%`
const zh = computed(() => props.locale === 'zh')
const branchTitle = computed(() => props.result?.branch_evidence.kind === 'gate'
  ? (zh.value ? 'Quality-aware gate 权重' : 'Quality-aware gate weights')
  : (zh.value ? '分支消融贡献' : 'Branch ablation contribution'))
const evidenceNote = computed(() => props.result?.branch_evidence.kind === 'gate'
  ? (zh.value ? 'M3 gate 仅调节语义与取证分支，不将质量特征直接送入分类器。' : 'The M3 gate only balances semantic and forensic branches; quality features are not sent directly to the classifier.')
  : (zh.value ? 'M2 没有 quality gate；该比例来自移除分支后的 logit 变化，并不表示因果权重。' : 'M2 has no quality gate. These shares come from logit changes after branch removal and are not causal weights.'))
</script>

<template>
  <section class="section-card analysis-section">
    <div class="section-heading">
      <span class="section-index">03</span>
      <div><p class="section-kicker">EVIDENCE & LIMITS</p><h2>{{ zh ? '证据与限制' : 'Evidence & Limits' }}</h2></div>
      <div class="mode-switch" :aria-label="zh ? '热图显示模式' : 'Heatmap display mode'">
        <button :class="{ active: heatmapMode === 'color' }" @click="heatmapMode = 'color'">{{ zh ? '彩色' : 'Color' }}</button>
        <button :class="{ active: heatmapMode === 'gray' }" @click="heatmapMode = 'gray'">{{ zh ? '原始' : 'Raw' }}</button>
      </div>
    </div>

    <div v-if="!result" class="empty-section"><ScanSearch :size="25" /><strong>{{ zh ? '等待取证证据' : 'Waiting for forensic evidence' }}</strong><span>{{ zh ? '上传图片后展示分支贡献、SRM 与 NPR 响应。' : 'Upload an image to view branch contributions and SRM/NPR responses.' }}</span></div>

    <template v-else>
      <div class="evidence-grid">
        <article class="branch-panel">
          <div class="panel-title"><Fingerprint :size="17" /><span>{{ branchTitle }}</span><em>{{ result.model.toUpperCase() }}</em></div>
          <div class="branch-row semantic"><div><span>{{ zh ? '语义分支' : 'Semantic branch' }}</span><strong>{{ pct(result.branch_evidence.semantic) }}</strong></div><div class="branch-track"><i :style="{ width: pct(result.branch_evidence.semantic) }"></i></div></div>
          <div class="branch-row forensic"><div><span>{{ zh ? '取证分支' : 'Forensic branch' }}</span><strong>{{ pct(result.branch_evidence.forensic) }}</strong></div><div class="branch-track"><i :style="{ width: pct(result.branch_evidence.forensic) }"></i></div></div>
          <div v-if="result.branch_evidence.kind === 'ablation'" class="effect-grid">
            <span>{{ zh ? '语义 logit 影响' : 'Semantic logit effect' }} <strong>{{ result.branch_evidence.semantic_logit_effect?.toFixed(4) }}</strong></span>
            <span>{{ zh ? '取证 logit 影响' : 'Forensic logit effect' }} <strong>{{ result.branch_evidence.forensic_logit_effect?.toFixed(4) }}</strong></span>
          </div>
          <p class="evidence-note">{{ evidenceNote }}</p>
        </article>

        <article class="heatmap-card"><div class="heatmap-title"><span>SRM</span><em>{{ zh ? '30 滤波器残差响应' : '30-filter residual response' }}</em></div><img :src="heatmapMode === 'color' ? result.heatmaps.srm_color : result.heatmaps.srm_gray" :alt="zh ? 'SRM 取证响应热图' : 'SRM forensic response heatmap'" /></article>
        <article class="heatmap-card"><div class="heatmap-title"><span>NPR</span><em>{{ zh ? '最近邻重建残差' : 'Nearest-neighbor residual' }}</em></div><img :src="heatmapMode === 'color' ? result.heatmaps.npr_color : result.heatmaps.npr_gray" :alt="zh ? 'NPR 取证响应热图' : 'NPR forensic response heatmap'" /></article>
      </div>

      <div class="limitations">
        <div class="limits-icon"><AlertTriangle :size="19" /></div>
        <div><strong>{{ zh ? '结合内容溯源解释，而不是替代内容溯源。' : 'Interpret with provenance, not instead of provenance.' }}</strong><p>{{ zh ? '这是概率信号，不是来源证明。重压缩、模糊、裁剪、低照噪声、未知生成器与分布外真实内容都可能造成误报或漏报。' : 'This is a probabilistic model signal, not proof of origin. Recompression, blur, cropping, low-light noise, unknown generators, and out-of-distribution real content can all cause false positives or false negatives.' }}</p><p>{{ zh ? '结果不能替代 C2PA、可信水印、元数据与发布链路等内容溯源；单图分数也不能证明 M2 或 M3 是通用检测器。' : 'The result cannot replace C2PA, trusted watermarks, metadata, publishing-chain evidence, or other provenance mechanisms. A single-image score cannot establish M2 or M3 as a universal detector.' }}</p></div>
      </div>
    </template>
  </section>
</template>
