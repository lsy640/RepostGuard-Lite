<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import { CheckCircle2, Download, Files, FolderInput, LoaderCircle, UploadCloud } from 'lucide-vue-next'
import { createBatch, downloadBatch, getBatch } from '../api'
import type { BatchJob, Language, ModelName } from '../types'

const props = withDefaults(defineProps<{ model: ModelName; locale?: Language }>(), { locale: 'zh' })
const selectedFiles = ref<File[]>([])
const selectedPaths = ref<string[]>([])
const selectionErrorCode = ref<'' | 'limit' | 'duplicate'>('')
const actionError = ref('')
const job = ref<BatchJob | null>(null)
const submitting = ref(false)
let timer: number | undefined

const progress = computed(() => (!job.value?.total ? 0 : Math.round((job.value.processed / job.value.total) * 100)))
const zh = computed(() => props.locale === 'zh')
const selectionError = computed(() => selectionErrorCode.value === 'limit'
  ? (zh.value ? '单批最多 100 张图片。' : 'A batch can contain at most 100 images.')
  : selectionErrorCode.value === 'duplicate'
    ? (zh.value ? '发现重复 image_path；请使用文件夹导入或移除同名文件。' : 'Duplicate image_path values found. Use folder import or remove files with the same basename.')
    : '')
const statusLabel = (status: BatchJob['status']) => zh.value
  ? ({ queued: '排队中', running: '处理中', complete: '已完成', failed: '失败' }[status])
  : status.toUpperCase()

function normalizeSelection(fileList: FileList | null, folder: boolean) {
  selectionErrorCode.value = ''
  actionError.value = ''
  job.value = null
  if (!fileList) return
  const files = Array.from(fileList)
  if (files.length > 100) {
    selectionErrorCode.value = 'limit'
    return
  }
  const paths = files.map((file) => {
    const relative = (file as File & { webkitRelativePath?: string }).webkitRelativePath
    return folder && relative ? relative.replaceAll('\\', '/') : file.name
  })
  if (new Set(paths).size !== paths.length) {
    selectionErrorCode.value = 'duplicate'
    return
  }
  selectedFiles.value = files
  selectedPaths.value = paths
}

async function start() {
  if (!selectedFiles.value.length || submitting.value) return
  actionError.value = ''
  submitting.value = true
  try {
    job.value = await createBatch(selectedFiles.value, selectedPaths.value, props.model)
    schedulePoll()
  } catch (error) {
    actionError.value = error instanceof Error ? error.message : String(error)
  } finally {
    submitting.value = false
  }
}

function schedulePoll() {
  window.clearTimeout(timer)
  timer = window.setTimeout(poll, 700)
}

async function poll() {
  if (!job.value || ['complete', 'failed'].includes(job.value.status)) return
  try {
    job.value = await getBatch(job.value.job_id)
    if (!['complete', 'failed'].includes(job.value.status)) schedulePoll()
  } catch (error) {
    actionError.value = error instanceof Error ? error.message : String(error)
  }
}

async function download() {
  if (!job.value || job.value.status !== 'complete') return
  try {
    await downloadBatch(job.value.job_id, job.value.model)
  } catch (error) {
    actionError.value = error instanceof Error ? error.message : String(error)
  }
}

onBeforeUnmount(() => window.clearTimeout(timer))
</script>

<template>
  <section class="section-card analysis-section batch-section">
    <div class="section-heading">
      <span class="section-index">04</span>
      <div><p class="section-kicker">BATCH JSON EXPORT</p><h2>{{ zh ? '批量导入与标准 JSON' : 'Batch Import & Standard JSON' }}</h2></div>
      <div class="schema-chip"><code>image_path</code><span>+</span><code>pred</code></div>
    </div>

    <div class="batch-layout">
      <div class="batch-input">
        <div class="batch-dropzones">
          <label><input type="file" multiple accept=".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.gif,image/*" @change="normalizeSelection(($event.target as HTMLInputElement).files, false)" /><Files :size="21" /><strong>{{ zh ? '多文件导入' : 'Import files' }}</strong><span>{{ zh ? '使用文件 basename' : 'Use file basenames' }}</span></label>
          <label><input type="file" multiple webkitdirectory directory @change="normalizeSelection(($event.target as HTMLInputElement).files, true)" /><FolderInput :size="21" /><strong>{{ zh ? '文件夹导入' : 'Import folder' }}</strong><span>{{ zh ? '保留相对路径' : 'Preserve relative paths' }}</span></label>
        </div>
        <div v-if="selectionError || actionError" class="inline-error">{{ selectionError || actionError }}</div>
        <div class="batch-selection"><span>{{ selectedFiles.length ? (zh ? `已选择 ${selectedFiles.length} 张图片` : `${selectedFiles.length} images selected`) : (zh ? '尚未选择文件' : 'No files selected') }}</span><em>{{ zh ? '任务模型：' : 'Job model:' }} {{ model.toUpperCase() }} · 20 MB / {{ zh ? '文件' : 'file' }} · {{ zh ? '最多 100 张' : '100 max' }}</em></div>
        <div v-if="selectedPaths.length" class="path-preview"><code v-for="path in selectedPaths.slice(0, 5)" :key="path">{{ path }}</code><span v-if="selectedPaths.length > 5">+ {{ selectedPaths.length - 5 }} {{ zh ? '项' : 'more' }}</span></div>
        <button class="primary-button" :disabled="!selectedFiles.length || submitting || !!selectionError" @click="start"><LoaderCircle v-if="submitting" class="spin" :size="16" /><UploadCloud v-else :size="16" />{{ submitting ? (zh ? '提交中…' : 'Submitting…') : (zh ? `使用 ${model.toUpperCase()} 开始批量推理` : `Start batch inference with ${model.toUpperCase()}`) }}</button>
      </div>

      <div class="batch-output">
        <div v-if="!job" class="batch-empty"><div class="json-brackets">[ ]</div><strong>{{ zh ? '标准评测 JSON' : 'Standard evaluation JSON' }}</strong><span>{{ zh ? '只输出成功图片的 image_path 与原始 pred，不混入诊断字段。' : 'Includes only image_path and raw pred for successful images, with no diagnostic fields.' }}</span></div>
        <template v-else>
          <div class="job-header"><div><span>{{ job.model.toUpperCase() }} · {{ statusLabel(job.status) }}</span><strong>{{ job.processed }} / {{ job.total }}</strong></div><CheckCircle2 v-if="job.status === 'complete'" :size="24" /><LoaderCircle v-else class="spin" :size="24" /></div>
          <div class="progress-track"><i :style="{ width: `${progress}%` }"></i></div>
          <div class="job-stats"><span>{{ zh ? '成功' : 'Success' }} <strong>{{ job.succeeded }}</strong></span><span>{{ zh ? '失败' : 'Failed' }} <strong>{{ job.failed }}</strong></span><span>{{ zh ? '进度' : 'Progress' }} <strong>{{ progress }}%</strong></span></div>
          <div v-if="job.results.length" class="json-preview"><div v-for="row in job.results.slice(0, 8)" :key="row.image_path"><code>"{{ row.image_path }}"</code><strong>{{ row.pred.toFixed(4) }}</strong></div></div>
          <div v-if="job.errors.length" class="batch-errors"><p v-for="item in job.errors.slice(0, 4)" :key="`${item.image_path}-${item.error_type}`"><code>{{ item.image_path }}</code> · {{ item.message }}</p></div>
          <button class="download-button" :disabled="job.status !== 'complete'" @click="download"><Download :size="15" /> {{ zh ? '下载 JSON' : 'Download JSON' }}</button>
        </template>
      </div>
    </div>
  </section>
</template>
