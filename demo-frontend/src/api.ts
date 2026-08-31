import type {
  BatchJob,
  HealthResponse,
  InferenceResult,
  ModelName,
  TransformControls,
} from './types'

async function request<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init)
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (payload.detail) detail = payload.detail
    } catch {
      // Preserve the HTTP status when the service did not return JSON.
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/api/health')
}

export function inferImage(file: File, model: ModelName, signal?: AbortSignal): Promise<InferenceResult> {
  const body = new FormData()
  body.append('image', file, file.name)
  body.append('model', model)
  return request<InferenceResult>('/api/infer', { method: 'POST', body, signal })
}

export function runRobustness(
  imageId: string,
  model: ModelName,
  controls: TransformControls,
  signal?: AbortSignal,
): Promise<InferenceResult> {
  return request<InferenceResult>(`/api/robustness/${imageId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, ...controls }),
    signal,
  })
}

export function createBatch(files: File[], paths: string[], model: ModelName): Promise<BatchJob> {
  const body = new FormData()
  files.forEach((file) => body.append('files', file, file.name))
  paths.forEach((path) => body.append('paths', path))
  body.append('model', model)
  return request<BatchJob>('/api/batches', { method: 'POST', body })
}

export function getBatch(jobId: string): Promise<BatchJob> {
  return request<BatchJob>(`/api/batches/${jobId}`)
}

export async function downloadBatch(jobId: string, model: ModelName): Promise<void> {
  const response = await fetch(`/api/batches/${jobId}/download`)
  if (!response.ok) throw new Error(`无法下载批量结果：${response.status}`)
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '-').slice(0, 15)
  anchor.href = url
  anchor.download = `aigi-detect-${model}-${stamp}.json`
  anchor.click()
  URL.revokeObjectURL(url)
}
