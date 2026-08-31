export type ModelName = 'm2' | 'm3'
export type Language = 'zh' | 'en'

export interface CalibrationStatus {
  available: boolean
  temperature: number | null
  calibrated_threshold: number | null
  reason: string | null
  samples: number | null
  views: number | null
}

export interface ModelHealth {
  available: boolean
  loaded: boolean
  verification: string
  error: string | null
  checkpoint_sha256: string
  config_sha256: string
  threshold: number
  clean_auroc: number
  train_version: string
  calibration: CalibrationStatus
}

export interface HealthResponse {
  status: string
  device: string
  mps_built: boolean
  mps_available: boolean
  models: Record<ModelName, ModelHealth>
}

export interface BranchEvidence {
  kind: 'gate' | 'ablation'
  title: string
  semantic: number
  forensic: number
  semantic_logit_effect?: number
  forensic_logit_effect?: number
  low_signal: boolean
  note: string
}

export interface Heatmaps {
  srm_color: string
  srm_gray: string
  npr_color: string
  npr_gray: string
}

export interface ImageFileMetadata {
  name: string
  width: number
  height: number
  format: string
  bytes: number
  animated_first_frame: boolean
}

export interface InferenceResult {
  image_id: string
  file?: ImageFileMetadata
  source_preview?: string
  model: ModelName
  checkpoint_sha256: string
  raw_score: number
  calibrated_score: number | null
  raw_threshold: number
  calibrated_threshold: number | null
  label: 'AIGC' | 'Real'
  uncertainty_entropy: number | null
  branch_evidence: BranchEvidence
  heatmaps: Heatmaps
  preview: string
  timing_ms: number
  device: string
  parameters: { total: number; trainable: number; frozen: number }
  applied_transforms?: Array<{ name: string; params: Record<string, string | number> }>
}

export interface TransformControls {
  jpeg: { enabled: boolean; quality: number }
  blur: { enabled: boolean; sigma: number }
  resize: { enabled: boolean; scale: number }
  noise: { enabled: boolean; sigma: number }
  jitter: {
    enabled: boolean
    brightness: number
    contrast: number
    saturation: number
  }
  crop: { enabled: boolean; ratio: number }
}

export interface BatchRecord {
  image_path: string
  pred: number
}

export interface BatchError {
  image_path: string
  status: string
  error_type: string
  message: string
}

export interface BatchJob {
  job_id: string
  model: ModelName
  status: 'queued' | 'running' | 'complete' | 'failed'
  total: number
  processed: number
  succeeded: number
  failed: number
  results: BatchRecord[]
  errors: BatchError[]
}
