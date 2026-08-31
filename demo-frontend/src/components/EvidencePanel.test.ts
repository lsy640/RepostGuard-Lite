import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import EvidencePanel from './EvidencePanel.vue'
import type { InferenceResult } from '../types'

function result(kind: 'gate' | 'ablation'): InferenceResult {
  const pixel = 'data:image/png;base64,iVBORw0KGgo='
  return {
    image_id: 'image',
    model: kind === 'gate' ? 'm3' : 'm2',
    checkpoint_sha256: 'a'.repeat(64),
    raw_score: 0.2,
    calibrated_score: 0.48,
    raw_threshold: 0.99,
    calibrated_threshold: 0.59,
    label: 'Real',
    uncertainty_entropy: 0.9,
    branch_evidence: {
      kind,
      title: kind === 'gate' ? 'Quality-aware gate 权重' : '分支消融贡献',
      semantic: 0.4,
      forensic: 0.6,
      low_signal: false,
      note: 'evidence note',
    },
    heatmaps: { srm_color: pixel, srm_gray: pixel, npr_color: pixel, npr_gray: pixel },
    preview: pixel,
    timing_ms: 10,
    device: 'cpu',
    parameters: { total: 1, trainable: 1, frozen: 0 },
  }
}

describe('EvidencePanel', () => {
  it('does not confuse M2 ablation with the M3 quality gate', () => {
    expect(mount(EvidencePanel, { props: { result: result('ablation') } }).text()).toContain('分支消融贡献')
    expect(mount(EvidencePanel, { props: { result: result('gate') } }).text()).toContain('Quality-aware gate 权重')
  })
})
