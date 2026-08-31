import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App.vue'
import type { InferenceResult } from './types'

const health = {
  status: 'ok',
  device: 'cpu',
  mps_built: true,
  mps_available: false,
  models: {
    m2: {
      available: true,
      loaded: false,
      verification: 'pending',
      error: null,
      checkpoint_sha256: 'm2',
      config_sha256: 'm2-config',
      threshold: 0.99658203125,
      clean_auroc: 0.978116,
      train_version: 'v3',
      calibration: { available: true, temperature: 14.5, calibrated_threshold: 0.5966, reason: null, samples: 2000, views: 4 },
    },
    m3: {
      available: true,
      loaded: false,
      verification: 'pending',
      error: null,
      checkpoint_sha256: 'm3',
      config_sha256: 'm3-config',
      threshold: 0.9970703125,
      clean_auroc: 0.9782205,
      train_version: 'v3',
      calibration: { available: true, temperature: 14.52, calibrated_threshold: 0.599, reason: null, samples: 2000, views: 4 },
    },
  },
}

const inferenceResult: InferenceResult = {
  image_id: 'image-id',
  file: { name: 'sample.png', width: 512, height: 512, format: 'PNG', bytes: 1024, animated_first_frame: false },
  model: 'm2',
  checkpoint_sha256: 'a'.repeat(64),
  raw_score: 0.9997,
  calibrated_score: 0.6376,
  raw_threshold: 0.99658203125,
  calibrated_threshold: 0.5966089452,
  label: 'AIGC',
  uncertainty_entropy: 0.9447,
  branch_evidence: {
    kind: 'ablation',
    title: '分支消融贡献',
    semantic: 0.5,
    forensic: 0.5,
    low_signal: false,
    note: 'test',
  },
  heatmaps: {
    srm_color: 'data:image/png;base64,',
    srm_gray: 'data:image/png;base64,',
    npr_color: 'data:image/png;base64,',
    npr_gray: 'data:image/png;base64,',
  },
  preview: 'data:image/jpeg;base64,',
  timing_ms: 61,
  device: 'cpu',
  parameters: { total: 1, trainable: 1, frozen: 0 },
}

afterEach(() => vi.unstubAllGlobals())

describe('AIGI Detect Demo', () => {
  it('renders the requested four work regions with M2 selected by default', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => health }))
    const wrapper = mount(App)
    await Promise.resolve()
    expect(wrapper.text()).toContain('AIGI Detect Demo')
    expect(wrapper.text()).toContain('Tiktok TechJam 2026')
    expect(wrapper.text()).toContain('输入与结果')
    expect(wrapper.text()).toContain('鲁棒性实验台')
    expect(wrapper.text()).toContain('证据与限制')
    expect(wrapper.text()).toContain('批量导入与标准 JSON')
    const modelButtons = wrapper.findAll('.model-switch button')
    expect(modelButtons[0].classes()).toContain('active')
    wrapper.unmount()
  })

  it('switches all four work regions from Chinese to English without reloading', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => health }))
    const wrapper = mount(App)
    await flushPromises()
    const languageButtons = wrapper.findAll('.language-switch button')
    expect(languageButtons[0].classes()).toContain('active')

    await languageButtons[1].trigger('click')

    expect(languageButtons[1].classes()).toContain('active')
    expect(wrapper.text()).toContain('Input & Result')
    expect(wrapper.text()).toContain('Robustness Lab')
    expect(wrapper.text()).toContain('Evidence & Limits')
    expect(wrapper.text()).toContain('Batch Import & Standard JSON')
    expect(wrapper.text()).toContain('Select image')
    expect(document.documentElement.lang).toBe('en')
    wrapper.unmount()
  })

  it('shows explicit raw and calibrated AIGC threshold directions', async () => {
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:sample'),
      revokeObjectURL: vi.fn(),
    })
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (input: RequestInfo | URL) => ({
      ok: true,
      json: async () => String(input) === '/api/infer' ? inferenceResult : health,
    })))
    const wrapper = mount(App)
    const input = wrapper.find<HTMLInputElement>('input[type="file"]')
    const file = new File(['image'], 'sample.png', { type: 'image/png' })
    Object.defineProperty(input.element, 'files', { value: [file] })
    await input.trigger('change')
    await flushPromises()

    const thresholds = wrapper.findAll('.metric-threshold')
    expect(thresholds).toHaveLength(2)
    expect(thresholds[0].text()).toContain('分类阈值')
    expect(thresholds[0].text()).toContain('AIGC ≥ 99.66%')
    expect(thresholds[1].text()).toContain('校准阈值')
    expect(thresholds[1].text()).toContain('AIGC ≥ 59.66%')
    wrapper.unmount()
  })
})
