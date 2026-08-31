import { mount } from '@vue/test-utils'
import { reactive } from 'vue'
import { describe, expect, it } from 'vitest'
import RobustnessLab from './RobustnessLab.vue'
import type { InferenceResult, TransformControls } from '../types'

const controls = (): TransformControls => ({
  jpeg: { enabled: false, quality: 70 },
  blur: { enabled: false, sigma: 1 },
  resize: { enabled: false, scale: 0.5 },
  noise: { enabled: false, sigma: 0.02 },
  jitter: { enabled: false, brightness: 1.2, contrast: 1.2, saturation: 1.2 },
  crop: { enabled: false, ratio: 0.8 },
})

const cleanResult = (): InferenceResult => ({
  image_id: 'image-id',
  model: 'm2',
  checkpoint_sha256: 'a'.repeat(64),
  raw_score: 0.9324,
  calibrated_score: 0.55,
  raw_threshold: 0.996,
  calibrated_threshold: 0.596,
  label: 'Real',
  uncertainty_entropy: 0.9,
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
  timing_ms: 1,
  device: 'cpu',
  parameters: { total: 1, trainable: 1, frozen: 0 },
})

describe('RobustnessLab', () => {
  it('renders the clean and perturbed labels as prominent state badges', () => {
    const wrapper = mount(RobustnessLab, {
      props: {
        controls: controls(),
        clean: cleanResult(),
        current: null,
        history: [0.9324],
        loading: false,
        disabled: false,
      },
    })

    const badges = wrapper.findAll('.prediction-badge')
    expect(badges).toHaveLength(2)
    expect(badges.every((badge) => badge.classes().includes('real'))).toBe(true)
    expect(badges.every((badge) => badge.text() === 'Real')).toBe(true)
  })

  it('adds localized response-index and raw-score axes to the history chart', async () => {
    const wrapper = mount(RobustnessLab, {
      props: {
        locale: 'en',
        controls: controls(),
        clean: cleanResult(),
        current: null,
        history: [0.91, 0.92, 0.915],
        loading: false,
        disabled: false,
      },
    })

    expect(wrapper.find('.x-title').text()).toBe('RESPONSE INDEX')
    expect(wrapper.find('.y-title').text()).toBe('RAW SCORE (%)')
    expect(wrapper.findAll('.y-tick')).toHaveLength(3)
    expect(wrapper.findAll('.latest-point')).toHaveLength(1)
    expect(wrapper.find('.aigc-zone-label').text()).toBe('AIGC ↑')
    expect(wrapper.find('.real-zone-label').text()).toBe('REAL ↓')
    expect(wrapper.find('.threshold-label').text()).toContain('threshold')
    expect(wrapper.find('polyline').attributes('points')).not.toBe('')

    await wrapper.setProps({ locale: 'zh' })
    expect(wrapper.find('.x-title').text()).toBe('响应序号')
    expect(wrapper.find('.y-title').text()).toBe('原始分数 (%)')
    expect(wrapper.find('.threshold-label').text()).toContain('阈值')
  })

  it('emits a plain control snapshot when a reactive prop is changed', async () => {
    const reactiveControls = reactive(controls())
    const wrapper = mount(RobustnessLab, {
      props: {
        controls: reactiveControls,
        clean: cleanResult(),
        current: null,
        history: [0.9324],
        loading: false,
        disabled: false,
      },
    })

    await wrapper.find('input[type="checkbox"]').setValue(true)

    const emitted = wrapper.emitted('change')
    expect(emitted).toHaveLength(1)
    expect((emitted?.[0][0] as TransformControls).jpeg.enabled).toBe(true)
    expect(reactiveControls.jpeg.enabled).toBe(false)
  })
})
