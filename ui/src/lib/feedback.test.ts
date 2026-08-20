import { describe, expect, it } from 'vitest';

import type { FeedbackItem } from './api/types.gen';
import { mirrorState, substitutionAnchor } from './feedback';

function item(partial: Partial<FeedbackItem>): FeedbackItem {
  return {
    id: 'x',
    kind: 'retry',
    severity: 'warning',
    confidence: 1,
    message_key: 'k',
    message: 'm',
    evidence: {},
    ...partial,
  };
}

describe('substitutionAnchor', () => {
  it('returns the anchored phone of a substitution verdict', () => {
    const feedback = [
      item({
        kind: 'segmental_substitution',
        anchor: { phone_index: 2, interval_s: [0.1, 0.2] },
      }),
    ];
    expect(substitutionAnchor(feedback)).toBe(2);
  });

  it('ignores retry items and anchor-less items', () => {
    const feedback = [
      item({ kind: 'retry' }),
      item({ kind: 'segmental_substitution', anchor: null }),
      item({ kind: 'segmental_substitution', anchor: {} }),
    ];
    expect(substitutionAnchor(feedback)).toBeNull();
  });

  it('returns null for an empty list', () => {
    expect(substitutionAnchor([])).toBeNull();
  });
});

describe('mirrorState', () => {
  it('returns null when no mirror item exists', () => {
    expect(mirrorState([item({ kind: 'retry' })])).toBeNull();
  });

  it('reads the heard-other state from the message', () => {
    const feedback = [
      item({
        kind: 'phone_hearing',
        message_key: 'feedback.phone_hearing_other',
        message: 'I heard /z/ where you were going for /ð/.',
        anchor: { phone_index: 0, interval_s: [0.1, 0.2] },
      }),
    ];
    expect(mirrorState(feedback)).toEqual({
      phoneIndex: 0,
      heard: 'z',
      onTarget: false,
      unsure: false,
    });
  });

  it('marks heard-as-intended and refuses to invent a heard phone', () => {
    const feedback = [
      item({
        kind: 'phone_hearing',
        message_key: 'feedback.phone_hearing_on_target',
        message: 'I heard /ð/ — right on target.',
        anchor: { phone_index: 1 },
      }),
    ];
    expect(mirrorState(feedback)).toEqual({
      phoneIndex: 1,
      heard: null,
      onTarget: true,
      unsure: false,
    });
  });

  it('marks the couldnt-tell refusal', () => {
    const feedback = [
      item({
        kind: 'phone_hearing',
        message_key: 'feedback.phone_hearing_unsure',
        message: "I couldn't tell what that /ð/ sounded like — try once more, a little slower.",
        anchor: { phone_index: 0 },
      }),
    ];
    expect(mirrorState(feedback)).toEqual({
      phoneIndex: 0,
      heard: null,
      onTarget: false,
      unsure: true,
    });
  });
});
