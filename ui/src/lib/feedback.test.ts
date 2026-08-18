import { describe, expect, it } from 'vitest';

import type { FeedbackItem } from './api/types.gen';
import { substitutionAnchor } from './feedback';

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
