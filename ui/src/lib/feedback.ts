// Feedback-to-highlight mapping: the one rule a rich client needs to point
// the user's eyes at the segment a verdict is about.
import type { FeedbackItem } from './api/types.gen';

/**
 * The phone index a substitution verdict anchors to, or null.
 *
 * The composer only anchors segmental_substitution items (retry items carry no
 * anchor: they report the engine's state, not a claim about any segment).
 */
export function substitutionAnchor(feedback: FeedbackItem[]): number | null {
  const item = feedback.find(
    (f) => f.kind === 'segmental_substitution' && f.anchor?.phone_index != null,
  );
  return item?.anchor?.phone_index ?? null;
}
