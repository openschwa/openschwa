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

/**
 * The mirror (phone_hearing) item for the focus slot, or null.
 *
 * The mirror is the shipped M1 feedback: one item per focus slot in one of
 * three states - heard-as-intended (praise), heard-other (warning), or the
 * honest couldn't-tell refusal. It always carries an anchor; the timeline
 * renders the heard phone at that slot instead of the drill's expected label.
 */
export function mirrorItem(feedback: FeedbackItem[]): FeedbackItem | null {
  return feedback.find((f) => f.kind === 'phone_hearing' && f.anchor?.phone_index != null) ?? null;
}

/**
 * The mirror's state as the timeline consumes it: the anchored phone index,
 * the phone the ear heard (from the message key), and whether the ear
 * refused to report. Null when no mirror item exists (no committed hearing
 * calibration yet - the engine is honestly silent).
 */
export function mirrorState(feedback: FeedbackItem[]): {
  phoneIndex: number;
  heard: string | null;
  onTarget: boolean;
  unsure: boolean;
} | null {
  const item = mirrorItem(feedback);
  if (item == null || item.anchor?.phone_index == null) return null;
  const onTarget = item.message_key === 'feedback.phone_hearing_on_target';
  const unsure = item.message_key === 'feedback.phone_hearing_unsure';
  return {
    phoneIndex: item.anchor.phone_index,
    heard: onTarget || unsure ? null : heardFromMessage(item),
    onTarget,
    unsure,
  };
}

// The heard phone lives in the message for items the composer authored
// ("I heard /z/ where you were going for /ð/."); a rich client may instead
// read contrasts[evidence.contrast_index].heard. Open-set ears say
// "I heard something else where ..." - there is no phone label to show, so
// the timeline keeps the expected label. This fallback keeps the helper
// self-contained for the panel/timeline.
function heardFromMessage(item: FeedbackItem): string | null {
  const match = /^I heard \/([^/]+)\//.exec(item.message);
  return match ? match[1] : null;
}
