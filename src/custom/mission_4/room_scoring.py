"""Mission 4 room descriptions and image-conditioned language-bias calibration."""
import math
from pathlib import Path
import tempfile

DESCRIPTION_PROMPT = 'Describe the visible furniture and objects in the room through the doorway.'
DESCRIPTIONS = {
    'classroom': 'The room contains student desks, chairs and a teaching board.',
    'medical': 'The room contains a hospital bed and medical equipment.',
    'cafeteria': 'The room contains dining tables, chairs and food equipment.',
}


def calibrated_observation(raw, neutral, size):
    """Subtract a measured neutral-image prior, preserving raw model evidence.

    This can change independent labels. A common per-label subtraction cannot
    change a one-of-each joint assignment or its margin; description scoring is
    the part which changes that decision. Scores are not confidence probabilities.
    """
    for values in (raw['scores'], neutral['scores']):
        if (set(values) != set(DESCRIPTIONS) or
                any(type(v) not in (float, int) or not math.isfinite(v)
                    for v in values.values())):
            raise ValueError('Invalid description likelihood scores')
    scores = {label: raw['scores'][label]-neutral['scores'][label]
              for label in DESCRIPTIONS}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError('Nonfinite calibrated description scores')
    label = max(scores, key=scores.get)
    return {**raw, 'label': label, 'text': label,
            'text_kind': 'highest_calibrated_description_likelihood',
            'scores': scores, 'raw_scores': dict(raw['scores']),
            'neutral_scores': dict(neutral['scores']),
            'score_kind': 'neutral_calibrated_description_log_likelihood',
            'calibration': {'rgb': [127, 127, 127], 'size': list(size)},
            'prompt': DESCRIPTION_PROMPT, 'candidate_descriptions': dict(DESCRIPTIONS)}


def score_room_image(image_path, score, calibration_cache):
    """Score one actual image; cache a neutral reference per resolution per worker."""
    from PIL import Image
    with Image.open(image_path) as image:
        size = image.size
    if size not in calibration_cache:
        with tempfile.TemporaryDirectory(prefix='haic-neutral-room-') as directory:
            neutral_path = Path(directory) / 'neutral.png'
            Image.new('RGB', size, (127, 127, 127)).save(neutral_path)
            calibration_cache[size] = score(neutral_path)
    raw = score(image_path)
    return calibrated_observation(raw, calibration_cache[size], size)
