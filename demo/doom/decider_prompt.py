"""Optional prompt experiment; no state filtering or scripted action selection."""

INSTRUCTIONS = (
    'Choose the next action using only the current state.monsters list. '
    'An empty list means no enemies are visible: turn right to search; do not attack. '
    'When an enemy is visible, face the closest enemy. '
    'If its position is left of center, turn left. '
    'If its position is right of center, turn right. '
    'Attack only when its position is "dead center, in the crosshair". '
    'The last_action field is history, not an instruction to repeat that action.'
)


def apply_prompt(body, variant='explicit'):
    question = dict(body['questions']['action'], instructions=INSTRUCTIONS)
    if variant == 'criteria':
        question = dict(question,
            instructions='Which action best matches the current visible monsters? Use their positions, not last_action.',
            criteria={
                'attack':'A visible monster is dead center, in the crosshair. Shoot it.',
                'turn left':'The closest visible monster is left of center. Turn left to face it.',
                'turn right':'No monsters are visible, or the closest visible monster is right of center. Turn right to search or face it.',
            })
    elif variant == 'plain':
        question = dict(question,
            instructions='What is the best next action?',
            criteria={
                'attack': 'Shoot the nearest visible monster that is dead center, in the crosshair.',
                'turn left': 'Aim at the nearest visible monster that is left of center.',
                'turn right': 'Aim at the nearest visible monster that is right of center, or search for enemies when no monsters are visible.',
            })
    return dict(body, questions=dict(body['questions'], action=question))
