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

FEEDBACK = (
    'Choose the next action using enemy positions and weapon_ready. '
    'ammo_used is ammo spent during the last action. '
    'last_turn_degrees is the actual turn: positive means left, negative means right.'
)


def apply_prompt(body, variant='explicit'):
    question = dict(body['questions']['action'], instructions=INSTRUCTIONS)
    if variant == 'criteria':
        question = dict(question,
            instructions=FEEDBACK,
            criteria={
                'attack':'A visible monster is dead center, in the crosshair. Keep aiming here and hold fire, even while the weapon cools down.',
                'turn left':'The closest visible monster is left of center. Turn left to face it.',
                'turn right':'No monsters are visible, or the closest visible monster is right of center. Turn right to search or face it.',
            })
    elif variant == 'range':
        question = dict(question,
            instructions=('Select the priority enemy by range to the player: point blank, close, '
                          'at medium range, far away (nearest first). Break ties by list order. '
                          'Which action faces or shoots that enemy? ' + FEEDBACK),
            criteria={
                'attack':'The priority enemy is dead center, in the crosshair. Keep aiming here and hold fire, even while the weapon cools down.',
                'turn left':'The priority enemy is left of center. Turn left to face that enemy.',
                'turn right':'The priority enemy is right of center, or no enemies are visible. Turn right to face that enemy or search.',
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
