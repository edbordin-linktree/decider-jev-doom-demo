"""Measure game feedback without loading a model."""
from unittest.mock import Mock

import pytest
import vizdoom as vzd

from game import Doom
from describe import state_dict


def test_real_game_feedback():
    doom = Doom('defend_the_center', seed=37)
    try:
        doom.new_episode()
        initial = state_dict(doom.snapshot(), None)
        assert initial['ammo_used'] == 0
        assert initial['last_turn_degrees'] == 0
        doom.step('turn left', 5)
        assert doom.snapshot().last_turn_degrees > 0
        doom.step('turn right', 5)
        assert doom.snapshot().last_turn_degrees < 0
        # Re-reading a snapshot must not consume or change step feedback.
        assert doom.snapshot().last_turn_degrees == doom.snapshot().last_turn_degrees
        saw_ammo_used = False
        for _ in range(10):
            previous = doom.snapshot()
            doom.step('attack', 5)
            current = doom.snapshot()
            assert current.ammo_used == previous.ammo - current.ammo
            assert current.weapon_ready == bool(doom.game.get_game_variable(vzd.GameVariable.ATTACK_READY))
            saw_ammo_used |= current.ammo_used > 0
        assert saw_ammo_used
        doom.new_episode()
        assert doom.snapshot().ammo_used == 0
        assert doom.snapshot().last_turn_degrees == 0
    finally:
        doom.close()


@pytest.mark.parametrize('before,after,expected', [(359, 2, 3), (2, 359, -3)])
def test_turn_wraparound_and_weapon_change(before, after, expected):
    doom = Doom.__new__(Doom)
    doom.game = Mock()
    # Ammo, weapon, angle before, then ammo, weapon, angle after.
    doom.game.get_game_variable.side_effect = [20, 2, before, 5, 3, after]
    doom.game.make_action.return_value = 0
    doom.total_reward = 0
    doom.step('turn left', 5)
    assert doom.last_turn_degrees == expected
    assert doom.ammo_used == 0  # A weapon switch is not evidence of spending ammo.
