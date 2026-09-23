"""Transport-only compatibility checks; these do not call a model."""
import unittest
from unittest.mock import patch
from client import Client

class LocalClientTests(unittest.TestCase):
    def test_attack_describes_holding_fire_without_adding_an_action(self):
        from decider_prompt import apply_prompt
        for variant in ('range',):
            body = {'state': {'weapon_ready': False}, 'questions': {'action': {'type': 'choice'}}}
            question = apply_prompt(body, variant)['questions']['action']
            self.assertEqual(list(question['criteria']), ['attack', 'turn left', 'turn right'])
            attack = question['criteria']['attack']
            self.assertIn('dead center, in the crosshair', attack)
            self.assertIn('hold fire, even while the weapon cools down', attack)

    def test_default_restores_baseline_question_and_observations(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import numpy as np
        from game import Snapshot
        from play import decide
        snap = Snapshot(frame=np.zeros((2, 2, 3)), health=100, ammo=10, kills=0, tic=5,
                        weapon_ready=False, ammo_used=1, last_turn_degrees=8.8)
        client = Mock()
        doom = SimpleNamespace(goal='', rules=[], examples=[], actions=['attack', 'turn left', 'turn right'])
        decide(client, 'systemone', snap, doom, 'attack', 'criteria')
        state, instruction, criteria = client.decide_systemone.call_args.args
        self.assertEqual(set(state), {'monsters', 'items', 'space', 'health', 'ammo', 'kills', 'last_action'})
        self.assertEqual(instruction, 'Which action best matches the current visible monsters? Use their positions, not last_action.')
        self.assertEqual(criteria['attack'], 'A visible monster is dead center, in the crosshair. Shoot it.')
        decide(client, 'systemone', snap, doom, 'attack', 'range')
        experimental = client.decide_systemone.call_args.args[0]
        self.assertEqual(experimental['ammo_used'], 1)
        self.assertEqual(experimental['last_turn_degrees'], 8.8)

    def test_range_targets_one_enemy_by_range_without_filtering_state(self):
        from decider_prompt import apply_prompt
        state = {'monsters': [
            {'kind': 'chainsaw marine', 'position': 'on the right', 'range': 'point blank'},
            {'kind': 'pinky demon', 'position': 'dead center, in the crosshair', 'range': 'far away'},
        ], 'last_action': 'attack'}
        body = {'state': state, 'questions': {'action': {'type': 'choice'}}}
        result = apply_prompt(body, 'range')
        question = result['questions']['action']
        self.assertIs(result['state'], state)
        self.assertIn('range to the player: point blank, close, at medium range, far away', question['instructions'])
        for description in question['criteria'].values():
            self.assertIn('priority enemy', description)
        self.assertIn('no enemies are visible', question['criteria']['turn right'])

    def test_prompt_variants_preserve_state_and_action_names(self):
        from decider_prompt import apply_prompt
        body={'model':'decider-0.8b','state':{'monsters':[]},'questions':{'action':{
            'type':'choice','instructions':'Original','criteria':{
                'attack':'fire','turn left':'rotate left','turn right':'rotate right'}}}}
        for variant in ('explicit','criteria','plain','range'):
            result=apply_prompt(body,variant)
            self.assertEqual(result['state'],body['state'])
            self.assertEqual(result['model'],body['model'])
            self.assertEqual(list(result['questions']['action']['criteria']),list(body['questions']['action']['criteria']))
        self.assertEqual(body['questions']['action']['instructions'],'Original')

    def test_model_selection_preserves_question_and_state(self):
        state={'monsters':[],'health':100}
        criteria={'attack':'fire','turn left':'rotate'}
        response={'answers':{'action':{'choice':'turn left','confidence':.8,
                  'probabilities':{'attack':.2,'turn left':.8}}},'usage':{}}
        bodies=[]
        for model in (None,'decider-0.8b'):
            client=Client('http://127.0.0.1:8000',api='systemone',model=model)
            with patch.object(client,'_post',return_value=response) as post:
                decision=client.decide_systemone(state,'Which action?',criteria)
                self.assertEqual(decision.best,'turn left')
                bodies.append(post.call_args.args[1])
        self.assertEqual(bodies[1],dict(bodies[0],model='decider-0.8b'))
        self.assertEqual(bodies[1]['state'],state)
        self.assertEqual(bodies[1]['questions']['action']['criteria'],criteria)

if __name__=='__main__': unittest.main()
