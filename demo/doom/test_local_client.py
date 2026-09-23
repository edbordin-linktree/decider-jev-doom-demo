"""Transport-only compatibility checks; these do not call a model."""
import unittest
from unittest.mock import patch
from client import Client

class LocalClientTests(unittest.TestCase):
    def test_prompt_variants_preserve_state_and_action_names(self):
        from decider_prompt import apply_prompt
        body={'model':'decider-0.8b','state':{'monsters':[]},'questions':{'action':{
            'type':'choice','instructions':'Original','criteria':{
                'attack':'fire','turn left':'rotate left','turn right':'rotate right'}}}}
        for variant in ('explicit','criteria','plain'):
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
