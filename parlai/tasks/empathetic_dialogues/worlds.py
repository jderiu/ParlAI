from parlai.tasks.self_chat.worlds import SelfChatWorld
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task

import random
from typing import List
import numpy as np

def load_contexts(opt):
    print('[ loading personas.. ]')
    # Create ConvAI2 data so we can assign personas.
    opt_copy = opt.copy()
    opt_copy['task'] = 'empathetic_dialogues'
    if opt_copy['datatype'].startswith('train'):
        opt_copy['datatype'] = 'train:eval'
    opt_copy['max-display-len'] = 1000
    opt_copy['display-ignore-fields'] = "agent_reply"
    opt_copy['interactive_task'] = False
    agent = RepeatLabelAgent(opt_copy)
    world = create_task(opt_copy, agent)
    contexts = list()
    while not world.epoch_done():
        turn_list = list()
        while True:
            world.parley()
            msg = world.get_acts()[0]

            turn_list.append(msg['text'])
            turn_list.append(msg['labels'][0])

            if world.episode_done() or world.acts[0].get('episode_done', True):
                break

        contexts.append(turn_list)

    print('[ loaded ' + str(len(contexts)) + ' personas ]')
    return list(contexts)


class InteractiveSelfchatWorld(SelfChatWorld):
    def __init__(self, opt, agents, shared=None):
        opt['random_order'] = False
        super(InteractiveSelfchatWorld, self).__init__(opt, agents, shared)

    def init_contexts(self, shared=None):
        self.context_list = load_contexts(self.opt)
        lengths = [int(len(convo) / 2) for convo in self.context_list]
        bin_lengths = np.bincount(lengths)
        self.p_vals = bin_lengths / np.sum(bin_lengths)
        self.lengths = np.arange(bin_lengths.shape[0])

    def sample_episode_length(self):
        # since empathetic dialogues are super small we make bot-bot dialogues at least 2 exchages long
        sampled_val = random.choices(self.lengths, weights=self.p_vals, k=1)[0] + 3
        sampled_val = max([sampled_val, 3])
        return sampled_val

    def get_contexts(self) -> List[str]:
        random.seed()
        context = random.choice(self.context_list)
        return [context[1], context[0]]
