from parlai.tasks.self_chat.worlds import SelfChatWorld
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task

import random
from typing import List
import numpy as np

def load_contexts(opt):
    print('[ loading personas.. ]')
    # Create ConvAI2 data so we can assign personas.
    dailydialog_opt = opt.copy()
    dailydialog_opt['task'] = 'dailydialog'
    if dailydialog_opt['datatype'].startswith('train'):
        dailydialog_opt['datatype'] = 'train:eval'
    dailydialog_opt['max-display-len'] = 1000
    dailydialog_opt['display-ignore-fields'] = "agent_reply"
    dailydialog_opt['interactive_task'] = False
    agent = RepeatLabelAgent(dailydialog_opt)
    world = create_task(dailydialog_opt, agent)
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

        if not turn_list[0] == '__SILENCE__':
            contexts.append(turn_list)

    print('[ loaded ' + str(len(contexts)) + ' personas ]')
    return list(contexts)


class InteractiveSelfchatWorld(SelfChatWorld):
    def __init__(self, opt, agents, shared=None):
        opt['random_order'] = False
        super(InteractiveSelfchatWorld, self).__init__(opt, agents, shared)

    def init_contexts(self):
        self.context_list = load_contexts(self.opt)
        lengths = [int(len(convo)/2) for convo in self.context_list]
        bin_lengths = np.bincount(lengths)
        self.p_vals = bin_lengths/np.sum(bin_lengths)
        self.lengths = np.arange(bin_lengths.shape[0])

    def sample_episode_length(self):
        # if length is 1 then only the context is rendered
        sampled_val = random.choices(self.lengths, weights=self.p_vals, k=1)[0] + 1
        #make sure there are at least 6 turns
        sampled_val = max([sampled_val, 4])
        return sampled_val

    def get_contexts(self, episode_num: int) -> List[str]:
        random.seed()
        context = random.choice(self.context_list)
        return [context[1], context[0]]
