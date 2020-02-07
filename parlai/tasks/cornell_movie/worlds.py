from parlai.tasks.self_chat.worlds import SelfChatBaseWorld
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task

import random
from typing import List

def load_contexts(opt):
    print('[ loading personas.. ]')
    # Create ConvAI2 data so we can assign personas.
    dailydialog_opt = opt.copy()
    dailydialog_opt['task'] = 'cornell_movie'
    if dailydialog_opt['datatype'].startswith('train'):
        dailydialog_opt['datatype'] = 'train:eval'
    dailydialog_opt['max-display-len'] = 1000
    dailydialog_opt['display-ignore-fields'] = "agent_reply"
    dailydialog_opt['interactive_task'] = False
    convai2_agent = RepeatLabelAgent(dailydialog_opt)
    convai2_world = create_task(dailydialog_opt, convai2_agent)
    contexts = list()
    while not convai2_world.epoch_done():
        convai2_world.parley()
        msg = convai2_world.get_acts()[0]
        # Find a new episode
        if msg.get('episode_done', False) and not convai2_world.epoch_done():
            convai2_world.parley()
            msg = convai2_world.get_acts()[0]
            if msg['text'] == '__SILENCE__':
                continue
            contexts.append((msg['text'], msg['labels'][0]))
    print('[ loaded ' + str(len(contexts)) + ' personas ]')
    return list(contexts)


class InteractiveSelfchatWorld(SelfChatBaseWorld):
    def __init__(self, opt, agents, shared=None):
        opt['random_order'] = False
        super(InteractiveSelfchatWorld, self).__init__(opt, agents, shared)

    def init_contexts(self):
        self.context_list = load_contexts(self.opt)

    def get_contexts(self, episode_num: int) -> List[str]:
        random.seed()
        context = random.choice(self.context_list)
        return [context[0], context[1]]
