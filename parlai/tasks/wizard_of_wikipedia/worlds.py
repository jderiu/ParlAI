#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import json
import random
import os
import string
import numpy as np

from parlai.core.agents import create_agent
from parlai.core.message import Message
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE
from parlai.tasks.self_chat.worlds import InteractiveWorld as SelfChatBaseWorld
from parlai.utils.misc import warn_once
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import (
    KnowledgeRetrieverAgent,
)

from typing import List


NO_TOPIC = '[NO TOPIC]'


class InteractiveWorld(DialogPartnerWorld):
    """
    Interactive world for wizard of wikipedia.

    Used for models trained on the task `-t wizard_of_wikipedia`. Automatically
    retrieves knowledge from Wikipedia based on the conversation history using a TF-IDF
    retriever. Then uses a Transformer-based model to select a checked sentence from
    these retrieved passages.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('WoW Interactive World Args')
        parser.add_argument(
            '--print-checked-sentence',
            type='bool',
            default=True,
            help='Print sentence that the model checks.',
        )

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print('[ loading topics.. ]')
        self.opt = opt
        self._load_topics(opt)
        self.num_topics = opt['num_topics']
        self.cnt = 0
        self.human_agent = self.agents[0]
        self.model_agent = self.agents[1]

        self._set_up_knowledge_agent(opt.get('add_token_knowledge', False))

        self.print_checked_sentence = opt['print_checked_sentence']

    @staticmethod
    def generate_world(opt, agents):
        return InteractiveWorld(opt, agents)

    def _set_up_knowledge_agent(self, add_token_knowledge=False):
        from parlai.core.params import ParlaiParser

        parser = ParlaiParser(False, False)
        KnowledgeRetrieverAgent.add_cmdline_args(parser)
        parser.set_params(
            model='projects:wizard_of_wikipedia:knowledge_retriever',
            add_token_knowledge=add_token_knowledge,
        )
        knowledge_opt = parser.parse_args([], print_args=False)
        self.knowledge_agent = create_agent(knowledge_opt)

    def _load_topics(self, opt):
        # Load possible chosen topics
        topics_path = os.path.join(
            opt['datapath'], 'wizard_of_wikipedia', 'topic_splits.json'
        )
        # Get training set topics
        datatype = opt['datatype'].split(':')[0]
        self.topic_list = json.load(open(topics_path, 'rb'))[datatype]

    def _get_new_topic(self):
        random.seed()
        topics = random.sample(self.topic_list, self.num_topics - 1)
        topics.append(NO_TOPIC)
        letters = list(string.ascii_uppercase)[: self.num_topics]
        topic_list = {x: y for x, y in zip(letters, topics)}
        topic_text = '\n'.join(['{}: {}'.format(k, v) for k, v in topic_list.items()])

        done = False
        while not done:
            self.human_agent.observe(
                {
                    'text': '\nPlease choose one of the following topics by typing '
                    'A, B, C, ..., etc. : \n\n{}\n'.format(topic_text)
                }
            )
            topic_act = self.human_agent.act()
            choice = topic_act['text'][0].upper()
            if choice in topic_list:
                done = True
            else:
                self.human_agent.observe(
                    {'text': 'Invalid response, please try again.'}
                )

        chosen_topic = topic_list[choice]
        print('[ Your chosen topic is: {} ]'.format(chosen_topic))
        return chosen_topic

    def _add_knowledge_to_act(self, act):
        self.knowledge_agent.observe(act, actor_id='apprentice')
        knowledge_act = self.knowledge_agent.act()
        act['knowledge'] = knowledge_act['text']
        act['checked_sentence'] = knowledge_act['checked_sentence']
        if self.print_checked_sentence:
            print(
                '[ Using chosen sentence from Wikpedia ]: {}'.format(
                    knowledge_act['checked_sentence']
                )
            )
        act['title'] = knowledge_act['title']
        return act

    def parley(self):
        """
        Loop between wizard and apprentice.

        Adds knowledge to the wizard observations. Assumes that the model agent is the
        wizard model.
        """

        if self.cnt == 0:
            self.topic = self._get_new_topic()
            self.acts = [None, None]
            self.human_first = random.choice([0, 1])

        # possibly get human act first
        if self.cnt == 0 and not self.human_first:
            self.acts[0] = act = Message({'text': '', 'episode_done': False})
            act = self.acts[0]
        else:
            self.acts[0] = self.human_agent.act()
            act = deepcopy(self.acts[0])

        # model agent observe
        if self.cnt == 0 and self.topic != NO_TOPIC:
            # add the chosen_topic to the message
            act['chosen_topic'] = self.topic
            act.force_set('text', '\n'.join([self.topic, act.get('text', 'hi')]))

        # add knowledge to the model observation
        act = self._add_knowledge_to_act(act)

        # model observes knowledge and human (apprentice) act
        self.model_agent.observe(validate(act))

        # model agent act
        self.acts[1] = self.model_agent.act()

        # add the model reply to the knowledge retriever's dialogue history
        self.knowledge_agent.observe(self.acts[1], actor_id='wizard')

        # human (apprentice) agent observes model act
        self.human_agent.observe(validate(self.acts[1]))

        self.update_counters()
        self.cnt += 1

        if self.episode_done():
            print('[ CHAT DONE ]')
            print('\n[ Preparing new chat... ]\n')
            self.cnt = 0
            self.model_agent.reset()


class InteractiveGeneratorWorld(InteractiveWorld):
    """
    Interactive world for generative models.

    Specifically a world for models trained on the task `-t wizard_of_wikipedia
    generator`.
    """

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print('[ loading topics.. ]')
        self.opt = opt
        self._load_topics(opt)
        self.num_topics = opt['num_topics']
        self.cnt = 0
        self.human_agent = self.agents[0]
        self.model_agent = self.agents[1]

        self._set_up_knowledge_agent(add_token_knowledge=True)

    def _add_knowledge_to_act(self, act):
        act = super()._add_knowledge_to_act(act)
        if self.opt.get('prepend_gold_knowledge', False):
            warn_once(
                'Prepending selected knowledge to dialogue input.'
                'If this was not intended behavior, please run with the '
                'flag --prepend-gold-knowledge False'
            )
            knowledge_text = ' '.join(
                [TOKEN_KNOWLEDGE, act['checked_sentence'], TOKEN_END_KNOWLEDGE]
            )
            new_text = '\n'.join([knowledge_text, act['text']])
            act.force_set('text', new_text)
        else:
            warn_once(
                'Not prepending selected knowledge to dialogue input.'
                'If this was not intended behavior, please run with the '
                'flag --prepend-gold-knowledge True'
            )
        return act


def compute_dialogue_lenghts(opt):
    wizard_opt = opt.copy()
    wizard_opt['task'] = 'wizard_of_wikipedia'
    if wizard_opt['datatype'].startswith('train'):
        wizard_opt['datatype'] = 'train:eval'
    wizard_opt['max-display-len'] = 1000
    wizard_opt['display-ignore-fields'] = "agent_reply"
    wizard_opt['interactive_task'] = False
    agent = RepeatLabelAgent(wizard_opt)
    world = create_task(wizard_opt, agent)
    convos = list()
    contexts = list()
    while not world.epoch_done():
        turn_list = list()
        while True:
            world.parley()
            msg = world.get_acts()[0]
            text = msg['text']
            label = msg['labels'][0]
            knwoledge = msg['knowledge']

            turn_list.append('\n'.join([knwoledge, text]))
            turn_list.append('\n'.join([knwoledge, label]))

            if world.episode_done() or world.acts[0].get('episode_done', True):
                break
        contexts.append(turn_list[:2])
        convos.append(turn_list)

    lengths = [int(len(convo) / 2) for convo in convos]
    bin_lengths = np.bincount(lengths)
    p_vals = bin_lengths / np.sum(bin_lengths)
    lengths = np.arange(bin_lengths.shape[0])
    return lengths, p_vals, contexts


class InteractiveSelfchatWorld(SelfChatBaseWorld):

    def __init__(self, opt, agents, shared=None):
        opt['random_order'] = False
        self._set_up_knowledge_agent(opt.get('add_token_knowledge', False))
        self.lengths, self.p_vals, self.seed_contexts = compute_dialogue_lenghts(opt)
        super(InteractiveSelfchatWorld, self).__init__(opt, agents, shared)

    def init_contexts(self):
        print('[ loading topics.. ]')
        # Load possible chosen topics
        topics_path = os.path.join(
            self.opt['datapath'], 'wizard_of_wikipedia', 'topic_splits.json'
        )
        # Get training set topics
        datatype = self.opt['datatype'].split(':')[0]
        self.topic_list = json.load(open(topics_path, 'rt', encoding='utf-8'))[datatype]

    def sample_episode_length(self):
        sampled_val = random.choices(self.lengths, weights=self.p_vals, k=1)[0] + 1
        sampled_val = max([sampled_val, 4])
        return sampled_val

    def get_contexts(self, episode_num: int) -> List[str]:
        random.seed()
        context = random.choice(self.seed_contexts)
        return context

    def _add_knowledge_to_act(self, act):
        self.knowledge_agent.observe(act, actor_id='apprentice')
        knowledge_act = self.knowledge_agent.act()
        act['knowledge'] = knowledge_act['text']
        act['checked_sentence'] = knowledge_act['checked_sentence']
        act['title'] = knowledge_act['title']
        return act

    def _set_up_knowledge_agent(self, add_token_knowledge=False):
        from parlai.core.params import ParlaiParser

        parser = ParlaiParser(False, False)
        KnowledgeRetrieverAgent.add_cmdline_args(parser)
        parser.set_params(
            model='projects:wizard_of_wikipedia:knowledge_retriever',
            add_token_knowledge=add_token_knowledge,
        )
        knowledge_opt = parser.parse_args([], print_args=False)
        self.knowledge_agent = create_agent(knowledge_opt)

    def parley(self):
        if self.episode_done():
            self.turn_cnt = 0
            self.episode_cnt += 1
            self.contexts = None
            self.seed_utterances = None
            agents = self.get_agents()
            for a in agents:
                a.reset()

        if self.turn_cnt == 0:
            self.acts = [None, None]
            # choose speaking order:
            self.agents_ordered = [self.agents[0], self.agents[1]]
            # get the beginning of the conversation, which can include contexts
            # and/or any number of starting messages
            self.contexts = self.get_contexts(self.episode_cnt)
            self.seed_utterances = self._get_seed_utt_acts(
                self.episode_cnt, self.agents_ordered
            )

        if self.contexts:
            assert len(self.contexts) == 2
            # initial context
            for i in range(0, 2):
                context = {
                    'text': self.contexts[i],
                    'episode_done': False,
                    'id': 'context',
                }
                self.acts[1 - i] = context
                self.agents_ordered[i].observe(validate(context))
            # clear contexts so they are only added once per episode
            self.contexts = None
        elif self.seed_utterances:
            # pop the next two seed messages (there may be less or more than 2 total)
            utts = self.seed_utterances[:2]
            self.seed_utterances = self.seed_utterances[2:]
            # process the turn
            for i in [0, 1]:
                # if we have a seed utterance, add it to the conversation
                if len(utts) > i:
                    self.acts[i] = utts[i]
                    if hasattr(self.agents_ordered[i], 'self_observe'):
                        self.agents_ordered[i].self_observe(self.acts[i])
                else:
                    self.acts[i] = self.agents_ordered[i].act()
                self.agents_ordered[1 - i].observe(validate(self.acts[i]))
        else:
            # do regular loop
            acts = self.acts
            agents = self.agents_ordered

            acts[0] = agents[0].act()
            acts[0] = self._add_knowledge_to_act(acts[0])
            agents[1].observe(validate(acts[0]))
            acts[1] = agents[1].act()
            acts[1] = self._add_knowledge_to_act(acts[1])
            agents[0].observe(validate(acts[1]))

        self.update_counters()
        self.turn_cnt += 1
