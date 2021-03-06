#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Allows a model to self-chat on a given task.
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.world_logging import WorldLogger
from parlai.utils.misc import TimeLogger
from parlai.core.message import Message

from pymongo import MongoClient

import math
import random
DATABASE_NAME = 'auto_judge'
COLLECTION_NAME = 'sampled-dialogues-amt-test3'


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Self chat with a model')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-d', '--display-examples', type='bool', default=True)
    parser.add_argument('-n', '-ne', '--num-examples', type=int, default=10)
    parser.add_argument('-nd', '--num-dialogues', type=int, default=10)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument('-host', '--mongo-host', type=str)
    parser.add_argument('-port', '--mongo-port', type=int)
    parser.add_argument('-user', '--user-name', type=str)
    parser.add_argument('-pw', '--password', type=str)
    parser.add_argument(
        '--display-ignore-fields',
        type=str,
        default='label_candidates,text_candidates',
        help='Do not display these fields',
    )
    parser.add_argument(
        '-st',
        '--selfchat-task',
        type='bool',
        default=True,
        help='Create a self chat version of the task',
    )
    parser.add_argument(
        '--num-self-chats', type=int, default=1, help='Number of self chats to run'
    )
    parser.add_argument(
        '--selfchat-max-turns',
        type=int,
        default=6,
        help='The number of dialogue turns before self chat ends',
    )
    parser.add_argument(
        '--seed-messages-from-task',
        action='store_true',
        help='Automatically seed conversation with messages from task dataset.',
    )
    parser.add_argument(
        '--outfile', type=str, default=None, help='File to save self chat logs'
    )
    parser.add_argument(
        '--save-format',
        type=str,
        default='conversations',
        choices=['conversations', 'parlai'],
        help='Format to save logs in. conversations is a jsonl format, parlai is a text format.',
    )
    parser.set_defaults(interactive_mode=True, task='self_chat')
    WorldLogger.add_cmdline_args(parser)
    return parser


def cap_context(turn_list, domain):
    if domain == 'dailydialog':
        return turn_list[2:]
    elif domain == 'personachat':
        return turn_list[2:]
    elif domain == 'wizard_of_wikipedia':
        return turn_list[2:]
    elif domain == 'empathetic_dialogues':
        return turn_list[2:]


def self_chat(opt, print_parser=None):
    client = MongoClient(
        opt['mongo_host'],
        opt['mongo_port'],
        username=opt['user_name'],
        password=opt['password'],
        #authSource=DATABASE_NAME
    )

    db = client[DATABASE_NAME]

    collection = db[COLLECTION_NAME]

    if print_parser is not None:
        if print_parser is True and isinstance(opt, ParlaiParser):
            print_parser = opt
        elif print_parser is False:
            print_parser = None
    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: self_chat should be passed opt not Parser ]')
        opt = opt.parse_args()

    # Create agents
    agent1 = create_agent(opt, requireModelExists=True)
    agent2 = agent1.clone()

    # Set IDs
    model_id = agent1.id
    agent1.id = model_id + "_1"
    agent2.id = model_id + "_2"

    world = create_task(opt, user_agents=[agent1, agent2])

    # Set up world logging
    logger = WorldLogger(opt)
    log_time = TimeLogger()

    # Run some self chats.
    max_dial_cnt = opt['num_dialogues']
    dial_cnt = 0
    while dial_cnt < max_dial_cnt:
        world.max_turn_cnt = world.sample_episode_length()
        world.turn_cnt = 0
        print('Dialogue Number: {}, Max Turn: {}\n'.format(dial_cnt, world.max_turn_cnt))
        while True:
            world.parley()
            logger.log(world)

            if opt.get('display_examples'):
                print(world.display())
            if world.episode_done():
                break

        print('\n\n')
        dial_cnt += 1

    if opt.get('display_examples'):
        print('-- end of episode --')

    logger.write(opt['outfile'], opt['format'])
    for convo in logger._logs:
        convo_data = {}
        convo_data['system_name0'] = opt['model_file']
        convo_data['system_name1'] = opt['model_file']

        convo_data['system_type0'] = opt['model_file'].split('/')[2]
        convo_data['system_type1'] = opt['model_file'].split('/')[2]

        convo_data['is_human0'] = False
        convo_data['is_human1'] = False

        convo_data['domain_name'] = opt['task'].split(':')[0]
        turn_list = []

        for eid, exchange in enumerate(convo):
            turn0 = exchange[0]
            turn1 = exchange[1]
            turn0['exchange_nr'] = eid
            turn1['exchange_nr'] = eid
            if type(turn0) == Message:
                turn0.force_set('episode_done', bool(turn0['episode_done']))
            else:
                turn0['episode_done'] = bool(turn0['episode_done'])
            if type(turn0) == Message:
                turn1.force_set('episode_done', bool(turn1['episode_done']))
            else:
                turn1['episode_done'] = bool(turn1['episode_done'])
            turn_list.append(turn0)
            turn_list.append(turn1)


        convo_data['convo'] = cap_context(turn_list, convo_data['domain_name'])
        collection.insert_one(convo_data)
        print(len(convo_data['convo']))


if __name__ == '__main__':
    parser = setup_args()
    self_chat(parser.parse_args(print_args=False))
