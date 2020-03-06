#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic example which iterates through the tasks specified and prints them out. Used for
verification of data loading and iteration.

For example, to make sure that bAbI task 1 (1k exs) loads one can run and to
see a few of them:

Examples
--------

.. code-block:: shell

  python display_data.py -t babi:task1k:1
"""

from parlai.core.params import ParlaiParser
from parlai.utils.world_logging import WorldLogger
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task

from pymongo import MongoClient
from tqdm import tqdm
import random

DATABASE_NAME = 'auto_judge'
COLLECTION_NAME = 'sampled-dialogues-amt-test1'
def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Display data from a task')
    parser.add_pytorch_datateacher_args()
    # Get command line arguments
    parser.add_argument('-n', '-ne', '--num-examples', type=int, default=10)
    parser.add_argument('-ns', '--num-stored', type=int, default=10)
    parser.add_argument('-mdl', '--max-display-len', type=int, default=1000)
    parser.add_argument('--display-ignore-fields', type=str, default='agent_reply')
    parser.set_defaults(datatype='train:stream')
    parser.add_argument('-host', '--mongo-host', type=str)
    parser.add_argument('-port', '--mongo-port', type=int)
    parser.add_argument('-user', '--user-name', type=str)
    parser.add_argument('-pw', '--password', type=str)
    WorldLogger.add_cmdline_args(parser)
    return parser



def display_data(opt):
    client = MongoClient(
        opt['mongo_host'],
        opt['mongo_port'],
        username=opt['user_name'],
        password=opt['password'],
        # authSource=DATABASE_NAME
    )

    db = client[DATABASE_NAME]

    collection = db[COLLECTION_NAME]

    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    logger = WorldLogger(opt)

    max_dial_count = opt['num_examples']
    dial_count = 0

    try:
        # print dataset size if available
        print(
            '[ loaded {} episodes with a total of {} examples ]'.format(
                world.num_episodes(), world.num_examples()
            )
        )
    except Exception:
        pass

    # Show some example dialogs.
    for _ in tqdm(range(max_dial_count)):
        while True:
            world.parley()
            logger.log(world)

            if world.episode_done() or world.acts[0].get('episode_done', False):
                break
        if world.epoch_done():
            break

        dial_count += 1

    domain =  opt['task'].split(':')[0]
    if domain == 'dailydialog':
        convo_list = [convo[1:] for convo in logger._logs if not convo[0] == '__SILENCE__']
    elif domain == 'wizard_of_wikipedia':
        convo_list = logger._logs
    elif domain == 'personachat':
        convo_list = [convo[1:] for convo in logger._logs]
    elif domain == 'empathetic_dialogues':
        convo_list = [convo[1:] for convo in logger._logs if len(convo) > 3]
    else:
        convo_list = logger._logs

    sampled_convos = random.sample(convo_list, k=opt['num_stored'])
    for did, convo in enumerate(sampled_convos):
        convo_data = {}
        convo_data['domain_name'] = opt['task'].split(':')[0]
        convo_data['system_name0'] = "{}/human".format(convo_data['domain_name'])
        convo_data['system_name1'] = "{}/human".format(convo_data['domain_name'])

        convo_data['system_type0'] = 'human'
        convo_data['system_type1'] = 'human'

        convo_data['is_human0'] = True
        convo_data['is_human1'] = True

        turn_list = []

        for eid, exchange in enumerate(convo):
            turn0 = exchange[0]
            turn1 = exchange[1]
            turn0['exchange_nr'] = eid
            turn1['exchange_nr'] = eid
            turn0.force_set('id', 'human')
            turn1.force_set('id', 'human')
            turn_list.append(turn0)
            turn_list.append(turn1)

        if domain == 'wizard_of_wikipedia':
            turn_list = turn_list[1:] #remove context word
        convo_data['convo'] = turn_list
        collection.insert_one(convo_data)
        print('Dialogue Number: {}\n'.format(did))
        for tid, turn in enumerate(convo_data['convo']):
            print(tid, turn['text'])
        print('\n\n')


if __name__ == '__main__':
    random.seed(42)

    # Get command line arguments
    parser = setup_args()
    opt = parser.parse_args()
    display_data(opt)
