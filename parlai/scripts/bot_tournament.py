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
import spacy

nlp = spacy.load("en_core_web_sm")

import warnings
warnings.filterwarnings("ignore")

from pymongo import MongoClient
from tqdm import tqdm
import random
DATABASE_NAME = 'auto_judge_final'


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
    parser.add_argument('-col', '--collection-name', type=str)
    parser.add_argument(
        '-mf1',
        '--model-file1',
        default=None,
        help='model file name for loading and saving models',
    )
    parser.add_argument(
        '-mf2',
        '--model-file2',
        default=None,
        help='model file name for loading and saving models',
    )
    parser.add_argument(
        '--display-ignore-fields',
        type=str,
        default='label_candidates,text_candidates',
        help='Do not display these fields',
    )
    parser.add_argument(
        '-it',
        '--interactive-task',
        type='bool',
        default=True,
        help='Create interactive version of task',
    )
    parser.add_argument(
        '--selfchat-max-turns',
        type=int,
        default=10,
        help="The number of dialogue turns before self chat ends.",
    )
    parser.add_argument(
        '--seed-messages-from-task',
        action='store_true',
        help="Automatically seed conversation with messages from task dataset.",
    )
    parser.add_argument('--outfile', type=str, default='/tmp/selfchat.json')
    parser.add_argument(
        '--format', type=str, default='json', choices={'parlai', 'json'}
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

def store_logger(opt, collection, logger: WorldLogger):
    for convo in logger._logs:
        convo_data = {}

        convo_data['is_human0'] = False
        convo_data['is_human1'] = False

        convo_data['domain_name'] = opt['task'].split(':')[0]
        turn_list = []

        convo_data['system_type0'] = opt['model_file1'].split('/')[2]
        convo_data['system_type1'] = opt['model_file2'].split('/')[2]

        for eid, exchange in enumerate(convo):
            turn0_db, turn1_db = {}, {}

            turn0 = exchange[0]
            turn1 = exchange[1]
            turn0_db['exchange_nr'] = eid
            turn1_db['exchange_nr'] = eid

            turn0_db['id'] = convo_data['system_type0']
            turn1_db['id'] = convo_data['system_type1']

            turn0_db['episode_done'] = bool(turn0['episode_done'])
            turn1_db['episode_done'] = bool(turn1['episode_done'])

            turn0_db['text'] = ' '.join([tok.text for tok in nlp(turn0['text'].lower())])
            turn1_db['text'] = ' '.join([tok.text for tok in nlp(turn1['text'].lower())])

            turn_list.append(turn0_db)
            turn_list.append(turn1_db)

        model1_starts = random.choice([False, True])

        if not model1_starts:
            convo_data['system_name1'] = opt['model_file1']
            convo_data['system_name0'] = opt['model_file2']

            convo_data['system_type1'] = opt['model_file1'].split('/')[2]
            convo_data['system_type0'] = opt['model_file2'].split('/')[2]
            convo_data['convo'] = cap_context(turn_list, convo_data['domain_name'])[1:]
        else:
            convo_data['system_name0'] = opt['model_file1']
            convo_data['system_name1'] = opt['model_file2']

            convo_data['system_type0'] = opt['model_file1'].split('/')[2]
            convo_data['system_type1'] = opt['model_file2'].split('/')[2]
            convo_data['convo'] = cap_context(turn_list, convo_data['domain_name'])

        collection.insert_one(convo_data)


def self_chat(opt, print_parser=None):
    client = MongoClient(
        opt['mongo_host'],
        opt['mongo_port'],
        username=opt['user_name'],
        password=opt['password'],
        #authSource=DATABASE_NAME
    )

    db = client[DATABASE_NAME]

    collection = db[opt['collection_name']]

    if print_parser is not None:
        if print_parser is True and isinstance(opt, ParlaiParser):
            print_parser = opt
        elif print_parser is False:
            print_parser = None
    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: self_chat should be passed opt not Parser ]')
        opt = opt.parse_args()

    random.seed(opt['seed'])
    # Create models
    opt['model_file'] = opt['model_file1']
    if opt['model_file'] == 'tmp/convai2/lost_in_conversation/last_checkpoint':
        parser.set_defaults(model='projects.convai2.baselines.transformer_chatbot.agent:TransformerAgent',
                            sample=False,
                            wild_mode=False,
                            replace_repeat=False,
                            replace_ngram=False,
                            detokenize=False,
                            emoji_prob=0,
                            add_questions=0,
                            clean_emoji=False,
                            check_grammar=False,
                            correct_generative=False,
                            split_into_sentences=False,
                            max_seq_len=256,
                            beam_size=3,
                            annealing_topk=None,
                            annealing=0.6,
                            length_penalty=0.6)
        opt = parser.parse_args()
        agent1 = create_agent(opt, requireModelExists=True)
    elif opt['model_file'] == 'tmp/convai2/huggingface/model':
        parser.set_params(model='projects.convai2.baselines.huggingface.convai_evaluation:TransformerAgent')
        opt = parser.parse_args()
        agent1 = create_agent(opt, requireModelExists=True)
    else:
        agent1 = create_agent(opt, requireModelExists=True)

    opt['model_file'] = opt['model_file2']
    if opt['model_file'] == 'tmp/convai2/lost_in_conversation/last_checkpoint':
        parser.set_defaults(model='projects.convai2.baselines.transformer_chatbot.agent:TransformerAgent',
                            sample=False,
                            wild_mode=False,
                            replace_repeat=False,
                            replace_ngram=False,
                            detokenize=False,
                            emoji_prob=0,
                            add_questions=0,
                            clean_emoji=False,
                            check_grammar=False,
                            correct_generative=False,
                            split_into_sentences=False,
                            max_seq_len=256,
                            beam_size=3,
                            annealing_topk=None,
                            annealing=0.6,
                            length_penalty=0.6)
        opt = parser.parse_args()
        agent2 = create_agent(opt, requireModelExists=True)
    elif opt['model_file'] == 'tmp/convai2/huggingface/model':
        parser.set_params( model='projects.convai2.baselines.huggingface.convai_evaluation:TransformerAgent')
        opt = parser.parse_args()
        agent2 = create_agent(opt, requireModelExists=True)
    else:
        agent2 = create_agent(opt, requireModelExists=True)

    if hasattr(agent2, 'id'):
        agent2.id = agent2.id + "2"

    opt['random_order'] = False
    world = create_task(opt, [agent1, agent2])

    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent1.opt
        print_parser.print_args()

    # set up logging
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()
    logger = WorldLogger(opt)

    # Run some self chats.
    max_dial_cnt = opt['num_dialogues']
    #dial_cnt = 0
    world.max_turn_cnt = world.sample_episode_length()
    for dial_cnt in tqdm(range(max_dial_cnt)):
    #while dial_cnt < max_dial_cnt:
        #world.max_turn_cnt = world.sample_episode_length()
        #world.turn_cnt = 0
        #print('Dialogue Number: {}, Max Turn: {}\n'.format(dial_cnt, world.max_turn_cnt))
        while True:
            world.parley()
            logger.log(world)

            if world.episode_done():
                break
        #dial_cnt += 1
        if dial_cnt % 20 == 0:
            store_logger(opt, collection, logger)
            logger = WorldLogger(opt)
    store_logger(opt, collection, logger)

if __name__ == '__main__':
    parser = setup_args()
    self_chat(parser.parse_args(print_args=False), print_parser=parser)
