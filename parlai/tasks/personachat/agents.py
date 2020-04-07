#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
PersonaChat task agents.

Persona can be 'none', 'self', 'other', or 'both'.
Format of persona can be 'original' or 'revised'.

This is specified in the following way:
--task personachat:{format}
...where {format} is one of...
- none
- self_original
- self_revised
- other_original
- other_revised
- both_original
- both_revised
"""


from parlai.core.teachers import FbDialogTeacher, create_task_agent_from_taskname
from .build import build

import copy
import os


def _path(opt, persona):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0] + '_' + persona
    return os.path.join(opt['datapath'], 'Persona-Chat', 'personachat', dt + '.txt')


class NoneTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'none_original')
        super().__init__(opt, shared)


class SelfOriginalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'self_original')
        super().__init__(opt, shared)


class SelfOriginal20Teacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'self_original_20')
        super().__init__(opt, shared)


class SelfOriginal60Teacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'self_original_60')
        super().__init__(opt, shared)


class SelfTeacher(SelfOriginalTeacher):
    pass


class SelfRevisedTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'self_revised')
        super().__init__(opt, shared)


class OtherOriginalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'other_original')
        super().__init__(opt, shared)


class OtherTeacher(OtherOriginalTeacher):
    pass


class OtherRevisedTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'other_revised')
        super().__init__(opt, shared)


class BothOriginalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'both_original')
        super().__init__(opt, shared)


class BothTeacher(BothOriginalTeacher):
    pass


class BothRevisedTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'both_revised')
        super().__init__(opt, shared)


class DefaultTeacher(SelfOriginalTeacher):
    pass


class SelfchatTeacher(SelfOriginalTeacher):
    # Dummy class to add arguments for interactive world.
    pass

def create_agents(opt):
    if not opt.get('interactive_task', False):
        return create_task_agent_from_taskname(opt)
    else:
        # interactive task has no task agents (they are attached as user agents)
        return []
