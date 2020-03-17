#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import uuid
import websocket
import time
import threading
from parlai.core.params import ParlaiParser


def _get_rand_id():
    """
    :return: The string of a random id using uuid4
    """
    return str(uuid.uuid4())


def _prBlueBG(text):
    """
    Print given in text with a blue background.

    :param text: The text to be printed
    """
    print("\033[44m{}\033[0m".format(text), sep="")


def on_message(ws, message):
    """
    Prints the incoming message from the server.

    :param ws: a WebSocketApp
    :param message: json with 'text' field to be printed
    """
    incoming_message = json.loads(message)
    print("\033[0m\nBot: " + incoming_message['text'], "\033[44m\n")


class CustomOnMessage:

    def __init__(self, json_path):
        self.json_path = json_path

    def __call__(self, ws, message):
        incoming_message = json.loads(message)
        with open(json_path, 'a') as fout:
            to_dump = incoming_message.copy()
            to_dump['speaker'] = 'bot'
            fout.write(json.dumps(to_dump))
            fout.write('\n')
        print("\033[0m\nBot: " + incoming_message['text'], "\033[44m\n")


def on_error(ws, error):
    """
    Prints an error, if occurs.

    :param ws: WebSocketApp
    :param error: An error
    """
    print(error)


def on_close(ws):
    """
    Cleanup before closing connection.

    :param ws: WebSocketApp
    """
    # Reset color formatting if necessary
    print("\033[0m")
    print("Connection closed")


def _run(ws, id, json_path):
    """
    Takes user input and sends it to a websocket.

    :param ws: websocket.WebSocketApp
    """
    while True:
        x = input("\033[44m Me: ")
        print("\033[0m", end="")
        data = {}
        data['id'] = id
        data['text'] = x
        json_data = json.dumps(data)
        with open(json_path, 'at', encoding='utf-8') as fout:
            data['speaker'] = 'human'
            fout.write(json.dumps(data))
            fout.write("\n")
        ws.send(json_data)
        time.sleep(1)
        if x == "[DONE]":
            break
    ws.close()


def on_open(ws):
    """
    Starts a new thread that loops, taking user input and sending it to the websocket.

    :param ws: websocket.WebSocketApp that sends messages to a terminal_manager
    """
    id = _get_rand_id()
    threading.Thread(target=_run, args=(ws, id)).start()


class CustomOnOpen:

    def __init__(self, json_path):
        self.json_path = json_path

    def __call__(self, ws):
        id = _get_rand_id()
        threading.Thread(target=_run, args=(ws, id, self.json_path)).start()


def setup_args():
    """
    Set up args, specifically for the port number.

    :return: A parser that parses the port from commandline arguments.
    """
    parser = ParlaiParser(False, False)
    parser_grp = parser.add_argument_group('Terminal Chat')
    parser_grp.add_argument(
        '--port', default=35496, type=int, help='Port to run the terminal chat server'
    )
    parser_grp.add_argument("--host", default="localhost", type=str, help="Host to run the terminal chat server")
    parser_grp.add_argument("--jsonl", default="conv.jsonl", type=str, help="Where to store conversation")
    return parser.parse_args()


if __name__ == "__main__":
    opt = setup_args()
    port = opt.get('port', 34596)
    host = opt.get('host', 'localhost')
    json_path = opt.get('jsonl', 'conv.jsonl')
    print("Connecting to port: ", port)
    ws = websocket.WebSocketApp(
        "ws://{}:{}/websocket".format(host, port),
        # on_message=on_message,
        on_message=CustomOnMessage(json_path=json_path),
        on_error=on_error,
        on_close=on_close,
    )
    # ws.on_open = on_open
    ws.on_open = CustomOnOpen(json_path=json_path)
    ws.run_forever()
