#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Check 3rd-party dependencies
try:
    import boto3  # noqa: F401
    import botocore  # noqa: F401
    import joblib  # noqa: F401
    import websocket  # noqa: F401
    try:
        import sh
    except ImportError:
        # fallback: emulate the sh API with pbs
        import pbs


        class Sh(object):
            def __getattr__(self, attr):
                return pbs.Command(attr)
        sh = Sh() # noqa: F401

except ImportError:
    raise SystemExit(
        "Please install 3rd-party dependencies by running: "
        "pip install boto3 joblib websocket-client sh"
    )
