#!/usr/bin/env python
"""Train m5C-SSNet with the default configuration."""

from utils.config import get_config
from utils.trainer import train

if __name__ == "__main__":
    train(get_config())
