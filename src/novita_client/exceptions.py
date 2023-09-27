#!/usr/bin/env python
# -*- coding: UTF-8 -*-

class NovitaError(Exception):
    pass


class NovitaResponseError(NovitaError):
    pass


class NovitaTimeoutError(NovitaError):
    pass
