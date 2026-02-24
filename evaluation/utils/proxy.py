import os
from contextlib import contextmanager

@contextmanager
def add_proxy():
    proxy_url = os.environ.get("proxy_url")
    if proxy_url is not None:
        os.environ['http_proxy'] = os.environ['HTTP_PROXY'] = os.environ['https_proxy'] = os.environ['HTTPS_PROXY'] = proxy_url
    try:
        yield
    finally:
        os.environ['http_proxy'] = os.environ['HTTP_PROXY'] = os.environ['https_proxy'] = os.environ['HTTPS_PROXY'] = ''


@contextmanager
def add_openai_proxy():
    proxy_url = os.environ.get("gpt_proxy_url")
    if proxy_url is not None:
        os.environ['http_proxy'] = os.environ['HTTP_PROXY'] = os.environ['https_proxy'] = os.environ['HTTPS_PROXY'] = proxy_url
    try:
        yield
    finally:
        os.environ['http_proxy'] = os.environ['HTTP_PROXY'] = os.environ['https_proxy'] = os.environ['HTTPS_PROXY'] = ''
