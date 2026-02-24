import os
from contextlib import contextmanager
from functools import wraps
import requests

@contextmanager
def add_proxy():
    proxy_url = os.environ.get("proxy_url")
    if proxy_url is not None:
        os.environ['http_proxy'] = os.environ['HTTP_PROXY'] = os.environ['https_proxy'] = os.environ['HTTPS_PROXY'] = proxy_url
    try:
        yield
    finally:
        os.environ['http_proxy'] = os.environ['HTTP_PROXY'] = os.environ['https_proxy'] = os.environ['HTTPS_PROXY'] = ''



def remove_proxy():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_http_proxy = os.environ.get('http_proxy')
            original_https_proxy = os.environ.get('https_proxy')
            null_proxy = ""
            os.environ['http_proxy'] = null_proxy
            os.environ['https_proxy'] = null_proxy

            try:
                result = func(*args, **kwargs)
            finally:
                if original_http_proxy is not None:
                    os.environ['http_proxy'] = original_http_proxy
                else:
                    del os.environ['http_proxy']

                if original_https_proxy is not None:
                    os.environ['https_proxy'] = original_https_proxy
                else:
                    del os.environ['https_proxy']

            return result
        return wrapper
    return decorator



def send_feishu_message(metrics):
    exp_name = metrics.pop("exp_name", None)
    webhook_url = os.environ.get("webhook_url")
    fields = [
        {"is_short": True, "text": {"content": f"pixas **{key}**\n{value}", "tag": "lark_md"}}
        for key, value in metrics.items()
    ]
    data = {
    "msg_type": "interactive",
    "card": {
        "elements": [{
            "tag": "div",
            "text": {"content": f"{exp_name}模型训练结果", "tag": "lark_md"}
        }, {
            "tag": "div",
            "fields": fields
        }],
        "header": {"title": {"content": "训练完成通知", "tag": "plain_text"}}
    }
}
    response = requests.post(webhook_url, json=data)
    return response.json()
