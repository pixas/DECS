from contextlib import contextmanager
from functools import wraps
import io
import json 
import os
import shutil

import torch 
from petrel_client.client import Client
import torch.distributed as dist


def print_on_main(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def proxy_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ori_http_proxy = os.environ.get('http_proxy')  # 获取原始的http_proxy值
        ori_https_proxy = os.environ.get("https_proxy")
        os.environ['http_proxy'] = ''  # 在函数执行前将http_proxy设为空字符串
        os.environ['https_proxy'] = ''
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        result = func(*args, **kwargs)  # 执行函数
        os.environ['http_proxy'] = ori_http_proxy if ori_http_proxy is not None else ''  # 函数执行后恢复原始的http_proxy值
        os.environ['https_proxy'] = ori_https_proxy if ori_https_proxy is not None else ''
        os.environ['HTTP_PROXY'] = ori_http_proxy if ori_http_proxy is not None else ''
        os.environ['HTTPS_PROXY'] = ori_https_proxy if ori_https_proxy is not None else ''
        return result
    return wrapper

@contextmanager
def add_proxy():
    proxy_url = os.environ.get("proxy_url")
    os.environ['http_proxy'] = os.environ['HTTP_PROXY'] = os.environ['https_proxy'] = os.environ['HTTPS_PROXY'] = proxy_url
    try:
        yield
    finally:
        os.environ['http_proxy'] = os.environ['HTTP_PROXY'] = os.environ['https_proxy'] = os.environ['HTTPS_PROXY'] = ''


class CephOSSClient:
    
    @proxy_decorator
    def __init__(self, conf_path: str = "~/petreloss.conf") -> None:
        self.client = Client(conf_path)
    
    @proxy_decorator
    def read_json(self, json_path, **kwargs):
        if json_path.startswith("s3://"):
            cur_bytes = self.client.get(json_path)
            if cur_bytes != "":
                data = json.loads(cur_bytes, **kwargs)
            else:
                data = []
            # data = json.loads(self.client.get(json_path), **kwargs)
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f, **kwargs)
        return data 

    @proxy_decorator
    def write_json(self, json_data, json_path, **kwargs):
        if json_path.startswith("s3://"):
            if json_data == []:
                self.client.put(json_path, "".encode("utf-8"))
            else:
                self.client.put(json_path, json.dumps(json_data, **kwargs).encode("utf-8"))
        else:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, **kwargs)
        return 1

    @proxy_decorator
    def read_jsonl(self, jsonl_path):
        if jsonl_path.startswith("s3://"):
            bytes = self.client.get(jsonl_path)
            data = bytes.decode('utf-8').split("\n")
            data = [json.loads(x) for x in data if x != ""]
        else:
            data = [json.loads(x) for x in open(jsonl_path, encoding='utf-8', mode='r')]
        return data 
    
    @proxy_decorator
    def write_jsonl(self, jsonl_data, jsonl_path, **kwargs):
        if jsonl_path.startswith("s3://"):
            if jsonl_data == []:
                self.client.put(jsonl_path, "".encode("utf-8"))
                return 1
            if isinstance(jsonl_data, list):
                large_bytes = "\n".join([json.dumps(x, ensure_ascii=False) for x in jsonl_data]).encode("utf-8")
            else:
                large_bytes = (json.dumps(x, ensure_ascii=False) + "\n").encode('utf-8')
            with io.BytesIO(large_bytes) as f:
                self.client.put(jsonl_path, f)
        else:
            with open(jsonl_path, 'w', **kwargs) as f:
                for x in jsonl_data:
                    f.write(json.dumps(x, ensure_ascii=False))
                    f.write("\n")
        return 1

    @proxy_decorator
    def read_txt(self, txt_path):
        if txt_path.startswith("s3://"):
            bytes = self.client.get(txt_path)
            data = bytes.decode('utf-8')
        else:
            with open(txt_path, 'r', encoding='utf-8') as f:
                data = f.read()
        return data 

    @proxy_decorator
    def write_text(self, txt_data, txt_path, mode='w'):
        if txt_path.startswith("s3://"):
            large_bytes = txt_data.encode("utf-8")
            with io.BytesIO(large_bytes) as f:
                self.client.put(txt_path, f)
        else:
            with open(txt_path, mode, encoding='utf-8') as f:
                f.write(txt_data)
        return 1
    
    @proxy_decorator
    def save_checkpoint(self, data, path, **kwargs):
        if "s3://" not in path:
            assert os.path.exists(path), f'No such file: {path}'
            torch.save(data, path, **kwargs)
        else:
            with io.BytesIO() as f:
                torch.save(data, f, **kwargs)
                self.client.put(path, f.getvalue())
        return 1 

    @proxy_decorator
    def load_checkpoint(self, path, map_location=None, **kwargs):
        if "s3://" not in path:
            assert os.path.exists(path), f'No such file: {path}'
            return torch.load(path, map_location=map_location, **kwargs)
        else:
            file_bytes = self.client.get(path)
            buffer = io.BytesIO(file_bytes)
            
            res = torch.load(buffer, map_location=map_location, **kwargs)
            return res
    
    @proxy_decorator
    def exists(self, file_path):
        if "s3://" not in file_path:
            return os.path.exists(file_path)
        else:
            file_exist = self.client.contains(file_path)
            if not file_exist:
                contents = list(self.client.list(file_path))
                if len(contents) == 0:
                    return False 
                else:
                    return True 
            else:
                return True
            
    
    @proxy_decorator
    def remove(self, file_path):
        if "s3://" not in file_path:
            return shutil.rmtree(file_path, ignore_errors=True)
            return os.remove(file_path)
        else:
            if self.client.isdir(file_path):
                all_uris = self.listdir(file_path)
                for uri in all_uris:
                    self.remove(file_path + ("/" if not file_path.endswith("/") else "") + uri)
            else:
                
                return self.client.delete(file_path)
    
    @proxy_decorator
    def abspath(self, file_path):
        if "s3://" not in file_path:
            return os.path.abspath(file_path)
        else:
            # For S3 paths, we return the path as is since it is already an absolute path
            return file_path
    
    @proxy_decorator
    def read_csv(self, path):
        if "s3://" in path:
            bytes = self.client.get(path)
            data = bytes.decode('utf-8').split("\n")
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = f.readlines()
        return data

    def read(self, path: str):
        s3_prefix = "s3://syj_test"
        local_prefix = "/mnt/petrelfs/jiangshuyang.p/oss"
        if "local_prefix" in path:
            path = path.replace(local_prefix, s3_prefix)
        mapping_processing = {
            "csv": self.read_csv,
            "json": self.read_json,
            "jsonl": self.read_jsonl,
            "txt": self.read_txt,
            "log": self.read_txt
        }
        suffix = path.split(".")[-1]
        try:
            return mapping_processing[suffix](path)
        except:
            s3_prefix = "s3://syj_test"
            local_prefix = "/mnt/petrelfs/jiangshuyang.p/oss"
            path = path.replace(local_prefix, s3_prefix)
            return mapping_processing[suffix](path)
    
    def write(self, data, path: str, **kwargs):
        mapping_processing = {
            "csv": self.write_text,
            "json": self.write_json,
            "jsonl": self.write_jsonl,
            "txt": self.write_text,
            "log": self.write_text
        }
        suffix = path.split(".")[-1]
        try:
            return mapping_processing[suffix](data, path, **kwargs)
        except Exception as e:
            print(e)
            s3_prefix = "s3://syj_test"
            local_prefix = "/mnt/petrelfs/jiangshuyang.p/oss"
            path = path.replace(local_prefix, s3_prefix)
            return mapping_processing[suffix](data, path, **kwargs)

    @proxy_decorator
    def listdir(self, path):
        if "s3://" in path:
            output = [x for x in list(self.client.list(path)) if x != ""]
            return output
        else:
            return os.listdir(path)
    
    @proxy_decorator
    def isdir(self, path):
        if "s3://" in path:
            return self.client.isdir(path)
        else:
            return os.path.isdir(path)
    
    


client = CephOSSClient("~/petreloss.conf")
