from modelscope import snapshot_download

def main():
    snapshot_download(repo_id="qwen/Qwen2-7B", cache_dir="/root/autodl-tmp/model/Llama-2-7b-chat") # 指定本地地址保存模型
    print('ok')
    
if __name__ == "__main__":
    main()