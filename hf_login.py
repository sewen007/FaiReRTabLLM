from huggingface_hub import login, HfApi

# Login to hugging face

token = "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"


def CheckLogin():
    api = HfApi()

    try:
        user_info = api.whoami()
        if user_info:
            print(f"Already logged in as {user_info['name']}")
        else:
            raise Exception("Not logged in")
    except:
        print("Not logged in. Logging in now...")
        login(token=token)
        print("Logged in successfully")
