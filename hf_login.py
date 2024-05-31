from huggingface_hub import login

# Login to hugging face

def login_to_hugging_face():
    # Get the API token from the environment
    token = "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"

    # Login to Hugging Face
    login(token=token)

    print("Logged in to Hugging Face")