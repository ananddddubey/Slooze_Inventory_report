import kagglehub

# Download latest version
path = kagglehub.dataset_download("sloozecareers/slooze-challenge")

print("Path to dataset files:", path)
