from roboflow import Roboflow

# 1. Initialize with the API Key from your snippet
rf = Roboflow(api_key="jn5jL4ugFlKzor0IPL9S")

# 2. Access Workspace and Project
# 'projectsdata' comes from workspace_name
# 'find-paperballs' is the project name
workspace = rf.workspace("projectsdata")
project = workspace.project("find-paperballs")

# 3. Download Version 2 (Matches 'find-paperballs-2')
# This will likely create a folder named 'find-paperballs-2' on your laptop
print("⬇️ Downloading Version 2...")
version = project.version(2)
dataset = version.download("yolov8")

print("✅ Download Complete! Check for the 'find-paperballs-2' folder.")