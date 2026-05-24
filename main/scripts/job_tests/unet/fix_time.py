# Author: M. El Aabaribaoune (@um6p)
import glob

for filepath in glob.glob("sweep_scripts/*.sh"):
 with open(filepath, "r") as f:
 content = f.read()
 
 content = content.replace("48:00:00", "24:00:00")
 
 with open(filepath, "w") as f:
 f.write(content)
