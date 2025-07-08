# This file generates a random bash history file with 500 commands.

import random

# Common command patterns
base_commands = [
    "ls", "cd", "mkdir", "rm", "touch", "cat", "nano", "vim", "clear",
    "git status", "git add .", "git commit -m \"{msg}\"", "git push",
    "python {script}", "python3 {script}", "pip install {pkg}", "pip uninstall {pkg}",
    "sudo apt update", "sudo apt install {pkg}", "brew install {pkg}",
    "ssh {user}@{host}", "scp {file} {user}@{host}:{path}", "tmux new -s {session}",
    "top", "htop", "make", "make clean", "make all", "./{script}", "clang {file}.c -o {file}",
    "g++ {file}.cpp -o {file}", "./{file}", "pytest", "source venv/bin/activate",
    "deactivate", "python3 -m venv venv", "cd ..", "cd ~", "cd /etc", "ls -la",
    "history", "chmod +x {file}", "chown {user}:{user} {file}", "curl {url}",
    "wget {url}", "whoami", "ps aux", "kill {pid}", "kill -9 {pid}", "tar -xzvf {file}.tar.gz",
    "zip {archive}.zip {file}", "unzip {archive}.zip"
]

# Fill-ins
msgs = ["initial commit", "update README", "fix bug", "add feature", "cleanup"]
scripts = ["train.py", "predict.py", "main.py", "script.py"]
pkgs = ["torch", "transformers", "numpy", "flask", "requests", "pytest"]
users = ["user", "dev", "admin"]
hosts = ["remote.server.com", "192.168.0.2", "dev.box"]
files = ["main", "utils", "app", "test"]
paths = ["~/workspace", "~/projects", "/home/dev"]
sessions = ["dev", "ml", "test"]
urls = ["http://example.com", "https://github.com", "https://api.openai.com"]
pids = [str(random.randint(1000, 9999)) for _ in range(10)]

# Generate command list
commands = []
while len(commands) < 500:
    cmd = random.choice(base_commands)
    cmd = cmd.replace("{msg}", random.choice(msgs))
    cmd = cmd.replace("{script}", random.choice(scripts))
    cmd = cmd.replace("{pkg}", random.choice(pkgs))
    cmd = cmd.replace("{user}", random.choice(users))
    cmd = cmd.replace("{host}", random.choice(hosts))
    cmd = cmd.replace("{file}", random.choice(files))
    cmd = cmd.replace("{path}", random.choice(paths))
    cmd = cmd.replace("{session}", random.choice(sessions))
    cmd = cmd.replace("{url}", random.choice(urls))
    cmd = cmd.replace("{pid}", random.choice(pids))
    commands.append(cmd)

# Shuffle to simulate realistic command ordering
random.shuffle(commands)

# Save to file
with open("/mnt/data/bash_history.txt", "w") as f:
    f.write("\n".join(commands))
"/mnt/data/bash_history.txt"
