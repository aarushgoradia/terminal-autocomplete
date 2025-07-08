import random

# Reinitialize after kernel reset

# Realistic command templates based on common CLI usage
command_templates = [
    "cd {path}",
    "ls {flags}",
    "mkdir {dir}",
    "rm {flags} {file}",
    "cp {file} {path}",
    "mv {file} {path}",
    "cat {file}",
    "touch {file}",
    "nano {file}",
    "vim {file}",
    "python {script}",
    "python3 {script}",
    "pip install {package}",
    "pip uninstall {package}",
    "git status",
    "git add {file}",
    "git commit -m \"{message}\"",
    "git push",
    "git pull",
    "git checkout {branch}",
    "git branch",
    "git merge {branch}",
    "ssh {user}@{host}",
    "scp {file} {user}@{host}:{path}",
    "sudo apt update",
    "sudo apt upgrade",
    "sudo apt install {package}",
    "brew install {package}",
    "top",
    "htop",
    "make",
    "make clean",
    "pytest",
    "./{script}",
    "chmod +x {file}",
    "chown {user}:{user} {file}",
    "tar -xzvf {archive}",
    "unzip {archive}",
    "zip {archive} {file}",
    "kill {pid}",
    "ps aux",
    "curl {url}",
    "wget {url}",
    "source venv/bin/activate",
    "deactivate",
    "python -m venv venv",
    "code {file}",
    "cd ..",
    "cd ~",
    "clear"
]

# Fill-in values
paths = ["projects", "src", "data", "bin", "~/workspace", "/etc"]
files = ["main.py", "test.py", "README.md", "config.json", "script.sh"]
dirs = ["build", "dist", "docs", "venv", "logs"]
flags = ["-la", "-rf", "-v", "-i"]
scripts = ["run.py", "setup.py", "train.py"]
packages = ["torch", "flask", "numpy", "pandas", "pytest"]
archives = ["archive.zip", "backup.tar.gz"]
messages = ["initial commit", "fix bug", "add feature", "update README"]
branches = ["main", "dev", "feature/login"]
users = ["user", "admin", "dev"]
hosts = ["192.168.0.2", "remote.server.com"]
urls = ["https://github.com", "https://example.com"]
pids = [str(random.randint(1000, 9999)) for _ in range(50)]

# Generate 2500+ commands
commands = []
while len(commands) < 2500:
    template = random.choice(command_templates)
    command = template.format(
        path=random.choice(paths),
        file=random.choice(files),
        dir=random.choice(dirs),
        flags=random.choice(flags),
        script=random.choice(scripts),
        package=random.choice(packages),
        archive=random.choice(archives),
        message=random.choice(messages),
        branch=random.choice(branches),
        user=random.choice(users),
        host=random.choice(hosts),
        url=random.choice(urls),
        pid=random.choice(pids)
    )
    commands.append(command)

# Shuffle and save
random.shuffle(commands)

with open("/mnt/data/bash_history_2500.txt", "w") as f:
    f.write("\n".join(commands))

"/mnt/data/bash_history_2500.txt"
