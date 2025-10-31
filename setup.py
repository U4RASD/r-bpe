from setuptools import setup, find_packages
import subprocess
from setuptools.command.install import install
from setuptools.command.develop import develop

# Hardcoded version
VERSION = "0.1.0"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

def run_camel_data_command():
    print("Running camel_data -i disambig-mle-all")
    try:
        subprocess.check_call(['camel_data', '-i', 'disambig-mle-all'])
        print("Command completed successfully")
    except Exception as e:
        print(f"Error running command: {e}")

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        self.execute(run_camel_data_command, [], msg="Running post-install command")

class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        self.execute(run_camel_data_command, [], msg="Running post-develop command")

setup(
    name="rbpe",
    version=VERSION,
    description="R-BPE: Improving BPE-Tokenizers with Token Reuse",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Unit for Research Studies In Arabic and Social Digital Spaces",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "": ["data/*"],
    },
    python_requires=">=3.10.12",
    install_requires=requirements,
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    }
) 