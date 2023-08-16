import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="conceptqr",
    version="0.0.1",
    author="Andrew Parry",
    author_email="a.parry.1@research.gla.ac.uk",
    description="Implementation of generative QR with LLM concept expansion",
    packages=setuptools.find_packages(['src']),
)