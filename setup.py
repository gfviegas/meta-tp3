import setuptools
from meta_tp3.version import Version

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="meta_tp3",
    version=Version("1.0.0").number,
    description="Trabalho Prático 3 de Metaheuristicas",
    long_description=open("README.md").read().strip(),
    author="Gustavo Viegas, Mário Sergio",
    author_email="gustavo.viegas@ufv.br, mario.cabral@ufv.br",
    #  url='http://path-to-my-meta-tp3',
    py_modules=["meta_tp3"],
    install_requires=required,
)
