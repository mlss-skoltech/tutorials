from distutils.core import setup

setup(
    name="mlss2019bdl",
    version="0.1",
    description="""Service code for MLSS2019 Tutorial on Bayesian Deep Learning""",
    license="MIT License",
    author="Ivan Nazarov, Yarin Gal",
    author_email="ivan.nazarov@skolkovotech.ru",
    packages=[
        "mlss2019bdl",
    ],
    install_requires=[
        "numpy",
        "tqdm",
        "matplotlib",
        "torch",
        "torchvision",
    ]
)
