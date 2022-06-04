from setuptools import setup
setup(
    name="dpr",
    packages=["dpr_mini"],
    setup_requires=[
        "setuptools>=18.0",
    ],
    install_requires=[
        "faiss-cpu>=1.6.1",
        "filelock",
        "numpy",
        "regex",
        "transformers>=4.3",
        "tqdm>=4.27",
        "wget",
        "spacy>=2.1.8",
        "hydra-core>=1.0.0",
        "omegaconf>=2.0.1",
        "jsonlines",
        "soundfile",
        "editdistance",
    ],
)