from setuptools import setup, find_packages

setup(
    name="rag_eval",
    version="0.1.0",
    description="A framework for evaluating Retrieval-Augmented Generation (RAG) pipelines.",
    author="Keshav Sreekantham, Kaya Tacer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "deepeval>=2.7.6",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
