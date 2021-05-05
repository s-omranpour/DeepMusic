from setuptools import setup, find_packages

from deepnote import __version__, __package_name__, __description__, __repository_url__

with open('requirements.txt', 'r') as f:
    dependencies = f.read().split('\n')

setup(
    name=__package_name__,  # Required
    version=__version__,  # Required
    description=__description__,  # Optional
    # long_description='',  # Optional
    # long_description_content_type='text/markdown',  # Optional (see note above)
    url=__repository_url__,  # Optional
    author='Soroush Omranpour',  # Optional
    author_email='soroush.333@gmail.com',  # Optional
    keywords='midi, deep learning, music',  # Optional
    package_dir={'': 'deepnote'},  # Optional
    packages=find_packages(where='deepnote'),  # Required
    python_requires='>=3.7, <4',
    install_requires=dependencies,  # Optional
    license="MIT license",

    project_urls={  # Optional
        'Bug Reports': __repository_url__ + '/issues',
        'Source': __repository_url__,
    },
)