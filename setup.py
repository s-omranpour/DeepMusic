from setuptools import setup


setup(
    name='deepnote',
    version='0.1.0',
    description='A python package to manipulate MIDI files, representing them and converting them into a proper format for feeding to machine learning models.',
    url='https://github.com/s-omranpour/DeepNote',
    author='Soroush Omranpour',
    author_email='soroush.333@gmail.com',
    keywords='midi, deep learning, music',
    packages=['deepnote'],
    python_requires='>=3.7, <4',
    install_requires=[
        'numpy >= 1.7.0', 
        'miditoolkit >= 0.1.14',
        'chorder >= 0.1.2',
        'midi2audio >= 0.1.1'
    ],
    license="MIT license",

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/s-omranpour/DeepNote/issues',
        'Source': 'https://github.com/s-omranpour/DeepNote',
    },
)