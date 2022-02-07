#!/usr/bin/env python3

import codecs
import os
import re
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from shutil import copyfile

__packagename__ = "aspired"

META_PATH = "pyproject.toml"

HERE = os.path.dirname(os.path.realpath(__file__))


# https://stackoverflow.com/questions/1612733/including-non-python-files-with-setup-py
def copy_files(target_path):
    source_path = HERE
    for fn in ["pyproject.toml", "LICENSE", "CHANGELOG.rst", "README.md"]:
        copyfile(os.path.join(source_path, fn), os.path.join(target_path, fn))


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        copy_files(os.path.abspath(__packagename__))


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        copy_files(
            os.path.abspath(os.path.join(self.install_lib, __packagename__))
        )


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^{meta} = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    meta_match_list = re.search(
        r"^{meta} = ([\[\"](?<=\[).+?(?=\])[\"\]])".format(meta=meta),
        meta_file,
        re.M,
    )
    if meta_match:
        return meta_match.group(1)
    elif meta_match_list:
        return eval(meta_match_list.group(1))
    else:
        raise RuntimeError(
            "Unable to find {meta} string or list.".format(meta=meta)
        )


def find_dependencies(meta_file=read(META_PATH)):
    _dependencies = (
        meta_file.split("[tool.poetry.dependencies]" + os.linesep)[1]
        .split(os.linesep + os.linesep + "[tool.poetry.dev-dependencies]")[0]
        .split(os.linesep)
    )
    _dependencies = [i.replace(' = "', "") for i in _dependencies]
    _dependencies = [i.replace('"', "") for i in _dependencies]
    dependencies = []
    for i in _dependencies:
        if i[:6] == "python":
            python_require = i[6:]
        else:
            dependencies.append(i)
    return python_require, dependencies


def find_dev_dependencies(meta_file=read(META_PATH)):
    dependencies = (
        meta_file.split("[tool.poetry.dev-dependencies]" + os.linesep)[1]
        .split(os.linesep + os.linesep + "[build-system]")[0]
        .split(os.linesep)
    )
    dependencies = [i.replace(' = "', "") for i in dependencies]
    dependencies = [i.replace('"', "") for i in dependencies]
    return dependencies


_authors = find_meta("authors")
author_names = [i.split("<")[0].rstrip() for i in _authors]
author_emails = [i.split("<")[1][:-1].rstrip() for i in _authors]

_maintainers = find_meta("authors")
maintainer_names = [i.split("<")[0].rstrip() for i in _maintainers]
maintainer_emails = [i.split("<")[1][:-1].rstrip() for i in _maintainers]

python_require, install_requires = find_dependencies()
extras_require = {"dev": find_dev_dependencies()}

setup(
    name=__packagename__,
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
    version=find_meta("version"),
    packages=find_packages(),
    author=author_names,
    author_email=author_emails,
    maintainer=maintainer_names,
    maintainer_email=maintainer_emails,
    url=find_meta("homepage"),
    license=find_meta("license"),
    description=find_meta("description"),
    long_description=read(os.path.join(HERE, "README.md")),
    long_description_content_type="text/markdown",
    zip_safe=False,
    data_files=[("", ["pyproject.toml"])],
    package_data={"": ["../pyproject.toml"]},
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=python_require,
)
