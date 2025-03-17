import os
import shutil
from pathlib import Path

import pytest
from pytest_lazyfixture import lazy_fixture

from remote_handler import file_handler


@pytest.fixture()
def remote_fs(config_dir):
    slurm_config_file = config_dir / "remote.json"
    fs = file_handler.RemoteFileSystem(slurm_config_file=slurm_config_file)
    target_dir = fs.slurm_config.get("savepath")
    if target_dir is None:
        target_dir = fs.create_tempdir()

    return fs, Path(target_dir)


@pytest.fixture()
def local_fs(data_dir):
    fs = file_handler.LocalFileSystem()
    target_dir = data_dir / "temp" / "TestFileSystem" / "temp"

    if fs.exists(target_dir):
        fs.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir, exist_ok=True)

    return fs, target_dir


@pytest.fixture()
def local_dir(data_dir):
    return data_dir / "assets" / "TestFileSystem"


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            lazy_fixture("remote_fs"),
            marks=[pytest.mark.remote],
        ),
        pytest.param(
            lazy_fixture("local_fs"),
            marks=[pytest.mark.local],
        ),
    ],
)
def instance(request):
    return request.param


class TestFileSystem:
    def test_put(self, instance, local_dir):
        fs, target_dir = instance
        filename = "single_file.txt"
        fs.put(local_dir / filename, target_dir / filename)
        assert fs.exists(target_dir / filename)

    def test_get(self, instance, local_dir):
        fs, target_dir = instance
        filename = "single_file.txt"
        filename2 = "single_file2.txt"

        if os.path.exists(local_dir / filename2):
            os.remove(local_dir / filename2)

        fs.put(local_dir / filename, target_dir / filename2)
        fs.get(target_dir / filename2, local_dir / filename2)

        assert os.path.exists(local_dir / filename2)
        # os.remove(local_dir / filename2)

    def test_put_dir(self, instance, local_dir):
        fs, target_dir = instance
        dirname = "directory_with_contents"

        fs.put_dir(local_dir / dirname, target_dir / dirname)

        assert fs.exists(target_dir / dirname)

    def test_get_dir(self, instance, local_dir):
        fs, target_dir = instance
        dirname = "directory_with_contents"
        dirname2 = "directory_with_contents2"

        if os.path.exists(local_dir / dirname2):
            shutil.rmtree(local_dir / dirname2)

        fs.put_dir(local_dir / dirname, target_dir / dirname2)
        fs.get_dir(target_dir / dirname2, local_dir / dirname2)

        assert os.path.exists(local_dir / dirname2)
        # shutil.rmtree(local_dir / dirname2)

    def test_create_tempdir(self, instance):
        fs, target_dir = instance
        temppath = fs.create_tempdir()
        print(temppath)
        assert fs.exists(temppath)


def test_reconnect(remote_fs, local_dir):
    fs, target_dir = remote_fs
    filename = "single_file.txt"
    fs.put(local_dir / filename, target_dir / filename)
    assert fs.exists(target_dir / filename)

    fs.sftp_client.close()

    fs.put(local_dir / filename, target_dir / filename)
    assert fs.exists(target_dir / filename)
