import paramiko

import os
from stat import S_ISDIR
import time


class Communicator:
    def __init__(self, host, port, username, password):
        try:
            self.transport = paramiko.Transport((host, port))
            self.transport.connect(None, username, password)
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(host, port, username, password)
            self.sftp = paramiko.SFTPClient.from_transport(self.transport)
            print('Created communicator')
        except Exception as e:
            self.transport = None
            self.ssh = None
            self.sftp = None
            self.error_msg = str(e)
            print('Failed creating communicator', self.error_msg)

    def _valid(self):
        return not self._invalid()

    def _invalid(self):
        return self.ssh is None or self.sftp is None

    def exists(self, remote_dir_path):
        # returns True if remote_dir_path exists
        assert self._valid()
        try:
            self.sftp.chdir(remote_dir_path)
            return True
        except IOError:
            return False

    def remove(self, remote_dir_path):
        assert self._valid()
        self.ssh.exec_command('rm -rf ' + remote_dir_path)

    def create_dir(self, remote_dir_path):
        # create dir and returns True if path doesn't exist already
        # doesn't create dir and returns False if path exists already
        assert self._valid()
        try:
            self.sftp.chdir(remote_dir_path)
            return False
        except IOError:
            # self.sftp.mkdir(remote_dir_path)
            self.ssh.exec_command('mkdir -p ' + remote_dir_path)
            return True

    def upload_dir(self, local_dir_path, remote_dir_path):
        assert self._valid()
        self.create_dir(remote_dir_path)
        for item in os.listdir(local_dir_path):
            local_path = os.path.join(local_dir_path, item)
            remote_path = os.path.join(remote_dir_path, item)
            if os.path.isfile(local_path):
                print('UPLOADING {} to {}'.format(local_path, remote_path))
                while True:
                    try:
                        self.sftp.put(local_path, remote_path)
                        print('UPLOADING SUCESSS')
                        break
                    except Exception as e:
                        print('UPLOADING FAIL:', e)
                        print('RETRY UPLOADING {} to {}'.format(local_path, remote_path))
                        time.sleep(3)
            else:
                self.upload_dir(local_path, remote_path)

    def download_dir(self, remote_dir_path, local_dir_path):
        assert self._valid()
        if not os.path.isdir(local_dir_path):
            os.makedirs(local_dir_path)
        for item in self.sftp.listdir(remote_dir_path):
            # print(item, S_ISDIR(item.st_mode))
            local_path = os.path.join(local_dir_path, item)
            remote_path = os.path.join(remote_dir_path, item)
            st_mode = self.sftp.stat(remote_path).st_mode
            if S_ISDIR(st_mode):
                self.download_dir(remote_path, local_path)
            else:
                if self.different(remote_path, local_path):
                    print('Downloading {} to {}'.format(remote_path, local_path))
                    self.sftp.get(remote_path, local_path)
                    mtime = self.sftp.lstat(remote_path).st_mtime
                    os.utime(local_path, (mtime, mtime))

    def different(self, remote_path, local_path):
        if not os.path.isfile(local_path):
            return True
        remote_attr = self.sftp.lstat(remote_path)
        local_stat = os.stat(local_path)
        # print('Different? {} {} {}=?{} {}=?{}'.format(remote_path, local_path, remote_attr.st_size, local_stat.st_size, remote_attr.st_mtime, local_stat.st_mtime))
        return remote_attr.st_size != local_stat.st_size or \
               remote_attr.st_mtime != local_stat.st_mtime

    def close(self):
        if self._valid():
            print('Closing sftp')
            self.sftp.close()
            print('Closing ssh')
            self.ssh.close()
            print('Closing transport')
            self.transport.close()


if __name__ == '__main__':
    # test communicator
    pass
