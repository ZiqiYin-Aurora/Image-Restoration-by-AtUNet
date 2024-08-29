# -*- coding:utf-8 -*-
import os
import stat
import traceback
import paramiko

# ssh -p 41576 root@region-5.autodl.com
# GCy5jMxvWh


def to_str(bytes_or_str):
    """
    把byte类型转换为str
    :param bytes_or_str:
    :return:
    """
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode('utf-8')
    else:
        value = bytes_or_str
    return value


class SSHConnection(object):

    def __init__(self, host_dict):
        self.host = host_dict['host']
        self.port = host_dict['port']
        self.username = host_dict['username']
        self.pwd = host_dict['pwd']
        self.__k = None

    def connect(self):
        transport = paramiko.Transport((self.host, self.port))
        transport.connect(username=self.username, password=self.pwd)
        self.transport = transport

    def close(self):
        self.transport.close()

    def run_cmd(self, command):
        """
         执行shell命令,返回字典
         return {'color': 'red','res':error}或
         return {'color': 'green', 'res':res}
        :param command:
        :return:
        """
        ssh = paramiko.SSHClient()
        ssh._transport = self.transport
        # 执行命令
        stdin, stdout, stderr = ssh.exec_command(command)
        # 获取命令结果
        res = to_str(stdout.read())
        # 获取错误信息
        error = to_str(stderr.read())
        # 如果有错误信息，返回error
        # 否则返回res
        if error.strip():
            return {'color': 'red', 'res': error}
        else:
            return {'color': 'green', 'res': res}

    def upload(self, local_path, target_path):
        # 连接，上传
        sftp = paramiko.SFTPClient.from_transport(self.transport)
        # 将location.py 上传至服务器 /tmp/test.py
        sftp.put(local_path, target_path, confirm=True)
        # print(os.stat(local_path).st_mode)
        # 增加权限
        # sftp.chmod(target_path, os.stat(local_path).st_mode)
        sftp.chmod(target_path, 0o755)  # 注意这里的权限是八进制的，八进制需要使用0o作为前缀

    def download(self, target_path, local_path):
        # 连接，下载
        sftp = paramiko.SFTPClient.from_transport(self.transport)
        # 将location.py 下载至服务器 /tmp/test.py
        sftp.get(target_path, local_path)

    # 递归遍历本地服务器指定目录下的所有文件
    def _get_all_files_in_local_dir(self, local_dir):
        all_files = list()

        for root, dirs, files in os.walk(local_dir, topdown=True):
            for file in files:
                filename = os.path.join(root, file)
                all_files.append(filename)

        return all_files

    # 递归遍历远程服务器指定目录下的所有文件
    def _get_all_files_in_remote_dir(self, sftp, remote_dir):
        all_files = list()
        if remote_dir[-1] == '/':
            remote_dir = remote_dir[0:-1]

        files = sftp.listdir_attr(remote_dir)
        for file in files:
            filename = remote_dir + '/' + file.filename

            if stat.S_ISDIR(file.st_mode):  # 如果是文件夹的话递归处理
                all_files.extend(self._get_all_files_in_remote_dir(sftp, filename))
            else:
                all_files.append(filename)

        return all_files

    def sftp_download_dir(self, remote_dir, local_dir):
        try:
            sftp = paramiko.SFTPClient.from_transport(self.transport)
            all_files = self._get_all_files_in_remote_dir(sftp, remote_dir)
            print("Downloading from remote server ......")
            print(local_dir + "  <<<<<  " + remote_dir)

            for file in all_files:
                if '.png' in file:
                    local_filename = file.replace(remote_dir, local_dir)
                    local_filepath = os.path.dirname(local_filename)

                    if not os.path.exists(local_filepath):
                        os.makedirs(local_filepath)

                    sftp.get(file, local_filename)
            print("---- Down! ----")
        except:
            print('ssh get dir from master failed.')
            print(traceback.format_exc())

    def sftp_upload_dir(self, local_dir, remote_dir):
        try:
            sftp = paramiko.SFTPClient.from_transport(self.transport)
            if remote_dir[-1] == "/":
                remote_dir = remote_dir[0:-1]
            all_files = self._get_all_files_in_local_dir(local_dir)
            print("Uploading to remote server ......")
            print(local_dir + "  >>>>>  " + remote_dir)

            for file in all_files:
                remote_filename = file.replace(local_dir, remote_dir)
                remote_path = os.path.dirname(remote_filename)

                try:
                    sftp.stat(remote_path)
                except:
                    os.popen('mkdir -p %s' % remote_path)

                sftp.put(file, remote_filename, confirm=True)
                sftp.chmod(remote_filename, 0o755)
            print("---- Down! ----")
        except:
            print('ssh get dir from master failed.')
            print(traceback.format_exc())

    # 销毁
    def __del__(self):
        self.close()


host = 'region-5.autodl.com'
port = 41576
username = 'root'
password = 'GCy5jMxvWh'
# remote_path = '/root/autodl-tmp/FYP-Yin/results'
# local_path = '/Volumes/Work Space/Codes/Python/FYP-Yin/results'

remote_dict = { 'host': host,
                'port': port,
                'username': username,
                'pwd': password}

if __name__ == '__main__':
    mySSH = SSHConnection(remote_dict)
    mySSH.connect()
    res = mySSH.run_cmd("ls -l")
    print(res['res'].strip('\n'))
    # mySSH.sftp_download_dir(remote_path, local_path)
    mySSH.close()
