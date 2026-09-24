"""Scoped server operations; the authorized SSH password is never persisted."""
import getpass
import json
from pathlib import Path

import paramiko


client = paramiko.SSHClient()
client.load_host_keys(str(Path.home() / '.ssh/known_hosts'))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect('connect.weste.seetacloud.com', port=35324, username='root',
               password=getpass.getpass('SSH password: '), look_for_keys=False,
               allow_agent=False, timeout=20)
try:
    while True:
        print('READY_JSON', flush=True)
        request = json.loads(input())
        if request.get('quit'):
            break
        try:
            if 'command' in request:
                _, out, err = client.exec_command(request['command'], timeout=55)
                print(out.read().decode('utf-8', errors='replace'), flush=True)
                print(err.read().decode('utf-8', errors='replace'), flush=True)
                print('EXIT', out.channel.recv_exit_status(), flush=True)
            with client.open_sftp() as sftp:
                for directory in request.get('mkdir', []):
                    try:
                        sftp.mkdir(directory)
                    except IOError:
                        sftp.stat(directory)
                for path in request.get('list', []):
                    print('LIST', path, json.dumps(sftp.listdir(path)), flush=True)
                for path in request.get('read', []):
                    print('READ', path, sftp.open(path).read().decode('utf-8', errors='replace'), flush=True)
                for source, destination in request.get('put', []):
                    with sftp.open(destination, 'wx') as stream:
                        stream.write(Path(source).read_bytes())
                    print('PUT', destination, flush=True)
                for source, destination in request.get('get', []):
                    target = Path(destination)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise ValueError('Local destination already exists')
                    sftp.get(source, str(target))
                    print('GET', destination, flush=True)
        except Exception as exc:
            print('ERROR', type(exc).__name__, str(exc), flush=True)
finally:
    client.close()
