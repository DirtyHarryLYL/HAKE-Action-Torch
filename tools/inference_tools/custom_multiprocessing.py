#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#############################################
from queue import Queue, Empty
from threading import Thread
import subprocess
import sys
import time
def custom_async(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
    return wrapper

class process_pool(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.processes = []
        self.message_queue = Queue()
        self.activated = False

    def start(self, cmd, idx, cwd):
        p = subprocess.Popen(cmd,  
                            shell=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.STDOUT,
                            cwd=cwd,
                            encoding='utf-8'
                            )
        self.activated |= True
        t = Thread(target=self.enqueue_output, args=(p.stdout, idx))
        t.daemon=True
        t.start()
        self.processes.append((idx, p, t))

    def apply(self, cmd_cwd_list):
        for idx, cmd_cwd in enumerate(cmd_cwd_list):
            cmd, cwd = cmd_cwd
            self.start(cmd, idx, cwd)
        self.daemon()

    def enqueue_output(self, out, i):
        for line in iter(out.readline, b''):
            line_strip = line.strip()
            if len(line_strip) > 0:
                self.message_queue.put_nowait((i, line_strip))
        out.close()
    
    @custom_async
    def daemon(self):
        self.process_num = len(self.processes)
        alive_pool = [1 for _ in range(self.process_num)]
        outputs = []
        while True:
            if sum(alive_pool) == 0 and self.message_queue.empty():
                break
            try: 
                i, out = self.message_queue.get_nowait()
            except Empty:
                pass
            else:
                if self.process_num > 1:
                    sys.stdout.write(' '.join(['pid: {:d}'.format(i), out]))
                else:
                    sys.stdout.write(out)
                sys.stdout.write('\n')
            for idx, p, t in self.processes:
                if p.poll() is not None:
                    alive_pool[idx] = 0

        self.reset()
    
    def wait(self):
        while True:
            if not self.activated:
                break
            else:
                time.sleep(0.1)