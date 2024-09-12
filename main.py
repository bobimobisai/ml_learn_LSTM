from load_model import start_update_thread
import queue
from test import get_qeustion, ptint_response
import threading


_queue = queue.Queue()
output_queue = queue.Queue()


if __name__ == "__main__":
    start_update_thread(_queue, output_queue)
    th1 = threading.Thread(target=get_qeustion, args=(_queue,))
    th2 = threading.Thread(target=ptint_response, args=(output_queue,))

    th1.start()
    th2.start()
