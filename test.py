import logging


logging.basicConfig(level=logging.INFO)


def get_qeustion(queue):
    while True:
        input_text = input("Ваш вопрос: ")
        queue.put(input_text)


def ptint_response(output_queue):
    while True:
        response = output_queue.get()
        if response:
            if response[1] > 0.5:
                logging.info(f"Ответ: {response[0]}")
                logging.info(f"Уверенность: {response[1]}")
            else:
                logging.info(f"Ответ: Не уверен...")
                logging.info(f"Уверенность: {response[1]}")
