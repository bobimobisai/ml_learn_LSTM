def get_qeustion(queue):
    while True:
        input_text = input("Ваш вопрос: ")
        queue.put(input_text)

def ptint_response(output_queue):
    while True:
        response = output_queue.get()
        if response:
            print(f"Ответ: {response[0]}")
            print(f"Уверенность: {response[1]}")
