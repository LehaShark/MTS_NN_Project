MTS-intrernship
Human detector
Video https://www.youtube.com/watch?v=6ueyBesTxTQ&ab_channel=UcchashSarkar

ans

Wrist detector
Video https://www.youtube.com/watch?v=whZwZ8jeq5E&ab_channel=StarTJ

To run the project
python demo_camera_V2.0.py
resultgirl

result

После обрезки видео программа выводит все кадры в черно-белые фото

frame74

Result
Peek 2021-02-21 21-45

Чтобы запустить demo_camera_V2.0.py произведем следующие шаги:
Шаг 1. Клонирование и установка зависимостей
$ git clone https://github.com/SyBorg91/pose-estimation-detection
$ cd pose-estimation-detection
$ pip3 install -r requirements.txt
Шаг 2. Поменять версию tensorflow на 1.13.2
$ pip install tensorflow==1.13.2
Шаг3. Запустить программу
$ python demo_camera_V2.0.py --model=mobilenet_thin --resize=432x368 --camera=0
