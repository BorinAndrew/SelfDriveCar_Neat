import sys
import pickle
from Car import Car
from ScreenRecorder import ScreenRecorder
from utils import *

import neat
import pygame

# Задаём пути к необходимым файлам

car_path = 'car2.png'
track_path = 'map.png'
config_path = 'config.txt'


def run_simulation(genomes, config, width=1920, height=1080, FPS=60, record_video=False,
                   save_video_path='best_genom.avi'):
    # Функция запуска симуляции
    nets = []  # Список нейронок, которые будут управлять машинами
    cars = []  # Список машин

    # Инициализируем PyGame и экран
    pygame.init()
    screen = pygame.display.set_mode((width, height))

    if record_video:
        recorder = ScreenRecorder(width, height, FPS, save_video_path)  # Объект для записи вождения лучшего генома

    # Для всех переданных геномов создать нейронную сеть
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    clock = pygame.time.Clock()

    game_map = pygame.image.load(track_path).convert()  # Загрузка карты

    # Счётчик для лимита по времени на каждую генерацию
    counter = 0

    while True:
        # Обработка выхода
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # Для каждой машины по данным с сенсоров получаем выход с нейронной сети
        # и выбираем действие с найбольшим значением
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += car.rotation_vel  # Поворот на лево
            elif choice == 1:
                car.angle -= car.rotation_vel  # На право
            elif choice == 2:
                if car.speed - car.acceleration >= 12:
                    car.speed -= car.acceleration  # Притормозить
            else:
                car.speed += car.acceleration  # Ускорится

        # Проверка остались ли ещё неразбитые машины
        # Увеличиваем Fitness если да и завершаем генерацию если нет
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40:  # Остонавливаем генерацию примерно через 20 секунд
            break

        # Отрисовываем карту и неразбитые машины
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        if record_video:
            recorder.capture_frame(screen)

        pygame.display.flip()
        clock.tick(FPS)  # 60 FPS

    if record_video:
        recorder.end_recording()


def run_best_genom(config_path, genome_path='best_genom.pkl'):
    # Функция для записи вождения лучшего агента
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)

    # Загружаем лучшего агента из файла
    with open(genome_path, 'rb') as file:
        genome = pickle.load(file)

    # Приводим данные к требуемому виду
    genomes = [(1, genome)]

    # Вызываем симуляцию с одним лучшим агентом и записываем её
    run_simulation(genomes, config, record_video=True)


def train(config_path, num_generations, best_genom_save_path='best_genom.pkl'):
    # Функция для обучения агентов
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # Создание популяции и вывод статистики по обучению
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Запуск симуляции для максимум num_generations генераций
    best_genom = population.run(run_simulation, num_generations)
    with open(best_genom_save_path, 'wb') as file:
        pickle.dump(best_genom, file)
        file.close()
    plot_stats(stats, view=True)
    draw_net(config, best_genom)


if __name__ == "__main__":
    train(config_path, 5)  # Запускаем обучение нейронных сетей и выводи график обучения
    run_best_genom(config_path)  # Записываем видео с игрой лучшего агента