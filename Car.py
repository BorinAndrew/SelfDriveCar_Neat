import math
import pygame


car_path = 'car2.png'


class Car:
    # Класс для создания машинок

    def __init__(self, car_image_path=car_path, car_size=(60, 60), car_start_pos=(900, 830), car_start_angle=0,
                 car_acceleration=2, car_rotation_vel=10, speed_set_flag=False,
                 radars_angle_list=(-75, -30, -15, -5, 0, 5, 15, 30, 75),
                 border_color=(255, 255, 255, 255), checkpoint_color=(255, 0, 0, 255)):

        self.car_size_x = car_size[0]
        self.car_size_y = car_size[1]

        self.border_color = border_color          # Цвет при столкновением с которым фиксируем аварию
        self.checkpoint_color = checkpoint_color  # Цвет при контакте с которым получаем доп. награду

        # Загрузка изображения машины
        self.sprite = pygame.image.load(car_image_path).convert()
        self.sprite = pygame.transform.scale(self.sprite, car_size)  # Изменяем размер под заданный
        self.rotated_sprite = self.sprite

        self.position = [car_start_pos[0], car_start_pos[1]]  # Начальная позиция
        self.angle = car_start_angle
        self.speed = 0

        self.acceleration = car_acceleration   # Ускорение
        self.rotation_vel = car_rotation_vel  # Угол поворота

        self.speed_set = speed_set_flag  # Флаг для постоянной скорости

        self.center = [self.position[0] + self.car_size_x / 2,
                       self.position[1] + self.car_size_y / 2]  # Вычисление центра машины

        self.radars_angle_list = radars_angle_list
        self.radars = []  # Список сенсоров
        self.drawing_radars = []  # Список для отрисовки сенсоров
        self.corners = []
        self.alive = True  # Для проверки аварии

        self.distance = 0   # Пройденный путь
        self.time = 0       # Счетчик времени
        self.reward = 0     # Полученная награда
        self.checkpoint = False  # Флаг для проверки преодоления чекпоинта

    def draw(self, screen):
        # Метод для отрисовки машины и сенсоров
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        # Метод для отрисовки сенсоров
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 127, 127), self.center, position, 1)
            pygame.draw.circle(screen, (0, 127, 127), position, 5)

    def check_collision(self, game_map):
        # Проверка колизии
        self.alive = True
        for point in self.corners:
            # Если одно из четырёх ядер задевает цвет стенки, происходит авария
            if game_map.get_at((int(point[0]), int(point[1]))) == self.border_color:
                self.alive = False
                break

            # Если ядро задевает цвет чекпоинта, получаем дополнительную награду и меняем флаг на True
            if game_map.get_at((int(point[0]), int(point[1]))) == self.checkpoint_color:
                self.reward += 200
                self.checkpoint = True

    def calculate_radar(self, degree, game_map):
        # Метод расчета сенсоров
        # Начальные значения
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Пока сенсор не достиг стенки и длинна не превышает 350,
        # увеличиваем длинну и вычисляем координаты конечной точки
        while not game_map.get_at((x, y)) == self.border_color and length < 350:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Вычисляем дистанцию сенсора и добавляем данные в список
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        # Устанавливаем начальную скорость для первой генерации, чтобы ускорить обучение
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Поворачиваем картинку
        # Вычисляем х координату, не даём ей уйти за край карты
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], game_map.get_width() - 20)

        # Тоже самое для y
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], game_map.get_height() - 20)

        # Увеличиваем путь и время
        self.distance += self.speed
        self.time += 1

        # Вычисляем новый центр
        self.center = [int(self.position[0]) + self.car_size_x / 2, int(self.position[1]) + self.car_size_y / 2]

        # Вычисляем четыре ядра разположенный по углам машины
        length = 0.5 * self.car_size_x
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Проверяем колизии и подчищаем сенсоры
        self.check_collision(game_map)
        self.radars.clear()

        # Вычисляем сенсоры по заданным углам
        for d in self.radars_angle_list:
            self.calculate_radar(d, game_map)

    def get_data(self):
        # Получаем расстояние до стенок
        radars = self.radars
        return_values = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        # Метод состояния машины
        return self.alive

    def get_reward(self):
        # Вычисляем награду как две сотых пройденного пути
        self.reward += self.distance / 50
        # Условие которое штрафует машину, если она вдруг закольцовываесть
        # или медленно едет не преодолев первого чекпоинта
        if self.time >= 80 and not self.checkpoint:
            self.reward -= 100
        return self.reward

    def rotate_center(self, image, angle):
        # Метод поворота картинки машины
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image
