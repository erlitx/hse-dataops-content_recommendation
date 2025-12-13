# Гайд, как выполнить

Делаю на Ubuntu-like системе

**утилита для работы с репами:**
sudo apt install -y software-properties-common


**Добавить репу ansible(в инетах пишут про флаг update, у меня с ним не взлетело)**
sudo add-apt-repository --yes ppa:ansible/ansible

**Так как в прошлом пунте без флага обновления, обнови вручную:**
sudo apt update

**Ставь ansible**
sudo apt install -y ansible

**Проверь:**
ansible --version


**Теперь выполни(если докера нет):**
ansible-galaxy collection install community.docker


Ну вот, твоя тачка - Control Node, дальше надо создавать playbook.yml, inventory и запускать их


Создание структуры на локалке:
```bash
cd ~
mkdir ansible-fastapi-hw
cd ansible-fastapi-hw
# это для папок, флаг -p для вложенности:
mkdir -p roles/docker/tasks
mkdir -p roles/fastapi_app/tasks
mkdir -p roles/fastapi_app/files/app
#### пустые файлики:
touch inventory.ini playbook.yml
touch roles/docker/tasks/main.yml
touch roles/fastapi_app/tasks/main.yml
touch roles/fastapi_app/files/app/Dockerfile
touch roles/fastapi_app/files/app/main.py
touch roles/fastapi_app/files/app/requirements.txt
```

### Все содержимое файлов в данной директории


После создани я виртуалки собрал контейнер:

ansible-playbook -i inventory.ini playbook.yml

и проверил:
ansible -i inventory.ini all -m ping

для проверка рабьоты модели подключился к адресу виртуалки (он динамический, поэтому указывать его бессмысленно)

Результат работы модельки по адресу:
http://[адрес]/recommend/baseline/1

Для проверки, подлючаюсь по ssh:
ssh [тут_мой_ssh_ник]@[адрес]

Проверка логов контейнера:
[тут_мой_ssh_ник]@compute-vm-2-1-10-hdd-1765643991026:~$ sudo docker logs -f fastapi_container









