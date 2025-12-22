# Neural Optimal Transport
---
## Структура репозитория
```bash
├── configs ### Папка с конфигами для моделей
│   ├── default.yaml
│   └── kaggle.yaml
├── data ### Классы для работы с данными
│   ├── __init__.py
│   ├── dataloader.py
│   └── dataset.py
├── inference ### Папка с изображениями для генерации примеров
│   ├── source ### Входные изображения
│   │   ├── 202592.jpg
│   │   ├── 202593.jpg
│   │   ├── 202594.jpg
│   │   ├── 202595.jpg
│   │   └── 202596.jpg
│   └── target ### Результат генерации + сравнения
│       └── output.pdf
├── experiments
│   └── default
│       └── logs
│           ├── 2025-12-18.log
├── checkpoints ### Чекпоинты моделей
│   ├── default_0.pth
├── logger ### Класс для логирования
│   └── logwriter.py
├── metrics ### Метрики качества генерации
│   ├── __init__.py
│   ├── loss.py
│   └── sim_metrics.py
├── models ### Классы с моделями
│   ├── __init__.py
│   ├── resnet.py
│   └── unet.py
├── outputs ### Сохраненные логи из консоли
│   ├── 2025-12-18
│   │   ├── 23-51-18
│   │   │   ├── .hydra
│   │   │   └── main.log
├── trainer ### Классы для обучения моделей
│   ├── base_trainer.py
│   └── not_trainer.py
├── utils ### Прочее: функции для отрисовки изображений
│   ├── __init__.py
│   └── plots.py
├── validation ### Случайные изображения для валидации
│   ├── source
│   │   ├── sample1.png
│   │   ├── sample2.png
│   │   └── sample3.png
│   └── target
│       ├── sample1.png
│       ├── sample2.png
│       └── sample3.png
├── README.md
├── main.py
├── requirements.txt
```
## Запуск обучения

Чтобы запустить пайплайн для обучения модели, результат которой был засабмичен для чекпоинта, необходимо выполнить следующую команду:

```bash
!python /NeuralOptimaTransport/main.py --config-name kaggle
```