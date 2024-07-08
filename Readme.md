Тестовое...

``` markdown
Для запуска проекта нужно.

Создать папку data в корне проекта 
Поместить данные с картинками в директорию /data/dataset, чтобы получилось
Создать папку data/db

├── ...
├── common
├── data                    
│   ├── dataset          
│       ├── bags         
│       ├── full_body     
│       ├── ... 
│       └── upper_body
│   ├── db  

ВАЖНО!!! Все скрипты запускаются из своих директорий. 
```

* Запустить скрипт components/create_df.py 
```bash
python create_df.py --path_to_images ../data/dataset --path_save_csv ../data/
```

Скрипт создаст df pandas в директории data

* Запустить скрипт components/create_db.py
```bash
python create_db.py --save_db_files_path ../data/db/ --df_path ../data/dataset.csv --device cuda
```

Скрипт создаст index файл и файлы json в директории data/db для работы с faiss

* Скрипт main.py запустит приложение.
По http://127.0.0.1:7860 откроется форма в которой можно добавить картинку и она выдаст ближайшие 2 картинки
Количество ближайших картинок равно 2 и это захардкожено, так как при данном значении получилась наилучшая метрика F1
```bash
python main.py --db_files_path ./data/db/ --df_path ./data/dataset.csv --device cuda
```

* Скрипт test/test_get_label.py для теста. Выводит в консоль название предсказанной вещи и правильный ответ
```bash
python test_get_label.py --db_files_path ../data/db/ --df_path ../data/dataset.csv --device cuda
```

В notebooks/pipeline.ipynb можно посмотреть как emb выглядят в 2D с помощью TSNE

