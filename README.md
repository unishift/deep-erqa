# Deep ERQA

Нейросетевая метрика качества восстановления границ

## Пример использования

1. Берем веса с Calypso: `//calypso/work/22e_lya/deep-erqa/weights/best_model.zip`
2. Разжимаем zip
3. Модифицируем `edge_similarity/test_benchmark.py` под свои нужды и запускаем. Текущий код для `${VSR_PATH}/test1_bicubic/full_result/`
```
python edge_similarity/test_benchmark.py --gpu 0 ${MODEL_PATH}/best_model/model ${DATA_PATH} ${RES_PATH}
```

TODO: Сделать более унифицированный запуск
