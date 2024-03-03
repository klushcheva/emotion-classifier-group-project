# МОВС, годовой проект 2023

## Участники: Павел Егоров, Мария Аугуст, Ксения Лущева

## Проект: Изучение социального настроения граждан c помощью машинного обучения

### Описание проекта

#### Проблема, которую решает проект



### Шаги выполненные на текущий момент:
#### Шаг 1. Bag of Words 

Подход, использующий метод Bag of Words (BoW) в сочетании с CountVectorizer и Logistic Regression для обучения языковых моделей, является прочным фундаментом в обработке естественного языка (NLP), поэтому мы решили начать с него. В результате использования (BoW) для столбца sentiment получили:

- Высокую точность при использовании 1-граммовой модели.
- Достижение точности 0,94 с помощью 1-граммовой модели впечатляет и указывает на то, что модель очень эффективно классифицирует настроения как положительные или отрицательные на основе отдельных слов. Это говорит о том, что для набора данных наличие или отсутствие определенных ключевых слов является сильным предиктором настроения.
- Мы получили снижение точности при использовании 3-граммовой модели.
- Снижение точности до 0,85 при использовании 3-граммовой модели скорее всего говорит о том, что включение контекста окружающих слов (до трех слов вместе) не только не улучшает, но даже может несколько ухудшить работу модели. 
- Хотя подход Bag of Words прост и эффективен для многих задач, он не учитывает порядок слов и семантические отношения между словами. Поэтому на следующих этапах мы постарались учесть более сложные модели.
#### Шаг 2. TfIdfVectorizer 

TfIdfVectorizer и расширение значений целевых переменных демонстрирует развитие и масштабирование проекта

- Использование TfIdfVectorizer с LogisticRegression сохраняет высокую точность бинарной классификации (положительная/отрицательная) как с 1-граммовой, так и с 3-граммовой моделями. 
- Небольшое увеличение точности 3-граммовой модели с использованием TfIdfVectorizer (с 0,85 до 0,86) по сравнению с подходом BoW позволяет предположить, что взвешивание TfIdf, которое подчеркивает важность менее частотных слов, может быть более эффективным для улавливания нюансов контекста в больших n-граммах.
- Получили первую проблему с многоклассовой классификацией. Расширение категории до "нейтральный" и обучение ряда классификаторов на векторах (1,2)-грамм показало заметное падение точности во всех моделях. 
- Это указывает на возросшую сложность различения трех классов настроений, особенно с добавлением категории "нейтральный", которая может пересекаться с характеристиками как "позитивных", так и "негативных" настроений.
- Также это может указывать на недостаточно хорошую разметку данных, в связи с чем, одной из потенциально решаемых задач в нашей работе - уточнение сентиментов.
- Низкие показатели F1 для позитивных/негативных настроений. Показатели f1 для положительных и отрицательных категорий ниже 0,3 свидетельствуют о значительных проблемах в достижении сбалансированной классификации с помощью текущих моделей, особенно в различении положительных и отрицательных настроений в трехклассовой системе. Это говорит о том, что модели предвзяты к "нейтральной" категории или пытаются найти отличительные признаки для "позитивного" и "негативного".
- Последующий анализ доказывает это.

#### Шаг 3. Включение дополнительных признаков в TfIdfVectorizer 

На следующем шаге мы постарались включить дополнительные параметры:
- Добавление новых признаков, таких как emotion, toxicity, is_congratulation и spam, наряду с существующими текстовыми данными представляет собой стратегический подход к улучшению понимания моделью глубинного контекста и нюансов текстовых данных. 

- Включая такие признаки, как эмоциональность, токсичность, is_congratulation и спам, мы позволяем модели улавливать более широкий спектр текстовых нюансов, что может значительно улучшить ее способность понимать и точно классифицировать текст. Эти особенности могут дать ценные сигналы, которые не улавливаются только текстом.

- Использование OHE для этих дополнительных признаков - подходящий выбор для категориальных данных, поскольку оно позволяет модели рассматривать каждую категорию как отдельную сущность, не подразумевая никакого порядка. Это может помочь в точном отражении влияния каждого признака на целевую переменную.

- Продолжение использования TfIdf для кодирования текста с помощью 1-грамм гарантирует, что модель учитывает важность каждого слова в корпусе, уменьшая при этом влияние часто встречающихся слов. Это позволяет сбалансировать набор признаков между вновь добавленными категориальными признаками и текстовыми данными.

- Мы применили TruncatedSVD для сжатия матрицы признаков до 100 признаков. Это поможет решить проблемы, связанные с высокой размерностью, такие как проклятие размерности и чрезмерная подгонка, что сделает вашу модель более обобщенной. Кроме того, SVD может выявить скрытые связи между признаками, что потенциально повышает производительность модели.

#### Шаг 4. Word2Vec

Мы предприняли переход к использованию nltk для удаления стоп-слов и модели Word2Vec для векторизации текста. Ниже подробный обзор этих шагов:

- Использование nltk для удаления стоп-слов. Удаление стоп-слов имеет решающее значение для уменьшения шума в текстовых данных. Это помогает сфокусироваться на словах, которые предлагают более значимые идеи или чувства.
  
- nltk, ведущая библиотека Python для обработки естественного языка, предоставляет список стоп-слов. Мы отфильтровали свои текстовые данные по этому списку, удалив эти слова перед дальнейшей обработкой.

- Использование Word2Vec для векторизации текста. Word2Vec - двухслойная нейронная сеть, которая обрабатывает текст, "обучаясь" векторным представлениям слов. Она учитывает контекстуальные нюансы и семантические связи между словами, в отличие от методов BoW и TfIdf, которые рассматривают слова как независимые сущности. Используя Word2Vec, мы преобразуем текст в векторы, которые представляют слова в непрерывном векторном пространстве. Это означает, что слова с похожими значениями расположены близко друг к другу в этом пространстве, что может значительно повысить способность модели понимать нюансы текста.
  
- Этот подход особенно эффективен для улавливания контекста слов, понимания синонимов и для улавливания определенного настроения. Мы рассчитываем, что  в нашем случае это приведет к созданию более эффективной модели, особенно для задач, требующих глубокого понимания семантики текста, таких как анализ настроения, классификация текстов и рекомендательные системы.


Потенциальные дальнейшие шаги к улучшению модели:

1. Разработка новых признаков.: Подумать о том, чтобы поэкспериментировать с более сложными методами извлечения признаков или добавить собственные признаки, которые могут более эффективно отражать настроения.
2. Использовать более продвинутые модели. Следующим шагом будет использование моделей на основе трансформеров (BERT  и библиотека Natasha), которые могут более эффективно отражать нюансы настроения.
3. Поиск возможностей проверить и улучшить разметку. Дисбаланс классов, который демонстрирует наша выборка может быть эффективна при работе с такими методами, как перевыборка класса меньшинства, недовыборка класса большинства или применение весовых коэффициентов класса во время обучения модели.
4. Оценка модели на основе матрицы смешения, чтобы понять закономерности неправильной классификации, которые могут дать представление о том, как модели работают с разными классами.
