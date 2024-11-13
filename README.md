# Transfer Learning para Classificação de Imagens de Gatos e Cachorros

Este projeto demonstra o uso de Transfer Learning para classificar imagens de gatos e cachorros utilizando redes neurais profundas em Python. Vamos explorar como carregar e treinar um modelo pré-existente no ambiente Google Colab e adaptar esse modelo para nosso problema específico.

## Visão Geral

Transfer Learning é uma técnica poderosa de aprendizado de máquina que permite usar o conhecimento de um modelo treinado em uma tarefa específica e aplicá-lo a uma nova tarefa com um esforço menor. Neste projeto, aplicaremos o Transfer Learning para distinguir entre imagens de gatos e cachorros usando um modelo de Deep Learning.

## Recursos Utilizados

- **Ambiente**: [Google Colab](https://colab.research.google.com)
- **Linguagem**: Python
- **Bibliotecas**: TensorFlow, Keras, entre outras
- **Modelo Base**: Utilizaremos um modelo pré-treinado (baseado no MNIST) como exemplo.
- **Dataset de Imagens**: [Cats vs. Dogs](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)

## Objetivos do Projeto

1. **Configurar o ambiente Colab** e carregar as bibliotecas necessárias.
2. **Baixar e preparar o dataset** de imagens de gatos e cachorros.
3. **Carregar um modelo pré-treinado** para transfer learning.
4. **Realizar o treinamento** do modelo com o dataset adaptado.
5. **Avaliar a performance** do modelo e gerar predições.

---

## Passo a Passo

### Passo 1: Configurando o Ambiente e Instalando as Bibliotecas Necessárias

Primeiramente, abra um novo notebook no Google Colab. Certifique-se de que o ambiente está com o runtime configurado para GPU para acelerar o processo de treinamento.

No início do notebook, execute o seguinte código para importar as bibliotecas principais:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
```

### Passo 2: Carregando e Preparando o Dataset de Imagens de Gatos e Cachorros

O dataset Cats vs. Dogs pode ser baixado através do [link da Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=54765). Este conjunto de dados já está organizado em duas classes: `gatos` e `cachorros`.

1. **Baixar o Dataset**: Faça o download do dataset e descompacte-o. Em seguida, faça o upload para o Google Colab.
2. **Carregar o Dataset no Colab**:

```python
dataset_path = "/path/to/cats_and_dogs"

train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)
```

### Passo 3: Escolhendo o Modelo Pré-Treinado para Transfer Learning

Neste exemplo, vamos utilizar a arquitetura **VGG16**, amplamente usada para classificação de imagens e disponível na biblioteca Keras. O modelo vem pré-treinado no dataset **ImageNet**.

```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congela os pesos do modelo base
```

### Passo 4: Adicionando Camadas para Transfer Learning

Agora, adicionamos novas camadas ao modelo pré-treinado para adaptar a rede ao nosso dataset de gatos e cachorros:

```python
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Saída binária para classificar gatos vs. cachorros
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Passo 5: Treinando o Modelo

Com o modelo pronto, iniciamos o processo de treinamento. Abaixo, definimos o número de épocas (quantas vezes o modelo verá os dados) e monitoramos o progresso:

```python
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)
```

### Passo 6: Avaliando o Modelo

Após o treinamento, avaliamos o desempenho do modelo nos dados de validação. Isso nos ajudará a entender a precisão do modelo ao classificar novas imagens de gatos e cachorros.

```python
val_loss, val_acc = model.evaluate(val_dataset)
print(f"Acurácia de validação: {val_acc * 100:.2f}%")
```

### Passo 7: Fazendo Predições

Podemos agora usar o modelo treinado para fazer previsões em novas imagens:

```python
def predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expande a dimensão para batch único

    prediction = model.predict(img_array)

    if prediction[0] < 0.5:
        print("É um Gato!")
    else:
        print("É um Cachorro!")
```

Para utilizar a função acima, passe o caminho de uma imagem de gato ou cachorro como parâmetro:

```python
predict_image("/path/to/image.jpg")
```

---

## Referências e Recursos

1. **Dataset de gatos e cachorros**: [Cats vs. Dogs no TensorFlow](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)
2. **Modelo VGG16**: [Keras Applications VGG16](https://keras.io/api/applications/vgg/#vgg16-function)

## Considerações Finais

Este projeto introduz os conceitos fundamentais de Transfer Learning, aplicando um modelo pré-treinado para resolver uma tarefa específica de classificação de imagens. Com Transfer Learning, conseguimos reduzir significativamente o tempo e o custo de treinamento, aproveitando um modelo já otimizado para identificar padrões complexos em imagens. 

Esperamos que este guia ajude você a entender o processo e a começar a explorar o Transfer Learning em seus próprios projetos!

---

**Observação**: Este projeto é um ponto de partida. Para melhores resultados, você pode experimentar outras arquiteturas ou realizar o fine-tuning, permitindo que algumas camadas do modelo pré-treinado sejam treinadas com os dados de gatos e cachorros.
