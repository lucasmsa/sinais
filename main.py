from imports import *

# redimensiona e achata uma imagem transformando-a 
# em uma lista de intensidade de pixels
def image_to_feature_vector(image, size=(10, 10)):
        
    return cv2.resize(image, size).flatten()


# construir o histograma da imagem para caracterizar a distribuicao de tons de preto desta
def extract_grayscale_histogram(image):
    # extrai os tons de cinza
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # para uma imagem em preto e branco 8-bits de informacao por pixel, existem 256 diferentes possibilidades 
    # de intensidade, assim, o histograma tem 256 valores
    hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])

    # normalizacao do histograma
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    else:
        cv2.normalize(hist, hist)

    return hist.flatten()


try:
    os.makedirs('fft_faces')
except Exception:
    pass

counter = 0
# passa por todas as imagens do banco de dados cedido e
# cria novas imagens utilizando transformada de fourier bidimensional
for s_value in range(1, 41):

    for img_value in range(1, 11):

        img = Image.open(f'orl_faces/s{s_value}/{img_value}.pgm')
        
        img_array = np.asarray(img)
        
        # aplicada a transformada de fourier bidimentsional
        fft_arr = abs(np.fft.rfft2(img_array))
        
        new_img = Image.fromarray(fft_arr)

        if new_img.mode != 'RGB':
            new_img = new_img.convert('RGB')

        new_img.save(f'fft_faces/s{s_value}_{img_value}.png')

        counter += 1
        
#print(fft_arr)

image_paths = sorted(list(paths.list_images('fft_faces')))

# intensidade dos pixels
pixelIntensities_train = []
pixelIntensities_test = []
# histograma
histFeatures_train = []
histFeatures_test = []
# uma das 41 pessoas do dataset
classLabels_train = []
classLabels_test = []

# para cada imagem no banco de dados preencher as 
# listas supracitadas com os valores determinados

counter = 0
random_image_num = random.randint(1, 10)
print(image_paths)

for (i, image_path) in enumerate(image_paths):

    if counter%10 == 0:
        random_image_num = random.randint(1, 10)
        
    image = cv2.imread(image_path)
    # pegar a pessoa da imagem pelo nome do arquivo
    label = image_path.split(os.path.sep)[-1].split('_')[0]
    number_image = image_path.split(os.path.sep)[-1].split('_')[1]
    number_image = number_image.split('.')[0]
    
    label_numeric = label.split('s')[1]
    
    # pegar valores de intensidade do pixel, depois o histograma
    # obtendo a distribuicao dos tons de preto na imagem
    pixels = image_to_feature_vector(image)
    histogram = extract_grayscale_histogram(image)

    counter += 1
    if random_image_num == int(number_image):
        
        pixelIntensities_test.append(pixels)
        histFeatures_test.append(histogram)
        classLabels_test.append(label_numeric)
        
    else: 
        
        pixelIntensities_train.append(pixels)
        histFeatures_train.append(histogram)
        classLabels_train.append(label_numeric)


print("Length test: ", len(pixelIntensities_test))
print("Length train: ", len(pixelIntensities_train))
# dividir os dados em treino e teste, 90% treino 10%
# a priori, utilizando a intensidade dos pixels

# (train_PI, test_PI, train_L, test_L) = train_test_split(
#     pixelIntensities, classLabels, test_size=0.1, random_state=42
# )

# utiliza-se aqui o histogrma dos tons de preto para treino 
# e teste

# (train_HF, test_HF, train_HF_L, test_HF_L) = train_test_split(
#     histFeatures, classLabels, test_size=0.1, random_state=42
# )
print('---------------------')

# treinamento do KNN com 1 vizinho, utilizando o Pixel Intensities
model = KNeighborsClassifier(n_neighbors=1)
# treina o modelo atraves desse .fit
model.fit(histFeatures_train, classLabels_train)
pred = model.predict(histFeatures_test)
pred = list(pred)
pred = [int(i) for i in pred]
accuracy = model.score(histFeatures_test, classLabels_test)
print("PREDICTIONS: ", pred)
classLabels_test = [int(i) for i in classLabels_test]
print("CLASS LABELS: ", classLabels_test)
model_accuracy = accuracy*100
mse = np.square(np.subtract(classLabels_test, pred)).mean()
print(f'Model accuracy Histogram: {model_accuracy}%')
print(f'Mean squared error: {mse}')


# Treinamento do modelo atraves do histograma da imagem
model.fit(train_HF, train_HF_L)
print(model)
pred = model.predict(test_HF)

print(np.asarray(test_HF_L, dtype=np.float64))
print(pred.astype(np.float64))
accuracy = model.score(test_HF, test_HF_L)
model_accuracy = accuracy*100
print(f'Model accuracy: {model_accuracy}')

# TODO: Descobrir como fazer o MSE, tentei pela funcao mas do sklearn mas deu erro
# TODO - talvez esse link ajude: https://machinelearningmastery.com/implement-machine-learning-algorithm-performance-metrics-scratch-python/
# ?: Problema eh que nao sei exatamente o que deve ir para os parametros do mean squared error

# TODO: Rodar os testes que ele pediu, acredito que essa parte nao seja tao dificil,
# TODO - o que gera a transformada de fourier bidimensional eh essa parte do codigo 
# TODO -- fft_arr = abs(np.fft.rfft2(img_array)), eh soh para cada imagem realizar os testes que ele pede, *acho*

# TODO: A parte de achar a sub-regiao com menor frequencia tambem nao sei se esta sendo feita 
# TODO - acredito que isso so vai requerer mudancas na funcao image_to_feature_vector, ela ja tem um parametro size
# TODO -- mas acho que ele so redimensiona a imagem, e nao trata de achar essa sub-regiao, ver esse link: https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/
# TODO --- talvez esse tambem ajude: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html, ou esse: https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/